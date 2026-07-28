#!/usr/bin/env python3
"""Regenerate the claim-critical paper-suite status page.

This is deliberately a read-only aggregator: workers write immutable atomic
rows, and this script only summarizes rows that already exist.  It never
converts an error or timeout into a mapping result.
"""
from __future__ import annotations

import csv
import datetime as dt
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "paper_suite_v1"
TRACKER = ROOT / "PAPER_RESULTS.md"
DEADLINE = dt.datetime.fromisoformat("2026-07-31T23:59:00+05:30")

TERMINAL_QUALITY = {"DONE", "VALID_MAPPING_FAILURE", "BUDGET_EXHAUSTED"}
TERMINAL = TERMINAL_QUALITY | {"TIMEOUT", "ERROR", "UNSUPPORTED"}


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def work_rows() -> list[dict[str, str]]:
    return read_rows(OUT / "work_manifest.csv")


def progress(rows: list[dict[str, str]]) -> dict[str, float]:
    done = [r for r in rows if r.get("status") in TERMINAL]
    quality = [r for r in rows if r.get("status") in TERMINAL_QUALITY]
    principal_methods = {"pathfinder", "dual_linear", "flow_proposal", "flow_top4", "flow_noparent_top4"}
    principal_expected = 12 * 4 * len(principal_methods) * 3
    principal_done = sum(r.get("method") in principal_methods and r.get("status") in TERMINAL_QUALITY for r in rows)
    sa_expected = 12 * 4 * 3 * 3
    sa_done = sum(r.get("method") in {"simulated_annealing", "sa"} and r.get("status") in TERMINAL_QUALITY for r in rows)
    full_done = sum(r.get("method") in {"full_relaxed_lookahead", "top2", "top4", "top6", "hybrid_top4", "noparent_top4"} and r.get("status") in TERMINAL_QUALITY for r in rows)
    a3_done = sum(r.get("architecture", "").startswith("A3") and r.get("status") in TERMINAL_QUALITY for r in rows)
    adapter = read_rows(OUT / "raw" / "roundtrip_validation.csv")
    adapter_legal = sum(str(r.get("flow_legality_success", "")).lower() in {"true", "1"} for r in adapter)
    simulations = sum(str(r.get("simulation_success", "")).lower() in {"true", "1"} for r in adapter)
    negative = read_rows(OUT / "raw" / "negative_legality_tests.csv")
    negative_ok = sum(str(r.get("flow_rejected", "")).lower() in {"true", "1"} for r in negative)
    artifact_files = [OUT / "tables", OUT / "plots", OUT / "raw"]
    artifact_score = sum(1 for p in artifact_files if p.exists()) / len(artifact_files)
    sections = {
        "adapter_functional_correctness": min(1.0, 0.5 * adapter_legal / 9 + 0.25 * simulations / 6 + 0.25 * negative_ok / 8),
        "principal_matrix": min(1.0, principal_done / max(1, principal_expected)),
        "conventional_baselines": min(1.0, sa_done / max(1, sa_expected)),
        "lookahead_and_parent_ablations": min(1.0, full_done / max(1, 12 * 6)),
        "portability_and_scaling": min(1.0, a3_done / max(1, 12 * 3)),
        "runtime_and_statistics": 1.0 if (OUT / "raw" / "bootstrap_results.csv").exists() else 0.0,
        "artifacts_and_reporting": artifact_score,
    }
    weights = {"adapter_functional_correctness": .20, "principal_matrix": .35, "conventional_baselines": .10, "lookahead_and_parent_ablations": .10, "portability_and_scaling": .10, "runtime_and_statistics": .05, "artifacts_and_reporting": .10}
    sections["overall"] = sum(sections[k] * weights[k] for k in weights)
    return {k: round(v, 6) for k, v in sections.items()} | {"rows": len(rows), "terminal": len(done), "quality_terminal": len(quality)}


def bar(value: float, width: int = 24) -> str:
    filled = max(0, min(width, round(value * width)))
    return "[" + "█" * filled + "░" * (width - filled) + "]"


def main() -> None:
    now = dt.datetime.now(dt.timezone.utc).astimezone(DEADLINE.tzinfo)
    hours = max(0.0, (DEADLINE - now).total_seconds() / 3600.0)
    rows = work_rows()
    p = progress(rows)
    status_counts: dict[str, int] = {}
    for row in rows:
        status = row.get("status", "UNKNOWN")
        status_counts[status] = status_counts.get(status, 0) + 1
    status_text = ", ".join(f"{k}={v}" for k, v in sorted(status_counts.items())) or "no jobs registered"
    content = f"""# FlowAdvantage Paper Results Index

Generated: `{now.isoformat()}`  
Deadline: `2026-07-31 23:59 IST`  
Hours remaining: **{hours:.2f}**

## Paper-readiness progress

**{bar(p['overall'])} {100*p['overall']:.1f}%** (fixed claim-critical weighted denominator)

| Evidence obligation | Completion |
|---|---:|
| Adapter and functional correctness | {100*p['adapter_functional_correctness']:.1f}% |
| Complete 12×4 principal matrix | {100*p['principal_matrix']:.1f}% |
| Conventional baseline coverage | {100*p['conventional_baselines']:.1f}% |
| Lookahead and parent ablations | {100*p['lookahead_and_parent_ablations']:.1f}% |
| Portability and scaling | {100*p['portability_and_scaling']:.1f}% |
| Runtime and statistics | {100*p['runtime_and_statistics']:.1f}% |
| Artifacts and reporting | {100*p['artifacts_and_reporting']:.1f}% |

## Queue status

- Registered work items: `{p['rows']}`
- Terminal work items: `{p['terminal']}`
- Quality-terminal items: `{p['quality_terminal']}`
- Status counts: `{status_text}`

Timeouts, errors, and unsupported rows are retained and excluded from quality
aggregates. Borderline values remain valid evidence and are marked AMBER in
the gate tables; no threshold is changed after observing results.

## Artifact index

- Grounding: [docs/HPCA_PAPER_RIGOR_GROUNDING.md](docs/HPCA_PAPER_RIGOR_GROUNDING.md)
- Execution plan: [docs/HPCA_PAPER_SUITE_PLAN.md](docs/HPCA_PAPER_SUITE_PLAN.md)
- Frozen config: [configs/paper_suite_v1.yaml](configs/paper_suite_v1.yaml)
- Atomic manifests: `results/paper_suite_v1/work_manifest.csv`, `results/paper_suite_v1/work_items/`
- Raw results: `results/paper_suite_v1/raw/`
- Tables: `results/paper_suite_v1/tables/`
- Plots: `results/paper_suite_v1/plots/`
- Checkpoints and logs: `results/paper_suite_v1/checkpoints/`, `results/paper_suite_v1/logs/`

## Current interpretation

The real Morpher bridge is not considered paper-ready until native legality,
independent legality, route/placement/II round-trip, and cycle-simulation
checks pass on the frozen reference corpus. A mapping timeout is operational
evidence, not a scientific mapping failure.
"""
    TRACKER.write_text(content, encoding="utf-8")
    (OUT / "progress.json").write_text(json.dumps({"generated_at": now.isoformat(), "hours_remaining": hours, **p}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
