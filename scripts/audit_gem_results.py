#!/usr/bin/env python3
"""Reconstruct the terminal gem-branch evidence without smoothing missing work.

This script intentionally treats TIMEOUT/ERROR as operational outcomes, not mapping
failures, and only computes paired comparisons on identical normally completed
kernel/architecture identities.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


NORMAL = {"DONE", "VALID_MAPPING_FAILURE"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def result_for_manifest_row(root: Path, row: dict[str, str]) -> dict[str, Any]:
    result = root / "work_items" / Path(row["result_path"]).name
    payload: dict[str, Any] = read_json(result) if result.exists() else {}
    payload.update(
        {
            "kernel": row["kernel"],
            "architecture": row["architecture"],
            "method": row["method"],
            "manifest_status": row["status"],
            "manifest_wall_seconds": number(row.get("wall_seconds")),
            "work_id": row["work_id"],
            "result_file": str(result),
        }
    )
    payload["status"] = row["status"]
    payload["normal_completion"] = row["status"] in NORMAL
    payload["success"] = bool(payload.get("success", row["status"] == "DONE"))
    return payload


def number(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_remote(snapshot: Path) -> list[dict[str, Any]]:
    return [result_for_manifest_row(snapshot, row) for row in read_csv(snapshot / "work_manifest.csv")]


def load_best_length(roots: Iterable[Path]) -> dict[tuple[str, str], dict[str, Any]]:
    """Select the strongest terminal attempt, using later retry roots for ties."""
    rank = {"DONE": 4, "VALID_MAPPING_FAILURE": 3, "TIMEOUT": 2, "ERROR": 1}
    selected: dict[tuple[str, str], tuple[int, int, dict[str, Any]]] = {}
    for root_index, root in enumerate(roots):
        for row in read_csv(root / "work_manifest.csv"):
            payload = result_for_manifest_row(root, row)
            key = (payload["kernel"], payload["architecture"])
            score = (rank.get(payload["status"], 0), root_index)
            previous = selected.get(key)
            if previous is None or score > previous[:2]:
                selected[key] = (score[0], score[1], payload)
    return {key: value[2] for key, value in selected.items()}


def bootstrap_success_delta(
    baseline: np.ndarray, method: np.ndarray, seed: int = 24072026, samples: int = 10_000
) -> tuple[float, float, float]:
    delta = method.astype(float) - baseline.astype(float)
    rng = np.random.default_rng(seed)
    draws = rng.choice(delta, size=(samples, len(delta)), replace=True).mean(axis=1)
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return float(delta.mean()), float(lo), float(hi)


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is None:
        columns = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    def cell(value: Any) -> str:
        if value is None:
            return "NA"
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    lines += ["| " + " | ".join(cell(row.get(c)) for c in columns) + " |" for row in rows]
    return "\n".join(lines) + "\n"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--length-root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    output = args.output
    tables = output / "tables"
    raw = output / "raw"
    tables.mkdir(parents=True, exist_ok=True)
    raw.mkdir(parents=True, exist_ok=True)

    remote = load_remote(args.snapshot)
    length = load_best_length(args.length_root)
    remote_by_method: dict[str, dict[tuple[str, str], dict[str, Any]]] = defaultdict(dict)
    for row in remote:
        remote_by_method[row["method"]][(row["kernel"], row["architecture"])] = row

    terminal_rows = []
    for row in remote:
        terminal_rows.append(
            {
                "kernel": row["kernel"],
                "architecture": row["architecture"],
                "method": row["method"],
                "status": row["status"],
                "normal_completion": row["normal_completion"],
                "success": row["success"],
                "legal": row.get("legal"),
                "ii": row.get("ii"),
                "route_cost": row.get("route_cost"),
                "compile_wall_seconds": row.get("compile_wall_seconds"),
                "routing_attempts": row.get("routing_attempts"),
                "parent_solves": row.get("parent_solves"),
                "child_solves": row.get("child_solves"),
                "cache_hits": row.get("relaxation_cache_hits"),
                "cache_misses": row.get("relaxation_cache_misses"),
                "termination": row.get("termination"),
                "scorer_rejection_reasons": json.dumps(row.get("scorer_rejection_reasons", {}), sort_keys=True),
                "work_id": row["work_id"],
                "result_file": row["result_file"],
            }
        )
    terminal_columns = list(terminal_rows[0])
    write_csv(raw / "terminal_flow_runs.csv", terminal_rows, terminal_columns)

    summary_rows: list[dict[str, Any]] = []
    for method, rows_by_id in sorted(remote_by_method.items()):
        rows = list(rows_by_id.values())
        normal = [r for r in rows if r["normal_completion"]]
        successes = [r for r in normal if r["success"]]
        summary_rows.append(
            {
                "method": method,
                "terminal_rows": len(rows),
                "normal_completions": len(normal),
                "done": sum(r["status"] == "DONE" for r in rows),
                "valid_mapping_failure": sum(r["status"] == "VALID_MAPPING_FAILURE" for r in rows),
                "timeouts": sum(r["status"] == "TIMEOUT" for r in rows),
                "errors": sum(r["status"] == "ERROR" for r in rows),
                "successes": len(successes),
                "success_rate_normal": len(successes) / len(normal) if normal else None,
                "median_success_wall_seconds": float(np.median([r["compile_wall_seconds"] for r in successes])) if successes else None,
            }
        )
    write_csv(tables / "real_method_summary.csv", summary_rows)
    (tables / "real_method_summary.md").write_text(
        markdown_table(summary_rows, list(summary_rows[0])), encoding="utf-8"
    )

    paired_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    for method in sorted(remote_by_method):
        method_rows = remote_by_method[method]
        ids = sorted(set(length) & set(method_rows))
        normal_ids = [key for key in ids if length[key]["normal_completion"] and method_rows[key]["normal_completion"]]
        for key in normal_ids:
            base, candidate = length[key], method_rows[key]
            paired_rows.append(
                {
                    "kernel": key[0],
                    "architecture": key[1],
                    "method": method,
                    "length_status": base["status"],
                    "method_status": candidate["status"],
                    "length_success": base["success"],
                    "method_success": candidate["success"],
                    "success_delta": int(candidate["success"]) - int(base["success"]),
                    "length_route_cost": base.get("route_cost"),
                    "method_route_cost": candidate.get("route_cost"),
                    "length_wall_seconds": base.get("compile_wall_seconds"),
                    "method_wall_seconds": candidate.get("compile_wall_seconds"),
                    "matched_budget": False,
                }
            )
        b = np.array([length[key]["success"] for key in normal_ids], dtype=bool)
        m = np.array([method_rows[key]["success"] for key in normal_ids], dtype=bool)
        delta, lo, hi = bootstrap_success_delta(b, m)
        common_success = [key for key in normal_ids if length[key]["success"] and method_rows[key]["success"]]
        route_delta = None
        if common_success:
            base_cost = np.array([length[k]["route_cost"] for k in common_success], dtype=float)
            method_cost = np.array([method_rows[k]["route_cost"] for k in common_success], dtype=float)
            route_delta = float(np.mean((method_cost - base_cost) / base_cost))
        bootstrap_rows.append(
            {
                "method": method,
                "paired_normal_instances": len(normal_ids),
                "length_successes": int(b.sum()),
                "method_successes": int(m.sum()),
                "success_delta_points": 100 * delta,
                "success_delta_ci_low_points": 100 * lo,
                "success_delta_ci_high_points": 100 * hi,
                "common_success_instances": len(common_success),
                "mean_common_success_route_cost_delta_fraction": route_delta,
                "matched_budget": False,
                "interpretation": "extended-budget diagnostic; 7200 s Flow watchdog versus heterogeneous length retries",
            }
        )
    write_csv(tables / "paired_length_comparison.csv", paired_rows)
    write_csv(tables / "paired_bootstrap_summary.csv", bootstrap_rows)
    (tables / "paired_bootstrap_summary.md").write_text(
        markdown_table(bootstrap_rows, list(bootstrap_rows[0])), encoding="utf-8"
    )

    readiness_rows = [
        {"category": "adapter_and_functional_correctness", "weight": 20, "earned": 12.0, "basis": "6/9 legality round trips; 8/8 negative tests; 0 cycle simulations"},
        {"category": "principal_12x4_real_matrix", "weight": 35, "earned": 5.5, "basis": "30 normal Flow cells across 10 pair identities; required matrix remains sparse"},
        {"category": "conventional_baselines", "weight": 10, "earned": 0.0, "basis": "no matched PathFinder and three-seed annealing matrix"},
        {"category": "lookahead_and_parent_ablations", "weight": 10, "earned": 1.0, "basis": "strong synthetic evidence; no real no-parent/full-lookahead comparison"},
        {"category": "portability_and_scaling", "weight": 10, "earned": 0.5, "basis": "A3 attempted but coverage failed or timed out"},
        {"category": "runtime_and_statistics", "weight": 5, "earned": 2.0, "basis": "atomic timings and bootstrap reconstruction; no matched budgets/cold-warm study"},
        {"category": "artifact_and_reporting", "weight": 10, "earned": 5.0, "basis": "hashed atomic artifacts exist; current paper tables/reproduction are incomplete"},
    ]
    write_csv(tables / "paper_readiness.csv", readiness_rows)
    (tables / "paper_readiness.md").write_text(
        markdown_table(readiness_rows, list(readiness_rows[0])), encoding="utf-8"
    )

    status_counts = Counter(row["status"] for row in remote)
    report = {
        "source_commit": remote[0].get("source_commit") if remote else None,
        "checkpoint_hashes": sorted({r.get("checkpoint_hash") for r in remote if r.get("checkpoint_hash")}),
        "flow_status_counts": dict(status_counts),
        "flow_pair_identities": len({(r["kernel"], r["architecture"]) for r in remote}),
        "flow_kernels": sorted({r["kernel"] for r in remote}),
        "flow_architectures": sorted({r["architecture"] for r in remote}),
        "best_available_length_pairs": len(length),
        "paper_readiness_percent": sum(row["earned"] for row in readiness_rows),
        "paired_summaries": bootstrap_rows,
    }
    (raw / "audit_summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    manifest_targets = [args.snapshot / "work_manifest.csv", *(root / "work_manifest.csv" for root in args.length_root)]
    manifest_targets += sorted((args.snapshot / "work_items").glob("*.json"))
    with (output / "SOURCE_ARTIFACTS.sha256").open("w", encoding="utf-8") as handle:
        for path in manifest_targets:
            handle.write(f"{sha256(path)}  {path}\n")


if __name__ == "__main__":
    main()
