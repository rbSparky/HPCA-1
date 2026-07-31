#!/usr/bin/env python3
"""Materialize the Hail-Mary real-pilot evidence from atomic queue outputs.

The script is deliberately fail-closed: it preserves every queue status,
keeps timeout/error/unsupported rows out of quality denominators, and emits
paired summaries only for identical kernel/architecture/seed keys.  It never
edits previous revision directories.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NORMAL = {"DONE", "VALID_MAPPING_FAILURE", "LEGAL_SUCCESS"}
QUALITY_SUCCESS = {"DONE", "LEGAL_SUCCESS"}


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    if not fields:
        fields = ["empty"]
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value for key, value in row.items()})
            handle.flush(); os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text); handle.flush(); os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_queue(queue: Path, label: str) -> list[dict[str, Any]]:
    manifest = queue / "work_manifest.csv"
    if not manifest.is_file():
        return []
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    out: list[dict[str, Any]] = []
    for row in rows:
        payload: dict[str, Any] = {}
        result = Path(row.get("result_path", ""))
        # Remote manifests contain absolute cluster paths.  After an
        # append-only rsync, resolve the same immutable basename in the local
        # queue directory rather than silently treating every row as missing.
        if not result.is_file():
            local_result = queue / "work_items" / result.name
            if local_result.is_file():
                result = local_result
        if result.is_file():
            try:
                payload = json.loads(result.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                row["status"] = "ERROR"
                row["error_type"] = "InvalidAtomicResult"
                row["error_message"] = str(error)
        merged = {**row, **payload}
        merged["source_queue"] = label
        merged["manifest_status"] = row.get("status", "")
        merged["result_path"] = str(result)
        # Static queue labels are implementation details; the analysis uses
        # the frozen scientific method names.
        merged["analysis_method"] = {
            "dual_linear_static": "dual_linear",
            "flow_proposal_static": "flow_proposal",
            "flow_top4_static": "flow_top4",
        }.get(str(merged.get("method")), str(merged.get("method", "")))
        out.append(merged)
    return out


def _num(row: dict[str, Any], key: str, default: float = math.nan) -> float:
    value = row.get(key, "")
    if value in (None, "", "null"):
        return default
    try: return float(value)
    except (TypeError, ValueError): return default


def paired_bootstrap(base: list[float], candidate: list[float], seed: int = 24072026, n: int = 10000) -> tuple[float, float, float, int]:
    if len(base) != len(candidate) or not base:
        return math.nan, math.nan, math.nan, 0
    x, y = np.asarray(base, dtype=float), np.asarray(candidate, dtype=float)
    delta = float(np.mean(y - x))
    rng = np.random.default_rng(seed)
    samples = np.empty(n, dtype=float)
    indices = np.arange(len(x))
    for i in range(n):
        take = rng.choice(indices, size=len(indices), replace=True)
        samples[i] = float(np.mean(y[take] - x[take]))
    return delta, float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975)), len(x)


def markdown(rows: list[dict[str, Any]]) -> str:
    if not rows: return "_No rows._\n"
    fields = list(rows[0])
    cell = lambda value: str(value if value is not None else "").replace("|", "\\|").replace("\n", " ")
    return "| " + " | ".join(fields) + " |\n|" + "|".join("---" for _ in fields) + "|\n" + "\n".join("| " + " | ".join(cell(r.get(f, "")) for f in fields) + " |" for r in rows) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT / "results/hail_mary_hpca1")
    parser.add_argument("--queues", nargs="+", required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    queues = [(Path(item).resolve(), Path(item).name) for item in args.queues]
    records: list[dict[str, Any]] = []
    for queue, label in queues:
        records.extend(load_queue(queue, label))
    # Work IDs are immutable semantic identities.  If the same item appears
    # in a diagnostic queue and a clean queue, keep the clean/current record
    # and retain the duplicate in the provenance census.
    by_id: dict[str, dict[str, Any]] = {}
    duplicates: list[dict[str, Any]] = []
    for row in records:
        wid = str(row.get("work_id", ""))
        if wid in by_id:
            duplicates.append({"work_id": wid, "first_queue": by_id[wid].get("source_queue", ""), "duplicate_queue": row.get("source_queue", "")})
        by_id[wid] = row
    records = list(by_id.values())
    for row in records:
        row["normal_completion"] = str(row.get("status")) in NORMAL
        row["quality_success"] = bool(row.get("success") is True or str(row.get("success", "")).lower() == "true") and str(row.get("status")) in QUALITY_SUCCESS
        row["legal_success"] = row["quality_success"] and str(row.get("legal", "")).lower() in {"true", "1"} and str(row.get("flowadvantage_legality", "")).lower() in {"true", "1", ""} and str(row.get("morpher_legality", "")).lower() in {"true", "1", ""}
    raw = root / "raw"
    atomic_csv(raw / "primary_atomic_results.csv", records)
    atomic_csv(raw / "duplicate_work_items.csv", duplicates)
    ii_rows: list[dict[str, Any]] = []
    for row in records:
        attempts = row.get("ii_attempts", [])
        if isinstance(attempts, str):
            try: attempts = json.loads(attempts)
            except json.JSONDecodeError: attempts = []
        for attempt in attempts or []:
            ii_rows.append({"work_id": row.get("work_id"), "kernel": row.get("kernel"), "architecture": row.get("architecture"), "analysis_method": row.get("analysis_method"), "seed": row.get("seed"), **attempt})
    atomic_csv(raw / "ii_attempts.csv", ii_rows)
    status_rows = [{"status": status, "count": count} for status, count in sorted(Counter(str(r.get("status", "")) for r in records).items())]
    atomic_csv(root / "tables/status_census.csv", status_rows)
    atomic_text(root / "tables/status_census.md", markdown(status_rows))
    coverage: list[dict[str, Any]] = []
    for method in sorted({str(r.get("analysis_method", "")) for r in records}):
        subset = [r for r in records if str(r.get("analysis_method")) == method]
        coverage.append({"method": method, "rows": len(subset), "normal_completions": sum(bool(r["normal_completion"]) for r in subset), "legal_successes": sum(bool(r["legal_success"]) for r in subset), "timeouts": sum(str(r.get("status")) == "TIMEOUT" for r in subset), "errors": sum(str(r.get("status")) in {"ERROR", "SOLVER_REJECTED", "VALIDATOR_FAILURE"} for r in subset), "valid_failures": sum(str(r.get("status")) == "VALID_MAPPING_FAILURE" for r in subset)})
    atomic_csv(root / "tables/method_coverage.csv", coverage)
    atomic_text(root / "tables/method_coverage.md", markdown(coverage))
    # Per-kernel/architecture coverage is the only acceptable denominator for
    # headline comparisons; no unequal aggregate is silently formed.
    pair_rows: list[dict[str, Any]] = []
    # A method is allowed to have been attempted in more than one append-only
    # queue while rescuing a stalled run.  Pairing by queue would silently
    # produce zero denominators (length and top-4 naturally live in different
    # queues).  Collapse only semantic duplicates here, retaining every raw
    # row above.  Prefer a current-source, terminal result and then the newest
    # finished result; this selection is deterministic and is recorded in the
    # pair table through ``source_queue``.
    def _selection_rank(row: dict[str, Any]) -> tuple[int, int, int, float]:
        status = str(row.get("status", ""))
        terminal = int(status in {"DONE", "VALID_MAPPING_FAILURE", "LEGAL_SUCCESS", "TIMEOUT", "ERROR", "SOLVER_REJECTED", "VALIDATOR_FAILURE", "UNSUPPORTED", "CANCELLED"})
        current = int(str(row.get("source_commit", "")) == "1dbc3377672ce3d31f3a0ba45aaea240d96490ed")
        queue = str(row.get("source_queue", ""))
        queue_generation = 3 if queue.endswith("mt8f") else (2 if queue.endswith("mt8e") else (1 if queue.endswith("mt8d") else 0))
        try:
            finished = float(row.get("finished_at") or row.get("end_time") or 0.0)
        except (TypeError, ValueError):
            finished = 0.0
        return (current, terminal, queue_generation, finished)

    selected_by_semantic: dict[tuple[str, str, int, str], dict[str, Any]] = {}
    duplicate_semantic: list[dict[str, Any]] = []
    for row in records:
        key = (str(row.get("kernel")), str(row.get("architecture")), int(float(row.get("seed") or 0)), str(row.get("analysis_method")))
        previous = selected_by_semantic.get(key)
        if previous is None or _selection_rank(row) > _selection_rank(previous):
            if previous is not None:
                duplicate_semantic.append({"semantic_key": "|".join(map(str, key)), "kept_queue": row.get("source_queue", ""), "discarded_queue": previous.get("source_queue", ""), "discarded_status": previous.get("status", "")})
            selected_by_semantic[key] = row
        else:
            duplicate_semantic.append({"semantic_key": "|".join(map(str, key)), "kept_queue": previous.get("source_queue", ""), "discarded_queue": row.get("source_queue", ""), "discarded_status": row.get("status", "")})
    records_for_pairing = list(selected_by_semantic.values())
    grouped: dict[tuple[str, str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in records_for_pairing:
        key = (str(row.get("kernel")), str(row.get("architecture")), int(float(row.get("seed") or 0)))
        grouped[key][str(row.get("analysis_method"))] = row
    for key, methods in sorted(grouped.items()):
        kernel, architecture, seed = key
        pair_rows.append({"kernel": kernel, "architecture": architecture, "seed": seed, **{f"{method}_source_queue": row.get("source_queue") for method, row in sorted(methods.items())}, **{f"{method}_status": row.get("status") for method, row in sorted(methods.items())}, **{f"{method}_success": row.get("legal_success") for method, row in sorted(methods.items())}})
    atomic_csv(root / "tables/pair_outcomes.csv", pair_rows)
    atomic_text(root / "tables/pair_outcomes.md", markdown(pair_rows))
    comparisons: list[dict[str, Any]] = []
    for baseline, candidate in (("length", "flow_top4"), ("length", "flow_proposal"), ("dual_linear", "flow_top4"), ("flow_proposal", "flow_top4")):
        pairs = []
        for key, methods in grouped.items():
            if baseline not in methods or candidate not in methods: continue
            left, right = methods[baseline], methods[candidate]
            if not left["normal_completion"] or not right["normal_completion"]: continue
            pairs.append((left, right))
        base_success = [1.0 if left["legal_success"] else 0.0 for left, _ in pairs]
        cand_success = [1.0 if right["legal_success"] else 0.0 for _, right in pairs]
        delta, lo, hi, n = paired_bootstrap(base_success, cand_success)
        route_pairs = [(left, right) for left, right in pairs if left["legal_success"] and right["legal_success"] and math.isfinite(_num(left, "route_cost")) and math.isfinite(_num(right, "route_cost"))]
        route_delta, route_lo, route_hi, route_n = paired_bootstrap([_num(left, "route_cost") for left, _ in route_pairs], [_num(right, "route_cost") for _, right in route_pairs])
        comparisons.append({"baseline": baseline, "candidate": candidate, "paired_normal_n": n, "success_delta": delta, "success_ci_low": lo, "success_ci_high": hi, "common_success_route_n": route_n, "route_cost_delta": route_delta, "route_cost_ci_low": route_lo, "route_cost_ci_high": route_hi})
    atomic_csv(raw / "bootstrap_results.csv", comparisons)
    atomic_csv(root / "tables/paired_comparisons.csv", comparisons)
    atomic_text(root / "tables/paired_comparisons.md", markdown(comparisons))
    atomic_csv(raw / "duplicate_semantic_work_items.csv", duplicate_semantic)
    summary = {"schema": "flowadvantage_hail_mary_results_v1", "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "queue_dirs": [str(q) for q, _ in queues], "rows": len(records), "pairing_rows": len(records_for_pairing), "normal_completions": sum(bool(r["normal_completion"]) for r in records), "legal_successes": sum(bool(r["legal_success"]) for r in records), "statuses": dict(Counter(str(r.get("status", "")) for r in records)), "duplicates": len(duplicates), "semantic_duplicates": len(duplicate_semantic), "bootstrap_seed": 24072026, "bootstrap_resamples": 10000}
    atomic_text(root / "real_pilot_aggregation.json", json.dumps(summary, indent=2, sort_keys=True) + "\n")
    report = ["# Hail-Mary real-pilot aggregation", "", f"Generated: {summary['generated_at_utc']}", "", "This report is generated exclusively from immutable atomic queue results. Operational timeouts/errors are retained and excluded from mapping-quality denominators.", "", "## Status census", "", markdown(status_rows), "## Method coverage", "", markdown(coverage), "## Paired comparisons", "", markdown(comparisons)]
    atomic_text(root / "HAIL_MARY_REAL_RESULTS.md", "\n".join(report))
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
