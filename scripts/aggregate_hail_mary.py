#!/usr/bin/env python3
"""Validate atomic paper results and atomically publish paired aggregates."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flowadvantage.paper_results import (
    ALL_STATUSES,
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    DEFAULT_PAIR_FIELDS,
    NONTERMINAL_STATUSES,
    comparison_rows,
    failure_census,
    load_atomic_records,
    record_rows,
    status_census,
)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _fields(rows: list[dict[str, Any]]) -> list[str]:
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for field in row:
            if field not in seen:
                fields.append(field)
                seen.add(field)
    return fields


def atomic_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str] | None = None) -> None:
    rows = list(rows)
    fields = fields or _fields(rows)
    if not fields:
        fields = ["empty"]
    output = []
    with tempfile.TemporaryFile(mode="w+", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: (
                        json.dumps(value, sort_keys=True, separators=(",", ":"))
                        if isinstance(value, (dict, list, tuple))
                        else value
                    )
                    for field, value in row.items()
                }
            )
        handle.seek(0)
        output.append(handle.read())
    atomic_text(path, "".join(output))


def markdown_table(rows: list[dict[str, Any]], fields: list[str] | None = None) -> str:
    fields = fields or _fields(rows)
    if not fields:
        return "_No rows._\n"
    def cell(value: Any) -> str:
        if isinstance(value, float):
            value = format(value, ".12g")
        return str(value if value is not None else "").replace("|", "\\|").replace("\n", " ")
    lines = [
        "| " + " | ".join(fields) + " |",
        "|" + "|".join("---" for _ in fields) + "|",
    ]
    lines.extend(
        "| " + " | ".join(cell(row.get(field, "")) for field in fields) + " |"
        for row in rows
    )
    return "\n".join(lines) + "\n"


def parse_comparison(value: str) -> tuple[str, str]:
    parts = value.split(":", 1)
    if len(parts) != 2 or not all(parts):
        raise argparse.ArgumentTypeError("comparison must be BASELINE:CANDIDATE")
    return parts[0], parts[1]


def update_result_index(root: Path, summary: dict[str, Any]) -> None:
    path = root / "result_index.csv"
    rows: list[dict[str, str]] = []
    if path.is_file():
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    fields = [
        "result_id", "category", "description", "state", "verdict",
        "source_paths", "paired_denominator", "primary_value",
        "confidence_interval", "config_hash", "source_hash",
        "generated_by", "last_updated_utc", "notes",
    ]
    row = {
        "result_id": "HAIL-ATOMIC-AGGREGATION",
        "category": "real_primary",
        "description": "Fail-closed atomic-result integrity and paired aggregation",
        "state": "VERIFIED" if summary["all_terminal"] else "RUNNING",
        "verdict": "PASS" if summary["all_terminal"] else "INCOMPLETE",
        "source_paths": "raw/atomic_results.csv;raw/ii_attempts.csv;raw/bootstrap_results.csv;tables/status_census.csv;tables/failure_census.csv",
        "paired_denominator": str(summary["normal_completions"]),
        "primary_value": f"{summary['terminal']}/{summary['manifest_rows']} terminal; {summary['normal_completions']} normal",
        "confidence_interval": "see raw/bootstrap_results.csv",
        "config_hash": "",
        "source_hash": summary["manifest_sha256"],
        "generated_by": "scripts/aggregate_hail_mary.py",
        "last_updated_utc": summary["generated_at_utc"],
        "notes": "Timeout/error/unsupported/cancelled rows retained and excluded from mapping-quality pairs",
    }
    by_id = {item.get("result_id"): item for item in rows}
    by_id[row["result_id"]] = row
    atomic_csv(path, by_id.values(), fields)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("results/hail_mary_hpca1"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--compare", type=parse_comparison, action="append", default=[])
    parser.add_argument("--pair-fields", default=",".join(DEFAULT_PAIR_FIELDS))
    parser.add_argument("--legality-policy", choices=("dual", "legacy"), default="dual")
    parser.add_argument("--no-update-tracker", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    manifest = (args.manifest or root / "work_manifest.csv").resolve()
    pair_fields = tuple(field.strip() for field in args.pair_fields.split(",") if field.strip())
    records, attempts = load_atomic_records(
        manifest, legality_policy=args.legality_policy
    )
    comparisons = comparison_rows(
        records,
        args.compare,
        pair_fields=pair_fields,
        resamples=BOOTSTRAP_RESAMPLES,
        seed=BOOTSTRAP_SEED,
    )
    statuses = status_census(records)
    failures = failure_census(records)
    rows = record_rows(records)
    min_ii = [
        {
            "work_id": row["work_id"], "kernel": row["kernel"],
            "architecture": row["architecture"], "method": row["method"],
            "seed": row["seed"], "status": row["status"],
            "success": row["success"], "minimum_ii": row["minimum_ii"],
            "lower_bound_ii": row.get("lower_bound_ii", ""),
            "distance_from_lower_bound": (
                int(row["minimum_ii"]) - int(row["lower_bound_ii"])
                if row.get("minimum_ii") not in (None, "")
                and row.get("lower_bound_ii") not in (None, "") else ""
            ),
            "result_sha256": row["result_sha256"],
        }
        for row in rows
    ]
    atomic_csv(root / "raw/atomic_results.csv", rows)
    atomic_csv(root / "raw/ii_attempts.csv", attempts)
    atomic_csv(root / "raw/bootstrap_results.csv", comparisons)
    for name, table in (
        ("status_census", statuses),
        ("failure_census", failures),
        ("minimum_ii", min_ii),
        ("paired_comparisons", comparisons),
    ):
        atomic_csv(root / f"tables/{name}.csv", table)
        atomic_text(root / f"tables/{name}.md", markdown_table(table))

    from datetime import datetime, timezone
    status_counts: dict[str, int] = {status: 0 for status in sorted(ALL_STATUSES)}
    for record in records:
        status_counts[record.status] += 1
    terminal = sum(
        count for status, count in status_counts.items()
        if status not in NONTERMINAL_STATUSES
    )
    normal = sum(status_counts[status] for status in ("LEGAL_SUCCESS", "VALID_MAPPING_FAILURE"))
    manifest_hash = __import__("hashlib").sha256(manifest.read_bytes()).hexdigest()
    summary = {
        "schema": "flowadvantage_hail_mary_aggregation_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "manifest": str(manifest),
        "manifest_sha256": manifest_hash,
        "manifest_rows": len(records),
        "terminal": terminal,
        "normal_completions": normal,
        "all_terminal": terminal == len(records),
        "status_counts": status_counts,
        "legality_policy": args.legality_policy,
        "pair_fields": pair_fields,
        "comparisons": [f"{a}:{b}" for a, b in args.compare],
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }
    atomic_text(root / "aggregation_summary.json", json.dumps(summary, indent=2, sort_keys=True) + "\n")
    update_result_index(root, summary)
    if not args.no_update_tracker and (root / "reviewer_actions.csv").is_file():
        subprocess.run(
            [sys.executable, str(ROOT / "scripts/update_hail_mary_tracker.py"), "--root", str(root)],
            check=True,
        )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
