#!/usr/bin/env python3
"""Materialize an honest report from an atomic native baseline queue.

This is deliberately a status/reporting utility rather than a mapper.  It
merges the durable manifest with each published result, preserves every
terminal status, and never converts timeout/error rows into mapping failures.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _payload(row: dict[str, str]) -> dict[str, Any]:
    result = Path(row["result_path"])
    if not result.is_file():
        return {}
    value = json.loads(result.read_text(encoding="utf-8"))
    if value.get("work_id") != row["work_id"]:
        raise RuntimeError(f"result identity mismatch: {result}")
    if value.get("config_hash") != row["config_hash"]:
        raise RuntimeError(f"result config mismatch: {result}")
    return value


def _number(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    queue = args.queue.resolve()
    output = args.output.resolve()
    rows = _read(queue / "work_manifest.csv")
    merged: list[dict[str, Any]] = []
    for row in rows:
        value = _payload(row)
        merged.append({**row, **value, "result_path": row["result_path"]})

    raw = output / "raw"
    tables = output / "tables"
    raw.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    _atomic_csv(raw / "length_medium_runs.csv", merged)

    status = Counter(row.get("status", "MISSING") for row in merged)
    by_pair: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_arch: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_kernel: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in merged:
        by_pair[(row.get("kernel", ""), row.get("architecture", ""))].append(row)
        by_arch[row.get("architecture", "")].append(row)
        by_kernel[row.get("kernel", "")].append(row)

    def summarize(group: list[dict[str, Any]], label: str, value: str) -> dict[str, Any]:
        done = [r for r in group if r.get("status") in {"DONE", "VALID_MAPPING_FAILURE"}]
        valid_failures = [r for r in group if r.get("status") == "VALID_MAPPING_FAILURE"]
        legal_success = [r for r in group if r.get("status") == "DONE" and str(r.get("success", "")).lower() == "true"]
        route_costs = [_number(r.get("route_cost")) for r in group if r.get("status") == "DONE"]
        route_costs = [x for x in route_costs if x is not None]
        return {
            label: value,
            "total_rows": len(group),
            "normal_terminal_rows": len(done),
            "done_rows": sum(r.get("status") == "DONE" for r in group),
            "valid_mapping_failure_rows": len(valid_failures),
            "timeout_rows": sum(r.get("status") == "TIMEOUT" for r in group),
            "error_rows": sum(r.get("status") == "ERROR" for r in group),
            "success_rows": len(legal_success),
            "success_rate_over_normal_rows": (len(legal_success) / len(done) if done else ""),
            "mean_route_cost_done": (sum(route_costs) / len(route_costs) if route_costs else ""),
            "mean_wall_seconds_terminal": (
                sum(_number(r.get("wall_seconds")) or 0.0 for r in group) / len(group)
                if group else ""
            ),
        }

    coverage = [summarize(merged, "scope", "all")]
    coverage.extend(summarize(group, "architecture", key) for key, group in sorted(by_arch.items()))
    coverage.extend(summarize(group, "kernel", key) for key, group in sorted(by_kernel.items()))
    _atomic_csv(tables / "length_medium_coverage.csv", coverage)

    pair_rows = [
        {
            **summarize(group, "kernel", kernel),
            "architecture": architecture,
            "work_ids": ";".join(sorted(r["work_id"] for r in group)),
            "statuses": ";".join(f"{k}:{v}" for k, v in sorted(Counter(r.get("status", "MISSING") for r in group).items())),
        }
        for (kernel, architecture), group in sorted(by_pair.items())
    ]
    _atomic_csv(tables / "length_medium_by_pair.csv", pair_rows)

    lines = [
        "# Real native length-medium baseline status",
        "",
        f"Queue: `{queue}`",
        "",
        "This report is generated only from atomic manifest rows and published result JSON files.",
        "Timeouts and errors are retained as operational statuses and are not counted as mapping failures.",
        "",
        "## Status counts",
        "",
        "| status | rows |",
        "|---|---:|",
    ]
    lines.extend(f"| {key} | {value} |" for key, value in sorted(status.items()))
    lines.extend([
        "",
        "## Interpretation",
        "",
        f"Normal terminal rows (`DONE` or `VALID_MAPPING_FAILURE`): {status['DONE'] + status['VALID_MAPPING_FAILURE']} / {len(merged)}.",
        "A legal success is counted only for a `DONE` result whose mapper result reports `success=true` and whose worker-side legality check passed.",
        "No scientific comparison is reported here until the paired methods use the same normally completed instance set.",
        "",
    ])
    (output / "PAPER_LENGTH_BASELINE_STATUS.md").parent.mkdir(parents=True, exist_ok=True)
    (output / "PAPER_LENGTH_BASELINE_STATUS.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"rows": len(merged), "status": status, "output": str(output)}, default=dict, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

