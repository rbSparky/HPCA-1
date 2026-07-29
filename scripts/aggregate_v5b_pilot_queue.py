#!/usr/bin/env python3
"""Materialize reviewable pilot/II tables from immutable atomic results."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from scripts.v5b_pilot_queue_common import read_manifest

PILOT_FIELDS = (
    "work_id",
    "kernel",
    "architecture",
    "method",
    "seed",
    "status",
    "success",
    "legal",
    "ii",
    "termination",
    "operation_count",
    "mapped_operations",
    "route_count",
    "route_cost",
    "expansions",
    "generated_actions",
    "routing_attempts",
    "parent_solves",
    "child_solves",
    "compile_wall_seconds",
    "feature_seconds",
    "proposal_seconds",
    "relaxation_seconds",
    "native_mapper_seconds",
    "config_hash",
    "source_commit",
    "source_tree_hash",
    "toolchain_hash",
    "dfg_hash",
    "architecture_hash",
    "checkpoint_hash",
    "result_path",
)
II_FIELDS = (
    "work_id",
    "kernel",
    "architecture",
    "method",
    "seed",
    "ii",
    "status",
    "started_offset_seconds",
    "config_hash",
)


def _write(path: Path, fields: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    output = arguments.output.resolve()
    manifest = read_manifest(output / "work_manifest.csv")
    pilot_rows: list[dict[str, Any]] = []
    ii_rows: list[dict[str, Any]] = []
    for manifest_row in manifest:
        result_path = Path(manifest_row["result_path"])
        payload: dict[str, Any] = {}
        if result_path.is_file():
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            if payload.get("work_id") != manifest_row["work_id"]:
                raise RuntimeError(
                    f"identity mismatch in atomic result {result_path}"
                )
            if payload.get("config_hash") != manifest_row["config_hash"]:
                raise RuntimeError(
                    f"configuration mismatch in atomic result {result_path}"
                )
        merged = {**manifest_row, **payload, "result_path": str(result_path)}
        pilot_rows.append({field: merged.get(field, "") for field in PILOT_FIELDS})
        for attempt in payload.get("ii_attempts", []):
            ii_rows.append(
                {
                    "work_id": manifest_row["work_id"],
                    "kernel": manifest_row["kernel"],
                    "architecture": manifest_row["architecture"],
                    "method": manifest_row["method"],
                    "seed": manifest_row["seed"],
                    "ii": attempt.get("ii", ""),
                    "status": attempt.get("status", ""),
                    "started_offset_seconds": attempt.get(
                        "started_offset_seconds", ""
                    ),
                    "config_hash": manifest_row["config_hash"],
                }
            )
    _write(output / "raw/pilot_runs.csv", PILOT_FIELDS, pilot_rows)
    _write(output / "raw/ii_attempts.csv", II_FIELDS, ii_rows)
    print(
        json.dumps(
            {
                "manifest_rows": len(manifest),
                "pilot_rows": len(pilot_rows),
                "ii_attempt_rows": len(ii_rows),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
