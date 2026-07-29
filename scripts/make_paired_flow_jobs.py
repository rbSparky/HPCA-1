#!/usr/bin/env python3
"""Build fresh, source-hash-safe FlowAdvantage jobs for completed baselines.

Only identities with a normally completed baseline row are selected.  Queue
state and source hashes are intentionally discarded so ``run_v5b_pilot_queue``
re-freezes them against the current checked-out implementation.  The parent
baseline manifests remain immutable evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


DERIVED_FIELDS = {
    "source_commit",
    "source_tree_hash",
    "config_hash",
    "work_id",
    "status",
    "attempt",
    "result_path",
    "heartbeat_path",
    "stdout_path",
    "stderr_path",
    "worker_pid",
    "worker_pid_create_time",
    "worker_host",
    "start_time",
    "last_heartbeat",
    "end_time",
    "wall_seconds",
    "cpu_seconds",
    "peak_rss_mb",
    "exit_code",
    "error_type",
    "error_message",
}


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--baseline-manifest", type=Path, action="append", required=True)
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    args = parser.parse_args()

    payload = json.loads(args.jobs.read_text(encoding="utf-8"))
    source_jobs = payload["jobs"] if isinstance(payload, dict) else payload
    baseline_rows = [row for manifest in args.baseline_manifest for row in rows(manifest)]
    paired = {
        (row["kernel"], row["architecture"], str(row.get("seed", "0")))
        for row in baseline_rows
        if row.get("status") in {"DONE", "VALID_MAPPING_FAILURE"}
    }
    selected: list[dict[str, object]] = []
    missing: list[tuple[str, str, str]] = []
    for identity in sorted(paired):
        kernel, architecture, seed = identity
        for method in args.methods:
            matches = [
                row
                for row in source_jobs
                if row.get("kernel") == kernel
                and row.get("architecture") == architecture
                and str(row.get("seed")) == seed
                and row.get("method") == method
            ]
            if not matches:
                missing.append((kernel, architecture, method))
                continue
            job = dict(matches[0])
            job["timeout_seconds"] = args.timeout_seconds
            job["budget_seconds"] = args.timeout_seconds
            job["method"] = method
            for field in DERIVED_FIELDS:
                job.pop(field, None)
            job["paired_baseline_manifests"] = [str(path.resolve()) for path in args.baseline_manifest]
            selected.append(job)
    if missing:
        raise SystemExit("missing requested method rows: " + ", ".join("/".join(x) for x in missing))
    if not selected:
        raise SystemExit("no normally completed baseline identities available")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "schema": "flowadvantage_v5b_paired_jobs_v1",
                "methods": list(args.methods),
                "baseline_manifests": [str(path.resolve()) for path in args.baseline_manifest],
                "paired_identities": sorted("/".join(x) for x in paired),
                "jobs": selected,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"paired_identities": len(paired), "jobs": len(selected), "methods": args.methods}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
