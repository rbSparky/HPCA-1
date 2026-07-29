#!/usr/bin/env python3
"""Select only operationally timed-out baseline items for a longer retry.

The source job list remains immutable.  This writes a new job document with a
longer watchdog and an explicit provenance record; no completed row is
silently replaced.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    parser.add_argument("--include-initialization-errors", action="store_true")
    args = parser.parse_args()
    source = json.loads(args.source.read_text(encoding="utf-8"))
    jobs = source["jobs"] if isinstance(source, dict) else source
    parent_rows = csv_rows(args.manifest)
    retry_rows = [
        row for row in parent_rows
        if row.get("status") == "TIMEOUT"
        or (
            args.include_initialization_errors
            and row.get("status") == "ERROR"
            and "deterministic length-prefix initialization failed: NO_LEGAL_ACTION"
            in row.get("error_message", "")
        )
    ]
    retry_identities = {
        (row["kernel"], row["architecture"], row["method"], str(row["seed"]))
        for row in retry_rows
    }
    selected = []
    for row in jobs:
        # work_id is not present until queue freeze; the immutable semantic
        # identity is kernel/architecture/method/seed for this baseline.
        identity = (row["kernel"], row["architecture"], row["method"], str(row["seed"]))
        if identity in retry_identities:
            updated = dict(row)
            updated["timeout_seconds"] = args.timeout_seconds
            updated["budget_seconds"] = args.timeout_seconds
            updated["retry_parent_queue"] = str(args.manifest.resolve())
            selected.append(updated)
    if not selected:
        raise SystemExit("no TIMEOUT rows found in immutable parent manifest")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"jobs": selected}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"jobs": len(selected), "timeout_seconds": args.timeout_seconds, "source": str(args.source), "parent_manifest": str(args.manifest)}, sort_keys=True))
    return 0


def csv_rows(path: Path) -> list[dict[str, str]]:
    import csv

    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


if __name__ == "__main__":
    raise SystemExit(main())
