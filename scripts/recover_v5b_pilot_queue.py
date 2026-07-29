#!/usr/bin/env python3
"""Recover stale revision-v5b pilot workers without rerunning completed work."""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.run_v5b_pilot_queue import recover
from scripts.v5b_pilot_queue_common import atomic_csv, read_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stale-seconds", type=float, default=35.0)
    arguments = parser.parse_args()
    manifest = arguments.output.resolve() / "work_manifest.csv"
    rows = read_manifest(manifest)
    recovered = recover(rows, arguments.stale_seconds)
    atomic_csv(manifest, rows)
    print(f"recovered={recovered} rows={len(rows)} manifest={manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
