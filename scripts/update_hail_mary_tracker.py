#!/usr/bin/env python3
"""Atomically refresh the generated progress block in HAIL_MARY.md."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import tempfile
from collections import Counter
from pathlib import Path


BEGIN = "<!-- GENERATED_PROGRESS_BEGIN -->"
END = "<!-- GENERATED_PROGRESS_END -->"
TERMINAL = {"VERIFIED"}
VALID_STATES = {"NOT_STARTED", "RUNNING", "VERIFIED", "BLOCKED"}
DEADLINE_UTC = dt.datetime(2026, 8, 1, 4, 30, tzinfo=dt.timezone.utc)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def atomic_text(path: Path, value: str) -> None:
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("results/hail_mary_hpca1"))
    args = parser.parse_args()
    root = args.root.resolve()
    actions = read_rows(root / "reviewer_actions.csv")
    results = read_rows(root / "result_index.csv")
    weights = [float(row["weight_percent"]) for row in actions]
    if abs(sum(weights) - 100.0) > 1e-9:
        raise ValueError(f"reviewer action weights must sum to 100, got {sum(weights)}")
    bad = sorted({row["state"] for row in actions if row["state"] not in VALID_STATES})
    if bad:
        raise ValueError(f"invalid reviewer-action states: {bad}")
    complete = sum(
        float(row["weight_percent"])
        for row in actions
        if row["state"] in TERMINAL
    )
    blocks = 25
    filled = min(blocks, int(round(complete / 100.0 * blocks)))
    bar = "█" * filled + "░" * (blocks - filled)
    now = dt.datetime.now(dt.timezone.utc)
    hours = max(0.0, (DEADLINE_UTC - now).total_seconds() / 3600.0)
    action_counts = Counter(row["state"] for row in actions)
    result_counts = Counter(row["state"] for row in results)
    generated = "\n".join(
        [
            BEGIN,
            f"**Paper-ready evidence progress:** `{bar}` **{complete:.1f}%**",
            "",
            f"**Hard deadline:** 2026-08-01 10:00 IST — **{hours:.2f} hours remaining**",
            "",
            f"**Last refreshed:** {now.isoformat(timespec='seconds')}",
            "",
            "| Ledger | NOT_STARTED | RUNNING | VERIFIED | BLOCKED |",
            "|---|---:|---:|---:|---:|",
            f"| Reviewer actions | {action_counts['NOT_STARTED']} | {action_counts['RUNNING']} | {action_counts['VERIFIED']} | {action_counts['BLOCKED']} |",
            f"| Indexed results | {result_counts['NOT_STARTED']} | {result_counts['RUNNING']} | {result_counts['VERIFIED']} | {result_counts['BLOCKED']} |",
            END,
        ]
    )
    tracker = root / "HAIL_MARY.md"
    text = tracker.read_text(encoding="utf-8")
    if text.count(BEGIN) != 1 or text.count(END) != 1:
        raise ValueError("HAIL_MARY.md must contain exactly one generated progress block")
    before, rest = text.split(BEGIN, 1)
    _, after = rest.split(END, 1)
    atomic_text(tracker, before + generated + after)
    print(f"updated {tracker}: {complete:.1f}% complete, {hours:.2f} hours remain")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
