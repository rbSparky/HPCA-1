#!/usr/bin/env python3
"""Uncensor only timed-out oracle mappings using their completed solve caches."""

from __future__ import annotations

import concurrent.futures
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_action_mapper_v2 import RAW, load_mapping_samples, run_oracle


def retry(sample):
    return run_oracle(sample, timeout_seconds=600.0)


def main():
    start = time.perf_counter()
    path = RAW / "action_mapper_runs.csv"
    frame = pd.read_csv(path)
    timed_ids = set(
        frame[
            (frame["method"] == "oracle_action_lookahead")
            & frame["timeout"].astype(bool)
        ]["sample_id"]
    )
    samples = [
        sample
        for sample in load_mapping_samples()
        if sample["sample_id"] in timed_ids
    ]
    kept = frame[
        ~(
            (frame["method"] == "oracle_action_lookahead")
            & frame["sample_id"].isin(timed_ids)
        )
    ]
    rows = kept.to_dict("records")
    with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:
        for row in executor.map(retry, samples):
            row["source"] = "revision_v2_cost_to_go_oracle_uncensored_600s"
            rows.append(row)
            pd.DataFrame(rows).to_csv(path, index=False)
    result = pd.DataFrame(rows)
    metadata = {
        "runtime_seconds": time.perf_counter() - start,
        "retried_ids": sorted(timed_ids),
        "retried_count": len(timed_ids),
        "remaining_timeouts": int(
            result[result["method"] == "oracle_action_lookahead"][
                "timeout"
            ].sum()
        ),
    }
    (ROOT / "results" / "revision_v2" / "oracle_uncensoring.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
