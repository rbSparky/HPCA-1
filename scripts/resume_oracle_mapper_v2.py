#!/usr/bin/env python3
"""Parallel, cache-resuming completion of the bounded oracle mapper subset."""

from __future__ import annotations

import concurrent.futures
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quotientflow.dfg import FAMILIES
from scripts.run_action_mapper_v2 import (
    OUTPUT,
    RAW,
    TEST_SPLITS,
    load_mapping_samples,
    run_oracle,
)


def main():
    start = time.perf_counter()
    output_path = RAW / "action_mapper_runs.csv"
    frame = pd.read_csv(output_path)
    samples = load_mapping_samples()
    oracle_ids = set()
    for split in ("val_seen", *TEST_SPLITS):
        subset = [sample for sample in samples if sample["split"] == split]
        for family in FAMILIES:
            oracle_ids.add(
                next(
                    sample["sample_id"]
                    for sample in subset
                    if sample["family"] == family
                )
            )
    completed = set(
        frame[frame["method"] == "oracle_action_lookahead"]["sample_id"]
    )
    remaining = [
        sample
        for sample in samples
        if sample["sample_id"] in oracle_ids
        and sample["sample_id"] not in completed
    ]
    rows = frame.to_dict("records")
    with concurrent.futures.ProcessPoolExecutor(max_workers=6) as executor:
        for row in executor.map(run_oracle, remaining):
            rows.append(row)
            pd.DataFrame(rows).to_csv(output_path, index=False)
    final = pd.DataFrame(rows)
    predictions = pd.read_csv(RAW / "action_predictions.csv")
    initial = predictions[predictions["selected_action"]].copy()
    method_lookup = {
        "length": ("length", -1),
        "heuristic_scarcity": ("heuristic_scarcity", -1),
        "pred_link_scarcity": ("predicted_link_price", -1),
        "mlp_action_lookahead": ("action_mlp", 11),
        "gnn_action_lookahead": ("action_gnn", 37),
        "oracle_action_lookahead": ("oracle_q_rel", -1),
    }
    regret = {
        (row.state_id, row.model, row.model_seed): row.selected_action_normalized_regret
        for row in initial.itertuples()
    }
    final["mean_selected_action_oracle_regret_available_state"] = [
        regret.get((row.sample_id, *method_lookup[row.method]), float("nan"))
        for row in final.itertuples()
    ]
    final.to_csv(output_path, index=False)
    runtime = {
        "parallel_oracle_resume_seconds": time.perf_counter() - start,
        "oracle_completed_before_resume": len(completed),
        "oracle_completed_after_resume": int(
            (final["method"] == "oracle_action_lookahead").sum()
        ),
        "total_rows": len(final),
    }
    (OUTPUT / "oracle_resume_runtime.json").write_text(
        json.dumps(runtime, indent=2) + "\n"
    )
    print(json.dumps(runtime, indent=2))


if __name__ == "__main__":
    main()
