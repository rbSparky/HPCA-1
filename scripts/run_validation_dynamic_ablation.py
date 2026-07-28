#!/usr/bin/env python
"""Validation-only oracle price-refresh ablation."""

from __future__ import annotations

import concurrent.futures
import pickle
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quotientflow.dfg import FAMILIES
from quotientflow.mapper import map_dfg


def worker(path):
    with path.open("rb") as handle:
        sample = pickle.load(handle)
    rows = []
    for method, prices, refresh in (
        ("length", None, 0),
        ("oracle_static", sample["normalized_prices"], 0),
        ("oracle_refresh2", sample["normalized_prices"], 2),
    ):
        result = map_dfg(
            sample["dfg"],
            sample["arch"],
            sample["state"],
            prices=prices,
            beam_width=4,
            action_limit=6,
            k_paths=4,
            max_expansions=5000,
            timeout_seconds=120.0,
            price_weight=0.5,
            price_mode="top2",
            dynamic_refresh_every=refresh,
        )
        rows.append(
            {
                "sample_id": sample["sample_id"],
                "architecture": sample["architecture"],
                "dfg_family": sample["family"],
                "method": method,
                **result.row(),
            }
        )
    return rows


def main():
    cache = ROOT / "results/cache/samples"
    paths = [
        sorted(cache.glob(f"val_seen__{arch}__{family}__*.pkl"))[0]
        for family in FAMILIES
        for arch in ("mesh3", "torus3")
    ]
    rows = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=6) as executor:
        for result in executor.map(worker, paths):
            rows.extend(result)
    frame = pd.DataFrame(rows)
    output = ROOT / "results/revision_v1"
    frame.to_csv(output / "validation_dynamic_ablation.csv", index=False)
    summary = (
        frame.groupby("method", as_index=False)
        .agg(
            success_rate=("success", "mean"),
            successes=("success", "sum"),
            median_expansions=("expansions", "median"),
            mean_cost=("best_cost", "mean"),
            median_runtime_ms=("runtime_ms", "median"),
            timeouts=("timeout", "sum"),
        )
    )
    summary.to_csv(output / "validation_dynamic_ablation_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
