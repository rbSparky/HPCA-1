#!/usr/bin/env python
"""Validation-only search-integration study; never reads test samples."""

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


def run_sample(path: Path):
    with path.open("rb") as handle:
        sample = pickle.load(handle)
    rows = []
    for beam in (4, 8, 16, 32):
        settings = [("length", None, 0.0, "sum")]
        settings += [
            (f"oracle_{mode}_w{weight}", sample["normalized_prices"], weight, mode)
            for mode in ("sum", "max", "top2")
            for weight in (0.5, 1.0, 2.0, 4.0)
        ]
        for method, prices, weight, mode in settings:
            result = map_dfg(
                sample["dfg"],
                sample["arch"],
                sample["state"],
                prices=prices,
                beam_width=beam,
                action_limit=6,
                k_paths=4,
                max_expansions=5000,
                timeout_seconds=120.0,
                price_weight=weight,
                price_mode=mode,
            )
            rows.append(
                {
                    "sample_id": sample["sample_id"],
                    "architecture": sample["architecture"],
                    "dfg_family": sample["family"],
                    "beam_width": beam,
                    "method": method,
                    **result.row(),
                }
            )
    return rows


def main():
    cache = ROOT / "results" / "cache" / "samples"
    # Exactly two validation samples per family, one from each seen architecture.
    paths = []
    for family in FAMILIES:
        for arch in ("mesh3", "torus3"):
            paths.append(sorted(cache.glob(f"val_seen__{arch}__{family}__*.pkl"))[0])
    output = ROOT / "results" / "revision_v1"
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=6) as executor:
        for result in executor.map(run_sample, paths):
            rows.extend(result)
            pd.DataFrame(rows).sort_values(
                ["beam_width", "method", "sample_id"]
            ).to_csv(output / "validation_mapper_ablation.csv", index=False)
    frame = pd.DataFrame(rows)
    summary = (
        frame.groupby(["beam_width", "method"], as_index=False)
        .agg(
            success_rate=("success", "mean"),
            timeouts=("timeout", "sum"),
            median_expansions=("expansions", "median"),
            median_runtime_ms=("runtime_ms", "median"),
            mean_best_cost=("best_cost", "mean"),
        )
    )
    baseline = summary[summary["method"] == "length"][
        ["beam_width", "success_rate", "median_expansions"]
    ].rename(
        columns={
            "success_rate": "length_success",
            "median_expansions": "length_expansions",
        }
    )
    summary = summary.merge(baseline, on="beam_width")
    summary["success_delta_points"] = 100 * (
        summary["success_rate"] - summary["length_success"]
    )
    summary["expansion_reduction_pct"] = 100 * (
        summary["length_expansions"] - summary["median_expansions"]
    ) / summary["length_expansions"]
    summary.to_csv(output / "validation_mapper_ablation_summary.csv", index=False)
    print(
        summary.sort_values(
            ["success_delta_points", "expansion_reduction_pct"],
            ascending=False,
        ).head(20).to_string(index=False)
    )


if __name__ == "__main__":
    main()
