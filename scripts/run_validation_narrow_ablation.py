#!/usr/bin/env python
"""Second validation-only study: narrow beams and critical-link predictions."""

from __future__ import annotations

import concurrent.futures
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quotientflow.dfg import FAMILIES
from quotientflow.mapper import map_dfg
from quotientflow.model import QuotientFlowGNN, predict_numpy


def worker(payload):
    sample, regression, critical = payload
    settings = {
        "length": None,
        "oracle_top2_w0.5": sample["normalized_prices"],
        "oracle_sum_w0.5": sample["normalized_prices"],
        "pred_reg_top2_w0.5": regression,
        "pred_critical_top2_w0.5": critical,
    }
    rows = []
    for beam in (2, 3):
        for method, prices in settings.items():
            mode = "sum" if "sum" in method else "top2"
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
                price_weight=0.5,
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
    checkpoint = torch.load(
        ROOT / "results/quick_v1/checkpoints/model_seed_11.pt",
        map_location="cuda",
        weights_only=False,
    )
    model = QuotientFlowGNN().cuda()
    model.load_state_dict(checkpoint["state_dict"])
    cache = ROOT / "results/cache/samples"
    samples = []
    for family in FAMILIES:
        for arch in ("mesh3", "torus3"):
            path = sorted(cache.glob(f"val_seen__{arch}__{family}__*.pkl"))[0]
            with path.open("rb") as handle:
                sample = pickle.load(handle)
            regression, logits = predict_numpy(
                model, sample["dfg"], sample["arch"], sample["state"]
            )
            samples.append((sample, regression, 1.0 / (1.0 + np.exp(-logits))))
    rows = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=6) as executor:
        for result in executor.map(worker, samples):
            rows.extend(result)
    frame = pd.DataFrame(rows)
    output = ROOT / "results/revision_v1"
    frame.to_csv(output / "validation_narrow_ablation.csv", index=False)
    summary = (
        frame.groupby(["beam_width", "method"], as_index=False)
        .agg(
            success_rate=("success", "mean"),
            successes=("success", "sum"),
            median_expansions=("expansions", "median"),
            mean_cost=("best_cost", "mean"),
            timeouts=("timeout", "sum"),
        )
    )
    base = summary[summary["method"] == "length"][
        ["beam_width", "success_rate", "median_expansions"]
    ].rename(
        columns={
            "success_rate": "length_success",
            "median_expansions": "length_expansions",
        }
    )
    summary = summary.merge(base, on="beam_width")
    summary["success_delta_points"] = 100 * (
        summary["success_rate"] - summary["length_success"]
    )
    summary["expansion_reduction_pct"] = 100 * (
        summary["length_expansions"] - summary["median_expansions"]
    ) / summary["length_expansions"]
    summary.to_csv(output / "validation_narrow_ablation_summary.csv", index=False)
    print(
        summary.sort_values(
            ["success_delta_points", "expansion_reduction_pct"],
            ascending=False,
        ).to_string(index=False)
    )


if __name__ == "__main__":
    main()
