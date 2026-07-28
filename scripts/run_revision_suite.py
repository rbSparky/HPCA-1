#!/usr/bin/env python
"""Frozen revision selected exclusively from validation_mapper_ablation.csv."""

from __future__ import annotations

import concurrent.futures
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quotientflow.dfg import FAMILIES
from quotientflow.mapper import map_dfg
from quotientflow.metrics import median, reduction_percent
from quotientflow.model import QuotientFlowGNN, predict_numpy

SELECTION = {
    "source": "results/revision_v1/validation_mapper_ablation_summary.csv",
    "beam_width": 4,
    "price_mode": "top2",
    "price_weight": 0.5,
    "selection_reason": (
        "Validation success improved from 92.857% to 100%; tied best success "
        "and slightly lower mean successful cost than sum_w0.5."
    ),
    "test_tuning_permitted": False,
}
METHODS = [
    "length",
    "quotient_length",
    "oracle_dual",
    "oracle_dual_quotient",
    "pred_dual",
    "pred_dual_quotient",
]
SPLITS = [
    "test_seen",
    "test_unseen_size",
    "test_unseen_topology",
    "test_unseen_asym",
]


def run_sample(payload):
    sample, predicted = payload
    definitions = {
        "length": (None, False),
        "quotient_length": (None, True),
        "oracle_dual": (sample["normalized_prices"], False),
        "oracle_dual_quotient": (sample["normalized_prices"], True),
        "pred_dual": (predicted, False),
        "pred_dual_quotient": (predicted, True),
    }
    rows = []
    for method in METHODS:
        prices, quotient = definitions[method]
        result = map_dfg(
            sample["dfg"],
            sample["arch"],
            sample["state"],
            prices=prices,
            beam_width=SELECTION["beam_width"],
            action_limit=6,
            k_paths=4,
            max_expansions=5000,
            timeout_seconds=120.0,
            price_weight=SELECTION["price_weight"],
            price_mode=SELECTION["price_mode"],
            quotient=quotient,
        )
        rows.append(
            {
                "sample_id": sample["sample_id"],
                "split": sample["split"],
                "architecture": sample["architecture"],
                "dfg_family": sample["family"],
                "method": method,
                **result.row(),
            }
        )
    return rows


def selected_samples():
    cache = ROOT / "results" / "cache" / "samples"
    samples = []
    for split in SPLITS:
        for family in FAMILIES:
            paths = sorted(cache.glob(f"{split}__*__{family}__*.pkl"))[:2]
            for path in paths:
                with path.open("rb") as handle:
                    samples.append(pickle.load(handle))
    return samples


def aggregate(frame):
    rows = []
    for split in SPLITS:
        subset = frame[frame["split"] == split]
        base = subset[subset["method"] == "length"].set_index("sample_id")
        for method in METHODS:
            current = subset[subset["method"] == method].set_index("sample_id")
            paired = base.join(current, lsuffix="_base", rsuffix="_method")
            common = paired[
                (paired["success_base"] == True) & (paired["success_method"] == True)  # noqa
            ]
            rows.append(
                {
                    "split": split,
                    "method": method,
                    "instances": len(paired),
                    "success_rate": float(paired["success_method"].mean()),
                    "success_delta_points": 100
                    * float(
                        paired["success_method"].mean()
                        - paired["success_base"].mean()
                    ),
                    "median_expansions": median(paired["expansions_method"]),
                    "expansion_reduction_pct": reduction_percent(
                        median(common["expansions_base"]),
                        median(common["expansions_method"]),
                    ),
                    "timeouts": int(paired["timeout_method"].sum()),
                    "legality_failures": int(
                        (
                            (paired["success_method"] == True)
                            & (paired["legal_method"] != True)
                        ).sum()
                    ),
                    "median_runtime_ms": median(paired["runtime_ms_method"]),
                }
            )
    return pd.DataFrame(rows)


def gates(summary):
    oracle = summary[summary["method"] == "oracle_dual"]
    overall_success = float(oracle["success_delta_points"].mean())
    overall_expansion = median(oracle["expansion_reduction_pct"])
    asym = oracle[oracle["split"] == "test_unseen_asym"].iloc[0]
    asym_ok = (
        asym["success_delta_points"] >= 10
        or asym["expansion_reduction_pct"] >= 25
    )
    g3 = (
        "GREEN"
        if overall_success >= -2
        and (overall_success >= 8 or overall_expansion >= 20)
        and asym_ok
        else "AMBER"
        if overall_success >= 3 or overall_expansion >= 8
        else "RED"
        if overall_success < -5 or overall_expansion < -10
        else "AMBER"
    )
    pred = summary[summary["method"] == "pred_dual_quotient"]
    qualifying = 0
    for split in SPLITS:
        p = pred[pred["split"] == split].iloc[0]
        o = oracle[oracle["split"] == split].iloc[0]
        retention_success = (
            p["success_delta_points"] / o["success_delta_points"]
            if o["success_delta_points"] > 0
            else float("nan")
        )
        retention_exp = (
            p["expansion_reduction_pct"] / o["expansion_reduction_pct"]
            if o["expansion_reduction_pct"] > 0
            else float("nan")
        )
        retention = np.nanmax([retention_success, retention_exp])
        if (
            p["success_delta_points"] >= -2
            and (
                p["success_delta_points"] >= 5
                or p["expansion_reduction_pct"] >= 25
            )
            and retention >= 0.60
        ):
            qualifying += 1
    g6 = "GREEN" if qualifying >= 3 else "AMBER"
    return {
        "G3_revision": g3,
        "G3_success_delta_points": overall_success,
        "G3_expansion_reduction_pct": overall_expansion,
        "G3_asym_condition": bool(asym_ok),
        "G6_revision": g6,
        "G6_qualifying_splits": qualifying,
    }


def main():
    output = ROOT / "results" / "revision_v1"
    output.mkdir(parents=True, exist_ok=True)
    (output / "selection.json").write_text(json.dumps(SELECTION, indent=2) + "\n")
    samples = selected_samples()
    checkpoint = torch.load(
        ROOT / "results" / "quick_v1" / "checkpoints" / "model_seed_11.pt",
        map_location="cuda",
        weights_only=False,
    )
    model = QuotientFlowGNN().cuda()
    model.load_state_dict(checkpoint["state_dict"])
    payloads = [
        (sample, predict_numpy(model, sample["dfg"], sample["arch"], sample["state"])[0])
        for sample in samples
    ]
    start = time.perf_counter()
    rows = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=6) as executor:
        for result in executor.map(run_sample, payloads):
            rows.extend(result)
            pd.DataFrame(rows).sort_values(
                ["split", "dfg_family", "sample_id", "method"]
            ).to_csv(output / "mappings.csv", index=False)
    frame = pd.DataFrame(rows)
    summary = aggregate(frame)
    summary.to_csv(output / "mapper_summary.csv", index=False)
    assessment = gates(summary)
    assessment["runtime_seconds"] = time.perf_counter() - start
    (output / "gate_assessment.json").write_text(
        json.dumps(assessment, indent=2) + "\n"
    )
    print(summary.to_string(index=False))
    print(json.dumps(assessment, indent=2))


if __name__ == "__main__":
    main()
