#!/usr/bin/env python3
"""Train and evaluate revision-v2 action scorers without test-set selection."""

from __future__ import annotations

import json
import math
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quotientflow.action_lookahead import (
    FEATURE_NAMES,
    scarcity_adjusted_sum,
)
from quotientflow.action_model import (
    decision_metrics,
    load_action_model,
    model_scores,
    train_action_model,
)
from quotientflow.mapper import Action
from quotientflow.model import QuotientFlowGNN, predict_numpy


OUTPUT = ROOT / "results" / "revision_v2"
RAW = OUTPUT / "raw"
CHECKPOINTS = OUTPUT / "checkpoints"
SEEDS = (11, 23, 37)
SPLITS = (
    "val_seen",
    "test_seen",
    "test_unseen_size",
    "test_unseen_topology",
    "test_unseen_asym",
)


def load_samples_and_groups():
    labels = pd.read_csv(RAW / "action_lookahead_labels.csv")
    manifest = pd.read_csv(RAW / "action_state_manifest.csv")
    paths = dict(zip(manifest["state_id"], manifest["sample_path"]))
    samples = {}
    for state_id, relative in paths.items():
        with (ROOT / relative).open("rb") as handle:
            samples[state_id] = pickle.load(handle)
    groups = []
    for state_id, frame in labels.groupby("state_id", sort=True):
        frame = frame.sort_values("action_id")
        groups.append(
            {
                "state_id": state_id,
                "split": frame["split"].iloc[0],
                "operation": int(frame["operation_id"].iloc[0]),
                "action_ids": frame["action_id"].tolist(),
                "targets": [
                    tuple(json.loads(value)) for value in frame["candidate_resource"]
                ],
                "new_links": [
                    tuple(json.loads(value)) for value in frame["route_edge_ids"]
                ],
                "features": frame[FEATURE_NAMES].to_numpy(dtype=np.float32),
                "target": frame["q_rel_normalized"].to_numpy(dtype=np.float32),
                "immediate_cost": frame["immediate_cost"].to_numpy(dtype=np.float32),
                "scarcity": frame[
                    "scarcity_adjusted_dual_sum"
                ].to_numpy(dtype=np.float32),
            }
        )
    return labels, samples, groups


def baseline_prediction(model_name: str, sample: dict, group: dict, price_model):
    if model_name == "length":
        return group["immediate_cost"].copy()
    if model_name == "heuristic_scarcity":
        return group["immediate_cost"] + 0.5 * group["scarcity"]
    if model_name == "oracle_q_rel":
        return group["target"].copy()
    predicted_prices, _ = predict_numpy(
        price_model, sample["dfg"], sample["arch"], sample["state"]
    )
    scores = []
    for target, links, immediate in zip(
        group["targets"], group["new_links"], group["immediate_cost"]
    ):
        proxy = Action(target, sample["state"], links, float(immediate))
        scores.append(
            float(immediate)
            + 0.5
            * scarcity_adjusted_sum(
                sample["arch"], sample["state"], proxy, predicted_prices
            )
        )
    return np.asarray(scores)


def prediction_rows(
    model_name,
    seed,
    groups,
    samples,
    learned=None,
    scaler=None,
    price_model=None,
):
    rows = []
    for group in groups:
        sample = samples[group["state_id"]]
        start = time.perf_counter()
        if learned is None:
            prediction = baseline_prediction(
                model_name, sample, group, price_model
            )
        else:
            learned.eval()
            with torch.no_grad():
                prediction = (
                    model_scores(learned, model_name, sample, group, scaler)
                    .detach()
                    .cpu()
                    .numpy()
                )
        inference_ms = (time.perf_counter() - start) * 1000.0
        target = group["target"]
        selected = int(np.argmin(prediction))
        oracle = int(np.argmin(target))
        top2 = set(np.argsort(prediction, kind="stable")[: min(2, len(target))])
        pair = decision_metrics(prediction, target)["pairwise_accuracy"]
        tau = kendalltau(-target, -prediction).statistic
        tau = 0.0 if not np.isfinite(tau) else float(tau)
        epsilon = float(np.max(np.abs(prediction - target)))
        ordered_target = np.sort(target)
        gap = (
            float(ordered_target[1] - ordered_target[0])
            if len(ordered_target) > 1
            else math.inf
        )
        margin = gap > 2.0 * epsilon
        for index, action_id in enumerate(group["action_ids"]):
            rows.append(
                {
                    "state_id": group["state_id"],
                    "split": group["split"],
                    "architecture": sample["architecture"],
                    "dfg_family": sample["family"],
                    "model": model_name,
                    "model_seed": seed,
                    "action_id": action_id,
                    "oracle_normalized_q": float(target[index]),
                    "predicted_q": float(prediction[index]),
                    "oracle_action": index == oracle,
                    "selected_action": index == selected,
                    "best_action_preservation": float(selected == oracle),
                    "top2_action_recall": float(oracle in top2),
                    "pairwise_ranking_accuracy": pair,
                    "kendall_tau": tau,
                    "selected_action_normalized_regret": float(target[selected]),
                    "zero_selected_regret": float(abs(float(target[selected])) <= 1e-8),
                    "epsilon_state": epsilon,
                    "gap_state": gap,
                    "empirical_margin_condition": margin,
                    "inference_ms_state": inference_ms,
                }
            )
    return rows


def aggregate_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    state = frame.drop_duplicates(["model", "model_seed", "state_id"])
    rows = []
    for keys, group in state.groupby(["model", "model_seed", "split"], dropna=False):
        observed = float(group["best_action_preservation"].mean())
        n = len(group)
        lower = observed - math.sqrt(math.log(1 / 0.05) / (2 * n))
        rows.append(
            {
                "model": keys[0],
                "model_seed": keys[1],
                "split": keys[2],
                "num_states": n,
                "best_action_preservation": observed,
                "top2_action_recall": float(group["top2_action_recall"].mean()),
                "pairwise_ranking_accuracy": float(
                    group["pairwise_ranking_accuracy"].mean()
                ),
                "kendall_tau": float(group["kendall_tau"].mean()),
                "mean_selected_normalized_regret": float(
                    group["selected_action_normalized_regret"].mean()
                ),
                "zero_regret_fraction": float(group["zero_selected_regret"].mean()),
                "margin_condition_fraction": float(
                    group["empirical_margin_condition"].mean()
                ),
                "action_preservation_lower_95": lower,
                "mean_inference_ms": float(group["inference_ms_state"].mean()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    start = time.perf_counter()
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    labels, samples, groups = load_samples_and_groups()
    train_groups = [group for group in groups if group["split"] == "train_seen"]
    validation_groups = [group for group in groups if group["split"] == "val_seen"]
    evaluation_groups = [group for group in groups if group["split"] in SPLITS]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    training_rows = []
    learned_models = {}
    for model_type in ("action_mlp", "action_gnn"):
        for seed in SEEDS:
            checkpoint = CHECKPOINTS / f"{model_type}_seed_{seed}.pt"
            model, scaler, metadata = train_action_model(
                model_type,
                train_groups,
                validation_groups,
                samples,
                seed,
                checkpoint,
                device,
            )
            learned_models[(model_type, seed)] = (model, scaler)
            training_rows.append(metadata)
            print(metadata)
    training = pd.DataFrame(training_rows)
    training.to_csv(OUTPUT / "training_summary.csv", index=False)

    # Selection is validation-only and is persisted before any test aggregation.
    ranked = training.sort_values(
        [
            "validation_best_action_preservation",
            "validation_selected_regret",
            "validation_pairwise_accuracy",
        ],
        ascending=[False, True, False],
    )
    selected = ranked.iloc[0].to_dict()
    selected["selection_split"] = "val_seen"
    selected["test_metrics_observed_at_selection"] = False
    (OUTPUT / "selected_action_model.json").write_text(
        json.dumps(selected, indent=2) + "\n"
    )

    price_checkpoint = torch.load(
        ROOT / "results" / "quick_v1" / "checkpoints" / "model_seed_11.pt",
        map_location=device,
        weights_only=False,
    )
    price_model = QuotientFlowGNN().to(device)
    price_model.load_state_dict(price_checkpoint["state_dict"])
    rows = []
    for model_name in (
        "length",
        "heuristic_scarcity",
        "predicted_link_price",
        "oracle_q_rel",
    ):
        rows.extend(
            prediction_rows(
                model_name,
                -1,
                evaluation_groups,
                samples,
                price_model=price_model,
            )
        )
    for (model_type, seed), (model, scaler) in learned_models.items():
        rows.extend(
            prediction_rows(
                model_type,
                seed,
                evaluation_groups,
                samples,
                learned=model,
                scaler=scaler,
            )
        )
    predictions = pd.DataFrame(rows)
    predictions.to_csv(RAW / "action_predictions.csv", index=False)
    summary = aggregate_predictions(predictions)
    summary.to_csv(OUTPUT / "action_prediction_summary.csv", index=False)
    runtime = {
        "training_and_prediction_seconds": time.perf_counter() - start,
        "device": device,
        "selected_model": selected["model_type"],
        "selected_seed": int(selected["seed"]),
    }
    (OUTPUT / "model_runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
    print(json.dumps(runtime, indent=2))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
