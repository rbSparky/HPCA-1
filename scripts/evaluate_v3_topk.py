#!/usr/bin/env python3
"""Add frozen revision-v2 baseline and evaluate exact top-k correction."""

from __future__ import annotations

import json
import math
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import kendalltau

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quotientflow.action_lookahead import FEATURE_NAMES, action_features  # noqa: E402
from quotientflow.action_model import load_action_model, model_scores  # noqa: E402
from quotientflow.flow_advantage import action_id  # noqa: E402
from quotientflow.flow_model import action_metrics  # noqa: E402
from quotientflow.flow_model import load_flow_model  # noqa: E402
from quotientflow.mapper import enumerate_actions  # noqa: E402
from scripts.run_v3_models import (  # noqa: E402
    append_prediction_rows, load_samples, make_groups, predict_raw,
)

OUT = ROOT / "results" / "revision_v3"
RAW = OUT / "raw"


def v2_rows(labels):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, scaler, model_type, _ = load_action_model(
        ROOT / "results/revision_v2/checkpoints/action_gnn_seed_37.pt", device
    )
    rows = []
    for state_id, group in labels.groupby("state_id", sort=True):
        with (ROOT / group.state_path.iloc[0]).open("rb") as handle:
            sample = pickle.load(handle)
        dfg, arch, state = sample["dfg"], sample["arch"], sample["state"]
        operation = int(group.operation_id.iloc[0])
        actions, _, _ = enumerate_actions(dfg, arch, state, operation, 4)
        lookup = {action_id(operation, action): action for action in actions}
        selected_actions, selected_rows = [], []
        for source in group.itertuples():
            action = lookup.get(source.action_id)
            if action is not None:
                selected_actions.append(action)
                selected_rows.append(source)
        if len(selected_actions) < 2:
            continue
        prices = np.asarray(sample["normalized_prices"])
        feature_values = [
            action_features(dfg, arch, state, operation, action, prices)
            for action in selected_actions
        ]
        model_group = {
            "features": np.asarray([
                [value[name] for name in FEATURE_NAMES]
                for value in feature_values
            ], np.float32),
            "operation": operation,
            "targets": [action.target for action in selected_actions],
            "new_links": [action.new_links for action in selected_actions],
        }
        with torch.no_grad():
            prediction = model_scores(
                model, model_type,
                {"dfg": dfg, "arch": arch, "state": state},
                model_group, scaler,
            ).cpu().numpy()
        delta = np.asarray([source.delta_star for source in selected_rows])
        epsilon = float(selected_rows[0].epsilon_state)
        metrics = action_metrics(prediction, delta, epsilon)
        tau = kendalltau(prediction, delta).statistic
        metrics["kendall_tau"] = float(tau) if math.isfinite(tau) else 0.0
        order = np.argsort(prediction, kind="stable")
        true_order = np.argsort(delta, kind="stable")
        selected = int(order[0])
        for index, source in enumerate(selected_rows):
            rows.append({
                "state_id": state_id,
                "split": source.split,
                "architecture": source.architecture,
                "dfg_family": source.dfg_family,
                "depth_bucket": source.depth_bucket,
                "trajectory_depth": source.trajectory_depth,
                "model": "revision_v2_action_gnn",
                "model_seed": "37",
                "action_id": source.action_id,
                "delta_star": delta[index],
                "dual_baseline": source.dual_baseline,
                "predicted_delta": prediction[index],
                "predicted_rank": int(np.where(order == index)[0][0]) + 1,
                "true_rank": int(np.where(true_order == index)[0][0]) + 1,
                "selected_action": index == selected,
                "true_best_action": index == true_order[0],
                "action_feasible": bool(source.child_feasible),
                **metrics,
            })
    return rows


def topk_rows(labels, predictions, selected_seed):
    chosen = predictions[
        (predictions.model == "residual_gnn")
        & (predictions.model_seed.astype(str) == str(selected_seed))
    ]
    rows = []
    audit = labels[labels.full_action_audit_state]
    for state_id, truth in audit.groupby("state_id", sort=True):
        pred = chosen[chosen.state_id == state_id].set_index("action_id")
        truth = truth[truth.action_id.isin(pred.index)].copy()
        if len(truth) < 2:
            continue
        truth["predicted_delta"] = [
            pred.loc[action, "predicted_delta"] for action in truth.action_id
        ]
        truth = truth.sort_values("action_id").reset_index(drop=True)
        true_order = np.argsort(truth.delta_star.to_numpy(), kind="stable")
        predicted_order = np.argsort(
            truth.predicted_delta.to_numpy(), kind="stable"
        )
        true_best = int(true_order[0])
        epsilon = float(truth.epsilon_state.iloc[0])
        for k in (2, 4):
            retained = predicted_order[:min(k, len(predicted_order))]
            corrected = min(
                retained,
                key=lambda index: (
                    truth.delta_star.iloc[index], truth.action_id.iloc[index]
                ),
            )
            rows.append({
                "state_id": state_id,
                "split": truth.split.iloc[0],
                "architecture": truth.architecture.iloc[0],
                "dfg_family": truth.dfg_family.iloc[0],
                "depth_bucket": truth.depth_bucket.iloc[0],
                "k": k,
                "legal_actions": len(truth),
                "full_oracle_action_id": truth.action_id.iloc[true_best],
                "corrected_action_id": truth.action_id.iloc[corrected],
                "exact_full_oracle_match": corrected == true_best,
                "epsilon_optimal_match": (
                    truth.delta_star.iloc[corrected]
                    <= truth.delta_star.iloc[true_best] + epsilon + 1e-12
                ),
                "corrected_raw_regret": (
                    truth.delta_star.iloc[corrected]
                    - truth.delta_star.iloc[true_best]
                ),
                "corrected_relative_regret": (
                    truth.delta_star.iloc[corrected]
                    - truth.delta_star.iloc[true_best]
                ) / max(float(truth.delta_star.max() - truth.delta_star.min()), 1.0),
                "child_solves": len(retained),
                "full_oracle_child_solves": len(truth),
                "child_solves_saved": len(truth) - len(retained),
                "child_solve_reduction": 1.0 - len(retained) / len(truth),
                "true_best_in_predicted_topk": true_best in retained,
            })
    return pd.DataFrame(rows)


def main():
    started = time.perf_counter()
    labels = pd.read_csv(RAW / "action_advantage_labels.csv")
    predictions = pd.read_csv(RAW / "action_predictions.csv")
    existing_v2 = predictions[
        predictions.model == "revision_v2_action_gnn"
    ].copy()
    predictions = predictions[
        predictions.model != "revision_v2_action_gnn"
    ]
    selection = json.loads((OUT / "selected_flow_model.json").read_text())
    selected = str(selection["principal_seed_or_ensemble"])
    # Emit predefined training-audit predictions for complete 44-state
    # top-k coverage. Held-out rows retain their original frozen predictions.
    if not (
        (predictions.model == "residual_gnn")
        & (predictions.model_seed.astype(str) == selected)
        & (predictions.split == "train_on_policy")
    ).any():
        states = pd.read_csv(RAW / "on_policy_states.csv")
        samples = load_samples(states)
        groups = make_groups(labels, samples, "train_on_policy", False)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        seeds = (11, 23, 37) if selected == "ensemble" else (int(selected),)
        seed_predictions = []
        for seed in seeds:
            model, scaler, data = load_flow_model(
                OUT / "checkpoints" / f"residual_gnn_seed_{seed}.pt", device
            )
            seed_predictions.append([
                predict_raw(
                    model, data["model_type"], group,
                    samples[group["state_id"]], scaler, data,
                )
                for group in groups
            ])
        combined = [
            np.mean([values[i] for values in seed_predictions], axis=0)
            for i in range(len(groups))
        ]
        extra = []
        append_prediction_rows(
            extra, "residual_gnn", selected, groups, combined
        )
        predictions = pd.concat(
            [predictions, pd.DataFrame(extra)], ignore_index=True
        )
    appended = (
        existing_v2
        if len(existing_v2) == len(labels)
        else pd.DataFrame(v2_rows(labels))
    )
    predictions = pd.concat([predictions, appended], ignore_index=True)
    predictions.to_csv(RAW / "action_predictions.csv", index=False)
    topk = topk_rows(
        labels, predictions, selection["principal_seed_or_ensemble"]
    )
    topk.to_csv(RAW / "topk_correction.csv", index=False)
    runtime = {
        "seconds": time.perf_counter() - started,
        "v2_prediction_rows": len(appended),
        "topk_states": int(topk.state_id.nunique()),
    }
    (OUT / "topk_runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
    print(json.dumps(runtime, indent=2))
    print(topk.groupby("k").agg(
        states=("state_id", "size"),
        exact=("exact_full_oracle_match", "mean"),
        epsilon=("epsilon_optimal_match", "mean"),
        solve_reduction=("child_solve_reduction", "mean"),
    ).to_string())


if __name__ == "__main__":
    main()
