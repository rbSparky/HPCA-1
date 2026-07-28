#!/usr/bin/env python3
"""Train FlowAdvantage residual models and evaluate frozen action ranking."""

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
from quotientflow.flow_advantage import (  # noqa: E402
    FLOW_FEATURE_NAMES,
    action_feature_rows,
    corrected_targets,
    dual_baseline,
    flow_state_hash,
    robust_residual_scaler,
)
from quotientflow.flow_model import (  # noqa: E402
    action_metrics,
    flow_scores,
    load_flow_model,
    train_flow_model,
)
from quotientflow.mapper import enumerate_actions  # noqa: E402
from quotientflow.relaxation import solve_relaxation  # noqa: E402

OUT = ROOT / "results" / "revision_v3"
RAW = OUT / "raw"
CHECKPOINTS = OUT / "checkpoints"
PARENT_CACHE = OUT / "cache" / "parents"
RELAXATION_CONFIG = {
    "tau": 0.001,
    "solver_primary": "CLARABEL",
    "solver_fallback": "OSQP",
    "max_seconds": 12.0,
}
SEEDS = (11, 23, 37)
TEST_SPLITS = (
    "test_seen", "test_unseen_size",
    "test_unseen_topology", "test_unseen_asym",
)


def cached_parent(dfg, arch, state):
    key = flow_state_hash(dfg, arch, state, RELAXATION_CONFIG)
    path = PARENT_CACHE / f"{key}.pkl"
    if path.exists():
        with path.open("rb") as handle:
            return pickle.load(handle)
    result = solve_relaxation(dfg, arch, state, **RELAXATION_CONFIG)
    with path.open("wb") as handle:
        pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return result


def load_samples(states):
    samples = {}
    for row in states.itertuples():
        with (ROOT / row.state_path).open("rb") as handle:
            sample = pickle.load(handle)
        samples[row.state_id] = sample
    return samples


def make_groups(frame, samples, split, selector_only=True):
    subset = frame[frame.split == split]
    groups = []
    for state_id, group in subset.groupby("state_id", sort=True):
        if selector_only and group.full_action_audit_state.iloc[0]:
            group = group[group.training_selector_action]
        group = group.sort_values("action_id").reset_index(drop=True)
        if len(group) < 2:
            continue
        parent_key = group.parent_cache_key.iloc[0]
        with (PARENT_CACHE / f"{parent_key}.pkl").open("rb") as handle:
            parent = pickle.load(handle)
        groups.append({
            "state_id": state_id,
            "frame": group,
            "parent": parent,
            "operation": int(group.operation_id.iloc[0]),
            "targets": [tuple(json.loads(value)) for value in group.candidate_resource],
            "new_links": [tuple(json.loads(value)) for value in group.route_edge_ids],
            "baseline": group.dual_baseline.to_numpy(float),
            "delta": group.delta_star.to_numpy(float),
            "residual_target": group.residual_target.to_numpy(float),
            "raw_regret": group.raw_regret.to_numpy(float),
            "soft_target": group.soft_target.to_numpy(float),
            "epsilon": float(group.epsilon_state.iloc[0]),
            "delta_range": float(group.delta_star.max() - group.delta_star.min()),
        })
    return groups


def prepare_initial_only():
    output = RAW / "initial_only_advantage_labels.csv"
    v2 = pd.read_csv(
        ROOT / "results/revision_v2/raw/action_lookahead_labels.csv"
    )
    manifest = pd.read_csv(
        ROOT / "results/revision_v2/raw/action_state_manifest.csv"
    ).set_index("state_id")
    rows, samples = [], {}
    for state_id, group in v2[v2.split.isin(["train_seen", "val_seen"])].groupby(
        "state_id", sort=True
    ):
        sample_path = ROOT / manifest.loc[state_id, "sample_path"]
        with sample_path.open("rb") as handle:
            sample = pickle.load(handle)
        samples[state_id] = sample
        dfg, arch, state = sample["dfg"], sample["arch"], sample["state"]
        parent = cached_parent(dfg, arch, state)
        if not parent.solved:
            continue
        operation = int(group.operation_id.iloc[0])
        actions, _, _ = enumerate_actions(dfg, arch, state, operation, 4)
        lookup = {}
        for action in actions:
            lookup.setdefault((action.target, action.new_links), action)
        selected_actions, source_rows = [], []
        for source in group.itertuples():
            signature = (
                tuple(json.loads(source.candidate_resource)),
                tuple(json.loads(source.route_edge_ids)),
            )
            if signature in lookup:
                selected_actions.append(lookup[signature])
                source_rows.append(source)
        features = action_feature_rows(
            dfg, arch, state, operation, selected_actions, parent
        )
        baselines = np.asarray([
            dual_baseline(action, parent)[0] for action in selected_actions
        ])
        delta = np.asarray([
            float(source.q_rel_raw) - parent.objective for source in source_rows
        ])
        regret = delta - delta.min()
        delta_range = float(np.ptp(delta))
        epsilon = max(1e-4, .01 * max(delta_range, 1.0))
        q25, q75 = np.quantile(regret, [.25, .75])
        state_scale = max(float(q75 - q25), .1, .05 * max(delta_range, 1.0))
        soft = np.exp(-(regret - regret.min()) / state_scale)
        soft /= soft.sum()
        for i, (source, action) in enumerate(zip(source_rows, selected_actions)):
            rows.append({
                "state_id": state_id,
                "split": "initial_train" if source.split == "train_seen"
                else "initial_validation",
                "architecture": source.architecture,
                "dfg_family": source.dfg_family,
                "depth_bucket": source.partial_depth,
                "trajectory_depth": source.partial_depth,
                "state_path": str(sample_path.relative_to(ROOT)),
                "operation_id": operation,
                "action_id": source.action_id,
                "candidate_resource": source.candidate_resource,
                "route_edge_ids": source.route_edge_ids,
                "dual_baseline": baselines[i],
                "delta_star": delta[i],
                "residual_advantage": delta[i] - baselines[i],
                "raw_regret": regret[i],
                "relative_regret": regret[i] / max(delta_range, 1.0),
                "epsilon_state": epsilon,
                "epsilon_optimal": regret[i] <= epsilon + 1e-12,
                "soft_target": soft[i],
                **features[i],
            })
    frame = pd.DataFrame(rows)
    center, scale = robust_residual_scaler(
        frame.loc[
            frame.split == "initial_train", "residual_advantage"
        ].to_numpy(float)
    )
    frame["residual_target"] = np.clip(
        (frame.residual_advantage - center) / scale, -8.0, 8.0
    )
    frame.to_csv(output, index=False)
    return frame, samples, center, scale


def initial_groups(frame, samples, split):
    groups = []
    for state_id, group in frame[frame.split == split].groupby("state_id"):
        group = group.sort_values("action_id").reset_index(drop=True)
        sample = samples[state_id]
        parent = cached_parent(
            sample["dfg"], sample["arch"], sample["state"]
        )
        groups.append({
            "state_id": state_id,
            "frame": group,
            "parent": parent,
            "operation": int(group.operation_id.iloc[0]),
            "targets": [tuple(json.loads(v)) for v in group.candidate_resource],
            "new_links": [tuple(json.loads(v)) for v in group.route_edge_ids],
            "baseline": group.dual_baseline.to_numpy(float),
            "delta": group.delta_star.to_numpy(float),
            "residual_target": group.residual_target.to_numpy(float),
            "raw_regret": group.raw_regret.to_numpy(float),
            "soft_target": group.soft_target.to_numpy(float),
            "epsilon": float(group.epsilon_state.iloc[0]),
            "delta_range": float(group.delta_star.max() - group.delta_star.min()),
        })
    return groups


def validation_key(metrics):
    return (
        metrics["top4_recall"], metrics["epsilon_top1"],
        -metrics["relative_regret"], metrics["exact_top1"],
    )


def predict_raw(model, model_type, group, sample, scaler, data):
    with torch.no_grad():
        scaled = flow_scores(
            model, model_type, group, sample, scaler,
            data["metadata"]["feature_names"],
        ).cpu().numpy()
    return (
        group["baseline"]
        + scaled * data["metadata"]["residual_scale"]
        + data["metadata"]["residual_center"]
    )


def evaluate_prediction(predicted, group):
    delta = group["delta"]
    metrics = action_metrics(predicted, delta, group["epsilon"])
    tau = kendalltau(predicted, delta).statistic
    metrics["kendall_tau"] = float(tau) if math.isfinite(tau) else 0.0
    return metrics


def select_seed_or_ensemble(family, checkpoints, validation_groups, samples, device):
    loaded = [
        load_flow_model(path, device) for path in checkpoints
    ]
    candidate_predictions = {}
    for seed, (model, scaler, data) in zip(SEEDS, loaded):
        candidate_predictions[str(seed)] = [
            predict_raw(
                model, data["model_type"], group,
                samples[group["state_id"]], scaler, data,
            )
            for group in validation_groups
        ]
    candidate_predictions["ensemble"] = [
        np.mean([
            candidate_predictions[str(seed)][i] for seed in SEEDS
        ], axis=0)
        for i in range(len(validation_groups))
    ]
    results = {}
    for candidate, predictions in candidate_predictions.items():
        values = [
            evaluate_prediction(prediction, group)
            for prediction, group in zip(predictions, validation_groups)
        ]
        results[candidate] = {
            key: float(np.mean([value[key] for value in values]))
            for key in values[0]
        }
    selected = max(results, key=lambda key: validation_key(results[key]))
    return selected, results


def append_prediction_rows(rows, model_name, seed_name, groups, predicted_list):
    for group, predicted in zip(groups, predicted_list):
        metrics = evaluate_prediction(predicted, group)
        order = np.argsort(predicted, kind="stable")
        true_order = np.argsort(group["delta"], kind="stable")
        selected = int(order[0])
        for i, source in group["frame"].iterrows():
            rows.append({
                "state_id": group["state_id"],
                "split": source["split"],
                "architecture": source["architecture"],
                "dfg_family": source["dfg_family"],
                "depth_bucket": source["depth_bucket"],
                "trajectory_depth": source["trajectory_depth"],
                "model": model_name,
                "model_seed": seed_name,
                "action_id": source["action_id"],
                "delta_star": group["delta"][i],
                "dual_baseline": group["baseline"][i],
                "predicted_delta": predicted[i],
                "predicted_rank": int(np.where(order == i)[0][0]) + 1,
                "true_rank": int(np.where(true_order == i)[0][0]) + 1,
                "selected_action": i == selected,
                "true_best_action": i == true_order[0],
                "action_feasible": bool(source["child_feasible"])
                if "child_feasible" in source else True,
                **metrics,
            })


def main():
    started = time.perf_counter()
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    labels = pd.read_csv(RAW / "action_advantage_labels.csv")
    states = pd.read_csv(RAW / "on_policy_states.csv")
    samples = load_samples(states)
    scaler_config = json.loads((OUT / "target_scaler.json").read_text())
    center, scale = (
        scaler_config["residual_center"], scaler_config["residual_scale"]
    )
    train = make_groups(labels, samples, "train_on_policy", True)
    validation = make_groups(labels, samples, "validation_on_policy", True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    summaries = []
    families = ("residual_mlp", "residual_gnn", "residual_gnn_no_dual")
    for family in families:
        for seed in SEEDS:
            _, _, metadata = train_flow_model(
                family, train, validation, samples, seed,
                CHECKPOINTS / f"{family}_seed_{seed}.pt",
                device, center, scale, epochs=80, patience=12,
            )
            metadata["training_regime"] = "on_policy"
            summaries.append(metadata)

    initial_frame, initial_samples, initial_center, initial_scale = (
        prepare_initial_only()
    )
    initial_train = initial_groups(
        initial_frame, initial_samples, "initial_train"
    )
    initial_validation = initial_groups(
        initial_frame, initial_samples, "initial_validation"
    )
    for seed in SEEDS:
        _, _, metadata = train_flow_model(
            "residual_gnn", initial_train, initial_validation,
            initial_samples, seed,
            CHECKPOINTS / f"initial_only_residual_gnn_seed_{seed}.pt",
            device, initial_center, initial_scale, epochs=80, patience=12,
        )
        metadata["training_regime"] = "initial_only"
        summaries.append(metadata)
    pd.DataFrame(summaries).to_csv(OUT / "training_summary.csv", index=False)

    selections = {}
    validation_results = {}
    for family in families:
        paths = [CHECKPOINTS / f"{family}_seed_{seed}.pt" for seed in SEEDS]
        choice, results = select_seed_or_ensemble(
            family, paths, validation, samples, device
        )
        selections[family] = choice
        validation_results[family] = results
    initial_paths = [
        CHECKPOINTS / f"initial_only_residual_gnn_seed_{seed}.pt"
        for seed in SEEDS
    ]
    initial_choice, initial_results = select_seed_or_ensemble(
        "initial_only_residual_gnn", initial_paths,
        initial_validation, initial_samples, device,
    )
    selections["initial_only_residual_gnn"] = initial_choice
    validation_results["initial_only_residual_gnn"] = initial_results
    selection = {
        "selection_split": "validation_on_policy",
        "test_metrics_observed_at_selection": False,
        "selected": selections,
        "validation_metrics": validation_results,
        "principal_model": "residual_gnn",
        "principal_seed_or_ensemble": selections["residual_gnn"],
    }
    (OUT / "selected_flow_model.json").write_text(
        json.dumps(selection, indent=2) + "\n"
    )

    # Frozen evaluation begins only after the selection record is durable.
    evaluation_groups = {
        split: make_groups(labels, samples, split, False)
        for split in ("validation_on_policy", *TEST_SPLITS)
    }
    rows = []
    for split, groups in evaluation_groups.items():
        append_prediction_rows(
            rows, "dual_linear", "-1", groups,
            [group["baseline"] for group in groups],
        )
        for family in families:
            loaded = [
                load_flow_model(
                    CHECKPOINTS / f"{family}_seed_{seed}.pt", device
                )
                for seed in SEEDS
            ]
            seed_predictions = {}
            for seed, (model, scaler, data) in zip(SEEDS, loaded):
                predictions = [
                    predict_raw(
                        model, data["model_type"], group,
                        samples[group["state_id"]], scaler, data,
                    )
                    for group in groups
                ]
                seed_predictions[seed] = predictions
                append_prediction_rows(rows, family, str(seed), groups, predictions)
            ensemble = [
                np.mean([seed_predictions[seed][i] for seed in SEEDS], axis=0)
                for i in range(len(groups))
            ]
            append_prediction_rows(
                rows, family, "ensemble", groups, ensemble
            )
        # Initial-only baseline is evaluated on on-policy states.
        loaded = [
            load_flow_model(
                CHECKPOINTS / f"initial_only_residual_gnn_seed_{seed}.pt", device
            )
            for seed in SEEDS
        ]
        seed_predictions = {}
        for seed, (model, scaler, data) in zip(SEEDS, loaded):
            predictions = [
                predict_raw(
                    model, data["model_type"], group,
                    samples[group["state_id"]], scaler, data,
                )
                for group in groups
            ]
            seed_predictions[seed] = predictions
            append_prediction_rows(
                rows, "initial_only_residual_gnn", str(seed),
                groups, predictions,
            )
        ensemble = [
            np.mean([seed_predictions[seed][i] for seed in SEEDS], axis=0)
            for i in range(len(groups))
        ]
        append_prediction_rows(
            rows, "initial_only_residual_gnn", "ensemble", groups, ensemble
        )
    predictions = pd.DataFrame(rows)
    predictions.to_csv(RAW / "action_predictions.csv", index=False)
    runtime = {
        "seconds": time.perf_counter() - started,
        "cuda": torch.cuda.is_available(),
        "models_trained": len(summaries),
        "selection": selections,
        "prediction_rows": len(predictions),
    }
    (OUT / "model_runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
    print(json.dumps(runtime, indent=2))


if __name__ == "__main__":
    main()
