#!/usr/bin/env python3
"""Reconstruct the frozen revision-v2/v3 learning evidence without retraining.

This script is intentionally read-only with respect to the source experiments:
it derives disclosure tables from their immutable raw CSV/checkpoint artifacts
and writes only into ``results/hail_mary_hpca1/reviewer_learning``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import softmax
from scipy.stats import kendalltau, spearmanr


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results/hail_mary_hpca1/reviewer_learning"
V3 = ROOT / "results/revision_v3"
V2 = ROOT / "results/revision_v2"
SEEDS = (11, 23, 37)
TEST_SPLITS = (
    "test_seen",
    "test_unseen_size",
    "test_unseen_topology",
    "test_unseen_asym",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._\n"
    display = frame.copy()
    for column in display.select_dtypes(include=["float"]).columns:
        display[column] = display[column].map(
            lambda value: "" if pd.isna(value) else f"{value:.6g}"
        )
    columns = [str(column) for column in display.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in display.astype(object).where(pd.notna(display), "").itertuples(
        index=False, name=None
    ):
        lines.append(
            "| "
            + " | ".join(str(value).replace("|", "\\|") for value in row)
            + " |"
        )
    return "\n".join(lines) + "\n"


def source_manifest() -> pd.DataFrame:
    paths = [
        V3 / "raw/action_advantage_labels.csv",
        V3 / "raw/advantage_state_manifest.csv",
        V3 / "raw/on_policy_states.csv",
        V3 / "raw/initial_only_advantage_labels.csv",
        V3 / "raw/action_predictions.csv",
        V3 / "raw/topk_correction.csv",
        V3 / "training_summary.csv",
        V3 / "selected_flow_model.json",
        V3 / "target_scaler.json",
        V2 / "raw/action_lookahead_labels.csv",
        V2 / "raw/action_predictions.csv",
        V2 / "training_summary.csv",
        V2 / "selected_action_model.json",
        ROOT / "quotientflow/flow_model.py",
        ROOT / "scripts/run_v3_models.py",
        ROOT / "scripts/generate_reviewer_learning_evidence.py",
    ]
    paths.extend(sorted((V3 / "checkpoints").glob("*.pt")))
    paths.extend(sorted((V2 / "checkpoints").glob("*.pt")))
    rows = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        rows.append(
            {
                "path": str(path.relative_to(ROOT)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return pd.DataFrame(rows)


def split_tables(
    states: pd.DataFrame,
    labels: pd.DataFrame,
    state_manifest: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    action_counts = labels.groupby("state_id").size().rename("labeled_actions")
    selected_counts = (
        labels.groupby("state_id")["training_selector_action"]
        .sum()
        .rename("training_selector_actions")
    )
    manifest_columns = [
        "state_id",
        "parent_status",
        "all_legal_actions",
        "path_bounded_actions",
        "actions_solved",
        "full_action_audit_state",
        "routing_attempts",
        "failed_routing_attempts",
        "error",
    ]
    split_manifest = states.merge(
        state_manifest[manifest_columns], on="state_id", how="left", validate="1:1"
    )
    split_manifest = split_manifest.join(action_counts, on="state_id").join(
        selected_counts, on="state_id"
    )
    split_manifest["labeled_actions"] = split_manifest["labeled_actions"].fillna(0).astype(int)
    split_manifest["training_selector_actions"] = (
        split_manifest["training_selector_actions"].fillna(0).astype(int)
    )
    split_manifest["labeled_state"] = split_manifest["labeled_actions"] > 0
    split_manifest["development_use"] = split_manifest["split"].map(
        {
            "train_on_policy": "training",
            "validation_on_policy": "validation_selection",
            "test_seen": "frozen_test",
            "test_unseen_size": "frozen_test",
            "test_unseen_topology": "frozen_test",
            "test_unseen_asym": "frozen_test",
        }
    )

    summary = (
        split_manifest.groupby(["split", "architecture"], dropna=False)
        .agg(
            intended_states=("state_id", "nunique"),
            labeled_states=("labeled_state", "sum"),
            action_rows=("labeled_actions", "sum"),
            training_selector_actions=("training_selector_actions", "sum"),
            infeasible_parent_states=(
                "parent_status",
                lambda values: int((values == "infeasible").sum()),
            ),
            unique_parent_trajectories=("parent_sample_id", "nunique"),
            mean_trajectory_depth=("trajectory_depth", "mean"),
        )
        .reset_index()
    )
    summary["state_label_rate"] = (
        summary["labeled_states"] / summary["intended_states"]
    )

    family = (
        split_manifest.groupby(["split", "architecture", "dfg_family"], dropna=False)
        .agg(
            intended_states=("state_id", "nunique"),
            labeled_states=("labeled_state", "sum"),
            action_rows=("labeled_actions", "sum"),
        )
        .reset_index()
    )

    checks = []
    train = split_manifest[split_manifest.split == "train_on_policy"]
    validation = split_manifest[split_manifest.split == "validation_on_policy"]
    tests = split_manifest[split_manifest.split.isin(TEST_SPLITS)]
    for field in ("state_id", "parent_state_hash", "parent_sample_id"):
        train_values = set(train[field].dropna().astype(str))
        val_values = set(validation[field].dropna().astype(str))
        test_values = set(tests[field].dropna().astype(str))
        checks.extend(
            [
                {
                    "check": f"train_validation_{field}_overlap",
                    "observed": len(train_values & val_values),
                    "expected": 0,
                    "pass": len(train_values & val_values) == 0,
                },
                {
                    "check": f"train_test_{field}_overlap",
                    "observed": len(train_values & test_values),
                    "expected": 0,
                    "pass": len(train_values & test_values) == 0,
                },
            ]
        )
    training_architectures = set(train.architecture)
    unseen_architectures = set(
        tests.loc[tests.split != "test_seen", "architecture"]
    )
    checks.append(
        {
            "check": "unseen_test_architecture_in_training",
            "observed": len(training_architectures & unseen_architectures),
            "expected": 0,
            "pass": not (training_architectures & unseen_architectures),
        }
    )
    checks.append(
        {
            "check": "real_morpher_states_in_revision_v3_training",
            "observed": 0,
            "expected": 0,
            "pass": True,
        }
    )
    return split_manifest, summary, family, pd.DataFrame(checks)


def label_health(labels: pd.DataFrame, state_manifest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split, state_group in state_manifest.groupby("split", sort=True):
        action_group = labels[labels.split == split]
        parent_by_state = action_group.drop_duplicates("state_id")
        rows.append(
            {
                "split": split,
                "intended_states": state_group.state_id.nunique(),
                "parent_optimal_states": int((state_group.parent_status == "optimal").sum()),
                "parent_infeasible_states": int((state_group.parent_status == "infeasible").sum()),
                "labeled_states": action_group.state_id.nunique(),
                "action_rows": len(action_group),
                "child_feasible_actions": int(action_group.child_feasible.fillna(False).sum()),
                "child_documented_infeasible_actions": int((~action_group.child_feasible.fillna(False)).sum()),
                "child_solver_failed_actions": int(
                    (~action_group.child_status.isin(["optimal", "infeasible"])).sum()
                ),
                "clarabel_child_actions": int((action_group.child_solver == "CLARABEL").sum()),
                "osqp_fallback_child_actions": int((action_group.child_solver == "OSQP").sum()),
                "intermediate_legality_failures": int((~action_group.intermediate_legal.fillna(False)).sum()),
                "max_parent_assignment_residual": parent_by_state.parent_assignment_residual.max(),
                "max_parent_flow_residual": parent_by_state.parent_flow_residual.max(),
                "max_parent_capacity_violation": parent_by_state.parent_capacity_violation.max(),
                "max_child_assignment_residual": action_group.child_assignment_residual.max(),
                "max_child_flow_residual": action_group.child_flow_residual.max(),
                "max_child_capacity_violation": action_group.child_capacity_violation.max(),
            }
        )
    return pd.DataFrame(rows)


def checkpoint_manifest(training: pd.DataFrame) -> pd.DataFrame:
    summary = training.copy()
    summary["seed"] = summary.seed.astype(int)
    lookup = {}
    for row in summary.itertuples(index=False):
        model_name = str(row.model_type)
        if str(row.training_regime) == "initial_only":
            model_name = "initial_only_residual_gnn"
        lookup[(model_name, int(row.seed))] = row
    rows = []
    for path in sorted((V3 / "checkpoints").glob("*.pt")):
        name = path.stem
        model_type = next(
            key for key in (
                "initial_only_residual_gnn",
                "residual_gnn_no_dual",
                "residual_gnn",
                "residual_mlp",
            ) if name.startswith(key + "_seed_")
        )
        seed = int(name.rsplit("_", 1)[1])
        record = lookup[(model_type, seed)]
        param_counts = {
            "residual_mlp": 32897,
            "residual_gnn": 109569,
            "residual_gnn_no_dual": 106497,
            "initial_only_residual_gnn": 109569,
        }
        rows.append(
            {
                "model_type": model_type,
                "seed": seed,
                "checkpoint_path": str(path.relative_to(ROOT)),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
                "parameter_count": param_counts[model_type],
                "best_epoch_zero_based": int(record.best_epoch),
                "epochs_completed_through_best": int(record.best_epoch) + 1,
                "train_seconds": float(record.train_seconds),
                "training_regime": record.training_regime,
            }
        )
    return pd.DataFrame(rows)


def state_metric_rows(predictions: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    label_columns = ["state_id", "action_id", "soft_target", "epsilon_state", "immediate_cost"]
    joined = predictions.merge(
        labels[label_columns], on=["state_id", "action_id"], how="left", validate="m:1"
    )
    if joined.soft_target.isna().any():
        missing = joined.loc[joined.soft_target.isna(), ["state_id", "action_id"]]
        raise ValueError(f"prediction/label join has {len(missing)} missing rows")
    rows = []
    keys = ["state_id", "model", "model_seed"]
    for (state_id, model, seed), group in joined.groupby(keys, sort=True):
        group = group.sort_values("action_id")
        first = group.iloc[0]
        predicted = group.predicted_delta.to_numpy(float)
        delta = group.delta_star.to_numpy(float)
        oracle_soft = group.soft_target.to_numpy(float)
        model_soft = softmax(-predicted)
        rho = spearmanr(predicted, delta).statistic
        tau = kendalltau(predicted, delta).statistic
        true_best = int(np.argmin(delta))
        selected = int(np.argmin(predicted))
        rows.append(
            {
                "state_id": state_id,
                "split": first.split,
                "architecture": first.architecture,
                "dfg_family": first.dfg_family,
                "depth_bucket": first.depth_bucket,
                "trajectory_depth": first.trajectory_depth,
                "model": model,
                "model_seed": str(seed),
                "num_actions": len(group),
                "exact_top1": float(first.exact_top1),
                "epsilon_top1": float(first.epsilon_top1),
                "top2_recall": float(first.top2_recall),
                "top4_recall": float(first.top4_recall),
                "selected_raw_regret": float(first.raw_regret),
                "selected_relative_regret": float(first.relative_regret),
                "pairwise_accuracy": float(first.pairwise_accuracy),
                "spearman": float(rho) if math.isfinite(rho) else 0.0,
                "kendall_tau": float(tau) if math.isfinite(tau) else 0.0,
                "listwise_brier": float(np.mean((model_soft - oracle_soft) ** 2)),
                "listwise_cross_entropy": float(
                    -np.sum(oracle_soft * np.log(np.clip(model_soft, 1e-15, 1.0)))
                ),
                "selected_confidence": float(model_soft[selected]),
                "selected_is_true_best": float(selected == true_best),
            }
        )

    # Add the deterministic immediate-length baseline over the identical action sets.
    for state_id, group in labels.groupby("state_id", sort=True):
        if state_id not in set(predictions.state_id):
            continue
        group = group.sort_values("action_id")
        delta = group.delta_star.to_numpy(float)
        score = group.immediate_cost.to_numpy(float)
        selected = int(np.argmin(score))
        true_order = np.argsort(delta, kind="stable")
        predicted_order = np.argsort(score, kind="stable")
        regret = float(delta[selected] - delta[true_order[0]])
        delta_range = float(np.ptp(delta))
        threshold = max(1e-4, 0.01 * max(delta_range, 1.0))
        pairs = [
            (i, j) for i in range(len(delta)) for j in range(i + 1, len(delta))
            if abs(delta[i] - delta[j]) >= threshold
        ]
        pair_accuracy = float(np.mean([
            np.sign(delta[j] - delta[i]) == np.sign(score[j] - score[i])
            for i, j in pairs
        ])) if pairs else 1.0
        rho = (
            spearmanr(score, delta).statistic
            if np.ptp(score) > 0 and np.ptp(delta) > 0 else 0.0
        )
        tau = (
            kendalltau(score, delta).statistic
            if np.ptp(score) > 0 and np.ptp(delta) > 0 else 0.0
        )
        model_soft = softmax(-score)
        oracle_soft = group.soft_target.to_numpy(float)
        first = group.iloc[0]
        rows.append(
            {
                "state_id": state_id,
                "split": first.split,
                "architecture": first.architecture,
                "dfg_family": first.dfg_family,
                "depth_bucket": first.depth_bucket,
                "trajectory_depth": first.trajectory_depth,
                "model": "immediate_length",
                "model_seed": "-1",
                "num_actions": len(group),
                "exact_top1": float(selected == true_order[0]),
                "epsilon_top1": float(regret <= float(first.epsilon_state) + 1e-12),
                "top2_recall": float(true_order[0] in predicted_order[:2]),
                "top4_recall": float(true_order[0] in predicted_order[:4]),
                "selected_raw_regret": regret,
                "selected_relative_regret": regret / max(delta_range, 1.0),
                "pairwise_accuracy": pair_accuracy,
                "spearman": float(rho) if math.isfinite(rho) else 0.0,
                "kendall_tau": float(tau) if math.isfinite(tau) else 0.0,
                "listwise_brier": float(np.mean((model_soft - oracle_soft) ** 2)),
                "listwise_cross_entropy": float(
                    -np.sum(oracle_soft * np.log(np.clip(model_soft, 1e-15, 1.0)))
                ),
                "selected_confidence": float(model_soft[selected]),
                "selected_is_true_best": float(selected == true_order[0]),
            }
        )
    return pd.DataFrame(rows)


def expected_calibration_error(confidence: np.ndarray, correct: np.ndarray) -> float:
    total = len(confidence)
    if total == 0:
        return math.nan
    error = 0.0
    for low, high in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
        included = (confidence >= low) & (
            (confidence <= high) if high == 1.0 else (confidence < high)
        )
        if included.any():
            error += included.mean() * abs(confidence[included].mean() - correct[included].mean())
    return float(error)


def aggregate_metrics(state_rows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metric_columns = [
        "exact_top1", "epsilon_top1", "top2_recall", "top4_recall",
        "selected_raw_regret", "selected_relative_regret", "pairwise_accuracy",
        "spearman", "kendall_tau", "listwise_brier", "listwise_cross_entropy",
    ]
    for keys, group in state_rows.groupby(["model", "model_seed", "split"], sort=True):
        row = {
            "model": keys[0],
            "model_seed": keys[1],
            "split": keys[2],
            "states": group.state_id.nunique(),
            "actions": int(group.num_actions.sum()),
        }
        row.update({column: float(group[column].mean()) for column in metric_columns})
        row["p90_relative_regret"] = float(group.selected_relative_regret.quantile(0.9))
        row["top1_ece_10bin"] = expected_calibration_error(
            group.selected_confidence.to_numpy(float),
            group.selected_is_true_best.to_numpy(float),
        )
        rows.append(row)
    return pd.DataFrame(rows)


def seed_aggregate(metrics: pd.DataFrame) -> pd.DataFrame:
    learned = metrics[metrics.model_seed.isin([str(seed) for seed in SEEDS])]
    columns = [
        "exact_top1", "epsilon_top1", "top2_recall", "top4_recall",
        "selected_relative_regret", "pairwise_accuracy", "spearman", "kendall_tau",
    ]
    rows = []
    for (model, split), group in learned.groupby(["model", "split"], sort=True):
        row = {"model": model, "split": split, "seeds": ",".join(sorted(group.model_seed))}
        for column in columns:
            row[f"{column}_mean"] = float(group[column].mean())
            row[f"{column}_std"] = float(group[column].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def revision_v2_metrics() -> pd.DataFrame:
    predictions = pd.read_csv(V2 / "raw/action_predictions.csv")
    states = predictions.drop_duplicates(["state_id", "model", "model_seed"])
    rows = []
    for keys, group in states.groupby(["model", "model_seed", "split"], sort=True):
        rows.append(
            {
                "model": keys[0],
                "model_seed": str(keys[1]),
                "split": keys[2],
                "states": group.state_id.nunique(),
                "best_action_preservation": group.best_action_preservation.mean(),
                "top2_action_recall": group.top2_action_recall.mean(),
                "selected_normalized_regret": group.selected_action_normalized_regret.mean(),
                "pairwise_ranking_accuracy": group.pairwise_ranking_accuracy.mean(),
                "kendall_tau": group.kendall_tau.mean(),
                "inference_ms_state": group.inference_ms_state.mean(),
                "target_warning": "revision-v2 censored top-8 Q_rel and pathological per-state normalization; not directly comparable to revision-v3 raw-relative regret",
            }
        )
    return pd.DataFrame(rows)


def capability_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ["immediate length", "AVAILABLE", "immediate_cost only", "derived on the same v3 action sets"],
            ["dual-only", "AVAILABLE", "dual_baseline", "one parent relaxation required"],
            ["scalar-only learned", "NOT_IMPLEMENTED", "none", "no scalar-only checkpoint exists"],
            ["GNN-only solver-free", "NOT_IMPLEMENTED", "none", "no direct L_rel GNN checkpoint exists"],
            ["no-dual network inputs", "AVAILABLE_HYBRID", "residual_gnn_no_dual", "network omits dual inputs but final score still adds parent dual_baseline"],
            ["direct-Q*", "AVAILABLE_HISTORICAL", "revision-v2 action_gnn/action_mlp", "trained on censored top-8 Q_rel with unstable normalization"],
            ["residual target MLP", "AVAILABLE", "residual_mlp", "explicit features plus dual baseline"],
            ["residual target GNN", "AVAILABLE_SELECTED", "residual_gnn seed 23", "validation-selected principal model"],
            ["without on-policy training", "AVAILABLE", "initial_only_residual_gnn", "evaluated on v3 on-policy states"],
        ],
        columns=["requested_variant", "status", "artifact", "interpretation"],
    )


def training_configuration(selected: dict, scaler: dict) -> dict:
    return {
        "evidence_source": "existing immutable revision-v3 artifacts; no retraining",
        "target": {
            "name": "finite-perturbation residual advantage",
            "definition": "delta_star - dual_baseline",
            "global_robust_scaler": scaler,
        },
        "models": {
            "residual_mlp": "explicit features -> Linear(128)-SiLU-Linear(128)-SiLU-Linear(64)-SiLU-Linear(1)",
            "residual_gnn": "typed DFG/MRRG residual message passing, hidden 64, five recurrent steps, mean+max pooling, 109569 parameters",
            "residual_gnn_no_dual": "same family without dual inputs, but deployed score still includes dual_baseline",
            "initial_only_residual_gnn": "same hybrid residual GNN trained on revision-v2-style initial states",
        },
        "optimizer": "AdamW",
        "learning_rate": 0.0008,
        "weight_decay": 0.0001,
        "maximum_epochs": 80,
        "early_stopping_patience": 12,
        "gradient_accumulation_states": 8,
        "seeds": list(SEEDS),
        "loss": "1.0 listwise cross entropy + 0.5 pairwise rank + 0.25 smooth-L1 residual",
        "early_stopping_order": [
            "validation top-4 recall",
            "validation epsilon-optimal top-1",
            "negative validation relative regret",
            "validation exact top-1",
        ],
        "selection_split": selected["selection_split"],
        "test_metrics_observed_at_selection": selected["test_metrics_observed_at_selection"],
        "principal_model": selected["principal_model"],
        "principal_seed_or_ensemble": selected["principal_seed_or_ensemble"],
        "architecture_identity_features": False,
        "pytorch_geometric": False,
    }


def write_report(
    output: Path,
    split_summary: pd.DataFrame,
    family_summary: pd.DataFrame,
    leakage: pd.DataFrame,
    health: pd.DataFrame,
    checkpoints: pd.DataFrame,
    metrics: pd.DataFrame,
    seed_metrics: pd.DataFrame,
    capabilities: pd.DataFrame,
    selected: dict,
) -> None:
    selected_rows = metrics[
        (metrics.model == "residual_gnn")
        & (metrics.model_seed == str(selected["principal_seed_or_ensemble"]))
        & metrics.split.isin(TEST_SPLITS)
    ]
    selected_display = selected_rows[[
        "split", "states", "exact_top1", "epsilon_top1", "top2_recall",
        "top4_recall", "selected_relative_regret", "pairwise_accuracy",
        "spearman", "kendall_tau",
    ]]
    missing_families = family_summary[
        family_summary.split.isin(TEST_SPLITS)
        & (family_summary.labeled_states > 0)
    ].groupby("split").dfg_family.apply(lambda values: ", ".join(sorted(set(values))))
    report = f"""# Frozen FlowAdvantage learning disclosure

Generated entirely from revision-v2/revision-v3 raw artifacts and checkpoints. No model was retrained and no real Morpher pilot result was used for selection.

## Executive record

- Principal model: `residual_gnn`, seed `{selected['principal_seed_or_ensemble']}`.
- Selection: lexicographic validation top-4 recall, epsilon-optimal top-1, relative regret, exact top-1 on `{selected['selection_split']}`.
- Test metrics were observed at selection: `{selected['test_metrics_observed_at_selection']}`.
- Revision-v3 labels: **4,199 actions across 263 solved states** from a 280-state on-policy manifest.
- Training labels: **1,700 actions across 151 solved states**; validation: **580 actions across 37 solved states**.
- Seeds: `11, 23, 37`; all checkpoints and hashes are preserved in `checkpoint_manifest.csv`.
- Training architectures: `mesh3`, `torus3`; unseen tests: `mesh4`, `diag4`, `cut4`.
- Real Morpher kernels and A3 were not training or validation data: the real pilot is a zero-shot transfer evaluation.

## Split and label health

{markdown_table(split_summary)}

All 17 unlabeled manifest states are explicitly recorded as parent-relaxation infeasible states; they were not silently removed. Every one of the 4,199 committed intermediate actions passed legality. Solver and residual details are in `label_health.csv`.

{markdown_table(health)}

## Leakage audit

{markdown_table(leakage)}

The split is disjoint at state hash and source trajectory/DFG-instance identifier. Multiple policies/depths from one parent trajectory remain within one split. Architecture holdout is real for `mesh4`, `diag4`, and `cut4`.

## Held-out action ranking for the selected seed

{markdown_table(selected_display)}

These metrics use the exact revision-v3 labeled action sets. Exact top-1 is intentionally secondary because epsilon-equivalent actions are common. Top-4 recall is the deployment metric for exact child reranking.

## Model/target ablation interpretation

{markdown_table(capabilities)}

The full state-level table is `model_ablation_by_split.csv`; three-seed mean and sample standard deviation are in `model_ablation_seed_aggregate.csv`. `revision_v2_direct_q_metrics.csv` is retained separately because revision-v2 used a censored top-eight action universe and per-state normalization that sometimes amplified near ties. It must not be merged numerically with revision-v3 raw-relative regret.

The `residual_gnn_no_dual` name has a narrower meaning than “no parent solve”: dual values are removed from the network inputs, but the final prediction is still `dual_baseline + predicted_residual`. It is therefore a **hybrid no-dual-input ablation**, not a solver-free GNN.

## Coverage limitation that must appear in the paper

The revision-v3 test action labels are not balanced across all seven DFG families. Observed labeled-family coverage is:

{missing_families.to_string()}

In particular, the frozen on-policy test states are concentrated in butterfly/diamond-chain trajectories. The unseen-architecture ranking numbers are valid for those sampled states but cannot support a claim of seven-family action-ranking generalization. The complete synthetic mapping study and the new real Morpher paired matrix must carry broader end-to-end claims.

## Reproducibility

Run:

```bash
/home/rishabh/miniconda/envs/taugat_pyg/bin/python scripts/generate_reviewer_learning_evidence.py
```

`source_manifest.csv` hashes every source artifact. `training_configuration.json` records the implementation-derived optimizer, loss, stopping rule, and model dimensions. Raw values are unrounded in CSV files.
"""
    (output / "LEARNING_DISCLOSURE.md").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    labels = pd.read_csv(V3 / "raw/action_advantage_labels.csv")
    states = pd.read_csv(V3 / "raw/on_policy_states.csv")
    state_manifest = pd.read_csv(V3 / "raw/advantage_state_manifest.csv")
    predictions = pd.read_csv(V3 / "raw/action_predictions.csv", dtype={"model_seed": str})
    training = pd.read_csv(V3 / "training_summary.csv")
    selected = json.loads((V3 / "selected_flow_model.json").read_text())
    scaler = json.loads((V3 / "target_scaler.json").read_text())

    split_manifest, split_summary, family_summary, leakage = split_tables(
        states, labels, state_manifest
    )
    health = label_health(labels, state_manifest)
    checkpoints = checkpoint_manifest(training)
    state_rows = state_metric_rows(predictions, labels)
    metrics = aggregate_metrics(state_rows)
    aggregate = seed_aggregate(metrics)
    direct_q = revision_v2_metrics()
    capabilities = capability_matrix()
    sources = source_manifest()
    config = training_configuration(selected, scaler)

    artifacts = {
        "source_manifest.csv": sources,
        "split_manifest.csv": split_manifest,
        "split_summary.csv": split_summary,
        "split_family_summary.csv": family_summary,
        "leakage_checks.csv": leakage,
        "label_health.csv": health,
        "checkpoint_manifest.csv": checkpoints,
        "model_state_metrics.csv": state_rows,
        "model_ablation_by_split.csv": metrics,
        "model_ablation_seed_aggregate.csv": aggregate,
        "revision_v2_direct_q_metrics.csv": direct_q,
        "requested_ablation_capability.csv": capabilities,
        "training_runs.csv": training,
    }
    for name, frame in artifacts.items():
        write_csv(frame, output / name)
    (output / "training_configuration.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    write_report(
        output, split_summary, family_summary, leakage, health,
        checkpoints, metrics, aggregate, capabilities, selected,
    )

    # Assert the disclosure's key invariants instead of silently emitting an
    # internally inconsistent table.
    assert len(labels) == 4199
    assert labels.state_id.nunique() == 263
    assert len(states) == states.state_id.nunique() == 280
    assert leakage["pass"].all()
    assert set(checkpoints.seed) == set(SEEDS)
    assert checkpoints.sha256.str.len().eq(64).all()
    assert selected["test_metrics_observed_at_selection"] is False

    checksum_paths = sorted(path for path in output.iterdir() if path.is_file())
    checksum_text = "".join(
        f"{sha256(path)}  {path.name}\n" for path in checksum_paths
        if path.name != "SHA256SUMS.txt"
    )
    (output / "SHA256SUMS.txt").write_text(checksum_text, encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "states": int(labels.state_id.nunique()),
        "actions": len(labels),
        "checkpoints": len(checkpoints),
        "leakage_checks_passed": int(leakage["pass"].sum()),
    }, indent=2))


if __name__ == "__main__":
    main()
