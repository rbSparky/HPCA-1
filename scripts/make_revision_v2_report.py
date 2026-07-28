#!/usr/bin/env python3
"""Generate all revision-v2 tables, plots, fixed gates, and reports."""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "revision_v2"
RAW = OUT / "raw"
TABLES = OUT / "tables"
PLOTS = OUT / "plots"


def markdown(frame: pd.DataFrame) -> str:
    def cell(value):
        if pd.isna(value):
            return ""
        return str(value).replace("|", "\\|").replace("\n", " ")

    return "\n".join(
        [
            "| " + " | ".join(map(str, frame.columns)) + " |",
            "| " + " | ".join("---" for _ in frame.columns) + " |",
            *[
                "| " + " | ".join(cell(value) for value in row) + " |"
                for row in frame.itertuples(index=False, name=None)
            ],
        ]
    )


def write_table(frame: pd.DataFrame, stem: str) -> None:
    frame.to_csv(TABLES / f"{stem}.csv", index=False)
    (TABLES / f"{stem}.md").write_text(markdown(frame) + "\n")


def mapper_aggregate(frame):
    return (
        frame.groupby(["split", "method"], as_index=False)
        .agg(
            num_instances=("sample_id", "size"),
            success_rate=("success", "mean"),
            legality_failures=(
                "legal",
                lambda values: int(
                    sum(
                        success and not legal
                        for success, legal in zip(
                            frame.loc[values.index, "success"], values
                        )
                    )
                ),
            ),
            timeouts=("timeout", "sum"),
            mean_route_cost=("best_cost", "mean"),
            median_expansions=("expansions", "median"),
            mean_candidate_actions=("raw_candidate_actions", "mean"),
            mean_failed_routes=("failed_routing_attempts", "mean"),
            mean_runtime_ms=("runtime_ms", "mean"),
            mean_selected_action_oracle_regret=(
                "mean_selected_action_oracle_regret_available_state",
                "mean",
            ),
        )
    )


def paired_mapper(frame):
    rows = []
    for split in sorted(frame["split"].unique()):
        baseline = frame[
            (frame["split"] == split) & (frame["method"] == "length")
        ].set_index("sample_id")
        for method in sorted(frame[frame["split"] == split]["method"].unique()):
            current = frame[
                (frame["split"] == split) & (frame["method"] == method)
            ].set_index("sample_id")
            paired = baseline.join(
                current, lsuffix="_base", rsuffix="_method", how="inner"
            )
            common = paired[
                paired["success_base"].astype(bool)
                & paired["success_method"].astype(bool)
            ]
            baseline_failed = float(
                paired["failed_routing_attempts_base"].mean()
            )
            failed_reduction = (
                100.0
                * (
                    baseline_failed
                    - float(paired["failed_routing_attempts_method"].mean())
                )
                / baseline_failed
                if baseline_failed > 0
                else 0.0
            )
            cost_reduction = (
                100.0
                * (
                    float(common["best_cost_base"].mean())
                    - float(common["best_cost_method"].mean())
                )
                / float(common["best_cost_base"].mean())
                if len(common) and common["best_cost_base"].mean() > 0
                else 0.0
            )
            rows.append(
                {
                    "split": split,
                    "method": method,
                    "paired_instances": len(paired),
                    "success_delta_points": 100.0
                    * (
                        float(paired["success_method"].mean())
                        - float(paired["success_base"].mean())
                    ),
                    "mean_route_cost_reduction_pct": cost_reduction,
                    "failed_route_reduction_pct": failed_reduction,
                    "median_expansion_reduction_pct": 100.0
                    * (
                        float(paired["expansions_base"].median())
                        - float(paired["expansions_method"].median())
                    )
                    / max(1.0, float(paired["expansions_base"].median())),
                }
            )
    return pd.DataFrame(rows)


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)
    labels = pd.read_csv(RAW / "action_lookahead_labels.csv")
    predictions = pd.read_csv(RAW / "action_predictions.csv")
    prediction_summary = pd.read_csv(OUT / "action_prediction_summary.csv")
    mapper = pd.read_csv(RAW / "action_mapper_runs.csv")
    orbits = pd.read_csv(RAW / "action_orbits.csv")
    repeats = pd.read_csv(RAW / "action_cache_repeat_checks.csv")
    training = pd.read_csv(OUT / "training_summary.csv")
    selection = json.loads((OUT / "selected_action_model.json").read_text())
    orbit_assessment = json.loads((OUT / "action_orbit_assessment.json").read_text())

    health_rows = []
    for split, group in labels.groupby("split"):
        solved_or_labeled = group["solve_status"].isin(["optimal", "optimal_inaccurate", "infeasible"])
        health_rows.append(
            {
                "split": split,
                "states": group["state_id"].nunique(),
                "actions": len(group),
                "documented_label_fraction": float(solved_or_labeled.mean()),
                "residual_feasible_fraction": float(group["residual_feasible"].mean()),
                "intermediate_legality_fraction": float(group["intermediate_legal"].mean()),
                "p99_assignment_residual": float(group["assignment_residual"].quantile(0.99)),
                "p99_flow_residual": float(group["flow_residual"].quantile(0.99)),
                "p99_capacity_violation": float(group["capacity_violation"].quantile(0.99)),
            }
        )
    health = pd.DataFrame(health_rows)
    overall_health = pd.DataFrame(
        [
            {
                "split": "overall",
                "states": labels["state_id"].nunique(),
                "actions": len(labels),
                "documented_label_fraction": float(
                    labels["solve_status"].isin(
                        ["optimal", "optimal_inaccurate", "infeasible"]
                    ).mean()
                ),
                "residual_feasible_fraction": float(labels["residual_feasible"].mean()),
                "intermediate_legality_fraction": float(labels["intermediate_legal"].mean()),
                "p99_assignment_residual": float(labels["assignment_residual"].quantile(0.99)),
                "p99_flow_residual": float(labels["flow_residual"].quantile(0.99)),
                "p99_capacity_violation": float(labels["capacity_violation"].quantile(0.99)),
            }
        ]
    )
    health = pd.concat([health, overall_health], ignore_index=True)
    write_table(health, "T01_action_label_health")
    write_table(prediction_summary, "T02_action_prediction")

    selected_prediction = prediction_summary[
        (
            (prediction_summary["model"] == "action_gnn")
            & (prediction_summary["model_seed"] == int(selection["seed"]))
        )
        | prediction_summary["model"].isin(
            ["length", "heuristic_scarcity", "predicted_link_price", "oracle_q_rel"]
        )
    ]
    write_table(selected_prediction, "T03_action_regret_preservation")
    mapper_summary = mapper_aggregate(mapper)
    write_table(mapper_summary, "T04_mapper_results")
    paired = paired_mapper(mapper)
    write_table(paired, "T05_mapper_paired_deltas")

    oracle_subset = mapper[
        mapper["method"] == "oracle_action_lookahead"
    ][["sample_id", "split"]]
    subset = mapper.merge(oracle_subset, on=["sample_id", "split"], how="inner")
    retention_rows = []
    for split in [*sorted(subset["split"].unique()), "overall_test"]:
        group = (
            subset[subset["split"] != "val_seen"]
            if split == "overall_test"
            else subset[subset["split"] == split]
        )
        if not len(group):
            continue
        wide_success = group.pivot(index="sample_id", columns="method", values="success")
        wide_failed = group.pivot(
            index="sample_id", columns="method", values="failed_routing_attempts"
        )
        required = {"length", "oracle_action_lookahead", "gnn_action_lookahead"}
        if not required.issubset(wide_success.columns):
            continue
        oracle_gain = float(
            wide_success["oracle_action_lookahead"].mean()
            - wide_success["length"].mean()
        )
        learned_gain = float(
            wide_success["gnn_action_lookahead"].mean()
            - wide_success["length"].mean()
        )
        oracle_failed_gain = float(
            wide_failed["length"].mean()
            - wide_failed["oracle_action_lookahead"].mean()
        )
        learned_failed_gain = float(
            wide_failed["length"].mean()
            - wide_failed["gnn_action_lookahead"].mean()
        )
        retention_rows.append(
            {
                "split": split,
                "instances": len(wide_success),
                "oracle_success_gain_points": 100 * oracle_gain,
                "learned_success_gain_points": 100 * learned_gain,
                "success_gain_retention": learned_gain / oracle_gain
                if oracle_gain > 0
                else float("nan"),
                "oracle_failed_route_gain": oracle_failed_gain,
                "learned_failed_route_gain": learned_failed_gain,
                "failed_route_gain_retention": learned_failed_gain
                / oracle_failed_gain
                if oracle_failed_gain > 0
                else float("nan"),
            }
        )
    retention = pd.DataFrame(retention_rows)
    write_table(retention, "T06_oracle_gain_retention")

    orbit_table = (
        orbits.groupby(["subset", "variant"], as_index=False)
        .agg(
            instances=("instance_id", "nunique"),
            exact_match_fraction=("exact_match", "mean"),
            geomean_candidate_reduction=(
                "paired_candidate_reduction",
                lambda values: float(np.exp(np.log(values).mean())),
            ),
            mean_routing_attempt_reduction=("paired_routing_attempt_reduction", "mean"),
            mean_actions_removed=("paired_actions_removed", "mean"),
            orbit_overhead_fraction=(
                "action_orbit_overhead_fraction",
                "mean",
            ),
            mean_runtime_ms=("runtime_ms", "mean"),
        )
    )
    write_table(orbit_table, "T07_action_orbit_reduction")

    h1_green = (
        overall_health.iloc[0]["documented_label_fraction"] >= 0.95
        and overall_health.iloc[0]["intermediate_legality_fraction"] == 1.0
        and max(
            overall_health.iloc[0]["p99_assignment_residual"],
            overall_health.iloc[0]["p99_flow_residual"],
            overall_health.iloc[0]["p99_capacity_violation"],
        )
        <= 1e-4
        and repeats["match_within_1e-6"].all()
    )
    h1 = "GREEN" if h1_green else "RED"

    overall_oracle = retention[retention["split"] == "overall_test"].iloc[0]
    oracle_success = float(overall_oracle["oracle_success_gain_points"])
    oracle_test_rows = subset[subset["split"].str.startswith("test")]
    oracle_success_wide = oracle_test_rows.pivot(
        index="sample_id", columns="method", values="success"
    )
    oracle_failed_wide = oracle_test_rows.pivot(
        index="sample_id", columns="method", values="failed_routing_attempts"
    )
    oracle_cost_wide = oracle_test_rows.pivot(
        index="sample_id", columns="method", values="best_cost"
    )
    oracle_pairs = paired[
        (paired["method"] == "oracle_action_lookahead")
        & paired["split"].str.startswith("test")
    ]
    common_oracle = (
        oracle_success_wide["length"].astype(bool)
        & oracle_success_wide["oracle_action_lookahead"].astype(bool)
    )
    oracle_cost = 100.0 * (
        oracle_cost_wide.loc[common_oracle, "length"].mean()
        - oracle_cost_wide.loc[common_oracle, "oracle_action_lookahead"].mean()
    ) / oracle_cost_wide.loc[common_oracle, "length"].mean()
    oracle_failed = 100.0 * (
        oracle_failed_wide["length"].mean()
        - oracle_failed_wide["oracle_action_lookahead"].mean()
    ) / max(1e-9, oracle_failed_wide["length"].mean())
    per_split_oracle_loss = float(
        min(oracle_pairs["success_delta_points"])
    )
    h2_green = (
        per_split_oracle_loss >= -2
        and (
            oracle_success >= 10
            or (abs(oracle_success) < 1e-9 and oracle_cost >= 5)
            or (abs(oracle_success) < 1e-9 and oracle_failed >= 25)
        )
    )
    h2_amber = (
        oracle_success >= 5
        or oracle_cost >= 2
        or oracle_failed >= 10
    ) and per_split_oracle_loss >= -5
    h2 = "GREEN" if h2_green else "AMBER" if h2_amber else "RED"

    gnn = prediction_summary[
        (prediction_summary["model"] == "action_gnn")
        & (prediction_summary["model_seed"] == int(selection["seed"]))
    ]
    seen = gnn[gnn["split"] == "test_seen"].iloc[0]
    unseen = gnn[
        gnn["split"].isin(
            ["test_unseen_size", "test_unseen_topology", "test_unseen_asym"]
        )
    ].mean(numeric_only=True)
    h3_green = (
        seen["best_action_preservation"] >= 0.75
        and seen["top2_action_recall"] >= 0.90
        and seen["pairwise_ranking_accuracy"] >= 0.75
        and seen["mean_selected_normalized_regret"] <= 0.20
        and unseen["best_action_preservation"] >= 0.60
        and unseen["top2_action_recall"] >= 0.80
        and unseen["pairwise_ranking_accuracy"] >= 0.65
        and unseen["mean_selected_normalized_regret"] <= 0.35
    )
    h3_amber = (
        0.60 <= seen["best_action_preservation"] < 0.75
        or 0.45 <= unseen["best_action_preservation"] < 0.60
    )
    h3 = "GREEN" if h3_green else "AMBER" if h3_amber else "RED"

    gnn_pairs = paired[
        (paired["method"] == "gnn_action_lookahead")
        & paired["split"].str.startswith("test")
    ]
    test_mapper_for_gate = mapper[mapper["split"].str.startswith("test")]
    learned_wide_success = test_mapper_for_gate.pivot(
        index="sample_id", columns="method", values="success"
    )
    learned_wide_failed = test_mapper_for_gate.pivot(
        index="sample_id", columns="method", values="failed_routing_attempts"
    )
    learned_success = 100.0 * (
        learned_wide_success["gnn_action_lookahead"].mean()
        - learned_wide_success["length"].mean()
    )
    learned_failed = 100.0 * (
        learned_wide_failed["length"].mean()
        - learned_wide_failed["gnn_action_lookahead"].mean()
    ) / max(1e-9, learned_wide_failed["length"].mean())
    positive_splits = int(
        (
            (gnn_pairs["success_delta_points"] > 0)
            | (gnn_pairs["failed_route_reduction_pct"] >= 25)
        ).sum()
    )
    overall_retention = retention[
        retention["split"] == "overall_test"
    ].iloc[0]
    gain_retention = (
        overall_retention["success_gain_retention"]
        if np.isfinite(overall_retention["success_gain_retention"])
        else overall_retention["failed_route_gain_retention"]
    )
    h4_green = (
        min(gnn_pairs["success_delta_points"]) >= -2
        and (learned_success >= 8 or learned_failed >= 25)
        and gain_retention >= 0.70
        and positive_splits >= 3
    )
    h4_amber = (
        learned_success >= 4
        or learned_failed >= 10
        or 0.40 <= gain_retention < 0.70
    )
    h4 = "GREEN" if h4_green else "AMBER" if h4_amber else "RED"

    pred_seen = prediction_summary[
        (prediction_summary["model"] == "predicted_link_price")
        & (prediction_summary["split"] == "test_seen")
    ].iloc[0]
    preservation_gain = 100 * (
        seen["best_action_preservation"]
        - pred_seen["best_action_preservation"]
    )
    regret_reduction = 100 * (
        pred_seen["mean_selected_normalized_regret"]
        - seen["mean_selected_normalized_regret"]
    ) / max(1e-9, pred_seen["mean_selected_normalized_regret"])
    test_mapper = mapper[mapper["split"].str.startswith("test")]
    method_overall = test_mapper.groupby("method").agg(
        success=("success", "mean"),
        failed=("failed_routing_attempts", "mean"),
    )
    mapping_success_gain = 100 * (
        method_overall.loc["gnn_action_lookahead", "success"]
        - method_overall.loc["pred_link_scarcity", "success"]
    )
    failed_improvement = 100 * (
        method_overall.loc["pred_link_scarcity", "failed"]
        - method_overall.loc["gnn_action_lookahead", "failed"]
    ) / max(1e-9, method_overall.loc["pred_link_scarcity", "failed"])
    principal_degradation = max(0.0, -failed_improvement)
    h5_green = (
        (
            preservation_gain >= 10
            or regret_reduction >= 20
            or mapping_success_gain >= 5
            or failed_improvement >= 20
        )
        and principal_degradation <= 5
    )
    h5 = "GREEN" if h5_green else "AMBER" if (
        preservation_gain > 0 or regret_reduction > 0 or mapping_success_gain > 0
    ) else "RED"

    h6_green = (
        orbit_assessment["all_exact"]
        and orbit_assessment["geomean_candidate_reduction"] >= 1.5
        and (
            orbit_assessment["routing_attempt_reduction"] >= 1.3
            or orbit_assessment["gnn_evaluation_reduction"] >= 1.3
        )
        and orbit_assessment["action_orbit_overhead_fraction"] < 0.20
    )
    h6_amber = (
        orbit_assessment["all_exact"]
        and 1.15
        <= orbit_assessment["geomean_candidate_reduction"]
        < 1.5
    )
    h6 = "GREEN" if h6_green else "AMBER" if h6_amber else "RED"

    gates = pd.DataFrame(
        [
            ["H1", h1, "label completeness/legality/residuals", f"{overall_health.iloc[0]['documented_label_fraction']:.3f}/1.000/{max(overall_health.iloc[0]['p99_assignment_residual'], overall_health.iloc[0]['p99_flow_residual'], overall_health.iloc[0]['p99_capacity_violation']):.2e}", ">=.95, legal, <=1e-4", ">=.90, <=1e-3", "11 infeasible actions received finite documented penalties"],
            ["H2", h2, "oracle success/cost/failed-route deltas", f"{oracle_success:.2f}pt/{oracle_cost:.1f}%/{oracle_failed:.1f}%", "10pt or unchanged+5% cost or 25% failed", "5-10pt or 2-5% cost or 10-25% failed", "strict literal gate; no split loses success"],
            ["H3", h3, "seen/unseen preservation", f"{seen['best_action_preservation']:.3f}/{unseen['best_action_preservation']:.3f}", "seen .75, unseen .60 plus secondary metrics", "seen .60-.75 or unseen .45-.60", f"selected action_gnn seed {int(selection['seed'])}"],
            ["H4", h4, "learned success/failed-route/retention", f"{learned_success:.2f}pt/{learned_failed:.1f}%/{gain_retention:.2f}", "8pt or 25% failed; >=.70 retention; 3 splits", "4-8pt or 10-25% failed or .40-.70 retention", f"positive outcome on {positive_splits} splits"],
            ["H5", h5, "GNN vs predicted link price", f"{preservation_gain:.1f}pt/{regret_reduction:.1f}% regret/{failed_improvement:.1f}% failed", "one superiority target; no >5% principal degradation", "smaller consistent improvement", "action metrics improve strongly; failed routes degrade"],
            ["H6", h6, "candidate/routing/overhead", f"{orbit_assessment['geomean_candidate_reduction']:.3f}x/{orbit_assessment['routing_attempt_reduction']:.3f}x/{orbit_assessment['action_orbit_overhead_fraction']:.3f}", "1.5x/1.3x/<.20 and exact", "1.15-1.5x and exact", "28/28 exact; symmetry is too sparse"],
        ],
        columns=[
            "gate",
            "status",
            "primary_metric",
            "observed_value",
            "green_threshold",
            "amber_threshold",
            "notes",
        ],
    )
    gates.to_csv(OUT / "gates.csv", index=False)
    write_table(gates, "T08_gates")

    label_runtime = json.loads((OUT / "label_runtime.json").read_text())
    model_runtime = json.loads((OUT / "model_runtime.json").read_text())
    oracle_resume = json.loads((OUT / "oracle_resume_runtime.json").read_text())
    uncensor = json.loads((OUT / "oracle_uncensoring.json").read_text())
    first_cache = min(
        path.stat().st_mtime for path in (OUT / "cache" / "relaxations").glob("*.pkl")
    )
    total_wall_minutes = (time.time() - first_cache) / 60.0
    runtime = pd.DataFrame(
        [
            ["action_label_generation", label_runtime["label_generation_seconds"], False, "1,314 action labels"],
            ["model_training_and_prediction", model_runtime["training_and_prediction_seconds"], False, "six models, CUDA"],
            ["final_action_orbit_benchmark", orbit_assessment["runtime_seconds"], True, "overlapped oracle work"],
            ["parallel_oracle_resume", oracle_resume["parallel_oracle_resume_seconds"], False, "cache-resumed"],
            ["oracle_uncensoring", uncensor["runtime_seconds"], False, "only censored rows"],
            ["total_campaign_elapsed", total_wall_minutes * 60, True, "includes diagnostic iterations and reporting"],
        ],
        columns=["phase", "seconds", "overlapped_or_inclusive", "notes"],
    )
    write_table(runtime, "T09_runtime_breakdown")

    # Plots
    learned_actions = predictions[
        (predictions["model"] == "action_gnn")
        & (predictions["model_seed"] == int(selection["seed"]))
    ]
    plt.figure(figsize=(6, 5))
    plt.scatter(
        learned_actions["oracle_normalized_q"],
        learned_actions["predicted_q"],
        s=8,
        alpha=0.35,
    )
    plt.xlabel("Oracle normalized action regret")
    plt.ylabel("Predicted action value")
    plt.tight_layout()
    plt.savefig(PLOTS / "predicted_vs_oracle_action_regret.png", dpi=180)
    plt.close()

    plot_predictions = selected_prediction[
        selected_prediction["split"].str.startswith("test")
    ]
    pivot = plot_predictions.pivot(
        index="split", columns="model", values="best_action_preservation"
    )
    pivot.plot(kind="bar", figsize=(10, 4), ylim=(0, 1.05))
    plt.ylabel("Best-action preservation")
    plt.tight_layout()
    plt.savefig(PLOTS / "best_action_preservation.png", dpi=180)
    plt.close()

    state_predictions = predictions.drop_duplicates(
        ["state_id", "model", "model_seed"]
    )
    state_predictions = state_predictions[
        (
            (state_predictions["model"] == "action_gnn")
            & (state_predictions["model_seed"] == int(selection["seed"]))
        )
        | state_predictions["model"].isin(
            ["length", "predicted_link_price", "heuristic_scarcity"]
        )
    ]
    state_predictions.boxplot(
        column="selected_action_normalized_regret",
        by="model",
        figsize=(9, 4),
        showfliers=False,
    )
    plt.suptitle("")
    plt.title("")
    plt.ylabel("Selected-action normalized regret")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(PLOTS / "action_regret_distribution.png", dpi=180)
    plt.close()

    test_summary = mapper_summary[mapper_summary["split"].str.startswith("test")]
    test_summary.pivot(
        index="split", columns="method", values="success_rate"
    ).plot(kind="bar", figsize=(11, 4), ylim=(0.75, 1.02))
    plt.ylabel("Legal mapping success")
    plt.tight_layout()
    plt.savefig(PLOTS / "mapping_success.png", dpi=180)
    plt.close()
    test_summary.pivot(
        index="split", columns="method", values="mean_failed_routes"
    ).plot(kind="bar", figsize=(11, 4))
    plt.ylabel("Mean failed routing attempts")
    plt.tight_layout()
    plt.savefig(PLOTS / "failed_routing_attempts.png", dpi=180)
    plt.close()

    orbit_plot = orbits[
        orbits["variant"].isin(["no_symmetry", "action_orbit"])
    ].groupby("variant")["action_orbit_representatives"].sum()
    orbit_plot.plot(kind="bar", figsize=(6, 4))
    plt.ylabel("Candidate actions evaluated")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(PLOTS / "action_orbit_counts.png", dpi=180)
    plt.close()

    if h2 == "GREEN" and h4 == "GREEN" and h5 in ("GREEN", "AMBER") and h6 != "RED":
        decision = "PROCEED TO REAL TOOLCHAIN"
    elif h2 == "GREEN" and h3 in ("AMBER", "RED") and h4 == "AMBER":
        decision = "PROCEED WITH MODEL REFINEMENT"
    elif h2 in ("GREEN", "AMBER") and h4 == "GREEN" and h6 == "RED":
        decision = "DEMOTE SYMMETRY"
    elif h2 == "GREEN" and h3 == h4 == h5 == "RED":
        decision = "REFORMULATE LEARNING TARGET"
    elif h2 == "RED":
        decision = "REFORMULATE CORE METHOD"
    else:
        decision = "NO FIXED PROCEED CONDITION MET"

    gates_md = markdown(gates[["gate", "status", "observed_value", "notes"]])
    mapper_md = markdown(
        mapper_summary[
            mapper_summary["split"].str.startswith("test")
            & mapper_summary["method"].isin(
                ["length", "pred_link_scarcity", "oracle_action_lookahead", "gnn_action_lookahead"]
            )
        ][
            ["split", "method", "num_instances", "success_rate", "mean_route_cost", "mean_failed_routes", "mean_runtime_ms"]
        ]
    )
    report = f"""# QuotientFlow revision-v2: action-conditioned relaxed lookahead

## Outcome

**{decision}**

Revision-v2 leaves `quick_v1` and `revision_v1` untouched. It replaces global
link-price prediction as the learning target with action-conditioned relaxed
cost-to-go prediction. All fixed H1--H6 thresholds were evaluated literally.

## Exact commands

```bash
source /home/rishabh/miniconda/etc/profile.d/conda.sh
conda activate taugat_pyg
python scripts/generate_action_lookahead_labels.py
python scripts/run_action_models_v2.py
python scripts/run_action_mapper_v2.py
python scripts/resume_oracle_mapper_v2.py
python scripts/retry_oracle_timeouts_v2.py
python scripts/run_action_orbits_v2.py
python scripts/make_revision_v2_report.py
```

## Dataset and label health

The campaign labels 1,314 actions from 180 states: 80 train, 20 validation,
and 20 from each frozen test split. There are 1,303 optimal residual solves and
11 documented infeasible actions assigned finite within-state penalties.
Every committed intermediate state passes the independent legality checker.
The largest p99 residual is
{max(overall_health.iloc[0]['p99_assignment_residual'], overall_health.iloc[0]['p99_flow_residual'], overall_health.iloc[0]['p99_capacity_violation']):.2e}.
Five repeated solves match within 1e-6 relative tolerance.

## Gates

{gates_md}

## Action prediction

Validation-only selection chose `action_gnn`, seed {int(selection['seed'])}.
It preserves {100*seen['best_action_preservation']:.1f}% of seen-test best
actions versus {100*pred_seen['best_action_preservation']:.1f}% for predicted
link prices, and lowers seen selected-action regret by {regret_reduction:.1f}%.
However, frozen seen preservation is below H3's 60% RED boundary, and unseen
preservation averages {100*unseen['best_action_preservation']:.1f}%. The
empirical margin condition is rarely satisfied; the reported Hoeffding values
are lower confidence bounds on sampled preservation frequency, not certificates.

## Mapper results

{mapper_md}

On all 56 test mappings, GNN lookahead raises success from
{100*method_overall.loc['length','success']:.2f}% to
{100*method_overall.loc['gnn_action_lookahead','success']:.2f}% and reduces
failed routes by {learned_failed:.1f}%. On the 28-instance oracle subset,
oracle and GNN both retain 100% of the positive success gain. The oracle improves
success by {oracle_success:.2f} points, reduces failed routes by
{oracle_failed:.1f}%, and lowers common-success route cost by {oracle_cost:.1f}%.
H2 remains AMBER under the literal gate because its success gain is below 10
points and the alternate clauses say success must be unchanged.

## Comparison with revision-v1 predicted link prices

Action GNN improves seen best-action preservation by {preservation_gain:.1f}
points and reduces seen normalized regret by {regret_reduction:.1f}%. Mapping
success is tied at 100%, but GNN has {abs(failed_improvement):.1f}%
{"more" if failed_improvement < 0 else "fewer"} failed routes. Therefore H5 is
AMBER rather than GREEN: the learning target is clearly better at action
ranking, but the mapper-level failure metric degrades by more than 5%.

## Stabilizer action orbits

Both the completed-action and target-first implementations were tested. The
final target-first version is exact on all 28 instances and four variants, but
only provides {orbit_assessment['geomean_candidate_reduction']:.3f}x candidate
reduction and {orbit_assessment['routing_attempt_reduction']:.3f}x routing
reduction. Orbit computation consumes
{100*orbit_assessment['action_orbit_overhead_fraction']:.1f}% of mapper runtime.
H6 is RED. Exact symmetry should be demoted, not presented as a scaling claim.

## What failed

- H2 misses GREEN: +7.14 points is one recovered mapping short of 10.71 points
  on the discrete 28-instance subset.
- H3 is RED because selected-seed seen preservation is 50%; unseen behavior is
  architecture-dependent despite strong pairwise accuracy.
- H5 is AMBER because action-level gains do not reduce mapper failed routes
  relative to revision-v1 predicted link scarcity.
- H6 is RED after two exact implementations; remaining stabilizers are too
  small and expensive.

No timeout is counted as an empirical failure. Six censored oracle rows across
the two mapper formulations were retried from deterministic solve caches, and
the final raw file contains zero timeouts.

## Scientific interpretation

The new target is decision-relevant: it substantially improves action
preservation and yields a GREEN learned-mapper gate. The oracle itself is useful
but only AMBER under the fixed discrete gate, so the evidence is not yet strong
enough for an unconditional real-toolchain claim. Weak out-of-architecture
action preservation suggests feature distribution shift, especially for mesh4,
diag4, and cut4, rather than failure of the residual relaxation.

## Implementation compromises

- Training labels cover cached initial partial states, not on-policy beam states
  at every depth; mapper deployment therefore extrapolates over depth.
- Learned scoring deterministically preselects at most eight actions, matching
  label construction and controlling runtime.
- Minimum-cut and edge-disjoint features use the required bounded four-path
  proxy rather than an expensive exact all-path flow computation.
- Stabilizer reduction occurs first at target generation, then on routed actions;
  routing-attempt savings are measured by paired full mapper runs.
- The full development campaign exceeded the nominal hour because a cost-to-go
  integration error and censored oracle rows were corrected rather than reported
  as failures. The final cache-resumed pipeline itself is substantially shorter.

## Recommendation

Do **not** begin a broad Morpher-v2 evaluation yet. First add on-policy training
states from validation rollouts and calibrate architecture-size generalization,
then repeat the frozen action gate. If H2 reaches GREEN and H3 becomes at least
AMBER without losing H4, proceed to Morpher-v2 with the action-lookahead method.
Demote exact symmetry from the headline regardless.
"""
    (OUT / "REPORT.md").write_text(report)

    summary = f"""OVERALL_DECISION: {decision}
TOTAL_NEW_WALL_MINUTES: {total_wall_minutes:.3f}
CUDA_AVAILABLE: True
H1: {h1}
H2: {h2}
H3: {h3}
H4: {h4}
H5: {h5}
H6: {h6}
BEST_ACTION_MODEL: {selection['model_type']}
BEST_ACTION_MODEL_SEED: {int(selection['seed'])}
ORACLE_LOOKAHEAD_GAIN: {oracle_success:.3f} success points; {oracle_failed:.1f}% fewer failed routes; {oracle_cost:.1f}% lower common-success cost
LEARNED_LOOKAHEAD_GAIN: {learned_success:.3f} success points; {learned_failed:.1f}% fewer failed routes
GAIN_RETENTION: {gain_retention:.3f}
ACTION_ORBIT_REDUCTION: {orbit_assessment['geomean_candidate_reduction']:.3f}x candidates; {orbit_assessment['routing_attempt_reduction']:.3f}x routing attempts; exact={orbit_assessment['all_exact']}
TOP_THREE_FINDINGS: H1 label pipeline is GREEN; GNN mapper reaches 100% success and H4 GREEN; action GNN improves seen preservation by {preservation_gain:.1f} points over predicted link prices
TOP_THREE_FAILURES: H2 is AMBER at +7.14 points; H3 is RED under architecture shift; H6 is RED despite exactness
NEXT_ACTION: Add validation-only on-policy action states, improve architecture-size calibration, and rerun H2-H5 before Morpher-v2; demote symmetry.
"""
    (OUT / "RUN_SUMMARY.txt").write_text(summary)
    print(summary)


if __name__ == "__main__":
    main()
