#!/usr/bin/env python3
"""Fail-fast structural and scientific-integrity audit for revision-v2."""

from pathlib import Path
import json

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "revision_v2"


def require(paths):
    missing = [str(p.relative_to(ROOT)) for p in paths if not p.exists()]
    assert not missing, f"missing required artifacts: {missing}"


required = [
    OUT / "REPORT.md",
    OUT / "RUN_SUMMARY.txt",
    OUT / "gates.csv",
    OUT / "config_deviations.md",
    OUT / "raw" / "action_lookahead_labels.csv",
    OUT / "raw" / "action_predictions.csv",
    OUT / "raw" / "action_mapper_runs.csv",
    OUT / "raw" / "action_orbits.csv",
]
required += [OUT / "tables" / f"T{i:02d}_{name}.csv" for i, name in [
    (1, "action_label_health"),
    (2, "action_prediction"),
    (3, "action_regret_preservation"),
    (4, "mapper_results"),
    (5, "mapper_paired_deltas"),
    (6, "oracle_gain_retention"),
    (7, "action_orbit_reduction"),
    (8, "gates"),
    (9, "runtime_breakdown"),
]]
required += [OUT / "plots" / name for name in [
    "predicted_vs_oracle_action_regret.png",
    "best_action_preservation.png",
    "action_regret_distribution.png",
    "mapping_success.png",
    "failed_routing_attempts.png",
    "action_orbit_counts.png",
]]
required += [
    OUT / "checkpoints" / f"{model}_seed_{seed}.pt"
    for model in ("action_mlp", "action_gnn")
    for seed in (11, 23, 37)
]
require(required)

labels = pd.read_csv(OUT / "raw" / "action_lookahead_labels.csv")
expected_label_columns = {
    "state_id", "split", "architecture", "dfg_family", "partial_depth",
    "action_id", "operation_id", "candidate_resource", "route_edge_ids",
    "immediate_cost", "residual_objective", "q_rel_raw",
    "q_rel_normalized", "action_regret", "residual_feasible",
    "oracle_action", "length_action", "scarcity_action",
    "oracle_price_action", "solve_status", "solve_ms",
    "num_candidate_actions",
}
assert expected_label_columns <= set(labels.columns)
assert labels.state_id.nunique() == 180
assert len(labels) == 1314
assert labels.groupby("state_id").size().max() <= 8
assert labels["intermediate_legal"].all()
assert set(labels.solve_status) <= {"optimal", "infeasible"}
assert labels.groupby("state_id").oracle_action.sum().eq(1).all()

split_states = labels.groupby("split").state_id.nunique().to_dict()
assert split_states == {
    "test_seen": 20,
    "test_unseen_asym": 20,
    "test_unseen_size": 20,
    "test_unseen_topology": 20,
    "train_seen": 80,
    "val_seen": 20,
}

pred = pd.read_csv(OUT / "raw" / "action_predictions.csv")
assert {"action_mlp", "action_gnn"} <= set(pred.model)
assert {11, 23, 37} <= set(pred.model_seed)
evaluation_states = set(labels.loc[labels.split != "train_seen", "state_id"])
assert evaluation_states <= set(pred.state_id)

mapper = pd.read_csv(OUT / "raw" / "action_mapper_runs.csv")
assert not mapper.duplicated(["split", "sample_id", "method"]).any()
assert not mapper.timeout.astype(bool).any()
for method in ("length", "heuristic_scarcity", "pred_link_scarcity",
               "mlp_action_lookahead", "gnn_action_lookahead"):
    test = mapper[(mapper.method == method) & mapper.split.str.startswith("test")]
    assert len(test) == 56, (method, len(test))
oracle = mapper[(mapper.method == "oracle_action_lookahead")
                & mapper.split.str.startswith("test")]
assert len(oracle) == 28
assert mapper.loc[mapper.success.astype(bool), "legal"].all()

orbits = pd.read_csv(OUT / "raw" / "action_orbits.csv")
assert orbits.instance_id.nunique() == 28
assert len(orbits) == 112
assert orbits["exact_match"].all()
assert not orbits.timeout.astype(bool).any()

gates = pd.read_csv(OUT / "gates.csv")
assert gates.set_index("gate").status.to_dict() == {
    "H1": "GREEN", "H2": "AMBER", "H3": "RED",
    "H4": "GREEN", "H5": "AMBER", "H6": "RED",
}

selection = json.loads((OUT / "selected_action_model.json").read_text())
assert selection["model_type"] == "action_gnn"
assert selection["seed"] == 37

lines = [
    "REVISION_V2_ARTIFACT_AUDIT: PASS",
    f"LABEL_ROWS: {len(labels)}",
    f"LABEL_STATES: {labels.state_id.nunique()}",
    f"MAPPER_ROWS: {len(mapper)}",
    f"ORBIT_ROWS: {len(orbits)}",
    "FINAL_TIMEOUT_ROWS: 0",
    "INTERMEDIATE_LEGALITY: PASS",
    "ORBIT_EXACTNESS: PASS",
    "CHECKPOINTS: 6",
]
(OUT / "ARTIFACT_AUDIT.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
