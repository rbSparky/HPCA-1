#!/usr/bin/env python3
"""Generate corrected FlowAdvantage parent/child labels and selector audit."""

from __future__ import annotations

import concurrent.futures
import json
import math
import multiprocessing as mp
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quotientflow.action_lookahead import action_cache_key, check_partial_legality  # noqa: E402
from quotientflow.flow_advantage import (  # noqa: E402
    FLOW_FEATURE_NAMES,
    action_feature_rows,
    action_id,
    corrected_targets,
    dual_baseline,
    flow_state_hash,
    robust_residual_scaler,
    select_training_actions,
)
from quotientflow.mapper import enumerate_actions  # noqa: E402
from quotientflow.relaxation import solve_relaxation  # noqa: E402

OUT = ROOT / "results" / "revision_v3"
RAW = OUT / "raw"
PARENT_CACHE = OUT / "cache" / "parents"
CHILD_CACHE = OUT / "cache" / "children"
RELAXATION_CONFIG = {
    "tau": 0.001,
    "solver_primary": "CLARABEL",
    "solver_fallback": "OSQP",
    "max_seconds": 12.0,
}
AUDIT_COUNTS = {
    # Forty were requested initially; 17/280 collected parents are documented
    # relaxation-infeasible. Precommitting 48 yields at least 40 feasible audit
    # states without selecting on child objectives.
    "train_on_policy": 11,
    "validation_on_policy": 11,
    "test_seen": 8,
    "test_unseen_size": 8,
    "test_unseen_topology": 8,
    "test_unseen_asym": 8,
}


def cached_parent(dfg, arch, state):
    key = flow_state_hash(dfg, arch, state, RELAXATION_CONFIG)
    path = PARENT_CACHE / f"{key}.pkl"
    hit = path.exists()
    if hit:
        with path.open("rb") as handle:
            result = pickle.load(handle)
    else:
        result = solve_relaxation(dfg, arch, state, **RELAXATION_CONFIG)
        with path.open("wb") as handle:
            pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return result, hit, key


def target_complete_path_bounded(actions, limit_per_target=4):
    grouped = {}
    for action in sorted(
        actions, key=lambda a: (a.target, a.base_cost, a.new_links, a.state.serialize())
    ):
        grouped.setdefault(action.target, []).append(action)
    return [
        action
        for target in sorted(grouped)
        for action in grouped[target][:limit_per_target]
    ]


def cached_child(dfg, arch, state, operation, action):
    key = action_cache_key(
        dfg, arch, state, operation, action, RELAXATION_CONFIG
    )
    path = CHILD_CACHE / f"{key}.pkl"
    hit = path.exists()
    if hit:
        with path.open("rb") as handle:
            result = pickle.load(handle)
    else:
        result = solve_relaxation(dfg, arch, action.state, **RELAXATION_CONFIG)
        with path.open("wb") as handle:
            pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return result, hit, key


def solve_state(task):
    record, full_audit = task
    with (ROOT / record["state_path"]).open("rb") as handle:
        sample = pickle.load(handle)
    dfg, arch, state = sample["dfg"], sample["arch"], sample["state"]
    remaining = [v for v in dfg.order if v not in state.placements]
    if not remaining:
        return [], {"state_id": record["state_id"], "error": "complete_state"}
    operation = remaining[0]
    parent, parent_hit, parent_key = cached_parent(dfg, arch, state)
    all_actions, attempts, failures = enumerate_actions(
        dfg, arch, state, operation, 4
    )
    all_actions = [
        action for action in all_actions
        if check_partial_legality(dfg, arch, action.state)[0]
    ]
    path_bounded = target_complete_path_bounded(all_actions, 4)
    if not parent.solved or not path_bounded:
        return [], {
            "state_id": record["state_id"],
            "split": record["split"],
            "architecture": record["architecture"],
            "dfg_family": record["dfg_family"],
            "full_action_audit_state": full_audit,
            "error": "parent_" + parent.status if not parent.solved else "no_actions",
            "parent_status": parent.status,
            "all_legal_actions": len(all_actions),
        }
    features = action_feature_rows(
        dfg, arch, state, operation, path_bounded, parent
    )
    selected = select_training_actions(path_bounded, features, 12, True)
    no_dual_selected = select_training_actions(
        path_bounded, features, 12, False
    )
    solve_indices = list(range(len(path_bounded))) if full_audit else selected
    rows = []
    for index in solve_indices:
        action = path_bounded[index]
        legal, legality_reason = check_partial_legality(
            dfg, arch, action.state
        )
        child, child_hit, child_key = cached_child(
            dfg, arch, state, operation, action
        ) if legal else (None, False, "")
        feasible = bool(child is not None and child.solved)
        baseline, routing_component, compute_component = dual_baseline(
            action, parent
        )
        rows.append({
            "state_id": record["state_id"],
            "split": record["split"],
            "architecture": record["architecture"],
            "dfg_family": record["dfg_family"],
            "depth_bucket": record["depth_bucket"],
            "trajectory_depth": record["trajectory_depth"],
            "source_trajectory_policy": record["source_trajectory_policy"],
            "parent_mapping_success": record["parent_mapping_success"],
            "parent_state_hash": record["parent_state_hash"],
            "state_path": record["state_path"],
            "operation_id": operation,
            "action_id": action_id(operation, action),
            "candidate_resource": json.dumps(action.target),
            "route_edge_ids": json.dumps(action.new_links),
            "action_index": index,
            "all_legal_actions": len(all_actions),
            "path_bounded_actions": len(path_bounded),
            "full_action_audit_state": full_audit,
            "training_selector_action": index in selected,
            "no_dual_selector_action": index in no_dual_selected,
            "intermediate_legal": legal,
            "intermediate_legality_reason": legality_reason,
            "J_parent": float(parent.objective),
            "parent_status": parent.status,
            "parent_solver": parent.solver,
            "parent_solve_ms": 1000.0 * parent.solve_seconds,
            "parent_cache_hit": parent_hit,
            "parent_cache_key": parent_key,
            "parent_assignment_residual": parent.assignment_residual,
            "parent_flow_residual": parent.flow_residual,
            "parent_capacity_violation": parent.capacity_violation,
            "immediate_cost": float(action.base_cost),
            "child_objective": float(child.objective) if feasible else math.nan,
            "child_status": child.status if child is not None else legality_reason,
            "child_solver": child.solver if child is not None else "",
            "child_solve_ms": 1000.0 * child.solve_seconds
            if child is not None else 0.0,
            "child_cache_hit": child_hit,
            "child_cache_key": child_key,
            "child_feasible": feasible,
            "child_assignment_residual": child.assignment_residual
            if feasible else math.nan,
            "child_flow_residual": child.flow_residual
            if feasible else math.nan,
            "child_capacity_violation": child.capacity_violation
            if feasible else math.nan,
            "dual_baseline": baseline,
            "routing_dual_component": routing_component,
            "compute_dual_component": compute_component,
            **features[index],
        })
    finite_delta = [
        row["immediate_cost"] + row["child_objective"] - row["J_parent"]
        for row in rows if math.isfinite(row["child_objective"])
    ]
    if finite_delta:
        margin = max(1.0, .25 * (max(finite_delta) - min(finite_delta)))
        penalty_delta = max(finite_delta) + margin
    else:
        penalty_delta = 1e3
    for row in rows:
        if math.isfinite(row["child_objective"]):
            delta = row["immediate_cost"] + row["child_objective"] - row["J_parent"]
        else:
            delta = penalty_delta
        row["delta_star"] = delta
        row["residual_advantage"] = delta - row["dual_baseline"]
    delta = np.asarray([row["delta_star"] for row in rows])
    baseline = np.asarray([row["dual_baseline"] for row in rows])
    immediate = np.asarray([row["immediate_cost"] for row in rows])
    child_objective = delta + float(parent.objective) - immediate
    targets = corrected_targets(
        immediate, child_objective, float(parent.objective), baseline
    )
    for i, row in enumerate(rows):
        row["raw_regret"] = targets["raw_regret"][i]
        row["relative_regret"] = targets["relative_regret"][i]
        row["epsilon_state"] = targets["epsilon"]
        row["epsilon_optimal"] = targets["epsilon_optimal"][i]
        row["state_scale"] = targets["state_scale"]
        row["soft_target"] = targets["soft_target"][i]
    manifest = {
        "state_id": record["state_id"],
        "split": record["split"],
        "parent_status": parent.status,
        "all_legal_actions": len(all_actions),
        "path_bounded_actions": len(path_bounded),
        "actions_solved": len(rows),
        "full_action_audit_state": full_audit,
        "routing_attempts": attempts,
        "failed_routing_attempts": failures,
        "error": "",
    }
    return rows, manifest


def select_audit_ids(states):
    ids = set()
    for split, count in AUDIT_COUNTS.items():
        subset = states[states.split == split].sort_values([
            "dfg_family", "architecture", "source_trajectory_policy",
            "depth_bucket", "state_id",
        ])
        ids.update(subset.head(count).state_id)
    assert len(ids) == 54
    return ids


def selector_audit(frame):
    rows = []
    audit = frame[frame.full_action_audit_state]
    for state_id, group in audit.groupby("state_id"):
        true = group.sort_values(["delta_star", "action_id"])
        best_id = true.action_id.iloc[0]
        epsilon_ids = set(true.loc[true.epsilon_optimal, "action_id"])
        top4 = set(true.action_id.iloc[:4])
        for selector, column in (
            ("normalized_12", "training_selector_action"),
            ("no_dual_12", "no_dual_selector_action"),
        ):
            retained = group[group[column]].sort_values("action_id")
            retained_ids = set(retained.action_id)
            rows.append({
                "state_id": state_id,
                "split": group.split.iloc[0],
                "architecture": group.architecture.iloc[0],
                "dfg_family": group.dfg_family.iloc[0],
                "selector": selector,
                "legal_actions": group.path_bounded_actions.iloc[0],
                "actions_retained": len(retained),
                "true_best_recall": best_id in retained_ids,
                "epsilon_optimal_recall": bool(epsilon_ids & retained_ids),
                "top2_set_recall": len(
                    set(true.action_id.iloc[:2]) & retained_ids
                ) / min(2, len(true)),
                "top4_set_recall": len(top4 & retained_ids) / min(4, len(true)),
                "best_retained_raw_regret": float(retained.raw_regret.min()),
            })
    return pd.DataFrame(rows)


def main():
    started = time.perf_counter()
    RAW.mkdir(parents=True, exist_ok=True)
    PARENT_CACHE.mkdir(parents=True, exist_ok=True)
    CHILD_CACHE.mkdir(parents=True, exist_ok=True)
    states = pd.read_csv(RAW / "on_policy_states.csv")
    audit_ids = select_audit_ids(states)
    tasks = [
        (row._asdict(), row.state_id in audit_ids)
        for row in states.itertuples(index=False)
    ]
    rows, manifests = [], []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=6, mp_context=mp.get_context("spawn")
    ) as pool:
        for state_rows, manifest in pool.map(solve_state, tasks):
            rows.extend(state_rows)
            manifests.append(manifest)
            if len(manifests) % 10 == 0:
                pd.DataFrame(rows).to_csv(
                    RAW / "action_advantage_labels.partial.csv", index=False
                )
    frame = pd.DataFrame(rows)
    # Training-only global robust residual scaler.
    training_values = frame.loc[
        frame.split == "train_on_policy", "residual_advantage"
    ].to_numpy(float)
    center, scale = robust_residual_scaler(training_values)
    frame["residual_target"] = np.clip(
        (frame.residual_advantage - center) / scale, -8.0, 8.0
    )
    frame = frame.sort_values(["split", "state_id", "action_id"])
    frame.to_csv(RAW / "action_advantage_labels.csv", index=False)
    pd.DataFrame(manifests).sort_values(
        ["split", "state_id"]
    ).to_csv(RAW / "advantage_state_manifest.csv", index=False)
    selector = selector_audit(frame)
    selector.to_csv(RAW / "candidate_selector_audit.csv", index=False)
    config = {
        "residual_center": center,
        "residual_scale": scale,
        "training_split_only": True,
        "relaxation_config": RELAXATION_CONFIG,
    }
    (OUT / "target_scaler.json").write_text(json.dumps(config, indent=2) + "\n")
    runtime = {
        "seconds": time.perf_counter() - started,
        "states_requested": len(states),
        "states_labeled": int(frame.state_id.nunique()),
        "actions": len(frame),
        "full_audit_states": int(frame.loc[
            frame.full_action_audit_state, "state_id"
        ].nunique()),
        "parent_cache_hits": int(frame.groupby("state_id").parent_cache_hit.first().sum()),
        "child_cache_hits": int(frame.child_cache_hit.sum()),
    }
    (OUT / "label_runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
    print(json.dumps(runtime, indent=2))
    print(selector.groupby("selector").agg(
        states=("state_id", "size"),
        true_best=("true_best_recall", "mean"),
        epsilon=("epsilon_optimal_recall", "mean"),
        top4=("top4_set_recall", "mean"),
    ).to_string())


if __name__ == "__main__":
    main()
