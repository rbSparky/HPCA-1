#!/usr/bin/env python3
"""Generate cached revision-v2 action-conditioned relaxation labels."""

from __future__ import annotations

import concurrent.futures
import json
import math
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quotientflow.action_lookahead import (
    FEATURE_NAMES,
    action_cache_key,
    action_features,
    action_flags,
    check_partial_legality,
    select_candidate_actions,
)
from quotientflow.dfg import FAMILIES
from quotientflow.mapper import enumerate_actions
from quotientflow.relaxation import solve_relaxation


OUTPUT = ROOT / "results" / "revision_v2"
RAW = OUTPUT / "raw"
CACHE = OUTPUT / "cache" / "relaxations"
RELAXATION_CONFIG = {
    "tau": 0.001,
    "solver_primary": "CLARABEL",
    "solver_fallback": "OSQP",
    "max_seconds": 8.0,
}
TARGETS = {
    "train_seen": 80,
    "val_seen": 20,
    "test_seen": 20,
    "test_unseen_size": 20,
    "test_unseen_topology": 20,
    "test_unseen_asym": 20,
}


def balanced_paths(split: str, target: int) -> list[Path]:
    source = ROOT / "results" / "cache" / "samples"
    buckets = {
        family: sorted(source.glob(f"{split}__*__{family}__*.pkl"))
        for family in FAMILIES
    }
    selected = []
    position = 0
    while len(selected) < target:
        changed = False
        for family in FAMILIES:
            if position < len(buckets[family]) and len(selected) < target:
                selected.append(buckets[family][position])
                changed = True
        if not changed:
            break
        position += 1
    return selected


def action_identifier(operation: int, target, links) -> str:
    return f"op{operation}_pe{target[0]}-{target[1]}-{target[2]}_r{'-'.join(map(str, links)) or 'none'}"


def solve_one(path: Path) -> tuple[list[dict], dict]:
    with path.open("rb") as handle:
        sample = pickle.load(handle)
    dfg, arch, state = sample["dfg"], sample["arch"], sample["state"]
    remaining = [op for op in dfg.order if op not in state.placements]
    if not remaining:
        return [], {"state_id": sample["sample_id"], "error": "complete_state"}
    operation = remaining[0]
    all_actions, attempts, failures = enumerate_actions(
        dfg, arch, state, operation, k_paths=4
    )
    selected = select_candidate_actions(
        dfg,
        arch,
        state,
        operation,
        all_actions,
        np.asarray(sample["normalized_prices"]),
        limit=8,
    )
    flags = action_flags(
        arch, state, all_actions, np.asarray(sample["normalized_prices"])
    )
    rows = []
    for action in selected:
        signature = (action.target, action.new_links)
        legal, legal_reason = check_partial_legality(dfg, arch, action.state)
        cache_key = action_cache_key(
            dfg, arch, state, operation, action, RELAXATION_CONFIG
        )
        cache_path = CACHE / f"{cache_key}.pkl"
        cache_hit = cache_path.exists()
        if legal:
            if cache_hit:
                with cache_path.open("rb") as handle:
                    relaxation = pickle.load(handle)
            else:
                relaxation = solve_relaxation(
                    dfg, arch, action.state, **RELAXATION_CONFIG
                )
                with cache_path.open("wb") as handle:
                    pickle.dump(relaxation, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            relaxation = None
        features = action_features(
            dfg,
            arch,
            state,
            operation,
            action,
            np.asarray(sample["normalized_prices"]),
        )
        residual_feasible = bool(relaxation is not None and relaxation.solved)
        residual_objective = (
            float(relaxation.objective) if residual_feasible else float("nan")
        )
        q_value = (
            float(action.base_cost + residual_objective)
            if residual_feasible
            else float("nan")
        )
        row = {
            "state_id": sample["sample_id"],
            "split": sample["split"],
            "architecture": sample["architecture"],
            "dfg_family": sample["family"],
            "partial_depth": len(state.placements) / len(dfg.graph),
            "action_id": action_identifier(operation, action.target, action.new_links),
            "operation_id": operation,
            "candidate_resource": json.dumps(action.target),
            "route_edge_ids": json.dumps(action.new_links),
            "immediate_cost": float(action.base_cost),
            "residual_objective": residual_objective,
            "q_rel_raw": q_value,
            "q_rel_normalized": float("nan"),
            "action_regret": float("nan"),
            "residual_feasible": residual_feasible,
            "oracle_action": False,
            "length_action": signature == flags["length"],
            "scarcity_action": signature == flags["scarcity"],
            "oracle_price_action": signature == flags["oracle_price"],
            "solve_status": relaxation.status if relaxation is not None else legal_reason,
            "solve_ms": (
                1000.0 * relaxation.solve_seconds if relaxation is not None else 0.0
            ),
            "num_candidate_actions": len(selected),
            "intermediate_legal": legal,
            "intermediate_legality_reason": legal_reason,
            "cache_key": cache_key,
            "cache_hit": cache_hit,
            "assignment_residual": (
                relaxation.assignment_residual if residual_feasible else float("nan")
            ),
            "flow_residual": (
                relaxation.flow_residual if residual_feasible else float("nan")
            ),
            "capacity_violation": (
                relaxation.capacity_violation if residual_feasible else float("nan")
            ),
            "min_variable": (
                relaxation.min_variable if residual_feasible else float("nan")
            ),
            "max_variable": (
                relaxation.max_variable if residual_feasible else float("nan")
            ),
            **features,
        }
        rows.append(row)
    finite = [row["q_rel_raw"] for row in rows if math.isfinite(row["q_rel_raw"])]
    if finite:
        minimum, maximum = min(finite), max(finite)
        margin = max(1.0, 0.25 * (maximum - minimum))
        penalty = maximum + margin
    else:
        minimum, penalty = 0.0, 1e3
    for row in rows:
        if not math.isfinite(row["q_rel_raw"]):
            row["q_rel_raw"] = penalty
        row["action_regret"] = row["q_rel_raw"] - min(
            r["q_rel_raw"] if math.isfinite(r["q_rel_raw"]) else penalty
            for r in rows
        )
    nonzero = [
        abs(row["action_regret"])
        for row in rows
        if abs(row["action_regret"]) > 1e-12
    ]
    denominator = float(np.median(nonzero)) + 1e-8 if nonzero else 1e-8
    for row in rows:
        row["q_rel_normalized"] = row["action_regret"] / denominator
    oracle_index = min(
        range(len(rows)),
        key=lambda index: (rows[index]["q_rel_raw"], rows[index]["action_id"]),
    )
    rows[oracle_index]["oracle_action"] = True
    manifest = {
        "state_id": sample["sample_id"],
        "sample_path": str(path.relative_to(ROOT)),
        "split": sample["split"],
        "architecture": sample["architecture"],
        "dfg_family": sample["family"],
        "actions_enumerated": len(all_actions),
        "actions_selected": len(selected),
        "routing_attempts": attempts,
        "failed_routing_attempts": failures,
        "error": "",
    }
    return rows, manifest


def repeat_checks(frame: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    checks = []
    selected = frame[frame["residual_feasible"]].head(5)
    paths = dict(zip(manifest["state_id"], manifest["sample_path"]))
    for _, row in selected.iterrows():
        with (ROOT / paths[row["state_id"]]).open("rb") as handle:
            sample = pickle.load(handle)
        operation = int(row["operation_id"])
        actions, _, _ = enumerate_actions(
            sample["dfg"], sample["arch"], sample["state"], operation, 4
        )
        target = tuple(json.loads(row["candidate_resource"]))
        links = tuple(json.loads(row["route_edge_ids"]))
        action = next(
            action
            for action in actions
            if action.target == target and action.new_links == links
        )
        repeated = solve_relaxation(
            sample["dfg"], sample["arch"], action.state, **RELAXATION_CONFIG
        )
        original = float(row["residual_objective"])
        relative = (
            abs(repeated.objective - original) / max(1.0, abs(original))
            if repeated.solved
            else float("inf")
        )
        checks.append(
            {
                "state_id": row["state_id"],
                "action_id": row["action_id"],
                "original_objective": original,
                "repeated_objective": repeated.objective,
                "relative_difference": relative,
                "match_within_1e-6": repeated.solved and relative <= 1e-6,
            }
        )
    return pd.DataFrame(checks)


def main() -> None:
    start = time.perf_counter()
    RAW.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    selected_paths = [
        path
        for split, target in TARGETS.items()
        for path in balanced_paths(split, target)
    ]
    rows = []
    manifests = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=6) as executor:
        for state_rows, manifest in executor.map(solve_one, selected_paths):
            rows.extend(state_rows)
            manifests.append(manifest)
            pd.DataFrame(rows).to_csv(
                RAW / "action_lookahead_labels.partial.csv", index=False
            )
    frame = pd.DataFrame(rows).sort_values(["split", "state_id", "action_id"])
    manifest = pd.DataFrame(manifests).sort_values(["split", "state_id"])
    frame.to_csv(RAW / "action_lookahead_labels.csv", index=False)
    manifest.to_csv(RAW / "action_state_manifest.csv", index=False)
    checks = repeat_checks(frame, manifest)
    checks.to_csv(RAW / "action_cache_repeat_checks.csv", index=False)
    runtime = {
        "label_generation_seconds": time.perf_counter() - start,
        "states": len(manifest),
        "actions": len(frame),
        "cache_hits": int(frame["cache_hit"].sum()),
        "repeat_checks_passed": bool(checks["match_within_1e-6"].all()),
    }
    (OUTPUT / "label_runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
    print(json.dumps(runtime, indent=2))
    print(frame.groupby("split").agg(states=("state_id", "nunique"), actions=("action_id", "size")).to_string())


if __name__ == "__main__":
    main()
