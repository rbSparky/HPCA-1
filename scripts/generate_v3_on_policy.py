#!/usr/bin/env python3
"""Collect deterministic on-policy partial states for FlowAdvantage."""

from __future__ import annotations

import hashlib
import json
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quotientflow.action_lookahead import (  # noqa: E402
    FEATURE_NAMES, action_features, select_candidate_actions,
)
from quotientflow.action_model import load_action_model, model_scores  # noqa: E402
from quotientflow.flow_advantage import (  # noqa: E402
    dual_baseline, flow_state_hash,
)
from quotientflow.mapper import enumerate_actions  # noqa: E402
from quotientflow.relaxation import solve_relaxation  # noqa: E402

OUT = ROOT / "results" / "revision_v3"
RAW = OUT / "raw"
STATE_CACHE = OUT / "cache" / "states"
PARENT_CACHE = OUT / "cache" / "parents"
RELAXATION_CONFIG = {
    "tau": 0.001,
    "solver_primary": "CLARABEL",
    "solver_fallback": "OSQP",
    "max_seconds": 12.0,
}
TARGETS = {
    "train_on_policy": ("train_seen", 160),
    "validation_on_policy": ("val_seen", 40),
    "test_seen": ("test_seen", 20),
    "test_unseen_size": ("test_unseen_size", 20),
    "test_unseen_topology": ("test_unseen_topology", 20),
    "test_unseen_asym": ("test_unseen_asym", 20),
}
POLICIES = ("length", "current_parent_dual", "revision_v2_action_gnn", "diverse")
DEPTHS = (0.20, 0.40, 0.60, 0.80)


def parent_relaxation(dfg, arch, state):
    key = flow_state_hash(dfg, arch, state, RELAXATION_CONFIG)
    path = PARENT_CACHE / f"{key}.pkl"
    if path.exists():
        with path.open("rb") as handle:
            return pickle.load(handle), True, key
    result = solve_relaxation(dfg, arch, state, **RELAXATION_CONFIG)
    with path.open("wb") as handle:
        pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return result, False, key


class FrozenV2Policy:
    def __init__(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.scaler, self.model_type, _ = load_action_model(
            ROOT / "results/revision_v2/checkpoints/action_gnn_seed_37.pt", device
        )

    def choose(self, sample, state, operation, actions):
        prices = np.asarray(sample["normalized_prices"])
        selected = select_candidate_actions(
            sample["dfg"], sample["arch"], state, operation,
            actions, prices, limit=8,
        )
        features = np.asarray([
            [
                action_features(
                    sample["dfg"], sample["arch"], state, operation, action, prices
                )[name]
                for name in FEATURE_NAMES
            ]
            for action in selected
        ], dtype=np.float32)
        group = {
            "features": features,
            "operation": operation,
            "targets": [action.target for action in selected],
            "new_links": [action.new_links for action in selected],
        }
        with torch.no_grad():
            scores = model_scores(
                self.model, self.model_type,
                {"dfg": sample["dfg"], "arch": sample["arch"], "state": state},
                group, self.scaler,
            ).cpu().numpy()
        return min(
            range(len(selected)),
            key=lambda i: (scores[i], selected[i].target, selected[i].new_links),
        ), selected


def choose_action(sample, state, operation, actions, policy, v2_policy):
    if policy == "length":
        return actions[0]
    if policy == "current_parent_dual":
        parent, _, _ = parent_relaxation(
            sample["dfg"], sample["arch"], state
        )
        if not parent.solved:
            return actions[0]
        return min(actions, key=lambda action: (
            dual_baseline(action, parent)[0], action.target, action.new_links
        ))
    if policy == "revision_v2_action_gnn":
        index, selected = v2_policy.choose(sample, state, operation, actions)
        return selected[index]
    # Seeded state hash selects among four lowest-length actions.
    digest = hashlib.sha256(
        (repr(state.serialize()) + "|" + sample["sample_id"]).encode()
    ).hexdigest()
    return actions[int(digest[:8], 16) % min(4, len(actions))]


def rollout(sample, policy, v2_policy):
    state = sample["state"].copy()
    visited = [state.copy()]
    success = True
    for operation in [v for v in sample["dfg"].order if v not in state.placements]:
        actions, _, _ = enumerate_actions(
            sample["dfg"], sample["arch"], state, operation, 4
        )
        if not actions:
            success = False
            break
        state = choose_action(
            sample, state, operation, actions, policy, v2_policy
        ).state.copy()
        visited.append(state.copy())
    selected = []
    for target in DEPTHS:
        nearest = min(
            visited,
            key=lambda candidate: (
                abs(len(candidate.placements) / len(sample["dfg"].graph) - target),
                candidate.serialize(),
            ),
        )
        selected.append((target, nearest))
    return success, selected


def sample_paths(split):
    paths = sorted((ROOT / "results/cache/samples").glob(f"{split}__*.pkl"))
    buckets = {}
    for path in paths:
        with path.open("rb") as handle:
            sample = pickle.load(handle)
        buckets.setdefault((sample["family"], sample["architecture"]), []).append(path)
    ordered = []
    position = 0
    while True:
        changed = False
        for key in sorted(buckets):
            if position < len(buckets[key]):
                ordered.append(buckets[key][position])
                changed = True
        if not changed:
            return ordered
        position += 1


def balanced_trim(rows, target):
    buckets = {}
    for row in rows:
        key = (
            row["dfg_family"], row["architecture"],
            row["source_trajectory_policy"], row["depth_bucket"],
        )
        buckets.setdefault(key, []).append(row)
    output, position = [], 0
    while len(output) < target:
        changed = False
        for key in sorted(buckets):
            if position < len(buckets[key]) and len(output) < target:
                output.append(buckets[key][position])
                changed = True
        if not changed:
            break
        position += 1
    return output


def main():
    started = time.perf_counter()
    RAW.mkdir(parents=True, exist_ok=True)
    STATE_CACHE.mkdir(parents=True, exist_ok=True)
    PARENT_CACHE.mkdir(parents=True, exist_ok=True)
    v2_policy = FrozenV2Policy()
    all_rows = []
    seen_hashes = set()
    for output_split, (source_split, target) in TARGETS.items():
        candidates = sample_paths(source_split)
        split_rows = []
        for path in candidates:
            with path.open("rb") as handle:
                sample = pickle.load(handle)
            for policy in POLICIES:
                success, states = rollout(sample, policy, v2_policy)
                for depth_bucket, state in states:
                    state_hash = flow_state_hash(
                        sample["dfg"], sample["arch"], state, RELAXATION_CONFIG
                    )
                    unique_key = (output_split, state_hash)
                    if unique_key in seen_hashes:
                        continue
                    seen_hashes.add(unique_key)
                    state_id = f"{output_split}__{state_hash[:16]}"
                    state_path = STATE_CACHE / f"{state_hash}.pkl"
                    if not state_path.exists():
                        payload = {
                            **sample,
                            "state": state,
                            "state_id": state_id,
                            "parent_sample_id": sample["sample_id"],
                            "source_trajectory_policy": policy,
                            "trajectory_depth": len(state.placements)
                            / len(sample["dfg"].graph),
                            "parent_mapping_success": success,
                            "parent_state_hash": state_hash,
                            "output_split": output_split,
                        }
                        with state_path.open("wb") as out:
                            pickle.dump(payload, out, protocol=pickle.HIGHEST_PROTOCOL)
                    split_rows.append({
                        "state_id": state_id,
                        "split": output_split,
                        "source_split": source_split,
                        "architecture": sample["architecture"],
                        "dfg_family": sample["family"],
                        "source_trajectory_policy": policy,
                        "depth_bucket": depth_bucket,
                        "trajectory_depth": len(state.placements)
                        / len(sample["dfg"].graph),
                        "parent_mapping_success": success,
                        "parent_state_hash": state_hash,
                        "state_path": str(state_path.relative_to(ROOT)),
                        "parent_sample_id": sample["sample_id"],
                    })
            if len(split_rows) >= target * 2:
                break
        trimmed = balanced_trim(split_rows, target)
        if len(trimmed) < target:
            raise RuntimeError(
                f"{output_split}: only {len(trimmed)} unique states for target {target}"
            )
        all_rows.extend(trimmed)
    frame = pd.DataFrame(all_rows).sort_values(["split", "state_id"])
    frame.to_csv(RAW / "on_policy_states.csv", index=False)
    runtime = {
        "seconds": time.perf_counter() - started,
        "states": len(frame),
        "by_split": frame.groupby("split").size().to_dict(),
    }
    (OUT / "on_policy_runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
    print(json.dumps(runtime, indent=2))
    print(frame.groupby([
        "split", "source_trajectory_policy", "depth_bucket"
    ]).size().to_string())


if __name__ == "__main__":
    main()
