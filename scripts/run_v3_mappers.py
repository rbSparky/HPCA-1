#!/usr/bin/env python3
"""Regression and stress mapping for FlowAdvantage top-k correction."""

from __future__ import annotations

import concurrent.futures
import json
import math
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quotientflow.action_lookahead import (  # noqa: E402
    FEATURE_NAMES, action_cache_key, action_features, select_candidate_actions,
)
from quotientflow.action_model import load_action_model, model_scores  # noqa: E402
from quotientflow.flow_advantage import (  # noqa: E402
    FLOW_FEATURE_NAMES,
    action_feature_rows,
    flow_state_hash,
    target_complete_path_bounded,
)
from quotientflow.flow_model import flow_scores, load_flow_model  # noqa: E402
from quotientflow.mapper import check_legality, map_dfg  # noqa: E402
from quotientflow.model import QuotientFlowGNN, predict_numpy  # noqa: E402
from quotientflow.relaxation import solve_relaxation  # noqa: E402
from scripts.run_revision2_scarcity_suite import adjust as scarcity_adjust  # noqa: E402

OUT = ROOT / "results" / "revision_v3"
RAW = OUT / "raw"
PARENT_CACHE = OUT / "cache" / "mapper_parents"
CHILD_CACHE = OUT / "cache" / "mapper_children"
RELAXATION_CONFIG = {
    "tau": 0.001,
    "solver_primary": "CLARABEL",
    "solver_fallback": "OSQP",
    "max_seconds": 12.0,
}
BUDGETS = {
    "tight": {"beam_width": 1, "action_limit": 3, "max_expansions": 600},
    "medium": {"beam_width": 2, "action_limit": 4, "max_expansions": 1500},
    "wide": {"beam_width": 4, "action_limit": 6, "max_expansions": 5000},
}
INEXPENSIVE = (
    "length", "revision_v1_pred_link", "revision_v2_action_gnn",
    "dual_linear", "residual_gnn_no_correction",
)
EXPENSIVE = (
    "residual_gnn_top2", "residual_gnn_top4", "full_action_oracle",
)


def atomic_pickle(path, value):
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    try:
        temporary.replace(path)
    except FileNotFoundError:
        pass


def cached_parent(dfg, arch, state):
    key = flow_state_hash(dfg, arch, state, RELAXATION_CONFIG)
    path = PARENT_CACHE / f"{key}.pkl"
    started = time.perf_counter()
    hit = path.exists()
    if hit:
        with path.open("rb") as handle:
            result = pickle.load(handle)
    else:
        result = solve_relaxation(dfg, arch, state, **RELAXATION_CONFIG)
        atomic_pickle(path, result)
    return result, hit, key, 1000.0 * (time.perf_counter() - started)


def cached_child(dfg, arch, state, operation, action):
    key = action_cache_key(
        dfg, arch, state, operation, action, RELAXATION_CONFIG
    )
    path = CHILD_CACHE / f"{key}.pkl"
    started = time.perf_counter()
    hit = path.exists()
    if hit:
        with path.open("rb") as handle:
            result = pickle.load(handle)
    else:
        result = solve_relaxation(
            dfg, arch, action.state, **RELAXATION_CONFIG
        )
        atomic_pickle(path, result)
    return result, hit, key, 1000.0 * (time.perf_counter() - started)


def load_selected_flow_models(device="cpu"):
    selection = json.loads((OUT / "selected_flow_model.json").read_text())
    selected = str(selection["principal_seed_or_ensemble"])
    seeds = (11, 23, 37) if selected == "ensemble" else (int(selected),)
    return [
        load_flow_model(
            OUT / "checkpoints" / f"residual_gnn_seed_{seed}.pt", device
        )
        for seed in seeds
    ], selected


class FlowScorer:
    def __init__(self, method, models, shared_feature_cache):
        self.method = method
        self.models = models
        self.shared_feature_cache = shared_feature_cache
        self.parent_solves = self.parent_cache_hits = 0
        self.child_solves = self.child_cache_hits = 0
        self.relaxation_ms = self.actual_relaxation_ms = 0.0
        self.feature_ms = self.actual_feature_ms = 0.0
        self.gnn_ms = 0.0

    def preselect(self, dfg, arch, state, operation, actions):
        return target_complete_path_bounded(actions, 4)

    def _parent_features(self, dfg, arch, state, operation, actions):
        parent, hit, parent_key, actual_ms = cached_parent(
            dfg, arch, state
        )
        self.parent_solves += 1
        self.parent_cache_hits += int(hit)
        self.actual_relaxation_ms += actual_ms
        self.relaxation_ms += 1000.0 * parent.solve_seconds
        signatures = tuple((a.target, a.new_links) for a in actions)
        cache_key = (parent_key, operation, signatures)
        cached = self.shared_feature_cache.get(cache_key)
        if cached is None:
            started = time.perf_counter()
            features = action_feature_rows(
                dfg, arch, state, operation, actions, parent
            ) if parent.solved else []
            elapsed = 1000.0 * (time.perf_counter() - started)
            self.shared_feature_cache[cache_key] = (features, elapsed)
            self.actual_feature_ms += elapsed
        else:
            features, elapsed = cached
        self.feature_ms += elapsed
        return parent, features

    def _predicted_delta(
        self, dfg, arch, state, operation, actions, parent, features
    ):
        frame = pd.DataFrame(features)
        group = {
            "frame": frame,
            "parent": parent,
            "operation": operation,
            "targets": [a.target for a in actions],
            "new_links": [a.new_links for a in actions],
        }
        sample = {"dfg": dfg, "arch": arch, "state": state}
        started = time.perf_counter()
        raw_predictions = []
        with torch.no_grad():
            for model, scaler, data in self.models:
                scaled = flow_scores(
                    model, data["model_type"], group, sample, scaler,
                    data["metadata"]["feature_names"],
                ).cpu().numpy()
                raw_predictions.append(
                    scaled * data["metadata"]["residual_scale"]
                    + data["metadata"]["residual_center"]
                )
        self.gnn_ms += 1000.0 * (time.perf_counter() - started)
        residual = np.mean(raw_predictions, axis=0)
        return frame.dual_baseline.to_numpy(float) + residual

    def __call__(self, dfg, arch, state, operation, actions):
        parent, features = self._parent_features(
            dfg, arch, state, operation, actions
        )
        if not parent.solved:
            return np.full(len(actions), 1e6)
        baseline = np.asarray([row["dual_baseline"] for row in features])
        if self.method == "dual_linear":
            return parent.objective + baseline
        predicted = self._predicted_delta(
            dfg, arch, state, operation, actions, parent, features
        )
        if self.method == "residual_gnn_no_correction":
            return parent.objective + predicted
        if self.method == "full_action_oracle":
            selected = list(range(len(actions)))
        else:
            k = 2 if self.method.endswith("top2") else 4
            selected = list(np.argsort(predicted, kind="stable")[:k])
        exact = np.full(len(actions), np.nan)
        finite = []
        def solve_index(index):
            return index, cached_child(
                dfg, arch, state, operation, actions[index]
            )
        # Child relaxations are independent. Concurrent solves make the
        # expensive full-action oracle practical on the available CPU while
        # retaining deterministic keys/results.
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(8, max(1, len(selected)))
        ) as child_pool:
            child_results = list(child_pool.map(solve_index, selected))
        for index, (result, hit, _, actual_ms) in child_results:
            self.child_solves += 1
            self.child_cache_hits += int(hit)
            self.actual_relaxation_ms += actual_ms
            self.relaxation_ms += 1000.0 * result.solve_seconds
            if result.solved:
                exact[index] = actions[index].base_cost + result.objective
                finite.append(exact[index])
        penalty = (
            max(finite) + max(1.0, .25 * np.ptp(finite))
            if finite else 1e6
        )
        return np.asarray([
            value if math.isfinite(value) else penalty
            for value in exact
        ])

    def stats(self):
        return {
            "parent_solves": self.parent_solves,
            "parent_cache_hits": self.parent_cache_hits,
            "child_solves": self.child_solves,
            "child_cache_hits": self.child_cache_hits,
            "relaxation_time_ms": self.relaxation_ms,
            "actual_relaxation_time_ms": self.actual_relaxation_ms,
            "feature_computation_ms": self.feature_ms,
            "actual_feature_computation_ms": self.actual_feature_ms,
            "gnn_inference_ms": self.gnn_ms,
        }


class FrozenV2Scorer:
    def __init__(self, sample, model, scaler, model_type):
        self.sample = sample
        self.model, self.scaler, self.model_type = model, scaler, model_type
        self.feature_ms = self.gnn_ms = 0.0

    def preselect(self, dfg, arch, state, operation, actions):
        bounded = target_complete_path_bounded(actions, 4)
        return select_candidate_actions(
            dfg, arch, state, operation, bounded,
            np.asarray(self.sample["normalized_prices"]), 8,
        )

    def __call__(self, dfg, arch, state, operation, actions):
        started = time.perf_counter()
        values = [
            action_features(
                dfg, arch, state, operation, action,
                np.asarray(self.sample["normalized_prices"]),
            )
            for action in actions
        ]
        self.feature_ms += 1000.0 * (time.perf_counter() - started)
        group = {
            "features": np.asarray([
                [row[name] for name in FEATURE_NAMES] for row in values
            ], np.float32),
            "operation": operation,
            "targets": [action.target for action in actions],
            "new_links": [action.new_links for action in actions],
        }
        started = time.perf_counter()
        with torch.no_grad():
            prediction = model_scores(
                self.model, self.model_type,
                {"dfg": dfg, "arch": arch, "state": state},
                group, self.scaler,
            ).cpu().numpy()
        self.gnn_ms += 1000.0 * (time.perf_counter() - started)
        return prediction


def base_row(instance, method, budget, result, stats, started):
    total_ms = 1000.0 * (time.perf_counter() - started)
    successful_legal = False
    if result.success and result.final_state is not None:
        successful_legal = check_legality(
            instance["dfg"], instance["arch"], result.final_state
        )[0]
    routing_ms = max(
        0.0,
        total_ms - stats.get("actual_relaxation_time_ms", 0.0)
        - stats.get("actual_feature_computation_ms", 0.0)
        - stats.get("gnn_inference_ms", 0.0),
    )
    return {
        "instance_id": instance["instance_id"],
        "split": instance["split"],
        "architecture": instance["architecture"],
        "dfg_family": instance["dfg_family"],
        "seed": instance["seed"],
        "budget": budget,
        "method": method,
        **result.row(),
        **stats,
        "routing_time_ms": routing_ms,
        "total_compile_time_ms": total_ms,
        "cold_equivalent_compile_time_ms": (
            total_ms - stats.get("actual_relaxation_time_ms", 0.0)
            + stats.get("relaxation_time_ms", 0.0)
        ),
        "final_independent_legality": successful_legal,
    }


def run_instance(instance_path, include_expensive):
    torch.set_num_threads(1)
    with (ROOT / instance_path).open("rb") as handle:
        instance = pickle.load(handle)
    flow_models, _ = load_selected_flow_models("cpu")
    v2_model, v2_scaler, v2_type, _ = load_action_model(
        ROOT / "results/revision_v2/checkpoints/action_gnn_seed_37.pt", "cpu"
    )
    price_checkpoint = torch.load(
        ROOT / "results/quick_v1/checkpoints/model_seed_11.pt",
        map_location="cpu", weights_only=False,
    )
    price_model = QuotientFlowGNN()
    price_model.load_state_dict(price_checkpoint["state_dict"])
    predicted_prices = predict_numpy(
        price_model, instance["dfg"], instance["arch"], instance["state"]
    )[0]
    initial_parent, _, _, _ = cached_parent(
        instance["dfg"], instance["arch"], instance["state"]
    )
    sample_for_v2 = {
        **instance,
        "normalized_prices": (
            initial_parent.normalized_prices
            if initial_parent.solved else np.zeros(len(instance["arch"].edge_list))
        ),
    }
    pred_link_prices = scarcity_adjust(sample_for_v2, predicted_prices)
    shared_features = {}
    rows = []
    override_methods = os.environ.get("FLOW_V4_METHODS", "")
    methods = tuple(x for x in override_methods.split(",") if x) if override_methods else list(INEXPENSIVE)
    if include_expensive:
        methods.extend(EXPENSIVE)
    override_budgets = os.environ.get("FLOW_V4_BUDGETS", "")
    requested_budgets = tuple(x for x in override_budgets.split(",") if x) if override_budgets else instance.get("run_budgets", tuple(BUDGETS))
    for budget_name in requested_budgets:
        budget = BUDGETS[budget_name]
        for method in methods:
            # Full oracle is run at the primary medium budget; top-k is run on
            # all budgets on the same deterministic 28-instance subset.
            if (
                method == "full_action_oracle"
                and budget_name != "medium"
                and len(requested_budgets) > 1
            ):
                continue
            started = time.perf_counter()
            if method == "length":
                result = map_dfg(
                    instance["dfg"], instance["arch"], instance["state"],
                    k_paths=4, timeout_seconds=600.0,
                    action_preselector=lambda d, a, s, o, actions:
                    target_complete_path_bounded(actions, 4),
                    **budget,
                )
                stats = {}
            elif method == "revision_v1_pred_link":
                result = map_dfg(
                    instance["dfg"], instance["arch"], instance["state"],
                    prices=pred_link_prices, price_weight=.5, price_mode="sum",
                    k_paths=4, timeout_seconds=600.0,
                    action_preselector=lambda d, a, s, o, actions:
                    target_complete_path_bounded(actions, 4),
                    **budget,
                )
                stats = {}
            elif method == "revision_v2_action_gnn":
                scorer = FrozenV2Scorer(
                    sample_for_v2, v2_model, v2_scaler, v2_type
                )
                result = map_dfg(
                    instance["dfg"], instance["arch"], instance["state"],
                    k_paths=4, timeout_seconds=600.0,
                    action_scorer=scorer,
                    action_preselector=scorer.preselect,
                    action_value_is_cost_to_go=True,
                    **budget,
                )
                stats = {
                    "feature_computation_ms": scorer.feature_ms,
                    "actual_feature_computation_ms": scorer.feature_ms,
                    "gnn_inference_ms": scorer.gnn_ms,
                }
            else:
                scorer = FlowScorer(method, flow_models, shared_features)
                result = map_dfg(
                    instance["dfg"], instance["arch"], instance["state"],
                    k_paths=4, timeout_seconds=600.0,
                    action_scorer=scorer,
                    action_preselector=scorer.preselect,
                    action_value_is_cost_to_go=True,
                    **budget,
                )
                stats = scorer.stats()
            rows.append(base_row(
                instance, method, budget_name, result, stats, started
            ))
    return rows


def regression_rows():
    old = pd.read_csv(RAW.parent.parent / "revision_v2/raw/action_mapper_runs.csv")
    keep = old[
        old.method.isin(["length", "pred_link_scarcity", "gnn_action_lookahead"])
        & old.split.str.startswith("test")
    ].copy()
    keep["method"] = keep.method.map({
        "length": "length",
        "pred_link_scarcity": "revision_v1_pred_link",
        "gnn_action_lookahead": "revision_v2_action_gnn",
    })
    keep["budget"] = "wide"
    keep["source"] = "immutable_revision_v2_regression"
    return keep


def main():
    started = time.perf_counter()
    RAW.mkdir(parents=True, exist_ok=True)
    PARENT_CACHE.mkdir(parents=True, exist_ok=True)
    CHILD_CACHE.mkdir(parents=True, exist_ok=True)
    stress = pd.read_csv(RAW / "stress_instances.csv")
    expensive_ids = set(
        stress.sort_values(["dfg_family", "architecture", "seed"])
        .groupby(["dfg_family", "architecture"], sort=True)
        .head(1).instance_id
    )
    tasks = [
        (row.instance_path, row.instance_id in expensive_ids)
        for row in stress.itertuples()
    ]
    partial_path = RAW / "stress_mapper_runs.partial.csv"
    rows = (
        pd.read_csv(partial_path).to_dict("records")
        if partial_path.exists() else []
    )
    completed_counts = pd.DataFrame(rows).groupby("instance_id").size().to_dict() if rows else {}
    remaining_tasks = []
    for path, expensive in tasks:
        instance_id = Path(path).stem
        expected = 22 if expensive else 15
        if completed_counts.get(instance_id, 0) == expected:
            continue
        # Partial instance rows are never emitted: pool results are appended
        # atomically per completed instance.
        remaining_tasks.append((path, expensive))
    with concurrent.futures.ProcessPoolExecutor(max_workers=6) as pool:
        for result in pool.map(lambda_args, remaining_tasks):
            rows.extend(result)
            pd.DataFrame(rows).to_csv(
                partial_path, index=False
            )
    frame = pd.DataFrame(rows).sort_values(
        ["budget", "method", "architecture", "dfg_family", "instance_id"]
    )
    frame.to_csv(RAW / "stress_mapper_runs.csv", index=False)
    regression_rows().to_csv(RAW / "regression_mapper_runs.csv", index=False)
    runtime = {
        "seconds": time.perf_counter() - started,
        "stress_rows": len(frame),
        "expensive_instances": len(expensive_ids),
        "timeouts": int(frame.timeout.sum()),
        "resumed_completed_instances": len(tasks) - len(remaining_tasks),
    }
    (OUT / "mapper_runtime.json").write_text(
        json.dumps(runtime, indent=2) + "\n"
    )
    print(json.dumps(runtime, indent=2))
    print(frame.groupby(["budget", "method"]).success.agg(["size", "mean"]).to_string())


def lambda_args(args):
    return run_instance(*args)


if __name__ == "__main__":
    main()
