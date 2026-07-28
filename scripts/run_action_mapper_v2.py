#!/usr/bin/env python3
"""Integrate frozen action scorers into the beam mapper for revision-v2."""

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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quotientflow.action_lookahead import (
    FEATURE_NAMES,
    action_cache_key,
    action_features,
    select_candidate_actions,
)
from quotientflow.action_model import load_action_model, model_scores
from quotientflow.dfg import FAMILIES
from quotientflow.mapper import Action, map_dfg
from quotientflow.model import QuotientFlowGNN, predict_numpy
from quotientflow.relaxation import solve_relaxation
from scripts.run_revision2_scarcity_suite import adjust as revision1_adjust


OUTPUT = ROOT / "results" / "revision_v2"
RAW = OUTPUT / "raw"
RELAXATION_CONFIG = {
    "tau": 0.001,
    "solver_primary": "CLARABEL",
    "solver_fallback": "OSQP",
    "max_seconds": 8.0,
}
TEST_SPLITS = (
    "test_seen",
    "test_unseen_size",
    "test_unseen_topology",
    "test_unseen_asym",
)


def load_mapping_samples():
    cache = ROOT / "results" / "cache" / "samples"
    samples = []
    for split in ("val_seen", *TEST_SPLITS):
        for family in FAMILIES:
            count = 1 if split == "val_seen" else 2
            for path in sorted(cache.glob(f"{split}__*__{family}__*.pkl"))[:count]:
                with path.open("rb") as handle:
                    samples.append(pickle.load(handle))
        if split == "val_seen":
            # Add the other seen architecture for the seven families.
            existing = {(sample["architecture"], sample["family"]) for sample in samples}
            for family in FAMILIES:
                for path in sorted(cache.glob(f"{split}__*__{family}__*.pkl")):
                    with path.open("rb") as handle:
                        candidate = pickle.load(handle)
                    key = (candidate["architecture"], family)
                    if key not in existing:
                        samples.append(candidate)
                        existing.add(key)
                        break
    return samples


class LearnedScorer:
    def __init__(
        self,
        model,
        model_type,
        scaler,
        prices,
        feature_cache=None,
        preselection_cache=None,
    ):
        self.model = model
        self.model_type = model_type
        self.scaler = scaler
        self.prices = np.asarray(prices)
        self.feature_cache = feature_cache if feature_cache is not None else {}
        self.preselection_cache = (
            preselection_cache if preselection_cache is not None else {}
        )

    def preselect(self, dfg, arch, state, operation, actions):
        key = (
            state.serialize(),
            operation,
            tuple((action.target, action.new_links) for action in actions),
        )
        signatures = self.preselection_cache.get(key)
        if signatures is None:
            selected = select_candidate_actions(
                dfg, arch, state, operation, actions, self.prices, limit=8
            )
            signatures = tuple(
                (action.target, action.new_links) for action in selected
            )
            self.preselection_cache[key] = signatures
            return selected
        lookup = {(action.target, action.new_links): action for action in actions}
        return [lookup[signature] for signature in signatures if signature in lookup]

    def __call__(self, dfg, arch, state, operation, actions):
        features = []
        for action in actions:
            key = (state.serialize(), operation, action.target, action.new_links)
            if key not in self.feature_cache:
                self.feature_cache[key] = action_features(
                    dfg, arch, state, operation, action, self.prices
                )
            features.append(
                [self.feature_cache[key][name] for name in FEATURE_NAMES]
            )
        group = {
            "features": np.asarray(features, dtype=np.float32),
            "operation": operation,
            "targets": [action.target for action in actions],
            "new_links": [action.new_links for action in actions],
        }
        sample = {"dfg": dfg, "arch": arch, "state": state}
        self.model.eval()
        with torch.no_grad():
            return (
                model_scores(
                    self.model, self.model_type, sample, group, self.scaler
                )
                .detach()
                .cpu()
                .numpy()
            )


class OracleScorer:
    def __init__(self, prices):
        self.prices = np.asarray(prices)
        self.cache = OUTPUT / "cache" / "mapper_oracle"
        self.cache.mkdir(parents=True, exist_ok=True)
        self.solves = 0
        self.cache_hits = 0

    def preselect(self, dfg, arch, state, operation, actions):
        return select_candidate_actions(
            dfg, arch, state, operation, actions, self.prices, limit=8
        )

    def __call__(self, dfg, arch, state, operation, actions):
        values = []
        finite = []
        for action in actions:
            key = action_cache_key(
                dfg, arch, state, operation, action, RELAXATION_CONFIG
            )
            path = self.cache / f"{key}.pkl"
            if path.exists():
                with path.open("rb") as handle:
                    result = pickle.load(handle)
                self.cache_hits += 1
            else:
                result = solve_relaxation(
                    dfg, arch, action.state, **RELAXATION_CONFIG
                )
                with path.open("wb") as handle:
                    pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
                self.solves += 1
            value = (
                float(action.base_cost + result.objective)
                if result.solved
                else float("nan")
            )
            values.append(value)
            if math.isfinite(value):
                finite.append(value)
        if finite:
            margin = max(1.0, 0.25 * (max(finite) - min(finite)))
            penalty = max(finite) + margin
        else:
            penalty = 1e3
        return np.asarray(
            [value if math.isfinite(value) else penalty for value in values]
        )


def run_learned(
    sample,
    method,
    model,
    scaler,
    model_type,
    feature_cache,
    preselection_cache,
):
    scorer = LearnedScorer(
        model,
        model_type,
        scaler,
        sample["normalized_prices"],
        feature_cache,
        preselection_cache,
    )
    result = map_dfg(
        sample["dfg"],
        sample["arch"],
        sample["state"],
        beam_width=4,
        action_limit=6,
        k_paths=4,
        max_expansions=5000,
        timeout_seconds=120.0,
        action_scorer=scorer,
        action_preselector=scorer.preselect,
        action_value_is_cost_to_go=True,
    )
    return {
        "sample_id": sample["sample_id"],
        "split": sample["split"],
        "architecture": sample["architecture"],
        "dfg_family": sample["family"],
        "method": method,
        "source": "revision_v2_cost_to_go_frozen_model",
        **result.row(),
    }


def run_oracle(sample, timeout_seconds=120.0):
    scorer = OracleScorer(sample["normalized_prices"])
    result = map_dfg(
        sample["dfg"],
        sample["arch"],
        sample["state"],
        beam_width=4,
        action_limit=6,
        k_paths=4,
        max_expansions=5000,
        timeout_seconds=timeout_seconds,
        action_scorer=scorer,
        action_preselector=scorer.preselect,
        action_value_is_cost_to_go=True,
    )
    row = {
        "sample_id": sample["sample_id"],
        "split": sample["split"],
        "architecture": sample["architecture"],
        "dfg_family": sample["family"],
        "method": "oracle_action_lookahead",
        "source": "revision_v2_cost_to_go_oracle",
        "oracle_new_solves": scorer.solves,
        "oracle_cache_hits": scorer.cache_hits,
        **result.row(),
    }
    return row


def reused_test_baselines():
    old = pd.read_csv(
        ROOT / "results" / "revision_v1" / "mappings_scarcity.csv"
    )
    rename = {
        "length": "length",
        "oracle_dual": "heuristic_scarcity",
        "pred_dual": "pred_link_scarcity",
    }
    frame = old[old["method"].isin(rename)].copy()
    frame["method"] = frame["method"].map(rename)
    frame["source"] = "reused_revision_v1_frozen"
    return frame.to_dict("records")


def main():
    start = time.perf_counter()
    selection = json.loads((OUTPUT / "selected_action_model.json").read_text())
    training = pd.read_csv(OUTPUT / "training_summary.csv")
    best = {}
    for model_type in ("action_mlp", "action_gnn"):
        row = (
            training[training["model_type"] == model_type]
            .sort_values(
                [
                    "validation_best_action_preservation",
                    "validation_selected_regret",
                    "validation_pairwise_accuracy",
                ],
                ascending=[False, True, False],
            )
            .iloc[0]
        )
        best[model_type] = int(row["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = {}
    for model_type, seed in best.items():
        model, scaler, _, _ = load_action_model(
            OUTPUT / "checkpoints" / f"{model_type}_seed_{seed}.pt", device
        )
        models[model_type] = (model, scaler)

    samples = load_mapping_samples()
    output_path = RAW / "action_mapper_runs.csv"
    rows = reused_test_baselines()
    if output_path.exists():
        existing = pd.read_csv(output_path)
        existing = existing[
            existing["source"].isin(
                [
                    "revision_v2_cost_to_go_frozen_model",
                    "revision_v2_cost_to_go_oracle",
                    "revision_v2_cost_to_go_oracle_uncensored_600s",
                ]
            )
        ]
        rows.extend(existing.to_dict("records"))
    completed = {
        (row["sample_id"], row["method"])
        for row in rows
    }
    validation_keep = {
        next(
            sample["sample_id"]
            for sample in samples
            if sample["split"] == "val_seen" and sample["family"] == family
        )
        for family in FAMILIES
    }
    for sample in samples:
        if sample["split"] == "val_seen" and sample["sample_id"] not in validation_keep:
            continue
        shared_features = {}
        shared_preselection = {}
        for model_type, method in (
            ("action_mlp", "mlp_action_lookahead"),
            ("action_gnn", "gnn_action_lookahead"),
        ):
            if (sample["sample_id"], method) in completed:
                continue
            model, scaler = models[model_type]
            rows.append(
                run_learned(
                    sample,
                    method,
                    model,
                    scaler,
                    model_type,
                    shared_features,
                    shared_preselection,
                )
            )
            pd.DataFrame(rows).to_csv(output_path, index=False)
            completed.add((sample["sample_id"], method))
    # Seven deterministic instances per split: one per DFG family.
    oracle_ids = set()
    for split in ("val_seen", *TEST_SPLITS):
        subset = [sample for sample in samples if sample["split"] == split]
        for family in FAMILIES:
            oracle_ids.add(
                next(
                    sample["sample_id"]
                    for sample in subset
                    if sample["family"] == family
                )
            )
    for sample in samples:
        if sample["sample_id"] in oracle_ids:
            if (
                sample["sample_id"],
                "oracle_action_lookahead",
            ) in completed:
                continue
            rows.append(run_oracle(sample))
            pd.DataFrame(rows).to_csv(output_path, index=False)
            completed.add(
                (sample["sample_id"], "oracle_action_lookahead")
            )
    frame = pd.DataFrame(rows)
    predictions = pd.read_csv(RAW / "action_predictions.csv")
    initial_regret = (
        predictions[predictions["selected_action"]]
        .groupby(["state_id", "model", "model_seed"], as_index=False)[
            "selected_action_normalized_regret"
        ]
        .first()
    )
    method_lookup = {
        "length": ("length", -1),
        "heuristic_scarcity": ("heuristic_scarcity", -1),
        "pred_link_scarcity": ("predicted_link_price", -1),
        "mlp_action_lookahead": ("action_mlp", best["action_mlp"]),
        "gnn_action_lookahead": ("action_gnn", best["action_gnn"]),
        "oracle_action_lookahead": ("oracle_q_rel", -1),
    }
    regret_map = {
        (row.state_id, row.model, row.model_seed): row.selected_action_normalized_regret
        for row in initial_regret.itertuples()
    }
    frame["mean_selected_action_oracle_regret_available_state"] = [
        regret_map.get(
            (row.sample_id, *method_lookup[row.method]), float("nan")
        )
        for row in frame.itertuples()
    ]
    frame.to_csv(output_path, index=False)
    runtime = {
        "action_mapping_seconds": time.perf_counter() - start,
        "selected_overall_model": selection["model_type"],
        "selected_overall_seed": int(selection["seed"]),
        "selected_seed_by_type": best,
        "rows": len(frame),
    }
    (OUTPUT / "mapper_runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
    print(json.dumps(runtime, indent=2))
    print(
        frame.groupby(["split", "method"]).agg(
            instances=("sample_id", "size"),
            success=("success", "mean"),
            failed_routes=("failed_routing_attempts", "mean"),
            runtime_ms=("runtime_ms", "mean"),
        ).to_string()
    )


if __name__ == "__main__":
    main()
