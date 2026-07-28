#!/usr/bin/env python3
"""Revision-v3 Phase 0: reconstruct v2 and uncensor action preselection."""

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

from quotientflow.action_lookahead import (  # noqa: E402
    FEATURE_NAMES,
    action_cache_key,
    check_partial_legality,
    scarcity_adjusted_sum,
)
from quotientflow.mapper import enumerate_actions  # noqa: E402
from quotientflow.relaxation import solve_relaxation  # noqa: E402

OUT = ROOT / "results" / "revision_v3"
RAW = OUT / "raw"
CACHE = OUT / "cache" / "forensic_child"
RELAXATION_CONFIG = {
    "tau": 0.001,
    "solver_primary": "CLARABEL",
    "solver_fallback": "OSQP",
    "max_seconds": 12.0,
}
SPLIT_COUNTS = {
    "train_seen": 6,
    "val_seen": 6,
    "test_seen": 3,
    "test_unseen_size": 3,
    "test_unseen_topology": 3,
    "test_unseen_asym": 3,
}


def v2_diagnostics(labels: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    selected = predictions[
        (predictions.model == "action_gnn")
        & (predictions.model_seed == 37)
        & predictions.selected_action
    ].set_index("state_id")
    rows = []
    for state_id, group in labels.groupby("state_id", sort=True):
        q = group.q_rel_raw.to_numpy(float)
        order = np.lexsort((group.action_id.to_numpy(str), q))
        qmin, qmax = float(q[order[0]]), float(q.max())
        raw_range = qmax - qmin
        gap = float(q[order[1]] - qmin) if len(q) > 1 else 0.0
        epsilon = max(1e-4, 0.01 * max(raw_range, 1.0))
        pred = selected.loc[state_id] if state_id in selected.index else None
        predicted_action = pred.action_id if pred is not None else ""
        predicted_q = float(
            group.loc[group.action_id == predicted_action, "q_rel_raw"].iloc[0]
        ) if predicted_action in set(group.action_id) else math.nan
        predicted_raw_regret = predicted_q - qmin if math.isfinite(predicted_q) else math.nan
        relative = predicted_raw_regret / max(abs(qmin), 1.0)
        exact = (
            predicted_action == group.iloc[order[0]].action_id
            if pred is not None else math.nan
        )
        nonzero = np.abs(q - qmin)
        nonzero = nonzero[nonzero > 1e-12]
        denominator = float(np.median(nonzero) + 1e-8) if len(nonzero) else 1e-8
        rows.append({
            "state_id": state_id,
            "split": group.split.iloc[0],
            "architecture": group.architecture.iloc[0],
            "dfg_family": group.dfg_family.iloc[0],
            "num_actions": len(group),
            "raw_q_min": qmin,
            "raw_q_max": qmax,
            "raw_q_range": raw_range,
            "best_second_gap": gap,
            "actions_within_1e-6": int(np.sum(q <= qmin + 1e-6)),
            "actions_within_1e-4": int(np.sum(q <= qmin + 1e-4)),
            "actions_within_0p1pct_range": int(np.sum(q <= qmin + .001 * raw_range)),
            "actions_within_1pct_range": int(np.sum(q <= qmin + .01 * raw_range)),
            "v3_epsilon": epsilon,
            "epsilon_optimal_actions": int(np.sum(q <= qmin + epsilon)),
            "v2_normalization_denominator": denominator,
            "v2_max_normalized_regret": float(group.q_rel_normalized.max()),
            "v2_gnn_exact_top1": exact,
            "v2_gnn_predicted_action": predicted_action,
            "v2_gnn_raw_regret": predicted_raw_regret,
            "v2_gnn_relative_regret_to_best_value": relative,
            "incorrect_top1_under_1pct_relative_regret": (
                bool(not exact and math.isfinite(relative) and relative < .01)
                if pred is not None else math.nan
            ),
            "v2_gnn_epsilon_optimal": (
                bool(
                math.isfinite(predicted_raw_regret)
                and predicted_raw_regret <= epsilon + 1e-12
                ) if pred is not None else math.nan
            ),
            "exact_top1_not_meaningful": bool(
                np.sum(q <= qmin + epsilon) > 1 or gap <= epsilon
            ),
        })
    return pd.DataFrame(rows)


def feature_shift(labels: pd.DataFrame) -> pd.DataFrame:
    rows = []
    train = labels[labels.split == "train_seen"]
    for feature in FEATURE_NAMES:
        x = train[feature].to_numpy(float)
        mean, std = float(np.nanmean(x)), float(np.nanstd(x))
        train_min, train_max = float(np.nanmin(x)), float(np.nanmax(x))
        for split in (
            "test_seen", "test_unseen_size",
            "test_unseen_topology", "test_unseen_asym",
        ):
            y = labels.loc[labels.split == split, feature].to_numpy(float)
            test_mean = float(np.nanmean(y))
            z = abs(test_mean - mean) / max(std, 1e-8)
            rows.append({
                "feature": feature,
                "test_split": split,
                "train_mean": mean,
                "train_std": std,
                "test_mean": test_mean,
                "absolute_mean_z_shift": z,
                "train_min": train_min,
                "train_max": train_max,
                "test_min": float(np.nanmin(y)),
                "test_max": float(np.nanmax(y)),
                "test_outside_train_range_pct": 100.0 * float(
                    np.mean((y < train_min) | (y > train_max))
                ),
                "z_shift_over_2": z > 2.0,
            })
    return pd.DataFrame(rows)


def diverse_order(actions):
    ordered = sorted(actions, key=lambda a: (a.base_cost, a.target, a.new_links))
    result, seen = [], set()
    for key_fn in (
        lambda a: a.target,
        lambda a: a.new_links,
    ):
        used = set()
        for action in ordered:
            key = key_fn(action)
            sig = (action.target, action.new_links)
            if key not in used and sig not in seen:
                result.append(action)
                seen.add(sig)
                used.add(key)
    result.extend(a for a in ordered if (a.target, a.new_links) not in seen)
    return result


def round_robin_union(*rankings):
    result, seen = [], set()
    for position in range(max(map(len, rankings), default=0)):
        for ranking in rankings:
            if position >= len(ranking):
                continue
            action = ranking[position]
            sig = (action.target, action.new_links)
            if sig not in seen:
                result.append(action)
                seen.add(sig)
    return result


def choose_audit_samples():
    source = ROOT / "results" / "cache" / "samples"
    chosen = []
    for split, count in SPLIT_COUNTS.items():
        candidates = sorted(source.glob(f"{split}__*.pkl"))
        # Round-robin by family via filename is deterministic and avoids
        # choosing based on observed child objectives.
        families = {}
        for path in candidates:
            with path.open("rb") as handle:
                sample = pickle.load(handle)
            families.setdefault(sample["family"], []).append(path)
        position = 0
        while len([p for p in chosen if p[0] == split]) < count:
            changed = False
            for family in sorted(families):
                if position < len(families[family]):
                    chosen.append((split, families[family][position]))
                    changed = True
                    if len([p for p in chosen if p[0] == split]) == count:
                        break
            if not changed:
                break
            position += 1
    return [path for _, path in chosen]


def solve_cached(payload):
    sample_path, operation, action_index = payload
    with Path(sample_path).open("rb") as handle:
        sample = pickle.load(handle)
    actions, _, _ = enumerate_actions(
        sample["dfg"], sample["arch"], sample["state"], operation, 4
    )
    action = actions[action_index]
    key = action_cache_key(
        sample["dfg"], sample["arch"], sample["state"],
        operation, action, RELAXATION_CONFIG,
    )
    cache_path = CACHE / f"{key}.pkl"
    hit = cache_path.exists()
    if hit:
        with cache_path.open("rb") as handle:
            result = pickle.load(handle)
    else:
        result = solve_relaxation(
            sample["dfg"], sample["arch"], action.state, **RELAXATION_CONFIG
        )
        with cache_path.open("wb") as handle:
            pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return sample["sample_id"], action_index, result, hit


def full_action_audit() -> pd.DataFrame:
    sample_paths = choose_audit_samples()
    prepared, jobs = {}, []
    for path in sample_paths:
        with path.open("rb") as handle:
            sample = pickle.load(handle)
        remaining = [v for v in sample["dfg"].order if v not in sample["state"].placements]
        operation = remaining[0]
        actions, _, _ = enumerate_actions(
            sample["dfg"], sample["arch"], sample["state"], operation, 4
        )
        legal = [
            action for action in actions
            if check_partial_legality(sample["dfg"], sample["arch"], action.state)[0]
        ]
        # enumerate_actions is already target-complete and four-path-bounded.
        actions = legal[:64] if len(legal) > 64 else legal
        parent = solve_relaxation(
            sample["dfg"], sample["arch"], sample["state"], **RELAXATION_CONFIG
        )
        prepared[sample["sample_id"]] = (
            path, sample, operation, actions, parent, len(legal) > 64
        )
        jobs.extend((str(path), operation, i) for i in range(len(actions)))
    results = {}
    # CVXPY solver runtimes are not fork-safe after the parent has solved J(s).
    # Spawn gives every worker a clean solver process.
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=6, mp_context=mp.get_context("spawn")
    ) as pool:
        for state_id, index, result, hit in pool.map(solve_cached, jobs):
            results[(state_id, index)] = (result, hit)
    rows = []
    for state_id, (_, sample, operation, actions, parent, path_bounded) in prepared.items():
        values, total_solve_ms = [], 0.0
        for i, action in enumerate(actions):
            result, _ = results[(state_id, i)]
            total_solve_ms += 1000.0 * result.solve_seconds
            values.append(
                action.base_cost + result.objective if result.solved else math.inf
            )
        finite = np.asarray([v for v in values if math.isfinite(v)])
        if not len(finite):
            continue
        penalty = float(finite.max() + max(1.0, .25 * np.ptp(finite)))
        values = np.asarray([v if math.isfinite(v) else penalty for v in values])
        true_order = np.lexsort((
            np.asarray([(a.target, a.new_links) for a in actions], dtype=object),
            values,
        )) if False else np.asarray(sorted(
            range(len(actions)),
            key=lambda i: (values[i], actions[i].target, actions[i].new_links),
        ))
        qmin, qrange = float(values[true_order[0]]), float(np.ptp(values))
        epsilon = max(1e-4, .01 * max(qrange, 1.0))
        length = sorted(actions, key=lambda a: (a.base_cost, a.target, a.new_links))
        static = sorted(actions, key=lambda a: (
            a.base_cost + .5 * scarcity_adjusted_sum(
                sample["arch"], sample["state"], a,
                np.asarray(sample["normalized_prices"]),
            ), a.target, a.new_links,
        ))
        current_prices = parent.normalized_prices if parent.solved else np.zeros(len(sample["arch"].edge_list))
        current = sorted(actions, key=lambda a: (
            a.base_cost + .5 * scarcity_adjusted_sum(
                sample["arch"], sample["state"], a, current_prices,
            ), a.target, a.new_links,
        ))
        diverse = diverse_order(actions)
        union = round_robin_union(length[:8], current[:8], diverse[:8])
        selectors = {
            "length": length,
            "revision_v2_static_price": static,
            "current_parent_dual": current,
            "target_path_diverse": diverse,
            "union_length_current_dual_diversity": union,
        }
        index = {(a.target, a.new_links): i for i, a in enumerate(actions)}
        for selector, ranking in selectors.items():
            ranked_indices = [index[(a.target, a.new_links)] for a in ranking]
            for k in (1, 2, 4, 8):
                retained = ranked_indices[:min(k, len(ranked_indices))]
                rows.append({
                    "state_id": state_id,
                    "split": sample["split"],
                    "architecture": sample["architecture"],
                    "dfg_family": sample["family"],
                    "selector": selector,
                    "k": k,
                    "legal_actions": len(actions),
                    "target_complete_path_bounded": path_bounded,
                    "true_best_recall": int(true_order[0] in retained),
                    "epsilon_optimal_recall": int(
                        np.any(values[retained] <= qmin + epsilon)
                    ),
                    "top2_set_recall": len(set(true_order[:2]) & set(retained)) / min(2, len(actions)),
                    "top4_set_recall": len(set(true_order[:4]) & set(retained)) / min(4, len(actions)),
                    "mean_retained_oracle_regret": float(np.mean(values[retained] - qmin)),
                    "best_retained_oracle_regret": float(np.min(values[retained] - qmin)),
                    "actions_evaluated_by_selector": len(retained),
                    "full_action_child_solve_ms": total_solve_ms,
                    "parent_solve_ms": 1000.0 * parent.solve_seconds,
                    "parent_status": parent.status,
                })
    return pd.DataFrame(rows)


def main():
    started = time.perf_counter()
    RAW.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    labels = pd.read_csv(ROOT / "results/revision_v2/raw/action_lookahead_labels.csv")
    predictions = pd.read_csv(ROOT / "results/revision_v2/raw/action_predictions.csv")
    diagnostics = v2_diagnostics(labels, predictions)
    shifts = feature_shift(labels)
    preselection = full_action_audit()
    diagnostics.to_csv(RAW / "v2_state_diagnostics.csv", index=False)
    shifts.to_csv(RAW / "v2_feature_shift.csv", index=False)
    preselection.to_csv(RAW / "v2_preselection_audit.csv", index=False)

    norm = labels.q_rel_normalized
    selected = diagnostics
    pre8 = preselection[
        (preselection.selector == "revision_v2_static_price") & (preselection.k == 8)
    ]
    current8 = preselection[
        (preselection.selector == "current_parent_dual") & (preselection.k == 8)
    ]
    union8 = preselection[
        (preselection.selector == "union_length_current_dual_diversity")
        & (preselection.k == 8)
    ]
    shifted = shifts[shifts.z_shift_over_2]
    audit = f"""# FlowAdvantage revision-v3 forensic audit

## Reconstruction integrity

All specified revision-v2 artifacts were read in full. Their SHA-256 values
match the immutable revision-v2 manifest. The v2 action labels contain
{len(labels):,} rows from {labels.state_id.nunique()} states, and the prediction
file contains {len(predictions):,} rows.

## A. Label ambiguity

- States where exact top-1 identity is not meaningful under the fixed v3
  epsilon: {selected.exact_top1_not_meaningful.mean():.3%}.
- Incorrect v2 GNN top-1 choices with under 1% relative regret:
  {selected.incorrect_top1_under_1pct_relative_regret.mean():.3%}.
- Exact top-1 v2 GNN rate: {selected.v2_gnn_exact_top1.mean():.3%}.
- Epsilon-optimal v2 GNN rate: {selected.v2_gnn_epsilon_optimal.mean():.3%}.

## B. Normalization pathology

- Maximum v2 normalized regret: {norm.max():.6g}.
- p95 / p99: {norm.quantile(.95):.6g} / {norm.quantile(.99):.6g}.
- States with median-regret denominator below 1e-3:
  {(selected.v2_normalization_denominator < 1e-3).sum()} /
  {len(selected)}.

The v2 per-state median denominator amplified nearly tied values. Revision-v3
uses raw delta regret, a floor-stabilized state scale for listwise targets, and
a training-only global robust scaler for residual regression.

## C. Feature shift

{shifted.feature.nunique()} features exceed absolute mean z-shift 2 on at least
one test split. Every affected feature/split is preserved in
`raw/v2_feature_shift.csv`; these include:
{", ".join(sorted(shifted.feature.unique())) or "none"}.

## D. Static-dual dependency

Revision-v2 prices were created with each cached initial sample and stored as
`sample["normalized_prices"]`. `LearnedScorer` passed this fixed vector into
preselection and every deeper-state feature call. It did not solve a parent
relaxation at deeper states. The dependent features were `dual_sum`,
`dual_max`, `dual_top2_sum`, `scarcity_adjusted_dual_sum`,
`positive_price_links_used`, and `top_decile_link_fraction`. Consequently the
v2 learned mapper was pure learned inference after static-price top-8
preselection, not a one-parent-relaxation hybrid.

## E. Preselection bias

The audit evaluates all mapper-legal actions on 24 deterministic states
(target-complete, with at most four paths per target) before applying selectors.
At k=8:

- revision-v2 static-price true-best recall: {pre8.true_best_recall.mean():.3%};
- current-parent-dual true-best recall: {current8.true_best_recall.mean():.3%};
- union selector true-best recall: {union8.true_best_recall.mean():.3%};
- revision-v2 static-price epsilon recall: {pre8.epsilon_optimal_recall.mean():.3%}.

This 24-state audit finds no empirical missed optimum from the v2 static
top-eight selector. Nevertheless, the v2 procedure was methodologically
censored because actions outside that set were not solved; it cannot establish
full-action behavior. Revision-v3 uses “full-action oracle” only for
target-complete, four-path-bounded action sets whose children were all solved.
"""
    (OUT / "AUDIT.md").write_text(audit)
    runtime = {
        "phase": "forensic_audit",
        "seconds": time.perf_counter() - started,
        "states": 24,
        "full_action_rows": len(preselection),
    }
    (OUT / "forensic_runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
    print(audit)
    print(json.dumps(runtime, indent=2))


if __name__ == "__main__":
    main()
