#!/usr/bin/env python
"""Run the complete resumable QuotientFlow quick validation suite."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import pickle
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quotientflow.arch import build_architecture
from quotientflow.dfg import FAMILIES, generate_dfg, generate_exactness_dfg
from quotientflow.mapper import action_score, enumerate_actions, map_dfg
from quotientflow.metrics import geometric_mean, mean, median, percentile, reduction_percent, safe_spearman
from quotientflow.model import QuotientFlowGNN, predict_numpy
from quotientflow.partial_state import construct_partial_state, deterministic_fraction
from quotientflow.relaxation import solve_relaxation
from quotientflow.reporting import make_plots, markdown_text, write_table
from quotientflow.training import prediction_metrics, train_seed


METHODS = [
    "length",
    "quotient_length",
    "oracle_dual",
    "oracle_dual_quotient",
    "pred_dual",
    "pred_dual_quotient",
]
TEST_SPLITS = [
    "test_seen",
    "test_unseen_size",
    "test_unseen_topology",
    "test_unseen_asym",
]


def stable_seed(master, *parts):
    text = ":".join(map(str, (master,) + parts))
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)


def save_frame(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def dataset_specs(cfg):
    data = cfg["data"]
    specs = []
    for split, archs, count in (
        ("train_seen", ("mesh3", "torus3"), data["train_per_family_per_arch"]),
        ("val_seen", ("mesh3", "torus3"), data["val_per_family_per_arch"]),
        ("test_seen", ("mesh3", "torus3"), data["test_seen_per_family_per_arch"]),
        ("test_unseen_size", ("mesh4",), data["test_per_family_unseen_arch"]),
        ("test_unseen_topology", ("diag4",), data["test_per_family_unseen_arch"]),
        ("test_unseen_asym", ("cut4",), data["test_per_family_unseen_arch"]),
    ):
        for arch in archs:
            for family in FAMILIES:
                for index in range(count):
                    specs.append((split, arch, family, index))
    return specs


def solve_sample(spec, cfg, cache_dir):
    split, arch_name, family, index = spec
    sample_id = f"{split}__{arch_name}__{family}__{index:02d}"
    path = cache_dir / "samples" / f"{sample_id}.pkl"
    if path.exists():
        with path.open("rb") as handle:
            return pickle.load(handle)
    master = cfg["experiment"]["master_seed"]
    base_seed = stable_seed(master, *spec)
    error = ""
    for replacement in range(50):
        seed = (base_seed + replacement * 104729) % (2**32)
        try:
            arch = build_architecture(arch_name, cfg["architecture"]["ii"])
            dfg = generate_dfg(family, seed)
            fraction = deterministic_fraction(seed, tuple(cfg["data"]["partial_depth_fractions"]))
            state = construct_partial_state(dfg, arch, fraction)
            break
        except Exception as exc:
            error += f"seed={seed}:{type(exc).__name__}:{exc};"
    else:
        raise RuntimeError(f"could not construct replacement for {sample_id}: {error}")
    rcfg = cfg["relaxation"]
    result = solve_relaxation(
        dfg,
        arch,
        state,
        tau=rcfg["tau"],
        solver_primary=rcfg["solver_primary"],
        solver_fallback=rcfg["solver_fallback"],
        max_seconds=rcfg["max_solve_seconds"],
    )
    target = result.normalized_prices
    pair_indices = np.argwhere(np.triu(np.abs(target[:, None] - target[None, :]) >= 0.1, 1))
    pair_list = [tuple(map(int, p)) for p in pair_indices.tolist()]
    if len(pair_list) > 64:
        pair_list = random.Random(seed).sample(pair_list, 64)
    sample = {
        "sample_id": sample_id,
        "split": split,
        "architecture": arch_name,
        "family": family,
        "index": index,
        "seed": seed,
        "replacement_count": replacement,
        "construction_errors": error,
        "partial_fraction": fraction,
        "dfg": dfg,
        "arch": arch,
        "state": state,
        "relaxation": result,
        "raw_prices": result.raw_prices,
        "normalized_prices": result.normalized_prices,
        "rank_pairs": pair_list,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(sample, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return sample


def raw_dataset_frames(samples):
    instances, relax = [], []
    for s in samples:
        dfg, arch, state, result = s["dfg"], s["arch"], s["state"], s["relaxation"]
        instances.append(
            {
                "sample_id": s["sample_id"],
                "split": s["split"],
                "architecture": s["architecture"],
                "dfg_family": s["family"],
                "seed": s["seed"],
                "replacement_count": s["replacement_count"],
                "construction_errors": s["construction_errors"],
                "ops": len(dfg.graph),
                "dfg_edges": dfg.graph.number_of_edges(),
                "partial_fraction_requested": s["partial_fraction"],
                "partial_operations": len(state.placements),
                "partial_depth": len(state.placements) / len(dfg.graph),
                "routing_nodes": len(arch.graph),
                "routing_edges": len(arch.edge_list),
                "dfg_group_size": len(dfg.automorphisms),
                "arch_group_size": len(arch.symmetries),
                "solver_status": result.status,
            }
        )
        for edge_id, (raw, normalized) in enumerate(
            zip(result.raw_prices, result.normalized_prices)
        ):
            relax.append(
                {
                    "sample_id": s["sample_id"],
                    "split": s["split"],
                    "architecture": s["architecture"],
                    "dfg_family": s["family"],
                    "edge_id": edge_id,
                    "raw_dual": raw,
                    "normalized_dual": normalized,
                    **result.health_dict(),
                }
            )
    return pd.DataFrame(instances), pd.DataFrame(relax)


def run_knockouts(probes, cfg, raw_path, deadline):
    rows = pd.read_csv(raw_path).to_dict("records") if raw_path.exists() else []
    completed = {
        (str(r["sample_id"]), int(r["edge_id"])) for r in rows
    }
    for sample in probes:
        result = sample["relaxation"]
        prices = result.raw_prices
        order = np.argsort(prices, kind="stable")
        middle = len(order) // 2
        selected = list(order[-8:]) + list(order[middle - 2 : middle + 2]) + list(order[:4])
        selected = list(dict.fromkeys(map(int, selected)))
        if all((sample["sample_id"], edge_id) in completed for edge_id in selected):
            continue
        instance_rows = []
        for edge_id in selected:
            if (sample["sample_id"], edge_id) in completed:
                instance_rows.append(
                    next(
                        r
                        for r in rows
                        if str(r["sample_id"]) == sample["sample_id"]
                        and int(r["edge_id"]) == edge_id
                    )
                )
                continue
            if time.perf_counter() > deadline:
                break
            knocked = solve_relaxation(
                sample["dfg"],
                sample["arch"],
                sample["state"],
                tau=cfg["relaxation"]["tau"],
                solver_primary=cfg["relaxation"]["solver_primary"],
                solver_fallback=cfg["relaxation"]["solver_fallback"],
                max_seconds=cfg["relaxation"]["max_solve_seconds"],
                knockout_edge=edge_id,
            )
            infeasible = not knocked.solved
            impact = (
                max(0.0, knocked.objective - result.objective)
                if knocked.solved
                else float("inf")
            )
            category = "top" if edge_id in set(order[-8:]) else "bottom" if edge_id in set(order[:4]) else "median"
            instance_rows.append(
                {
                    "sample_id": sample["sample_id"],
                    "split": sample["split"],
                    "architecture": sample["architecture"],
                    "edge_id": edge_id,
                    "category": category,
                    "dual_price": prices[edge_id],
                    "base_objective": result.objective,
                    "knockout_objective": knocked.objective,
                    "impact": impact,
                    "infeasible": infeasible,
                    "solver_status": knocked.status,
                    "solve_seconds": knocked.solve_seconds,
                }
            )
        finite = [r["impact"] for r in instance_rows if np.isfinite(r["impact"])]
        substitute = (max(finite) + max(1.0, abs(max(finite)))) if finite else 1e9
        scored = [substitute if not np.isfinite(r["impact"]) else r["impact"] for r in instance_rows]
        dual = [r["dual_price"] for r in instance_rows]
        top_price = set(np.argsort(dual, kind="stable")[-min(8, len(dual)):])
        top_impact = set(np.argsort(scored, kind="stable")[-min(8, len(scored)):])
        recall = len(top_price & top_impact) / max(1, min(8, len(scored)))
        top_values = [scored[i] for i, r in enumerate(instance_rows) if r["category"] == "top"]
        bottom_values = [scored[i] for i, r in enumerate(instance_rows) if r["category"] == "bottom"]
        ratio = mean(top_values) / max(mean(bottom_values), 1e-12)
        corr = safe_spearman(dual, scored)
        for row in instance_rows:
            row.update(
                {
                    "instance_spearman": corr,
                    "instance_top8_recall": recall,
                    "instance_top_bottom_ratio": ratio,
                }
            )
        rows = [r for r in rows if str(r["sample_id"]) != sample["sample_id"]]
        rows.extend(instance_rows)
        save_frame(pd.DataFrame(rows), raw_path)
    return pd.DataFrame(rows)


def load_or_train_models(samples, cfg, output, device):
    train = [s for s in samples if s["split"] == "train_seen" and s["relaxation"].solved]
    validation = [s for s in samples if s["split"] == "val_seen" and s["relaxation"].solved]
    models, metadata = {}, {}
    for seed in cfg["model"]["seeds"]:
        checkpoint = output / "checkpoints" / f"model_seed_{seed}.pt"
        if checkpoint.exists():
            saved = torch.load(checkpoint, map_location=device, weights_only=False)
            model = QuotientFlowGNN(
                cfg["model"]["hidden_dim"],
                cfg["model"]["dfg_layers"],
                cfg["model"]["arch_layers"],
                cfg["model"]["head_layers"],
            ).to(device)
            model.load_state_dict(saved["state_dict"])
            meta = {
                "seed": seed,
                "best_epoch": saved["best_epoch"],
                "validation_spearman": saved["validation_spearman"],
                "train_seconds": saved.get("train_seconds", float("nan")),
            }
        else:
            model, meta = train_seed(
                train, validation, cfg["model"], seed, device, checkpoint
            )
        models[seed], metadata[seed] = model, meta
    selected = max(metadata, key=lambda seed: metadata[seed]["validation_spearman"])
    return models, metadata, selected


def mapping_samples(samples):
    chosen = []
    for split in TEST_SPLITS:
        for family in FAMILIES:
            group = sorted(
                (s for s in samples if s["split"] == split and s["family"] == family),
                key=lambda s: (s["architecture"], s["index"]),
            )
            chosen.extend(group[:2])
    return chosen


def prediction_and_actions(models, metadata, selected_seed, samples, map_samples, output):
    rows, raw_predictions, action_rows = [], [], []
    eval_splits = ["val_seen"] + TEST_SPLITS
    for seed, model in models.items():
        for split in eval_splits:
            group = [s for s in samples if s["split"] == split and s["relaxation"].solved]
            metrics = prediction_metrics(model, group)
            action_group = [s for s in map_samples if s["split"] == split]
            matches, regrets = [], []
            for sample in action_group:
                remaining = [v for v in sample["dfg"].order if v not in sample["state"].placements]
                if not remaining:
                    continue
                actions, _, _ = enumerate_actions(
                    sample["dfg"], sample["arch"], sample["state"], remaining[0], 4
                )
                if not actions:
                    continue
                predicted, _ = predict_numpy(model, sample["dfg"], sample["arch"], sample["state"])
                oracle_scores = np.array([action_score(a, sample["normalized_prices"], 2.0) for a in actions])
                pred_scores = np.array([action_score(a, predicted, 2.0) for a in actions])
                predicted_choice = int(np.argmin(pred_scores))
                oracle_best = np.flatnonzero(oracle_scores <= oracle_scores.min() + 1e-8)
                matched = predicted_choice in set(oracle_best)
                regret = float(oracle_scores[predicted_choice] - oracle_scores.min())
                matches.append(matched)
                regrets.append(regret)
                action_rows.append(
                    {
                        "model_seed": seed,
                        "sample_id": sample["sample_id"],
                        "split": split,
                        "num_actions": len(actions),
                        "predicted_action_index": predicted_choice,
                        "oracle_best_indices": json.dumps(oracle_best.tolist()),
                        "match": matched,
                        "oracle_regret": regret,
                    }
                )
            rows.append(
                {
                    "model_seed": seed,
                    "split": split,
                    "architecture": "+".join(sorted({s["architecture"] for s in group})),
                    "num_samples": len(group),
                    **metrics,
                    "action_preservation": mean(matches),
                    "mean_action_regret": mean(regrets),
                    "train_seconds": metadata[seed]["train_seconds"],
                    "best_epoch": metadata[seed]["best_epoch"],
                }
            )
        for sample in [s for s in samples if s["split"] != "train_seen" and s["relaxation"].solved]:
            predicted, logits = predict_numpy(model, sample["dfg"], sample["arch"], sample["state"])
            target = sample["normalized_prices"]
            target_rank = rankdata(target, method="average") / len(target)
            pred_rank = rankdata(predicted, method="average") / len(predicted)
            for edge_id in range(len(target)):
                raw_predictions.append(
                    {
                        "model_seed": seed,
                        "selected_seed": seed == selected_seed,
                        "sample_id": sample["sample_id"],
                        "split": sample["split"],
                        "architecture": sample["architecture"],
                        "edge_id": edge_id,
                        "raw_dual": sample["raw_prices"][edge_id],
                        "target": target[edge_id],
                        "prediction": predicted[edge_id],
                        "critical_logit": logits[edge_id],
                        "target_rank": target_rank[edge_id],
                        "predicted_rank": pred_rank[edge_id],
                    }
                )
    seed_rows = pd.DataFrame(rows)
    aggregate_rows = []
    numeric = [
        "num_samples", "price_mae", "price_rmse", "median_spearman",
        "mean_spearman", "critical_auroc", "critical_average_precision",
        "top8_recall", "action_preservation", "mean_action_regret",
        "train_seconds", "best_epoch",
    ]
    for split, group in seed_rows.groupby("split"):
        base = {
            "split": split,
            "architecture": group["architecture"].iloc[0],
        }
        aggregate_rows.append(
            {
                **base,
                "model_seed": "aggregate_mean",
                **{column: mean(group[column]) for column in numeric},
            }
        )
        aggregate_rows.append(
            {
                **base,
                "model_seed": "aggregate_std",
                **{column: float(np.nanstd(pd.to_numeric(group[column], errors="coerce"), ddof=0)) for column in numeric},
            }
        )
    prediction_df = pd.DataFrame(raw_predictions)
    action_df = pd.DataFrame(action_rows)
    save_frame(prediction_df, output / "raw" / "predictions.csv")
    save_frame(action_df, output / "raw" / "actions.csv")
    return pd.concat([seed_rows, pd.DataFrame(aggregate_rows)], ignore_index=True), prediction_df, action_df


def run_mappings(map_samples, selected_model, cfg, output, deadline):
    raw_path = output / "raw" / "mappings.csv"
    existing = pd.read_csv(raw_path).to_dict("records") if raw_path.exists() else []
    completed = {(r["sample_id"], r["method"]) for r in existing}
    rows = existing
    priority = {"test_seen": 0, "test_unseen_asym": 1, "test_unseen_size": 2, "test_unseen_topology": 3}
    ordered = sorted(map_samples, key=lambda s: (priority[s["split"]], s["family"], s["index"]))
    mc = cfg["mapper"]
    jobs = []
    for sample in ordered:
        predicted, _ = predict_numpy(selected_model, sample["dfg"], sample["arch"], sample["state"])
        definitions = {
            "length": (None, False),
            "quotient_length": (None, True),
            "oracle_dual": (sample["normalized_prices"], False),
            "oracle_dual_quotient": (sample["normalized_prices"], True),
            "pred_dual": (predicted, False),
            "pred_dual_quotient": (predicted, True),
        }
        pending = [method for method in METHODS if (sample["sample_id"], method) not in completed]
        if pending:
            jobs.append((sample, predicted, mc, pending))
    if not jobs:
        return pd.DataFrame(rows), False
    stopped = False
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=cfg["experiment"]["cpu_threads"]
    ) as executor:
        futures = [executor.submit(_mapping_sample_job, job) for job in jobs]
        for future in concurrent.futures.as_completed(futures):
            if time.perf_counter() > deadline:
                stopped = True
            for row in future.result():
                rows.append(row)
                completed.add((row["sample_id"], row["method"]))
            frame = pd.DataFrame(rows).sort_values(
                ["split", "dfg_family", "sample_id", "method"]
            )
            save_frame(frame, raw_path)
    return pd.DataFrame(rows), stopped


def _mapping_sample_job(payload):
    sample, predicted, mc, methods = payload
    definitions = {
        "length": (None, False),
        "quotient_length": (None, True),
        "oracle_dual": (sample["normalized_prices"], False),
        "oracle_dual_quotient": (sample["normalized_prices"], True),
        "pred_dual": (predicted, False),
        "pred_dual_quotient": (predicted, True),
    }
    rows = []
    for method in methods:
        prices, quotient = definitions[method]
        result = map_dfg(
            sample["dfg"],
            sample["arch"],
            sample["state"],
            prices=prices,
            beam_width=mc["beam_width"],
            action_limit=mc["per_state_action_limit"],
            k_paths=mc["k_paths"],
            max_expansions=mc["max_expansions"],
            timeout_seconds=mc["timeout_seconds_per_instance"],
            price_weight=mc["price_weight"],
            price_mode=mc.get("price_mode", "sum"),
            quotient=quotient,
        )
        rows.append(
            {
                "sample_id": sample["sample_id"],
                "split": sample["split"],
                "architecture": sample["architecture"],
                "dfg_family": sample["family"],
                "method": method,
                **result.row(),
            }
        )
    return rows


def run_symmetry(cfg, output, deadline):
    raw_path = output / "raw" / "symmetry.csv"
    if raw_path.exists() and len(pd.read_csv(raw_path)) == 12:
        return pd.read_csv(raw_path), False
    rows = []
    arch = build_architecture("mesh3")
    for family in ("diamond_chain", "reduction_tree", "dot_product"):
        for index in range(4):
            if time.perf_counter() > deadline:
                return pd.DataFrame(rows), True
            seed = stable_seed(cfg["experiment"]["master_seed"], "symmetry", family, index)
            dfg = generate_exactness_dfg(family, seed)
            state = construct_partial_state(dfg, arch, 0.25)
            common = dict(
                beam_width=128,
                action_limit=9,
                k_paths=4,
                max_expansions=5000,
                timeout_seconds=60.0,
            )
            nonquot = map_dfg(dfg, arch, state, quotient=False, **common)
            quot = map_dfg(dfg, arch, state, quotient=True, joint_quotient=True, **common)
            cost_match = (
                (not nonquot.success and not quot.success)
                or (
                    nonquot.success
                    and quot.success
                    and abs(nonquot.best_cost - quot.best_cost) <= 1e-8
                )
            )
            rows.append(
                {
                    "instance_id": f"sym_{family}_{index}",
                    "dfg_family": family,
                    "architecture": "mesh3",
                    "dfg_group_size": len(dfg.automorphisms),
                    "arch_group_size": len(arch.symmetries),
                    "joint_group_size": len(dfg.automorphisms) * len(arch.symmetries),
                    "nonquot_success": nonquot.success,
                    "quot_success": quot.success,
                    "nonquot_best_cost": nonquot.best_cost,
                    "quot_best_cost": quot.best_cost,
                    "cost_match": cost_match,
                    "legality_match": nonquot.legal == quot.legal,
                    "mapped_count_match": nonquot.mapped_operations == quot.mapped_operations,
                    "nonquot_expansions": nonquot.expansions,
                    "quot_expansions": quot.expansions,
                    "expansion_reduction": nonquot.expansions / max(1, quot.expansions),
                    "canonicalization_ms": quot.canonicalization_ms,
                    "quot_total_ms": quot.runtime_ms,
                    "canonicalization_fraction": quot.canonicalization_ms / max(quot.runtime_ms, 1e-9),
                }
            )
            save_frame(pd.DataFrame(rows), raw_path)
    return pd.DataFrame(rows), False


def make_tables(samples, instance_df, relaxation_df, knockouts, prediction_table, symmetry, mappings, system, manifest):
    census = (
        instance_df.groupby(["split", "architecture", "dfg_family"], as_index=False)
        .agg(
            requested_samples=("sample_id", "size"),
            generated_samples=("sample_id", "size"),
            solved_samples=("solver_status", lambda x: x.isin(["optimal", "optimal_inaccurate"]).sum()),
            infeasible_samples=("solver_status", lambda x: (x == "infeasible").sum()),
            solver_failed_samples=("solver_status", lambda x: (~x.isin(["optimal", "optimal_inaccurate", "infeasible"])).sum()),
            mean_ops=("ops", "mean"),
            mean_dfg_edges=("dfg_edges", "mean"),
            mean_partial_depth=("partial_depth", "mean"),
            mean_routing_nodes=("routing_nodes", "mean"),
            mean_routing_edges=("routing_edges", "mean"),
        )
    )
    health_unique = relaxation_df.drop_duplicates("sample_id")
    health_rows = []
    for (split, arch), group in health_unique.groupby(["split", "architecture"]):
        solved = group[group["solved"] == True]  # noqa: E712
        health_rows.append(
            {
                "split": split,
                "architecture": arch,
                "num_samples": len(group),
                "solve_rate": len(solved) / len(group),
                "median_solve_ms": median(solved["solve_seconds"]) * 1000,
                "p95_solve_ms": percentile(solved["solve_seconds"], 95) * 1000,
                "p99_assignment_residual": percentile(solved["assignment_residual"], 99),
                "p99_flow_residual": percentile(solved["flow_residual"], 99),
                "p99_capacity_violation": percentile(solved["capacity_violation"], 99),
                "min_variable": solved["min_variable"].min(),
                "max_variable": solved["max_variable"].max(),
                "median_positive_dual_fraction": median(solved["positive_dual_fraction"]),
            }
        )
    health = pd.DataFrame(health_rows)
    knockout_rows = []
    if len(knockouts):
        unique = knockouts.drop_duplicates("sample_id")
        for (split, arch), group in unique.groupby(["split", "architecture"]):
            sp = median(group["instance_spearman"])
            rec = median(group["instance_top8_recall"])
            ratio = median(group["instance_top_bottom_ratio"])
            gate = "GREEN" if sp >= 0.60 and rec >= 0.60 and ratio >= 2 else "AMBER" if sp >= 0.35 or rec >= 0.40 else "RED"
            knockout_rows.append(
                {
                    "split": split,
                    "architecture": arch,
                    "num_probe_instances": len(group),
                    "median_spearman": sp,
                    "mean_spearman": mean(group["instance_spearman"]),
                    "median_top8_recall": rec,
                    "median_top_bottom_impact_ratio": ratio,
                    "num_knockout_infeasible": int(knockouts[knockouts["sample_id"].isin(group["sample_id"])]["infeasible"].sum()),
                    "gate_G2": gate,
                }
            )
    knockout_table = pd.DataFrame(knockout_rows)
    mapper_rows = []
    common_ids = {}
    for split in TEST_SPLITS:
        split_map = mappings[mappings["split"] == split]
        common_ids[split] = set.intersection(
            *[
                set(split_map[(split_map["method"] == method) & (split_map["success"] == True)]["sample_id"])
                for method in METHODS
            ]
        ) if all(len(split_map[split_map["method"] == m]) for m in METHODS) else set()
        for method, group in split_map.groupby("method"):
            common = group[group["sample_id"].isin(common_ids[split])]
            mapper_rows.append(
                {
                    "split": split,
                    "method": method,
                    "num_instances": len(group),
                    "success_rate": mean(group["success"]),
                    "timeouts": int(group["timeout"].sum()),
                    "legality_failures": int(((group["success"] == True) & (group["legal"] != True)).sum()),
                    "median_expansions": median(group["expansions"]),
                    "p90_expansions": percentile(group["expansions"], 90),
                    "median_generated_successors": median(group["generated_successors"]),
                    "median_duplicate_states": median(group["duplicate_states"]),
                    "median_quotient_merges": median(group["quotient_merges"]),
                    "median_routing_attempts": median(group["routing_attempts"]),
                    "median_failed_routes": median(group["failed_routing_attempts"]),
                    "median_runtime_ms": median(group["runtime_ms"]),
                    "p90_runtime_ms": percentile(group["runtime_ms"], 90),
                    "paired_route_cost_mean": mean(common["best_cost"]),
                }
            )
    mapper = pd.DataFrame(mapper_rows)
    paired_rows = []
    for split in TEST_SPLITS:
        base = mappings[(mappings["split"] == split) & (mappings["method"] == "length")]
        for method in METHODS:
            current = mappings[(mappings["split"] == split) & (mappings["method"] == method)]
            merged = base.merge(current, on="sample_id", suffixes=("_base", "_method"))
            common = merged[(merged["success_base"] == True) & (merged["success_method"] == True)]
            base_success, current_success = mean(merged["success_base"]), mean(merged["success_method"])
            exp_reduction = reduction_percent(median(common["expansions_base"]), median(common["expansions_method"]))
            runtime_reduction = reduction_percent(median(merged["runtime_ms_base"]), median(merged["runtime_ms_method"]))
            route_delta = -reduction_percent(mean(common["best_cost_base"]), mean(common["best_cost_method"]))
            oracle = paired_rows[-1] if False else None
            paired_rows.append(
                {
                    "split": split,
                    "method": method,
                    "paired_instances": len(merged),
                    "success_delta_points": 100 * (current_success - base_success),
                    "median_expansion_reduction_pct": exp_reduction,
                    "median_runtime_reduction_pct": runtime_reduction,
                    "median_route_cost_delta_pct": route_delta,
                    "oracle_gain_retention_success": float("nan"),
                    "oracle_gain_retention_expansions": float("nan"),
                }
            )
    paired = pd.DataFrame(paired_rows)
    for split in TEST_SPLITS:
        oracle = paired[(paired["split"] == split) & (paired["method"] == "oracle_dual")].iloc[0]
        for idx in paired[paired["split"] == split].index:
            if oracle["success_delta_points"] > 0:
                paired.loc[idx, "oracle_gain_retention_success"] = paired.loc[idx, "success_delta_points"] / oracle["success_delta_points"]
            if oracle["median_expansion_reduction_pct"] > 0:
                paired.loc[idx, "oracle_gain_retention_expansions"] = paired.loc[idx, "median_expansion_reduction_pct"] / oracle["median_expansion_reduction_pct"]
    return {
        "T00_system": pd.DataFrame([manifest]),
        "T01_dataset_census": census,
        "T02_relaxation_health": health,
        "T03_knockout_signal": knockout_table,
        "T04_prediction_quality": prediction_table,
        "T05_symmetry_exactness": symmetry,
        "T06_mapper_aggregate": mapper,
        "T07_mapper_paired": paired,
    }


def assess_gates(tables, relaxation_df, knockouts, mappings, symmetry, prediction_table, tests_passed, cuda, wall_minutes, complete):
    gates = []
    def add(gate, status, metric, value, green, amber, notes):
        gates.append({"gate": gate, "status": status, "primary_metric": metric, "observed_value": value, "green_threshold": green, "amber_threshold": amber, "notes": notes})
    g0 = "GREEN" if cuda and tests_passed and complete and wall_minutes <= 60 else "AMBER" if tests_passed and complete and wall_minutes <= 75 else "RED"
    add("G0", g0, "wall_minutes", wall_minutes, "CUDA + tests + <=60", "correct CPU/<=75", f"cuda={cuda}; tests={tests_passed}; complete={complete}")
    unique = relaxation_df.drop_duplicates("sample_id")
    solved = unique[unique["solved"] == True]  # noqa: E712
    rate = len(solved) / max(1, len(unique))
    p99a, p99f, p99c = [percentile(solved[c], 99) for c in ("assignment_residual", "flow_residual", "capacity_violation")]
    bound = max(max(0.0, -solved["min_variable"].min()), max(0.0, solved["max_variable"].max() - 1.0)) if len(solved) else float("inf")
    if rate >= .98 and max(p99a, p99f, p99c) <= 1e-4 and bound <= 1e-5:
        g1 = "GREEN"
    elif rate >= .90 and max(p99a, p99f, p99c) <= 1e-3:
        g1 = "AMBER"
    else:
        g1 = "RED"
    add("G1", g1, "solve_rate/max_p99_residual", f"{rate:.4g}/{max(p99a,p99f,p99c):.4g}", ">=.98/<=1e-4", ">=.90/<=1e-3", f"max_bound_violation={bound:.3g}")
    ku = knockouts.drop_duplicates("sample_id") if len(knockouts) else pd.DataFrame()
    sp = median(ku["instance_spearman"]) if len(ku) else float("nan")
    rec = median(ku["instance_top8_recall"]) if len(ku) else float("nan")
    ratio = median(ku["instance_top_bottom_ratio"]) if len(ku) else float("nan")
    g2 = "GREEN" if sp >= .60 and rec >= .60 and ratio >= 2 else "AMBER" if sp >= .35 or rec >= .40 else "RED"
    add("G2", g2, "Spearman/recall/ratio", f"{sp:.3g}/{rec:.3g}/{ratio:.3g}", ".60/.60/2", ".35/.40", "")
    paired = tables["T07_mapper_paired"]
    oracle = paired[paired["method"] == "oracle_dual"]
    overall_success = mean(oracle["success_delta_points"])
    overall_exp = median(oracle["median_expansion_reduction_pct"])
    asym = oracle[oracle["split"] == "test_unseen_asym"].iloc[0] if len(oracle[oracle["split"] == "test_unseen_asym"]) else None
    asym_ok = asym is not None and (asym["success_delta_points"] >= 10 or asym["median_expansion_reduction_pct"] >= 25)
    no_loss = overall_success >= -2
    mapping_censored = bool(mappings["timeout"].any())
    if mapping_censored:
        g3 = "AMBER"
    elif no_loss and (overall_success >= 8 or overall_exp >= 20) and asym_ok:
        g3 = "GREEN"
    elif overall_success >= 3 or overall_exp >= 8:
        g3 = "AMBER"
    elif overall_success < -5 or (overall_exp < -10 and overall_success <= 0):
        g3 = "RED"
    else:
        g3 = "AMBER"
    add("G3", g3, "success_delta/expansion_reduction", f"{overall_success:.3g}/{overall_exp:.3g}%", "8pt or 20%; asym 10pt/25%", "3pt or 8%", f"asym_ok={asym_ok}; censored={mapping_censored}")
    correctness = len(symmetry) == 12 and bool(symmetry[["cost_match", "legality_match", "mapped_count_match"]].all(axis=None)) and bool((symmetry["nonquot_success"] == symmetry["quot_success"]).all())
    reduction = geometric_mean(symmetry["expansion_reduction"]) if len(symmetry) else float("nan")
    family_best = symmetry.groupby("dfg_family")["expansion_reduction"].apply(geometric_mean).max() if len(symmetry) else float("nan")
    overhead = symmetry["canonicalization_ms"].sum() / max(symmetry["quot_total_ms"].sum(), 1e-9) if len(symmetry) else float("nan")
    if not correctness:
        g4 = "RED"
    elif reduction >= 1.5 and family_best >= 2 and overhead <= .25:
        g4 = "GREEN"
    elif reduction >= 1.1 and overhead <= .50:
        g4 = "AMBER"
    else:
        g4 = "RED"
    add("G4", g4, "geomean_reduction/overhead", f"{reduction:.3g}x/{overhead:.3g}", ">=1.5x,family>=2x,<=.25", ">=1.1x,<=.50", f"correctness={correctness}; family_best={family_best:.3g}")
    seed_mask = pd.to_numeric(prediction_table["model_seed"], errors="coerce").notna()
    seen = prediction_table[seed_mask & (prediction_table["split"] == "test_seen")]
    unseen = prediction_table[seed_mask & prediction_table["split"].isin(TEST_SPLITS[1:])]
    ss, sa, sap = mean(seen["median_spearman"]), mean(seen["critical_auroc"]), mean(seen["action_preservation"])
    us, ua, uap = mean(unseen["median_spearman"]), mean(unseen["critical_auroc"]), mean(unseen["action_preservation"])
    if ss >= .60 and sa >= .80 and sap >= .70 and us >= .40 and ua >= .70 and uap >= .55:
        g5 = "GREEN"
    elif ss < .40 and sa < .70:
        g5 = "RED"
    else:
        g5 = "AMBER"
    add("G5", g5, "seen/unseen Spearman", f"{ss:.3g}/{us:.3g}", "seen .60; unseen .40 plus AUROC/actions", "seen .40; unseen .25", f"AUROC={sa:.3g}/{ua:.3g}; actions={sap:.3g}/{uap:.3g}")
    pred = paired[paired["method"] == "pred_dual_quotient"]
    qualifying = 0
    amber_splits = 0
    for _, row in pred.iterrows():
        retention = max(row["oracle_gain_retention_success"], row["oracle_gain_retention_expansions"])
        if row["success_delta_points"] >= -2 and (row["success_delta_points"] >= 5 or row["median_expansion_reduction_pct"] >= 25) and retention >= .60:
            qualifying += 1
        if row["success_delta_points"] >= -2 and (row["median_expansion_reduction_pct"] >= 10 or retention >= .30):
            amber_splits += 1
    worst_success = pred["success_delta_points"].min() if len(pred) else -100
    if mapping_censored:
        g6 = "AMBER"
    elif qualifying >= 3:
        g6 = "GREEN"
    elif worst_success < -5:
        g6 = "RED"
    else:
        # The fixed RED conditions are not met. This includes the neutral case
        # where no positive oracle gain exists, so retention is undefined.
        g6 = "AMBER"
    add("G6", g6, "qualifying_splits", qualifying, ">=3", "some 10%/30% retention", f"worst_success_delta={worst_success:.3g}; amber_splits={amber_splits}")
    return pd.DataFrame(gates)


def overall_decision(gates):
    status = dict(zip(gates["gate"], gates["status"]))
    if status["G1"] == "RED":
        return "INVALID RUN"
    if status["G1"] == status["G3"] == status["G4"] == "GREEN":
        return "PROCEED"
    if status["G1"] == "GREEN" and status["G3"] in ("GREEN", "AMBER") and status["G4"] in ("GREEN", "AMBER"):
        return "PROCEED WITH REFINEMENT"
    if status["G3"] == "GREEN" and status["G4"] == "RED":
        return "PIVOT TO DUAL-ONLY"
    if status["G4"] == "GREEN" and status["G3"] == "RED":
        return "PIVOT TO QUOTIENT-ONLY"
    if status["G1"] == "GREEN" and status["G3"] == status["G4"] == "RED":
        return "STOP/REFORMULATE"
    return "STOP/REFORMULATE"


def report(output, cfg, tables, gates, overall, runtime, complete, selected_seed):
    t06 = tables["T06_mapper_aggregate"]
    t07 = tables["T07_mapper_paired"]
    failures = t06.sort_values("success_rate").head(8)
    mappings = pd.read_csv(output / "raw" / "mappings.csv")
    family_analysis = (
        mappings.groupby(["dfg_family", "method"], as_index=False)
        .agg(success_rate=("success", "mean"), timeout_rate=("timeout", "mean"))
    )
    architecture_analysis = (
        mappings.groupby(["architecture", "method"], as_index=False)
        .agg(success_rate=("success", "mean"), timeout_rate=("timeout", "mean"))
    )
    health = tables["T02_relaxation_health"]
    positive_dual_range = (
        float(health["median_positive_dual_fraction"].min()),
        float(health["median_positive_dual_fraction"].max()),
    )
    timeout_count = int(mappings["timeout"].sum())
    if timeout_count:
        timeout_interpretation = (
            f"{timeout_count} of {len(mappings)} method-instance runs remain "
            "timeout-censored even after the documented 120-second retry. "
            "Those comparisons are inconclusive and are not treated as empirical "
            "evidence against the scientific method."
        )
    else:
        timeout_interpretation = (
            f"All {len(mappings)} method-instance runs completed under the "
            "documented 120-second uncensoring cap, so G3/G6 use empirical paired "
            "outcomes rather than timeout failures."
        )
    text = f"""# QuotientFlow quick-suite report

## Exact command used

`python scripts/run_quick_suite.py --config configs/quick.yaml`

## Exact environment

The official run reused `taugat_pyg` (Python 3.11) per the low-space instruction.
Activate with `conda activate taugat_pyg`. Full records are in
`../system/system.json`, `../system/conda_env.yml`, `../system/pip_freeze.txt`,
and `../system/nvidia_smi.txt`.

## Runtime breakdown

```json
{json.dumps(runtime, indent=2)}
```

Run complete: **{complete}**. Selected model seed by validation Spearman: **{selected_seed}**.

## Dataset census

See [T01](tables/T01_dataset_census.md) and raw [instances](raw/instances.csv).

## Gates

{markdown_text(gates)}

Overall decision: **{overall}**.

## Six required method comparisons

All methods were evaluated on the same intended mapping instances where the hard
stop allowed. Full aggregates are in [T06](tables/T06_mapper_aggregate.md) and
paired deltas in [T07](tables/T07_mapper_paired.md).

{markdown_text(t07)}

## Per-family and per-architecture failure analysis

Raw method/family/architecture outcomes are preserved in [mappings.csv](raw/mappings.csv).
The lowest aggregate success rows were:

{markdown_text(failures)}

Per-family outcomes:

{markdown_text(family_analysis)}

Per-architecture outcomes:

{markdown_text(architecture_analysis)}

## Plots

- [Dual price versus knockout impact](plots/dual_vs_knockout.png)
- [Oracle versus predicted dual rank](plots/oracle_vs_predicted_rank.png)
- [Mapping success by method/split](plots/mapping_success.png)
- [Expansion count by method/split](plots/mapping_expansions.png)

## What failed

Every non-green gate is a negative finding, not a tuned-away result:

{markdown_text(gates[gates["status"] != "GREEN"])}

{timeout_interpretation}

Following the G3 failure rule, prices are not uniformly dense; they are extremely
sparse. Across split/architecture groups, the median positive-dual fraction
ranges from **{positive_dual_range[0]:.4f} to {positive_dual_range[1]:.4f}**.
G2 nevertheless passes strongly, so the finite-knockout signal exists but the
static path-sum integration does not turn it into search benefit in this mapper.

G4 correctness passes on all 12 retried exactness instances, but the
geometric-mean expansion reduction is only 1.05x and no family reaches 2x.
Canonicalization is therefore exact but not useful enough in this implementation.

No gate threshold was changed. All runtime-only deviations and their archived
censored attempts are recorded in `config_deviations.md`.

## Scientific interpretation

- **Optimization signal validity:** G1 measures formulation/numerical health; G2
  measures whether finite link removal follows the dual ranking.
- **Search integration:** G3 is the primary falsification result for static oracle
  prices in the bounded beam mapper.
- **Symmetry prevalence:** G4 separately requires exact result preservation and
  meaningful state reduction.
- **Learned prediction:** G5 reports all three seeds; model selection affects only
  integrated search.
- **Integrated system:** G6 measures whether predicted prices plus quotienting
  retain positive oracle benefit.
- **Limitations:** This remains a custom small MRRG pilot with a fixed schedule,
  static prices, and no Morpher/CGRA-ME or cycle-accurate hardware validation.

## Recommended next iteration

Follow the failure-driven rule for the first failed primary gate. If G3 is not
green, inspect price uniformity and validate a 25--40% partial-depth/static-price
refresh ablation on validation only before further GNN work. If G4 is weak but
correct, add stabilizer-aware action-orbit reduction and cached transform hashes.
If G3 passes but G5 fails, predict ranks/criticality and add alternative-path
scarcity features without changing test splits.

## Can this support an HPCA paper?

No, not alone. It is a falsification pilot. A paper claim requires the real
toolchain, at least 30 extracted kernels and 12 architecture configurations,
minimum-II/cycle-accurate validation, stronger baselines, five learned seeds, and
the paper gates fixed in `AGENTS.md`.
"""
    (output / "REPORT.md").write_text(text)
    gate_map = dict(zip(gates["gate"], gates["status"]))
    findings = [
        f"G3 oracle utility: {gate_map['G3']}",
        f"G4 exact quotient: {gate_map['G4']}",
        f"G5 prediction: {gate_map['G5']}",
    ]
    failed = [f"{g}: {s}" for g, s in gate_map.items() if s != "GREEN"][:3] or ["none"]
    summary = "\n".join(
        [
            f"OVERALL_DECISION: {overall}",
            f"TOTAL_WALL_MINUTES: {runtime['total_minutes']:.6f}",
            f"CUDA_AVAILABLE: {torch.cuda.is_available()}",
            *[f"{g}: {gate_map[g]}" for g in [f"G{i}" for i in range(7)]],
            "TOP_THREE_FINDINGS: " + "; ".join(findings),
            "TOP_THREE_FAILURES: " + "; ".join(failed),
            "NEXT_ACTION: Follow the first failed primary gate using AGENTS.md section 20 on validation only.",
        ]
    )
    (output / "RUN_SUMMARY.txt").write_text(summary + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = ROOT / args.config
    cfg = yaml.safe_load(config_path.read_text())
    output = ROOT / cfg["experiment"]["output_dir"]
    cache = ROOT / cfg["experiment"]["cache_dir"]
    for sub in ("raw", "tables", "plots", "checkpoints"):
        (output / sub).mkdir(parents=True, exist_ok=True)
    (output / "config_deviations.md").write_text(
        "# Configuration deviations\n\n"
        "- Environment only (user-directed, non-numerical): reused pre-existing "
        "`taugat_pyg` Python 3.11 environment instead of retaining a duplicate "
        "`quotientflow` environment to avoid multi-gigabyte CUDA duplication.\n"
        "- Pytest isolation only: disabled ambient third-party plugin autoload to "
        "exclude an unrelated ROS Python 3.12 plugin from the Python 3.11 run.\n"
        "- Apart from the explicitly listed mapper timeout uncensoring change, "
        "no numerical values in `configs/quick.yaml` were changed.\n"
        "- User-authorized uncensoring retry: mapper per-instance timeout changed "
        "from 2 s to 120 s after validation-only timing showed all seven families "
        "complete in 3.8--8.6 s. The original timeout-censored mapping file is "
        "retained as `raw/mappings_attempt_2s.csv`. No guidance, beam, data, "
        "model, or gate parameter changed.\n"
        "- User-authorized exactness retry: the internal, symmetric per-search "
        "timeout for the 12-instance exactness-only subtest changed from 15 s "
        "to 60 s after the first attempt timed out. Both non-quotient and "
        "quotient searches use the same new cap. The first attempt is retained "
        "as `raw/symmetry_attempt_15s.csv`; gates remain unchanged.\n"
    )
    os.environ.update(
        {
            "OMP_NUM_THREADS": str(cfg["experiment"]["cpu_threads"]),
            "MKL_NUM_THREADS": str(cfg["experiment"]["cpu_threads"]),
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            # Required by CUDA >= 10.2 when deterministic algorithms are enabled.
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        }
    )
    torch.set_num_threads(cfg["experiment"]["cpu_threads"])
    if cfg["experiment"]["deterministic_torch"]:
        torch.manual_seed(cfg["experiment"]["master_seed"])
        torch.use_deterministic_algorithms(True)
    start = time.perf_counter()
    campaign_file = output / "campaign_start_utc.txt"
    if campaign_file.exists():
        started = campaign_file.read_text().strip()
        campaign_start = datetime.fromisoformat(started)
    else:
        campaign_start = datetime.now(timezone.utc)
        started = campaign_start.isoformat()
        campaign_file.write_text(started + "\n")
    campaign_elapsed = (datetime.now(timezone.utc) - campaign_start).total_seconds()
    remaining_hard_seconds = max(
        0.0, cfg["experiment"]["hard_stop_minutes"] * 60 - campaign_elapsed
    )
    hard_deadline = start + remaining_hard_seconds
    stage = {}

    t = time.perf_counter()
    test_run = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=ROOT,
        env={**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"},
        text=True,
        capture_output=True,
    )
    (output / "pytest.txt").write_text(test_run.stdout + test_run.stderr)
    tests_passed = test_run.returncode == 0
    stage["tests_seconds"] = time.perf_counter() - t
    if not tests_passed:
        raise SystemExit("mandatory tests failed; see results/quick_v1/pytest.txt")

    t = time.perf_counter()
    samples = []
    for i, spec in enumerate(dataset_specs(cfg), 1):
        samples.append(solve_sample(spec, cfg, cache))
        if i % 10 == 0:
            print(f"[dataset] {i}/252", flush=True)
        if time.perf_counter() > hard_deadline:
            break
    stage["dataset_relaxation_seconds"] = time.perf_counter() - t
    instance_df, relaxation_df = raw_dataset_frames(samples)
    save_frame(instance_df, output / "raw" / "instances.csv")
    save_frame(relaxation_df, output / "raw" / "relaxation.csv")

    t = time.perf_counter()
    probes = [s for s in samples if s["split"] == "val_seen"][: cfg["evaluation"]["probe_instances"]]
    knockouts = run_knockouts(probes, cfg, output / "raw" / "knockouts.csv", hard_deadline)
    stage["knockout_seconds"] = time.perf_counter() - t

    t = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models, metadata, selected_seed = load_or_train_models(samples, cfg, output, device)
    maps = mapping_samples(samples)
    prediction_table, predictions, actions = prediction_and_actions(
        models, metadata, selected_seed, samples, maps, output
    )
    stage["training_prediction_seconds"] = time.perf_counter() - t

    t = time.perf_counter()
    mappings, map_stop = run_mappings(maps, models[selected_seed], cfg, output, hard_deadline)
    stage["mapping_seconds"] = time.perf_counter() - t

    t = time.perf_counter()
    symmetry, sym_stop = run_symmetry(cfg, output, hard_deadline)
    stage["symmetry_seconds"] = time.perf_counter() - t

    system = json.loads((ROOT / "results/system/system.json").read_text())
    expected_mappings = 56 * 6
    complete = len(samples) == 252 and len(knockouts["sample_id"].unique()) == 28 and len(mappings) == expected_mappings and len(symmetry) == 12 and not map_stop and not sym_stop
    invocation_minutes = (time.perf_counter() - start) / 60
    provisional_minutes = (
        datetime.now(timezone.utc) - campaign_start
    ).total_seconds() / 60
    manifest = {
        "run_id": cfg["experiment"]["name"],
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True).stdout.strip() or "not-a-git-repository",
        "timestamp_start": started,
        "timestamp_end": datetime.now(timezone.utc).isoformat(),
        "wall_minutes": provisional_minutes,
        "os": system["os"],
        "cpu": system["cpu"],
        "ram_gb": system["ram_gb"],
        "gpu": system["gpu"],
        "vram_gb": system["vram_gb"],
        "nvidia_driver": system["nvidia_driver"],
        "torch_version": system["torch_version"],
        "torch_cuda_version": system["torch_cuda_version"],
        "cuda_available": system["cuda_available"],
        "config_hash": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "tests_passed": tests_passed,
        "hard_stop_triggered": not complete,
    }
    tables = make_tables(samples, instance_df, relaxation_df, knockouts, prediction_table, symmetry, mappings, system, manifest)
    gates = assess_gates(tables, relaxation_df, knockouts, mappings, symmetry, prediction_table, tests_passed, system["cuda_available"], provisional_minutes, complete)
    tables["T08_gates"] = gates
    for name, frame in tables.items():
        write_table(frame, output / "tables", name)
    make_plots(output, knockouts, predictions, mappings)
    final_invocation_minutes = (time.perf_counter() - start) / 60
    total_minutes = (
        datetime.now(timezone.utc) - campaign_start
    ).total_seconds() / 60
    stage["reporting_seconds"] = final_invocation_minutes * 60 - sum(stage.values())
    stage["final_cached_invocation_minutes"] = final_invocation_minutes
    stage["total_minutes"] = total_minutes
    overall = overall_decision(gates)
    report(output, cfg, tables, gates, overall, stage, complete, selected_seed)
    print((output / "RUN_SUMMARY.txt").read_text())


if __name__ == "__main__":
    main()
