#!/usr/bin/env python3
"""Exact stabilizer-aware action-orbit evaluation for revision-v2."""

from __future__ import annotations

import concurrent.futures
import json
import math
import pickle
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quotientflow.arch import build_architecture
from quotientflow.dfg import generate_exactness_dfg
from quotientflow.mapper import map_dfg
from quotientflow.partial_state import construct_partial_state
from scripts.run_quick_suite import stable_seed


OUTPUT = ROOT / "results" / "revision_v2"
RAW = OUTPUT / "raw"
VARIANTS = {
    "no_symmetry": (False, False),
    "successor_canonicalization": (True, False),
    "action_orbit": (False, True),
    "both": (True, True),
}


def length_scores(dfg, arch, state, operation, actions):
    return [action.base_cost for action in actions]


def run_instance(payload):
    instance_id, kind, dfg, arch, state = payload
    rows = []
    common = dict(
        beam_width=128 if kind == "prior_exactness" else 4,
        action_limit=9 if kind == "prior_exactness" else 6,
        k_paths=4,
        max_expansions=5000,
        timeout_seconds=120.0,
        action_scorer=length_scores,
    )
    for variant, (quotient, orbit) in VARIANTS.items():
        result = map_dfg(
            dfg,
            arch,
            state,
            quotient=quotient,
            joint_quotient=quotient,
            action_orbit_reduction=orbit,
            **common,
        )
        rows.append(
            {
                "instance_id": instance_id,
                "subset": kind,
                "dfg_family": dfg.family,
                "architecture": arch.name,
                "variant": variant,
                "dfg_group_size": len(dfg.automorphisms),
                "arch_group_size": len(arch.symmetries),
                "success": result.success,
                "legal": result.legal,
                "best_cost": result.best_cost,
                "mapped_operations": result.mapped_operations,
                "expansions": result.expansions,
                "raw_candidate_actions": result.raw_candidate_actions,
                "action_orbit_representatives": result.action_orbit_representatives,
                "actions_removed": result.action_orbit_actions_removed,
                "candidate_reduction": result.raw_candidate_actions
                / max(1, result.action_orbit_representatives),
                "candidate_removed_pct": 100.0
                * result.action_orbit_actions_removed
                / max(1, result.raw_candidate_actions),
                "gnn_evaluations_avoided_if_enabled": result.action_orbit_actions_removed,
                "routing_attempts": result.routing_attempts,
                "routing_attempts_avoided": result.routing_attempts_avoided,
                "action_orbit_ms": result.action_orbit_ms,
                "runtime_ms": result.runtime_ms,
                "action_orbit_overhead_fraction": result.action_orbit_ms
                / max(result.runtime_ms, 1e-9),
                "timeout": result.timeout,
            }
        )
    return rows


def jobs():
    payloads = []
    for family in ("diamond_chain", "reduction_tree", "dot_product"):
        for index in range(4):
            arch = build_architecture("mesh3")
            seed = stable_seed(24072026, "symmetry", family, index)
            dfg = generate_exactness_dfg(family, seed)
            state = construct_partial_state(dfg, arch, 0.10)
            payloads.append(
                (
                    f"prior_{family}_{index}",
                    "prior_exactness",
                    dfg,
                    arch,
                    state,
                )
            )
    cache = ROOT / "results" / "cache" / "samples"
    additional = []
    for architecture in ("mesh3", "torus3"):
        for family in (
            "diamond_chain",
            "reduction_tree",
            "dot_product",
            "butterfly",
        ):
            paths = sorted(
                cache.glob(f"train_seen__{architecture}__{family}__*.pkl")
            )[:2]
            for path in paths:
                with path.open("rb") as handle:
                    sample = pickle.load(handle)
                additional.append(
                    (
                        f"additional_{sample['sample_id']}",
                        "additional_symmetric",
                        sample["dfg"],
                        sample["arch"],
                        sample["state"],
                    )
                )
    return payloads + additional


def main():
    start = time.perf_counter()
    rows = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=6) as executor:
        for result in executor.map(run_instance, jobs()):
            rows.extend(result)
            pd.DataFrame(rows).to_csv(
                RAW / "action_orbits.partial.csv", index=False
            )
    frame = pd.DataFrame(rows)
    baseline = frame[frame["variant"] == "no_symmetry"].set_index("instance_id")
    checks = []
    paired_candidate_reduction = []
    paired_routing_reduction = []
    paired_removed = []
    for row in frame.itertuples():
        reference = baseline.loc[row.instance_id]
        cost_match = (
            row.success == reference.success
            and (
                not row.success
                or math.isclose(
                    row.best_cost, reference.best_cost, abs_tol=1e-8
                )
            )
        )
        checks.append(
            row.success == reference.success
            and row.legal == reference.legal
            and cost_match
        )
        candidate_reduction = reference.raw_candidate_actions / max(
            1, row.action_orbit_representatives
        )
        routing_reduction = reference.routing_attempts / max(
            1, row.routing_attempts
        )
        paired_candidate_reduction.append(candidate_reduction)
        paired_routing_reduction.append(routing_reduction)
        paired_removed.append(
            max(0, reference.raw_candidate_actions - row.action_orbit_representatives)
        )
    frame["exact_match"] = checks
    frame["paired_candidate_reduction"] = paired_candidate_reduction
    frame["paired_routing_attempt_reduction"] = paired_routing_reduction
    frame["paired_actions_removed"] = paired_removed
    frame["gnn_evaluations_avoided_if_enabled"] = paired_removed
    frame.to_csv(RAW / "action_orbits.csv", index=False)
    orbit = frame[frame["variant"].isin(["action_orbit", "both"])]
    assessment = {
        "runtime_seconds": time.perf_counter() - start,
        "instances": int(frame["instance_id"].nunique()),
        "prior_exactness_instances": int(
            frame[frame["subset"] == "prior_exactness"]["instance_id"].nunique()
        ),
        "additional_instances": int(
            frame[frame["subset"] == "additional_symmetric"][
                "instance_id"
            ].nunique()
        ),
        "all_exact": bool(frame["exact_match"].all()),
        "timeouts": int(frame["timeout"].sum()),
        "geomean_candidate_reduction": float(
            orbit["paired_candidate_reduction"].clip(lower=1e-12).prod()
            ** (1.0 / len(orbit))
        ),
        "mean_removed_pct": float(
            100.0
            * orbit["paired_actions_removed"].sum()
            / (2.0 * frame[
                frame["variant"].isin(["no_symmetry"])
            ]["raw_candidate_actions"].sum())
        ),
        "gnn_evaluation_reduction": float(
            2.0
            * frame[frame["variant"] == "no_symmetry"][
                "raw_candidate_actions"
            ].sum()
            / max(1, orbit["action_orbit_representatives"].sum())
        ),
        "routing_attempt_reduction": float(
            2.0
            * frame[frame["variant"] == "no_symmetry"][
                "routing_attempts"
            ].sum()
            / max(1, orbit["routing_attempts"].sum())
        ),
        "action_orbit_overhead_fraction": float(
            orbit["action_orbit_ms"].sum() / orbit["runtime_ms"].sum()
        ),
    }
    (OUTPUT / "action_orbit_assessment.json").write_text(
        json.dumps(assessment, indent=2) + "\n"
    )
    print(json.dumps(assessment, indent=2))


if __name__ == "__main__":
    main()
