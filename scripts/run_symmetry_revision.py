#!/usr/bin/env python
"""Post-baseline exactness study using exactly one canonical anchor."""

from __future__ import annotations

import concurrent.futures
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quotientflow.arch import build_architecture
from quotientflow.dfg import generate_exactness_dfg
from quotientflow.mapper import map_dfg
from quotientflow.metrics import geometric_mean
from quotientflow.partial_state import construct_partial_state
from scripts.run_quick_suite import stable_seed


def worker(payload):
    family, index = payload
    arch = build_architecture("mesh3")
    seed = stable_seed(24072026, "symmetry", family, index)
    dfg = generate_exactness_dfg(family, seed)
    # ceil(0.10*n) is one for all 7--10 operation exactness DFGs.
    state = construct_partial_state(dfg, arch, 0.10)
    common = dict(
        beam_width=128,
        action_limit=9,
        k_paths=4,
        max_expansions=5000,
        timeout_seconds=120.0,
    )
    nonquot = map_dfg(dfg, arch, state, quotient=False, **common)
    quot = map_dfg(
        dfg, arch, state, quotient=True, joint_quotient=True, **common
    )
    return {
        "instance_id": f"sym_anchor1_{family}_{index}",
        "dfg_family": family,
        "architecture": "mesh3",
        "anchored_operations": len(state.placements),
        "dfg_group_size": len(dfg.automorphisms),
        "arch_group_size": len(arch.symmetries),
        "joint_group_size": len(dfg.automorphisms) * len(arch.symmetries),
        "nonquot_success": nonquot.success,
        "quot_success": quot.success,
        "nonquot_best_cost": nonquot.best_cost,
        "quot_best_cost": quot.best_cost,
        "cost_match": (
            nonquot.success == quot.success
            and (
                not nonquot.success
                or abs(nonquot.best_cost - quot.best_cost) <= 1e-8
            )
        ),
        "legality_match": nonquot.legal == quot.legal,
        "mapped_count_match": nonquot.mapped_operations == quot.mapped_operations,
        "nonquot_expansions": nonquot.expansions,
        "quot_expansions": quot.expansions,
        "expansion_reduction": nonquot.expansions / max(1, quot.expansions),
        "canonicalization_ms": quot.canonicalization_ms,
        "quot_total_ms": quot.runtime_ms,
        "canonicalization_fraction": quot.canonicalization_ms
        / max(quot.runtime_ms, 1e-9),
        "nonquot_timeout": nonquot.timeout,
        "quot_timeout": quot.timeout,
    }


def main():
    jobs = [
        (family, index)
        for family in ("diamond_chain", "reduction_tree", "dot_product")
        for index in range(4)
    ]
    start = time.perf_counter()
    rows = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=6) as executor:
        for row in executor.map(worker, jobs):
            rows.append(row)
    frame = pd.DataFrame(rows)
    output = ROOT / "results/revision_v1"
    frame.to_csv(output / "symmetry_anchor1.csv", index=False)
    correctness = bool(
        len(frame) == 12
        and frame[
            ["cost_match", "legality_match", "mapped_count_match"]
        ].all(axis=None)
        and (frame["nonquot_success"] == frame["quot_success"]).all()
        and not frame[["nonquot_timeout", "quot_timeout"]].any(axis=None)
    )
    reduction = geometric_mean(frame["expansion_reduction"])
    family = (
        frame.groupby("dfg_family")["expansion_reduction"]
        .apply(geometric_mean)
        .to_dict()
    )
    overhead = frame["canonicalization_ms"].sum() / frame["quot_total_ms"].sum()
    gate = (
        "RED"
        if not correctness
        else "GREEN"
        if reduction >= 1.5 and max(family.values()) >= 2 and overhead <= 0.25
        else "AMBER"
        if reduction >= 1.1 and overhead <= 0.50
        else "RED"
    )
    assessment = {
        "correctness": correctness,
        "geomean_expansion_reduction": reduction,
        "family_geomeans": family,
        "canonicalization_fraction": overhead,
        "G4_revision": gate,
        "runtime_seconds": time.perf_counter() - start,
    }
    (output / "symmetry_anchor1_assessment.json").write_text(
        json.dumps(assessment, indent=2) + "\n"
    )
    print(frame.to_string(index=False))
    print(json.dumps(assessment, indent=2))


if __name__ == "__main__":
    main()
