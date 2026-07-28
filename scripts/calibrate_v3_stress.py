#!/usr/bin/env python3
"""Validation-only stress calibration followed by frozen test generation."""

from __future__ import annotations

import concurrent.futures
import json
import pickle
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quotientflow.arch import build_architecture  # noqa: E402
from quotientflow.dfg import FAMILIES, generate_stress_dfg  # noqa: E402
from quotientflow.flow_advantage import target_complete_path_bounded  # noqa: E402
from quotientflow.mapper import map_dfg  # noqa: E402
from quotientflow.partial_state import PartialState, construct_partial_state  # noqa: E402

OUT = ROOT / "results" / "revision_v3"
RAW = OUT / "raw"
CACHE = OUT / "cache" / "stress"
ARCHITECTURES = ("mesh3", "mesh4", "cut4", "diag4")
FRACTIONS = (0.0, 0.10, 0.20)
MEDIUM = {
    "beam_width": 2,
    "action_limit": 4,
    "max_expansions": 1500,
}


def common_preselector(dfg, arch, state, operation, actions):
    return target_complete_path_bounded(actions, 4)


def build_instance(split, family, architecture, seed, fraction):
    dfg = generate_stress_dfg(family, seed)
    arch = build_architecture(architecture)
    try:
        state = (
            PartialState()
            if fraction == 0.0
            else construct_partial_state(dfg, arch, fraction)
        )
        error = ""
    except Exception as exc:
        state = PartialState()
        error = f"{type(exc).__name__}: {exc}"
    instance_id = (
        f"{split}__{architecture}__{family}__{seed}__d{fraction:.2f}"
    )
    return {
        "instance_id": instance_id,
        "split": split,
        "architecture": architecture,
        "dfg_family": family,
        "seed": seed,
        "start_fraction": fraction,
        "dfg": dfg,
        "arch": arch,
        "state": state,
        "construction_error": error,
    }


def evaluate_length(instance):
    if instance["construction_error"]:
        return {
            "instance_id": instance["instance_id"],
            "success": False,
            "legal": False,
            "timeout": False,
            "reason": instance["construction_error"],
            "runtime_ms": 0.0,
        }
    result = map_dfg(
        instance["dfg"], instance["arch"], instance["state"],
        k_paths=4, timeout_seconds=600.0,
        action_preselector=common_preselector,
        **MEDIUM,
    )
    return {"instance_id": instance["instance_id"], **result.row()}


def main():
    started = time.perf_counter()
    RAW.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    validation_instances = []
    for fraction in FRACTIONS:
        for family_index, family in enumerate(FAMILIES):
            for arch_index, architecture in enumerate(ARCHITECTURES):
                seed = 330000 + 100 * family_index + 10 * arch_index
                validation_instances.append(build_instance(
                    "stress_validation", family, architecture, seed, fraction
                ))
    with concurrent.futures.ProcessPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(evaluate_length, validation_instances))
    lookup = {row["instance_id"]: row for row in results}
    calibration = []
    for instance in validation_instances:
        result = lookup[instance["instance_id"]]
        calibration.append({
            "instance_id": instance["instance_id"],
            "architecture": instance["architecture"],
            "dfg_family": instance["dfg_family"],
            "seed": instance["seed"],
            "start_fraction": instance["start_fraction"],
            **result,
        })
    frame = pd.DataFrame(calibration)
    rates = frame.groupby("start_fraction").success.mean()
    valid = rates[(rates >= .40) & (rates <= .80)]
    if len(valid):
        # Fixed deterministic preference: shallowest valid starting depth.
        selected_fraction = float(valid.index.min())
        validity = "VALID"
    else:
        # This is a calibration outcome, not test filtering. Choose closest to
        # the center and mark difficulty invalid if even the frozen test remains
        # outside the mandated range.
        selected_fraction = float((rates - .60).abs().sort_values().index[0])
        validity = "VALIDATION_TARGET_MISSED"
    selection = {
        "selected_start_fraction": selected_fraction,
        "validation_length_medium_success": float(rates[selected_fraction]),
        "validation_status": validity,
        "allowed_candidates": list(FRACTIONS),
        "selection_rule": "shallowest fraction in [0.40,0.80], else closest to 0.60",
        "test_observed_at_selection": False,
        "medium_budget": MEDIUM,
    }
    (OUT / "stress_selection.json").write_text(
        json.dumps(selection, indent=2) + "\n"
    )
    frame.to_csv(RAW / "stress_validation_calibration.csv", index=False)

    # Test generation happens only after the validation selection is durable.
    test_rows = []
    for family_index, family in enumerate(FAMILIES):
        for arch_index, architecture in enumerate(ARCHITECTURES):
            for replicate in range(2):
                seed = (
                    440000 + 1000 * replicate
                    + 100 * family_index + 10 * arch_index
                )
                instance = build_instance(
                    "stress_test", family, architecture, seed, selected_fraction
                )
                path = CACHE / f"{instance['instance_id']}.pkl"
                with path.open("wb") as handle:
                    pickle.dump(instance, handle, protocol=pickle.HIGHEST_PROTOCOL)
                test_rows.append({
                    key: instance[key] for key in (
                        "instance_id", "split", "architecture", "dfg_family",
                        "seed", "start_fraction", "construction_error",
                    )
                } | {"instance_path": str(path.relative_to(ROOT))})
    pd.DataFrame(test_rows).to_csv(RAW / "stress_instances.csv", index=False)
    runtime = {
        "seconds": time.perf_counter() - started,
        "selected_fraction": selected_fraction,
        "validation_rates": {str(k): float(v) for k, v in rates.items()},
        "test_instances": len(test_rows),
    }
    (OUT / "stress_calibration_runtime.json").write_text(
        json.dumps(runtime, indent=2) + "\n"
    )
    print(json.dumps(selection, indent=2))
    print(json.dumps(runtime, indent=2))


if __name__ == "__main__":
    main()
