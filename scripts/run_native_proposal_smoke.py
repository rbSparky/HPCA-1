#!/usr/bin/env python3
"""Reproduce frozen seed-23 inference on the authoritative fixed17 state."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
import time
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flowadvantage.morpher_adapter.native_beam_mapper import ScheduleHorizon
from flowadvantage.morpher_adapter.native_mapper import (
    NativeMappingState,
    NativeMorpherProblem,
)
from flowadvantage.morpher_adapter.native_proposal import (
    FrozenFlowAdvantageNativeProposalScorer,
    NativeRelaxationParentContextProvider,
    compatibility_report,
)
from flowadvantage.morpher_adapter.native_relaxation import (
    NativeRelaxationConfig,
    NativeRelaxationSolver,
)


DEFAULT_INPUT = (
    ROOT
    / "results/revision_v5b/reference_mappings/array_add"
    / "A0_hycube4x4_standard_fixed17_smoke"
)
DEFAULT_OUTPUT = (
    ROOT / "results/revision_v5b/raw/native_proposal_fixed17_smoke.json"
)


def _partial_state(problem, mapping):
    excluded = problem.operation_order()[-1]
    excluded_id = int(problem.nodes[excluded]["dfg_node_id"])
    placed = {
        operation["native_node_key"]
        for operation in mapping["operations"]
        if operation["native_node_key"] != excluded
    }
    partial = dict(mapping)
    partial["operations"] = [
        operation
        for operation in mapping["operations"]
        if operation["native_node_key"] in placed
    ]
    partial["routes"] = [
        route
        for route in mapping["routes"]
        if route["source_node_key"] in placed
        and route["destination_node_key"] in placed
    ]
    partial["port_state"] = []
    for record in mapping["port_state"]:
        signals = [
            signal
            for signal in record["signals"]
            if signal["source_node_key"] in placed
            and int(signal["destination_node"]) != excluded_id
        ]
        if signals:
            partial["port_state"].append(
                {
                    "native_port_id": record["native_port_id"],
                    "signals": signals,
                }
            )
    return excluded, NativeMappingState.from_mapping(problem, partial)


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=ROOT / "results/revision_v5b/cache/native_relaxation",
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    dfg, mrrg, mapping = (
        json.loads((args.input / name).read_text())
        for name in ("dfg.json", "mrrg.json", "mapping.json")
    )
    problem = NativeMorpherProblem(dfg, mrrg)
    operation, state = _partial_state(problem, mapping)
    earliest, latest = ScheduleHorizon().bounds(problem, operation)
    actions = []
    for placement in problem.placement_candidates(
        operation,
        earliest_latency=earliest,
        latest_latency=latest,
        state=state,
    ):
        actions.extend(
            problem.actions_for_placement(
                placement,
                state,
                k_paths=4,
                max_route_combinations=4,
                max_route_expansions=200_000,
            )
        )
    if not actions:
        raise RuntimeError("fixed17 smoke produced no legal native action")

    solver = NativeRelaxationSolver(
        NativeRelaxationConfig(max_solve_seconds=120),
        cache_dir=args.cache_dir,
    )
    provider = NativeRelaxationParentContextProvider(solver)
    scorer = FrozenFlowAdvantageNativeProposalScorer(
        provider, device=args.device
    )
    start = time.perf_counter()
    first = np.asarray(
        scorer.score_actions(problem, state, actions), dtype=np.float64
    )
    first_timing = scorer.last_timing
    second = np.asarray(
        scorer.score_actions(problem, state, actions), dtype=np.float64
    )
    second_timing = scorer.last_timing
    if not np.array_equal(first, second):
        raise RuntimeError("fixed17 frozen inference is not bitwise deterministic")
    result = provider.last_result
    if result is None or not result.feasible:
        raise RuntimeError("fixed17 parent relaxation did not pass solver health")
    payload = {
        "schema": "flowadvantage_native_proposal_smoke_v1",
        "input": str(args.input.relative_to(ROOT)),
        "operation_key": operation,
        "actions": len(actions),
        "scores": first.tolist(),
        "selected_action_index": int(np.argmin(first)),
        "selected_action_key": repr(actions[int(np.argmin(first))].stable_key()),
        "deterministic_repeat": bool(np.array_equal(first, second)),
        "compatibility": compatibility_report(problem).__dict__,
        "checkpoint_path": str(scorer.checkpoint_path.relative_to(ROOT)),
        "checkpoint_sha256": scorer.checkpoint_sha256,
        "feature_schema_hash": scorer.feature_schema_hash,
        "device": str(scorer.device),
        "parent_relaxation": result.health_dict(),
        "first_inference_timing": first_timing.__dict__,
        "warm_inference_timing": second_timing.__dict__,
        "total_wall_seconds": time.perf_counter() - start,
    }
    _atomic_json(args.output, payload)
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
