#!/usr/bin/env python3
"""Benchmark native Clarabel/OSQP root solves without changing pilot outputs."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from flowadvantage.morpher_adapter.native_mapper import NativeMappingState, NativeMorpherProblem
from flowadvantage.morpher_adapter.native_relaxation import NativeRelaxationConfig, NativeRelaxationSolver


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dfg", type=Path, required=True)
    parser.add_argument("--mrrg", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-seconds", type=float, default=1800.0)
    args = parser.parse_args()
    dfg = json.loads(args.dfg.read_text(encoding="utf-8"))
    mrrg = json.loads(args.mrrg.read_text(encoding="utf-8"))
    problem = NativeMorpherProblem.from_native_documents(dfg, mrrg)
    state = NativeMappingState(problem)
    rows = []
    for primary, fallback in (("CLARABEL", "OSQP"), ("OSQP", "CLARABEL")):
        cache = args.cache / primary.lower()
        config = NativeRelaxationConfig(
            solver_primary=primary,
            solver_fallback=fallback,
            extra_ii_periods=0,
            fallback_extra_ii_periods=1,
            max_solve_seconds=args.max_seconds,
            max_threads=1,
            structure_cache_enabled=True,
        )
        start = time.perf_counter()
        result = NativeRelaxationSolver(config, cache_dir=cache).solve(problem, state)
        rows.append({"solver_primary": primary, "solver_used": result.solver, "status": result.status, "wall_seconds": time.perf_counter() - start, "canonicalization_seconds": result.canonicalization_seconds, "objective": result.objective, "feasible": result.feasible, "assignment_residual": result.assignment_residual, "flow_residual": result.flow_residual, "capacity_violation": max(result.compute_capacity_violation, result.routing_capacity_violation), "error": result.error})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"schema": "native_solver_benchmark_v1", "dfg": str(args.dfg), "mrrg": str(args.mrrg), "rows": rows}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(rows, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
