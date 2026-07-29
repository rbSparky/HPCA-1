"""Numerical tests for the exact port-level Morpher residual relaxation."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from flowadvantage.morpher_adapter.native_mapper import (
    NativeMappingState,
    NativeMorpherProblem,
)
from flowadvantage.morpher_adapter.native_relaxation import (
    NativeExactChildEvaluator,
    NativeRelaxationConfig,
    NativeRelaxationSolver,
    _native_failure,
)


ROOT = Path(__file__).resolve().parents[1]
FIXED17 = (
    ROOT
    / "results/revision_v5b/reference_mappings/array_add"
    / "A0_hycube4x4_standard_fixed17_smoke"
)


def test_native_relaxation_requires_explicit_positive_solver_threads():
    assert NativeRelaxationConfig().max_threads == 1
    with pytest.raises(ValueError, match="max_threads"):
        NativeRelaxationConfig(max_threads=0)


@pytest.fixture(scope="module")
def fixed17_residual():
    dfg, mrrg, mapping = (
        json.loads((FIXED17 / name).read_text())
        for name in ("dfg.json", "mrrg.json", "mapping.json")
    )
    problem = NativeMorpherProblem(dfg, mrrg)
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
    return (
        problem,
        mapping,
        excluded,
        NativeMappingState.from_mapping(problem, partial),
    )


@pytest.fixture(scope="module")
def solved_fixed17_residual(fixed17_residual, tmp_path_factory):
    problem, _, _, state = fixed17_residual
    solver = NativeRelaxationSolver(
        NativeRelaxationConfig(max_solve_seconds=120.0),
        cache_dir=tmp_path_factory.mktemp("native-relaxation-cache"),
    )
    return solver, solver.solve(problem, state)


def test_native_relaxation_uses_complete_exported_mrrg(
    fixed17_residual, solved_fixed17_residual
):
    problem, _, _, _ = fixed17_residual
    _, result = solved_fixed17_residual
    unique_edges = {
        (str(edge["src"]), str(edge["dst"]))
        for edge in problem.mrrg["edges"]
    }
    assert result.remaining_operations == 1
    assert result.remaining_dependencies == 2
    assert result.full_flow_variables == 2 * len(unique_edges)
    assert result.flow_variables < result.full_flow_variables
    assert result.flow_edge_reduction_fraction > 0.5
    assert set(result.routing_resource_duals) == set(problem.resources)
    assert set(result.capacity_slacks) == set(problem.resources)
    assert len(result.routing_edge_duals) == len(unique_edges)


def test_native_relaxation_solver_health(solved_fixed17_residual):
    _, result = solved_fixed17_residual
    assert result.status == "optimal"
    assert result.feasible
    assert math.isfinite(result.objective)
    assert result.objective > 0
    assert result.assignment_residual <= 1e-7
    assert result.flow_residual <= 1e-7
    assert result.compute_capacity_violation <= 1e-7
    assert result.routing_capacity_violation <= 1e-7
    assert result.min_variable >= -1e-7
    assert result.max_variable <= 1.0 + 1e-7
    assert all(value >= 0.0 for value in result.routing_resource_duals.values())
    assert all(value >= 0.0 for value in result.routing_edge_duals.values())
    assert all(value >= 0.0 for value in result.compute_duals.values())
    assert all(value >= 0.0 for value in result.capacity_slacks.values())
    assert 0.0 < result.fractional_flow_concentration <= 1.0


def test_native_relaxation_cache_repeat_is_exact(
    fixed17_residual, solved_fixed17_residual
):
    problem, _, _, state = fixed17_residual
    solver, first = solved_fixed17_residual
    repeated = solver.solve(problem, state)
    assert repeated.cache_hit
    assert solver.solve_calls >= 2
    assert solver.cache_hits >= 1
    assert solver.cache_misses >= 1
    assert repeated.cache_key == first.cache_key
    assert repeated.objective == first.objective
    assert repeated.routing_resource_duals == first.routing_resource_duals
    assert repeated.routing_edge_duals == first.routing_edge_duals
    reloaded = NativeRelaxationSolver(
        solver.config, cache_dir=solver.cache_dir
    ).solve(problem, state)
    assert reloaded.cache_hit
    assert reloaded.objective == first.objective


def test_solver_failure_preserves_pruned_problem_dimensions():
    result = _native_failure(
        "user_limit",
        "CLARABEL",
        0.0,
        "failure-key",
        10,
        12,
        "documented limit",
        placement_variables=640,
        flow_variables=2_075,
        full_flow_variables=9_216,
    )
    assert not result.solved
    assert result.flow_variables == 2_075
    assert result.full_flow_variables == 9_216
    assert result.flow_edge_reduction_fraction == pytest.approx(
        1.0 - 2_075 / 9_216
    )


def test_reachable_edge_pruning_preserves_deep_fixed17_value(
    fixed17_residual, tmp_path
):
    problem, _, _, state = fixed17_residual
    full = NativeRelaxationSolver(
        NativeRelaxationConfig(reachable_edge_pruning=False),
        cache_dir=tmp_path / "full",
    ).solve(problem, state)
    pruned = NativeRelaxationSolver(
        NativeRelaxationConfig(reachable_edge_pruning=True),
        cache_dir=tmp_path / "pruned",
    ).solve(problem, state)
    assert full.feasible and pruned.feasible
    assert abs(full.objective - pruned.objective) / max(1.0, abs(full.objective)) <= 1e-6
    assert pruned.flow_variables < full.flow_variables
    assert pruned.flow_edge_reduction_fraction >= 0.5
    assert pruned.assignment_residual <= 1e-6
    assert pruned.flow_residual <= 1e-6
    assert pruned.routing_capacity_violation <= 1e-6


def test_exact_child_evaluator_produces_finite_native_action_ranking(
    fixed17_residual, tmp_path
):
    problem, mapping, excluded, state = fixed17_residual
    reference = next(
        operation
        for operation in mapping["operations"]
        if operation["native_node_key"] == excluded
    )
    actions = []
    for placement in problem.placement_candidates(
        excluded,
        earliest_latency=int(reference["latency"]),
        latest_latency=int(reference["latency"]),
        state=state,
    ):
        actions.extend(
            problem.actions_for_placement(
                placement,
                state,
                k_paths=2,
                max_route_combinations=2,
                max_route_expansions=100_000,
            )
        )
    assert len(actions) == 2
    evaluator = NativeExactChildEvaluator(
        NativeRelaxationSolver(cache_dir=tmp_path / "children")
    )
    scores = evaluator.evaluate_children(problem, state, actions)
    assert all(math.isfinite(score) for score in scores)
    assert len(set(scores)) == 2
    assert scores == tuple(record.q_rel for record in evaluator.last_evaluations)
    assert all(record.residual_feasible for record in evaluator.last_evaluations)
    assert all(record.solve_status == "optimal" for record in evaluator.last_evaluations)
    assert all(record.residual_objective == 0.0 for record in evaluator.last_evaluations)
    assert evaluator.total_evaluations == len(actions)
    assert evaluator.total_cache_hits == sum(
        record.cache_hit for record in evaluator.last_evaluations
    )
    assert evaluator.total_request_wall_seconds > 0.0
    assert evaluator.total_cache_read_seconds >= 0.0
    assert evaluator.total_solve_seconds >= 0.0
