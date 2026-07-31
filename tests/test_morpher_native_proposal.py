"""Frozen seed-23 FlowAdvantage inference on the authoritative fixed17 dump."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from flowadvantage.morpher_adapter.native_beam_mapper import (
    NativeActionScorer,
    ScheduleHorizon,
)
from flowadvantage.morpher_adapter.native_mapper import (
    NativeMappingState,
    NativeMorpherProblem,
)
from flowadvantage.morpher_adapter.native_proposal import (
    FrozenFlowAdvantageNativeProposalScorer,
    NativeDualLinearActionScorer,
    NativeParentRelaxationContext,
    NativeProposalCompatibilityError,
    NativeRelaxationParentContextProvider,
    StaticRootParentContextProvider,
    SELECTED_CHECKPOINT_SHA256,
    compatibility_report,
    native_state_hash,
)


ROOT = Path(__file__).resolve().parents[1]
FIXED17 = (
    ROOT
    / "results/revision_v5b/reference_mappings/array_add"
    / "A0_hycube4x4_standard_fixed17_smoke"
)


class _DeterministicParentProvider:
    """Numerically valid parent fixture for testing the inference boundary.

    This is not reported as experimental relaxation evidence.  Production
    mapping supplies the values from ``NativeRelaxationSolver``.
    """

    def parent_context(self, problem, state):
        route_duals = {
            resource_id: (
                int.from_bytes(resource_id.encode()[:2], "little") % 7
            ) * 1e-3
            for resource_id in problem.resources
        }
        compute_duals = {
            resource_id: (
                int.from_bytes(resource_id.encode()[-2:], "little") % 5
            ) * 1e-3
            for resource_id in problem.dp_by_id
        }
        return NativeParentRelaxationContext(
            state_hash=native_state_hash(state),
            architecture_hash=str(problem.mrrg["architecture_hash"]),
            dfg_hash=str(problem.dfg["dfg_hash"]),
            ii=problem.ii,
            objective=12.0,
            routing_duals=route_duals,
            compute_duals=compute_duals,
            capacity_slacks={
                resource_id: float(
                    resource_id not in state.resource_signals
                    or not state.resource_signals[resource_id]
                )
                for resource_id in problem.resources
            },
            fractional_flow_concentration=0.125,
            solver_status="test_exact_context",
        )


@pytest.fixture(scope="module")
def native_batch():
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
    state = NativeMappingState.from_mapping(problem, partial)
    earliest, latest = ScheduleHorizon().bounds(problem, excluded)
    actions = []
    for placement in problem.placement_candidates(
        excluded,
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
    assert actions
    return problem, state, tuple(actions[:16])


@pytest.fixture(scope="module")
def scorer():
    return FrozenFlowAdvantageNativeProposalScorer(
        _DeterministicParentProvider(),
        device="cpu",
    )


def test_fixed17_capability_projection_is_compatible(native_batch):
    problem, _, _ = native_batch
    report = compatibility_report(problem)
    assert report.compatible
    assert report.total_nodes == 20
    assert report.unsupported_compute_nodes == 0
    assert report.fixed_terminal_nodes >= 6
    assert report.arithmetic_compute_nodes >= 10


def test_unknown_compute_fraction_above_ten_percent_is_rejected():
    problem = SimpleNamespace(
        nodes={
            str(index): {"opcode": "STATEFUL" if index < 2 else "ADD"}
            for index in range(10)
        }
    )
    report = compatibility_report(problem)
    assert not report.compatible
    assert report.unsupported_fraction == pytest.approx(0.2)


def test_checkpoint_hash_and_schema_are_frozen(scorer):
    assert scorer.checkpoint_sha256 == SELECTED_CHECKPOINT_SHA256
    assert scorer.model.include_duals
    assert len(scorer.feature_names) == 62
    assert scorer.feature_schema_hash
    assert scorer.residual_scale > 0


def test_wrong_checkpoint_hash_fails_closed():
    with pytest.raises(
        NativeProposalCompatibilityError, match="SHA-256 mismatch"
    ):
        FrozenFlowAdvantageNativeProposalScorer(
            _DeterministicParentProvider(),
            checkpoint_sha256="0" * 64,
            device="cpu",
        )


def test_selected_hybrid_refuses_missing_parent_provider():
    with pytest.raises(TypeError, match="NativeParentContextProvider"):
        FrozenFlowAdvantageNativeProposalScorer(object(), device="cpu")


def test_parent_context_rejects_stale_state(native_batch):
    problem, state, _ = native_batch
    context = _DeterministicParentProvider().parent_context(problem, state)
    with pytest.raises(
        NativeProposalCompatibilityError, match="state hash"
    ):
        replace(context, state_hash="stale").validate(problem, state)


def test_native_relaxation_result_adapter_preserves_exact_parent_fields(
    native_batch,
):
    problem, state, _ = native_batch
    result = SimpleNamespace(
        feasible=True,
        status="optimal",
        error="",
        objective=3.25,
        routing_resource_duals={rid: 0.0 for rid in problem.resources},
        compute_duals={rid: 0.0 for rid in problem.dp_by_id},
        capacity_slacks={rid: 1.0 for rid in problem.resources},
        fractional_flow_concentration=0.375,
    )

    class Solver:
        def solve(self, checked_problem, checked_state):
            assert checked_problem is problem
            assert checked_state is state
            return result

    provider = NativeRelaxationParentContextProvider(Solver())
    context = provider.parent_context(problem, state)
    assert context.objective == 3.25
    assert context.fractional_flow_concentration == 0.375
    assert context.routing_duals == result.routing_resource_duals
    assert context.compute_duals == result.compute_duals
    assert context.capacity_slacks == result.capacity_slacks
    assert provider.calls == 1
    assert provider.cache_hits == 0
    assert provider.request_wall_seconds > 0.0
    assert provider.cache_read_seconds == 0.0


def test_static_root_parent_solves_once_and_rehashes_successor(native_batch):
    problem, state, _ = native_batch
    result = SimpleNamespace(
        feasible=True,
        status="optimal",
        error="",
        objective=7.0,
        routing_resource_duals={rid: 0.0 for rid in problem.resources},
        compute_duals={rid: 0.0 for rid in problem.dp_by_id},
        capacity_slacks={rid: 1.0 for rid in problem.resources},
        fractional_flow_concentration=0.2,
        solve_seconds=0.25,
        canonicalization_seconds=0.1,
    )

    class Solver:
        def __init__(self):
            self.calls = 0

        def solve(self, checked_problem, checked_state):
            assert checked_problem is problem
            assert not checked_state.placements
            self.calls += 1
            return result

    solver = Solver()
    provider = StaticRootParentContextProvider(solver, problem)
    first = provider.parent_context(problem, state)
    second = provider.parent_context(problem, state)
    assert solver.calls == 1
    assert provider.calls == 2
    assert provider.cache_hits == 1
    assert first.schema == "flowadvantage_native_static_root_parent_v1"
    assert first.state_hash == native_state_hash(state)
    assert second.semantic_hash == first.semantic_hash


def test_fixed17_scores_are_deterministic_finite_and_batched(
    native_batch, scorer
):
    problem, state, actions = native_batch
    first = np.asarray(scorer.score_actions(problem, state, actions))
    cold_timing = scorer.last_timing
    second = np.asarray(scorer.score_actions(problem, state, actions))
    warm_timing = scorer.last_timing
    assert np.array_equal(first, second)
    assert np.isfinite(first).all()
    assert first.shape == (len(actions),)
    assert cold_timing is not None and not cold_timing.state_cache_hit
    assert warm_timing is not None and warm_timing.state_cache_hit
    assert warm_timing.dfg_encoding_seconds == 0
    assert warm_timing.mrrg_encoding_seconds == 0
    assert warm_timing.actions == len(actions)
    # This is a generous operational smoke bound, not a paper result.
    assert cold_timing.total_seconds < 10.0
    assert warm_timing.total_seconds < 2.0


def test_native_proposal_satisfies_mapper_scorer_protocol(scorer):
    assert isinstance(scorer, NativeActionScorer)


def test_dual_linear_scores_exact_native_actions_without_model(native_batch):
    problem, state, actions = native_batch
    provider = _DeterministicParentProvider()
    scorer = NativeDualLinearActionScorer(provider)
    values = np.asarray(scorer.score_actions(problem, state, actions))
    assert isinstance(scorer, NativeActionScorer)
    assert values.shape == (len(actions),)
    assert np.isfinite(values).all()
    for action, value in zip(actions, values):
        immediate = sum(max(0, len(route.resource_ids) - 1) for route in action.routes)
        consumed = {(route.source_key, resource_id)
                    for route in action.routes
                    for resource_id in route.resource_ids}
        expected = immediate + sum(
            provider.parent_context(problem, state).routing_duals[resource_id]
            for _, resource_id in consumed
        ) + provider.parent_context(problem, state).compute_duals[action.placement.dp_id]
        assert value == pytest.approx(expected)


def test_mixed_operation_batch_is_rejected(native_batch, scorer):
    problem, state, actions = native_batch
    other = actions[0]
    mutated = replace(
        other,
        placement=replace(other.placement, node_key=problem.operation_order()[0]),
    )
    with pytest.raises(ValueError, match="one operation"):
        scorer.score_actions(problem, state, (actions[0], mutated))
