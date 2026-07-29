"""Fixed17 native-contract smoke tests for deterministic Morpher beam search."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from flowadvantage.morpher_adapter.native_beam_mapper import (
    DeterministicNativeBeamMapper,
    LengthActionScorer,
    NativeBeamConfig,
    NativeActionScorer,
    ScheduleHorizon,
    TopKExactRerankScorer,
)
from flowadvantage.morpher_adapter.native_mapper import (
    NativeMappingState,
    NativeMorpherProblem,
)


ROOT = Path(__file__).resolve().parents[1]
FIXED17 = (
    ROOT
    / "results/revision_v5b/reference_mappings/array_add"
    / "A0_hycube4x4_standard_fixed17_smoke"
)


@pytest.fixture(scope="module")
def fixed17():
    assert FIXED17.exists(), "authoritative fixed17 smoke export is missing"
    dfg, mrrg, mapping = (
        json.loads((FIXED17 / name).read_text())
        for name in ("dfg.json", "mrrg.json", "mapping.json")
    )
    problem = NativeMorpherProblem(dfg, mrrg)
    return problem, mapping


def _one_operation_partial(problem, mapping):
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


def test_fixed17_has_complete_exact_fu_latency_metadata(fixed17):
    problem, _ = fixed17
    assert problem.fu_operation_latencies
    for dp in problem.dp_records:
        for opcode in dp["supported_operations"]:
            assert (dp["fu_id"], opcode) in problem.fu_operation_latencies


def test_asap_alap_horizon_contains_every_native_reference_placement(fixed17):
    problem, mapping = fixed17
    horizon = ScheduleHorizon(extra_ii_periods=1)
    for operation in mapping["operations"]:
        earliest, latest = horizon.bounds(
            problem, operation["native_node_key"]
        )
        assert earliest <= int(operation["latency"]) <= latest


def test_beam_mapper_completes_fixed17_partial_and_checks_legality(fixed17):
    problem, mapping = fixed17
    excluded, partial = _one_operation_partial(problem, mapping)
    events = []
    result = DeterministicNativeBeamMapper(
        problem,
        scorer=LengthActionScorer(),
        config=NativeBeamConfig(
            beam_width=4,
            k_paths=4,
            max_route_combinations_per_target=4,
            per_state_action_limit=8,
            max_expansions=20,
        ),
        progress_callback=events.append,
    ).map(partial)
    assert result.metrics.success
    assert result.metrics.legal
    assert result.metrics.termination == "DONE"
    assert result.mapping is not None
    assert excluded in result.state.placements
    assert result.metrics.generated_targets > 0
    assert result.metrics.generated_actions > 0
    assert [event.sequence for event in events] == list(range(len(events)))
    assert [event.stage for event in events] == [
        "START",
        "OPERATION_COMMITTED",
        "DONE",
    ]


def test_beam_mapper_is_bitwise_deterministic_on_fixed17_partial(fixed17):
    problem, mapping = fixed17
    _, partial = _one_operation_partial(problem, mapping)
    config = NativeBeamConfig(
        beam_width=2,
        k_paths=2,
        max_route_combinations_per_target=2,
        per_state_action_limit=4,
        max_expansions=20,
    )
    first = DeterministicNativeBeamMapper(
        problem, scorer=LengthActionScorer(), config=config
    ).map(partial)
    second = DeterministicNativeBeamMapper(
        problem, scorer=LengthActionScorer(), config=config
    ).map(partial)
    assert first.metrics.success == second.metrics.success
    assert first.state.stable_key() == second.state.stable_key()
    assert first.mapping == second.mapping
    # Wall time is intentionally excluded from deterministic equality.
    assert first.metrics.route_cost == second.metrics.route_cost
    assert first.metrics.expansions == second.metrics.expansions
    assert first.metrics.generated_actions == second.metrics.generated_actions


def test_mapper_requires_concrete_batch_scorer(fixed17):
    problem, _ = fixed17
    with pytest.raises(TypeError, match="concrete NativeActionScorer"):
        DeterministicNativeBeamMapper(problem, scorer=object())  # type: ignore


def test_topk_interface_refuses_missing_exact_child_evaluator():
    with pytest.raises(TypeError, match="child_evaluator"):
        TopKExactRerankScorer(
            proposal=LengthActionScorer(),
            child_evaluator=object(),  # type: ignore
            k=4,
        )


def test_length_scorer_satisfies_batch_protocol():
    assert isinstance(LengthActionScorer(), NativeActionScorer)
