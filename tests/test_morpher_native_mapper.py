"""Authoritative tests for the Morpher-native FlowAdvantage mapping core."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from flowadvantage.morpher_adapter.mapping_export import (
    export_flowadvantage_mapping,
)
from flowadvantage.morpher_adapter.native_mapper import (
    NativeContractError,
    NativeMappingState,
    NativeMorpherProblem,
    NativeSignal,
)
from flowadvantage.morpher_adapter.legality_bridge import validate_mapping


ROOT = Path(__file__).resolve().parents[1]
DUMP = (
    ROOT
    / "results/revision_v5b/reference_mappings/array_add"
    / "A0_full_pipeline_fixed11/native_dump"
)


@pytest.fixture(scope="module")
def native_documents():
    return tuple(
        json.loads((DUMP / name).read_text())
        for name in ("dfg.json", "mrrg.json", "mapping.json")
    )


@pytest.fixture(scope="module")
def problem(native_documents):
    dfg, mrrg, mapping = native_documents
    return NativeMorpherProblem(
        dfg, mrrg, reference_mapping=mapping
    )


def _native_placement(problem, mapping, node_key):
    state = NativeMappingState.from_mapping(problem, mapping)
    return state.placements[node_key]


def test_authoritative_problem_preserves_native_counts(problem):
    assert len(problem.nodes) == 20
    assert len(problem.dependencies) == 23
    # The native module walk contains repeated identical port records.  The
    # mapper indexes the 2,177 stable native IDs without losing any of the
    # 7,175 raw traversal records.
    assert len(problem.mrrg["resources"]) == 7175
    assert len(problem.resources) == 2177
    assert sum(map(len, problem.adjacency.values())) == 4656


def test_production_problem_rejects_legacy_dump_without_fu_latencies(
    native_documents,
):
    dfg, mrrg, _ = native_documents
    with pytest.raises(NativeContractError, match="operation-latency metadata"):
        NativeMorpherProblem(dfg, mrrg)


def test_native_operation_order_is_stable_and_complete(problem):
    first = problem.operation_order()
    second = problem.operation_order()
    assert first == second
    assert len(first) == len(problem.nodes)
    assert set(first) == set(problem.nodes)


def test_placement_candidates_are_exact_fu_dp_time_assignments(
    problem, native_documents
):
    mapping = native_documents[2]
    # ADD has native latency witnesses on both compute and memory FU classes
    # in this authoritative reference, so every supported target is exact.
    operation = next(
        value for value in mapping["operations"] if value["opcode"] == "ADD"
    )
    node_key = operation["native_node_key"]
    candidates = problem.placement_candidates(
        node_key,
        earliest_latency=operation["latency"],
        latest_latency=operation["latency"],
    )
    matching = [
        candidate
        for candidate in candidates
        if candidate.fu_id == operation["fu_id"]
        and candidate.pe_id == operation["pe_id"]
        and candidate.modulo_time == operation["modulo_time"]
    ]
    assert len(matching) == 1
    assert matching[0].dp_id.endswith(".DP0.DP0")


def test_placement_enumeration_requires_explicit_schedule_window(
    problem,
):
    with pytest.raises(TypeError):
        problem.placement_candidates(problem.operation_order()[0])  # type: ignore


def test_native_operation_latency_is_witnessed_by_reference(problem):
    assert problem.operation_latency("ADD", "compute") == 1
    assert problem.operation_latency("ADD", "memory") == 2
    assert problem.operation_latency("LOAD", "memory") == 2
    assert problem.operation_latency("STORE", "memory") == 2


def test_operand_specific_route_reaches_exact_datapath_input(
    problem, native_documents
):
    dfg, _, mapping = native_documents
    dependency = dfg["dependencies"][0]
    source_key = dependency["source_node_key"]
    destination_key = dependency["destination_node_key"]
    source = _native_placement(problem, mapping, source_key)
    destination = _native_placement(problem, mapping, destination_key)
    state = NativeMappingState(problem)
    assert state.place(source)
    assert state.place(destination)
    routes = problem.enumerate_routes(
        dependency, source, destination, state, k=4
    )
    assert routes
    route = routes[0]
    expected_native_prefix = tuple(mapping["routes"][0]["ordered_resource_ids"])
    assert route.resource_ids[: len(expected_native_prefix)] == expected_native_prefix
    assert route.resource_ids[-1].endswith(".DP0.P")
    assert route.resource_latencies[-1] == destination.latency


def test_route_uses_only_directed_native_edges(problem, native_documents):
    dfg, _, mapping = native_documents
    dependency = dfg["dependencies"][0]
    source = _native_placement(
        problem, mapping, dependency["source_node_key"]
    )
    destination = _native_placement(
        problem, mapping, dependency["destination_node_key"]
    )
    state = NativeMappingState(problem)
    assert state.place(source)
    assert state.place(destination)
    route = problem.enumerate_routes(
        dependency, source, destination, state, k=1
    )[0]
    assert all(
        right in problem.adjacency[left]
        for left, right in zip(route.resource_ids, route.resource_ids[1:])
    )


def test_port_capacity_distinguishes_broadcast_and_distinct_values(problem):
    port = next(
        rid
        for rid, record in problem.resources.items()
        if record.get("resource_type") == "port"
        and not record.get("allows_operand_mux")
    )
    state = NativeMappingState(problem)
    first = NativeSignal("source-a", 1, 0, 1)
    same_value_fanout = NativeSignal("source-a", 2, 0, 1)
    other_value = NativeSignal("source-b", 2, 0, 2)
    state.resource_signals[port].add(first)
    assert state.can_occupy(port, same_value_fanout)
    assert not state.can_occupy(port, other_value)


def test_operand_mux_accepts_distinct_sources(problem):
    port = next(
        rid
        for rid, record in problem.resources.items()
        if record.get("resource_type") == "port"
        and record.get("allows_operand_mux")
    )
    state = NativeMappingState(problem)
    state.resource_signals[port].add(NativeSignal("source-a", 1, 0, 1))
    assert state.can_occupy(
        port, NativeSignal("source-b", 2, 0, 2)
    )


def test_conflict_port_occupancy_is_enforced(problem):
    non_mux_ports = [
        rid
        for rid, record in problem.resources.items()
        if record.get("resource_type") == "port"
        and not record.get("allows_operand_mux")
    ]
    resource_id, conflicting_id = non_mux_ports[:2]
    # The checked-in fixed11 evidence predates conflict-set serialization.
    # Inject one declared native conflict relation into the authoritative graph
    # and restore it after exercising the exact production code path.
    old_left, old_right = (
        problem.conflicts[resource_id],
        problem.conflicts[conflicting_id],
    )
    try:
        problem.conflicts[resource_id] = frozenset((conflicting_id,))
        problem.conflicts[conflicting_id] = frozenset((resource_id,))
        state = NativeMappingState(problem)
        state.resource_signals[conflicting_id].add(
            NativeSignal("source-a", 1, 0, 1)
        )
        assert not state.can_occupy(
            resource_id, NativeSignal("source-b", 2, 0, 2)
        )
    finally:
        problem.conflicts[resource_id] = old_left
        problem.conflicts[conflicting_id] = old_right


def test_committed_route_populates_replayable_port_state(
    problem, native_documents
):
    dfg, _, mapping = native_documents
    dependency = dfg["dependencies"][0]
    source = _native_placement(
        problem, mapping, dependency["source_node_key"]
    )
    destination = _native_placement(
        problem, mapping, dependency["destination_node_key"]
    )
    state = NativeMappingState(problem)
    assert state.place(source)
    assert state.place(destination)
    route = problem.enumerate_routes(
        dependency, source, destination, state, k=1
    )[0]
    assert state.add_route(route)
    document = state.to_mapping()
    port_state = {
        record["native_port_id"]: record["signals"]
        for record in document["port_state"]
    }
    for resource_id in route.resource_ids:
        assert any(
            signal["source_node_key"] == route.source_key
            and signal["destination_node"] == route.destination_node
            for signal in port_state[resource_id]
        )
    export_flowadvantage_mapping(document)


def test_action_generation_commits_operand_specific_incoming_route(
    problem, native_documents
):
    dfg, _, mapping = native_documents
    dependency = dfg["dependencies"][0]
    source = _native_placement(
        problem, mapping, dependency["source_node_key"]
    )
    destination = _native_placement(
        problem, mapping, dependency["destination_node_key"]
    )
    state = NativeMappingState(problem)
    assert state.place(source)
    action = problem.action_for_placement(destination, state, k_paths=4)
    assert action is not None
    assert action.placement == destination
    assert len(action.routes) == 1
    assert action.routes[0].resource_ids[-1].endswith(".DP0.P")


def test_authoritative_mapping_semantic_roundtrip(
    problem, native_documents
):
    original = native_documents[2]
    state = NativeMappingState.from_mapping(problem, original)
    replay = state.to_mapping()
    original_placements = {
        operation["native_node_key"]: (
            operation["pe_id"],
            operation["fu_id"],
            operation["modulo_time"],
            operation["latency"],
        )
        for operation in original["operations"]
    }
    replay_placements = {
        operation["native_node_key"]: (
            operation["pe_id"],
            operation["fu_id"],
            operation["modulo_time"],
            operation["latency"],
        )
        for operation in replay["operations"]
    }
    original_routes = {
        route["edge_id"]: (
            tuple(route["ordered_resource_ids"]),
            tuple(route["ordered_resource_latencies"]),
        )
        for route in original["routes"]
    }
    replay_routes = {
        route["edge_id"]: (
            tuple(route["ordered_resource_ids"]),
            tuple(route["ordered_resource_latencies"]),
        )
        for route in replay["routes"]
    }
    assert replay["ii"] == original["ii"]
    assert replay_placements == original_placements
    assert replay_routes == original_routes
    assert replay["port_state"] == original["port_state"]
    legality = validate_mapping(replay, problem.dfg, problem.mrrg)
    assert legality["legal"], legality["violations"]


def test_imported_native_mapping_marks_every_datapath_occupied(
    problem, native_documents
):
    state = NativeMappingState.from_mapping(problem, native_documents[2])
    assert len(state.placements) == len(problem.nodes)
    assert len(state.dp_occupancy) == len(problem.nodes)
