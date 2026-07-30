"""Exhaustive transition-equivalence checks for exact mapper symmetries.

Graph automorphism is necessary but not sufficient for quotienting a search
transition system.  In particular, the deterministic operation order, resource
semantics, action scores, and successor relation must all commute with the
claimed transform.  This module supplies the stricter, executable definition
used by the paper-review validation.  It intentionally does not weaken a
failed transform into an approximate symmetry.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import networkx as nx
import numpy as np

from .arch import Architecture, RoutingNode, verify_arch_permutation
from .dfg import DFG, verify_dfg_permutation
from .mapper import Action, action_score, check_legality, enumerate_actions
from .partial_state import PartialState, residual_graph
from .symmetry import (
    architecture_edge_map,
    preserves_total_operation_order,
    transform_state,
)


@dataclass(frozen=True)
class TransformRejection:
    dfg_transform: int
    architecture_transform: int
    reason: str


@dataclass(frozen=True)
class TransitionEquivalenceResult:
    case: str
    reachable_states: int
    terminal_states: int
    admitted_transforms: int
    rejected_transforms: int
    checked_state_transforms: int
    checked_actions: int
    action_bijection: bool
    score_equivariance: bool
    successor_equivariance: bool
    terminal_legality: bool
    terminal_cost: bool
    passed: bool
    error: str = ""


def _semantic_attrs(data: Mapping, ignored: Iterable[str] = ()) -> tuple:
    """Canonicalize every public semantic attribute, including nested values."""

    def freeze(value):
        if isinstance(value, Mapping):
            return tuple(sorted((str(k), freeze(v)) for k, v in value.items()))
        if isinstance(value, (list, tuple, set, frozenset)):
            return tuple(freeze(v) for v in value)
        return value

    ignored_set = set(ignored)
    return tuple(
        sorted(
            (str(key), freeze(value))
            for key, value in data.items()
            if key not in ignored_set
        )
    )


def verify_semantic_dfg_permutation(dfg: DFG, perm: Dict[int, int]) -> bool:
    """Verify all operation/dependency attributes, not just opcode and slot."""
    if not verify_dfg_permutation(dfg, perm):
        return False
    graph = dfg.graph
    for node in graph:
        if _semantic_attrs(graph.nodes[node]) != _semantic_attrs(graph.nodes[perm[node]]):
            return False
    for source, sink, data in graph.edges(data=True):
        if not graph.has_edge(perm[source], perm[sink]):
            return False
        if _semantic_attrs(data) != _semantic_attrs(
            graph.edges[perm[source], perm[sink]]
        ):
            return False
    return True


def verify_semantic_arch_permutation(
    arch: Architecture, perm: Dict[RoutingNode, RoutingNode]
) -> bool:
    """Verify the complete resource contract, including time and mutex metadata."""
    if not verify_arch_permutation(arch, perm):
        return False
    graph = arch.graph
    for node in graph:
        if _semantic_attrs(graph.nodes[node]) != _semantic_attrs(graph.nodes[perm[node]]):
            return False
    for source, sink, data in graph.edges(data=True):
        if not graph.has_edge(perm[source], perm[sink]):
            return False
        # Native edge indices and coordinate deltas transform covariantly; the
        # directed endpoint permutation above verifies them structurally.
        if _semantic_attrs(data, ("index", "dx", "dy")) != _semantic_attrs(
            graph.edges[perm[source], perm[sink]], ("index", "dx", "dy")
        ):
            return False
    return True


def admitted_joint_transforms(
    dfg: DFG, arch: Architecture
) -> Tuple[List[Tuple[Dict[int, int], Dict[RoutingNode, RoutingNode]]], List[TransformRejection]]:
    admitted = []
    rejected = []
    for dfg_index, dfg_perm in enumerate(dfg.automorphisms):
        for arch_index, arch_perm in enumerate(arch.symmetries):
            reason = ""
            if not verify_semantic_dfg_permutation(dfg, dfg_perm):
                reason = "dfg_semantics_not_preserved"
            elif not preserves_total_operation_order(dfg, dfg_perm):
                reason = "total_operation_order_not_preserved"
            elif not verify_semantic_arch_permutation(arch, arch_perm):
                reason = "architecture_semantics_not_preserved"
            if reason:
                rejected.append(TransformRejection(dfg_index, arch_index, reason))
            else:
                admitted.append((dfg_perm, arch_perm))
    return admitted, rejected


def _full_state_signature(state: PartialState) -> tuple:
    return (
        tuple(sorted(state.placements.items())),
        tuple(sorted(state.occupied_compute)),
        tuple(sorted(state.occupied_links)),
        tuple(sorted(state.routed_edges.items())),
        round(float(state.route_cost), 12),
    )


def _action_signature(action: Action) -> tuple:
    return (
        action.target,
        tuple(action.new_links),
        round(float(action.base_cost), 12),
        _full_state_signature(action.state),
    )


def _path_semantics_valid(
    dfg: DFG, arch: Architecture, state: PartialState, dependency: Tuple[int, int]
) -> bool:
    source, sink = dependency
    data = dfg.graph.edges[dependency]
    path = state.routed_edges.get(dependency)
    if path is None:
        return False
    distance = int(data.get("distance", 0))
    required_hops = data.get("required_hops")
    if required_hops is not None and len(path) != int(required_hops):
        return False
    # If a recurrence distance is supplied, bound the legal modulo traversal.
    # This admits the normal direct traversal and exactly ``distance`` wraps.
    if distance:
        source_t = state.placements[source][2]
        sink_t = state.placements[sink][2]
        minimum = (sink_t - source_t) % arch.ii
        if len(path) != minimum + distance * arch.ii:
            return False
    return True


def semantic_state_valid(dfg: DFG, arch: Architecture, state: PartialState) -> bool:
    """Check extra review cases not represented by the original basic checker."""
    mutex_owners = {}
    broadcast_sources = {}
    for op, resource in state.placements.items():
        op_data = dfg.graph.nodes[op]
        resource_data = arch.graph.nodes[resource]
        if op_data.get("memory_role", "none") != "none":
            if resource_data.get("memory_role", "none") != op_data["memory_role"]:
                return False
        asap = int(op_data.get("asap", op_data.get("level", 0)))
        alap = int(op_data.get("alap", asap))
        absolute = int(op_data.get("absolute_schedule", op_data.get("level", 0)))
        if not asap <= absolute <= alap:
            return False
    for dependency, path in state.routed_edges.items():
        if not _path_semantics_valid(dfg, arch, state, dependency):
            return False
        for edge_id in path:
            edge = arch.edge_list[edge_id]
            mutex = arch.graph.edges[edge].get("mutex_group")
            if mutex is not None:
                owner = mutex_owners.setdefault(mutex, dependency)
                if owner != dependency:
                    return False
            broadcast = arch.graph.edges[edge].get("broadcast_group")
            if broadcast is not None:
                producer = broadcast_sources.setdefault(broadcast, dependency[0])
                if producer != dependency[0]:
                    return False
    return True


def enumerate_semantic_actions(
    dfg: DFG,
    arch: Architecture,
    state: PartialState,
    op: int,
    k_paths: int = 8,
) -> List[Action]:
    # A k-shortest prefix is generally not closed under a symmetry when the
    # kth boundary contains tied paths.  Exact validation must enumerate the
    # complete finite simple-path action universe.
    slot = dfg.graph.nodes[op]["slot"]
    targets = [
        resource for resource in arch.compute_slots
        if resource[2] == slot and resource not in state.occupied_compute
    ]
    incoming = sorted(
        predecessor for predecessor in dfg.graph.predecessors(op)
        if predecessor in state.placements
    )
    actions = []
    for target in targets:
        initial = state.copy()
        initial.placements[op] = target
        initial.occupied_compute.add(target)
        partials = [(initial, tuple(), 0.0)]
        for predecessor in incoming:
            next_partials = []
            for candidate, links, cost in partials:
                graph = residual_graph(arch, candidate.occupied_links)
                paths = list(
                    nx.all_simple_paths(
                        graph,
                        candidate.placements[predecessor],
                        target,
                        cutoff=len(graph) - 1,
                    )
                )
                edge_paths = sorted(
                    tuple(arch.edge_index[(a, b)] for a, b in zip(path, path[1:]))
                    for path in paths
                )
                for path in edge_paths:
                    updated = candidate.copy()
                    updated.routed_edges[(predecessor, op)] = path
                    updated.occupied_links.update(path)
                    path_cost = sum(
                        arch.graph.edges[arch.edge_list[edge]]["base_cost"]
                        for edge in path
                    )
                    updated.route_cost += path_cost
                    next_partials.append((updated, links + path, cost + path_cost))
            partials = next_partials
        actions.extend(Action(target, candidate, links, cost) for candidate, links, cost in partials)
    actions.sort(key=lambda action: (action.base_cost, action.target, action.new_links))
    filtered = []
    operation = dfg.graph.nodes[op]
    for action in actions:
        resource = arch.graph.nodes[action.target]
        if operation["op"] not in resource["supports"]:
            continue
        if operation.get("memory_role", "none") != "none" and resource.get(
            "memory_role", "none"
        ) != operation["memory_role"]:
            continue
        if semantic_state_valid(dfg, arch, action.state):
            filtered.append(action)
    return filtered


def enumerate_reachable_states(
    dfg: DFG, arch: Architecture, k_paths: int = 8
) -> List[PartialState]:
    """Enumerate the complete bounded transition system without beam pruning."""
    frontier = [PartialState()]
    reachable = [PartialState()]
    for op in dfg.order:
        successors = {}
        for state in frontier:
            for action in enumerate_semantic_actions(dfg, arch, state, op, k_paths):
                successors[_full_state_signature(action.state)] = action.state
        frontier = [successors[key] for key in sorted(successors)]
        reachable.extend(frontier)
        if not frontier:
            break
    return reachable


def invariant_prices(arch: Architecture) -> np.ndarray:
    """Prices derived only from attributes preserved by admitted transforms."""
    kind_value = {"wait": 0.25, "cardinal": 0.5, "diagonal": 0.75, "wrap": 1.0}
    return np.asarray(
        [
            float(arch.graph.edges[edge]["base_cost"])
            + kind_value.get(arch.graph.edges[edge].get("kind", ""), 0.0)
            for edge in arch.edge_list
        ],
        dtype=float,
    )


def validate_transition_equivalence(
    case: str, dfg: DFG, arch: Architecture, k_paths: int = 8
) -> Tuple[TransitionEquivalenceResult, List[TransformRejection]]:
    admitted, rejected = admitted_joint_transforms(dfg, arch)
    states = enumerate_reachable_states(dfg, arch, k_paths=k_paths)
    state_sets = {}
    for state in states:
        state_sets.setdefault(len(state.placements), set()).add(_full_state_signature(state))
    prices = invariant_prices(arch)
    checked_state_transforms = checked_actions = terminal_states = 0
    action_bijection = score_equivariance = successor_equivariance = True
    terminal_legality = terminal_cost = True
    error = ""
    try:
        for state in states:
            depth = len(state.placements)
            terminal = depth == len(dfg.graph)
            if terminal:
                terminal_states += 1
            for dfg_perm, arch_perm in admitted:
                checked_state_transforms += 1
                edge_map = architecture_edge_map(arch, arch_perm)
                transformed = transform_state(
                    state, arch, dfg_perm, arch_perm, edge_map=edge_map
                )
                if _full_state_signature(transformed) not in state_sets[depth]:
                    successor_equivariance = False
                    raise AssertionError("transformed reachable state is unreachable")
                if terminal:
                    legal_a = check_legality(dfg, arch, state)[0] and semantic_state_valid(
                        dfg, arch, state
                    )
                    legal_b = check_legality(dfg, arch, transformed)[0] and semantic_state_valid(
                        dfg, arch, transformed
                    )
                    if legal_a != legal_b:
                        terminal_legality = False
                        raise AssertionError("terminal legality is not invariant")
                    if not math.isclose(
                        state.route_cost, transformed.route_cost, abs_tol=1e-10
                    ):
                        terminal_cost = False
                        raise AssertionError("terminal cost is not invariant")
                    continue
                op = dfg.order[depth]
                transformed_op = dfg_perm[op]
                actions = enumerate_semantic_actions(dfg, arch, state, op, k_paths)
                target_actions = enumerate_semantic_actions(
                    dfg, arch, transformed, transformed_op, k_paths
                )
                target_by_signature = {
                    _action_signature(action): action for action in target_actions
                }
                images = []
                for action in actions:
                    checked_actions += 1
                    image_state = transform_state(
                        action.state, arch, dfg_perm, arch_perm, edge_map=edge_map
                    )
                    image = Action(
                        arch_perm[action.target],
                        image_state,
                        tuple(edge_map[edge] for edge in action.new_links),
                        action.base_cost,
                    )
                    signature = _action_signature(image)
                    images.append(signature)
                    counterpart = target_by_signature.get(signature)
                    if counterpart is None:
                        successor_equivariance = False
                        raise AssertionError("action successor has no transformed counterpart")
                    score = action_score(action, prices, price_weight=0.7)
                    transformed_prices = np.empty_like(prices)
                    for edge_id, mapped_id in edge_map.items():
                        transformed_prices[mapped_id] = prices[edge_id]
                    image_score = action_score(image, transformed_prices, price_weight=0.7)
                    if not math.isclose(score, image_score, abs_tol=1e-10):
                        score_equivariance = False
                        raise AssertionError("action score is not equivariant")
                if len(images) != len(set(images)) or set(images) != set(target_by_signature):
                    action_bijection = False
                    raise AssertionError("action transform is not a bijection")
    except AssertionError as exc:
        error = str(exc)
    passed = all(
        (
            bool(admitted),
            action_bijection,
            score_equivariance,
            successor_equivariance,
            terminal_legality,
            terminal_cost,
            not error,
        )
    )
    return (
        TransitionEquivalenceResult(
            case=case,
            reachable_states=len(states),
            terminal_states=terminal_states,
            admitted_transforms=len(admitted),
            rejected_transforms=len(rejected),
            checked_state_transforms=checked_state_transforms,
            checked_actions=checked_actions,
            action_bijection=action_bijection,
            score_equivariance=score_equivariance,
            successor_equivariance=successor_equivariance,
            terminal_legality=terminal_legality,
            terminal_cost=terminal_cost,
            passed=passed,
            error=error,
        ),
        rejected,
    )
