"""Deterministic beam mapper and independent legality checker."""

from __future__ import annotations

import itertools
import math
import time
from dataclasses import asdict, dataclass
from typing import Callable, Iterable, List, Optional, Tuple

import networkx as nx
import numpy as np

from .arch import Architecture, RoutingNode
from .dfg import DFG
from .partial_state import PartialState, residual_graph
from .symmetry import canonical_key

_PATH_CACHE = {}


@dataclass
class Action:
    target: RoutingNode
    state: PartialState
    new_links: Tuple[int, ...]
    base_cost: float


@dataclass
class MappingResult:
    success: bool
    legal: bool
    best_cost: float
    mapped_operations: int
    expansions: int
    generated_successors: int
    duplicate_states: int
    quotient_merges: int
    canonicalization_ms: float
    routing_attempts: int
    failed_routing_attempts: int
    raw_candidate_actions: int
    action_orbit_representatives: int
    action_orbit_actions_removed: int
    gnn_evaluations_avoided: int
    routing_attempts_avoided: int
    action_orbit_ms: float
    raw_candidate_targets: int
    target_orbit_representatives: int
    runtime_ms: float
    timeout: bool
    reason: str
    final_state: Optional[PartialState] = None

    def row(self) -> dict:
        row = asdict(self)
        row.pop("final_state")
        return row


def _k_paths(
    arch: Architecture,
    source: RoutingNode,
    sink: RoutingNode,
    occupied: Iterable[int],
    k: int,
) -> List[Tuple[int, ...]]:
    key = (arch.name, source, sink, tuple(sorted(occupied)), k)
    cached = _PATH_CACHE.get(key)
    if cached is not None:
        return cached
    graph = residual_graph(arch, occupied)
    try:
        generator = nx.shortest_simple_paths(graph, source, sink, weight="base_cost")
        paths = list(itertools.islice(generator, k))
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        _PATH_CACHE[key] = []
        return []
    result = [
        tuple(arch.edge_index[(u, v)] for u, v in zip(path, path[1:]))
        for path in paths
    ]
    result = sorted(
        result,
        key=lambda p: (
            sum(arch.graph.edges[arch.edge_list[e]]["base_cost"] for e in p),
            p,
        ),
    )
    _PATH_CACHE[key] = result
    return result


def enumerate_actions(
    dfg: DFG,
    arch: Architecture,
    state: PartialState,
    op: int,
    k_paths: int = 4,
    targets_override: Optional[Iterable[RoutingNode]] = None,
) -> Tuple[List[Action], int, int]:
    slot = dfg.graph.nodes[op]["slot"]
    targets = [
        r
        for r in arch.compute_slots
        if r[2] == slot and r not in state.occupied_compute
    ]
    if targets_override is not None:
        allowed_targets = set(targets_override)
        targets = [target for target in targets if target in allowed_targets]
    incoming = sorted(p for p in dfg.graph.predecessors(op) if p in state.placements)
    actions, attempts, failures = [], 0, 0
    for target in targets:
        initial = state.copy()
        initial.placements[op] = target
        initial.occupied_compute.add(target)
        partials = [(initial, tuple(), 0.0)]
        for pred in incoming:
            next_partials = []
            for candidate, links, cost in partials:
                attempts += 1
                paths = _k_paths(
                    arch,
                    candidate.placements[pred],
                    target,
                    candidate.occupied_links,
                    k_paths,
                )
                if not paths:
                    failures += 1
                for path in paths:
                    updated = candidate.copy()
                    updated.routed_edges[(pred, op)] = path
                    updated.occupied_links.update(path)
                    path_cost = sum(
                        arch.graph.edges[arch.edge_list[e]]["base_cost"] for e in path
                    )
                    updated.route_cost += path_cost
                    next_partials.append((updated, links + path, cost + path_cost))
            partials = next_partials
            if not partials:
                break
        for candidate, links, cost in partials:
            actions.append(Action(target, candidate, links, cost))
    actions.sort(key=lambda a: (a.base_cost, a.target, a.new_links))
    return actions, attempts, failures


def action_score(
    action: Action,
    prices: Optional[np.ndarray],
    price_weight: float,
    price_mode: str = "sum",
) -> float:
    values = (
        np.asarray(prices)[list(action.new_links)]
        if prices is not None and action.new_links
        else np.zeros(0)
    )
    if not len(values):
        scarcity = 0.0
    elif price_mode == "sum":
        scarcity = float(values.sum())
    elif price_mode == "max":
        scarcity = float(values.max())
    elif price_mode == "top2":
        scarcity = float(np.sort(values)[-2:].sum())
    else:
        raise ValueError(f"unknown price aggregation mode {price_mode}")
    return action.base_cost + price_weight * scarcity


def map_dfg(
    dfg: DFG,
    arch: Architecture,
    initial: PartialState,
    prices: Optional[np.ndarray] = None,
    beam_width: int = 32,
    action_limit: int = 6,
    k_paths: int = 4,
    max_expansions: int = 5000,
    timeout_seconds: float = 2.0,
    price_weight: float = 2.0,
    price_mode: str = "sum",
    quotient: bool = False,
    joint_quotient: bool = False,
    dynamic_refresh_every: int = 0,
    action_scorer: Optional[
        Callable[[DFG, Architecture, PartialState, int, List[Action]], Iterable[float]]
    ] = None,
    action_preselector: Optional[
        Callable[[DFG, Architecture, PartialState, int, List[Action]], List[Action]]
    ] = None,
    action_orbit_reduction: bool = False,
    action_value_is_cost_to_go: bool = False,
) -> MappingResult:
    start = time.perf_counter()
    beam = [(0.0, initial.copy())]
    expansions = generated = duplicates = merges = attempts = failures = 0
    canon_ms = 0.0
    raw_candidate_actions = orbit_representatives = orbit_removed = 0
    orbit_ms = 0.0
    raw_candidate_targets = target_orbit_representatives = 0
    timeout = False
    reason = ""
    current_prices = prices
    remaining_order = [v for v in dfg.order if v not in initial.placements]
    for depth, op in enumerate(remaining_order):
        if dynamic_refresh_every and depth > 0 and depth % dynamic_refresh_every == 0:
            # Small failure-driven ablation: refresh on the deterministic best
            # beam representative, then use that conditioned signal for this
            # level. Static behavior remains the default.
            from .relaxation import solve_relaxation

            refreshed = solve_relaxation(dfg, arch, beam[0][1])
            if refreshed.solved:
                current_prices = refreshed.normalized_prices
        successors = {}
        quotient_candidates = []
        for parent_score, state in beam:
            if time.perf_counter() - start > timeout_seconds:
                timeout, reason = True, "timeout"
                break
            if expansions >= max_expansions:
                reason = "max_expansions"
                break
            expansions += 1
            targets_override = None
            if action_orbit_reduction:
                from .action_lookahead import stabilizer_target_orbits

                targets_override, target_info = stabilizer_target_orbits(
                    dfg, arch, state, op
                )
                raw_candidate_targets += target_info["raw_targets"]
                target_orbit_representatives += target_info[
                    "target_orbit_representatives"
                ]
                orbit_ms += target_info["target_orbit_ms"]
            actions, used_attempts, used_failures = enumerate_actions(
                dfg, arch, state, op, k_paths, targets_override=targets_override
            )
            attempts += used_attempts
            failures += used_failures
            raw_candidate_actions += len(actions)
            if action_orbit_reduction and actions:
                from .action_lookahead import stabilizer_action_orbits

                actions, orbit_info = stabilizer_action_orbits(
                    dfg, arch, state, op, actions
                )
                orbit_ms += orbit_info["action_orbit_ms"]
                orbit_removed += orbit_info["actions_removed"]
            orbit_representatives += len(actions)
            if action_preselector is not None and actions:
                actions = action_preselector(dfg, arch, state, op, actions)
            if action_scorer is not None and actions:
                action_values = list(action_scorer(dfg, arch, state, op, actions))
                if len(action_values) != len(actions):
                    raise ValueError("action scorer returned wrong number of scores")
            else:
                action_values = [
                    action_score(
                        action, current_prices, price_weight, price_mode
                    )
                    for action in actions
                ]
            ranked = sorted(
                zip(actions, action_values),
                key=lambda item: (
                    item[1],
                    item[0].target,
                    item[0].new_links,
                ),
            )[:action_limit]
            generated += len(ranked)
            for action, action_value in ranked:
                score = (
                    state.route_cost + float(action_value)
                    if action_value_is_cost_to_go
                    else parent_score + float(action_value)
                )
                if quotient:
                    quotient_candidates.append((score, action.state))
                else:
                    key = action.state.serialize()
                    old = successors.get(key)
                    if old is not None:
                        duplicates += 1
                    if old is None or score < old[0] - 1e-12:
                        successors[key] = (score, action.state)
        if timeout or reason == "max_expansions":
            break
        if quotient:
            # Canonicalization is needed only through the score boundary that
            # fills the next beam. Candidates are stable-sorted by score, and
            # every tie at the boundary is processed, so no state capable of
            # entering the exact final beam is skipped.
            quotient_candidates.sort(key=lambda item: item[0])
            group_start = 0
            while group_start < len(quotient_candidates):
                group_score = quotient_candidates[group_start][0]
                group_end = group_start + 1
                while (
                    group_end < len(quotient_candidates)
                    and quotient_candidates[group_end][0] == group_score
                ):
                    group_end += 1
                for score, candidate_state in quotient_candidates[
                    group_start:group_end
                ]:
                    key, elapsed = canonical_key(
                        candidate_state, dfg, arch, joint=joint_quotient
                    )
                    canon_ms += elapsed
                    old = successors.get(key)
                    if old is not None:
                        duplicates += 1
                        merges += 1
                    if old is None or score < old[0] - 1e-12:
                        successors[key] = (score, candidate_state)
                if len(successors) >= beam_width:
                    break
                group_start = group_end
        if not successors:
            reason = f"no_successor_at_{op}"
            beam = []
            break
        beam = sorted(
            successors.values(),
            key=lambda item: (item[0], item[1].serialize()),
        )[:beam_width]
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    complete = [
        state for _, state in beam if len(state.placements) == len(dfg.graph)
    ]
    if complete:
        best = min(complete, key=lambda s: (s.route_cost, s.serialize()))
        legal, legal_reason = check_legality(dfg, arch, best)
        success = legal
        reason = legal_reason if not legal else ""
    else:
        best = max((s for _, s in beam), key=lambda s: len(s.placements), default=initial)
        legal, success = False, False
    return MappingResult(
        success=success,
        legal=legal,
        best_cost=float(best.route_cost) if success else float("nan"),
        mapped_operations=len(best.placements),
        expansions=expansions,
        generated_successors=generated,
        duplicate_states=duplicates,
        quotient_merges=merges,
        canonicalization_ms=canon_ms,
        routing_attempts=attempts,
        failed_routing_attempts=failures,
        raw_candidate_actions=raw_candidate_actions,
        action_orbit_representatives=orbit_representatives,
        action_orbit_actions_removed=orbit_removed,
        gnn_evaluations_avoided=orbit_removed if action_scorer is not None else 0,
        routing_attempts_avoided=0,
        action_orbit_ms=orbit_ms,
        raw_candidate_targets=raw_candidate_targets,
        target_orbit_representatives=target_orbit_representatives,
        runtime_ms=elapsed_ms,
        timeout=timeout,
        reason=reason,
        final_state=best,
    )


def check_legality(dfg: DFG, arch: Architecture, state: PartialState) -> Tuple[bool, str]:
    if set(state.placements) != set(dfg.graph):
        return False, "not_all_operations_placed"
    if len(set(state.placements.values())) != len(state.placements):
        return False, "compute_capacity"
    for op, node in state.placements.items():
        if node not in arch.graph:
            return False, "unknown_compute_node"
        if node[2] != dfg.graph.nodes[op]["slot"]:
            return False, "wrong_modulo_slot"
        if dfg.graph.nodes[op]["op"] not in arch.graph.nodes[node]["supports"]:
            return False, "unsupported_operation"
    if set(state.routed_edges) != set(dfg.graph.edges):
        return False, "missing_dependency_route"
    used = set()
    for (u, v), path in state.routed_edges.items():
        cursor = state.placements[u]
        for edge_id in path:
            if edge_id in used:
                return False, "routing_capacity"
            used.add(edge_id)
            a, b = arch.edge_list[edge_id]
            if a != cursor:
                return False, "disconnected_route"
            cursor = b
        if cursor != state.placements[v]:
            return False, "wrong_route_sink"
    if used != state.occupied_links:
        return False, "occupied_link_accounting"
    expected_cost = sum(
        arch.graph.edges[arch.edge_list[e]]["base_cost"] for e in used
    )
    if not math.isclose(expected_cost, state.route_cost, abs_tol=1e-8):
        return False, "route_cost_accounting"
    return True, ""
