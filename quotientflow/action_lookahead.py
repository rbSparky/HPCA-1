"""Action-conditioned relaxed lookahead, explicit features, and exact action orbits."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import networkx as nx
import numpy as np

from .arch import Architecture, RoutingNode
from .dfg import DFG
from .mapper import Action
from .partial_state import PartialState, residual_graph
from .symmetry import (
    architecture_edge_map,
    preserves_total_operation_order,
    transform_state,
)


FEATURE_NAMES = [
    "immediate_route_length",
    "new_link_count",
    "candidate_x",
    "candidate_y",
    "candidate_time",
    "operation_is_add",
    "operation_is_mul",
    "operation_indegree",
    "operation_outdegree",
    "partial_depth",
    "remaining_operation_count",
    "occupied_link_fraction",
    "dual_sum",
    "dual_max",
    "dual_top2_sum",
    "scarcity_adjusted_dual_sum",
    "positive_price_links_used",
    "top_decile_link_fraction",
    "alt_count_min",
    "alt_count_mean",
    "alt_count_max",
    "alt_min_length_min",
    "alt_min_length_mean",
    "alt_min_length_max",
    "next_length_gap_min",
    "next_length_gap_mean",
    "next_length_gap_max",
    "edge_disjoint_alt_min",
    "edge_disjoint_alt_mean",
    "edge_disjoint_alt_max",
    "min_cut_proxy_min",
    "min_cut_proxy_mean",
    "min_cut_proxy_max",
    "no_replacement_fraction_min",
    "no_replacement_fraction_mean",
    "no_replacement_fraction_max",
    "remaining_routing_capacity",
    "timeslice_components_mean",
    "timeslice_components_max",
    "largest_residual_component",
    "saturated_link_count",
    "remaining_cut_dependencies",
    "mean_shortest_residual_distance",
    "remaining_no_path_count",
]

_ALTERNATIVE_COUNT_CACHE: Dict[tuple, int] = {}


def check_partial_legality(
    dfg: DFG, arch: Architecture, state: PartialState
) -> Tuple[bool, str]:
    """Independent legality check for a committed intermediate state."""
    if not set(state.placements).issubset(set(dfg.graph)):
        return False, "unknown_operation"
    if len(set(state.placements.values())) != len(state.placements):
        return False, "compute_capacity"
    for op, node in state.placements.items():
        if node not in arch.graph:
            return False, "unknown_compute_node"
        data = dfg.graph.nodes[op]
        if node[2] != data["slot"]:
            return False, "wrong_modulo_slot"
        if data["op"] not in arch.graph.nodes[node]["supports"]:
            return False, "unsupported_operation"
    used: set[int] = set()
    for dependency, path in state.routed_edges.items():
        if dependency not in dfg.graph.edges:
            return False, "unknown_dependency"
        u, v = dependency
        if u not in state.placements or v not in state.placements:
            return False, "route_without_placed_endpoint"
        cursor = state.placements[u]
        for edge_id in path:
            if edge_id < 0 or edge_id >= len(arch.edge_list):
                return False, "unknown_routing_edge"
            if edge_id in used:
                return False, "routing_capacity"
            used.add(edge_id)
            a, b = arch.edge_list[edge_id]
            if cursor != a:
                return False, "disconnected_route"
            cursor = b
        if cursor != state.placements[v]:
            return False, "wrong_route_sink"
    if used != state.occupied_links:
        return False, "occupied_link_accounting"
    if set(state.occupied_compute) != set(state.placements.values()):
        return False, "occupied_compute_accounting"
    expected_cost = sum(
        arch.graph.edges[arch.edge_list[edge_id]]["base_cost"]
        for edge_id in used
    )
    if not math.isclose(expected_cost, state.route_cost, abs_tol=1e-8):
        return False, "route_cost_accounting"
    return True, ""


def _state_payload(state: PartialState) -> dict:
    return {
        "placements": sorted((op, list(node)) for op, node in state.placements.items()),
        "occupied_compute": sorted(map(list, state.occupied_compute)),
        "occupied_links": sorted(state.occupied_links),
        "routed_edges": sorted(
            ([u, v], list(path)) for (u, v), path in state.routed_edges.items()
        ),
        "route_cost": state.route_cost,
    }


def action_cache_key(
    dfg: DFG,
    arch: Architecture,
    state: PartialState,
    op: int,
    action: Action,
    relaxation_config: dict,
) -> str:
    payload = {
        "architecture": {
            "name": arch.name,
            "ii": arch.ii,
            "edges": [[list(u), list(v)] for u, v in arch.edge_list],
        },
        "dfg": {
            "family": dfg.family,
            "seed": dfg.seed,
            "nodes": sorted(
                (v, dict(sorted(dfg.graph.nodes[v].items()))) for v in dfg.graph
            ),
            "edges": sorted((u, v, dict(sorted(data.items()))) for u, v, data in dfg.graph.edges(data=True)),
        },
        "state": _state_payload(state),
        "action": {
            "operation": op,
            "target": list(action.target),
            "new_links": list(action.new_links),
            "committed_state": _state_payload(action.state),
        },
        "relaxation": relaxation_config,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _edge_alternative_count(
    arch: Architecture,
    state: PartialState,
    edge_id: int,
    cap: int = 4,
) -> int:
    cache_key = (
        arch.name,
        arch.width,
        arch.height,
        arch.ii,
        tuple(sorted(state.occupied_links)),
        edge_id,
        cap,
    )
    cached = _ALTERNATIVE_COUNT_CACHE.get(cache_key)
    if cached is not None:
        return cached
    graph = residual_graph(arch, state.occupied_links)
    u, v = arch.edge_list[edge_id]
    if not graph.has_edge(u, v):
        _ALTERNATIVE_COUNT_CACHE[cache_key] = 0
        return 0
    graph.remove_edge(u, v)
    try:
        result = sum(
            1
            for _ in itertools.islice(
                nx.shortest_simple_paths(graph, u, v, weight="base_cost"), cap
            )
        )
        _ALTERNATIVE_COUNT_CACHE[cache_key] = result
        return result
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        _ALTERNATIVE_COUNT_CACHE[cache_key] = 0
        return 0


def scarcity_adjusted_sum(
    arch: Architecture,
    state: PartialState,
    action: Action,
    prices: np.ndarray,
) -> float:
    total = 0.0
    for edge_id in action.new_links:
        alternatives = _edge_alternative_count(arch, state, edge_id, 4)
        total += float(prices[edge_id]) / max(1, alternatives)
    return total


def select_candidate_actions(
    dfg: DFG,
    arch: Architecture,
    state: PartialState,
    op: int,
    actions: Sequence[Action],
    prices: np.ndarray,
    limit: int = 8,
) -> List[Action]:
    """Deterministic diverse subset with all required anchor actions."""
    ordered = sorted(actions, key=lambda action: (action.base_cost, action.target, action.new_links))
    if len(ordered) <= limit:
        return ordered
    chosen: List[Action] = []
    seen = set()

    def add(action: Action) -> None:
        signature = (action.target, action.new_links)
        if signature not in seen and len(chosen) < limit:
            seen.add(signature)
            chosen.append(action)

    add(ordered[0])
    add(
        min(
            ordered,
            key=lambda action: (
                action.base_cost
                + 0.5 * scarcity_adjusted_sum(arch, state, action, prices),
                action.target,
                action.new_links,
            ),
        )
    )
    add(
        min(
            ordered,
            key=lambda action: (
                action.base_cost
                + 0.5
                * float(np.asarray(prices)[list(action.new_links)].sum())
                if action.new_links
                else action.base_cost,
                action.target,
                action.new_links,
            ),
        )
    )
    used_targets = {action.target for action in chosen}
    for action in ordered:
        if action.target not in used_targets:
            add(action)
            used_targets.add(action.target)
    used_paths = {action.new_links for action in chosen}
    for action in ordered:
        if action.new_links not in used_paths:
            add(action)
            used_paths.add(action.new_links)
    for action in ordered:
        add(action)
    return chosen


def action_flags(
    arch: Architecture,
    state: PartialState,
    actions: Sequence[Action],
    prices: np.ndarray,
) -> Dict[str, Tuple[RoutingNode, Tuple[int, ...]]]:
    key = lambda action: (action.target, action.new_links)
    length = min(actions, key=lambda a: (a.base_cost, a.target, a.new_links))
    scarcity = min(
        actions,
        key=lambda a: (
            a.base_cost + 0.5 * scarcity_adjusted_sum(arch, state, a, prices),
            a.target,
            a.new_links,
        ),
    )
    oracle_price = min(
        actions,
        key=lambda a: (
            a.base_cost
            + (0.5 * float(prices[list(a.new_links)].sum()) if a.new_links else 0.0),
            a.target,
            a.new_links,
        ),
    )
    return {
        "length": key(length),
        "scarcity": key(scarcity),
        "oracle_price": key(oracle_price),
    }


def _aggregate(values: Sequence[float], default: float = 0.0) -> Tuple[float, float, float]:
    if not values:
        return default, default, default
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    if not finite.size:
        return default, default, default
    return float(finite.min()), float(finite.mean()), float(finite.max())


def _path_cost(arch: Architecture, path: Iterable[int]) -> float:
    return float(
        sum(arch.graph.edges[arch.edge_list[edge]]["base_cost"] for edge in path)
    )


def _dependency_path_features(
    dfg: DFG,
    arch: Architecture,
    state: PartialState,
    action: Action,
) -> List[float]:
    graph = residual_graph(arch, state.occupied_links)
    new_dependencies = sorted(set(action.state.routed_edges) - set(state.routed_edges))
    alt_counts: List[float] = []
    alt_lengths: List[float] = []
    length_gaps: List[float] = []
    disjoint_counts: List[float] = []
    cut_proxies: List[float] = []
    no_replacement: List[float] = []
    for u, v in new_dependencies:
        selected = action.state.routed_edges[(u, v)]
        source, sink = state.placements[u], action.target
        try:
            node_paths = list(
                itertools.islice(
                    nx.shortest_simple_paths(graph, source, sink, weight="base_cost"),
                    5,
                )
            )
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            node_paths = []
        candidates = [
            tuple(
                arch.edge_index[(a, b)]
                for a, b in zip(node_path, node_path[1:])
            )
            for node_path in node_paths
        ]
        alternatives = [path for path in candidates if path != selected][:4]
        alt_counts.append(float(len(alternatives)))
        lengths = [_path_cost(arch, path) for path in alternatives]
        selected_length = _path_cost(arch, selected)
        alt_lengths.append(min(lengths) if lengths else selected_length + 4.0)
        length_gaps.append(
            max(0.0, min(lengths) - selected_length) if lengths else 4.0
        )
        selected_set = set(selected)
        disjoint = sum(not selected_set.intersection(path) for path in alternatives)
        disjoint_counts.append(float(disjoint))
        cut_proxies.append(float(min(4, disjoint + 1)))
        if selected:
            no_replacement.append(
                float(
                    np.mean(
                        [
                            not any(edge not in alternative for alternative in alternatives)
                            for edge in selected
                        ]
                    )
                )
                if alternatives
                else 1.0
            )
        else:
            no_replacement.append(0.0)
    return [
        *_aggregate(alt_counts),
        *_aggregate(alt_lengths),
        *_aggregate(length_gaps),
        *_aggregate(disjoint_counts),
        *_aggregate(cut_proxies),
        *_aggregate(no_replacement),
    ]


def _endpoint_region(
    dfg: DFG,
    arch: Architecture,
    state: PartialState,
    op: int,
) -> List[RoutingNode]:
    if op in state.placements:
        return [state.placements[op]]
    slot = dfg.graph.nodes[op]["slot"]
    return [
        node
        for node in arch.compute_slots
        if node[2] == slot and node not in state.occupied_compute
    ]


def _residual_features(
    dfg: DFG, arch: Architecture, committed: PartialState
) -> List[float]:
    graph = residual_graph(arch, committed.occupied_links)
    component_counts = []
    largest_components = []
    for time_slot in range(arch.ii):
        physical = nx.Graph()
        physical.add_nodes_from(
            (x, y) for x in range(arch.width) for y in range(arch.height)
        )
        for u, v in graph.edges:
            if u[2] == time_slot and (u[0], u[1]) != (v[0], v[1]):
                physical.add_edge((u[0], u[1]), (v[0], v[1]))
        components = list(nx.connected_components(physical))
        component_counts.append(float(len(components)))
        largest_components.append(float(max(map(len, components), default=0)))
    try:
        distances = dict(nx.all_pairs_dijkstra_path_length(graph, weight="base_cost"))
    except Exception:
        distances = {}
    remaining_dependencies = [
        edge for edge in sorted(dfg.graph.edges) if edge not in committed.routed_edges
    ]
    minimum_distances = []
    no_path = 0
    cut_dependencies = 0
    for u, v in remaining_dependencies:
        sources = _endpoint_region(dfg, arch, committed, u)
        sinks = _endpoint_region(dfg, arch, committed, v)
        values = [
            distances.get(source, {}).get(sink, math.inf)
            for source in sources
            for sink in sinks
        ]
        finite = [value for value in values if math.isfinite(value)]
        if finite:
            minimum_distances.append(float(min(finite)))
        else:
            no_path += 1
        if arch.name == "cut4" and sources and sinks:
            source_left = all(node[0] <= 1 for node in sources)
            source_right = all(node[0] >= 2 for node in sources)
            sink_left = all(node[0] <= 1 for node in sinks)
            sink_right = all(node[0] >= 2 for node in sinks)
            cut_dependencies += int(
                (source_left and sink_right) or (source_right and sink_left)
            )
    return [
        (len(arch.edge_list) - len(committed.occupied_links))
        / max(1, len(arch.edge_list)),
        float(np.mean(component_counts)),
        float(max(component_counts, default=0.0)),
        float(max(largest_components, default=0.0)),
        float(len(committed.occupied_links)),
        float(cut_dependencies),
        float(np.mean(minimum_distances)) if minimum_distances else 0.0,
        float(no_path),
    ]


def action_features(
    dfg: DFG,
    arch: Architecture,
    state: PartialState,
    op: int,
    action: Action,
    prices: np.ndarray,
) -> Dict[str, float]:
    values = (
        np.asarray(prices, dtype=float)[list(action.new_links)]
        if action.new_links
        else np.zeros(0)
    )
    positive = np.asarray(prices)[np.asarray(prices) > 1e-10]
    threshold = float(np.quantile(positive, 0.9)) if positive.size else math.inf
    basic = [
        float(action.base_cost),
        float(len(set(action.new_links))),
        action.target[0] / max(1, arch.width - 1),
        action.target[1] / max(1, arch.height - 1),
        action.target[2] / max(1, arch.ii - 1),
        float(dfg.graph.nodes[op]["op"] == "ADD"),
        float(dfg.graph.nodes[op]["op"] == "MUL"),
        float(dfg.graph.in_degree(op)),
        float(dfg.graph.out_degree(op)),
        len(state.placements) / max(1, len(dfg.graph)),
        float(len(dfg.graph) - len(state.placements) - 1),
        len(state.occupied_links) / max(1, len(arch.edge_list)),
    ]
    scarcity = [
        float(values.sum()) if values.size else 0.0,
        float(values.max()) if values.size else 0.0,
        float(np.sort(values)[-2:].sum()) if values.size else 0.0,
        scarcity_adjusted_sum(arch, state, action, np.asarray(prices)),
        float(np.sum(values > 1e-10)),
        float(np.mean(values >= threshold)) if values.size else 0.0,
    ]
    feature_values = [
        *basic,
        *scarcity,
        *_dependency_path_features(dfg, arch, state, action),
        *_residual_features(dfg, arch, action.state),
    ]
    assert len(feature_values) == len(FEATURE_NAMES)
    return dict(zip(FEATURE_NAMES, map(float, feature_values)))


def action_signature(action: Action) -> tuple:
    return action.target, action.new_links, _full_state_signature(action.state)


def _full_state_signature(state: PartialState) -> tuple:
    return (
        tuple(sorted(state.placements.items())),
        tuple(sorted(state.occupied_compute)),
        tuple(sorted(state.occupied_links)),
        tuple(sorted(state.routed_edges.items())),
    )


def stabilizer_action_orbits(
    dfg: DFG,
    arch: Architecture,
    state: PartialState,
    op: int,
    actions: Sequence[Action],
) -> Tuple[List[Action], dict]:
    """Return exact deterministic action-orbit representatives."""
    start = time.perf_counter()
    identity_state = _full_state_signature(state)
    stabilizer = []
    for dfg_perm, arch_perm in itertools.product(
        dfg.automorphisms, arch.symmetries
    ):
        if not preserves_total_operation_order(dfg, dfg_perm):
            continue
        if dfg_perm[op] != op:
            continue
        edge_map = architecture_edge_map(arch, arch_perm)
        transformed = transform_state(
            state, arch, dfg_perm, arch_perm, edge_map=edge_map
        )
        if _full_state_signature(transformed) == identity_state:
            stabilizer.append((dfg_perm, arch_perm, edge_map))
    signature_to_index = {action_signature(action): i for i, action in enumerate(actions)}
    adjacency = [set([i]) for i in range(len(actions))]
    for index, action in enumerate(actions):
        for dfg_perm, arch_perm, edge_map in stabilizer:
            transformed_state = transform_state(
                action.state,
                arch,
                dfg_perm,
                arch_perm,
                edge_map=edge_map,
            )
            transformed_signature = (
                arch_perm[action.target],
                tuple(edge_map[edge] for edge in action.new_links),
                _full_state_signature(transformed_state),
            )
            other = signature_to_index.get(transformed_signature)
            if other is not None:
                adjacency[index].add(other)
                adjacency[other].add(index)
    representatives = []
    visited = set()
    orbit_sizes = []
    for root in range(len(actions)):
        if root in visited:
            continue
        stack = [root]
        component = []
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            component.append(current)
            stack.extend(adjacency[current] - visited)
        representative = min(
            component,
            key=lambda idx: (
                actions[idx].base_cost,
                actions[idx].target,
                actions[idx].new_links,
            ),
        )
        representatives.append(actions[representative])
        orbit_sizes.append(len(component))
    representatives.sort(key=lambda action: (action.base_cost, action.target, action.new_links))
    return representatives, {
        "stabilizer_size": len(stabilizer),
        "raw_actions": len(actions),
        "orbit_representatives": len(representatives),
        "actions_removed": len(actions) - len(representatives),
        "orbit_sizes": json.dumps(orbit_sizes),
        "action_orbit_ms": (time.perf_counter() - start) * 1000.0,
    }


def stabilizer_target_orbits(
    dfg: DFG,
    arch: Architecture,
    state: PartialState,
    op: int,
) -> Tuple[List[RoutingNode], dict]:
    """Exact PE/time target representatives under the current-state stabilizer."""
    start = time.perf_counter()
    slot = dfg.graph.nodes[op]["slot"]
    targets = [
        node
        for node in arch.compute_slots
        if node[2] == slot and node not in state.occupied_compute
    ]
    target_set = set(targets)
    identity_state = _full_state_signature(state)
    stabilizer = []
    for dfg_perm, arch_perm in itertools.product(
        dfg.automorphisms, arch.symmetries
    ):
        if not preserves_total_operation_order(dfg, dfg_perm):
            continue
        if dfg_perm[op] != op:
            continue
        edge_map = architecture_edge_map(arch, arch_perm)
        transformed = transform_state(
            state, arch, dfg_perm, arch_perm, edge_map=edge_map
        )
        if _full_state_signature(transformed) == identity_state:
            stabilizer.append(arch_perm)
    adjacency = {target: {target} for target in targets}
    for target in targets:
        for arch_perm in stabilizer:
            image = arch_perm[target]
            if image in target_set:
                adjacency[target].add(image)
                adjacency[image].add(target)
    representatives = []
    visited = set()
    for root in targets:
        if root in visited:
            continue
        component = set()
        stack = [root]
        while stack:
            current = stack.pop()
            if current in component:
                continue
            component.add(current)
            stack.extend(adjacency[current] - component)
        visited.update(component)
        representatives.append(min(component))
    representatives.sort()
    return representatives, {
        "stabilizer_size": len(stabilizer),
        "raw_targets": len(targets),
        "target_orbit_representatives": len(representatives),
        "targets_removed": len(targets) - len(representatives),
        "target_orbit_ms": (time.perf_counter() - start) * 1000.0,
    }
