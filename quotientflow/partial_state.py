"""Deterministic valid partial mapping construction."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Set, Tuple

import networkx as nx

from .arch import Architecture, RoutingNode
from .dfg import DFG

DFGEdge = Tuple[int, int]


@dataclass
class PartialState:
    placements: Dict[int, RoutingNode] = field(default_factory=dict)
    occupied_compute: Set[RoutingNode] = field(default_factory=set)
    occupied_links: Set[int] = field(default_factory=set)
    routed_edges: Dict[DFGEdge, Tuple[int, ...]] = field(default_factory=dict)
    route_cost: float = 0.0

    def copy(self) -> "PartialState":
        return PartialState(
            dict(self.placements),
            set(self.occupied_compute),
            set(self.occupied_links),
            dict(self.routed_edges),
            self.route_cost,
        )

    def serialize(self) -> tuple:
        return (
            tuple(sorted(self.placements.items())),
            tuple(sorted(self.occupied_compute)),
            tuple(sorted(self.occupied_links)),
            tuple(sorted(self.routed_edges)),
        )


def residual_graph(arch: Architecture, occupied: Iterable[int]) -> nx.DiGraph:
    blocked = set(occupied)
    graph = nx.DiGraph()
    graph.add_nodes_from(arch.graph.nodes)
    for u, v, data in arch.graph.edges(data=True):
        if data["index"] not in blocked:
            graph.add_edge(u, v, **data)
    return graph


def shortest_available_path(
    arch: Architecture,
    source: RoutingNode,
    sink: RoutingNode,
    occupied: Iterable[int],
) -> Optional[Tuple[int, ...]]:
    graph = residual_graph(arch, occupied)
    try:
        paths = nx.shortest_simple_paths(graph, source, sink, weight="base_cost")
        path = next(paths)
    except (nx.NetworkXNoPath, nx.NodeNotFound, StopIteration):
        return None
    return tuple(arch.edge_index[(a, b)] for a, b in zip(path, path[1:]))


def _distance(arch: Architecture, src: RoutingNode, dst: RoutingNode) -> float:
    try:
        return float(nx.shortest_path_length(arch.graph, src, dst, weight="base_cost"))
    except nx.NetworkXNoPath:
        return math.inf


def place_operation(
    dfg: DFG, arch: Architecture, state: PartialState, op: int, target: RoutingNode
) -> Optional[PartialState]:
    if target in state.occupied_compute:
        return None
    if target[2] != dfg.graph.nodes[op]["slot"]:
        return None
    result = state.copy()
    result.placements[op] = target
    result.occupied_compute.add(target)
    for pred in sorted(dfg.graph.predecessors(op)):
        if pred not in result.placements:
            continue
        dep = (pred, op)
        path = shortest_available_path(
            arch, result.placements[pred], target, result.occupied_links
        )
        if path is None:
            return None
        result.routed_edges[dep] = path
        result.occupied_links.update(path)
        result.route_cost += sum(
            arch.graph.edges[arch.edge_list[e]]["base_cost"] for e in path
        )
    return result


def construct_partial_state(
    dfg: DFG, arch: Architecture, fraction: float
) -> PartialState:
    count = max(1, min(len(dfg.graph) - 1, math.ceil(len(dfg.graph) * fraction)))
    state = PartialState()
    for depth, op in enumerate(dfg.order[:count]):
        slot = dfg.graph.nodes[op]["slot"]
        candidates = [
            node
            for node in arch.compute_slots
            if node[2] == slot and node not in state.occupied_compute
        ]
        preds = [p for p in dfg.graph.predecessors(op) if p in state.placements]
        if depth == 0:
            # Lexicographically least representative of all architecture orbits.
            candidates.sort(
                key=lambda n: (
                    min(p[n] for p in arch.symmetries),
                    n,
                )
            )
        else:
            candidates.sort(
                key=lambda n: (
                    sum(_distance(arch, state.placements[p], n) for p in preds),
                    n,
                )
            )
        placed = None
        for candidate in candidates:
            placed = place_operation(dfg, arch, state, op, candidate)
            if placed is not None:
                break
        if placed is None:
            raise RuntimeError(f"partial construction failed at operation {op}")
        state = placed
    return state


def deterministic_fraction(seed: int, choices=(0.10, 0.25, 0.40)) -> float:
    return choices[random.Random(seed).randrange(len(choices))]


def state_capacity_valid(state: PartialState) -> bool:
    return (
        len(state.occupied_compute) == len(state.placements)
        and sum(len(path) for path in state.routed_edges.values())
        == len(state.occupied_links)
    )
