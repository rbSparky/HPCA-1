"""Typed time-expanded CGRA routing architectures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import networkx as nx

RoutingNode = Tuple[int, int, int]


@dataclass
class Architecture:
    name: str
    width: int
    height: int
    ii: int
    graph: nx.DiGraph
    edge_list: List[Tuple[RoutingNode, RoutingNode]]
    edge_index: Dict[Tuple[RoutingNode, RoutingNode], int]
    symmetries: List[Dict[RoutingNode, RoutingNode]]

    @property
    def compute_slots(self) -> List[RoutingNode]:
        return sorted(self.graph.nodes)


def _physical_links(name: str, width: int, height: int) -> set:
    links = set()

    def add(a: Tuple[int, int], b: Tuple[int, int]) -> None:
        links.add(tuple(sorted((a, b))))

    for x in range(width):
        for y in range(height):
            if x + 1 < width:
                if not (name == "cut4" and x == 1 and y not in (0, height - 1)):
                    add((x, y), (x + 1, y))
            if y + 1 < height:
                add((x, y), (x, y + 1))
            if name == "diag4" and x + 1 < width and y + 1 < height:
                add((x, y), (x + 1, y + 1))
                add((x + 1, y), (x, y + 1))
    if name == "torus3":
        for x in range(width):
            for y in range(height):
                add((x, y), ((x + 1) % width, y))
                add((x, y), (x, (y + 1) % height))
    return links


def _d4_xy(x: int, y: int, n: int, kind: int) -> Tuple[int, int]:
    transforms = (
        (x, y),
        (n - 1 - y, x),
        (n - 1 - x, n - 1 - y),
        (y, n - 1 - x),
        (n - 1 - x, y),
        (x, n - 1 - y),
        (y, x),
        (n - 1 - y, n - 1 - x),
    )
    return transforms[kind]


def verify_arch_permutation(
    arch: Architecture, perm: Dict[RoutingNode, RoutingNode]
) -> bool:
    nodes = set(arch.graph.nodes)
    if set(perm) != nodes or set(perm.values()) != nodes:
        return False
    for node, image in perm.items():
        if image[2] != node[2]:
            return False
        if arch.graph.nodes[node]["capacity"] != arch.graph.nodes[image]["capacity"]:
            return False
        if arch.graph.nodes[node]["supports"] != arch.graph.nodes[image]["supports"]:
            return False
    for u, v, data in arch.graph.edges(data=True):
        a, b = perm[u], perm[v]
        if not arch.graph.has_edge(a, b):
            return False
        other = arch.graph.edges[a, b]
        if (data["kind"], data["capacity"], data["base_cost"]) != (
            other["kind"],
            other["capacity"],
            other["base_cost"],
        ):
            return False
    return True


def _symmetries(arch: Architecture) -> List[Dict[RoutingNode, RoutingNode]]:
    candidates = []
    n = arch.width
    for kind in range(8):
        translations: Iterable[Tuple[int, int]]
        translations = (
            ((dx, dy) for dx in range(n) for dy in range(n))
            if arch.name == "torus3"
            else ((0, 0),)
        )
        for dx, dy in translations:
            perm = {}
            for x, y, t in arch.graph.nodes:
                xx, yy = _d4_xy(x, y, n, kind)
                perm[(x, y, t)] = ((xx + dx) % n, (yy + dy) % n, t)
            if verify_arch_permutation(arch, perm):
                candidates.append(perm)
    unique = {}
    order = sorted(arch.graph.nodes)
    for perm in candidates:
        unique[tuple(perm[n] for n in order)] = perm
    return [unique[k] for k in sorted(unique)]


def build_architecture(name: str, ii: int = 3) -> Architecture:
    specs = {
        "mesh3": (3, 3),
        "torus3": (3, 3),
        "mesh4": (4, 4),
        "diag4": (4, 4),
        "cut4": (4, 4),
    }
    if name not in specs:
        raise ValueError(f"unknown architecture {name}")
    width, height = specs[name]
    graph = nx.DiGraph(name=name)
    for x in range(width):
        for y in range(height):
            for t in range(ii):
                graph.add_node(
                    (x, y, t), capacity=1.0, supports=("ADD", "MUL")
                )
    for x in range(width):
        for y in range(height):
            for t in range(ii):
                u, v = (x, y, t), (x, y, (t + 1) % ii)
                graph.add_edge(
                    u,
                    v,
                    kind="wait",
                    capacity=1.0,
                    base_cost=1.05,
                    dx=0,
                    dy=0,
                    wrap=False,
                )
    for a, b in sorted(_physical_links(name, width, height)):
        for src, dst in ((a, b), (b, a)):
            raw_dx, raw_dy = dst[0] - src[0], dst[1] - src[1]
            coordinate_seam = abs(raw_dx) > 1 or abs(raw_dy) > 1
            # A torus has no distinguished seam: marking coordinate-seam links
            # would incorrectly destroy its exact translation symmetries.
            wrap = coordinate_seam and name != "torus3"
            dx = 1 if raw_dx > 0 else -1 if raw_dx < 0 else 0
            dy = 1 if raw_dy > 0 else -1 if raw_dy < 0 else 0
            if coordinate_seam:
                dx = -1 if raw_dx > 1 else 1 if raw_dx < -1 else dx
                dy = -1 if raw_dy > 1 else 1 if raw_dy < -1 else dy
            diagonal = dx != 0 and dy != 0
            kind = "diagonal" if diagonal else ("wrap" if wrap else "cardinal")
            cost = 1.15 if diagonal else 1.0
            for t in range(ii):
                u = (src[0], src[1], t)
                v = (dst[0], dst[1], (t + 1) % ii)
                graph.add_edge(
                    u,
                    v,
                    kind=kind,
                    capacity=1.0,
                    base_cost=cost,
                    dx=dx,
                    dy=dy,
                    wrap=wrap,
                )
    edge_list = sorted(graph.edges)
    edge_index = {edge: i for i, edge in enumerate(edge_list)}
    for edge, idx in edge_index.items():
        graph.edges[edge]["index"] = idx
    arch = Architecture(name, width, height, ii, graph, edge_list, edge_index, [])
    arch.symmetries = _symmetries(arch)
    if not arch.symmetries:
        raise RuntimeError(f"no verified identity symmetry for {name}")
    return arch


EXPECTED_COUNTS = {
    "mesh3": (27, 99),
    "torus3": (27, 135),
    "mesh4": (48, 192),
    "diag4": (48, 300),
    "cut4": (48, 180),
}
