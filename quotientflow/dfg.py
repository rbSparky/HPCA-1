"""Deterministic typed DAG families and verified construction symmetries."""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
from typing import Dict, List

import networkx as nx

FAMILIES = [
    "diamond_chain",
    "reduction_tree",
    "dot_product",
    "butterfly",
    "fir",
    "gemm_tile",
    "series_parallel",
]


@dataclass
class DFG:
    family: str
    seed: int
    graph: nx.DiGraph
    automorphisms: List[Dict[int, int]]

    @property
    def order(self) -> List[int]:
        return sorted(
            self.graph.nodes,
            key=lambda v: (
                self.graph.nodes[v]["level"],
                -self.graph.out_degree(v),
                v,
            ),
        )


def _annotate(graph: nx.DiGraph, ii: int = 3) -> None:
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("DFG construction is cyclic")
    levels = {}
    for v in nx.topological_sort(graph):
        preds = list(graph.predecessors(v))
        levels[v] = 0 if not preds else 1 + max(levels[u] for u in preds)
        graph.nodes[v]["level"] = levels[v]
        graph.nodes[v]["slot"] = levels[v] % ii
    nx.set_edge_attributes(graph, "data", "kind")


def verify_dfg_permutation(dfg: DFG, perm: Dict[int, int]) -> bool:
    graph = dfg.graph
    nodes = set(graph)
    if set(perm) != nodes or set(perm.values()) != nodes:
        return False
    for v in nodes:
        if graph.nodes[v]["op"] != graph.nodes[perm[v]]["op"]:
            return False
        if graph.nodes[v]["slot"] != graph.nodes[perm[v]]["slot"]:
            return False
    original = {(u, v, d["kind"]) for u, v, d in graph.edges(data=True)}
    image = {(perm[u], perm[v], d["kind"]) for u, v, d in graph.edges(data=True)}
    return original == image


def _all_typed_automorphisms(graph: nx.DiGraph, family: str, seed: int) -> List[Dict[int, int]]:
    matcher = nx.algorithms.isomorphism.DiGraphMatcher(
        graph,
        graph,
        node_match=lambda a, b: a["op"] == b["op"] and a["slot"] == b["slot"],
        edge_match=lambda a, b: a["kind"] == b["kind"],
    )
    holder = DFG(family, seed, graph, [])
    perms = [dict(p) for p in matcher.isomorphisms_iter()]
    assert all(verify_dfg_permutation(holder, p) for p in perms)
    return sorted(perms, key=lambda p: tuple(p[v] for v in sorted(graph)))


def _diamond(rng: random.Random, seed: int) -> DFG:
    blocks = 3 + rng.randrange(2)
    g = nx.DiGraph()
    g.add_node(0, op="ADD")
    root, nxt = 0, 1
    for i in range(blocks):
        b, c, d = nxt, nxt + 1, nxt + 2
        branch_op = "MUL" if i % 2 == 0 else "ADD"
        g.add_nodes_from(((b, {"op": branch_op}), (c, {"op": branch_op}), (d, {"op": "ADD"})))
        g.add_edges_from(((root, b), (root, c), (b, d), (c, d)))
        root, nxt = d, nxt + 3
    _annotate(g)
    return DFG("diamond_chain", seed, g, _all_typed_automorphisms(g, "diamond_chain", seed))


def _tree_graph(leaves: int) -> nx.DiGraph:
    g = nx.DiGraph()
    current = []
    for _ in range(leaves):
        v = len(g)
        g.add_node(v, op="MUL")
        current.append(v)
    while len(current) > 1:
        nxt = []
        for a, b in zip(current[::2], current[1::2]):
            v = len(g)
            g.add_node(v, op="ADD")
            g.add_edges_from(((a, v), (b, v)))
            nxt.append(v)
        current = nxt
    return g


def _reduction(seed: int) -> DFG:
    g = _tree_graph(8)
    _annotate(g)
    return DFG("reduction_tree", seed, g, _all_typed_automorphisms(g, "reduction_tree", seed))


def _dot(seed: int) -> DFG:
    g = _tree_graph(4)
    out = len(g)
    g.add_node(out, op="ADD")
    g.add_edge(out - 1, out)
    _annotate(g)
    return DFG("dot_product", seed, g, _all_typed_automorphisms(g, "dot_product", seed))


def _butterfly(rng: random.Random, seed: int) -> DFG:
    stages = 2 + rng.randrange(2)
    g = nx.DiGraph()
    prev = []
    for _ in range(4):
        v = len(g)
        g.add_node(v, op="MUL")
        prev.append(v)
    for stage in range(stages):
        cur = []
        # Width-four radix-2 butterfly: the third stage repeats the
        # nearest-neighbour pairing after the distance-two exchange.
        stride = 1 << (stage % 2)
        for lane in range(4):
            v = len(g)
            g.add_node(v, op="ADD")
            g.add_edge(prev[lane], v)
            g.add_edge(prev[lane ^ stride], v)
            cur.append(v)
        prev = cur
    _annotate(g)
    identity = {v: v for v in g}
    # Construction-preserved lane reflection; retain only after exact verification.
    reflect = {}
    for v in range(4):
        reflect[v] = 3 - v
    for stage in range(stages):
        base = 4 + 4 * stage
        for lane in range(4):
            reflect[base + lane] = base + 3 - lane
    dfg = DFG("butterfly", seed, g, [])
    dfg.automorphisms = [p for p in (identity, reflect) if verify_dfg_permutation(dfg, p)]
    return dfg


def _fir(seed: int) -> DFG:
    g = nx.DiGraph()
    taps = []
    for _ in range(4):
        v = len(g)
        g.add_node(v, op="MUL")
        taps.append(v)
    for output in range(2):
        acc = len(g)
        g.add_node(acc, op="ADD")
        g.add_edges_from(((taps[0], acc), (taps[1], acc)))
        for tap in taps[2:]:
            nxt = len(g)
            g.add_node(nxt, op="ADD")
            g.add_edges_from(((acc, nxt), (tap, nxt)))
            acc = nxt
    _annotate(g)
    return DFG("fir", seed, g, [{v: v for v in g}])


def _gemm(seed: int) -> DFG:
    g = nx.DiGraph()
    mul = {}
    for i, j, k in itertools.product(range(2), repeat=3):
        v = len(g)
        mul[i, j, k] = v
        g.add_node(v, op="MUL")
    add = {}
    for i, j in itertools.product(range(2), repeat=2):
        v = len(g)
        add[i, j] = v
        g.add_node(v, op="ADD")
        g.add_edges_from(((mul[i, j, 0], v), (mul[i, j, 1], v)))
    _annotate(g)
    perms = []
    for swap_i, swap_j in itertools.product((False, True), repeat=2):
        p = {}
        for (i, j, k), v in mul.items():
            p[v] = mul[1 - i if swap_i else i, 1 - j if swap_j else j, k]
        for (i, j), v in add.items():
            p[v] = add[1 - i if swap_i else i, 1 - j if swap_j else j]
        perms.append(p)
    dfg = DFG("gemm_tile", seed, g, perms)
    assert all(verify_dfg_permutation(dfg, p) for p in perms)
    return dfg


def _series_parallel(rng: random.Random, seed: int) -> DFG:
    n = rng.randint(8, 16)
    g = nx.DiGraph()
    for v in range(n):
        g.add_node(v, op="MUL" if v % 3 == 0 else "ADD")
    g.add_edges_from((v, v + 1) for v in range(n - 1))
    target = min(int(1.4 * n), n * (n - 1) // 2)
    candidates = [(u, v) for u in range(n) for v in range(u + 2, n)]
    rng.shuffle(candidates)
    for u, v in candidates:
        if g.number_of_edges() >= target:
            break
        if g.out_degree(u) < 3 and g.in_degree(v) < 3:
            g.add_edge(u, v)
    _annotate(g)
    return DFG("series_parallel", seed, g, [{v: v for v in g}])


def generate_dfg(family: str, seed: int) -> DFG:
    rng = random.Random(seed)
    if family == "diamond_chain":
        dfg = _diamond(rng, seed)
    elif family == "reduction_tree":
        dfg = _reduction(seed)
    elif family == "dot_product":
        dfg = _dot(seed)
    elif family == "butterfly":
        dfg = _butterfly(rng, seed)
    elif family == "fir":
        dfg = _fir(seed)
    elif family == "gemm_tile":
        dfg = _gemm(seed)
    elif family == "series_parallel":
        dfg = _series_parallel(rng, seed)
    else:
        raise ValueError(f"unknown DFG family {family}")
    if not (8 <= len(dfg.graph) <= 16):
        raise AssertionError(f"{family} produced {len(dfg.graph)} operations")
    if not nx.is_directed_acyclic_graph(dfg.graph):
        raise AssertionError(f"{family} is cyclic")
    if not all(verify_dfg_permutation(dfg, p) for p in dfg.automorphisms):
        raise AssertionError(f"{family} proposed an invalid automorphism")
    return dfg


def generate_exactness_dfg(family: str, seed: int) -> DFG:
    """Small 6--12 operation constructions for the required exactness subtest."""
    if family == "reduction_tree":
        graph = _tree_graph(4)
        _annotate(graph)
        return DFG(
            family,
            seed,
            graph,
            _all_typed_automorphisms(graph, family, seed),
        )
    if family == "dot_product":
        return _dot(seed)
    if family == "diamond_chain":
        # Three diamonds always produce ten operations.
        graph = nx.DiGraph()
        graph.add_node(0, op="ADD")
        root, nxt = 0, 1
        for i in range(3):
            b, c, d = nxt, nxt + 1, nxt + 2
            op = "MUL" if i % 2 == 0 else "ADD"
            graph.add_nodes_from(((b, {"op": op}), (c, {"op": op}), (d, {"op": "ADD"})))
            graph.add_edges_from(((root, b), (root, c), (b, d), (c, d)))
            root, nxt = d, nxt + 3
        _annotate(graph)
        return DFG(
            family,
            seed,
            graph,
            _all_typed_automorphisms(graph, family, seed),
        )
    raise ValueError(f"{family} is not an exactness family")


def generate_stress_dfg(family: str, seed: int) -> DFG:
    """Deterministic 13--20 operation stress variants for revision-v3."""
    rng = random.Random(seed)
    g = nx.DiGraph()
    if family == "diamond_chain":
        g.add_node(0, op="ADD")
        root, nxt = 0, 1
        for block in range(4):
            left, right, merge = nxt, nxt + 1, nxt + 2
            op = "MUL" if block % 2 == 0 else "ADD"
            g.add_nodes_from((
                (left, {"op": op}), (right, {"op": op}),
                (merge, {"op": "ADD"}),
            ))
            g.add_edges_from((
                (root, left), (root, right),
                (left, merge), (right, merge),
            ))
            root, nxt = merge, nxt + 3
    elif family == "reduction_tree":
        g = _tree_graph(8)  # 15 operations
    elif family == "dot_product":
        # Four lanes, each with two products, followed by an eight-leaf sum.
        products = []
        for _ in range(8):
            node = len(g)
            g.add_node(node, op="MUL")
            products.append(node)
        current = products
        while len(current) > 1:
            next_level = []
            for a, b in zip(current[::2], current[1::2]):
                node = len(g)
                g.add_node(node, op="ADD")
                g.add_edges_from(((a, node), (b, node)))
                next_level.append(node)
            current = next_level
    elif family == "butterfly":
        previous = []
        for _ in range(4):
            node = len(g)
            g.add_node(node, op="MUL")
            previous.append(node)
        for stage in range(4):
            current = []
            stride = 1 << (stage % 2)
            for lane in range(4):
                node = len(g)
                g.add_node(node, op="ADD")
                g.add_edges_from((
                    (previous[lane], node),
                    (previous[lane ^ stride], node),
                ))
                current.append(node)
            previous = current
    elif family == "fir":
        taps = []
        for _ in range(4):
            node = len(g)
            g.add_node(node, op="MUL")
            taps.append(node)
        # Four outputs share taps but have independent three-add reductions.
        for output in range(4):
            accumulator = len(g)
            g.add_node(accumulator, op="ADD")
            g.add_edges_from(((taps[0], accumulator), (taps[1], accumulator)))
            for tap in taps[2:]:
                nxt = len(g)
                g.add_node(nxt, op="ADD")
                g.add_edges_from(((accumulator, nxt), (tap, nxt)))
                accumulator = nxt
    elif family == "gemm_tile":
        multiply = {}
        for i, j, k in itertools.product(range(2), range(2), range(3)):
            node = len(g)
            g.add_node(node, op="MUL")
            multiply[i, j, k] = node
        for i, j in itertools.product(range(2), repeat=2):
            first = len(g)
            g.add_node(first, op="ADD")
            g.add_edges_from((
                (multiply[i, j, 0], first),
                (multiply[i, j, 1], first),
            ))
            second = len(g)
            g.add_node(second, op="ADD")
            g.add_edges_from(((first, second), (multiply[i, j, 2], second)))
    elif family == "series_parallel":
        n = 18 + seed % 3
        for node in range(n):
            g.add_node(node, op="MUL" if node % 4 == 0 else "ADD")
        g.add_edges_from((node, node + 1) for node in range(n - 1))
        candidates = [
            (u, v) for u in range(n) for v in range(u + 2, min(n, u + 6))
        ]
        rng.shuffle(candidates)
        for u, v in candidates:
            if g.number_of_edges() >= int(1.45 * n):
                break
            if g.out_degree(u) < 3 and g.in_degree(v) < 3:
                g.add_edge(u, v)
    else:
        raise ValueError(f"unknown stress DFG family {family}")
    _annotate(g)
    dfg = DFG(family, seed, g, [{node: node for node in g}])
    if not (13 <= len(g) <= 20):
        raise AssertionError(f"stress {family} has {len(g)} operations")
    if not nx.is_directed_acyclic_graph(g):
        raise AssertionError(f"stress {family} is cyclic")
    return dfg
