"""Verified exact automorphism actions and state canonicalization."""

from __future__ import annotations

import itertools
import time
from typing import Dict, Iterable, Optional, Tuple

from .arch import Architecture, RoutingNode, verify_arch_permutation
from .dfg import DFG, verify_dfg_permutation
from .partial_state import PartialState

def architecture_edge_map(
    arch: Architecture, perm: Dict[RoutingNode, RoutingNode]
) -> Dict[int, int]:
    if not verify_arch_permutation(arch, perm):
        raise ValueError("invalid architecture transform")
    return {
        idx: arch.edge_index[(perm[u], perm[v])]
        for idx, (u, v) in enumerate(arch.edge_list)
    }


def transform_state(
    state: PartialState,
    arch: Architecture,
    dfg_perm: Dict[int, int],
    arch_perm: Dict[RoutingNode, RoutingNode],
    edge_map: Optional[Dict[int, int]] = None,
) -> PartialState:
    edge_map = edge_map or architecture_edge_map(arch, arch_perm)
    return PartialState(
        placements={
            dfg_perm[op]: arch_perm[node] for op, node in state.placements.items()
        },
        occupied_compute={arch_perm[node] for node in state.occupied_compute},
        occupied_links={edge_map[e] for e in state.occupied_links},
        routed_edges={
            (dfg_perm[u], dfg_perm[v]): tuple(edge_map[e] for e in path)
            for (u, v), path in state.routed_edges.items()
        },
        route_cost=state.route_cost,
    )


def canonical_key(
    state: PartialState,
    dfg: DFG,
    arch: Architecture,
    joint: bool = False,
) -> Tuple[tuple, float]:
    """Return an exact orbit key without allocating transformed states.

    The key encodes every field used by ``PartialState.serialize``. Integer
    arrays and fixed-position tuples replace transformed dictionaries, sets,
    and repeated general-purpose sorting. Compiled actions are cached on the
    graph objects themselves, so their lifetime is bounded by those objects
    and Python object-id reuse cannot corrupt a global cache.
    """
    start = time.perf_counter()
    nodes = getattr(arch, "_qf_canonical_nodes", None)
    arch_actions = getattr(arch, "_qf_canonical_actions", None)
    if nodes is None or arch_actions is None:
        nodes = tuple(sorted(arch.graph.nodes))
        node_index = {node: idx for idx, node in enumerate(nodes)}
        arch_actions = []
        for perm in arch.symmetries:
            edge_map = architecture_edge_map(arch, perm)
            arch_actions.append(
                (
                    tuple(node_index[perm[node]] for node in nodes),
                    tuple(edge_map[idx] for idx in range(len(arch.edge_list))),
                )
            )
        arch_actions = tuple(arch_actions)
        arch._qf_canonical_nodes = nodes
        arch._qf_canonical_node_index = node_index
        arch._qf_canonical_actions = arch_actions
    else:
        node_index = arch._qf_canonical_node_index

    operations = getattr(dfg, "_qf_canonical_operations", None)
    dfg_actions = getattr(dfg, "_qf_canonical_actions", None)
    if operations is None or dfg_actions is None:
        operations = tuple(sorted(dfg.graph.nodes))
        operation_index = {op: idx for idx, op in enumerate(operations)}
        dfg_edges = tuple(sorted(dfg.graph.edges))
        dfg_edge_index = {edge: idx for idx, edge in enumerate(dfg_edges)}
        dfg_actions = []
        for perm in dfg.automorphisms:
            dfg_actions.append(
                (
                    tuple(operation_index[perm[op]] for op in operations),
                    tuple(
                        dfg_edge_index[(perm[u], perm[v])] for u, v in dfg_edges
                    ),
                )
            )
        dfg_actions = tuple(dfg_actions)
        dfg._qf_canonical_operations = operations
        dfg._qf_canonical_operation_index = operation_index
        dfg._qf_canonical_edges = dfg_edges
        dfg._qf_canonical_edge_index = dfg_edge_index
        dfg._qf_canonical_actions = dfg_actions
    else:
        operation_index = dfg._qf_canonical_operation_index
        dfg_edges = dfg._qf_canonical_edges
        dfg_edge_index = dfg._qf_canonical_edge_index

    if not joint:
        dfg_actions = (
            (
                tuple(range(len(operations))),
                tuple(range(len(dfg_edges))),
            ),
        )

    placements = [-1] * len(operations)
    for op, node in state.placements.items():
        placements[operation_index[op]] = node_index[node]
    occupied_compute = tuple(node_index[node] for node in state.occupied_compute)
    occupied_links = tuple(state.occupied_links)
    # PartialState.serialize intentionally records routed dependency keys, not
    # their path values: future feasibility depends on the occupied-link set,
    # while the retained representative still carries its independently legal
    # concrete paths.
    routed_dependencies = [False] * len(dfg_edges)
    for edge in state.routed_edges:
        routed_dependencies[dfg_edge_index[edge]] = True

    best = None
    for (operation_map, dfg_edge_map), (
        node_map,
        arch_edge_map,
    ) in itertools.product(dfg_actions, arch_actions):
        transformed_placements = [-1] * len(placements)
        for op_idx, node_idx in enumerate(placements):
            if node_idx >= 0:
                transformed_placements[operation_map[op_idx]] = node_map[node_idx]
        transformed_dependencies = [False] * len(routed_dependencies)
        for edge_idx, present in enumerate(routed_dependencies):
            if present:
                transformed_dependencies[dfg_edge_map[edge_idx]] = True
        key = (
            tuple(transformed_placements),
            tuple(sorted(node_map[node] for node in occupied_compute)),
            tuple(sorted(arch_edge_map[link] for link in occupied_links)),
            tuple(transformed_dependencies),
        )
        if best is None or key < best:
            best = key
    assert best is not None
    return best, (time.perf_counter() - start) * 1000.0


def verify_all_symmetries(dfg: DFG, arch: Architecture) -> bool:
    return all(verify_dfg_permutation(dfg, p) for p in dfg.automorphisms) and all(
        verify_arch_permutation(arch, p) for p in arch.symmetries
    )
