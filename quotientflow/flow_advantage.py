"""FlowAdvantage targets, normalized features, and deterministic selectors."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from dataclasses import dataclass
from typing import Dict, List, Sequence

import networkx as nx
import numpy as np

from .action_lookahead import check_partial_legality, scarcity_adjusted_sum
from .arch import Architecture
from .dfg import DFG
from .mapper import Action
from .partial_state import PartialState, residual_graph
from .relaxation import RelaxationResult


BASE_FEATURE_NAMES = [
    "immediate_cost",
    "route_length_normalized",
    "new_link_fraction",
    "candidate_x",
    "candidate_y",
    "candidate_time",
    "operation_is_add",
    "operation_is_mul",
    "operation_indegree_fraction",
    "operation_outdegree_fraction",
    "partial_depth",
    "remaining_operation_fraction",
    "occupied_link_fraction",
    "remaining_routing_capacity_fraction",
    "largest_component_fraction",
    "saturated_link_fraction",
    "remaining_no_path_fraction",
    "cut_dependency_fraction",
    "alternative_count_fraction_min",
    "alternative_count_fraction_mean",
    "alternative_count_fraction_max",
    "mean_residual_distance_normalized",
    "dual_sum_per_new_link",
    "dual_max",
    "dual_top2_mean",
    "scarcity_adjusted_dual_mean",
    "positive_dual_link_fraction",
    "top_dual_decile_fraction",
    "selected_edge_betweenness_mean",
    "selected_edge_betweenness_max",
    "selected_bridge_fraction",
    "remaining_pair_shortest_path_exposure_mean",
    "remaining_pair_shortest_path_exposure_max",
    "residual_alternative_path_fraction",
    "demand_to_min_cut_ratio_mean",
    "demand_to_min_cut_ratio_max",
    "dual_weighted_path_congestion_mean",
    "dual_weighted_path_congestion_max",
    "active_capacity_link_fraction",
    "parent_objective_per_remaining_dependency",
    "parent_fractional_flow_concentration",
    "parent_slack_q0",
    "parent_slack_q25",
    "parent_slack_q50",
    "parent_slack_q75",
    "parent_slack_q100",
    "compute_dual_target",
    "routing_dual_baseline_component",
    "compute_dual_baseline_component",
    "dual_baseline",
]

RELATIVE_FEATURE_NAMES = [
    "immediate_cost_percentile",
    "dual_baseline_percentile",
    "route_length_percentile",
    "scarcity_cost_percentile",
    "immediate_cost_from_min",
    "dual_baseline_from_min",
    "route_length_from_min",
    "scarcity_cost_from_min",
    "immediate_cost_candidate_z",
    "dual_baseline_candidate_z",
    "route_length_candidate_z",
    "scarcity_cost_candidate_z",
]

FLOW_FEATURE_NAMES = BASE_FEATURE_NAMES + RELATIVE_FEATURE_NAMES
NO_DUAL_FEATURE_NAMES = [
    name for name in FLOW_FEATURE_NAMES
    if "dual" not in name
    and not name.startswith("parent_objective")
    and not name.startswith("parent_fractional")
    and not name.startswith("parent_slack")
]


def state_payload(state: PartialState) -> dict:
    return {
        "placements": sorted((int(k), list(v)) for k, v in state.placements.items()),
        "occupied_compute": sorted(map(list, state.occupied_compute)),
        "occupied_links": sorted(map(int, state.occupied_links)),
        "routed_edges": sorted(
            ([int(u), int(v)], list(map(int, path)))
            for (u, v), path in state.routed_edges.items()
        ),
        "route_cost": float(state.route_cost),
    }


def flow_state_hash(
    dfg: DFG, arch: Architecture, state: PartialState, config: dict
) -> str:
    payload = {
        "architecture": arch.name,
        "dfg_family": dfg.family,
        "dfg_seed": dfg.seed,
        "dfg_nodes": [
            (int(v), dict(sorted(dfg.graph.nodes[v].items())))
            for v in sorted(dfg.graph)
        ],
        "dfg_edges": sorted(map(list, dfg.graph.edges)),
        "state": state_payload(state),
        "relaxation": config,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def action_id(operation: int, action: Action) -> str:
    target = "-".join(map(str, action.target))
    route = "-".join(map(str, action.new_links)) or "none"
    state_suffix = hashlib.sha256(
        repr(action.state.serialize()).encode()
    ).hexdigest()[:10]
    return f"op{operation}_pe{target}_r{route}_s{state_suffix}"


def dual_baseline(action: Action, parent: RelaxationResult) -> tuple[float, float, float]:
    routing = (
        float(parent.raw_prices[list(action.new_links)].sum())
        if action.new_links else 0.0
    )
    compute = float(getattr(parent, "compute_duals", {}).get(action.target, 0.0))
    return float(action.base_cost + routing + compute), routing, compute


def robust_residual_scaler(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    center = float(np.median(values))
    q25, q75 = np.quantile(values, [0.25, 0.75])
    return center, max(float((q75 - q25) / 1.349), 1e-3)


def corrected_targets(
    immediate: np.ndarray,
    child_objective: np.ndarray,
    parent_objective: float,
    baseline: np.ndarray,
    residual_center: float | None = None,
    residual_scale: float | None = None,
) -> dict:
    delta = np.asarray(immediate) + np.asarray(child_objective) - parent_objective
    residual = delta - np.asarray(baseline)
    regret = delta - np.nanmin(delta)
    delta_range = float(np.nanmax(delta) - np.nanmin(delta))
    relative = regret / max(delta_range, 1.0)
    epsilon = max(1e-4, 0.01 * max(delta_range, 1.0))
    q25, q75 = np.nanquantile(regret, [0.25, 0.75])
    state_scale = max(float(q75 - q25), 0.1, 0.05 * max(delta_range, 1.0))
    logits = -regret / state_scale
    logits -= np.nanmax(logits)
    soft = np.exp(logits)
    soft /= np.nansum(soft)
    result = {
        "delta_star": delta,
        "residual_advantage": residual,
        "raw_regret": regret,
        "relative_regret": relative,
        "epsilon": epsilon,
        "epsilon_optimal": regret <= epsilon + 1e-12,
        "state_scale": state_scale,
        "soft_target": soft,
    }
    if residual_center is not None and residual_scale is not None:
        result["residual_target"] = np.clip(
            (residual - residual_center) / residual_scale, -8.0, 8.0
        )
    return result


def _endpoint_region(dfg, arch, state, operation):
    if operation in state.placements:
        return [state.placements[operation]]
    slot = dfg.graph.nodes[operation]["slot"]
    return [
        node for node in arch.compute_slots
        if node[2] == slot and node not in state.occupied_compute
    ]


@dataclass
class StateFeatureContext:
    residual: nx.DiGraph
    diameter: float
    edge_betweenness: Dict[tuple, float]
    bridges: set
    shortest_exposure: np.ndarray
    remaining_dependencies: List[tuple]
    remaining_no_path: int
    remaining_cut_dependencies: int
    mean_residual_distance: float
    largest_component: int
    residual_nodes: int
    remaining_capacity_fraction: float
    slack_quantiles: np.ndarray


def build_state_context(
    dfg: DFG,
    arch: Architecture,
    state: PartialState,
    parent: RelaxationResult,
) -> StateFeatureContext:
    graph = residual_graph(arch, state.occupied_links)
    undirected = graph.to_undirected()
    try:
        lengths = dict(nx.all_pairs_dijkstra_path_length(graph, weight="base_cost"))
        diameter = max(
            (distance for values in lengths.values() for distance in values.values()),
            default=1.0,
        )
    except Exception:
        lengths, diameter = {}, 1.0
    betweenness = nx.edge_betweenness_centrality(
        graph, normalized=True, weight="base_cost"
    ) if graph.number_of_edges() else {}
    bridges = set()
    if undirected.number_of_edges():
        for u, v in nx.bridges(undirected):
            bridges.add((u, v))
            bridges.add((v, u))
    remaining = [
        edge for edge in sorted(dfg.graph.edges)
        if edge not in state.routed_edges
    ]
    exposure = np.zeros(len(arch.edge_list), dtype=float)
    no_path = cut_count = 0
    minimum_distances = []
    for u, v in remaining:
        sources = _endpoint_region(dfg, arch, state, u)
        sinks = _endpoint_region(dfg, arch, state, v)
        best = None
        for source, sink in itertools.product(sources, sinks):
            try:
                path = nx.shortest_path(graph, source, sink, weight="base_cost")
                cost = nx.path_weight(graph, path, weight="base_cost")
                candidate = (cost, path)
                if best is None or candidate < best:
                    best = candidate
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
        if best is None:
            no_path += 1
        else:
            minimum_distances.append(float(best[0]))
            for a, b in zip(best[1], best[1][1:]):
                exposure[arch.edge_index[(a, b)]] += 1.0
        if arch.name == "cut4" and sources and sinks:
            left_s = all(node[0] <= 1 for node in sources)
            right_s = all(node[0] >= 2 for node in sources)
            left_t = all(node[0] <= 1 for node in sinks)
            right_t = all(node[0] >= 2 for node in sinks)
            cut_count += int((left_s and right_t) or (right_s and left_t))
    exposure /= max(1, len(remaining))
    components = list(nx.connected_components(undirected))
    slack = np.asarray(
        getattr(parent, "capacity_slack_quantiles", np.zeros(5)), dtype=float
    )
    if slack.shape != (5,) or not np.isfinite(slack).all():
        slack = np.zeros(5)
    return StateFeatureContext(
        residual=graph,
        diameter=max(float(diameter), 1.0),
        edge_betweenness=betweenness,
        bridges=bridges,
        shortest_exposure=exposure,
        remaining_dependencies=remaining,
        remaining_no_path=no_path,
        remaining_cut_dependencies=cut_count,
        mean_residual_distance=float(np.mean(minimum_distances))
        if minimum_distances else 0.0,
        largest_component=max(map(len, components), default=0),
        residual_nodes=max(1, graph.number_of_nodes()),
        remaining_capacity_fraction=graph.number_of_edges()
        / max(1, len(arch.edge_list)),
        slack_quantiles=slack,
    )


def _alternative_stats(arch, state, action, k_paths=4):
    graph = residual_graph(arch, state.occupied_links)
    new_dependencies = sorted(set(action.state.routed_edges) - set(state.routed_edges))
    counts, cuts = [], []
    for u, v in new_dependencies:
        source, sink = state.placements[u], action.target
        try:
            paths = list(itertools.islice(
                nx.shortest_simple_paths(graph, source, sink, weight="base_cost"),
                k_paths,
            ))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            paths = []
        counts.append(len(paths) / max(1, k_paths))
        # Bounded min-cut proxy consistent with the enumerated route universe.
        edge_sets = [set(zip(path, path[1:])) for path in paths]
        disjoint = 0
        used = set()
        for path_edges in edge_sets:
            if not path_edges & used:
                disjoint += 1
                used |= path_edges
        cuts.append(max(1, disjoint))
    if not counts:
        return (0.0, 0.0, 0.0), (0.0, 0.0)
    return (
        (float(min(counts)), float(np.mean(counts)), float(max(counts))),
        (float(np.mean([1.0 / c for c in cuts])), float(max(1.0 / c for c in cuts))),
    )


def action_feature_rows(
    dfg: DFG,
    arch: Architecture,
    state: PartialState,
    operation: int,
    actions: Sequence[Action],
    parent: RelaxationResult,
    context: StateFeatureContext | None = None,
) -> List[dict]:
    context = context or build_state_context(dfg, arch, state, parent)
    remaining_ops = len(dfg.graph) - len(state.placements) - 1
    remaining_deps = max(1, len(context.remaining_dependencies))
    prices = np.asarray(parent.raw_prices, dtype=float)
    positive = prices[prices > 1e-10]
    top_threshold = float(np.quantile(positive, .9)) if len(positive) else math.inf
    rows = []
    for action in actions:
        link_ids = list(action.new_links)
        values = prices[link_ids] if link_ids else np.zeros(0)
        alt, demand_cut = _alternative_stats(arch, state, action)
        bet = [
            context.edge_betweenness.get(arch.edge_list[e], 0.0) for e in link_ids
        ]
        bridge = [
            float(arch.edge_list[e] in context.bridges) for e in link_ids
        ]
        exposure = context.shortest_exposure[link_ids] if link_ids else np.zeros(0)
        baseline, routing_component, compute_component = dual_baseline(action, parent)
        scarcity = action.base_cost + scarcity_adjusted_sum(
            arch, state, action, prices
        )
        active = values > 1e-8
        row = {
            "immediate_cost": float(action.base_cost),
            "route_length_normalized": action.base_cost / context.diameter,
            "new_link_fraction": len(set(link_ids)) / max(1, len(arch.edge_list)),
            "candidate_x": action.target[0] / max(1, arch.width - 1),
            "candidate_y": action.target[1] / max(1, arch.height - 1),
            "candidate_time": action.target[2] / max(1, arch.ii - 1),
            "operation_is_add": float(dfg.graph.nodes[operation]["op"] == "ADD"),
            "operation_is_mul": float(dfg.graph.nodes[operation]["op"] == "MUL"),
            "operation_indegree_fraction": dfg.graph.in_degree(operation)
            / max(1, len(dfg.graph) - 1),
            "operation_outdegree_fraction": dfg.graph.out_degree(operation)
            / max(1, len(dfg.graph) - 1),
            "partial_depth": len(state.placements) / max(1, len(dfg.graph)),
            "remaining_operation_fraction": remaining_ops / max(1, len(dfg.graph)),
            "occupied_link_fraction": len(state.occupied_links)
            / max(1, len(arch.edge_list)),
            "remaining_routing_capacity_fraction": context.remaining_capacity_fraction,
            "largest_component_fraction": context.largest_component
            / context.residual_nodes,
            "saturated_link_fraction": len(action.state.occupied_links)
            / max(1, len(arch.edge_list)),
            "remaining_no_path_fraction": context.remaining_no_path / remaining_deps,
            "cut_dependency_fraction": context.remaining_cut_dependencies / remaining_deps,
            "alternative_count_fraction_min": alt[0],
            "alternative_count_fraction_mean": alt[1],
            "alternative_count_fraction_max": alt[2],
            "mean_residual_distance_normalized": context.mean_residual_distance
            / context.diameter,
            "dual_sum_per_new_link": float(values.sum()) / max(1, len(link_ids)),
            "dual_max": float(values.max()) if len(values) else 0.0,
            "dual_top2_mean": float(np.sort(values)[-2:].mean())
            if len(values) else 0.0,
            "scarcity_adjusted_dual_mean": (
                scarcity - action.base_cost
            ) / max(1, len(link_ids)),
            "positive_dual_link_fraction": float(np.mean(values > 1e-10))
            if len(values) else 0.0,
            "top_dual_decile_fraction": float(np.mean(values >= top_threshold))
            if len(values) else 0.0,
            "selected_edge_betweenness_mean": float(np.mean(bet)) if bet else 0.0,
            "selected_edge_betweenness_max": float(max(bet)) if bet else 0.0,
            "selected_bridge_fraction": float(np.mean(bridge)) if bridge else 0.0,
            "remaining_pair_shortest_path_exposure_mean": float(np.mean(exposure))
            if len(exposure) else 0.0,
            "remaining_pair_shortest_path_exposure_max": float(np.max(exposure))
            if len(exposure) else 0.0,
            "residual_alternative_path_fraction": alt[1],
            "demand_to_min_cut_ratio_mean": demand_cut[0],
            "demand_to_min_cut_ratio_max": demand_cut[1],
            "dual_weighted_path_congestion_mean": float(
                np.mean(values * (1.0 + exposure))
            ) if len(values) else 0.0,
            "dual_weighted_path_congestion_max": float(
                np.max(values * (1.0 + exposure))
            ) if len(values) else 0.0,
            "active_capacity_link_fraction": float(np.mean(active))
            if len(active) else 0.0,
            "parent_objective_per_remaining_dependency": parent.objective
            / remaining_deps,
            "parent_fractional_flow_concentration": float(
                getattr(parent, "fractional_flow_concentration", 0.0)
            ),
            "parent_slack_q0": context.slack_quantiles[0],
            "parent_slack_q25": context.slack_quantiles[1],
            "parent_slack_q50": context.slack_quantiles[2],
            "parent_slack_q75": context.slack_quantiles[3],
            "parent_slack_q100": context.slack_quantiles[4],
            "compute_dual_target": compute_component,
            "routing_dual_baseline_component": routing_component,
            "compute_dual_baseline_component": compute_component,
            "dual_baseline": baseline,
            "_scarcity_cost": scarcity,
        }
        rows.append(row)
    for source, prefix in (
        ("immediate_cost", "immediate_cost"),
        ("dual_baseline", "dual_baseline"),
        ("route_length_normalized", "route_length"),
        ("_scarcity_cost", "scarcity_cost"),
    ):
        values = np.asarray([row[source] for row in rows], dtype=float)
        order = np.argsort(values, kind="stable")
        ranks = np.empty(len(values), dtype=float)
        ranks[order] = np.arange(len(values), dtype=float)
        ranks /= max(1, len(values) - 1)
        mean, std = float(values.mean()), float(values.std())
        for i, row in enumerate(rows):
            row[f"{prefix}_percentile"] = ranks[i]
            row[f"{prefix}_from_min"] = values[i] - float(values.min())
            row[f"{prefix}_candidate_z"] = (
                (values[i] - mean) / std if std > 1e-8 else 0.0
            )
    for row in rows:
        row.pop("_scarcity_cost")
        assert set(row) == set(FLOW_FEATURE_NAMES)
    return rows


def select_training_actions(
    actions: Sequence[Action],
    features: Sequence[dict],
    limit: int = 12,
    use_parent_dual: bool = True,
) -> List[int]:
    """Solver-independent union; parent dual is optional, child values never used."""
    if len(actions) <= limit:
        return list(range(len(actions)))
    indices = list(range(len(actions)))
    chosen, seen = [], set()

    def add(index):
        signature = (actions[index].target, actions[index].new_links)
        if signature not in seen and len(chosen) < limit:
            chosen.append(index)
            seen.add(signature)
    length_rank = sorted(indices, key=lambda i: (
        actions[i].base_cost, actions[i].target, actions[i].new_links
    ))
    used_targets = set()
    target_rank = []
    for index in length_rank:
        if actions[index].target not in used_targets:
            target_rank.append(index)
            used_targets.add(actions[index].target)
    used_routes = set()
    route_rank = []
    for index in length_rank:
        route = actions[index].new_links
        if route not in used_routes:
            route_rank.append(index)
            used_routes.add(route)
    bottleneck_rank = sorted(indices, key=lambda i: (
        -features[i]["remaining_pair_shortest_path_exposure_max"],
        -features[i]["selected_edge_betweenness_max"],
        actions[i].target,
        actions[i].new_links,
    ))
    if use_parent_dual:
        dual_rank = sorted(indices, key=lambda i: (
            features[i]["dual_baseline"], actions[i].target, actions[i].new_links
        ))
        # Fixed on development audit before model fitting: four length, four
        # current-parent-dual, two bottleneck, two target-diverse, two
        # route-diverse. Deduplication keeps the final union at most 12.
        ranked_groups = (
            (length_rank, 4), (dual_rank, 4), (bottleneck_rank, 2),
            (target_rank, 2), (route_rank, 2),
        )
    else:
        ranked_groups = (
            (length_rank, 4), (bottleneck_rank, 4),
            (target_rank, 4), (route_rank, 4),
        )
    for ranking, count in ranked_groups:
        for index in ranking[:count]:
            add(index)
    for ranking, _ in ranked_groups:
        for index in ranking:
            add(index)
    return chosen


def target_complete_path_bounded(
    actions: Sequence[Action], limit_per_target: int = 4
) -> List[Action]:
    """Keep every target and at most the cheapest paths per target."""
    grouped = {}
    for action in sorted(
        actions,
        key=lambda a: (a.target, a.base_cost, a.new_links, a.state.serialize()),
    ):
        grouped.setdefault(action.target, []).append(action)
    return [
        action
        for target in sorted(grouped)
        for action in grouped[target][:limit_per_target]
    ]
