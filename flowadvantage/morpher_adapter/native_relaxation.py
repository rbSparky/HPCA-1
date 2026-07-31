"""Regularized fractional residual mapping on Morpher's native MRRG.

This module is intentionally separate from :mod:`quotientflow.relaxation`.
It uses the actual port-level, II-expanded graph exported by Morpher and never
projects that graph onto the synthetic CGRA representation.

The relaxation couples fractional operation placement to multi-commodity
routing at the exact native DataPath output and operand-port endpoints.  Native
resource sharing is represented per *source value*, matching Morpher's
broadcast semantics.  Basic-block mutex values may share a resource, while
incompatible values consume its capacity.  Both native resource and directed
edge capacities are enforced, including occupation fixed by the partial state.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import cvxpy as cp
import networkx as nx
import numpy as np
import scipy.sparse as sp

from quotientflow.relaxation import normalize_prices

from .native_mapper import (
    NativeAction,
    NativeMappingState,
    NativeMorpherProblem,
    NativeSignal,
    _dependency_key,
)


@dataclass(frozen=True)
class NativeRelaxationConfig:
    """Numerical and finite-schedule configuration of the native relaxation."""

    tau: float = 1e-3
    extra_ii_periods: int = 1
    fallback_extra_ii_periods: int = 0
    solver_primary: str = "CLARABEL"
    solver_fallback: str = "OSQP"
    max_threads: int = 1
    max_solve_seconds: float = 120.0
    feasibility_tolerance: float = 1e-7
    reachable_edge_pruning: bool = True
    structure_cache_enabled: bool = True
    cache_version: str = "native_mrrg_fractional_v4"

    def __post_init__(self) -> None:
        if self.tau <= 0:
            raise ValueError("tau must be positive")
        if self.extra_ii_periods < 0:
            raise ValueError("extra_ii_periods must be nonnegative")
        if self.fallback_extra_ii_periods < 0:
            raise ValueError("fallback_extra_ii_periods must be nonnegative")
        if self.max_solve_seconds <= 0:
            raise ValueError("max_solve_seconds must be positive")
        if self.max_threads <= 0:
            raise ValueError("max_threads must be positive")


@dataclass
class NativeRelaxationResult:
    status: str
    solver: str
    solve_seconds: float
    canonicalization_seconds: float
    objective: float
    assignment_residual: float
    flow_residual: float
    compute_capacity_violation: float
    routing_capacity_violation: float
    min_variable: float
    max_variable: float
    routing_resource_duals: dict[str, float] = field(default_factory=dict)
    routing_edge_duals: dict[str, float] = field(default_factory=dict)
    compute_duals: dict[str, float] = field(default_factory=dict)
    normalized_resource_duals: dict[str, float] = field(default_factory=dict)
    normalized_edge_duals: dict[str, float] = field(default_factory=dict)
    capacity_slacks: dict[str, float] = field(default_factory=dict)
    fractional_flow_concentration: float = 0.0
    remaining_operations: int = 0
    remaining_dependencies: int = 0
    placement_variables: int = 0
    flow_variables: int = 0
    full_flow_variables: int = 0
    flow_edge_reduction_fraction: float = 0.0
    cache_key: str = ""
    cache_hit: bool = False
    error: str = ""

    @property
    def solved(self) -> bool:
        return self.status in {"optimal", "optimal_inaccurate"}

    @property
    def feasible(self) -> bool:
        return self.solved and max(
            self.assignment_residual,
            self.flow_residual,
            self.compute_capacity_violation,
            self.routing_capacity_violation,
        ) <= 1e-4

    def health_dict(self) -> dict[str, Any]:
        result = asdict(self)
        for key in (
            "routing_resource_duals",
            "routing_edge_duals",
            "compute_duals",
            "normalized_resource_duals",
            "normalized_edge_duals",
            "capacity_slacks",
        ):
            result.pop(key)
        result["solved"] = self.solved
        result["feasible"] = self.feasible
        return result


@dataclass(frozen=True)
class NativeChildEvaluation:
    action_key: tuple[Any, ...]
    immediate_cost: float
    residual_objective: float
    q_rel: float
    residual_feasible: bool
    solve_status: str
    solve_seconds: float
    cache_hit: bool
    error: str = ""


@dataclass(frozen=True)
class _NativeProblemIndex:
    """Immutable sparse indexing shared by every state of one native problem."""

    resource_ids: tuple[str, ...]
    resource_index: dict[str, int]
    native_edges: tuple[tuple[str, str], ...]
    edge_index: dict[tuple[str, str], int]
    edge_ids: tuple[str, ...]


def _jsonable(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset, list)):
        return [_jsonable(item) for item in sorted(value, key=repr)]
    return repr(value)


def native_relaxation_cache_key(
    problem: NativeMorpherProblem,
    state: NativeMappingState,
    config: NativeRelaxationConfig,
) -> str:
    payload = {
        "architecture_hash": problem.mrrg.get("architecture_hash"),
        "dfg_hash": problem.dfg.get("dfg_hash"),
        "ii": problem.ii,
        "state": _jsonable(state.stable_key()),
        "config": asdict(config),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


class NativeRelaxationSolver:
    """Exact native-contract relaxation with deterministic result caching."""

    def __init__(
        self,
        config: NativeRelaxationConfig = NativeRelaxationConfig(),
        *,
        cache_dir: str | Path | None = None,
    ) -> None:
        self.config = config
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._memory_cache: dict[str, NativeRelaxationResult] = {}
        # A worker maps many states of the same immutable DFG/MRRG.  Keep the
        # expensive deterministic native-ID indexing and state-independent
        # sparse structures once.  Entries retain the problem object alongside
        # its id so an id recycled by Python can never alias another problem.
        self._problem_indices: dict[
            int, tuple[NativeMorpherProblem, _NativeProblemIndex]
        ] = {}
        self._clique_cache: dict[
            tuple[int, tuple[str, ...]], tuple[tuple[str, ...], ...]
        ] = {}
        self._incidence_cache: OrderedDict[
            tuple[int, tuple[int, ...]],
            tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix],
        ] = OrderedDict()
        self._corridor_cache: OrderedDict[
            tuple[Any, ...], np.ndarray
        ] = OrderedDict()
        self.structure_cache_hits = 0
        self.structure_cache_misses = 0
        # Native MRRG topology is immutable across states.  Reachability used
        # to rebuild phase tables and adjacency lists for every dependency;
        # retain the exact indexed graph once per problem/edge ordering.
        self._reachability_graph_cache: dict[
            tuple[int, tuple[tuple[str, str], ...]],
            tuple[dict[str, int], dict[str, tuple[tuple[int, str, int], ...]], dict[str, tuple[tuple[int, str, int], ...]]],
        ] = {}
        self.solve_calls = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.schedule_padding_fallbacks = 0

    def _problem_index(self, problem: NativeMorpherProblem) -> _NativeProblemIndex:
        identity = id(problem)
        cached = (
            self._problem_indices.get(identity)
            if self.config.structure_cache_enabled
            else None
        )
        if cached is not None and cached[0] is problem:
            self.structure_cache_hits += 1
            return cached[1]
        resource_ids = tuple(sorted(problem.resources))
        native_edges = tuple(
            sorted(
                {
                    (str(edge["src"]), str(edge["dst"]))
                    for edge in problem.mrrg.get("edges", [])
                }
            )
        )
        index = _NativeProblemIndex(
            resource_ids=resource_ids,
            resource_index={rid: column for column, rid in enumerate(resource_ids)},
            native_edges=native_edges,
            edge_index={edge: column for column, edge in enumerate(native_edges)},
            edge_ids=tuple(f"{left}->{right}" for left, right in native_edges),
        )
        if self.config.structure_cache_enabled:
            self._problem_indices[identity] = (problem, index)
        self.structure_cache_misses += 1
        return index

    @staticmethod
    def _bounded_put(cache: OrderedDict, key: Any, value: Any, limit: int) -> None:
        """Insert an immutable derived structure without unbounded worker RSS."""

        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > limit:
            cache.popitem(last=False)

    def solve(
        self,
        problem: NativeMorpherProblem,
        state: NativeMappingState,
    ) -> NativeRelaxationResult:
        if state.problem is not problem:
            raise ValueError("state belongs to a different NativeMorpherProblem")
        self.solve_calls += 1
        key = native_relaxation_cache_key(problem, state, self.config)
        cached = self._load_cache(key)
        if cached is not None:
            self.cache_hits += 1
            cached.cache_hit = True
            return cached
        self.cache_misses += 1
        result = self._solve_uncached(problem, state, key)
        # Some Morpher kernels expose a finite ASAP/ALAP window whose legal
        # route corridor is empty at the zero-padding interval, while the
        # native mapper legitimately uses one additional modulo period for a
        # long cross-PE dependency.  Try the explicitly configured expanded
        # interval only for that structural infeasibility.  This is not a
        # heuristic relaxation: it is the exact original model's feasible
        # interval, reached lazily after the smaller exact interval proves it
        # has no temporal corridor.
        if (
            result.status == "infeasible"
            and self.config.fallback_extra_ii_periods
            and "no legal temporal native route edges" in result.error
            and self.config.fallback_extra_ii_periods > self.config.extra_ii_periods
        ):
            fallback_config = NativeRelaxationConfig(
                **{
                    **asdict(self.config),
                    "extra_ii_periods": self.config.fallback_extra_ii_periods,
                    "fallback_extra_ii_periods": 0,
                }
            )
            fallback_solver = NativeRelaxationSolver(
                fallback_config, cache_dir=self.cache_dir
            )
            fallback_result = fallback_solver.solve(problem, state)
            self.schedule_padding_fallbacks += 1
            fallback_result.error = (
                f"{fallback_result.error}; schedule_padding_fallback="
                f"{self.config.fallback_extra_ii_periods}"
            ).strip("; ")
            result = fallback_result
        self._memory_cache[key] = result
        if self.cache_dir is not None:
            self._write_cache(key, result)
        return result

    def _cache_path(self, key: str) -> Path:
        assert self.cache_dir is not None
        return self.cache_dir / key[:2] / f"{key}.json"

    def _load_cache(self, key: str) -> NativeRelaxationResult | None:
        if key in self._memory_cache:
            return NativeRelaxationResult(**asdict(self._memory_cache[key]))
        if self.cache_dir is None:
            return None
        path = self._cache_path(key)
        if not path.exists():
            return None
        try:
            raw = json.loads(path.read_text())
            if raw.pop("_cache_schema") != self.config.cache_version:
                return None
            result = NativeRelaxationResult(**raw)
            self._memory_cache[key] = result
            return NativeRelaxationResult(**asdict(result))
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            return None

    def _write_cache(self, key: str, result: NativeRelaxationResult) -> None:
        path = self._cache_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"_cache_schema": self.config.cache_version, **asdict(result)}
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        with temporary.open("w") as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)

    @staticmethod
    def _pad_native_result(
        result: NativeRelaxationResult,
        problem_index: _NativeProblemIndex,
    ) -> NativeRelaxationResult:
        """Restore complete native-ID dictionaries after active-subgraph solve."""

        result.routing_resource_duals = {
            rid: float(result.routing_resource_duals.get(rid, 0.0))
            for rid in problem_index.resource_ids
        }
        result.normalized_resource_duals = {
            rid: float(result.normalized_resource_duals.get(rid, 0.0))
            for rid in problem_index.resource_ids
        }
        result.capacity_slacks = {
            rid: float(result.capacity_slacks.get(rid, 1.0))
            for rid in problem_index.resource_ids
        }
        result.routing_edge_duals = {
            edge_id: float(result.routing_edge_duals.get(edge_id, 0.0))
            for edge_id in problem_index.edge_ids
        }
        result.normalized_edge_duals = {
            edge_id: float(result.normalized_edge_duals.get(edge_id, 0.0))
            for edge_id in problem_index.edge_ids
        }
        result.full_flow_variables = (
            result.remaining_dependencies * len(problem_index.edge_ids)
        )
        result.flow_edge_reduction_fraction = (
            0.0
            if result.full_flow_variables <= 0
            else 1.0
            - float(result.flow_variables) / float(result.full_flow_variables)
        )
        return result

    def _solve_uncached(
        self,
        problem: NativeMorpherProblem,
        state: NativeMappingState,
        cache_key: str,
    ) -> NativeRelaxationResult:
        start = time.perf_counter()
        remaining = [
            key for key in problem.operation_order() if key not in state.placements
        ]
        dependencies = [
            dependency
            for key, dependency in sorted(problem.dependencies.items())
            if key not in state.routes
            and bool(
                dependency.get(
                    "requires_route",
                    str(dependency.get("edge_type", "")).upper() != "PS",
                )
            )
        ]
        if not remaining and not dependencies:
            return NativeRelaxationResult(
                status="optimal",
                solver="closed_form_complete",
                solve_seconds=time.perf_counter() - start,
                canonicalization_seconds=0.0,
                objective=0.0,
                assignment_residual=0.0,
                flow_residual=0.0,
                compute_capacity_violation=0.0,
                routing_capacity_violation=0.0,
                min_variable=0.0,
                max_variable=0.0,
                remaining_operations=0,
                remaining_dependencies=0,
                cache_key=cache_key,
            )

        problem_index = self._problem_index(problem)
        # Keep the complete native index for provenance/dual padding, but use
        # a state-local active subgraph for CVXPY.  Resources and links that
        # cannot lie on any legal source-to-sink corridor have identically zero
        # flow and zero dual; omitting them is an exact sparse reduction.
        all_resource_ids = problem_index.resource_ids
        all_native_edges = problem_index.native_edges

        # Placement candidates are exact native FU/DataPath/absolute-latency
        # assignments within the declared Morpher schedule horizon.
        candidates: list[Any] = []
        candidate_by_operation: dict[str, list[int]] = {}
        for node_key in remaining:
            node = problem.nodes[node_key]
            earliest = int(node.get("asap", 0))
            latest = max(earliest, int(node.get("alap", earliest)))
            latest += self.config.extra_ii_periods * problem.ii
            values = problem.placement_candidates(
                node_key,
                earliest_latency=earliest,
                latest_latency=latest,
                state=state,
            )
            if not values:
                return _native_failure(
                    "infeasible",
                    "none",
                    start,
                    cache_key,
                    len(remaining),
                    len(dependencies),
                    f"no legal native placement candidate for {node_key}",
                )
            indices = []
            for placement in values:
                indices.append(len(candidates))
                candidates.append(placement)
            candidate_by_operation[node_key] = indices

        # Necessary-condition pruning before any CVXPY variable is created.
        # A placement whose native endpoint is absent from the exact temporal
        # corridor of one of its incident dependencies can never carry a
        # legal route.  Removing only those candidates is algebraically exact
        # (the corresponding fractional variables are forced to zero), while
        # substantially reducing the root-state canonicalization problem.
        if dependencies:
            provisional_corridors: list[np.ndarray] = []
            for dependency in dependencies:
                columns = self._reachable_edge_columns(
                    problem,
                    state,
                    candidates,
                    candidate_by_operation,
                    dependency,
                    all_native_edges,
                )
                if columns.size == 0:
                    source_key, destination_key = _dependency_key(dependency)
                    return _native_failure(
                        "infeasible",
                        "none",
                        start,
                        cache_key,
                        len(remaining),
                        len(dependencies),
                        f"no legal temporal native route edges for {source_key}->{destination_key}",
                        placement_variables=len(candidates),
                    )
                provisional_corridors.append(columns)
            kept_old: set[int] = set()
            for node_key in remaining:
                incident = [
                    (
                        dependency,
                        {
                            all_native_edges[int(column)][side]
                            for column in columns
                            for side in (0, 1)
                        },
                    )
                    for dependency, columns in zip(
                        dependencies, provisional_corridors
                    )
                    if node_key in _dependency_key(dependency)
                ]
                for old_index in candidate_by_operation[node_key]:
                    placement = candidates[old_index]
                    valid = True
                    for dependency, edge_resources in incident:
                        source_key, destination_key = _dependency_key(dependency)
                        edge_type = str(dependency.get("edge_type", ""))
                        endpoint = (
                            problem.output_port(placement)
                            if source_key == node_key
                            else problem.operand_port(placement, edge_type)
                        )
                        if endpoint not in edge_resources:
                            valid = False
                            break
                    if valid:
                        kept_old.add(old_index)
            if len(kept_old) < len(candidates):
                remap = {
                    old_index: new_index
                    for new_index, old_index in enumerate(
                        index for index in range(len(candidates))
                        if index in kept_old
                    )
                }
                candidates = [
                    placement
                    for index, placement in enumerate(candidates)
                    if index in kept_old
                ]
                candidate_by_operation = {
                    node_key: [remap[index] for index in indices if index in remap]
                    for node_key, indices in candidate_by_operation.items()
                }
                if any(not indices for indices in candidate_by_operation.values()):
                    return _native_failure(
                        "infeasible",
                        "none",
                        start,
                        cache_key,
                        len(remaining),
                        len(dependencies),
                        "endpoint-corridor candidate pruning removed an operation's support",
                    )

        x = cp.Variable(len(candidates), name="native_x")
        constraints: list[cp.Constraint] = [x >= 0.0, x <= 1.0]
        assignment_constraints: list[tuple[str, list[int], cp.Constraint]] = []
        for node_key in remaining:
            indices = candidate_by_operation[node_key]
            constraint = cp.sum(x[indices]) == 1.0
            constraints.append(constraint)
            assignment_constraints.append((node_key, indices, constraint))

        dp_to_candidates: dict[str, list[int]] = {}
        for index, placement in enumerate(candidates):
            dp_to_candidates.setdefault(placement.dp_id, []).append(index)
        compute_constraints: list[tuple[str, list[int], cp.Constraint]] = []
        for dp_id, indices in sorted(dp_to_candidates.items()):
            residual_capacity = 0.0 if state.compute_occupied(dp_id) else 1.0
            constraint = cp.sum(x[indices]) <= residual_capacity
            constraints.append(constraint)
            compute_constraints.append((dp_id, indices, constraint))

        # Absolute schedule feasibility is retained even though routing
        # capacity is expressed on Morpher's modulo-expanded native graph.
        # This prevents the fractional endpoint coupling from assigning mass
        # to a producer/consumer pair that no native temporal route can meet.
        for dependency in dependencies:
            source_key, destination_key = _dependency_key(dependency)
            distance = int(dependency.get("iteration_distance", 0) or 0)
            source_options: list[tuple[int | None, Any]]
            destination_options: list[tuple[int | None, Any]]
            if source_key in state.placements:
                source_options = [(None, state.placements[source_key])]
            else:
                source_options = [
                    (index, candidates[index])
                    for index in candidate_by_operation[source_key]
                ]
            if destination_key in state.placements:
                destination_options = [
                    (None, state.placements[destination_key])
                ]
            else:
                destination_options = [
                    (index, candidates[index])
                    for index in candidate_by_operation[destination_key]
                ]
            for source_candidate, source_placement in source_options:
                for destination_candidate, destination_placement in destination_options:
                    if (
                        source_placement.output_latency
                        <= destination_placement.latency + distance * problem.ii
                    ):
                        continue
                    if source_candidate is None and destination_candidate is None:
                        return _native_failure(
                            "infeasible",
                            "none",
                            start,
                            cache_key,
                            len(remaining),
                            len(dependencies),
                            "fixed partial state violates dependency timing "
                            f"{source_key}->{destination_key}",
                        )
                    if source_candidate is None:
                        constraints.append(x[destination_candidate] == 0.0)
                    elif destination_candidate is None:
                        constraints.append(x[source_candidate] == 0.0)
                    else:
                        constraints.append(
                            x[source_candidate] + x[destination_candidate] <= 1.0
                        )

        if not dependencies:
            objective = cp.Minimize(self.config.tau / 2.0 * cp.sum_squares(x))
            cvx_problem = cp.Problem(objective, constraints)
            return self._pad_native_result(self._run_problem(
                cvx_problem,
                start=start,
                canonical_seconds=0.0,
                cache_key=cache_key,
                x=x,
                f=None,
                z=None,
                w=None,
                candidates=candidates,
                dependencies=dependencies,
                resource_ids=resource_ids,
                edge_ids=edge_ids,
                incidence=None,
                endpoint_rhs=[],
                assignment_constraints=assignment_constraints,
                compute_constraints=compute_constraints,
                routing_constraints=[],
                dependency_source_rows=(),
            ), problem_index)

        source_keys = tuple(
            sorted({_dependency_key(dependency)[0] for dependency in dependencies})
        )
        source_index = {key: index for index, key in enumerate(source_keys)}
        global_columns_by_dependency: list[np.ndarray] = []
        for dependency in dependencies:
            columns = self._reachable_edge_columns(
                problem,
                state,
                candidates,
                candidate_by_operation,
                dependency,
                all_native_edges,
            )
            if columns.size == 0:
                source_key, destination_key = _dependency_key(dependency)
                return _native_failure(
                    "infeasible",
                    "none",
                    start,
                    cache_key,
                    len(remaining),
                    len(dependencies),
                    f"no legal temporal native route edges for {source_key}->{destination_key}",
                    placement_variables=len(candidates),
                )
            global_columns_by_dependency.append(columns)
        active_global_edges = np.asarray(
            sorted({int(column) for columns in global_columns_by_dependency for column in columns}),
            dtype=int,
        )
        native_edges = tuple(all_native_edges[int(column)] for column in active_global_edges)
        global_to_local_edge = {
            int(global_column): local_column
            for local_column, global_column in enumerate(active_global_edges)
        }
        edge_ids = tuple(f"{left}->{right}" for left, right in native_edges)
        endpoint_resources: set[str] = set()
        for dependency in dependencies:
            source_key, destination_key = _dependency_key(dependency)
            edge_type = str(dependency.get("edge_type", ""))
            for endpoint_key, is_source in ((source_key, True), (destination_key, False)):
                placements = (
                    [state.placements[endpoint_key]]
                    if endpoint_key in state.placements
                    else [candidates[index] for index in candidate_by_operation[endpoint_key]]
                )
                for placement in placements:
                    endpoint_resources.add(
                        problem.output_port(placement)
                        if is_source
                        else problem.operand_port(placement, edge_type)
                    )
        active_resources = set(endpoint_resources)
        for left, right in native_edges:
            active_resources.update((left, right))
        resource_ids = tuple(sorted(active_resources))
        resource_index = {rid: column for column, rid in enumerate(resource_ids)}
        r_count, e_count = len(resource_ids), len(native_edges)
        z = cp.Variable((len(source_keys), r_count), nonneg=True, name="native_z")
        w = cp.Variable((len(source_keys), e_count), nonneg=True, name="native_w")
        # Flattened views make the native capacity matrices explicit sparse
        # linear operators.  Advanced 2-D CVXPY indexing otherwise expands a
        # large dense expression graph during canonicalization.
        z_flat = cp.reshape(
            z, (len(source_keys) * r_count,), order="C"
        )
        w_flat = cp.reshape(
            w, (len(source_keys) * e_count,), order="C"
        )
        constraints.extend((z <= 1.0, w <= 1.0))

        endpoint_rhs: list[Any] = []
        flow_constraints: list[cp.Constraint] = []
        flows: list[cp.Variable] = []
        commodity_columns: list[np.ndarray] = []
        commodity_incidences: list[sp.csr_matrix] = []
        commodity_outgoing: list[sp.csr_matrix] = []
        commodity_incoming: list[sp.csr_matrix] = []
        for dependency_index, dependency in enumerate(dependencies):
            source_key, destination_key = _dependency_key(dependency)
            global_columns = global_columns_by_dependency[dependency_index]
            columns = np.asarray(
                [global_to_local_edge[int(value)] for value in global_columns],
                dtype=int,
            )
            commodity_columns.append(columns)
            incidence_key = (id(problem), tuple(int(value) for value in columns))
            cached_incidence = (
                self._incidence_cache.get(incidence_key)
                if self.config.structure_cache_enabled
                else None
            )
            if cached_incidence is None:
                local_rows: list[int] = []
                local_cols: list[int] = []
                local_values: list[float] = []
                for local_column, global_column in enumerate(columns):
                    source_resource, destination_resource = native_edges[
                        int(global_column)
                    ]
                    local_rows.extend(
                        (
                            resource_index[source_resource],
                            resource_index[destination_resource],
                        )
                    )
                    local_cols.extend((local_column, local_column))
                    local_values.extend((1.0, -1.0))
                local_incidence = sp.csr_matrix(
                    (local_values, (local_rows, local_cols)),
                    shape=(r_count, len(columns)),
                )
                local_outgoing = local_incidence.maximum(0.0).tocsr()
                local_incoming = (-local_incidence).maximum(0.0).tocsr()
                if self.config.structure_cache_enabled:
                    self._bounded_put(
                        self._incidence_cache,
                        incidence_key,
                        (local_incidence, local_outgoing, local_incoming),
                        256,
                    )
                self.structure_cache_misses += 1
            else:
                self._incidence_cache.move_to_end(incidence_key)
                local_incidence, local_outgoing, local_incoming = cached_incidence
                self.structure_cache_hits += 1
            commodity_incidences.append(local_incidence)
            commodity_outgoing.append(local_outgoing)
            commodity_incoming.append(local_incoming)
            flow = cp.Variable(len(columns), name=f"native_f_{dependency_index}")
            flows.append(flow)
            constraints.extend((flow >= 0.0, flow <= 1.0))
            source_distribution = self._endpoint_distribution(
                problem,
                state,
                candidates,
                candidate_by_operation,
                x,
                source_key,
                str(dependency.get("edge_type", "")),
                source=True,
                resource_index=resource_index,
            )
            destination_distribution = self._endpoint_distribution(
                problem,
                state,
                candidates,
                candidate_by_operation,
                x,
                destination_key,
                str(dependency.get("edge_type", "")),
                source=False,
                resource_index=resource_index,
            )
            rhs = source_distribution - destination_distribution
            endpoint_rhs.append(rhs)
            flow_constraint = local_incidence @ flow == rhs
            constraints.append(flow_constraint)
            flow_constraints.append(flow_constraint)
            source_row = source_index[source_key]
            # ``max(outgoing, incoming) <= z`` is exactly equivalent to the
            # two sparse inequalities below.  Avoiding a per-resource
            # elementwise CVXPY maximum is important: its epigraph expansion
            # dominates canonicalization for large native MRRGs.
            constraints.extend(
                (
                    local_outgoing @ flow <= z[source_row, :],
                    local_incoming @ flow <= z[source_row, :],
                    flow <= w[source_row, columns],
                )
            )

        # Fixed occupancy and native broadcast/mutex semantics.
        fixed_resource_sources = {
            rid: {signal.source_key for signal in state.resource_signals.get(rid, ())}
            for rid in resource_ids
        }
        fixed_edge_sources: dict[tuple[str, str], set[str]] = {
            edge: set() for edge in native_edges
        }
        for route in state.routes.values():
            for edge in zip(route.resource_ids, route.resource_ids[1:]):
                if edge in fixed_edge_sources:
                    fixed_edge_sources[edge].add(route.source_key)

        resource_allowed = np.ones((len(source_keys), r_count), dtype=float)
        edge_allowed = np.ones((len(source_keys), e_count), dtype=float)
        for source_row, source_key in enumerate(source_keys):
            source_node = problem.nodes[source_key]
            dummy = NativeSignal(
                source_key=source_key,
                destination_node=int(source_node["dfg_node_id"]),
                latency=0,
                source_node=int(source_node["dfg_node_id"]),
            )
            for resource_column, rid in enumerate(resource_ids):
                if not state.can_occupy(rid, dummy):
                    resource_allowed[source_row, resource_column] = 0.0
            for edge_column, edge in enumerate(native_edges):
                existing = fixed_edge_sources[edge]
                if existing and not all(
                    state._source_mutex(source_key, other) for other in existing
                ):
                    edge_allowed[source_row, edge_column] = 0.0
        # Keep these masks sparse.  The previous implementation emitted a
        # dense CVXPY elementwise inequality for every source/resource and
        # source/edge pair.  On native Morpher MRRGs that expression graph is
        # needlessly enormous (and can consume many GB before the solver is
        # reached).  A zero equality on the forbidden indices is exactly the
        # same constraint, while retaining sparse indexed CVXPY expressions.
        for source_row in range(len(source_keys)):
            forbidden_resources = np.flatnonzero(
                resource_allowed[source_row] <= 0.0
            )
            if forbidden_resources.size:
                flat_indices = source_row * r_count + forbidden_resources
                constraints.append(z_flat[flat_indices] == 0.0)
            forbidden_edges = np.flatnonzero(edge_allowed[source_row] <= 0.0)
            if forbidden_edges.size:
                flat_indices = source_row * e_count + forbidden_edges
                constraints.append(w_flat[flat_indices] == 0.0)

        clique_key = (id(problem), source_keys)
        incompatibility_cliques = (
            self._clique_cache.get(clique_key)
            if self.config.structure_cache_enabled
            else None
        )
        if incompatibility_cliques is None:
            incompatibility_cliques = self._incompatibility_cliques(
                problem, source_keys
            )
            if self.config.structure_cache_enabled:
                self._clique_cache[clique_key] = incompatibility_cliques
            self.structure_cache_misses += 1
        else:
            self.structure_cache_hits += 1
        routing_constraints: list[
            tuple[
                str,
                tuple[int, ...],
                np.ndarray,
                cp.Constraint,
                np.ndarray,
                np.ndarray,
            ]
        ] = []
        non_mux_columns = np.asarray(
            [
                index
                for index, rid in enumerate(resource_ids)
                if not bool(problem.resources[rid].get("allows_operand_mux"))
            ],
            dtype=int,
        )
        for clique in incompatibility_cliques:
            rows_in_clique = tuple(source_index[key] for key in clique)
            coefficients = np.ones(
                (len(rows_in_clique), len(non_mux_columns)), dtype=float
            )
            fixed_count = np.zeros(len(non_mux_columns), dtype=float)
            for local_column, resource_column in enumerate(non_mux_columns):
                rid = resource_ids[resource_column]
                fixed = fixed_resource_sources[rid]
                for row_offset, key in enumerate(clique):
                    if key in fixed:
                        coefficients[row_offset, local_column] = 0.0
                if any(key in fixed for key in clique):
                    fixed_count[local_column] = 1.0
            # All non-fixed entries in a clique have unit coefficient.  The
            # fixed entries are represented in ``fixed_count``.  Avoiding a
            # dense multiply here preserves the exact native capacity model
            # but makes the expression graph proportional to the indexed
            # variables rather than to a dense clique-by-resource matrix.
            # A fixed source may be allowed to keep its own native signal in
            # the resource variable, so it cannot be removed by the generic
            # forbidden mask.  Remove fixed rows explicitly for each column;
            # the grouped expression below is exact because its RHS is zero
            # on columns with a fixed occupant and the fixed rows are omitted.
            resource_rows: list[int] = []
            resource_cols: list[int] = []
            resource_data: list[float] = []
            for column_offset, resource_column in enumerate(non_mux_columns):
                rid = resource_ids[resource_column]
                for row, key in zip(rows_in_clique, clique):
                    if key not in fixed_resource_sources[rid]:
                        resource_rows.append(column_offset)
                        resource_cols.append(row * r_count + int(resource_column))
                        resource_data.append(1.0)
            resource_matrix = sp.csr_matrix(
                (resource_data, (resource_rows, resource_cols)),
                shape=(len(non_mux_columns), len(source_keys) * r_count),
            )
            constraint = resource_matrix @ z_flat <= 1.0 - fixed_count
            constraints.append(constraint)
            routing_constraints.append(
                (
                    "resource",
                    rows_in_clique,
                    non_mux_columns,
                    constraint,
                    coefficients,
                    fixed_count,
                )
            )

            edge_coefficients = np.ones(
                (len(rows_in_clique), e_count), dtype=float
            )
            edge_fixed_count = np.zeros(e_count, dtype=float)
            for edge_column, edge in enumerate(native_edges):
                fixed = fixed_edge_sources[edge]
                for row_offset, key in enumerate(clique):
                    if key in fixed:
                        edge_coefficients[row_offset, edge_column] = 0.0
                if any(key in fixed for key in clique):
                    edge_fixed_count[edge_column] = 1.0
            edge_rows: list[int] = []
            edge_cols: list[int] = []
            edge_data: list[float] = []
            for edge_column, edge in enumerate(native_edges):
                fixed = fixed_edge_sources[edge]
                for row, key in zip(rows_in_clique, clique):
                    if key not in fixed:
                        edge_rows.append(edge_column)
                        edge_cols.append(row * e_count + edge_column)
                        edge_data.append(1.0)
            edge_matrix = sp.csr_matrix(
                (edge_data, (edge_rows, edge_cols)),
                shape=(e_count, len(source_keys) * e_count),
            )
            edge_constraint = edge_matrix @ w_flat <= 1.0 - edge_fixed_count
            constraints.append(edge_constraint)
            routing_constraints.append(
                (
                    "edge",
                    rows_in_clique,
                    np.arange(e_count),
                    edge_constraint,
                    edge_coefficients,
                    edge_fixed_count,
                )
            )

        # Morpher's initial native routing cost is one per directed MRRG
        # transition.  The strongly convex term makes primal routing and dual
        # prices deterministic when several native paths have equal length.
        objective = cp.Minimize(
            sum((cp.sum(flow) for flow in flows))
            + self.config.tau
            / 2.0
            * (
                cp.sum_squares(x)
                + sum((cp.sum_squares(flow) for flow in flows))
            )
        )
        cvx_problem = cp.Problem(objective, constraints)
        return self._pad_native_result(self._run_problem(
            cvx_problem,
            start=start,
            canonical_seconds=0.0,
            cache_key=cache_key,
            x=x,
            f=flows,
            z=z,
            w=w,
            candidates=candidates,
            dependencies=dependencies,
            resource_ids=resource_ids,
            edge_ids=edge_ids,
            incidence=commodity_incidences,
            commodity_columns=commodity_columns,
            commodity_incidences=commodity_incidences,
            commodity_outgoing=commodity_outgoing,
            commodity_incoming=commodity_incoming,
            endpoint_rhs=endpoint_rhs,
            assignment_constraints=assignment_constraints,
            compute_constraints=compute_constraints,
            routing_constraints=routing_constraints,
            actual_resource_usage=(),
            dependency_source_rows=tuple(
                source_index[_dependency_key(dependency)[0]]
                for dependency in dependencies
            ),
        ), problem_index)

    def _reachable_edge_columns(
        self,
        problem: NativeMorpherProblem,
        state: NativeMappingState,
        candidates: Sequence[Any],
        candidate_by_operation: Mapping[str, Sequence[int]],
        dependency: Mapping[str, Any],
        native_edges: Sequence[tuple[str, str]],
    ) -> np.ndarray:
        """Return every native edge on a legal temporal source/sink corridor.

        The traversal is a graph-theoretic reachability computation, not a
        shortest-path heuristic: all source candidates, all destination
        candidates, all directed native edges, and every latency state within
        the configured window are considered.  Removing an edge is therefore
        safe because no legal native temporal route can use it.
        """
        if not self.config.reachable_edge_pruning:
            return np.arange(len(native_edges), dtype=int)
        source_key, destination_key = _dependency_key(dependency)
        edge_type = str(dependency.get("edge_type", ""))
        distance = int(dependency.get("iteration_distance", 0) or 0)

        def options(node_key: str, source: bool) -> list[tuple[str, int]]:
            if node_key in state.placements:
                placement_values = [state.placements[node_key]]
            else:
                placement_values = [
                    candidates[index]
                    for index in candidate_by_operation[node_key]
                ]
            result = []
            for placement in placement_values:
                resource = (
                    problem.output_port(placement)
                    if source
                    else problem.operand_port(placement, edge_type)
                )
                latency = (
                    placement.output_latency
                    if source
                    else placement.latency + distance * problem.ii
                )
                result.append((resource, int(latency)))
            return result

        source_options = options(source_key, True)
        destination_options = options(destination_key, False)
        if not source_options or not destination_options:
            return np.zeros(0, dtype=int)
        corridor_key = (
            id(problem),
            source_key,
            destination_key,
            edge_type,
            distance,
            tuple(sorted(source_options)),
            tuple(sorted(destination_options)),
            len(native_edges),
        )
        cached_corridor = (
            self._corridor_cache.get(corridor_key)
            if self.config.structure_cache_enabled
            else None
        )
        if cached_corridor is not None:
            self._corridor_cache.move_to_end(corridor_key)
            self.structure_cache_hits += 1
            return cached_corridor
        min_time = min(value for _, value in source_options)
        max_time = max(value for _, value in destination_options)
        if min_time > max_time:
            return np.zeros(0, dtype=int)
        graph_key = (id(problem), tuple(native_edges))
        cached_graph = self._reachability_graph_cache.get(graph_key)
        if cached_graph is None:
            resource_phase = {
                resource_id: int(record.get("time_slot", -1))
                for resource_id, record in problem.resources.items()
            }
            adjacency_mut: dict[str, list[tuple[int, str, int]]] = {}
            reverse_mut: dict[str, list[tuple[int, str, int]]] = {}
            for global_column, (left, right) in enumerate(native_edges):
                left_phase = resource_phase[left]
                right_phase = resource_phase[right]
                if left_phase < 0 or right_phase < 0:
                    continue
                delta = (right_phase - left_phase) % problem.ii
                adjacency_mut.setdefault(left, []).append((global_column, right, delta))
                reverse_mut.setdefault(right, []).append((global_column, left, delta))
            cached_graph = (
                resource_phase,
                {key: tuple(value) for key, value in adjacency_mut.items()},
                {key: tuple(value) for key, value in reverse_mut.items()},
            )
            self._reachability_graph_cache[graph_key] = cached_graph
        resource_phase, adjacency, reverse = cached_graph
        forward: set[tuple[str, int]] = set(source_options)
        forward = {
            (resource, timestamp)
            for resource, timestamp in forward
            if min_time <= timestamp <= max_time
        }
        queue = list(forward)
        cursor = 0
        while cursor < len(queue):
            resource, timestamp = queue[cursor]
            cursor += 1
            for _, nxt, delta in adjacency.get(resource, ()):
                next_timestamp = timestamp + delta
                if next_timestamp > max_time:
                    continue
                key = (nxt, next_timestamp)
                if key not in forward:
                    forward.add(key)
                    queue.append(key)
        backward: set[tuple[str, int]] = set(destination_options)
        queue = list(backward)
        cursor = 0
        while cursor < len(queue):
            resource, timestamp = queue[cursor]
            cursor += 1
            for _, previous, delta in reverse.get(resource, ()):
                previous_timestamp = timestamp - delta
                if previous_timestamp < min_time:
                    continue
                key = (previous, previous_timestamp)
                if key not in backward:
                    backward.add(key)
                    queue.append(key)
        forward_by_resource: dict[str, set[int]] = {}
        for resource, timestamp in forward:
            forward_by_resource.setdefault(resource, set()).add(timestamp)
        relevant: set[int] = set()
        for global_column, (left, right) in enumerate(native_edges):
            left_phase = resource_phase[left]
            right_phase = resource_phase[right]
            if left_phase < 0 or right_phase < 0:
                continue
            delta = (right_phase - left_phase) % problem.ii
            for timestamp in forward_by_resource.get(left, ()):
                if (right, timestamp + delta) in backward:
                    relevant.add(global_column)
                    break
        result = np.asarray(sorted(relevant), dtype=int)
        result.setflags(write=False)
        if self.config.structure_cache_enabled:
            self._bounded_put(
                self._corridor_cache, corridor_key, result, 512
            )
        self.structure_cache_misses += 1
        return result

    @staticmethod
    def _incompatibility_cliques(
        problem: NativeMorpherProblem, source_keys: Sequence[str]
    ) -> tuple[tuple[str, ...], ...]:
        def source_mutex(left: str, right: str) -> bool:
            if left == right:
                return True
            left_bb = str(problem.nodes.get(left, {}).get("bb", ""))
            right_bb = str(problem.nodes.get(right, {}).get("bb", ""))
            return (
                bool(left_bb)
                and bool(right_bb)
                and frozenset((left_bb, right_bb)) in problem.mutex_pairs
            )

        graph = nx.Graph()
        graph.add_nodes_from(source_keys)
        for left_index, left in enumerate(source_keys):
            for right in source_keys[left_index + 1 :]:
                if not source_mutex(left, right):
                    graph.add_edge(left, right)
        cliques = [
            tuple(sorted(clique))
            for clique in nx.find_cliques(graph)
            if clique
        ]
        return tuple(sorted(set(cliques)))

    @staticmethod
    def _endpoint_distribution(
        problem: NativeMorpherProblem,
        state: NativeMappingState,
        candidates: Sequence[Any],
        candidate_by_operation: Mapping[str, Sequence[int]],
        x: cp.Variable,
        node_key: str,
        edge_type: str,
        *,
        source: bool,
        resource_index: Mapping[str, int],
    ) -> Any:
        """Return a sparse affine endpoint distribution.

        Native MRRGs contain thousands of port resources.  Building a Python
        list of scalar CVXPY expressions for every endpoint creates a very
        large expression DAG dominated by zeros.  A sparse candidate-to-port
        incidence matrix is mathematically identical and lets CVXPY retain a
        sparse coefficient matrix through canonicalization.
        """
        if node_key in state.placements:
            placement = state.placements[node_key]
            rid = (
                problem.output_port(placement)
                if source
                else problem.operand_port(placement, edge_type)
            )
            vector = np.zeros(len(resource_index), dtype=float)
            vector[resource_index[rid]] = 1.0
            return vector
        rows: list[int] = []
        columns: list[int] = []
        values: list[float] = []
        for candidate_index in candidate_by_operation[node_key]:
            placement = candidates[candidate_index]
            rid = (
                problem.output_port(placement)
                if source
                else problem.operand_port(placement, edge_type)
            )
            rows.append(resource_index[rid])
            columns.append(int(candidate_index))
            values.append(1.0)
        matrix = sp.csr_matrix(
            (values, (rows, columns)),
            shape=(len(resource_index), int(x.shape[0])),
        )
        return matrix @ x

    def _run_problem(
        self,
        problem: cp.Problem,
        *,
        start: float,
        canonical_seconds: float,
        cache_key: str,
        x: cp.Variable,
        f: Sequence[cp.Variable] | None,
        z: cp.Variable | None,
        w: cp.Variable | None,
        candidates: Sequence[Any],
        dependencies: Sequence[Mapping[str, Any]],
        resource_ids: Sequence[str],
        edge_ids: Sequence[str],
        incidence: Sequence[sp.csr_matrix] | None,
        commodity_columns: Sequence[np.ndarray] = (),
        commodity_incidences: Sequence[sp.csr_matrix] = (),
        commodity_outgoing: Sequence[sp.csr_matrix] = (),
        commodity_incoming: Sequence[sp.csr_matrix] = (),
        endpoint_rhs: Sequence[Any],
        assignment_constraints: Sequence[tuple[str, list[int], cp.Constraint]],
        compute_constraints: Sequence[tuple[str, list[int], cp.Constraint]],
        routing_constraints: Sequence[
            tuple[
                str,
                tuple[int, ...],
                np.ndarray,
                cp.Constraint,
                np.ndarray,
                np.ndarray,
            ]
        ],
        actual_resource_usage: Sequence[Any] = (),
        dependency_source_rows: Sequence[int] = (),
    ) -> NativeRelaxationResult:
        used = ""
        errors = []
        canonical_total = float(canonical_seconds)
        solve_seconds = 0.0
        for solver in (self.config.solver_primary, self.config.solver_fallback):
            try:
                used = solver
                # CVXPY caches canonical problem data on the Problem object.
                # Timing this call separates graph canonicalization from the
                # numerical solver; the subsequent solve reuses the cached
                # representation.  A template is not shared across different
                # residual operation/dependency sets because that would alter
                # matrix dimensions and is therefore not semantically valid.
                canonical_start = time.perf_counter()
                problem.get_problem_data(solver)
                canonical_total += time.perf_counter() - canonical_start
                kwargs: dict[str, Any] = {"verbose": False, "warm_start": True}
                if solver == "CLARABEL":
                    kwargs.update(
                        {
                            "time_limit": self.config.max_solve_seconds,
                            "max_threads": self.config.max_threads,
                            "tol_gap_abs": self.config.feasibility_tolerance,
                            "tol_feas": self.config.feasibility_tolerance,
                        }
                    )
                elif solver == "OSQP":
                    kwargs.update(
                        {
                            "time_limit": self.config.max_solve_seconds,
                            "eps_abs": self.config.feasibility_tolerance,
                            "eps_rel": self.config.feasibility_tolerance,
                            "max_iter": 200_000,
                            "polishing": True,
                        }
                    )
                solve_start = time.perf_counter()
                problem.solve(solver=solver, **kwargs)
                solve_seconds += time.perf_counter() - solve_start
                # A Clarabel ``user_limit`` is an operationally meaningful
                # terminal result.  Falling through to OSQP after a large
                # native conic problem hits its time limit rebuilds an even
                # larger QP and can spend another hour in setup without
                # improving the scientific result.  Preserve the solver
                # status and diagnostics instead of silently converting a
                # bounded solver limit into an unbounded fallback attempt.
                if problem.status in {
                    "optimal",
                    "optimal_inaccurate",
                    "infeasible",
                    "unbounded",
                    "user_limit",
                    "solver_error",
                }:
                    break
            except Exception as exc:  # preserve every solver failure
                errors.append(f"{solver}: {type(exc).__name__}: {exc}")
        if problem.status not in {"optimal", "optimal_inaccurate"}:
            return _native_failure(
                str(problem.status or "solver_failed"),
                used,
                start,
                cache_key,
                len(assignment_constraints),
                len(dependencies),
                "; ".join(errors),
                canonical_seconds=canonical_total,
                solve_seconds=solve_seconds,
                placement_variables=len(candidates),
                flow_variables=0 if f is None else int(sum(variable.shape[0] for variable in f)),
                full_flow_variables=0 if f is None else len(dependencies) * len(edge_ids),
            )

        xv = np.asarray(x.value, dtype=float).reshape(-1)
        fv = np.zeros((len(dependencies), len(edge_ids)), dtype=float)
        if f is not None:
            for index, (variable, columns) in enumerate(zip(f, commodity_columns)):
                fv[index, columns] = np.asarray(variable.value, dtype=float).reshape(-1)
        assignment_residual = max(
            (
                abs(float(xv[indices].sum()) - 1.0)
                for _, indices, _ in assignment_constraints
            ),
            default=0.0,
        )
        compute_violation = max(
            (
                max(0.0, float(xv[indices].sum()) - 1.0)
                for _, indices, _ in compute_constraints
            ),
            default=0.0,
        )
        flow_residual = 0.0
        if f is not None and incidence is not None:
            for index, rhs in enumerate(endpoint_rhs):
                rhs_value = np.asarray(
                    rhs.value if hasattr(rhs, "value") else rhs, dtype=float
                ).reshape(-1)
                flow_residual = max(
                    flow_residual,
                    float(
                        np.max(
                            np.abs(
                                incidence[index]
                                @ np.asarray(f[index].value, dtype=float).reshape(-1)
                                - rhs_value
                            ),
                            initial=0.0,
                        )
                    ),
                )
        routing_violation = max(
            (
                float(
                    np.max(
                        np.maximum(
                            np.asarray(constraint.violation(), dtype=float), 0.0
                        ),
                        initial=0.0,
                    )
                )
                for _, _, _, constraint, _, _ in routing_constraints
            ),
            default=0.0,
        )
        variables = [xv, fv.ravel()]
        if z is not None:
            variables.append(np.asarray(z.value, dtype=float).ravel())
        if w is not None:
            variables.append(np.asarray(w.value, dtype=float).ravel())
        all_values = np.concatenate(variables)

        resource_duals = np.zeros(len(resource_ids), dtype=float)
        edge_duals = np.zeros(len(edge_ids), dtype=float)
        resource_slacks = np.ones(len(resource_ids), dtype=float)
        if actual_resource_usage:
            actual_usage_values = np.vstack(
                [
                    np.asarray(value.value, dtype=float).reshape(-1)
                    for value in actual_resource_usage
                ]
            )
        elif f is not None and commodity_outgoing and commodity_incoming:
            # Compute the max(outgoing,incoming) usage numerically after the
            # solve.  This preserves capacity/slack diagnostics without
            # putting a CVXPY maximum in the canonicalized problem.
            actual_usage_values = np.vstack(
                [
                    np.maximum(
                        commodity_outgoing[index]
                        @ np.asarray(f[index].value, dtype=float).reshape(-1),
                        commodity_incoming[index]
                        @ np.asarray(f[index].value, dtype=float).reshape(-1),
                    )
                    for index in range(len(f))
                ]
            )
            # ``routing_constraints`` indexes rows by source key, so reduce
            # all commodities belonging to each source below before use.
            source_count = max(dependency_source_rows, default=-1) + 1
            if source_count != len(dependency_source_rows):
                reduced = np.zeros((source_count, len(resource_ids)))
                for index, row in enumerate(dependency_source_rows):
                    reduced[row] = np.maximum(
                        reduced[row], actual_usage_values[index]
                    )
                actual_usage_values = reduced
        else:
            actual_usage_values = np.zeros((0, len(resource_ids)), dtype=float)
        for kind, rows, columns, constraint, coefficients, fixed_count in routing_constraints:
            dual = np.maximum(
                np.asarray(constraint.dual_value, dtype=float).reshape(-1), 0.0
            )
            if kind == "resource":
                resource_duals[columns] += dual
                actual_lhs = np.sum(
                    coefficients
                    * actual_usage_values[np.ix_(rows, columns)],
                    axis=0,
                )
                slack = np.maximum(1.0 - fixed_count - actual_lhs, 0.0)
                resource_slacks[columns] = np.minimum(
                    resource_slacks[columns], slack
                )
            else:
                edge_duals[columns] += dual
        compute_duals = {
            dp_id: max(
                0.0, float(np.asarray(constraint.dual_value).reshape(()))
            )
            for dp_id, _, constraint in compute_constraints
        }
        normalized_resource = normalize_prices(resource_duals)
        normalized_edge = normalize_prices(edge_duals)
        positive_flow = np.maximum(fv, 0.0).ravel()
        flow_total = float(positive_flow.sum())
        flow_concentration = (
            float(np.square(positive_flow / flow_total).sum())
            if flow_total > 1e-12
            else 0.0
        )
        return NativeRelaxationResult(
            status=str(problem.status),
            solver=used,
            solve_seconds=solve_seconds,
            canonicalization_seconds=canonical_total,
            objective=float(problem.value),
            assignment_residual=float(assignment_residual),
            flow_residual=float(flow_residual),
            compute_capacity_violation=float(compute_violation),
            routing_capacity_violation=float(routing_violation),
            min_variable=float(all_values.min(initial=0.0)),
            max_variable=float(all_values.max(initial=0.0)),
            routing_resource_duals={
                rid: float(value)
                for rid, value in zip(resource_ids, resource_duals)
            },
            routing_edge_duals={
                rid: float(value) for rid, value in zip(edge_ids, edge_duals)
            },
            compute_duals=compute_duals,
            normalized_resource_duals={
                rid: float(value)
                for rid, value in zip(resource_ids, normalized_resource)
            },
            normalized_edge_duals={
                rid: float(value)
                for rid, value in zip(edge_ids, normalized_edge)
            },
            capacity_slacks={
                rid: float(value)
                for rid, value in zip(resource_ids, resource_slacks)
            },
            fractional_flow_concentration=flow_concentration,
            remaining_operations=len(assignment_constraints),
            remaining_dependencies=len(dependencies),
            placement_variables=len(candidates),
            flow_variables=0 if f is None else int(sum(variable.shape[0] for variable in f)),
            full_flow_variables=0 if f is None else len(dependencies) * len(edge_ids),
            flow_edge_reduction_fraction=(
                0.0
                if f is None or not dependencies
                else 1.0
                - float(sum(variable.shape[0] for variable in f))
                / float(len(dependencies) * len(edge_ids))
            ),
            cache_key=cache_key,
            error="; ".join(errors),
        )


def _native_failure(
    status: str,
    solver: str,
    start: float,
    cache_key: str,
    remaining_operations: int,
    remaining_dependencies: int,
    error: str,
    *,
    canonical_seconds: float = 0.0,
    solve_seconds: float | None = None,
    placement_variables: int = 0,
    flow_variables: int = 0,
    full_flow_variables: int = 0,
) -> NativeRelaxationResult:
    return NativeRelaxationResult(
        status=status,
        solver=solver,
        solve_seconds=(
            time.perf_counter() - start
            if solve_seconds is None
            else solve_seconds
        ),
        canonicalization_seconds=canonical_seconds,
        objective=float("nan"),
        assignment_residual=float("nan"),
        flow_residual=float("nan"),
        compute_capacity_violation=float("nan"),
        routing_capacity_violation=float("nan"),
        min_variable=float("nan"),
        max_variable=float("nan"),
        remaining_operations=remaining_operations,
        remaining_dependencies=remaining_dependencies,
        placement_variables=placement_variables,
        flow_variables=flow_variables,
        full_flow_variables=full_flow_variables,
        flow_edge_reduction_fraction=(
            0.0
            if full_flow_variables <= 0
            else 1.0 - float(flow_variables) / float(full_flow_variables)
        ),
        cache_key=cache_key,
        error=error,
    )


class NativeExactChildEvaluator:
    """Evaluate strict ``immediate_cost + exact child residual`` scores."""

    def __init__(
        self,
        solver: NativeRelaxationSolver,
        *,
        parallelism: int = 1,
    ) -> None:
        if parallelism <= 0:
            raise ValueError("child relaxation parallelism must be positive")
        self.solver = solver
        self.parallelism = int(parallelism)
        # Each lane owns its solver and all CVXPY/native-solver state.  Sharing
        # one solver between threads would race mutable warm starts, counters,
        # and in-memory caches.  The disk cache remains process-safe through
        # atomic rename and semantic content hashes.
        self._lane_solvers = [solver] + [
            NativeRelaxationSolver(solver.config, cache_dir=solver.cache_dir)
            for _ in range(self.parallelism - 1)
        ]
        self.last_evaluations: tuple[NativeChildEvaluation, ...] = ()
        self.total_evaluations = 0
        self.total_cache_hits = 0
        self.total_solve_seconds = 0.0
        self.total_request_wall_seconds = 0.0
        self.total_batch_wall_seconds = 0.0
        self.total_cache_read_seconds = 0.0

    @staticmethod
    def _evaluate_one(
        solver: NativeRelaxationSolver,
        problem: NativeMorpherProblem,
        state: NativeMappingState,
        action: NativeAction,
    ) -> tuple[NativeChildEvaluation, float]:
        request_start = time.perf_counter()
        child = state.apply_action(action)
        immediate = float(
            sum(max(0, len(route.resource_ids) - 1) for route in action.routes)
        )
        if child is None:
            record = NativeChildEvaluation(
                action_key=action.stable_key(),
                immediate_cost=immediate,
                residual_objective=float("nan"),
                q_rel=math.inf,
                residual_feasible=False,
                solve_status="illegal_child",
                solve_seconds=0.0,
                cache_hit=False,
                error="action could not be committed atomically",
            )
        else:
            result = solver.solve(problem, child)
            q_rel = (
                immediate + result.objective
                if result.feasible and math.isfinite(result.objective)
                else math.inf
            )
            record = NativeChildEvaluation(
                action_key=action.stable_key(),
                immediate_cost=immediate,
                residual_objective=result.objective,
                q_rel=q_rel,
                residual_feasible=result.feasible,
                solve_status=result.status,
                solve_seconds=result.solve_seconds,
                cache_hit=result.cache_hit,
                error=result.error,
            )
        return record, time.perf_counter() - request_start

    def evaluate_children(
        self,
        problem: NativeMorpherProblem,
        state: NativeMappingState,
        actions: Sequence[NativeAction],
    ) -> Sequence[float]:
        batch_start = time.perf_counter()
        indexed_records: list[tuple[int, NativeChildEvaluation, float]] = []
        if self.parallelism == 1 or len(actions) <= 1:
            for index, action in enumerate(actions):
                record, request_seconds = self._evaluate_one(
                    self.solver, problem, state, action
                )
                indexed_records.append((index, record, request_seconds))
        else:
            # Assign one sequential slice to each solver lane.  A lane solver
            # is therefore never invoked concurrently, while independent child
            # relaxations overlap in native solver code.
            lanes = min(self.parallelism, len(actions))
            slices = [
                list(range(lane, len(actions), lanes)) for lane in range(lanes)
            ]

            def run_lane(lane: int) -> list[tuple[int, NativeChildEvaluation, float]]:
                values = []
                lane_solver = self._lane_solvers[lane]
                for index in slices[lane]:
                    record, request_seconds = self._evaluate_one(
                        lane_solver, problem, state, actions[index]
                    )
                    values.append((index, record, request_seconds))
                return values

            with ThreadPoolExecutor(max_workers=lanes) as executor:
                futures = [executor.submit(run_lane, lane) for lane in range(lanes)]
                for future in futures:
                    indexed_records.extend(future.result())
        indexed_records.sort(key=lambda value: value[0])
        records = [value[1] for value in indexed_records]
        scores = [record.q_rel for record in records]
        for _, record, request_seconds in indexed_records:
            self.total_request_wall_seconds += request_seconds
            if record.cache_hit:
                self.total_cache_read_seconds += request_seconds
            else:
                self.total_solve_seconds += float(record.solve_seconds)
        self.total_batch_wall_seconds += time.perf_counter() - batch_start
        self.last_evaluations = tuple(records)
        self.total_evaluations += len(records)
        self.total_cache_hits += sum(record.cache_hit for record in records)
        return tuple(scores)
