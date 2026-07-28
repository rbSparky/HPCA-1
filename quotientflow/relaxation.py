"""Regularized fractional residual placement/routing optimization."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Dict, Optional, Tuple

import cvxpy as cp
import numpy as np
import scipy.sparse as sp

from .arch import Architecture
from .dfg import DFG
from .partial_state import PartialState


@dataclass
class RelaxationResult:
    status: str
    solver: str
    solve_seconds: float
    objective: float
    raw_prices: np.ndarray
    normalized_prices: np.ndarray
    assignment_residual: float
    flow_residual: float
    capacity_violation: float
    min_variable: float
    max_variable: float
    positive_dual_fraction: float
    error: str = ""
    compute_duals: Dict[tuple, float] = field(default_factory=dict)
    fractional_flow_concentration: float = 0.0
    capacity_slack_quantiles: np.ndarray = field(
        default_factory=lambda: np.zeros(5, dtype=float)
    )

    @property
    def solved(self) -> bool:
        return self.status in ("optimal", "optimal_inaccurate")

    def health_dict(self) -> dict:
        result = asdict(self)
        result.pop("raw_prices")
        result.pop("normalized_prices")
        result["solved"] = self.solved
        return result


def normalize_prices(raw: np.ndarray) -> np.ndarray:
    positive = raw[raw > 1e-10]
    scale = float(np.median(positive)) if positive.size else 1.0
    return np.log1p(np.maximum(raw, 0.0) / (scale + 1e-8))


def _incidence(arch: Architecture) -> sp.csr_matrix:
    nodes = sorted(arch.graph.nodes)
    node_idx = {v: i for i, v in enumerate(nodes)}
    rows, cols, vals = [], [], []
    for j, (u, v) in enumerate(arch.edge_list):
        rows.extend((node_idx[u], node_idx[v]))
        cols.extend((j, j))
        vals.extend((1.0, -1.0))
    return sp.csr_matrix((vals, (rows, cols)), shape=(len(nodes), len(arch.edge_list)))


def solve_relaxation(
    dfg: DFG,
    arch: Architecture,
    state: PartialState,
    tau: float = 0.001,
    solver_primary: str = "CLARABEL",
    solver_fallback: str = "OSQP",
    max_seconds: float = 8.0,
    knockout_edge: Optional[int] = None,
) -> RelaxationResult:
    start = time.perf_counter()
    nodes = sorted(arch.graph.nodes)
    node_idx = {v: i for i, v in enumerate(nodes)}
    rem = [v for v in dfg.order if v not in state.placements]
    deps = [e for e in sorted(dfg.graph.edges) if e not in state.routed_edges]
    if not rem and not deps:
        elapsed = time.perf_counter() - start
        zeros = np.zeros(len(arch.edge_list))
        return RelaxationResult(
            status="optimal",
            solver="closed_form_complete",
            solve_seconds=elapsed,
            objective=0.0,
            raw_prices=zeros.copy(),
            normalized_prices=zeros.copy(),
            assignment_residual=0.0,
            flow_residual=0.0,
            capacity_violation=0.0,
            min_variable=0.0,
            max_variable=0.0,
            positive_dual_fraction=0.0,
            compute_duals={},
            fractional_flow_concentration=0.0,
            capacity_slack_quantiles=np.ones(5, dtype=float),
        )
    pairs = [
        (v, r)
        for v in rem
        for r in nodes
        if r[2] == dfg.graph.nodes[v]["slot"] and r not in state.occupied_compute
    ]
    pair_idx = {pair: i for i, pair in enumerate(pairs)}
    if any(not any(v == p[0] for p in pairs) for v in rem):
        return _failure("infeasible", "none", start, len(arch.edge_list), "no compute slot")

    x = cp.Variable(len(pairs), name="x")
    f = cp.Variable((len(deps), len(arch.edge_list)), name="f")
    constraints = [x >= 0, x <= 1, f >= 0, f <= 1]
    assign_constraints = []
    for v in rem:
        indices = [i for i, pair in enumerate(pairs) if pair[0] == v]
        c = cp.sum(x[indices]) == 1
        constraints.append(c)
        assign_constraints.append((v, indices))
    compute_constraints = []
    for r in nodes:
        indices = [i for i, pair in enumerate(pairs) if pair[1] == r]
        if indices:
            c = cp.sum(x[indices]) <= 1.0
            constraints.append(c)
            compute_constraints.append((r, indices, c))

    def placement_expr(op: int):
        if op in state.placements:
            value = np.zeros(len(nodes))
            value[node_idx[state.placements[op]]] = 1.0
            return value
        return cp.hstack(
            [x[pair_idx[(op, r)]] if (op, r) in pair_idx else 0.0 for r in nodes]
        )

    incidence = _incidence(arch)
    flow_constraints = []
    for i, (u, v) in enumerate(deps):
        rhs = placement_expr(u) - placement_expr(v)
        c = incidence @ f[i, :] == rhs
        constraints.append(c)
        flow_constraints.append(c)

    capacity = np.ones(len(arch.edge_list))
    if state.occupied_links:
        capacity[list(state.occupied_links)] = 0.0
    if knockout_edge is not None:
        capacity[knockout_edge] = 0.0
    capacity_constraint = cp.sum(f, axis=0) <= capacity
    constraints.append(capacity_constraint)
    costs = np.array(
        [arch.graph.edges[e]["base_cost"] for e in arch.edge_list], dtype=float
    )
    objective = cp.Minimize(
        cp.sum(cp.multiply(f, costs[None, :]))
        + tau / 2.0 * (cp.sum_squares(x) + cp.sum_squares(f))
    )
    problem = cp.Problem(objective, constraints)
    used = ""
    error = ""
    for solver in (solver_primary, solver_fallback):
        try:
            used = solver
            kwargs = {"verbose": False}
            if solver == "CLARABEL":
                kwargs.update(
                    {
                        "time_limit": max_seconds,
                        "tol_gap_abs": 1e-8,
                        "tol_feas": 1e-8,
                    }
                )
            elif solver == "OSQP":
                kwargs.update(
                    {
                        "time_limit": max_seconds,
                        "eps_abs": 1e-7,
                        "eps_rel": 1e-7,
                        "max_iter": 100000,
                        "polishing": True,
                    }
                )
            problem.solve(solver=solver, **kwargs)
            if problem.status in ("optimal", "optimal_inaccurate", "infeasible"):
                break
        except Exception as exc:  # solver failures must be preserved in result
            error += f"{solver}: {type(exc).__name__}: {exc}; "
    elapsed = time.perf_counter() - start
    if problem.status not in ("optimal", "optimal_inaccurate"):
        return _failure(problem.status or "solver_failed", used, start, len(costs), error)
    xv = np.asarray(x.value).reshape(-1)
    fv = np.asarray(f.value)
    assignment = max(
        (abs(float(xv[idx].sum()) - 1.0) for _, idx in assign_constraints),
        default=0.0,
    )
    flow = 0.0
    for i, (u, v) in enumerate(deps):
        src = _placement_value(u, state, pairs, xv, nodes, node_idx)
        dst = _placement_value(v, state, pairs, xv, nodes, node_idx)
        flow = max(flow, float(np.max(np.abs(incidence @ fv[i] - src + dst))))
    cap_violation = float(np.maximum(fv.sum(axis=0) - capacity, 0.0).max(initial=0.0))
    all_values = np.concatenate((xv, fv.ravel()))
    dual = np.maximum(np.asarray(capacity_constraint.dual_value).reshape(-1), 0.0)
    compute_duals = {
        r: max(0.0, float(np.asarray(constraint.dual_value).reshape(())))
        for r, _, constraint in compute_constraints
    }
    positive_flow = np.maximum(fv, 0.0).ravel()
    flow_total = float(positive_flow.sum())
    flow_concentration = (
        float(np.square(positive_flow / flow_total).sum())
        if flow_total > 1e-12 else 0.0
    )
    capacity_slack = np.maximum(capacity - fv.sum(axis=0), 0.0)
    return RelaxationResult(
        status=str(problem.status),
        solver=used,
        solve_seconds=elapsed,
        objective=float(problem.value),
        raw_prices=dual,
        normalized_prices=normalize_prices(dual),
        assignment_residual=float(assignment),
        flow_residual=float(flow),
        capacity_violation=cap_violation,
        min_variable=float(all_values.min(initial=0.0)),
        max_variable=float(all_values.max(initial=0.0)),
        positive_dual_fraction=float(np.mean(dual > 1e-8)),
        error=error,
        compute_duals=compute_duals,
        fractional_flow_concentration=flow_concentration,
        capacity_slack_quantiles=np.quantile(
            capacity_slack, [0.0, 0.25, 0.5, 0.75, 1.0]
        ),
    )


def _placement_value(op, state, pairs, xv, nodes, node_idx):
    value = np.zeros(len(nodes))
    if op in state.placements:
        value[node_idx[state.placements[op]]] = 1.0
    else:
        for i, (v, r) in enumerate(pairs):
            if v == op:
                value[node_idx[r]] = xv[i]
    return value


def _failure(status, solver, start, edge_count, error):
    return RelaxationResult(
        status=str(status),
        solver=solver,
        solve_seconds=time.perf_counter() - start,
        objective=float("nan"),
        raw_prices=np.zeros(edge_count),
        normalized_prices=np.zeros(edge_count),
        assignment_residual=float("nan"),
        flow_residual=float("nan"),
        capacity_violation=float("nan"),
        min_variable=float("nan"),
        max_variable=float("nan"),
        positive_dual_fraction=0.0,
        error=error,
        compute_duals={},
        fractional_flow_concentration=0.0,
        capacity_slack_quantiles=np.full(5, np.nan),
    )
