import numpy as np

from quotientflow.arch import build_architecture
from quotientflow.dfg import generate_dfg
from quotientflow.flow_advantage import (
    FLOW_FEATURE_NAMES,
    action_feature_rows,
    corrected_targets,
    dual_baseline,
    select_training_actions,
)
from quotientflow.mapper import enumerate_actions
from quotientflow.partial_state import PartialState
from quotientflow.relaxation import solve_relaxation


def small_problem():
    dfg = generate_dfg("diamond_chain", seed=3)
    arch = build_architecture("mesh3")
    state = PartialState()
    operation = dfg.order[0]
    actions, _, _ = enumerate_actions(dfg, arch, state, operation, 4)
    parent = solve_relaxation(dfg, arch, state)
    return dfg, arch, state, operation, actions, parent


def test_relaxation_exposes_compute_and_primal_diagnostics():
    *_, parent = small_problem()
    assert parent.solved
    assert parent.compute_duals
    assert parent.capacity_slack_quantiles.shape == (5,)
    assert 0.0 <= parent.fractional_flow_concentration <= 1.0


def test_flow_features_are_complete_and_selector_is_deterministic():
    dfg, arch, state, operation, actions, parent = small_problem()
    rows = action_feature_rows(
        dfg, arch, state, operation, actions, parent
    )
    assert all(set(row) == set(FLOW_FEATURE_NAMES) for row in rows)
    first = select_training_actions(actions, rows, 12)
    second = select_training_actions(actions, rows, 12)
    assert first == second
    assert len(first) <= 12
    assert all(np.isfinite(list(rows[i].values())).all() for i in first)


def test_corrected_target_uses_raw_stable_scale():
    immediate = np.array([1.0, 1.0, 2.0])
    child = np.array([4.0, 4.00001, 4.0])
    baseline = np.array([4.5, 4.5, 5.0])
    target = corrected_targets(immediate, child, 3.0, baseline, 0.0, 1.0)
    assert target["raw_regret"][1] < 1e-4
    assert target["epsilon_optimal"][1]
    assert target["state_scale"] >= 0.1
    assert np.isclose(target["soft_target"].sum(), 1.0)


def test_dual_baseline_has_compute_and_routing_terms():
    dfg, arch, state, operation, actions, parent = small_problem()
    value, routing, compute = dual_baseline(actions[0], parent)
    assert np.isclose(value, actions[0].base_cost + routing + compute)
    assert routing >= 0.0
    assert compute >= 0.0
