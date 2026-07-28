import numpy as np
import torch

from quotientflow.action_lookahead import (
    FEATURE_NAMES,
    action_features,
    check_partial_legality,
    stabilizer_action_orbits,
    stabilizer_target_orbits,
)
from quotientflow.action_model import ActionGNN, ActionMLP, combined_action_loss
from quotientflow.arch import build_architecture
from quotientflow.dfg import generate_dfg
from quotientflow.mapper import enumerate_actions, map_dfg
from quotientflow.partial_state import construct_partial_state
from quotientflow.relaxation import solve_relaxation


def fixture():
    arch = build_architecture("mesh3")
    dfg = generate_dfg("dot_product", 101)
    state = construct_partial_state(dfg, arch, 0.10)
    operation = next(op for op in dfg.order if op not in state.placements)
    actions, _, _ = enumerate_actions(dfg, arch, state, operation, 4)
    return dfg, arch, state, operation, actions


def test_committed_action_is_partial_legal_and_feature_complete():
    dfg, arch, state, operation, actions = fixture()
    action = actions[0]
    assert check_partial_legality(dfg, arch, action.state) == (True, "")
    features = action_features(
        dfg,
        arch,
        state,
        operation,
        action,
        np.zeros(len(arch.edge_list)),
    )
    assert list(features) == FEATURE_NAMES
    assert np.isfinite(list(features.values())).all()


def test_action_models_shape_and_combined_loss():
    dfg, arch, state, operation, actions = fixture()
    chosen = actions[:3]
    features = torch.zeros((len(chosen), len(FEATURE_NAMES)))
    mlp = ActionMLP()
    assert mlp(features).shape == (len(chosen),)
    gnn = ActionGNN()
    prediction = gnn(
        dfg,
        arch,
        state,
        operation,
        [action.target for action in chosen],
        [action.new_links for action in chosen],
        features,
    )
    assert prediction.shape == (len(chosen),)
    loss = combined_action_loss(prediction, torch.tensor([0.0, 1.0, 2.0]))
    assert torch.isfinite(loss)


def test_stabilizer_orbits_are_verified_and_mapping_exact():
    dfg, arch, state, operation, actions = fixture()
    targets, target_info = stabilizer_target_orbits(
        dfg, arch, state, operation
    )
    representatives, action_info = stabilizer_action_orbits(
        dfg, arch, state, operation, actions
    )
    assert 0 < len(targets) <= len(arch.compute_slots)
    assert target_info["stabilizer_size"] >= 1
    assert 0 < len(representatives) <= len(actions)
    assert action_info["stabilizer_size"] >= 1
    baseline = map_dfg(
        dfg, arch, state, beam_width=16, action_limit=9, timeout_seconds=20
    )
    orbit = map_dfg(
        dfg,
        arch,
        state,
        beam_width=16,
        action_limit=9,
        timeout_seconds=20,
        action_orbit_reduction=True,
    )
    assert not baseline.timeout and not orbit.timeout
    assert (baseline.success, baseline.legal, baseline.best_cost) == (
        orbit.success,
        orbit.legal,
        orbit.best_cost,
    )
