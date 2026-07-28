import itertools

from quotientflow.arch import build_architecture
from quotientflow.dfg import generate_dfg
from quotientflow.mapper import enumerate_actions
from quotientflow.partial_state import PartialState, construct_partial_state
from quotientflow.symmetry import canonical_key, transform_state, verify_all_symmetries


def test_identity_and_verified_automorphisms():
    arch = build_architecture("mesh3")
    dfg = generate_dfg("diamond_chain", 3)
    state = construct_partial_state(dfg, arch, 0.25)
    identity_d = {v: v for v in dfg.graph}
    identity_a = {v: v for v in arch.graph}
    assert transform_state(state, arch, identity_d, identity_a).serialize() == state.serialize()
    assert verify_all_symmetries(dfg, arch)


def test_canonical_equivalent_and_noncollision():
    arch = build_architecture("mesh3")
    dfg = generate_dfg("diamond_chain", 4)
    state = construct_partial_state(dfg, arch, 0.25)
    identity_d = {v: v for v in dfg.graph}
    equivalent = transform_state(state, arch, identity_d, arch.symmetries[-1])
    assert canonical_key(state, dfg, arch)[0] == canonical_key(equivalent, dfg, arch)[0]
    other = state.copy()
    unused = next(n for n in arch.compute_slots if n not in other.occupied_compute)
    other.occupied_compute.add(unused)
    assert canonical_key(state, dfg, arch)[0] != canonical_key(other, dfg, arch)[0]


def test_compact_key_preserves_full_state_orbit_partition():
    arch = build_architecture("mesh3")
    dfg = generate_dfg("dot_product", 9)
    initial = construct_partial_state(dfg, arch, 0.10)
    next_op = next(op for op in dfg.order if op not in initial.placements)
    actions, _, _ = enumerate_actions(dfg, arch, initial, next_op)
    states = [initial, *(action.state for action in actions[:18])]

    reference = []
    for state in states:
        keys = []
        for dfg_perm, arch_perm in itertools.product(
            dfg.automorphisms, arch.symmetries
        ):
            keys.append(
                transform_state(state, arch, dfg_perm, arch_perm).serialize()
            )
        reference.append(min(keys))
    compact = [canonical_key(state, dfg, arch, joint=True)[0] for state in states]

    # The representation changes, but equality classes must be identical.
    for left, right in itertools.product(range(len(states)), repeat=2):
        assert (reference[left] == reference[right]) == (
            compact[left] == compact[right]
        )

    routed = initial
    for op in (op for op in dfg.order if op not in initial.placements):
        routed_actions, _, _ = enumerate_actions(dfg, arch, routed, op)
        assert routed_actions
        routed = routed_actions[0].state
        if routed.routed_edges:
            break
    assert routed.routed_edges
    same_serialized_state = routed.copy()
    dependency = next(iter(same_serialized_state.routed_edges))
    same_serialized_state.routed_edges[dependency] = tuple(
        reversed(same_serialized_state.routed_edges[dependency])
    )
    assert routed.serialize() == same_serialized_state.serialize()
    assert canonical_key(routed, dfg, arch, joint=True)[0] == canonical_key(
        same_serialized_state, dfg, arch, joint=True
    )[0]
