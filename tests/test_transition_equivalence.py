from scripts.run_symmetry_transition_equivalence import micro_architecture, micro_dfg

from quotientflow.transition_equivalence import (
    admitted_joint_transforms,
    preserves_total_operation_order,
    validate_transition_equivalence,
    verify_semantic_arch_permutation,
)


def test_tied_priority_dfg_swap_is_rejected_by_total_order():
    dfg = micro_dfg("tied_priority")
    assert len(dfg.automorphisms) == 2
    assert preserves_total_operation_order(dfg, dfg.automorphisms[0])
    assert not preserves_total_operation_order(dfg, dfg.automorphisms[1])
    admitted, rejected = admitted_joint_transforms(
        dfg, micro_architecture("tied_priority")
    )
    assert admitted
    assert any(item.reason == "total_operation_order_not_preserved" for item in rejected)


def test_memory_column_transform_is_rejected_semantically():
    arch = micro_architecture("memory")
    assert verify_semantic_arch_permutation(arch, arch.symmetries[0])
    assert verify_semantic_arch_permutation(arch, arch.symmetries[1])
    assert not verify_semantic_arch_permutation(arch, arch.symmetries[2])


def test_exhaustive_transition_equivalence_review_cases():
    for case in ("tied_priority", "recurrence", "memory"):
        result, rejected = validate_transition_equivalence(
            case, micro_dfg(case), micro_architecture(case)
        )
        assert result.passed, (case, result.error)
        assert result.reachable_states > 1
        assert result.checked_state_transforms > 0
        assert result.checked_actions > 0
        assert rejected
