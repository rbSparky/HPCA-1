import pytest

from quotientflow.arch import EXPECTED_COUNTS, build_architecture, verify_arch_permutation


@pytest.mark.parametrize("name", EXPECTED_COUNTS)
def test_architecture_counts_and_time(name):
    arch = build_architecture(name)
    assert (len(arch.graph), arch.graph.number_of_edges()) == EXPECTED_COUNTS[name]
    assert arch.symmetries
    assert all(verify_arch_permutation(arch, p) for p in arch.symmetries)
    for u, v in arch.graph.edges:
        assert v[2] == (u[2] + 1) % arch.ii


def test_exact_expected_architecture_groups():
    assert len(build_architecture("mesh3").symmetries) == 8
    assert len(build_architecture("torus3").symmetries) == 72
    assert len(build_architecture("cut4").symmetries) == 4
