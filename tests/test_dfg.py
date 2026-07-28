import networkx as nx
import pytest

from quotientflow.dfg import (
    FAMILIES,
    generate_dfg,
    generate_stress_dfg,
    verify_dfg_permutation,
)


@pytest.mark.parametrize("family", FAMILIES)
def test_dfg_family(family):
    dfg = generate_dfg(family, 1234)
    assert nx.is_directed_acyclic_graph(dfg.graph)
    assert 8 <= len(dfg.graph) <= 16
    assert set(nx.get_node_attributes(dfg.graph, "op").values()) <= {"ADD", "MUL"}
    assert all(verify_dfg_permutation(dfg, p) for p in dfg.automorphisms)


@pytest.mark.parametrize("family", FAMILIES)
def test_stress_dfg_family(family):
    dfg = generate_stress_dfg(family, 24072026)
    assert nx.is_directed_acyclic_graph(dfg.graph)
    assert 13 <= len(dfg.graph) <= 20
    assert all(verify_dfg_permutation(dfg, p) for p in dfg.automorphisms)
