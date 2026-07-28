import networkx as nx

from quotientflow.arch import Architecture
from quotientflow.dfg import DFG
from quotientflow.partial_state import PartialState
from quotientflow.relaxation import solve_relaxation


def test_one_edge_flow_conservation_solves_exactly():
    graph = nx.DiGraph()
    a, b = (0, 0, 0), (1, 0, 1)
    graph.add_node(a, capacity=1.0, supports=("ADD", "MUL"))
    graph.add_node(b, capacity=1.0, supports=("ADD", "MUL"))
    graph.add_edge(a, b, kind="cardinal", capacity=1.0, base_cost=1.0, dx=1, dy=0, wrap=False, index=0)
    identity = {a: a, b: b}
    arch = Architecture("toy", 2, 1, 2, graph, [(a, b)], {(a, b): 0}, [identity])
    dag = nx.DiGraph()
    dag.add_node(0, op="ADD", level=0, slot=0)
    dag.add_node(1, op="MUL", level=1, slot=1)
    dag.add_edge(0, 1, kind="data")
    dfg = DFG("toy", 0, dag, [{0: 0, 1: 1}])
    state = PartialState({0: a}, {a}, set(), {})
    result = solve_relaxation(dfg, arch, state)
    assert result.solved
    assert result.assignment_residual < 1e-6
    assert result.flow_residual < 1e-6
    assert abs(result.objective - 1.001) < 1e-5
