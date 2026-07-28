import networkx as nx

from quotientflow.arch import build_architecture
from quotientflow.dfg import DFG, generate_dfg
from quotientflow.mapper import map_dfg
from quotientflow.partial_state import PartialState, construct_partial_state, state_capacity_valid
from quotientflow.relaxation import solve_relaxation
from quotientflow.symmetry import transform_state


def test_partial_capacity_and_relaxation_health():
    arch = build_architecture("mesh3")
    dfg = generate_dfg("diamond_chain", 7)
    state = construct_partial_state(dfg, arch, 0.25)
    assert state_capacity_valid(state)
    result = solve_relaxation(dfg, arch, state)
    assert result.solved
    assert result.assignment_residual <= 1e-5
    assert result.flow_residual <= 1e-5
    assert result.capacity_violation <= 1e-5


def test_known_compute_capacity_infeasible():
    arch = build_architecture("mesh3")
    graph = nx.DiGraph()
    for v in range(10):
        graph.add_node(v, op="ADD", level=0, slot=0)
    graph.add_edges_from((v, v + 1, {"kind": "data"}) for v in range(9))
    dfg = DFG("overfull", 0, graph, [{v: v for v in graph}])
    result = solve_relaxation(dfg, arch, PartialState())
    assert result.status == "infeasible"


def test_transformed_relaxation_objective_invariant():
    arch = build_architecture("mesh3")
    dfg = generate_dfg("diamond_chain", 8)
    state = construct_partial_state(dfg, arch, 0.25)
    identity = {v: v for v in dfg.graph}
    transformed = transform_state(state, arch, identity, arch.symmetries[-1])
    first = solve_relaxation(dfg, arch, state)
    second = solve_relaxation(dfg, arch, transformed)
    assert first.solved and second.solved
    assert abs(first.objective - second.objective) <= 1e-5


def test_complete_state_has_zero_residual_objective():
    arch = build_architecture("mesh3")
    dfg = generate_dfg("dot_product", 17)
    state = construct_partial_state(dfg, arch, 0.10)
    mapped = map_dfg(
        dfg, arch, state, beam_width=64, action_limit=9, timeout_seconds=20
    )
    assert mapped.success
    result = solve_relaxation(dfg, arch, mapped.final_state)
    assert result.solved
    assert result.objective == 0.0
    assert result.solver == "closed_form_complete"
