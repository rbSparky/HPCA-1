from quotientflow.arch import build_architecture
from quotientflow.dfg import generate_dfg
from quotientflow.mapper import check_legality, map_dfg
from quotientflow.partial_state import construct_partial_state


def test_mapper_success_is_independently_legal():
    arch = build_architecture("mesh3")
    dfg = generate_dfg("dot_product", 5)
    state = construct_partial_state(dfg, arch, 0.25)
    result = map_dfg(dfg, arch, state, beam_width=64, action_limit=9, timeout_seconds=10)
    assert result.success
    assert result.legal
    assert check_legality(dfg, arch, result.final_state) == (True, "")
