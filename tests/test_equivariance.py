import numpy as np
import pytest
import torch

from quotientflow.arch import build_architecture
from quotientflow.dfg import generate_dfg
from quotientflow.model import QuotientFlowGNN, predict_numpy
from quotientflow.partial_state import construct_partial_state


def test_gnn_output_shape():
    arch = build_architecture("mesh3")
    dfg = generate_dfg("fir", 2)
    state = construct_partial_state(dfg, arch, 0.25)
    model = QuotientFlowGNN()
    price, logits = model(dfg, arch, state)
    assert price.shape == logits.shape == (len(arch.edge_list),)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cpu_cuda_forward_agree():
    torch.manual_seed(0)
    arch = build_architecture("mesh3")
    dfg = generate_dfg("fir", 2)
    state = construct_partial_state(dfg, arch, 0.25)
    model = QuotientFlowGNN().eval()
    cpu, _ = predict_numpy(model, dfg, arch, state)
    model.cuda()
    cuda, _ = predict_numpy(model, dfg, arch, state)
    np.testing.assert_allclose(cpu, cuda, rtol=2e-4, atol=2e-5)
