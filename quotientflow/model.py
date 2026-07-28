"""Small pure-PyTorch two-tower directed message-passing network."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .arch import Architecture
from .dfg import DFG
from .partial_state import PartialState


class DirectedLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.self_linear = nn.Linear(input_dim, output_dim)
        self.in_linear = nn.Linear(input_dim, output_dim, bias=False)
        self.out_linear = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, h: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        n = h.shape[0]
        src, dst = edges
        incoming = torch.zeros((n, h.shape[1]), device=h.device, dtype=h.dtype)
        outgoing = torch.zeros_like(incoming)
        in_count = torch.zeros((n, 1), device=h.device, dtype=h.dtype)
        out_count = torch.zeros_like(in_count)
        incoming.index_add_(0, dst, h[src])
        outgoing.index_add_(0, src, h[dst])
        ones = torch.ones((src.numel(), 1), device=h.device, dtype=h.dtype)
        in_count.index_add_(0, dst, ones)
        out_count.index_add_(0, src, ones)
        incoming = incoming / in_count.clamp_min(1.0)
        outgoing = outgoing / out_count.clamp_min(1.0)
        return F.relu(
            self.self_linear(h)
            + self.in_linear(incoming)
            + self.out_linear(outgoing)
        )


class Tower(nn.Module):
    def __init__(self, input_dim: int, hidden: int, layers: int):
        super().__init__()
        dims = [input_dim] + [hidden] * layers
        self.layers = nn.ModuleList(
            DirectedLayer(dims[i], dims[i + 1]) for i in range(layers)
        )

    def forward(self, x, edges):
        for layer in self.layers:
            x = layer(x, edges)
        return x


class QuotientFlowGNN(nn.Module):
    def __init__(self, hidden_dim=64, dfg_layers=3, arch_layers=3, head_layers=2):
        super().__init__()
        self.dfg_tower = Tower(11, hidden_dim, dfg_layers)
        self.arch_tower = Tower(12, hidden_dim, arch_layers)
        head = []
        input_dim = hidden_dim * 3 + 9
        for i in range(head_layers - 1):
            head.extend((nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim), nn.ReLU()))
        head.append(nn.Linear(hidden_dim if head_layers > 1 else input_dim, 2))
        self.head = nn.Sequential(*head)

    def forward(self, dfg: DFG, arch: Architecture, state: PartialState):
        device = next(self.parameters()).device
        df, de = dfg_tensors(dfg, arch, state, device)
        af, ae, edge_src, edge_dst, ef = arch_tensors(arch, state, device)
        dh = self.dfg_tower(df, de)
        ah = self.arch_tower(af, ae)
        pooled = dh.mean(dim=0, keepdim=True).expand(len(arch.edge_list), -1)
        combined = torch.cat((ah[edge_src], ah[edge_dst], pooled, ef), dim=1)
        output = self.head(combined)
        return F.softplus(output[:, 0]), output[:, 1]


def dfg_tensors(dfg: DFG, arch: Architecture, state: PartialState, device):
    nodes = sorted(dfg.graph)
    idx = {v: i for i, v in enumerate(nodes)}
    max_level = max(dfg.graph.nodes[v]["level"] for v in nodes) or 1
    features = []
    for v in nodes:
        data = dfg.graph.nodes[v]
        anchored = v in state.placements
        node = state.placements.get(v, (0, 0, 0))
        features.append(
            [
                float(data["op"] == "ADD"),
                float(data["op"] == "MUL"),
                dfg.graph.in_degree(v) / max(1, len(nodes) - 1),
                dfg.graph.out_degree(v) / max(1, len(nodes) - 1),
                data["level"] / max_level,
                *[float(data["slot"] == t) for t in range(3)],
                float(anchored),
                node[0] / max(1, arch.width - 1) if anchored else 0.0,
                node[1] / max(1, arch.height - 1) if anchored else 0.0,
            ]
        )
    edges = [(idx[u], idx[v]) for u, v in dfg.graph.edges]
    edge_tensor = torch.tensor(edges, dtype=torch.long, device=device).T
    return torch.tensor(features, dtype=torch.float32, device=device), edge_tensor


def arch_tensors(arch: Architecture, state: PartialState, device):
    nodes = sorted(arch.graph)
    idx = {v: i for i, v in enumerate(nodes)}
    features = []
    for x, y, t in nodes:
        features.append(
            [
                x / max(1, arch.width - 1),
                y / max(1, arch.height - 1),
                *[float(t == k) for k in range(3)],
                arch.graph.in_degree((x, y, t)) / max(1, len(nodes) - 1),
                arch.graph.out_degree((x, y, t)) / max(1, len(nodes) - 1),
                float((x, y, t) in state.occupied_compute),
                float(y == arch.height - 1),
                float(y == 0),
                float(x == arch.width - 1),
                float(x == 0),
            ]
        )
    graph_edges = [(idx[u], idx[v]) for u, v in arch.edge_list]
    edge_src = torch.tensor([a for a, _ in graph_edges], dtype=torch.long, device=device)
    edge_dst = torch.tensor([b for _, b in graph_edges], dtype=torch.long, device=device)
    edge_features = []
    kinds = ("wait", "cardinal", "diagonal", "wrap")
    for edge_id, edge in enumerate(arch.edge_list):
        data = arch.graph.edges[edge]
        edge_features.append(
            [
                *[float(data["kind"] == kind) for kind in kinds],
                float(edge_id in state.occupied_links),
                float(edge_id not in state.occupied_links),
                float(data["dx"]),
                float(data["dy"]),
                float(data["wrap"]),
            ]
        )
    return (
        torch.tensor(features, dtype=torch.float32, device=device),
        torch.tensor(graph_edges, dtype=torch.long, device=device).T,
        edge_src,
        edge_dst,
        torch.tensor(edge_features, dtype=torch.float32, device=device),
    )


@torch.no_grad()
def predict_numpy(model, dfg, arch, state) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    price, logit = model(dfg, arch, state)
    return price.detach().cpu().numpy(), logit.detach().cpu().numpy()
