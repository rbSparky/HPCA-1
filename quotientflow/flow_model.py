"""Pure-PyTorch FlowAdvantage residual models and listwise training."""

from __future__ import annotations

import copy
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .flow_advantage import FLOW_FEATURE_NAMES, NO_DUAL_FEATURE_NAMES
from .model import dfg_tensors


@dataclass
class StandardScaler:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, groups: Sequence[dict], feature_names: Sequence[str]):
        values = np.concatenate([
            group["frame"][list(feature_names)].to_numpy(np.float32)
            for group in groups
        ])
        mean, scale = values.mean(0), values.std(0)
        scale[scale < 1e-6] = 1.0
        return cls(mean.astype(np.float32), scale.astype(np.float32))

    def transform(self, values):
        return ((np.asarray(values, np.float32) - self.mean) / self.scale).astype(
            np.float32
        )


class ResidualNodeLayer(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.self_linear = nn.Linear(hidden, hidden)
        self.in_linear = nn.Linear(hidden, hidden, bias=False)
        self.out_linear = nn.Linear(hidden, hidden, bias=False)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, h, edges):
        src, dst = edges
        incoming = torch.zeros_like(h)
        outgoing = torch.zeros_like(h)
        incoming.index_add_(0, dst, h[src])
        outgoing.index_add_(0, src, h[dst])
        update = F.silu(
            self.self_linear(h) + self.in_linear(incoming)
            + self.out_linear(outgoing)
        )
        return self.norm(h + update)


class ResidualEdgeLayer(nn.Module):
    def __init__(self, hidden, edge_dim):
        super().__init__()
        self.in_message = nn.Linear(hidden + edge_dim, hidden)
        self.out_message = nn.Linear(hidden + edge_dim, hidden)
        self.self_linear = nn.Linear(hidden, hidden)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, h, edges, edge_features):
        src, dst = edges
        incoming = torch.zeros_like(h)
        outgoing = torch.zeros_like(h)
        incoming.index_add_(
            0, dst, self.in_message(torch.cat((h[src], edge_features), 1))
        )
        outgoing.index_add_(
            0, src, self.out_message(torch.cat((h[dst], edge_features), 1))
        )
        update = F.silu(self.self_linear(h) + incoming + outgoing)
        return self.norm(h + update)


def architecture_tensors(arch, state, parent, device, include_duals=True):
    nodes = sorted(arch.graph)
    index = {node: i for i, node in enumerate(nodes)}
    node_features = []
    compute_duals = getattr(parent, "compute_duals", {})
    positive_compute = np.asarray(list(compute_duals.values()), dtype=float)
    compute_scale = max(float(np.median(positive_compute[positive_compute > 1e-10]))
                        if np.any(positive_compute > 1e-10) else 1.0, 1e-8)
    for x, y, t in nodes:
        node_features.append([
            x / max(1, arch.width - 1),
            y / max(1, arch.height - 1),
            t / max(1, arch.ii - 1),
            arch.graph.in_degree((x, y, t)) / max(1, len(nodes) - 1),
            arch.graph.out_degree((x, y, t)) / max(1, len(nodes) - 1),
            float((x, y, t) in state.occupied_compute),
            float(x in (0, arch.width - 1)),
            float(y in (0, arch.height - 1)),
            (
                math.log1p(max(0.0, compute_duals.get((x, y, t), 0.0))
                           / compute_scale)
                if include_duals else 0.0
            ),
        ])
    graph_edges = [(index[u], index[v]) for u, v in arch.edge_list]
    raw = np.asarray(parent.raw_prices)
    positive = raw[raw > 1e-10]
    dual_scale = max(float(np.median(positive)) if len(positive) else 1.0, 1e-8)
    edge_features = []
    kinds = ("wait", "cardinal", "diagonal", "wrap")
    for edge_id, edge in enumerate(arch.edge_list):
        data = arch.graph.edges[edge]
        edge_features.append([
            *[float(data["kind"] == kind) for kind in kinds],
            float(edge_id in state.occupied_links),
            float(edge_id not in state.occupied_links),
            float(data["dx"]),
            float(data["dy"]),
            (
                math.log1p(max(0.0, raw[edge_id]) / dual_scale)
                if include_duals else 0.0
            ),
        ])
    edges = torch.tensor(graph_edges, dtype=torch.long, device=device).T
    return (
        torch.tensor(node_features, dtype=torch.float32, device=device),
        edges,
        torch.tensor(edge_features, dtype=torch.float32, device=device),
        {node: i for i, node in enumerate(nodes)},
    )


class ResidualMLP(nn.Module):
    def __init__(self, input_dim=len(FLOW_FEATURE_NAMES)):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128), nn.SiLU(),
            nn.Linear(128, 128), nn.SiLU(),
            nn.Linear(128, 64), nn.SiLU(),
            nn.Linear(64, 1),
        )

    def forward(self, features):
        return self.network(features).squeeze(-1)


class ResidualGNN(nn.Module):
    def __init__(self, feature_dim=len(FLOW_FEATURE_NAMES), hidden=64,
                 include_duals=True):
        super().__init__()
        self.include_duals = include_duals
        self.dfg_input = nn.Linear(11, hidden)
        self.arch_input = nn.Linear(9, hidden)
        self.dfg_layer = ResidualNodeLayer(hidden)
        self.arch_layer = ResidualEdgeLayer(hidden, 9)
        self.steps = 5
        self.head = nn.Sequential(
            nn.Linear(hidden * 8 + feature_dim, 128), nn.SiLU(),
            nn.Linear(128, 64), nn.SiLU(),
            nn.Linear(64, 1),
        )

    def forward(self, dfg, arch, state, parent, operation, targets, new_links,
                features):
        device = next(self.parameters()).device
        dfg_x, dfg_edges = dfg_tensors(dfg, arch, state, device)
        arch_x, arch_edges, arch_edge_features, arch_index = architecture_tensors(
            arch, state, parent, device, self.include_duals
        )
        dh = self.dfg_input(dfg_x)
        ah = self.arch_input(arch_x)
        for _ in range(self.steps):
            dh = self.dfg_layer(dh, dfg_edges)
            ah = self.arch_layer(ah, arch_edges, arch_edge_features)
        dfg_nodes = sorted(dfg.graph)
        dfg_index = {node: i for i, node in enumerate(dfg_nodes)}
        count = len(targets)
        operation_h = dh[dfg_index[operation]].unsqueeze(0).expand(count, -1)
        target_h = ah[torch.tensor(
            [arch_index[target] for target in targets],
            dtype=torch.long, device=device,
        )]
        dfg_mean = dh.mean(0, keepdim=True).expand(count, -1)
        dfg_max = dh.max(0).values.unsqueeze(0).expand(count, -1)
        arch_mean = ah.mean(0, keepdim=True).expand(count, -1)
        arch_max = ah.max(0).values.unsqueeze(0).expand(count, -1)
        src, dst = arch_edges
        edge_h = (ah[src] + ah[dst]) / 2.0
        routed_mean, routed_max = [], []
        for links in new_links:
            if links:
                selected = edge_h[torch.tensor(links, dtype=torch.long, device=device)]
                routed_mean.append(selected.mean(0))
                routed_max.append(selected.max(0).values)
            else:
                routed_mean.append(torch.zeros_like(ah[0]))
                routed_max.append(torch.zeros_like(ah[0]))
        combined = torch.cat((
            operation_h, target_h, dfg_mean, dfg_max, arch_mean, arch_max,
            torch.stack(routed_mean), torch.stack(routed_max), features,
        ), 1)
        return self.head(combined).squeeze(-1)


def flow_scores(model, model_type, group, sample, scaler, feature_names):
    device = next(model.parameters()).device
    values = group["frame"][list(feature_names)].to_numpy(np.float32)
    features = torch.tensor(
        scaler.transform(values), dtype=torch.float32, device=device
    )
    if model_type == "residual_mlp":
        return model(features)
    return model(
        sample["dfg"], sample["arch"], sample["state"], group["parent"],
        group["operation"], group["targets"], group["new_links"], features,
    )


def flow_loss(prediction, predicted_delta, residual_target, raw_regret,
              soft_target, delta_range):
    model_log = F.log_softmax(-predicted_delta, dim=0)
    listwise = -(soft_target * model_log).sum()
    differences = raw_regret[:, None] - raw_regret[None, :]
    threshold = max(1e-4, .01 * max(float(delta_range), 1.0))
    upper = torch.triu(torch.ones_like(differences, dtype=torch.bool), diagonal=1)
    valid = upper & (differences.abs() >= threshold)
    if valid.any():
        i, j = valid.nonzero(as_tuple=True)
        target = torch.sign(raw_regret[j] - raw_regret[i])
        relative = (raw_regret[i] - raw_regret[j]).abs() / max(
            float(delta_range), 1.0
        )
        weights = torch.clamp(relative + .5, max=5.0)
        pairwise = (
            weights
            * F.softplus(-target * (predicted_delta[j] - predicted_delta[i]))
        ).sum() / weights.sum()
    else:
        pairwise = prediction.sum() * 0.0
    regression = F.smooth_l1_loss(prediction, residual_target)
    return listwise + .5 * pairwise + .25 * regression


def action_metrics(predicted_delta, delta, epsilon):
    predicted_order = np.argsort(predicted_delta, kind="stable")
    true_order = np.argsort(delta, kind="stable")
    selected = predicted_order[0]
    regret = float(delta[selected] - delta[true_order[0]])
    delta_range = float(np.ptp(delta))
    pairs = [
        (i, j) for i in range(len(delta)) for j in range(i + 1, len(delta))
        if abs(delta[i] - delta[j]) >= max(1e-4, .01 * max(delta_range, 1.0))
    ]
    pair_accuracy = np.mean([
        np.sign(delta[j] - delta[i])
        == np.sign(predicted_delta[j] - predicted_delta[i])
        for i, j in pairs
    ]) if pairs else 1.0
    return {
        "exact_top1": float(selected == true_order[0]),
        "epsilon_top1": float(regret <= epsilon + 1e-12),
        "top2_recall": float(true_order[0] in predicted_order[:2]),
        "top4_recall": float(true_order[0] in predicted_order[:4]),
        "raw_regret": regret,
        "relative_regret": regret / max(delta_range, 1.0),
        "pairwise_accuracy": float(pair_accuracy),
    }


@torch.no_grad()
def evaluate(model, model_type, groups, samples, scaler, feature_names,
             residual_center, residual_scale):
    model.eval()
    metrics = []
    for group in groups:
        residual_scaled = flow_scores(
            model, model_type, group, samples[group["state_id"]],
            scaler, feature_names,
        ).cpu().numpy()
        residual = residual_scaled * residual_scale + residual_center
        predicted_delta = group["baseline"] + residual
        metrics.append(action_metrics(
            predicted_delta, group["delta"], group["epsilon"]
        ))
    return {
        key: float(np.mean([metric[key] for metric in metrics]))
        for key in metrics[0]
    }


def train_flow_model(model_type, train_groups, validation_groups, samples,
                     seed, checkpoint: Path, device, residual_center,
                     residual_scale, epochs=120, patience=18):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    include_duals = model_type != "residual_gnn_no_dual"
    feature_names = (
        FLOW_FEATURE_NAMES if include_duals else NO_DUAL_FEATURE_NAMES
    )
    scaler = StandardScaler.fit(train_groups, feature_names)
    if model_type == "residual_mlp":
        model = ResidualMLP(len(feature_names))
    else:
        model = ResidualGNN(
            len(feature_names), include_duals=include_duals
        )
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    rng = random.Random(seed)
    best_key = (-math.inf, -math.inf, -math.inf, -math.inf)
    best_state, best_epoch, stale = copy.deepcopy(model.state_dict()), 0, 0
    started = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        order = list(range(len(train_groups)))
        rng.shuffle(order)
        optimizer.zero_grad(set_to_none=True)
        for position, index in enumerate(order):
            group = train_groups[index]
            prediction = flow_scores(
                model, model_type, group, samples[group["state_id"]],
                scaler, feature_names,
            )
            device = prediction.device
            baseline = torch.tensor(
                group["baseline"], dtype=torch.float32, device=device
            )
            predicted_delta = (
                baseline + prediction * residual_scale + residual_center
            )
            loss = flow_loss(
                prediction,
                predicted_delta,
                torch.tensor(group["residual_target"], dtype=torch.float32, device=device),
                torch.tensor(group["raw_regret"], dtype=torch.float32, device=device),
                torch.tensor(group["soft_target"], dtype=torch.float32, device=device),
                group["delta_range"],
            ) / 8.0
            loss.backward()
            if (position + 1) % 8 == 0 or position + 1 == len(order):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        validation = evaluate(
            model, model_type, validation_groups, samples, scaler, feature_names,
            residual_center, residual_scale,
        )
        key = (
            validation["top4_recall"],
            validation["epsilon_top1"],
            -validation["relative_regret"],
            validation["exact_top1"],
        )
        if key > best_key:
            best_key = key
            best_state = copy.deepcopy(model.state_dict())
            best_epoch, stale = epoch, 0
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    metadata = {
        "model_type": model_type,
        "seed": seed,
        "best_epoch": best_epoch,
        "validation_top4_recall": best_key[0],
        "validation_epsilon_top1": best_key[1],
        "validation_relative_regret": -best_key[2],
        "validation_exact_top1": best_key[3],
        "train_seconds": time.perf_counter() - started,
        "feature_names": list(feature_names),
        "residual_center": residual_center,
        "residual_scale": residual_scale,
    }
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_type": model_type,
        "seed": seed,
        "state_dict": best_state,
        "scaler": {"mean": scaler.mean, "scale": scaler.scale},
        "metadata": metadata,
    }, checkpoint)
    return model, scaler, metadata


def load_flow_model(path: Path, device):
    data = torch.load(path, map_location=device, weights_only=False)
    metadata = data["metadata"]
    feature_names = metadata["feature_names"]
    if data["model_type"] == "residual_mlp":
        model = ResidualMLP(len(feature_names))
    else:
        model = ResidualGNN(
            len(feature_names),
            include_duals=data["model_type"] != "residual_gnn_no_dual",
        )
    model.load_state_dict(data["state_dict"])
    scaler = StandardScaler(
        np.asarray(data["scaler"]["mean"]),
        np.asarray(data["scaler"]["scale"]),
    )
    return model.to(device), scaler, data
