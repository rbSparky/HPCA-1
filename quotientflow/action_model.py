"""Pure-PyTorch action-value models and decision-calibrated training."""

from __future__ import annotations

import copy
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .action_lookahead import FEATURE_NAMES
from .model import Tower, arch_tensors, dfg_tensors


@dataclass
class FeatureScaler:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, groups: Sequence[dict]) -> "FeatureScaler":
        values = np.concatenate([group["features"] for group in groups], axis=0)
        mean = values.mean(axis=0)
        scale = values.std(axis=0)
        scale[scale < 1e-6] = 1.0
        return cls(mean.astype(np.float32), scale.astype(np.float32))

    def transform(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.mean) / self.scale).astype(np.float32)

    def state_dict(self) -> dict:
        return {"mean": self.mean, "scale": self.scale}

    @classmethod
    def from_state_dict(cls, state: dict) -> "FeatureScaler":
        return cls(np.asarray(state["mean"]), np.asarray(state["scale"]))


class ActionMLP(nn.Module):
    def __init__(self, input_dim: int = len(FEATURE_NAMES)):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


class ActionGNN(nn.Module):
    def __init__(self, feature_dim: int = len(FEATURE_NAMES), hidden_dim: int = 64):
        super().__init__()
        self.dfg_tower = Tower(11, hidden_dim, 3)
        self.arch_tower = Tower(12, hidden_dim, 3)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 5 + feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        dfg,
        arch,
        state,
        operation: int,
        targets: Sequence[tuple],
        new_links: Sequence[Sequence[int]],
        features: torch.Tensor,
    ) -> torch.Tensor:
        device = next(self.parameters()).device
        dfg_features, dfg_edges = dfg_tensors(dfg, arch, state, device)
        arch_features, arch_edges, edge_src, edge_dst, _ = arch_tensors(
            arch, state, device
        )
        dfg_hidden = self.dfg_tower(dfg_features, dfg_edges)
        arch_hidden = self.arch_tower(arch_features, arch_edges)
        dfg_nodes = sorted(dfg.graph)
        arch_nodes = sorted(arch.graph)
        dfg_index = {node: idx for idx, node in enumerate(dfg_nodes)}
        arch_index = {node: idx for idx, node in enumerate(arch_nodes)}
        operation_embedding = dfg_hidden[dfg_index[operation]].unsqueeze(0).expand(
            len(targets), -1
        )
        target_indices = torch.tensor(
            [arch_index[target] for target in targets],
            dtype=torch.long,
            device=device,
        )
        target_embedding = arch_hidden[target_indices]
        pooled_dfg = dfg_hidden.mean(dim=0, keepdim=True).expand(len(targets), -1)
        pooled_arch = arch_hidden.mean(dim=0, keepdim=True).expand(len(targets), -1)
        routing_edge_hidden = (arch_hidden[edge_src] + arch_hidden[edge_dst]) / 2.0
        used_edge_embeddings = []
        for links in new_links:
            if links:
                indices = torch.tensor(links, dtype=torch.long, device=device)
                used_edge_embeddings.append(routing_edge_hidden[indices].mean(dim=0))
            else:
                used_edge_embeddings.append(torch.zeros_like(pooled_arch[0]))
        used_edge_embedding = torch.stack(used_edge_embeddings)
        combined = torch.cat(
            (
                operation_embedding,
                target_embedding,
                pooled_dfg,
                pooled_arch,
                used_edge_embedding,
                features,
            ),
            dim=1,
        )
        return self.head(combined).squeeze(-1)


def model_scores(
    model: nn.Module,
    model_type: str,
    sample: dict,
    group: dict,
    scaler: FeatureScaler,
) -> torch.Tensor:
    device = next(model.parameters()).device
    features = torch.as_tensor(
        scaler.transform(group["features"]), dtype=torch.float32, device=device
    )
    if model_type == "action_mlp":
        return model(features)
    return model(
        sample["dfg"],
        sample["arch"],
        sample["state"],
        group["operation"],
        group["targets"],
        group["new_links"],
        features,
    )


def combined_action_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    differences = target[:, None] - target[None, :]
    upper = torch.triu(torch.ones_like(differences, dtype=torch.bool), diagonal=1)
    valid = upper & (differences.abs() >= 0.05)
    if valid.any():
        i, j = valid.nonzero(as_tuple=True)
        # target_pair = sign(Q_j - Q_i)
        target_pair = torch.sign(target[j] - target[i])
        weights = torch.clamp((target[i] - target[j]).abs() + 0.5, max=5.0)
        pairwise = (
            weights
            * F.softplus(-target_pair * (prediction[j] - prediction[i]))
        ).sum() / weights.sum()
    else:
        pairwise = prediction.sum() * 0.0
    regression = F.smooth_l1_loss(prediction, target)
    oracle_distribution = F.softmax(-target / 0.25, dim=0)
    model_log_distribution = F.log_softmax(-prediction / 0.5, dim=0)
    best_cross_entropy = -(oracle_distribution * model_log_distribution).sum()
    return pairwise + 0.5 * regression + 0.5 * best_cross_entropy


def decision_metrics(prediction: np.ndarray, target: np.ndarray) -> dict:
    order = np.argsort(prediction, kind="stable")
    oracle = np.argsort(target, kind="stable")
    selected = int(order[0])
    best = int(oracle[0])
    pairs_total = pairs_correct = 0
    for i in range(len(target)):
        for j in range(i + 1, len(target)):
            if abs(float(target[i] - target[j])) < 0.05:
                continue
            pairs_total += 1
            pairs_correct += int(
                np.sign(target[j] - target[i])
                == np.sign(prediction[j] - prediction[i])
            )
    return {
        "best_action_preservation": float(selected == best),
        "top2_recall": float(best in set(order[: min(2, len(order))])),
        "pairwise_accuracy": pairs_correct / max(1, pairs_total),
        "selected_regret": float(target[selected]),
        "zero_regret": float(abs(float(target[selected])) <= 1e-8),
    }


@torch.no_grad()
def evaluate_groups(
    model: nn.Module,
    model_type: str,
    groups: Sequence[dict],
    samples: Dict[str, dict],
    scaler: FeatureScaler,
) -> dict:
    model.eval()
    metrics = []
    for group in groups:
        prediction = (
            model_scores(model, model_type, samples[group["state_id"]], group, scaler)
            .detach()
            .cpu()
            .numpy()
        )
        metrics.append(decision_metrics(prediction, group["target"]))
    return {
        key: float(np.mean([metric[key] for metric in metrics]))
        for key in metrics[0]
    }


def train_action_model(
    model_type: str,
    train_groups: Sequence[dict],
    validation_groups: Sequence[dict],
    samples: Dict[str, dict],
    seed: int,
    checkpoint: Path,
    device: str,
    epochs: int = 100,
    patience: int = 15,
) -> Tuple[nn.Module, FeatureScaler, dict]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    scaler = FeatureScaler.fit(train_groups)
    model: nn.Module = ActionMLP() if model_type == "action_mlp" else ActionGNN()
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    generator = random.Random(seed)
    best_key = (-math.inf, -math.inf, -math.inf)
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    stale = 0
    start = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        order = list(range(len(train_groups)))
        generator.shuffle(order)
        total_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        for position, index in enumerate(order):
            group = train_groups[index]
            prediction = model_scores(
                model, model_type, samples[group["state_id"]], group, scaler
            )
            target = torch.as_tensor(
                group["target"], dtype=prediction.dtype, device=prediction.device
            )
            loss = combined_action_loss(prediction, target) / 8.0
            loss.backward()
            total_loss += float(loss.detach())
            if (position + 1) % 8 == 0 or position + 1 == len(order):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        validation = evaluate_groups(
            model, model_type, validation_groups, samples, scaler
        )
        key = (
            validation["best_action_preservation"],
            -validation["selected_regret"],
            validation["pairwise_accuracy"],
        )
        if key > best_key:
            best_key = key
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    metadata = {
        "model_type": model_type,
        "seed": seed,
        "best_epoch": best_epoch,
        "validation_best_action_preservation": best_key[0],
        "validation_selected_regret": -best_key[1],
        "validation_pairwise_accuracy": best_key[2],
        "train_seconds": time.perf_counter() - start,
    }
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_type": model_type,
            "seed": seed,
            "state_dict": best_state,
            "scaler": scaler.state_dict(),
            "metadata": metadata,
        },
        checkpoint,
    )
    return model, scaler, metadata


def load_action_model(checkpoint: Path, device: str):
    data = torch.load(checkpoint, map_location=device, weights_only=False)
    model = ActionMLP() if data["model_type"] == "action_mlp" else ActionGNN()
    model.load_state_dict(data["state_dict"])
    return (
        model.to(device),
        FeatureScaler.from_state_dict(data["scaler"]),
        data["model_type"],
        data,
    )
