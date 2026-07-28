"""Training and prediction metrics for the pure-PyTorch model."""

from __future__ import annotations

import copy
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from .model import QuotientFlowGNN, predict_numpy


def critical_labels(raw):
    raw = np.asarray(raw)
    labels = np.zeros(len(raw), dtype=np.float32)
    positive = np.flatnonzero(raw > 1e-8)
    if not len(positive):
        return labels, False
    count = max(1, int(math.ceil(0.1 * len(positive))))
    chosen = positive[np.argsort(raw[positive], kind="stable")[-count:]]
    labels[chosen] = 1.0
    return labels, True


def graph_loss(model, sample, cfg, generator):
    pred, logits = model(sample["dfg"], sample["arch"], sample["state"])
    target = torch.as_tensor(sample["normalized_prices"], dtype=pred.dtype, device=pred.device)
    mse = F.mse_loss(pred, target)
    valid_pairs = sample.get("rank_pairs")
    if valid_pairs is None:
        values = target.detach().cpu().numpy()
        indices = np.argwhere(np.triu(np.abs(values[:, None] - values[None, :]) >= 0.1, 1))
        valid_pairs = [tuple(x) for x in indices.tolist()]
    if valid_pairs:
        chosen = random.Random(generator).sample(valid_pairs, min(64, len(valid_pairs)))
        a = torch.tensor([x for x, _ in chosen], device=pred.device)
        b = torch.tensor([y for _, y in chosen], device=pred.device)
        sign = torch.sign(target[a] - target[b])
        rank = F.softplus(-sign * (pred[a] - pred[b])).mean()
    else:
        rank = pred.sum() * 0.0
    labels, has_critical = critical_labels(sample["raw_prices"])
    if has_critical:
        y = torch.as_tensor(labels, dtype=pred.dtype, device=pred.device)
        positives = max(1.0, float(y.sum()))
        pos_weight = torch.tensor((len(y) - positives) / positives, device=pred.device)
        bce = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)
    else:
        bce = logits.sum() * 0.0
    return (
        cfg["loss_mse_weight"] * mse
        + cfg["loss_rank_weight"] * rank
        + cfg["loss_critical_weight"] * bce
    )


def _median_spearman(model, samples):
    values = []
    for sample in samples:
        pred, _ = predict_numpy(model, sample["dfg"], sample["arch"], sample["state"])
        target = sample["normalized_prices"]
        corr = spearmanr(target, pred).statistic
        values.append(0.0 if not np.isfinite(corr) else corr)
    return float(np.median(values)) if values else float("nan")


def train_seed(train, validation, cfg, seed, device, checkpoint: Path):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    model = QuotientFlowGNN(
        cfg["hidden_dim"], cfg["dfg_layers"], cfg["arch_layers"], cfg["head_layers"]
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"]
    )
    best_score, best_epoch, stale = -float("inf"), 0, 0
    best_state = copy.deepcopy(model.state_dict())
    start = time.perf_counter()
    order_rng = random.Random(seed)
    for epoch in range(cfg["epochs"]):
        model.train()
        order = list(range(len(train)))
        order_rng.shuffle(order)
        optimizer.zero_grad(set_to_none=True)
        for position, idx in enumerate(order):
            loss = graph_loss(model, train[idx], cfg, seed * 100000 + epoch * 1000 + idx)
            (loss / cfg["batch_size"]).backward()
            if (position + 1) % cfg["batch_size"] == 0 or position + 1 == len(order):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        score = _median_spearman(model, validation)
        if score > best_score + 1e-6:
            best_score, best_epoch, stale = score, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= cfg["early_stop_patience"]:
                break
    model.load_state_dict(best_state)
    train_seconds = time.perf_counter() - start
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state,
            "seed": seed,
            "best_epoch": best_epoch,
            "validation_spearman": best_score,
            "train_seconds": train_seconds,
            "config": cfg,
        },
        checkpoint,
    )
    return model, {
        "seed": seed,
        "best_epoch": best_epoch,
        "validation_spearman": best_score,
        "train_seconds": train_seconds,
    }


def prediction_metrics(model, samples):
    targets, predictions, labels, logits = [], [], [], []
    graph_corr, recalls = [], []
    for sample in samples:
        pred, logit = predict_numpy(model, sample["dfg"], sample["arch"], sample["state"])
        target = np.asarray(sample["normalized_prices"])
        corr = spearmanr(target, pred).statistic
        graph_corr.append(0.0 if not np.isfinite(corr) else corr)
        top = min(8, len(target))
        true_top = set(np.argsort(target, kind="stable")[-top:])
        pred_top = set(np.argsort(pred, kind="stable")[-top:])
        recalls.append(len(true_top & pred_top) / top)
        label, valid = critical_labels(sample["raw_prices"])
        targets.extend(target)
        predictions.extend(pred)
        if valid:
            labels.extend(label)
            logits.extend(logit)
    targets, predictions = np.asarray(targets), np.asarray(predictions)
    labels, logits = np.asarray(labels), np.asarray(logits)
    return {
        "price_mae": float(np.mean(np.abs(targets - predictions))),
        "price_rmse": float(np.sqrt(np.mean((targets - predictions) ** 2))),
        "median_spearman": float(np.median(graph_corr)),
        "mean_spearman": float(np.mean(graph_corr)),
        "critical_auroc": float(roc_auc_score(labels, logits)) if len(np.unique(labels)) == 2 else float("nan"),
        "critical_average_precision": float(average_precision_score(labels, logits)) if len(labels) else float("nan"),
        "top8_recall": float(np.mean(recalls)),
    }
