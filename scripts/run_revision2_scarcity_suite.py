#!/usr/bin/env python
"""Final frozen test: validation-selected alternative-path scarcity."""

from __future__ import annotations

import concurrent.futures
import itertools
import json
import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quotientflow.mapper import map_dfg
from quotientflow.model import QuotientFlowGNN, predict_numpy
from quotientflow.partial_state import residual_graph
from scripts.run_revision_suite import (
    METHODS,
    aggregate,
    gates,
    selected_samples,
)

SELECTION = {
    "source": "results/revision_v1/validation_scarcity_ablation_summary.csv",
    "beam_width": 4,
    "price_mode": "sum",
    "price_weight": 0.5,
    "alternative_paths_cap": 4,
    "selection_reason": (
        "All scarcity candidates retained 14/14 validation successes; sum_w0.5 "
        "had the lowest mean successful route cost (17.089 vs 17.961)."
    ),
    "final_test_revision": True,
}


def adjust(sample, values):
    arch, state = sample["arch"], sample["state"]
    adjusted = np.asarray(values).copy()
    graph = residual_graph(arch, state.occupied_links)
    for edge_id in np.flatnonzero(adjusted > 1e-10):
        u, v = arch.edge_list[int(edge_id)]
        if not graph.has_edge(u, v):
            adjusted[edge_id] = 0.0
            continue
        reduced = graph.copy()
        reduced.remove_edge(u, v)
        try:
            alternatives = sum(
                1
                for _ in itertools.islice(
                    nx.shortest_simple_paths(reduced, u, v, weight="base_cost"),
                    SELECTION["alternative_paths_cap"],
                )
            )
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            alternatives = 0
        adjusted[edge_id] /= 1.0 + alternatives
    return adjusted


def worker(payload):
    sample, predicted = payload
    oracle = adjust(sample, sample["normalized_prices"])
    predicted = adjust(sample, predicted)
    definitions = {
        "length": (None, False),
        "quotient_length": (None, True),
        "oracle_dual": (oracle, False),
        "oracle_dual_quotient": (oracle, True),
        "pred_dual": (predicted, False),
        "pred_dual_quotient": (predicted, True),
    }
    rows = []
    for method in METHODS:
        prices, quotient = definitions[method]
        result = map_dfg(
            sample["dfg"],
            sample["arch"],
            sample["state"],
            prices=prices,
            beam_width=SELECTION["beam_width"],
            action_limit=6,
            k_paths=4,
            max_expansions=5000,
            timeout_seconds=120.0,
            price_weight=SELECTION["price_weight"],
            price_mode=SELECTION["price_mode"],
            quotient=quotient,
        )
        rows.append(
            {
                "sample_id": sample["sample_id"],
                "split": sample["split"],
                "architecture": sample["architecture"],
                "dfg_family": sample["family"],
                "method": method,
                **result.row(),
            }
        )
    return rows


def main():
    output = ROOT / "results/revision_v1"
    (output / "selection_scarcity.json").write_text(
        json.dumps(SELECTION, indent=2) + "\n"
    )
    checkpoint = torch.load(
        ROOT / "results/quick_v1/checkpoints/model_seed_11.pt",
        map_location="cuda",
        weights_only=False,
    )
    model = QuotientFlowGNN().cuda()
    model.load_state_dict(checkpoint["state_dict"])
    samples = selected_samples()
    payloads = [
        (
            sample,
            predict_numpy(
                model, sample["dfg"], sample["arch"], sample["state"]
            )[0],
        )
        for sample in samples
    ]
    start = time.perf_counter()
    rows = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=6) as executor:
        for result in executor.map(worker, payloads):
            rows.extend(result)
            pd.DataFrame(rows).sort_values(
                ["split", "dfg_family", "sample_id", "method"]
            ).to_csv(output / "mappings_scarcity.csv", index=False)
    frame = pd.DataFrame(rows)
    summary = aggregate(frame)
    summary.to_csv(output / "mapper_scarcity_summary.csv", index=False)
    assessment = gates(summary)
    assessment["runtime_seconds"] = time.perf_counter() - start
    (output / "gate_scarcity_assessment.json").write_text(
        json.dumps(assessment, indent=2) + "\n"
    )
    print(summary.to_string(index=False))
    print(json.dumps(assessment, indent=2))


if __name__ == "__main__":
    main()
