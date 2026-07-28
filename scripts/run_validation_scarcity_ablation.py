#!/usr/bin/env python
"""Validation-only residual alternative-path scarcity ablation."""

from __future__ import annotations

import concurrent.futures
import itertools
import pickle
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quotientflow.dfg import FAMILIES
from quotientflow.mapper import map_dfg
from quotientflow.partial_state import residual_graph


def scarcity_adjusted(sample):
    arch, state = sample["arch"], sample["state"]
    adjusted = np.asarray(sample["normalized_prices"]).copy()
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
                    nx.shortest_simple_paths(
                        reduced, u, v, weight="base_cost"
                    ),
                    4,
                )
            )
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            alternatives = 0
        adjusted[edge_id] /= 1.0 + alternatives
    return adjusted


def worker(path):
    with path.open("rb") as handle:
        sample = pickle.load(handle)
    scarcity = scarcity_adjusted(sample)
    rows = []
    settings = {
        "length": (None, 0.0),
        "oracle_top2_w0.5": (sample["normalized_prices"], 0.5),
        "scarcity_top2_w0.5": (scarcity, 0.5),
        "scarcity_top2_w1.0": (scarcity, 1.0),
        "scarcity_sum_w0.5": (scarcity, 0.5),
        "scarcity_sum_w1.0": (scarcity, 1.0),
    }
    for method, (prices, weight) in settings.items():
        result = map_dfg(
            sample["dfg"],
            sample["arch"],
            sample["state"],
            prices=prices,
            beam_width=4,
            action_limit=6,
            k_paths=4,
            max_expansions=5000,
            timeout_seconds=120.0,
            price_weight=weight,
            price_mode="sum" if "_sum_" in method else "top2",
        )
        rows.append(
            {
                "sample_id": sample["sample_id"],
                "architecture": sample["architecture"],
                "dfg_family": sample["family"],
                "method": method,
                **result.row(),
            }
        )
    return rows


def main():
    cache = ROOT / "results/cache/samples"
    paths = [
        sorted(cache.glob(f"val_seen__{arch}__{family}__*.pkl"))[0]
        for family in FAMILIES
        for arch in ("mesh3", "torus3")
    ]
    rows = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=6) as executor:
        for result in executor.map(worker, paths):
            rows.extend(result)
    frame = pd.DataFrame(rows)
    output = ROOT / "results/revision_v1"
    frame.to_csv(output / "validation_scarcity_ablation.csv", index=False)
    summary = (
        frame.groupby("method", as_index=False)
        .agg(
            success_rate=("success", "mean"),
            successes=("success", "sum"),
            mean_cost=("best_cost", "mean"),
            median_expansions=("expansions", "median"),
            median_runtime_ms=("runtime_ms", "median"),
            timeouts=("timeout", "sum"),
        )
    )
    summary.to_csv(
        output / "validation_scarcity_ablation_summary.csv", index=False
    )
    print(
        summary.sort_values(
            ["success_rate", "mean_cost"], ascending=[False, True]
        ).to_string(index=False)
    )


if __name__ == "__main__":
    main()
