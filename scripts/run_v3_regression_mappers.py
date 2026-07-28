#!/usr/bin/env python3
"""Run FlowAdvantage methods on the immutable 56-instance regression suite."""

from __future__ import annotations

import concurrent.futures
import json
import pickle
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_v3_mappers import run_instance  # noqa: E402

OUT = ROOT / "results" / "revision_v3"
RAW = OUT / "raw"
CACHE = OUT / "cache" / "regression"


def worker(args):
    return run_instance(*args)


def main():
    started = time.perf_counter()
    CACHE.mkdir(parents=True, exist_ok=True)
    v2 = pd.read_csv(
        ROOT / "results/revision_v2/raw/action_mapper_runs.csv"
    )
    ids = sorted(v2[
        (v2.method == "length") & v2.split.str.startswith("test")
    ].sample_id.unique())
    records = []
    for sample_id in ids:
        path = ROOT / "results/cache/samples" / f"{sample_id}.pkl"
        with path.open("rb") as handle:
            sample = pickle.load(handle)
        instance = {
            "instance_id": sample_id,
            "split": sample["split"],
            "architecture": sample["architecture"],
            "dfg_family": sample["family"],
            "seed": sample["dfg"].seed,
            "dfg": sample["dfg"],
            "arch": sample["arch"],
            "state": sample["state"],
            "normalized_prices": sample["normalized_prices"],
            "run_budgets": ("wide",),
        }
        output = CACHE / f"{sample_id}.pkl"
        with output.open("wb") as handle:
            pickle.dump(instance, handle, protocol=pickle.HIGHEST_PROTOCOL)
        records.append({
            "instance_id": sample_id,
            "split": sample["split"],
            "dfg_family": sample["family"],
            "instance_path": str(output.relative_to(ROOT)),
        })
    manifest = pd.DataFrame(records)
    expensive = set(
        manifest.sort_values("instance_id")
        .groupby(["split", "dfg_family"], sort=True)
        .head(1).instance_id
    )
    tasks = [
        (row.instance_path, row.instance_id in expensive)
        for row in manifest.itertuples()
    ]
    rows = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
        for result in pool.map(worker, tasks):
            rows.extend(result)
            pd.DataFrame(rows).to_csv(
                RAW / "regression_mapper_runs.partial.csv", index=False
            )
    frame = pd.DataFrame(rows).sort_values(
        ["method", "split", "dfg_family", "instance_id"]
    )
    frame.to_csv(RAW / "regression_mapper_runs.csv", index=False)
    runtime = {
        "seconds": time.perf_counter() - started,
        "rows": len(frame),
        "instances": len(manifest),
        "expensive_instances": len(expensive),
        "timeouts": int(frame.timeout.sum()),
    }
    (OUT / "regression_mapper_runtime.json").write_text(
        json.dumps(runtime, indent=2) + "\n"
    )
    print(json.dumps(runtime, indent=2))
    print(frame.groupby("method").success.agg(["size", "mean"]).to_string())


if __name__ == "__main__":
    main()
