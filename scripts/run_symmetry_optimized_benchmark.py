#!/usr/bin/env python3
"""Measure optimized exact canonicalization against the preserved 12-case run."""

from __future__ import annotations

import concurrent.futures
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quotientflow.metrics import geometric_mean
from scripts.run_symmetry_revision import worker


def main() -> None:
    output = ROOT / "results" / "revision_v1"
    baseline = pd.read_csv(output / "symmetry_anchor1.csv").sort_values(
        "instance_id"
    )
    jobs = [
        (family, index)
        for family in ("diamond_chain", "reduction_tree", "dot_product")
        for index in range(4)
    ]
    start = time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(max_workers=6) as executor:
        rows = list(executor.map(worker, jobs))
    elapsed = time.perf_counter() - start
    optimized = pd.DataFrame(rows).sort_values("instance_id")
    optimized.to_csv(output / "symmetry_optimized.csv", index=False)

    exact_columns = [
        "nonquot_success",
        "quot_success",
        "nonquot_best_cost",
        "quot_best_cost",
        "cost_match",
        "legality_match",
        "mapped_count_match",
        "nonquot_expansions",
        "quot_expansions",
        "nonquot_timeout",
        "quot_timeout",
    ]
    identical_search = bool(
        baseline["instance_id"].tolist() == optimized["instance_id"].tolist()
        and baseline[exact_columns].reset_index(drop=True).equals(
            optimized[exact_columns].reset_index(drop=True)
        )
    )
    old_fraction = float(
        baseline["canonicalization_ms"].sum() / baseline["quot_total_ms"].sum()
    )
    new_fraction = float(
        optimized["canonicalization_ms"].sum()
        / optimized["quot_total_ms"].sum()
    )
    assessment = {
        "identical_search_results": identical_search,
        "cases": len(optimized),
        "all_legal_cost_matches": bool(
            optimized[
                ["cost_match", "legality_match", "mapped_count_match"]
            ].all(axis=None)
        ),
        "old_canonicalization_fraction": old_fraction,
        "new_canonicalization_fraction": new_fraction,
        "old_canonicalization_ms_sum": float(
            baseline["canonicalization_ms"].sum()
        ),
        "new_canonicalization_ms_sum": float(
            optimized["canonicalization_ms"].sum()
        ),
        "canonicalization_work_speedup": float(
            baseline["canonicalization_ms"].sum()
            / optimized["canonicalization_ms"].sum()
        ),
        "old_quotient_runtime_ms_sum": float(baseline["quot_total_ms"].sum()),
        "new_quotient_runtime_ms_sum": float(optimized["quot_total_ms"].sum()),
        "quotient_runtime_speedup": float(
            baseline["quot_total_ms"].sum()
            / optimized["quot_total_ms"].sum()
        ),
        "geomean_expansion_reduction": geometric_mean(
            optimized["expansion_reduction"]
        ),
        "benchmark_wall_seconds": elapsed,
        "confirmed_below_25_percent": new_fraction <= 0.25,
    }
    (output / "symmetry_optimized_assessment.json").write_text(
        json.dumps(assessment, indent=2) + "\n"
    )
    print(json.dumps(assessment, indent=2))


if __name__ == "__main__":
    main()
