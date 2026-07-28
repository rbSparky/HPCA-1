#!/usr/bin/env python3
"""Create the immutable claim-critical paper-suite work universe."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "paper_suite_v1"
ARCHITECTURES = ["A0_hycube4x4", "A1_stdnoc4x4", "A2_hycube4x4_mem_variant", "A3_hycube8x8"]
KERNELS = ["array_add", "array_cond", "hpcg", "trmm", "gemm_nt", "fix_fft", "fir", "conv2", "conv3", "mac", "matrixmultiply", "dct"]
BUDGETS = [30, 120, 600]
FIELDS = ["work_id", "kernel", "architecture", "method", "seed", "budget_seconds", "status", "attempt", "worker_pid", "start_time", "last_heartbeat", "end_time", "wall_seconds", "cpu_seconds", "peak_rss_mb", "exit_code", "timeout_seconds", "result_path", "stdout_path", "stderr_path", "error_type", "error_message", "config_hash", "source_commit", "checkpoint_hash", "architecture_hash", "kernel_hash"]


def work_id(kernel: str, architecture: str, method: str, seed: int, budget: int) -> str:
    raw = f"paper_suite_v1|{kernel}|{architecture}|{method}|{seed}|{budget}"
    return hashlib.sha256(raw.encode()).hexdigest()[:20]


def add(rows, kernel, arch, method, seed, budget):
    rows.append({"work_id": work_id(kernel, arch, method, seed, budget), "kernel": kernel, "architecture": arch, "method": method, "seed": seed, "budget_seconds": budget, "status": "PENDING", "attempt": 0, "worker_pid": "", "start_time": "", "last_heartbeat": "", "end_time": "", "wall_seconds": "", "cpu_seconds": "", "peak_rss_mb": "", "exit_code": "", "timeout_seconds": budget, "result_path": "", "stdout_path": "", "stderr_path": "", "error_type": "", "error_message": "", "config_hash": "", "source_commit": "", "checkpoint_hash": "", "architecture_hash": "", "kernel_hash": ""})


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = OUT / "work_manifest.csv"
    if manifest.exists():
        raise SystemExit(f"refusing to overwrite immutable manifest: {manifest}")
    rows = []
    deterministic = ["pathfinder", "dual_linear", "flow_proposal", "flow_top4", "flow_noparent_top4"]
    for kernel in KERNELS:
        for arch in ARCHITECTURES:
            for method in deterministic:
                for budget in BUDGETS:
                    add(rows, kernel, arch, method, 0, budget)
            for seed in (11, 23, 37):
                for budget in BUDGETS:
                    add(rows, kernel, arch, "simulated_annealing", seed, budget)
    for kernel in ("array_add", "gemm_nt", "fix_fft"):
        for arch in ARCHITECTURES:
            add(rows, kernel, arch, "full_relaxed_lookahead", 0, 3600)
    # Stable round-robin order, not family-sorted prefixes.
    rows.sort(key=lambda r: (ARCHITECTURES.index(r["architecture"]), KERNELS.index(r["kernel"]), r["method"], int(r["seed"]), int(r["budget_seconds"])))
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader(); writer.writerows(rows)
    (OUT / "work_manifest_metadata.json").write_text(json.dumps({"schema": "flowadvantage_paper_work_manifest_v1", "rows": len(rows), "kernels": KERNELS, "architectures": ARCHITECTURES, "methods": deterministic + ["simulated_annealing", "full_relaxed_lookahead"], "frozen": True}, indent=2) + "\n", encoding="utf-8")
    print(f"created {manifest} with {len(rows)} immutable work items")


if __name__ == "__main__":
    main()
