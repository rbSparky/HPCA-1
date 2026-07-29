#!/usr/bin/env python3
"""Build a paper manifest only from frozen, semantic native artifacts.

This command intentionally emits a *partial* manifest when the repository
does not contain all requested real-kernel contracts.  It never invents a path,
hash, architecture, or result row.  Missing work is written to the manifest
metadata as an explicit blocker.
"""
from __future__ import annotations

import argparse, hashlib, json, math, os, subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.v5b_pilot_queue_common import (
    config_hash,
    sha256_file,
    source_tree_hash,
    work_id,
)
METHODS = ("length", "dual_linear", "flow_proposal", "flow_top4", "full_relaxed_lookahead")
KERNELS = ("array_add", "array_cond", "hpcg", "trmm", "gemm_nt", "fix_fft")
ARCHES = ("A0_hycube4x4", "A1_stdnoc4x4", "A2_hycube4x4_mem_variant", "A3_hycube8x8")

def _git() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()

def _wid(row: dict[str, Any]) -> str:
    raw = "paper_suite_v2|" + "|".join(str(row[k]) for k in ("kernel", "architecture", "method", "seed", "budget_seconds"))
    return hashlib.sha256(raw.encode()).hexdigest()[:20]

def _canonical_dirs(root: Path, kernel: str, arch: str) -> list[Path]:
    """Return only complete native JSON contract directories.

    Native queue exports place the contract under ``export/``; older frozen
    references may expose the same three files directly.  Incomplete pulls and
    mapper text outputs are never accepted.
    """
    candidates: list[tuple[int, str, Path]] = []
    kernel_root = root / kernel
    for path in sorted(kernel_root.glob(f"{arch}*")):
        for directory in (path / "export", path, path / "native_dump"):
            if not all((directory / name).is_file() for name in ("dfg.json", "mrrg.json", "mapping.json")):
                continue
            try:
                dfg = json.loads((directory / "dfg.json").read_text())
                mrrg = json.loads((directory / "mrrg.json").read_text())
                mapping = json.loads((directory / "mapping.json").read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if dfg.get("schema") != "flowadvantage_morpher_dfg_v1":
                continue
            if mrrg.get("schema") != "flowadvantage_morpher_mrrg_v1":
                continue
            if mapping.get("schema") != "flowadvantage_morpher_mapping_v1":
                continue
            if mapping.get("dfg_hash") != dfg.get("dfg_hash") or mapping.get("architecture_hash") != mrrg.get("architecture_hash"):
                continue
            resource_ids = {
                str(resource.get("native_resource_id", resource.get("resource_id", resource.get("id", ""))))
                for resource in mrrg.get("resources", [])
            }
            if not resource_ids or any(
                str(edge.get("src")) not in resource_ids
                or str(edge.get("dst")) not in resource_ids
                for edge in mrrg.get("edges", [])
            ):
                continue
            priority = 0 if directory.name == "export" else (1 if directory.name == path.name else 2)
            candidates.append((priority, str(directory), directory))
    return [item[2] for item in sorted(candidates)]

def _select(root: Path) -> tuple[dict[tuple[str,str], Path], list[str]]:
    selected, blockers = {}, []
    for kernel in KERNELS:
        for arch in ARCHES:
            found = _canonical_dirs(root, kernel, arch)
            if found:
                selected[(kernel, arch)] = found[0]
            else:
                blockers.append(f"missing canonical dfg/mrrg/mapping: {kernel}/{arch}")
    return selected, blockers

def build(args: argparse.Namespace) -> tuple[Path, Path]:
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    refs, blockers = _select(args.reference_root.resolve())
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        blockers.append(f"missing selected checkpoint: {checkpoint}")
    source_commit = _git()
    source_hash = source_tree_hash(ROOT)
    jobs = []
    for (kernel, arch), directory in sorted(refs.items()):
        dfg = directory / "dfg.json"; mrrg = directory / "mrrg.json"; witness = directory / "mapping.json"
        dfg_doc = json.loads(dfg.read_text()); mrrg_doc = json.loads(mrrg.read_text())
        # Prefix depth is frozen before execution and built from the empty
        # state by the deterministic length beam.  It is not copied from the
        # native witness mapping.
        anchor = max(1, min(
            len(dfg_doc.get("nodes", [])) - 1,
            int(math.ceil(0.40 * len(dfg_doc.get("nodes", []))))
        ))
        for method in METHODS:
            if method == "full_relaxed_lookahead" and (kernel, arch) not in {("array_add", "A0_hycube4x4"), ("array_add", "A1_stdnoc4x4"), ("array_add", "A2_hycube4x4_mem_variant"), ("gemm_nt", "A0_hycube4x4"), ("gemm_nt", "A1_stdnoc4x4"), ("gemm_nt", "A2_hycube4x4_mem_variant"), ("fix_fft", "A0_hycube4x4"), ("fix_fft", "A1_stdnoc4x4"), ("fix_fft", "A2_hycube4x4_mem_variant")}:
                continue
            row = {
                "kernel": kernel, "architecture": arch, "method": method,
                "seed": 0, "budget_seconds": 600 if method != "full_relaxed_lookahead" else 3600,
                "timeout_seconds": 600 if method != "full_relaxed_lookahead" else 3600,
                "dfg_path": str(dfg.resolve()), "architecture_path": str(mrrg.resolve()),
                "reference_mapping_path": str(witness.resolve()), "dfg_hash": sha256_file(dfg),
                "architecture_hash": sha256_file(mrrg), "reference_mapping_hash": sha256_file(witness),
                "checkpoint_path": str(checkpoint),
                # The complete algorithm starts at the empty state, constructs
                # a frozen deterministic length prefix, then applies the
                # selected completion policy.
                "evaluation_mode": "deterministic_length_prefix_then_complete",
                "anchor_depth_fraction": anchor / max(1, len(dfg_doc.get("nodes", []))),
                "initialization_policy": "deterministic_length_prefix",
                "initialization_depth_fraction": 0.40,
                "full_end_to_end": True,
                "checkpoint_hash": sha256_file(checkpoint) if checkpoint.is_file() else "",
                "source_commit": source_commit, "source_tree_hash": source_hash,
                "beam_width": 4, "k_paths": 4, "action_limit": 24,
                "max_expansions": 5000, "initial_ii": int(mrrg_doc["ii"]),
                "x": max((int(r.get("x", 0)) for r in mrrg_doc.get("resources", [])), default=0)+1,
                "y": max((int(r.get("y", 0)) for r in mrrg_doc.get("resources", [])), default=0)+1,
                "pe_type": "HY_CUBE", "native_method": 0,
                "relaxation_cache_dir": str((out / "cache").resolve()),
                "relaxation_tau": 0.001, "relaxation_timeout": 120.0,
                "device": args.device,
            }
            row["work_id"] = work_id(row); row["status"] = "PENDING"; row["attempt"] = 0
            row["result_path"] = str((out / "work_items" / f"{row['work_id']}.json").resolve())
            row["heartbeat_path"] = str((out / "heartbeats" / f"{row['work_id']}.json").resolve())
            row["stdout_path"] = str((out / "logs" / f"{row['work_id']}.stdout.log").resolve())
            row["stderr_path"] = str((out / "logs" / f"{row['work_id']}.stderr.log").resolve())
            row["config_hash"] = config_hash(row)
            jobs.append(row)
    jobs.sort(key=lambda r: (r["architecture"], r["kernel"], r["method"]))
    jobs_path = out / "jobs.json"; jobs_path.write_text(json.dumps({"jobs": jobs}, indent=2) + "\n")
    metadata = {"schema": "flowadvantage_paper_manifest_v2", "frozen": True, "source_commit": source_commit, "source_tree_hash": source_hash, "jobs": len(jobs), "blockers": blockers, "selected": {f"{k[0]}/{k[1]}": str(v) for k,v in refs.items()}, "unsupported_methods": ["native_pathfinder", "simulated_annealing", "flow_noparent_top4"], "launch": args.launch, "evaluation_mode": "deterministic_length_prefix_then_complete", "initialization_policy": "deterministic_length_prefix", "initialization_depth_fraction": 0.40, "full_end_to_end": True, "protocol_note": "current MRRGs provide native FU latencies; the witness is provenance only, while all placements/routes are built from the empty state by the frozen deterministic length prefix"}
    (out / "manifest_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return jobs_path, out / "manifest_metadata.json"

def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--output", type=Path, default=ROOT / "results/paper_suite_v2"); p.add_argument("--reference-root", type=Path, default=ROOT / "results/revision_v5b/reference_mappings"); p.add_argument("--checkpoint", type=Path, default=ROOT / "results/revision_v3/checkpoints/residual_gnn_seed_23.pt"); p.add_argument("--device", default="cpu"); p.add_argument("--launch", choices=("none", "local", "cluster"), default="none"); a = p.parse_args(); jobs, meta = build(a); print(json.dumps({"jobs": str(jobs), "metadata": str(meta)}, indent=2));
    if a.launch == "local": print("launch: python scripts/run_v5b_pilot_queue.py --jobs", jobs)
    if a.launch == "cluster": print("launch: use tools/cluster_queue.sh with jobs.json; no remote launch performed")
if __name__ == "__main__": main()
