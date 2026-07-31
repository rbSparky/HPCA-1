#!/usr/bin/env python3
"""Build a native Morpher PathFinder/SA manifest for the Hail-Mary run.

Unlike the FlowAdvantage manifest, native jobs do not require a serialized
problem-only MRRG.  Morpher constructs its own II-expanded MRRG from the
native architecture JSON, and ``-i 0`` asks the native mapper to derive the
resource/recurrence lower bound.  This makes the builder usable for held-out
architectures for which a problem-only export was intentionally not available.

The output is consumed by :mod:`scripts.run_v5b_pilot_queue`; no result is
written by this builder and an existing non-empty output directory is never
overwritten.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.v5b_pilot_queue_common import config_hash, source_tree_hash

ROOT = Path(__file__).resolve().parents[1]
KERNELS = ("array_add", "array_cond", "gemm_nt", "fix_fft", "hpcg", "trmm")
ARCHES = ("A0_hycube4x4", "A1_stdnoc4x4", "A2_hycube4x4_mem_variant", "A3_hycube8x8")
ARCH_SPEC = {
    "A0_hycube4x4": (4, 4, "HyCUBE_4REG", "source/Morpher_CGRA_Mapper/json_arch/hycube_original.json"),
    "A1_stdnoc4x4": (4, 4, "STDNOC_4REG", "source/Morpher_CGRA_Mapper/json_arch/stdnoc.json"),
    "A3_hycube8x8": (8, 8, "HyCUBE_4REG", "source/Morpher_CGRA_Mapper/json_arch/hycube_8x8.json"),
}


def _sha(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _base(
    *,
    kernel: str,
    architecture: str,
    method: str,
    seed: int,
    dfg: Path,
    arch: Path,
    mapper: Path,
    toolchain: Path,
    source_commit: str,
    source_hash: str,
) -> dict[str, Any]:
    x, y, pe_type, _ = ARCH_SPEC[architecture]
    row: dict[str, Any] = {
        "kernel": kernel,
        "architecture": architecture,
        "method": method,
        "seed": seed,
        "budget_seconds": 1200,
        "timeout_seconds": 1800,
        "dfg_path": str(dfg.resolve()),
        "architecture_path": str(arch.resolve()),
        "x": x,
        "y": y,
        "initial_ii": 0,
        "ii_delta_max": 4,
        "max_ii": 5,
        "pe_type": pe_type,
        "native_method": 0,
        "native_max_iter": 30,
        "evaluation_mode": "native_empty_state_ii_grid",
        "full_end_to_end": True,
        "paper_strict_legality": False,
        "initialization_policy": "deterministic_length_prefix",
        "initialization_depth_fraction": 0.0,
        "beam_width": 4,
        "k_paths": 4,
        "action_limit": 24,
        "max_expansions": 5000,
        "mapper_binary": str(mapper.resolve()),
        "toolchain_path": str(toolchain.resolve()),
        "dfg_hash": _sha(dfg),
        "architecture_hash": _sha(arch),
        "native_dfg_path": "",
        "native_architecture_path": "",
        "native_dfg_hash": "",
        "native_architecture_hash": "",
        "toolchain_hash": _sha(mapper),
        "source_commit": source_commit,
        "source_tree_hash": source_hash,
        "checkpoint_path": "",
        "checkpoint_hash": "",
        "relaxation_cache_dir": "",
        "relaxation_tau": 0.001,
        "relaxation_extra_ii_periods": 0,
        "relaxation_fallback_extra_ii_periods": 1,
        "relaxation_timeout": 120.0,
        "relaxation_max_threads": 1,
        "reachable_edge_pruning": True,
        "device": "cpu",
        "native_legality_required": True,
    }
    row["config_hash"] = config_hash(row)
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--native-root", type=Path, required=True)
    parser.add_argument("--toolchain", type=Path, required=True)
    parser.add_argument("--mapper", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--kernels", default=",".join(KERNELS))
    parser.add_argument("--architectures", default=",".join(ARCHES))
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    kernels = tuple(v.strip() for v in args.kernels.split(",") if v.strip())
    architectures = tuple(v.strip() for v in args.architectures.split(",") if v.strip())
    if set(kernels) - set(KERNELS) or set(architectures) - set(ARCHES):
        raise SystemExit("unknown kernel or architecture")
    source_hash = source_tree_hash(ROOT)
    jobs: list[dict[str, Any]] = []
    for kernel in kernels:
        dfg = (args.native_root / "dfg" / f"{kernel}.xml").resolve()
        if not dfg.is_file():
            raise FileNotFoundError(dfg)
        for architecture in architectures:
            if architecture == "A2_hycube4x4_mem_variant":
                repaired = args.native_root.parent / "a2_repaired_arch" / kernel / "hycube_original_mem.json"
                if repaired.is_file():
                    arch = repaired.resolve()
                else:
                    arch = (args.toolchain / "source/Morpher_CGRA_Mapper/applications/hycube/array_add/hycube_original_mem.json").resolve()
            else:
                arch = (args.toolchain / ARCH_SPEC[architecture][3]).resolve()
            if not arch.is_file():
                raise FileNotFoundError(arch)
            jobs.append(_base(kernel=kernel, architecture=architecture, method="native_pathfinder", seed=0, dfg=dfg, arch=arch, mapper=args.mapper, toolchain=args.toolchain, source_commit=args.source_commit, source_hash=source_hash))
            for seed in (11, 23, 37):
                jobs.append(_base(kernel=kernel, architecture=architecture, method="native_simulated_annealing", seed=seed, dfg=dfg, arch=arch, mapper=args.mapper, toolchain=args.toolchain, source_commit=args.source_commit, source_hash=source_hash))
    (output / "jobs.json").write_text(json.dumps({"jobs": jobs}, indent=2) + "\n", encoding="utf-8")
    (output / "manifest_metadata.json").write_text(json.dumps({"schema": "flowadvantage_hail_native_manifest_v1", "jobs": len(jobs), "source_commit": args.source_commit, "source_tree_hash": source_hash, "kernels": kernels, "architectures": architectures, "seeds": [11, 23, 37]}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"jobs": len(jobs), "output": str(output), "source_tree_hash": source_hash}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
