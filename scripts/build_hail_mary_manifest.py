#!/usr/bin/env python3
"""Build immutable Hail-Mary pilot work manifests from native problem exports.

The builder emits one fixed-II FlowAdvantage wave at a time.  This makes each
II attempt atomic and resumable while the analysis layer joins attempts by the
base kernel/architecture/method/seed identity.  Native PathFinder/SA rows use
Morpher's own lower-bound-to-upper-bound search in one atomic item.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
KERNELS = ("array_add", "array_cond", "gemm_nt", "fix_fft", "hpcg", "trmm")
ARCHES_CORE = ("A0_hycube4x4", "A1_stdnoc4x4", "A2_hycube4x4_mem_variant")
ARCHES_ALL = ARCHES_CORE + ("A3_hycube8x8",)
PE = {
    "A0_hycube4x4": (4, 4, "HyCUBE_4REG"),
    "A1_stdnoc4x4": (4, 4, "STDNOC_4REG"),
    "A2_hycube4x4_mem_variant": (4, 4, "HyCUBE_4REG"),
    "A3_hycube8x8": (8, 8, "HyCUBE_4REG"),
}
METHODS_FLOW = ("length", "dual_linear", "flow_proposal", "flow_top4")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_tree_hash(root: Path) -> str:
    from scripts.v5b_pilot_queue_common import source_tree_hash as calculate

    return calculate(root)


def row_base(
    *,
    kernel: str,
    architecture: str,
    method: str,
    seed: int,
    budget: int,
    timeout: int,
    dfg: Path,
    mrrg: Path,
    native_dfg: Path,
    native_arch: Path,
    mapper: str,
    toolchain: str,
    checkpoint: Path | None,
    source_commit: str,
    source_hash: str,
    delta: int,
    lower_bound: int,
    strict: bool,
) -> dict[str, object]:
    x, y, pe_type = PE[architecture]
    row: dict[str, object] = {
        "kernel": kernel,
        "architecture": architecture,
        "method": method,
        "seed": seed,
        "budget_seconds": budget,
        "timeout_seconds": timeout,
        "dfg_path": str(dfg),
        "architecture_path": str(mrrg),
        "native_dfg_path": str(native_dfg),
        "native_architecture_path": str(native_arch),
        "x": x,
        "y": y,
        "initial_ii": lower_bound + delta if method in METHODS_FLOW else lower_bound,
        "ii_delta_max": 4,
        "max_ii": lower_bound + 5,
        "pe_type": pe_type,
        "native_method": 0,
        "native_max_iter": 30,
        "evaluation_mode": "empty_state_zero_prefix_fixed_ii" if method in METHODS_FLOW else "native_empty_state_ii_grid",
        "full_end_to_end": True,
        "paper_strict_legality": strict,
        "initialization_policy": "deterministic_length_prefix",
        "initialization_depth_fraction": 0.0,
        "beam_width": 4,
        "k_paths": 4,
        "action_limit": 24,
        "max_expansions": 5000,
        "mapper_binary": mapper,
        "toolchain_path": toolchain,
        "dfg_hash": sha256(dfg),
        "architecture_hash": sha256(mrrg),
        "native_dfg_hash": sha256(native_dfg),
        "native_architecture_hash": sha256(native_arch),
        "toolchain_hash": "",
        "source_commit": source_commit,
        "source_tree_hash": source_hash,
        "checkpoint_path": str(checkpoint) if checkpoint else "",
        "checkpoint_hash": sha256(checkpoint) if checkpoint else "",
        "relaxation_cache_dir": "",
        "relaxation_tau": 0.001,
        # Morpher exports finite ASAP/ALAP latency bounds.  Keep the pilot
        # interval explicit instead of adding artificial II-period padding,
        # which creates native placement variables outside the exported
        # schedule contract and dominates root relaxation cost.
        "relaxation_extra_ii_periods": 0,
        "relaxation_fallback_extra_ii_periods": 1,
        "relaxation_timeout": 120.0,
        "reachable_edge_pruning": True,
        "device": "cpu",
        "requested_ii_delta": delta,
        "lower_bound_ii": lower_bound,
    }
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem-root", type=Path, required=True)
    parser.add_argument("--native-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--remote-root", required=True)
    parser.add_argument("--toolchain", required=True)
    parser.add_argument("--mapper", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--include-a3", action="store_true")
    parser.add_argument("--kernels", default=",".join(KERNELS), help="comma-separated frozen kernel subset")
    parser.add_argument(
        "--architectures",
        help="comma-separated architecture subset (defaults to all core, or core+A3 with --include-a3)",
    )
    parser.add_argument("--delta", type=int)
    parser.add_argument("--native-only", action="store_true")
    args = parser.parse_args()
    problem_root = args.problem_root.resolve()
    native_root = args.native_root.resolve()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    source_hash = source_tree_hash(ROOT)
    selected_kernels = tuple(value.strip() for value in args.kernels.split(",") if value.strip())
    if any(value not in KERNELS for value in selected_kernels):
        raise SystemExit(f"unknown kernel in --kernels: {selected_kernels}")
    if args.architectures:
        arches = tuple(value.strip() for value in args.architectures.split(",") if value.strip())
        unknown_arches = sorted(set(arches) - set(ARCHES_ALL))
        if unknown_arches:
            raise SystemExit(f"unknown architecture in --architectures: {unknown_arches}")
        if not args.include_a3 and "A3_hycube8x8" in arches:
            raise SystemExit("--architectures includes A3; pass --include-a3 explicitly")
    else:
        arches = ARCHES_ALL if args.include_a3 else ARCHES_CORE
    jobs: list[dict[str, object]] = []
    for kernel in selected_kernels:
        native_dfg = native_root / "dfg" / f"{kernel}.xml"
        for architecture in arches:
            pair = problem_root / kernel / architecture
            lb_doc = json.loads((pair / "ii0" / "problem_manifest.json").read_text())
            lower_bound = int(lb_doc["ii"])
            native_arch = Path(args.toolchain) / (
                "source/Morpher_CGRA_Mapper/" +
                {
                    "A0_hycube4x4": "json_arch/hycube_original.json",
                    "A1_stdnoc4x4": "json_arch/stdnoc.json",
                    "A2_hycube4x4_mem_variant": "applications/hycube/array_add/hycube_original_mem.json",
                    "A3_hycube8x8": "json_arch/hycube_8x8.json",
                }[architecture]
            )
            if not native_dfg.is_file() or not native_arch.is_file():
                raise FileNotFoundError(f"native inputs missing for {kernel}/{architecture}")
            def remote_problem(ii: int) -> str:
                return f"{args.remote_root}/problem_exports/{kernel}/{architecture}/ii{ii}"
            if args.delta is None and not args.native_only:
                for method in METHODS_FLOW:
                    ii = lower_bound
                    dfg = pair / "ii0" / "dfg.json"
                    mrrg = pair / "ii0" / "mrrg.json"
                    row = row_base(
                        kernel=kernel, architecture=architecture, method=method,
                        seed=0, budget=600, timeout=600, dfg=Path(remote_problem(0)) / "dfg.json",
                        mrrg=Path(remote_problem(0)) / "mrrg.json", native_dfg=Path(args.remote_root) / "native_inputs/dfg" / f"{kernel}.xml",
                        native_arch=Path(str(native_arch).replace(str(args.toolchain), args.toolchain)), mapper=args.mapper,
                        toolchain=args.toolchain, checkpoint=args.checkpoint,
                        source_commit=args.source_commit, source_hash=source_hash, delta=0,
                        lower_bound=lower_bound, strict=True,
                    )
                    # The selected checkpoint/hash are immutable, but paths in
                    # the manifest must be valid on the remote host.
                    row["checkpoint_path"] = str(Path(args.remote_root) / "checkpoint" / args.checkpoint.name)
                    jobs.append(row)
            if args.delta is not None and not args.native_only:
                ii = lower_bound + args.delta
                for method in METHODS_FLOW:
                    jobs.append(row_base(
                        kernel=kernel, architecture=architecture, method=method,
                        seed=0, budget=600, timeout=600, dfg=Path(remote_problem(ii)) / "dfg.json",
                        mrrg=Path(remote_problem(ii)) / "mrrg.json", native_dfg=Path(args.remote_root) / "native_inputs/dfg" / f"{kernel}.xml",
                        native_arch=native_arch, mapper=args.mapper, toolchain=args.toolchain,
                        checkpoint=args.checkpoint, source_commit=args.source_commit,
                        source_hash=source_hash, delta=args.delta, lower_bound=lower_bound, strict=True,
                    ))
            if args.native_only or args.delta is None:
                for method, seeds in (("native_pathfinder", (0,)), ("native_simulated_annealing", (11, 23, 37))):
                    jobs.extend(row_base(
                        kernel=kernel, architecture=architecture, method=method, seed=seed,
                        budget=600, timeout=600, dfg=Path(args.remote_root) / "native_inputs/dfg" / f"{kernel}.xml",
                        mrrg=native_arch, native_dfg=Path(args.remote_root) / "native_inputs/dfg" / f"{kernel}.xml",
                        native_arch=native_arch, mapper=args.mapper, toolchain=args.toolchain,
                        checkpoint=None, source_commit=args.source_commit, source_hash=source_hash,
                        delta=0, lower_bound=lower_bound, strict=False,
                    ) for seed in seeds)
    # Resolve semantic hashes/config hashes exactly as the remote queue does.
    from scripts.v5b_pilot_queue_common import config_hash, work_id
    for row in jobs:
        row["toolchain_hash"] = ""  # filled by the remote queue's freeze step
        row["config_hash"] = config_hash(row)
        row["work_id"] = work_id(row)
        row["status"] = "PENDING"
        row["attempt"] = 0
    fields = sorted({key for row in jobs for key in row})
    with (output / "jobs.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(jobs)
    (output / "jobs.json").write_text(json.dumps({"jobs": jobs}, indent=2) + "\n")
    print(json.dumps({"jobs": len(jobs), "output": str(output), "source_tree_hash": source_hash}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
