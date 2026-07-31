#!/usr/bin/env python3
"""Export the native, mapping-free Morpher problem grid for the frozen pilot.

The export is deliberately performed by the pinned Morpher container.  No MRRG
is reconstructed in Python: every ``mrrg.json`` is emitted by Morpher's native
II expansion and is validated against its problem manifest before admission.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMAGE = "quotientflow-morpher-hail-problem-export:dev"
KERNELS = {
    "array_add": ROOT / "results/revision_v5b/morpher_patch/Morpher_CGRA_Mapper/applications/array_add/array_add_INNERMOST_LN1_PartPred_DFG.xml",
    "array_cond": ROOT / "results/revision_v5b/generated_dfg_v2/array_cond/array_cond_PartPredDFG.xml",
    "gemm_nt": ROOT / "results/revision_v5b/morpher_patch/Morpher_CGRA_Mapper/applications/gemm_nt/gemm_nt_INNERMOST_LN111_PartPred_DFG.xml",
    "fix_fft": ROOT / "results/revision_v5b/morpher_patch/Morpher_CGRA_Mapper/applications/fix_fft_npb/fix_fft_INNERMOST_LN111_PartPred_DFG.xml",
    "hpcg": ROOT / "results/revision_v5b/generated_dfg_v2/hpcg/hpcg_PartPredDFG.xml",
    "trmm": ROOT / "results/revision_v5b/generated_dfg_v2/trmm/trmm_PartPredDFG.xml",
}
ARCHES = {
    "A0_hycube4x4": ("json_arch/hycube_original.json", 4, 4, "HyCUBE_4REG"),
    "A1_stdnoc4x4": ("json_arch/stdnoc.json", 4, 4, "STDNOC_4REG"),
    "A2_hycube4x4_mem_variant": (
        "applications/hycube/array_add/hycube_original_mem.json",
        4,
        4,
        "HyCUBE_4REG",
    ),
    "A3_hycube8x8": ("json_arch/hycube_8x8.json", 8, 8, "HyCUBE_4REG"),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_one(
    image: str,
    stage: Path,
    output: Path,
    kernel: str,
    architecture: str,
    dfg: Path,
    arch_ref: str,
    x: int,
    y: int,
    pe_type: str,
    ii: int,
) -> dict[str, object]:
    pair_out = output / kernel / architecture / f"ii{ii}"
    if pair_out.exists():
        complete = all(
            (pair_out / name).is_file()
            for name in ("problem_manifest.json", "dfg.json", "mrrg.json")
        )
        if not complete:
            partial = pair_out.with_name(
                f"{pair_out.name}.partial_{os.getpid()}"
            )
            pair_out.rename(partial)
        else:
            manifest = json.loads((pair_out / "problem_manifest.json").read_text())
            dfg_doc = json.loads((pair_out / "dfg.json").read_text())
            mrrg_doc = json.loads((pair_out / "mrrg.json").read_text())
            actual_ii = int(manifest.get("ii", -1))
            if (
                actual_ii <= 0
                or dfg_doc.get("ii") != actual_ii
                or mrrg_doc.get("ii") != actual_ii
                or manifest.get("dfg_hash") != dfg_doc.get("dfg_hash")
                or manifest.get("architecture_hash") != mrrg_doc.get("architecture_hash")
            ):
                raise RuntimeError(f"invalid existing native export: {pair_out}")
            return {
                "kernel": kernel,
                "architecture": architecture,
                "ii": ii,
                "actual_ii": actual_ii,
                "status": "CACHED",
                "elapsed_seconds": 0.0,
                "dfg_hash": sha256(pair_out / "dfg.json"),
                "mrrg_hash": sha256(pair_out / "mrrg.json"),
                "manifest_hash": sha256(pair_out / "problem_manifest.json"),
                "native_dfg_hash": sha256(dfg),
                "native_architecture_ref": arch_ref,
                "native_dfg_nodes": len(dfg_doc.get("nodes", [])),
                "mrrg_resources": len(mrrg_doc.get("resources", [])),
                "mrrg_edges": len(mrrg_doc.get("edges", [])),
                "stdout": "resumed cached export",
            }
    pair_out.parent.mkdir(parents=True, exist_ok=True)
    container_dfg = f"/work/dfg/{dfg.name}"
    container_out = f"/out/{kernel}/{architecture}/ii{ii}"
    command = [
        "docker",
        "run",
        "--rm",
        "--network=none",
        "-v",
        f"{stage}:/work:ro",
        "-v",
        f"{output}:/out",
        image,
        "/home/user/morpher/Morpher_CGRA_Mapper/build/src/cgra_xml_mapper",
        "-d",
        container_dfg,
        "-x",
        str(x),
        "-y",
        str(y),
        "-j",
        f"/home/user/morpher/Morpher_CGRA_Mapper/{arch_ref}",
        "-i",
        str(ii),
        "-t",
        pe_type,
        "-m",
        "0",
        "--dump-flowadvantage-problem",
        container_out,
    ]
    started = time.monotonic()
    completed = subprocess.run(
        command, cwd=ROOT, text=True, capture_output=True, check=False
    )
    elapsed = time.monotonic() - started
    if completed.returncode != 0:
        raise RuntimeError(
            f"native export failed {kernel}/{architecture}/ii{ii} "
            f"rc={completed.returncode}: {completed.stdout[-1000:]} "
            f"{completed.stderr[-1000:]}"
        )
    manifest = pair_out / "problem_manifest.json"
    dfg_json = pair_out / "dfg.json"
    mrrg_json = pair_out / "mrrg.json"
    if not all(path.is_file() for path in (manifest, dfg_json, mrrg_json)):
        raise RuntimeError(f"incomplete native export: {pair_out}")
    manifest_doc = json.loads(manifest.read_text())
    dfg_doc = json.loads(dfg_json.read_text())
    mrrg_doc = json.loads(mrrg_json.read_text())
    actual_ii = int(manifest_doc.get("ii", -1))
    if actual_ii <= 0 or dfg_doc.get("ii") != actual_ii or mrrg_doc.get("ii") != actual_ii:
        raise RuntimeError(f"II mismatch in native export: {pair_out}")
    if manifest_doc.get("dfg_hash") != dfg_doc.get("dfg_hash"):
        raise RuntimeError(f"DFG hash mismatch in native export: {pair_out}")
    if manifest_doc.get("architecture_hash") != mrrg_doc.get("architecture_hash"):
        raise RuntimeError(f"architecture hash mismatch in native export: {pair_out}")
    return {
        "kernel": kernel,
        "architecture": architecture,
        "ii": ii,
        "actual_ii": actual_ii,
        "status": "DONE",
        "elapsed_seconds": elapsed,
        "dfg_hash": sha256(dfg_json),
        "mrrg_hash": sha256(mrrg_json),
        "manifest_hash": sha256(manifest),
        "native_dfg_hash": sha256(dfg),
        "native_architecture_ref": arch_ref,
        "native_dfg_nodes": len(dfg_doc.get("nodes", [])),
        "mrrg_resources": len(mrrg_doc.get("resources", [])),
        "mrrg_edges": len(mrrg_doc.get("edges", [])),
        "stdout": completed.stdout[-2000:],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "results/hail_mary_hpca1/problem_exports")
    parser.add_argument("--image", default=IMAGE)
    parser.add_argument("--delta-max", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--kernels", default=",".join(KERNELS))
    parser.add_argument("--architectures", default=",".join(ARCHES))
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise SystemExit(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    stage = output / "_native_inputs"
    (stage / "dfg").mkdir(parents=True, exist_ok=True)
    for kernel, path in KERNELS.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        shutil.copy2(path, stage / "dfg" / f"{kernel}.xml")
    selected_kernels = tuple(
        value.strip() for value in args.kernels.split(",") if value.strip()
    )
    selected_architectures = tuple(
        value.strip() for value in args.architectures.split(",") if value.strip()
    )
    unknown_kernels = set(selected_kernels) - set(KERNELS)
    unknown_architectures = set(selected_architectures) - set(ARCHES)
    if unknown_kernels or unknown_architectures:
        raise SystemExit(
            f"unknown export selection kernels={sorted(unknown_kernels)} "
            f"architectures={sorted(unknown_architectures)}"
        )

    def export_pair(item: tuple[str, str]) -> list[dict[str, object]]:
        kernel, architecture = item
        source_dfg = KERNELS[kernel]
        arch_ref, x, y, pe_type = ARCHES[architecture]
        pair_rows: list[dict[str, object]] = []
        # Each pair is sequential because the i=0 export determines the
        # native resource/recurrence lower bound.  Independent pairs run in
        # two isolated Docker containers to use the host's available cores.
        
        for requested in (0,):
            lb_row = run_one(args.image, stage, output, kernel, architecture, stage / "dfg" / f"{kernel}.xml", arch_ref, x, y, pe_type, requested)
            lb_manifest = json.loads((output / kernel / architecture / "ii0" / "problem_manifest.json").read_text())
            lower_bound = int(lb_manifest["ii"])
            lb_row["requested_ii"] = 0
            lb_row["lower_bound_ii"] = lower_bound
            pair_rows.append(lb_row)
        lower_bound = int(pair_rows[0]["lower_bound_ii"])
        for delta in range(1, args.delta_max + 1):
            target = lower_bound + delta
            row = run_one(args.image, stage, output, kernel, architecture, stage / "dfg" / f"{kernel}.xml", arch_ref, x, y, pe_type, target)
            row["requested_ii"] = target
            row["lower_bound_ii"] = lower_bound
            pair_rows.append(row)
        return pair_rows

    rows: list[dict[str, object]] = []
    pairs = [
        (kernel, architecture)
        for kernel in selected_kernels
        for architecture in selected_architectures
    ]
    if args.workers <= 0:
        raise SystemExit("--workers must be positive")
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(export_pair, pair): pair for pair in pairs}
        for future in as_completed(futures):
            rows.extend(future.result())
    rows.sort(key=lambda row: (str(row["kernel"]), str(row["architecture"]), int(row["requested_ii"])))
    manifest_path = output / "problem_export_manifest.csv"
    fields = sorted({key for row in rows for key in row})
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (output / "export_metadata.json").write_text(json.dumps({
        "schema": "flowadvantage_hail_mary_problem_grid_v1",
        "image": args.image,
        "image_digest": subprocess.check_output(["docker", "image", "inspect", args.image, "--format", "{{.Id}}"], text=True).strip(),
        "kernels": list(KERNELS),
        "architectures": list(ARCHES),
        "delta_max": args.delta_max,
        "rows": len(rows),
    }, indent=2) + "\n")
    print(json.dumps({"output": str(output), "rows": len(rows), "manifest": str(manifest_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
