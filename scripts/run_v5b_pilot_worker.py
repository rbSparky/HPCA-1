#!/usr/bin/env python3
"""Execute exactly one revision-v5b real-pilot work item.

The queue owns the hard wall timeout and process-tree termination.  This
worker owns native/FlowAdvantage execution, progress heartbeats, schema
validation, and atomic publication of a single result.
"""

from __future__ import annotations

import argparse
import faulthandler
import json
import os
import re
import signal
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

# These limits must be set before importing scientific libraries.
for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
    "TORCH_NUM_THREADS",
):
    os.environ[_name] = "1"

import psutil

from scripts.v5b_pilot_queue_common import (
    atomic_json,
    config_hash,
    fsync_directory,
    sha256_file,
    source_tree_hash,
)

ROOT = Path(__file__).resolve().parents[1]
PATHFINDER_METHODS = frozenset({"native_pathfinder", "pathfinder"})
SUPPORTED_EXECUTORS = PATHFINDER_METHODS | frozenset({"length"})
DECLARED_BUT_UNWIRED = frozenset(
    {
        "dual",
        "dual_linear",
        "proposal",
        "flow_proposal",
        "top4",
        "flow_top4",
        "full",
        "full_relaxed_lookahead",
    }
)


def _resolve_mapper(spec: dict[str, Any]) -> Path:
    if spec.get("mapper_binary"):
        return Path(spec["mapper_binary"]).resolve()
    toolchain = Path(spec["toolchain_path"]).resolve()
    candidates = (
        toolchain / "bin/cgra_xml_mapper",
        toolchain / "source/Morpher_CGRA_Mapper/build/src/cgra_xml_mapper",
    )
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    raise FileNotFoundError(
        "no executable cgra_xml_mapper in immutable toolchain prefix "
        f"{toolchain}"
    )


def _preflight(spec: dict[str, Any]) -> tuple[Path | None, Path, Path]:
    method = str(spec["method"])
    if method in DECLARED_BUT_UNWIRED:
        raise NotImplementedError(
            f"{method} is declared but has no paper-grade native scorer/evaluator "
            "binding; refusing to synthesize proxy results"
        )
    if method not in SUPPORTED_EXECUTORS:
        raise ValueError(f"unknown pilot method {method!r}")
    dfg = Path(spec["dfg_path"]).resolve()
    architecture = Path(spec["architecture_path"]).resolve()
    for kind, path in (("DFG", dfg), ("architecture", architecture)):
        if not path.is_file():
            raise FileNotFoundError(f"{kind} input does not exist: {path}")
    if sha256_file(dfg) != spec["dfg_hash"]:
        raise ValueError("DFG hash changed after manifest freeze")
    if sha256_file(architecture) != spec["architecture_hash"]:
        raise ValueError("architecture hash changed after manifest freeze")
    mapper = _resolve_mapper(spec) if method in PATHFINDER_METHODS else None
    if mapper is not None and sha256_file(mapper) != spec["toolchain_hash"]:
        raise ValueError("native mapper binary hash changed after manifest freeze")
    if config_hash(spec) != spec["config_hash"]:
        raise ValueError("semantic work-item configuration hash mismatch")
    if source_tree_hash(ROOT) != spec["source_tree_hash"]:
        raise ValueError("pilot source tree changed after manifest freeze")
    return mapper, dfg, architecture


def _process_totals(process: psutil.Process) -> tuple[float, float]:
    cpu_seconds = 0.0
    rss_bytes = 0
    processes = [process]
    try:
        processes.extend(process.children(recursive=True))
    except psutil.Error:
        pass
    for value in processes:
        try:
            times = value.cpu_times()
            cpu_seconds += times.user + times.system
            rss_bytes += value.memory_info().rss
        except psutil.Error:
            pass
    return cpu_seconds, rss_bytes / 1024**2


def _native_pathfinder(
    spec: dict[str, Any],
    mapper: Path,
    dfg: Path,
    architecture: Path,
    artifact_directory: Path,
    progress: dict[str, Any],
) -> dict[str, Any]:
    scratch = artifact_directory.with_name(f".{artifact_directory.name}.native.tmp")
    if scratch.exists():
        raise FileExistsError(f"stale native scratch directory exists: {scratch}")
    scratch.mkdir(parents=True)
    export = scratch / "export"
    export.mkdir()
    command = [
        str(mapper),
        "-d",
        str(dfg),
        "-x",
        str(int(spec["x"])),
        "-y",
        str(int(spec["y"])),
        "-j",
        str(architecture),
        "-i",
        str(int(spec["initial_ii"])),
        "-t",
        str(spec["pe_type"]),
        "-m",
        str(spec.get("native_method", 0)),
        "--dump-flowadvantage-state",
        str(export),
    ]
    progress["stage"] = "native_pathfinder"
    started = time.monotonic()
    stdout_path = scratch / "native.stdout.log"
    stderr_path = scratch / "native.stderr.log"
    attempts: list[dict[str, Any]] = []
    success_ii: int | None = None
    current_ii: int | None = None
    with stdout_path.open("w", encoding="utf-8", buffering=1) as stdout_handle, (
        stderr_path.open("w", encoding="utf-8", buffering=1)
    ) as stderr_handle:
        process = subprocess.Popen(
            command,
            cwd=scratch,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=stderr_handle,
            text=True,
            bufsize=1,
            start_new_session=False,
        )
        assert process.stdout is not None
        for line in process.stdout:
            stdout_handle.write(line)
            print(line, end="", flush=True)
            match = re.search(r"Current target II =\s*(\d+)", line)
            if match:
                candidate_ii = int(match.group(1))
                if candidate_ii != current_ii:
                    current_ii = candidate_ii
                    attempts.append(
                        {
                            "ii": candidate_ii,
                            "status": "ATTEMPTED",
                            "started_offset_seconds": time.monotonic() - started,
                        }
                    )
                progress["ii"] = current_ii
            match = re.search(r"Map Success with II =\s*(\d+)", line)
            if match:
                success_ii = int(match.group(1))
                if attempts:
                    attempts[-1]["status"] = "MAPPED"
        return_code = process.wait()
        stdout_handle.flush()
        os.fsync(stdout_handle.fileno())
        stderr_handle.flush()
        os.fsync(stderr_handle.fileno())
    if return_code != 0:
        raise RuntimeError(f"native PathFinder exited with code {return_code}")
    if success_ii is None:
        # A normal native exit with explicit II attempts but no export is a
        # valid search failure.  It is not an infrastructure exception.
        if attempts and not any(export.glob("*.json")):
            return {
                "status": "VALID_MAPPING_FAILURE",
                "legal": False,
                "success": False,
                "termination": "NATIVE_SEARCH_EXHAUSTED",
                "ii_attempts": attempts,
                "compile_wall_seconds": time.monotonic() - started,
                "native_mapper_seconds": time.monotonic() - started,
                "command": command,
            }
        raise RuntimeError(
            "native mapper exited zero without a success marker or a valid "
            "search-exhaustion record"
        )
    documents: dict[str, dict[str, Any]] = {}
    hashes: dict[str, str] = {}
    for name in ("dfg.json", "mrrg.json", "mapping.json"):
        path = export / name
        if not path.is_file():
            raise RuntimeError(f"native success omitted required export {name}")
        documents[name] = json.loads(path.read_text(encoding="utf-8"))
        hashes[name] = sha256_file(path)
    mapping = documents["mapping.json"]
    if int(mapping["ii"]) != success_ii:
        raise RuntimeError("native success II does not match exported mapping II")
    progress["stage"] = "independent_legality"
    from flowadvantage.morpher_adapter.legality_bridge import validate_mapping

    legality = validate_mapping(
        mapping, documents["dfg.json"], documents["mrrg.json"]
    )
    if not legality["legal"]:
        raise RuntimeError(
            "native PathFinder export failed independent legality: "
            + json.dumps(legality["violations"][:5], sort_keys=True)
        )
    for path in (stdout_path, stderr_path, *(export / value for value in documents)):
        if path.is_file():
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
    os.replace(scratch, artifact_directory)
    fsync_directory(artifact_directory.parent)
    elapsed = time.monotonic() - started
    return {
        "status": "DONE",
        "legal": True,
        "success": True,
        "termination": "DONE",
        "ii": success_ii,
        "ii_attempts": attempts,
        "operation_count": len(mapping.get("operations", [])),
        "route_count": len(mapping.get("routes", [])),
        "route_cost": sum(
            max(0, len(route.get("ordered_resource_ids", [])) - 1)
            for route in mapping.get("routes", [])
        ),
        "compile_wall_seconds": elapsed,
        "native_mapper_seconds": elapsed,
        "artifact_directory": str(artifact_directory),
        "document_hashes": hashes,
        "command": command,
    }


def _native_length(
    spec: dict[str, Any],
    dfg_path: Path,
    architecture_path: Path,
    artifact_directory: Path,
    progress: dict[str, Any],
) -> dict[str, Any]:
    # For native Python search, dfg_path and architecture_path are canonical
    # JSON documents and a sibling mapping.json is the latency witness only.
    dfg = json.loads(dfg_path.read_text(encoding="utf-8"))
    mrrg = json.loads(architecture_path.read_text(encoding="utf-8"))
    witness_path = Path(spec.get("reference_mapping_path", ""))
    if not witness_path.is_file():
        raise FileNotFoundError(
            "length search requires an explicit native reference_mapping_path "
            "for FU latency witnesses"
        )
    if sha256_file(witness_path) != spec.get("reference_mapping_hash"):
        raise ValueError("native reference-mapping hash changed after freeze")
    witness = json.loads(witness_path.read_text(encoding="utf-8"))
    from flowadvantage.morpher_adapter.native_beam_mapper import (
        DeterministicNativeBeamMapper,
        LengthActionScorer,
        NativeBeamConfig,
    )
    from flowadvantage.morpher_adapter.native_mapper import NativeMorpherProblem

    problem = NativeMorpherProblem(dfg, mrrg, reference_mapping=witness)

    def callback(event: Any) -> None:
        progress.update(
            {
                "stage": event.stage,
                "mapped_operations": event.depth,
                "expansions": event.expansions,
                "candidate_actions": event.generated_actions,
                "routing_attempts": event.routed_actions + event.failed_targets,
            }
        )

    mapper = DeterministicNativeBeamMapper(
        problem,
        scorer=LengthActionScorer(),
        config=NativeBeamConfig(
            beam_width=int(spec.get("beam_width") or 4),
            k_paths=int(spec.get("k_paths") or 4),
            per_state_action_limit=(
                int(spec["action_limit"]) if spec.get("action_limit") else None
            ),
            max_expansions=(
                int(spec["max_expansions"]) if spec.get("max_expansions") else None
            ),
        ),
        progress_callback=callback,
    )
    started = time.monotonic()
    result = mapper.map()
    elapsed = time.monotonic() - started
    payload = {
        "status": "DONE" if result.metrics.success else "VALID_MAPPING_FAILURE",
        "legal": result.metrics.legal,
        "success": result.metrics.success,
        "termination": result.metrics.termination,
        "ii": problem.ii,
        "ii_attempts": [
            {
                "ii": problem.ii,
                "status": "MAPPED" if result.metrics.success else "FAILED",
                "started_offset_seconds": 0.0,
            }
        ],
        "mapped_operations": result.metrics.mapped_operations,
        "operation_count": result.metrics.total_operations,
        "route_cost": result.metrics.route_cost,
        "expansions": result.metrics.expansions,
        "generated_actions": result.metrics.generated_actions,
        "routing_attempts": (
            result.metrics.routed_actions + result.metrics.failed_targets
        ),
        "compile_wall_seconds": elapsed,
        "feature_seconds": 0.0,
        "proposal_seconds": 0.0,
        "relaxation_seconds": 0.0,
    }
    if result.mapping is not None:
        artifact_directory.mkdir(parents=True)
        atomic_json(artifact_directory / "mapping.json", result.mapping)
        payload["artifact_directory"] = str(artifact_directory)
        payload["mapping_hash"] = sha256_file(artifact_directory / "mapping.json")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    arguments = parser.parse_args()
    spec = json.loads(arguments.spec.read_text(encoding="utf-8"))
    heartbeat_path = Path(spec["heartbeat_path"])
    result_path = Path(spec["result_path"])
    artifact_directory = Path(spec["artifact_directory"])
    started_wall = time.time()
    started_monotonic = time.monotonic()
    own_process = psutil.Process()
    progress: dict[str, Any] = {
        "stage": "preflight",
        "mapped_operations": 0,
        "expansions": 0,
        "candidate_actions": 0,
        "routing_attempts": 0,
        "parent_solves": 0,
        "child_solves": 0,
        "cache_hits": 0,
        "cache_misses": 0,
        "ii": None,
    }
    stop_heartbeat = threading.Event()

    def heartbeat() -> None:
        while not stop_heartbeat.is_set():
            cpu_seconds, rss_mb = _process_totals(own_process)
            atomic_json(
                heartbeat_path,
                {
                    **progress,
                    "schema": "flowadvantage_v5b_heartbeat_v1",
                    "work_id": spec["work_id"],
                    "pid": os.getpid(),
                    "timestamp": time.time(),
                    "wall_seconds": time.monotonic() - started_monotonic,
                    "cpu_seconds": cpu_seconds,
                    "rss_mb": rss_mb,
                },
            )
            stop_heartbeat.wait(15.0)

    faulthandler.enable()
    faulthandler.register(signal.SIGUSR1, all_threads=True)
    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        mapper, dfg, architecture = _preflight(spec)
        if result_path.exists():
            raise FileExistsError(f"refusing to overwrite atomic result {result_path}")
        if artifact_directory.exists():
            raise FileExistsError(
                f"refusing to overwrite artifact directory {artifact_directory}"
            )
        if spec["method"] in PATHFINDER_METHODS:
            assert mapper is not None
            row = _native_pathfinder(
                spec, mapper, dfg, architecture, artifact_directory, progress
            )
        else:
            row = _native_length(
                spec, dfg, architecture, artifact_directory, progress
            )
        row.update(
            {
                "schema": "flowadvantage_v5b_pilot_result_v1",
                "work_id": spec["work_id"],
                "kernel": spec["kernel"],
                "architecture": spec["architecture"],
                "method": spec["method"],
                "seed": int(spec["seed"]),
                "config_hash": spec["config_hash"],
                "source_commit": spec["source_commit"],
                "toolchain_hash": spec["toolchain_hash"],
                "dfg_hash": spec["dfg_hash"],
                "architecture_hash": spec["architecture_hash"],
                "checkpoint_hash": spec.get("checkpoint_hash", ""),
                "started_at": started_wall,
                "finished_at": time.time(),
            }
        )
        if row["status"] not in {"DONE", "VALID_MAPPING_FAILURE"}:
            raise RuntimeError(f"executor returned invalid status {row['status']}")
        atomic_json(result_path, row)
        return 0
    except NotImplementedError as error:
        atomic_json(
            result_path,
            {
                "schema": "flowadvantage_v5b_pilot_result_v1",
                "work_id": spec["work_id"],
                "status": "UNSUPPORTED",
                "error_type": type(error).__name__,
                "error_message": str(error),
                "config_hash": spec.get("config_hash"),
            },
        )
        return 3
    except BaseException as error:
        atomic_json(
            result_path,
            {
                "schema": "flowadvantage_v5b_pilot_result_v1",
                "work_id": spec.get("work_id"),
                "status": "ERROR",
                "error_type": type(error).__name__,
                "error_message": str(error),
                "config_hash": spec.get("config_hash"),
            },
        )
        raise
    finally:
        progress["stage"] = "terminal"
        stop_heartbeat.set()
        thread.join(timeout=2.0)


if __name__ == "__main__":
    raise SystemExit(main())
