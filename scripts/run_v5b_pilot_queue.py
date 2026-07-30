#!/usr/bin/env python3
"""Atomic, resumable queue for revision-v5b real Morpher work items."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

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
    MANIFEST_FIELDS,
    RUNNABLE_STATUSES,
    TERMINAL_STATUSES,
    atomic_csv,
    atomic_json,
    balanced_order,
    config_hash,
    host_name,
    read_manifest,
    sha256_file,
    source_tree_hash,
    work_id,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results/revision_v5b/pilot_queue"
THREAD_ENVIRONMENTS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
    "TORCH_NUM_THREADS",
)


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        supplied = os.environ.get("FLOWADVANTAGE_SOURCE_COMMIT", "")
        if len(supplied) == 40 and all(
            character in "0123456789abcdef" for character in supplied.lower()
        ):
            return supplied.lower()
        raise RuntimeError(
            "source checkout has no Git metadata; set "
            "FLOWADVANTAGE_SOURCE_COMMIT to the exact 40-hex source commit "
            "(the source_tree_hash still records working-tree contents)"
        )


def _mapper_path(job: dict[str, Any]) -> Path:
    if job.get("mapper_binary"):
        return Path(job["mapper_binary"]).expanduser().resolve()
    toolchain = Path(job["toolchain_path"]).expanduser().resolve()
    candidates = (
        toolchain / "bin/cgra_xml_mapper",
        toolchain / "source/Morpher_CGRA_Mapper/build/src/cgra_xml_mapper",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def freeze_job(job: dict[str, Any], output: Path) -> dict[str, Any]:
    row: dict[str, Any] = {field: "" for field in MANIFEST_FIELDS}
    row.update(job)
    required = {
        "kernel",
        "architecture",
        "method",
        "seed",
        "timeout_seconds",
        "dfg_path",
        "architecture_path",
        "x",
        "y",
        "initial_ii",
        "pe_type",
    }
    missing = sorted(field for field in required if field not in row)
    if missing:
        raise ValueError(f"job is missing required fields: {', '.join(missing)}")
    dfg = Path(row["dfg_path"]).expanduser().resolve()
    architecture = Path(row["architecture_path"]).expanduser().resolve()
    if not dfg.is_file() or not architecture.is_file():
        raise FileNotFoundError(
            f"missing immutable inputs: dfg={dfg}, architecture={architecture}"
        )
    row["dfg_path"], row["architecture_path"] = str(dfg), str(architecture)
    row["dfg_hash"], row["architecture_hash"] = (
        sha256_file(dfg),
        sha256_file(architecture),
    )
    native_dfg_value = row.get("native_dfg_path")
    native_architecture_value = row.get("native_architecture_path")
    if bool(native_dfg_value) != bool(native_architecture_value):
        raise ValueError(
            "native_dfg_path and native_architecture_path must be supplied together"
        )
    if native_dfg_value:
        native_dfg = Path(native_dfg_value).expanduser().resolve()
        native_architecture = Path(native_architecture_value).expanduser().resolve()
        if not native_dfg.is_file() or not native_architecture.is_file():
            raise FileNotFoundError(
                "missing native reimport inputs: "
                f"dfg={native_dfg}, architecture={native_architecture}"
            )
        row["native_dfg_path"] = str(native_dfg)
        row["native_architecture_path"] = str(native_architecture)
        row["native_dfg_hash"] = sha256_file(native_dfg)
        row["native_architecture_hash"] = sha256_file(native_architecture)
    if row.get("reference_mapping_path"):
        witness = Path(row["reference_mapping_path"]).expanduser().resolve()
        if not witness.is_file():
            raise FileNotFoundError(f"reference mapping does not exist: {witness}")
        row["reference_mapping_path"] = str(witness)
        row["reference_mapping_hash"] = sha256_file(witness)
    row["source_commit"] = row.get("source_commit") or _git_commit()
    row["source_tree_hash"] = row.get("source_tree_hash") or source_tree_hash(ROOT)
    row["native_method"] = row.get("native_method", 0)
    row["checkpoint_hash"] = row.get("checkpoint_hash", "")
    if row.get("checkpoint_path"):
        checkpoint = Path(row["checkpoint_path"]).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")
        row["checkpoint_path"] = str(checkpoint)
        row["checkpoint_hash"] = sha256_file(checkpoint)
    if row.get("method") in {"flow_proposal", "flow_top4", "full_relaxed_lookahead"}:
        if not row.get("checkpoint_path"):
            raise ValueError("FlowAdvantage work items require checkpoint_path")
    if row.get("method") in {
        "length",
        "dual_linear",
        "flow_proposal",
        "flow_top4",
        "full_relaxed_lookahead",
    }:
        if row.get("initialization_policy") != "deterministic_length_prefix":
            raise ValueError(
                "Python native mapping requires the frozen "
                "deterministic_length_prefix initialization policy"
            )
        depth_fraction = float(row.get("initialization_depth_fraction", 0.0))
        if not 0.0 <= depth_fraction < 1.0:
            raise ValueError(
                "initialization_depth_fraction must be frozen in [0,1)"
            )
        if row.get("full_end_to_end") is not True:
            raise ValueError(
                "paper mapping jobs must explicitly declare full_end_to_end=true"
            )
        if row.get("anchor_operations"):
            raise ValueError(
                "witness anchor_operations is prohibited by the paper protocol"
            )
    row["reachable_edge_pruning"] = row.get("reachable_edge_pruning", True)
    native_methods = {
        "native_pathfinder",
        "pathfinder",
        "native_simulated_annealing",
        "simulated_annealing",
        "native_lisa",
        "lisa",
    }
    paper_flow_methods = {
        "length",
        "dual_linear",
        "flow_proposal",
        "flow_top4",
        "full_relaxed_lookahead",
    }
    if row["method"] in native_methods or native_dfg_value:
        mapper = _mapper_path(row)
        if not mapper.is_file():
            raise FileNotFoundError(f"native mapper binary not found: {mapper}")
        row["mapper_binary"] = str(mapper)
        row["toolchain_hash"] = sha256_file(mapper)
    else:
        row["toolchain_hash"] = row.get("toolchain_hash", "")
    if row["method"] in paper_flow_methods and row.get("paper_strict_legality"):
        if not native_dfg_value:
            raise ValueError(
                "paper_strict_legality Flow jobs require native DFG/architecture inputs"
            )
        if not row.get("mapper_binary"):
            raise ValueError(
                "paper_strict_legality Flow jobs require a native mapper binary"
            )
    for field, default in (
        ("beam_width", 4),
        ("k_paths", 4),
        ("action_limit", 24),
        ("max_expansions", 5000),
    ):
        row[field] = row.get(field) or default
    row["config_hash"] = config_hash(row)
    row["work_id"] = work_id(row)
    row.update(
        {
            "status": "PENDING",
            "attempt": 0,
            "result_path": str(output / "work_items" / f"{row['work_id']}.json"),
            "heartbeat_path": str(
                output / "heartbeats" / f"{row['work_id']}.json"
            ),
            "stdout_path": str(output / "logs" / f"{row['work_id']}.stdout.log"),
            "stderr_path": str(output / "logs" / f"{row['work_id']}.stderr.log"),
        }
    )
    return row


def initialize_manifest(
    jobs_path: Path, output: Path, manifest_path: Path
) -> list[dict[str, Any]]:
    payload = json.loads(jobs_path.read_text(encoding="utf-8"))
    jobs = payload["jobs"] if isinstance(payload, dict) else payload
    if not isinstance(jobs, list):
        raise TypeError("jobs document must be a list or {'jobs': [...]}")
    frozen = balanced_order(freeze_job(job, output) for job in jobs)
    existing = read_manifest(manifest_path)
    by_id = {row["work_id"]: row for row in existing}
    for row in frozen:
        previous = by_id.get(row["work_id"])
        if previous:
            if previous["config_hash"] != row["config_hash"]:
                raise ValueError(
                    f"immutable work item changed: {row['work_id']}"
                )
            continue
        existing.append(row)
    # Preserve execution state while applying deterministic order to new and
    # existing items alike.
    ordered = balanced_order(existing)
    atomic_csv(manifest_path, ordered)
    atomic_json(
        output / "manifest_metadata.json",
        {
            "schema": "flowadvantage_v5b_pilot_manifest_v1",
            "manifest": str(manifest_path),
            "job_source": str(jobs_path.resolve()),
            "job_source_sha256": sha256_file(jobs_path),
            "work_items": len(ordered),
            "source_commit": _git_commit(),
            "thread_limits": {name: "1" for name in THREAD_ENVIRONMENTS},
            "balanced_order": "kernel_architecture_latin_round_robin",
        },
    )
    return ordered


def _pid_matches(row: dict[str, Any]) -> bool:
    try:
        process = psutil.Process(int(row["worker_pid"]))
        return abs(
            process.create_time() - float(row["worker_pid_create_time"])
        ) < 0.01
    except (psutil.Error, ValueError, TypeError):
        return False


def recover(rows: list[dict[str, Any]], stale_seconds: float) -> int:
    now = time.time()
    recovered = 0
    for row in rows:
        if row["status"] != "RUNNING":
            continue
        heartbeat_path = Path(row.get("heartbeat_path") or "")
        if heartbeat_path.is_file():
            try:
                heartbeat = json.loads(
                    heartbeat_path.read_text(encoding="utf-8")
                )
                if heartbeat.get("work_id") == row["work_id"]:
                    row["last_heartbeat"] = heartbeat["timestamp"]
            except (OSError, ValueError, KeyError):
                pass
        heartbeat_age = now - float(row.get("last_heartbeat") or 0)
        if not _pid_matches(row) or heartbeat_age > stale_seconds:
            row["status"] = (
                "RETRYABLE" if int(row.get("attempt") or 0) < 2 else "ERROR"
            )
            row["error_type"] = "RecoveredStaleWorker"
            row["error_message"] = (
                "RUNNING worker absent or heartbeat stale; prior logs preserved"
            )
            row["end_time"] = now
            recovered += 1
    return recovered


def _kill_process_tree(process: subprocess.Popen[Any], stderr_path: Path) -> None:
    try:
        root = psutil.Process(process.pid)
        descendants = root.children(recursive=True)
    except psutil.NoSuchProcess:
        return
    try:
        os.kill(process.pid, signal.SIGUSR1)
    except (ProcessLookupError, PermissionError):
        pass
    time.sleep(1)
    with stderr_path.open("a", encoding="utf-8") as handle:
        handle.write(
            "\n[queue] hard timeout: requested traceback; terminating process tree\n"
        )
    for child in reversed(descendants):
        try:
            child.terminate()
        except psutil.Error:
            pass
    try:
        root.terminate()
    except psutil.Error:
        pass
    _, alive = psutil.wait_procs([*descendants, root], timeout=5)
    for process_value in alive:
        try:
            process_value.kill()
        except psutil.Error:
            pass


def _publish_operational_result(
    row: dict[str, Any], result_path: Path
) -> None:
    """Publish terminal infrastructure evidence when the child cannot do so."""

    if result_path.exists():
        return
    payload = {
        "schema": "flowadvantage_v5b_pilot_result_v1",
        "work_id": row["work_id"],
        "kernel": row["kernel"],
        "architecture": row["architecture"],
        "method": row["method"],
        "seed": int(row["seed"]),
        "status": row["status"],
        "success": None,
        "legal": False,
        "config_hash": row["config_hash"],
        "source_commit": row.get("source_commit", ""),
        "source_tree_hash": row.get("source_tree_hash", ""),
        "toolchain_hash": row.get("toolchain_hash", ""),
        "dfg_hash": row.get("dfg_hash", ""),
        "architecture_hash": row.get("architecture_hash", ""),
        "checkpoint_hash": row.get("checkpoint_hash", ""),
        "error_type": row.get("error_type", ""),
        "error_message": row.get("error_message", ""),
        "compile_wall_seconds": float(row.get("wall_seconds") or 0.0),
        "cpu_seconds": float(row.get("cpu_seconds") or 0.0),
        "peak_rss_mb": float(row.get("peak_rss_mb") or 0.0),
        "finished_at": float(row.get("end_time") or time.time()),
    }
    atomic_json(result_path, payload)


def _run_one(
    row: dict[str, Any],
    output: Path,
    state_callback: Any,
) -> dict[str, Any]:
    row = dict(row)
    row["attempt"] = int(row.get("attempt") or 0) + 1
    row["status"] = "RUNNING"
    row["start_time"] = row["last_heartbeat"] = time.time()
    row["worker_host"] = host_name()
    # Publish the claim before creating a child.  A queue crash can therefore
    # never leave an active child behind a manifest row that still says
    # PENDING.
    state_callback(dict(row))
    result_path = Path(row["result_path"])
    heartbeat_path = Path(row["heartbeat_path"])
    stdout_path = Path(row["stdout_path"])
    stderr_path = Path(row["stderr_path"])
    artifact_directory = (
        output / "artifacts" / row["work_id"] / f"attempt{row['attempt']}"
    )
    spec = {
        **row,
        "artifact_directory": str(artifact_directory),
    }
    spec_path = output / "specs" / f"{row['work_id']}.attempt{row['attempt']}.json"
    atomic_json(spec_path, spec)
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT)
    for name in THREAD_ENVIRONMENTS:
        environment[name] = "1"
    started_monotonic = time.monotonic()
    peak_rss = 0.0
    cpu_seconds = 0.0
    timed_out = False
    with stdout_path.open("a", encoding="utf-8") as stdout_handle, (
        stderr_path.open("a", encoding="utf-8")
    ) as stderr_handle:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "scripts.run_v5b_pilot_worker",
                "--spec",
                str(spec_path),
            ],
            cwd=ROOT,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=stdout_handle,
            stderr=stderr_handle,
            start_new_session=True,
        )
        row["worker_pid"] = process.pid
        row["worker_pid_create_time"] = psutil.Process(process.pid).create_time()
        state_callback(dict(row))
        deadline = started_monotonic + float(row["timeout_seconds"])
        while process.poll() is None:
            try:
                root_process = psutil.Process(process.pid)
                family = [root_process, *root_process.children(recursive=True)]
                peak_rss = max(
                    peak_rss,
                    sum(
                        value.memory_info().rss
                        for value in family
                        if value.is_running()
                    )
                    / 1024**2,
                )
                cpu_seconds = sum(
                    sum(value.cpu_times()[:2])
                    for value in family
                    if value.is_running()
                )
            except psutil.Error:
                pass
            if heartbeat_path.exists():
                try:
                    heartbeat = json.loads(
                        heartbeat_path.read_text(encoding="utf-8")
                    )
                    row["last_heartbeat"] = heartbeat["timestamp"]
                except (OSError, ValueError, KeyError):
                    pass
            if time.monotonic() >= deadline:
                timed_out = True
                _kill_process_tree(process, stderr_path)
                break
            time.sleep(1)
        return_code = process.poll()
    row["end_time"] = time.time()
    row["wall_seconds"] = time.monotonic() - started_monotonic
    row["cpu_seconds"] = cpu_seconds
    row["peak_rss_mb"] = peak_rss
    row["exit_code"] = return_code if return_code is not None else -signal.SIGKILL
    if timed_out:
        row["status"] = "TIMEOUT"
        row["error_type"] = "HardWallTimeout"
        row["error_message"] = (
            f"process tree terminated after {row['timeout_seconds']} seconds"
        )
        _publish_operational_result(row, result_path)
    elif result_path.is_file():
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        if payload.get("work_id") != row["work_id"]:
            row["status"] = "ERROR"
            row["error_type"] = "ResultIdentityMismatch"
            row["error_message"] = "atomic result belongs to a different work item"
        elif payload.get("config_hash") != row["config_hash"]:
            row["status"] = "ERROR"
            row["error_type"] = "ResultConfigurationMismatch"
            row["error_message"] = "atomic result configuration hash differs"
        elif payload.get("status") in TERMINAL_STATUSES:
            row["status"] = payload["status"]
            row["error_type"] = payload.get("error_type", "")
            row["error_message"] = payload.get("error_message", "")
        else:
            row["status"] = "ERROR"
            row["error_type"] = "InvalidAtomicResult"
            row["error_message"] = f"invalid result status {payload.get('status')!r}"
    else:
        row["status"] = "ERROR"
        row["error_type"] = "NoAtomicResult"
        row["error_message"] = "worker exited without publishing a result"
        _publish_operational_result(row, result_path)
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--jobs", type=Path)
    parser.add_argument("--initialize-only", action="store_true")
    parser.add_argument("--recover-only", action="store_true")
    parser.add_argument("--concurrency", type=int, choices=(1, 2), default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--method", action="append")
    arguments = parser.parse_args()
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "work_manifest.csv"
    rows = (
        initialize_manifest(arguments.jobs, output, manifest_path)
        if arguments.jobs
        else read_manifest(manifest_path)
    )
    if not rows:
        parser.error("no manifest rows; pass --jobs for first initialization")
    recovered_count = recover(rows, stale_seconds=35.0)
    atomic_csv(manifest_path, rows)
    if arguments.initialize_only or arguments.recover_only:
        print(
            json.dumps(
                {"work_items": len(rows), "recovered": recovered_count},
                sort_keys=True,
            )
        )
        return 0
    selected = [
        row
        for row in rows
        if row["status"] in RUNNABLE_STATUSES
        and int(row.get("attempt") or 0) < 2
        and (not arguments.method or row["method"] in arguments.method)
    ]
    if arguments.limit is not None:
        selected = selected[: arguments.limit]

    manifest_lock = threading.Lock()

    def record_state(updated: dict[str, Any]) -> None:
        with manifest_lock:
            index = next(
                index
                for index, row in enumerate(rows)
                if row["work_id"] == updated["work_id"]
            )
            rows[index] = updated
            atomic_csv(manifest_path, rows)

    def publish(updated: dict[str, Any]) -> None:
        record_state(updated)
        print(
            json.dumps(
                {
                    "work_id": updated["work_id"],
                    "status": updated["status"],
                    "wall_seconds": updated["wall_seconds"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    with ThreadPoolExecutor(max_workers=arguments.concurrency) as pool:
        futures = {
            pool.submit(_run_one, row, output, record_state): row["work_id"]
            for row in selected
        }
        for future in as_completed(futures):
            try:
                publish(future.result())
            except BaseException as error:
                work_item = next(
                    row for row in rows if row["work_id"] == futures[future]
                )
                work_item["status"] = "ERROR"
                work_item["error_type"] = type(error).__name__
                work_item["error_message"] = str(error)
                work_item["end_time"] = time.time()
                _publish_operational_result(
                    work_item, Path(work_item["result_path"])
                )
                publish(work_item)
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    print(json.dumps({"status_counts": counts}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
