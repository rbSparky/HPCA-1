"""Shared durable-state primitives for the revision-v5b real pilot.

This module intentionally uses only the Python standard library.  In
particular, it can be imported before NumPy, PyTorch, CVXPY, or a native solver
is loaded by a spawned worker.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import socket
import tempfile
from pathlib import Path
from typing import Any, Iterable

TERMINAL_STATUSES = frozenset(
    {
        "DONE",
        "VALID_MAPPING_FAILURE",
        "TIMEOUT",
        "ERROR",
        "UNSUPPORTED",
    }
)
RUNNABLE_STATUSES = frozenset({"PENDING", "RETRYABLE"})
ALL_STATUSES = frozenset(
    {
        "PENDING",
        "RUNNING",
        "DONE",
        "VALID_MAPPING_FAILURE",
        "TIMEOUT",
        "ERROR",
        "RETRYABLE",
        "UNSUPPORTED",
    }
)

MANIFEST_FIELDS = (
    "work_id",
    "kernel",
    "architecture",
    "method",
    "seed",
    "budget_seconds",
    "status",
    "attempt",
    "worker_pid",
    "worker_pid_create_time",
    "worker_host",
    "start_time",
    "last_heartbeat",
    "end_time",
    "wall_seconds",
    "cpu_seconds",
    "peak_rss_mb",
    "exit_code",
    "timeout_seconds",
    "result_path",
    "heartbeat_path",
    "stdout_path",
    "stderr_path",
    "error_type",
    "error_message",
    "toolchain_path",
    "mapper_binary",
    "dfg_path",
    "architecture_path",
    "native_dfg_path",
    "native_architecture_path",
    "reference_mapping_path",
    "x",
    "y",
    "initial_ii",
    "ii_delta_max",
    "max_ii",
    "pe_type",
    "native_method",
    "native_max_iter",
    "evaluation_mode",
    "full_end_to_end",
    "initialization_policy",
    "initialization_depth_fraction",
    "beam_width",
    "k_paths",
    "action_limit",
    "max_expansions",
    "config_hash",
    "source_commit",
    "source_tree_hash",
    "toolchain_hash",
    "dfg_hash",
    "architecture_hash",
    "native_dfg_hash",
    "native_architecture_hash",
    "paper_strict_legality",
    "reference_mapping_hash",
    "checkpoint_hash",
    "checkpoint_path",
    "anchor_operations",
    "relaxation_cache_dir",
    "relaxation_tau",
    "relaxation_extra_ii_periods",
    "relaxation_fallback_extra_ii_periods",
    "relaxation_timeout",
    "reachable_edge_pruning",
    "device",
)

SEMANTIC_FIELDS = (
    "kernel",
    "architecture",
    "method",
    "seed",
    "budget_seconds",
    "timeout_seconds",
    "toolchain_path",
    "mapper_binary",
    "dfg_path",
    "architecture_path",
    "native_dfg_path",
    "native_architecture_path",
    "reference_mapping_path",
    "x",
    "y",
    "initial_ii",
    "ii_delta_max",
    "max_ii",
    "pe_type",
    "native_method",
    "native_max_iter",
    "evaluation_mode",
    "full_end_to_end",
    "initialization_policy",
    "initialization_depth_fraction",
    "beam_width",
    "k_paths",
    "action_limit",
    "max_expansions",
    "source_commit",
    "source_tree_hash",
    "toolchain_hash",
    "dfg_hash",
    "architecture_hash",
    "native_dfg_hash",
    "native_architecture_hash",
    "paper_strict_legality",
    "reference_mapping_hash",
    "checkpoint_hash",
    "checkpoint_path",
    "anchor_operations",
    "relaxation_cache_dir",
    "relaxation_tau",
    "relaxation_extra_ii_periods",
    "relaxation_fallback_extra_ii_periods",
    "relaxation_timeout",
    "reachable_edge_pruning",
    "device",
)

_SEMANTIC_INTEGER_FIELDS = frozenset(
    {
        "seed",
        "budget_seconds",
        "timeout_seconds",
        "x",
        "y",
        "initial_ii",
        "ii_delta_max",
        "max_ii",
        "native_method",
        "native_max_iter",
        "beam_width",
        "k_paths",
        "action_limit",
        "max_expansions",
        "anchor_operations",
    }
)
_SEMANTIC_FLOAT_FIELDS = frozenset(
    {
        "initialization_depth_fraction",
        "relaxation_tau",
        "relaxation_extra_ii_periods",
        "relaxation_fallback_extra_ii_periods",
        "relaxation_timeout",
    }
)
_SEMANTIC_BOOLEAN_FIELDS = frozenset(
    {"full_end_to_end", "reachable_edge_pruning", "paper_strict_legality"}
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_tree_hash(root: Path) -> str:
    """Hash every Python source that can affect native pilot semantics."""

    paths = sorted((root / "flowadvantage").rglob("*.py"))
    paths.extend(
        root / "scripts" / name
        for name in (
            "run_v5b_pilot_worker.py",
            "run_v5b_pilot_queue.py",
            "recover_v5b_pilot_queue.py",
            "v5b_pilot_queue_common.py",
        )
    )
    digest = hashlib.sha256()
    for path in sorted(set(paths)):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        payload = path.read_bytes()
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _semantic_value(field: str, value: Any) -> Any:
    """Canonicalize typed config values across durable CSV round trips."""

    if value is None or (isinstance(value, str) and not value.strip()):
        return ""
    if field in _SEMANTIC_INTEGER_FIELDS:
        return int(value)
    if field in _SEMANTIC_FLOAT_FIELDS:
        return float(value)
    if field in _SEMANTIC_BOOLEAN_FIELDS:
        if isinstance(value, bool):
            return value
        normalized = str(value).strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
        raise ValueError(f"invalid semantic Boolean {field}={value!r}")
    return str(value)


def semantic_config(row: dict[str, Any]) -> dict[str, Any]:
    return {
        field: _semantic_value(field, row.get(field, ""))
        for field in SEMANTIC_FIELDS
    }


def config_hash(row: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(semantic_config(row))).hexdigest()


def work_id(row: dict[str, Any]) -> str:
    prefix = "__".join(
        str(row[field]).replace("/", "_")
        for field in ("kernel", "architecture", "method", "seed")
    )
    return f"{prefix}__{config_hash(row)[:16]}"


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        fsync_directory(path.parent)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def atomic_text(path: Path, value: str) -> None:
    """Durably publish UTF-8 text without exposing a partial file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        fsync_directory(path.parent)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def atomic_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=MANIFEST_FIELDS, extrasaction="ignore"
            )
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row.get(field, "") for field in MANIFEST_FIELDS})
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        fsync_directory(path.parent)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def read_manifest(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        if row.get("status") not in ALL_STATUSES:
            raise ValueError(
                f"manifest {path} has invalid status {row.get('status')!r}"
            )
    ids = [row["work_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"manifest {path} contains duplicate work IDs")
    return rows


def balanced_order(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Round-robin work over kernel/architecture pairs.

    Pair order uses a deterministic Latin-square traversal so an early tranche
    covers kernel and architecture axes instead of a family-sorted prefix.
    Methods and seeds are then consumed one at a time per pair.
    """

    rows = list(rows)
    kernels = sorted({str(row["kernel"]) for row in rows})
    architectures = sorted({str(row["architecture"]) for row in rows})
    pair_order: list[tuple[str, str]] = []
    for offset in range(max(1, len(architectures))):
        for kernel_index, kernel in enumerate(kernels):
            if architectures:
                architecture = architectures[
                    (kernel_index + offset) % len(architectures)
                ]
                pair = (kernel, architecture)
                if pair not in pair_order:
                    pair_order.append(pair)
    for pair in sorted(
        {(str(row["kernel"]), str(row["architecture"])) for row in rows}
    ):
        if pair not in pair_order:
            pair_order.append(pair)
    queues = {
        pair: sorted(
            (
                row
                for row in rows
                if (str(row["kernel"]), str(row["architecture"])) == pair
            ),
            key=lambda row: (
                str(row["method"]),
                int(row.get("seed") or 0),
                str(row["work_id"]),
            ),
        )
        for pair in pair_order
    }
    ordered: list[dict[str, Any]] = []
    while any(queues.values()):
        for pair in pair_order:
            if queues[pair]:
                ordered.append(queues[pair].pop(0))
    return ordered


def host_name() -> str:
    return socket.getfqdn()
