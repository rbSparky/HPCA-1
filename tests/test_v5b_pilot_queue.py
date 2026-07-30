"""Reliability tests for atomic revision-v5b pilot execution."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import pytest

from scripts.run_v5b_pilot_queue import freeze_job, recover
from scripts.run_v5b_pilot_worker import (
    _native_search,
    _optional_bool,
    _optional_float,
    _prefix_depth,
)
from scripts.v5b_pilot_queue_common import (
    atomic_csv,
    atomic_json,
    balanced_order,
    config_hash,
    read_manifest,
)


def _job(tmp_path: Path, **changes):
    dfg = tmp_path / "dfg.xml"
    architecture = tmp_path / "architecture.json"
    mapper = tmp_path / "cgra_xml_mapper"
    dfg.write_text("<DFG />\n", encoding="utf-8")
    architecture.write_text("{}\n", encoding="utf-8")
    mapper.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    mapper.chmod(0o755)
    value = {
        "kernel": "kernel",
        "architecture": "architecture",
        "method": "native_pathfinder",
        "seed": 11,
        "timeout_seconds": 120,
        "mapper_binary": str(mapper),
        "dfg_path": str(dfg),
        "architecture_path": str(architecture),
        "x": 4,
        "y": 4,
        "initial_ii": 0,
        "pe_type": "HyCUBE_4REG",
    }
    value.update(changes)
    return value


def test_atomic_json_is_complete_and_leaves_no_temporary_file(tmp_path):
    path = tmp_path / "nested/result.json"
    atomic_json(path, {"work_id": "work", "value": [1, 2, 3]})
    assert json.loads(path.read_text())["value"] == [1, 2, 3]
    assert not list(path.parent.glob("*.tmp"))


def test_manifest_freezes_all_input_hashes(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "scripts.run_v5b_pilot_queue._git_commit", lambda: "a" * 40
    )
    row = freeze_job(_job(tmp_path), tmp_path / "output")
    assert len(row["dfg_hash"]) == 64
    assert len(row["architecture_hash"]) == 64
    assert len(row["toolchain_hash"]) == 64
    assert row["config_hash"] == config_hash(row)


def test_manifest_accepts_native_simulated_annealing_executor(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "scripts.run_v5b_pilot_queue._git_commit", lambda: "a" * 40
    )
    row = freeze_job(
        _job(
            tmp_path,
            method="native_simulated_annealing",
            native_method=1,
        ),
        tmp_path / "output",
    )
    assert row["method"] == "native_simulated_annealing"
    assert row["native_method"] == 1
    assert row["toolchain_hash"]
    assert row["work_id"].endswith(row["config_hash"][:16])


def test_semantic_change_produces_new_work_identity(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "scripts.run_v5b_pilot_queue._git_commit", lambda: "a" * 40
    )
    first = freeze_job(_job(tmp_path), tmp_path / "output")
    second = freeze_job(
        _job(tmp_path, initial_ii=5), tmp_path / "output"
    )
    assert first["work_id"] != second["work_id"]
    assert first["config_hash"] != second["config_hash"]


def test_semantic_hash_survives_csv_type_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "scripts.run_v5b_pilot_queue._git_commit", lambda: "a" * 40
    )
    row = freeze_job(
        _job(
            tmp_path,
            budget_seconds=3600,
            initialization_depth_fraction=0.4,
            full_end_to_end=True,
            reachable_edge_pruning=True,
        ),
        tmp_path / "output",
    )
    manifest = tmp_path / "manifest.csv"
    atomic_csv(manifest, [row])
    restored = read_manifest(manifest)[0]
    assert config_hash(restored) == row["config_hash"]


def test_balanced_order_covers_axes_in_early_tranche():
    rows = [
        {
            "kernel": kernel,
            "architecture": architecture,
            "method": "native_pathfinder",
            "seed": 11,
            "work_id": f"{kernel}-{architecture}",
        }
        for kernel in ("a", "b", "c")
        for architecture in ("x", "y")
    ]
    ordered = balanced_order(rows)
    assert {row["kernel"] for row in ordered[:3]} == {"a", "b", "c"}
    assert {row["architecture"] for row in ordered[:3]} == {"x", "y"}
    assert len({row["work_id"] for row in ordered}) == len(rows)


def test_recovery_never_changes_terminal_rows():
    rows = [
        {"status": "DONE", "attempt": "1"},
        {
            "status": "RUNNING",
            "attempt": "1",
            "worker_pid": "99999999",
            "worker_pid_create_time": "0",
            "last_heartbeat": "0",
        },
    ]
    assert recover(rows, stale_seconds=35) == 1
    assert rows[0]["status"] == "DONE"
    assert rows[1]["status"] == "RETRYABLE"


def test_recovery_reads_atomic_heartbeat_before_declaring_stale(
    tmp_path, monkeypatch
):
    import time

    heartbeat = tmp_path / "heartbeat.json"
    atomic_json(
        heartbeat,
        {
            "schema": "flowadvantage_v5b_heartbeat_v1",
            "work_id": "running",
            "timestamp": time.time(),
        },
    )
    rows = [
        {
            "work_id": "running",
            "status": "RUNNING",
            "attempt": "1",
            "worker_pid": "123",
            "worker_pid_create_time": "1",
            "last_heartbeat": "0",
            "heartbeat_path": str(heartbeat),
        }
    ]
    monkeypatch.setattr(
        "scripts.run_v5b_pilot_queue._pid_matches", lambda row: True
    )
    assert recover(rows, stale_seconds=35) == 0
    assert rows[0]["status"] == "RUNNING"
    assert float(rows[0]["last_heartbeat"]) > 0


def test_atomic_manifest_roundtrip(tmp_path):
    path = tmp_path / "manifest.csv"
    row = {field: "" for field in __import__(
        "scripts.v5b_pilot_queue_common", fromlist=["MANIFEST_FIELDS"]
    ).MANIFEST_FIELDS}
    row.update(
        {
            "work_id": "one",
            "status": "PENDING",
            "kernel": "array_add",
            "architecture": "A0",
            "method": "native_pathfinder",
            "seed": 11,
        }
    )
    atomic_csv(path, [row])
    loaded = read_manifest(path)
    assert loaded[0]["work_id"] == "one"
    assert not list(tmp_path.glob("*.tmp"))


def test_worker_manifest_defaults_survive_csv_empty_cells():
    assert _optional_float("", 0.001) == 0.001
    assert _optional_float(None, 120.0) == 120.0
    assert _optional_float("3.5", 1.0) == 3.5
    assert _optional_bool("", True) is True
    assert _optional_bool("False", True) is False
    assert _optional_bool("yes", False) is True
    with pytest.raises(ValueError, match="invalid boolean"):
        _optional_bool("maybe", True)


def test_zero_initialization_fraction_is_the_true_empty_root():
    assert _prefix_depth(25, 0.0) == 0
    assert _prefix_depth(1, 0.0) == 0


def test_positive_initialization_fraction_keeps_prior_rounding_contract():
    assert _prefix_depth(25, 0.20) == 5
    assert _prefix_depth(25, 0.01) == 1
    assert _prefix_depth(4, 0.99) == 3
    with pytest.raises(ValueError, match=r"\[0,1\)"):
        _prefix_depth(4, 1.0)


def test_native_search_zero_prefix_executes_without_hidden_initialization(
    tmp_path,
):
    root = Path(__file__).resolve().parents[1]
    corpus = (
        root
        / "results/revision_v5b/reference_mappings/array_add"
        / "A0_hycube4x4_standard_fixed17_smoke"
    )
    payload = _native_search(
        {
            "method": "length",
            "initialization_policy": "deterministic_length_prefix",
            "initialization_depth_fraction": 0.0,
            "beam_width": 1,
            "k_paths": 1,
            "action_limit": 1,
            "max_expansions": 1,
        },
        None,
        corpus / "dfg.json",
        corpus / "mrrg.json",
        tmp_path / "artifacts",
        {},
    )
    assert payload["initialization_mapped_operations"] == 0
    assert payload["initialization_expansions"] == 0
    assert payload["initialization_generated_actions"] == 0
    assert payload["initialization_routing_attempts"] == 0
    assert payload["initialization_frontier_size"] == 1
    assert not any(
        key.startswith("initialization_")
        for key in payload["stage_seconds"]
    )


def test_optional_reference_mapping_is_provenance_only(tmp_path):
    root = Path(__file__).resolve().parents[1]
    corpus = (
        root
        / "results/revision_v5b/reference_mappings/array_add"
        / "A0_hycube4x4_standard_fixed17_smoke"
    )
    # No reference_mapping_path is supplied.  The native DFG/MRRG contract is
    # sufficient to construct and execute the mapper.
    payload = _native_search(
        {
            "method": "length",
            "initialization_policy": "deterministic_length_prefix",
            "initialization_depth_fraction": 0.0,
            "beam_width": 1,
            "k_paths": 1,
            "action_limit": 1,
            "max_expansions": 1,
        },
        None,
        corpus / "dfg.json",
        corpus / "mrrg.json",
        tmp_path / "artifacts",
        {},
    )
    assert payload["operation_count"] > 0


def test_canonical_flow_methods_are_not_legacy_aliases():
    from scripts.run_v5b_pilot_worker import FLOW_METHODS, SUPPORTED_EXECUTORS

    assert {"flow_proposal", "flow_top4", "full_relaxed_lookahead"} <= FLOW_METHODS
    assert {"proposal", "top4", "full"}.isdisjoint(SUPPORTED_EXECUTORS)


@pytest.mark.parametrize(
    "method",
    ["dual", "proposal", "top4", "full", "noparent_proposal", "noparent_top4"],
)
def test_paper_method_aliases_are_strict_unwired_hooks(
    tmp_path, monkeypatch, method
):
    monkeypatch.setattr(
        "scripts.run_v5b_pilot_queue._git_commit", lambda: "a" * 40
    )
    row = freeze_job(_job(tmp_path, method=method), tmp_path / "output")
    spec = tmp_path / f"{method}.json"
    result = tmp_path / f"{method}.result.json"
    row.update(
        {
            "heartbeat_path": str(tmp_path / f"{method}.heartbeat.json"),
            "result_path": str(result),
            "artifact_directory": str(tmp_path / f"{method}.artifacts"),
        }
    )
    spec.write_text(json.dumps(row), encoding="utf-8")
    import subprocess
    import sys

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.run_v5b_pilot_worker",
            "--spec",
            str(spec),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
    )
    assert completed.returncode == 3
    assert json.loads(result.read_text())["status"] == "UNSUPPORTED"
