"""Reliability tests for atomic revision-v5b pilot execution."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import pytest

from scripts.run_v5b_pilot_queue import freeze_job, recover
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


def test_unwired_method_fails_preflight_instead_of_returning_proxy(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "scripts.run_v5b_pilot_queue._git_commit", lambda: "a" * 40
    )
    row = freeze_job(
        _job(tmp_path, method="flow_top4"), tmp_path / "output"
    )
    spec = tmp_path / "spec.json"
    row.update(
        {
            "heartbeat_path": str(tmp_path / "heartbeat.json"),
            "result_path": str(tmp_path / "result.json"),
            "artifact_directory": str(tmp_path / "artifacts"),
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
    payload = json.loads((tmp_path / "result.json").read_text())
    assert completed.returncode == 3
    assert payload["status"] == "UNSUPPORTED"
    assert "refusing to synthesize proxy results" in payload["error_message"]


@pytest.mark.parametrize("method", ["dual", "proposal", "top4", "full"])
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
