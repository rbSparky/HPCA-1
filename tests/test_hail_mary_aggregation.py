from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from flowadvantage.paper_results import (
    AggregationError,
    comparison_rows,
    failure_census,
    load_atomic_records,
    paired_bootstrap,
    paired_records,
)


FIELDS = [
    "work_id", "kernel", "architecture", "method", "seed", "budget_seconds",
    "initialization_depth_fraction", "replicate", "status", "result_path",
    "config_hash", "source_commit", "dfg_hash", "architecture_hash",
]


def _write_case(tmp_path: Path, rows: list[tuple[dict, dict]]) -> Path:
    items = tmp_path / "work_items"
    items.mkdir()
    manifest_rows = []
    for manifest, result in rows:
        path = items / f"{manifest['work_id']}.json"
        path.write_text(json.dumps(result, sort_keys=True) + "\n")
        manifest_rows.append({**{field: "" for field in FIELDS}, **manifest, "result_path": str(path)})
    path = tmp_path / "work_manifest.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(manifest_rows)
    return path


def _row(work_id: str, method: str, status: str, *, success: bool, ii: int = 3, **metrics):
    config_hash = f"cfg-{work_id}"
    manifest = {
        "work_id": work_id, "kernel": "k", "architecture": "A0",
        "method": method, "seed": "0", "budget_seconds": "600",
        "initialization_depth_fraction": "0.0", "replicate": "0",
        "status": status, "config_hash": config_hash,
        "source_commit": "commit", "dfg_hash": "dfg", "architecture_hash": "arch",
    }
    result = {
        "work_id": work_id, "kernel": "k", "architecture": "A0",
        "method": method, "seed": 0, "status": status,
        "config_hash": config_hash, "source_commit": "commit",
        "dfg_hash": "dfg", "architecture_hash": "arch", "success": success,
        "schema": "test_result_v1",
        **metrics,
    }
    if success:
        result.update({
            "legal": True, "flowadvantage_legality": True,
            "morpher_legality": True, "ii": ii,
            "mapping_hash": f"mapping-{work_id}",
            "ii_attempts": [
                {"ii": ii - 1, "status": "FAILED"},
                {"ii": ii, "status": "MAPPED"},
            ],
        })
    else:
        result.update({"legal": False, "ii_attempts": [{"ii": ii, "status": "FAILED"}]})
    return manifest, result


def test_validates_and_uses_only_equal_normal_pair_intersection(tmp_path):
    rows = [
        _row("b0", "length", "VALID_MAPPING_FAILURE", success=False, compile_wall_seconds=5, failed_routes=4),
        _row("c0", "top4", "DONE", success=True, ii=4, compile_wall_seconds=8, failed_routes=1, route_cost=7),
    ]
    # A timeout has no scientific mapping outcome and cannot enlarge the pair.
    manifest, result = _row("b1", "length", "TIMEOUT", success=False)
    manifest["kernel"] = result["kernel"] = "other"
    result.pop("legal")
    rows.append((manifest, result))
    manifest, result = _row("c1", "top4", "DONE", success=True, ii=3, compile_wall_seconds=9, route_cost=6)
    manifest["kernel"] = result["kernel"] = "other"
    rows.append((manifest, result))
    records, attempts = load_atomic_records(_write_case(tmp_path, rows))
    pairs = paired_records(records, "length", "top4")
    assert [(left.work_id, right.work_id) for left, right in pairs] == [("b0", "c0")]
    table = comparison_rows(records, [("length", "top4")], resamples=100)
    success = next(row for row in table if row["metric"] == "success")
    assert success["paired_instances"] == 1
    assert success["point_estimate"] == 1.0
    assert len(attempts) == 6
    failures = failure_census(records)
    assert any(row["status"] == "TIMEOUT" for row in failures)


def test_bootstrap_is_seeded_and_paired():
    first = paired_bootstrap([0, 1, 1], [1, 1, 0], resamples=1_000)
    second = paired_bootstrap([0, 1, 1], [1, 1, 0], resamples=1_000)
    assert first == second
    assert first[0] == pytest.approx(0.0)


def test_dual_legality_is_required_for_paper_success(tmp_path):
    manifest, result = _row("x", "top4", "DONE", success=True)
    del result["morpher_legality"]
    path = _write_case(tmp_path, [(manifest, result)])
    with pytest.raises(AggregationError, match="independent and native"):
        load_atomic_records(path)
    records, _ = load_atomic_records(path, legality_policy="legacy")
    assert records[0].success is True


def test_rejects_manifest_result_hash_or_status_mismatch(tmp_path):
    manifest, result = _row("x", "top4", "DONE", success=True)
    result["config_hash"] = "wrong"
    with pytest.raises(AggregationError, match="config_hash mismatch"):
        load_atomic_records(_write_case(tmp_path, [(manifest, result)]))


def test_strict_provenance_requires_frozen_hashes_in_atomic_result(tmp_path):
    manifest, result = _row("x", "top4", "DONE", success=True)
    del result["source_commit"]
    path = _write_case(tmp_path, [(manifest, result)])
    with pytest.raises(AggregationError, match="omitted frozen source_commit"):
        load_atomic_records(path)
    records, _ = load_atomic_records(path, legality_policy="legacy")
    assert len(records) == 1


def test_rejects_duplicate_semantic_work(tmp_path):
    one = _row("x", "top4", "DONE", success=True)
    two = _row("y", "top4", "DONE", success=True)
    with pytest.raises(AggregationError, match="duplicate semantic work"):
        load_atomic_records(_write_case(tmp_path, [one, two]))


def test_rejects_nonmonotonic_or_post_success_ii_search(tmp_path):
    manifest, result = _row("x", "top4", "DONE", success=True)
    result["ii_attempts"] = [
        {"ii": 3, "status": "MAPPED"},
        {"ii": 4, "status": "FAILED"},
    ]
    with pytest.raises(AggregationError, match="continued after"):
        load_atomic_records(_write_case(tmp_path, [(manifest, result)]))


def test_terminal_row_requires_atomic_result(tmp_path):
    path = tmp_path / "work_manifest.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerow({
            **{field: "" for field in FIELDS}, "work_id": "missing",
            "kernel": "k", "architecture": "A0", "method": "length",
            "seed": "0", "status": "TIMEOUT",
        })
    with pytest.raises(AggregationError, match="has no atomic result"):
        load_atomic_records(path)


def test_cli_atomically_emits_censuses_and_bootstrap_tables(tmp_path):
    root = tmp_path / "hail"
    root.mkdir()
    manifest = _write_case(
        root,
        [
            _row("b", "length", "VALID_MAPPING_FAILURE", success=False, compile_wall_seconds=2),
            _row("c", "top4", "DONE", success=True, ii=3, compile_wall_seconds=3, route_cost=4),
        ],
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/aggregate_hail_mary.py",
            "--root", str(root),
            "--manifest", str(manifest),
            "--compare", "length:top4",
            "--no-update-tracker",
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )
    assert (root / "raw/bootstrap_results.csv").is_file()
    assert (root / "tables/failure_census.md").is_file()
    summary = json.loads((root / "aggregation_summary.json").read_text())
    assert summary["normal_completions"] == 2
    with (root / "raw/bootstrap_results.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    success = next(row for row in rows if row["metric"] == "success")
    assert success["paired_instances"] == "1"
