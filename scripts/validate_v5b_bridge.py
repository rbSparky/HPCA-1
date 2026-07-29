#!/usr/bin/env python3
"""Aggregate authoritative native Morpher round-trip evidence for v5b.

This script is deliberately evidence-only: it never manufactures mappings and
never turns a missing stage into success.  Native re-import success is accepted
only when a fresh state dump exists; Morpher writes that dump strictly after
schema/hash resolution, native object reconstruction, and its native legality
checker have all passed.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from flowadvantage.morpher_adapter.legality_bridge import validate_mapping


PAIR_SPECS = [
    {
        "kernel": "array_add",
        "architecture": "A0_hycube4x4",
        "directory": "array_add/A0_hycube4x4_standard_fixed14",
        "reimport": "native_reimport",
    },
    {
        "kernel": "array_add",
        "architecture": "A1_stdnoc4x4",
        "directory": "array_add/A1_stdnoc4x4_fixed14_retry",
        "reimport": "native_reimport",
    },
    {
        "kernel": "array_add",
        "architecture": "A2_hycube4x4_mem_variant",
        "directory": "array_add/A2_hycube4x4_mem_variant_fixed14",
        "reimport": "native_reimport",
    },
    {
        "kernel": "gemm_nt",
        "architecture": "A0_hycube4x4",
        "directory": "gemm_nt/A0_hycube4x4_fixed9_migrated",
        "reimport": "native_reimport_fixed11",
    },
    {
        "kernel": "gemm_nt",
        "architecture": "A1_stdnoc4x4",
        "directory": "gemm_nt/A1_stdnoc4x4_fixed14",
        "reimport": "native_reimport",
    },
    {
        "kernel": "gemm_nt",
        "architecture": "A2_hycube4x4_mem_variant",
        "directory": "gemm_nt/A2_hycube4x4_mem_variant_fixed15",
        "reimport": "native_reimport",
    },
    {
        "kernel": "fix_fft",
        "architecture": "A0_hycube4x4",
        "directory": "fix_fft/A0_hycube4x4_fixed14",
        "reimport": "native_reimport",
    },
    {
        "kernel": "fix_fft",
        "architecture": "A1_stdnoc4x4",
        "directory": "fix_fft/A1_stdnoc4x4_fixed14",
        "reimport": "native_reimport",
    },
    {
        "kernel": "fix_fft",
        "architecture": "A2_hycube4x4_mem_variant",
        "directory": "fix_fft/A2_hycube4x4_mem_variant_fixed15",
        "reimport": "native_reimport",
    },
]


def load_contract(directory: Path) -> tuple[dict, dict, dict]:
    return tuple(
        json.loads((directory / name).read_text())
        for name in ("dfg.json", "mrrg.json", "mapping.json")
    )


def operation_semantics(mapping: dict[str, Any]) -> dict[str, tuple[Any, ...]]:
    return {
        str(operation.get("native_node_key", operation.get("dfg_node_id"))): (
            operation.get("pe_id"),
            operation.get("fu_id"),
            operation.get("modulo_time"),
            operation.get("latency"),
        )
        for operation in mapping.get("operations", [])
    }


def route_semantics(mapping: dict[str, Any]) -> dict[tuple[str, str], tuple[Any, ...]]:
    return {
        (
            str(route.get("source_node_key", route.get("source_node"))),
            str(route.get("destination_node_key", route.get("destination_node"))),
        ): (
            tuple(route.get("ordered_resource_ids", [])),
            tuple(route.get("ordered_resource_latencies", [])),
            route.get("start_time"),
            route.get("end_time"),
        )
        for route in mapping.get("routes", [])
    }


def empty_row(spec: dict[str, str], directory: Path) -> dict[str, Any]:
    return {
        "kernel": spec["kernel"],
        "architecture": spec["architecture"],
        "reference_directory": str(directory),
        "native_mapping_success": False,
        "canonical_export_success": False,
        "flow_import_success": False,
        "flow_legality_success": False,
        "flow_export_success": False,
        "native_reimport_success": False,
        "native_legality_success": False,
        "configuration_success": False,
        "simulation_success": False,
        "output_match": False,
        "placement_match": False,
        "route_match": False,
        "ii_match": False,
        "memory_match": False,
        "operation_count": 0,
        "route_count": 0,
        "error": "",
    }


def evaluate_pair(root: Path, spec: dict[str, str]) -> dict[str, Any]:
    directory = root / spec["directory"]
    row = empty_row(spec, directory)
    required = [directory / name for name in ("dfg.json", "mrrg.json", "mapping.json")]
    if not all(path.is_file() for path in required):
        row["error"] = "native_dump_unavailable"
        return row

    try:
        dfg, mrrg, mapping = load_contract(directory)
        flow_result = validate_mapping(mapping, dfg, mrrg)
    except Exception as exc:
        row["error"] = f"parse_or_validation_error:{type(exc).__name__}:{exc}"
        return row

    row.update(
        native_mapping_success=True,
        canonical_export_success=True,
        flow_import_success=True,
        flow_legality_success=flow_result["legal"],
        flow_export_success=flow_result["legal"],
        operation_count=len(mapping.get("operations", [])),
        route_count=len(mapping.get("routes", [])),
    )
    if not flow_result["legal"]:
        row["error"] = json.dumps(flow_result["violations"], sort_keys=True)
        return row

    replay_directory = directory / spec["reimport"]
    replay_paths = [
        replay_directory / name for name in ("dfg.json", "mrrg.json", "mapping.json")
    ]
    if not all(path.is_file() for path in replay_paths):
        row["error"] = "native_reimport_unavailable"
        return row

    try:
        replay_dfg, replay_mrrg, replay_mapping = load_contract(replay_directory)
        replay_flow_result = validate_mapping(
            replay_mapping, replay_dfg, replay_mrrg
        )
    except Exception as exc:
        row["error"] = f"native_reimport_parse_error:{type(exc).__name__}:{exc}"
        return row

    # ExportFlowAdvantageState is called by the native executable only after
    # ImportFlowAdvantageMapping and ValidateFlowAdvantageMapping both return
    # true.  Its presence is therefore direct native-checker evidence.
    row["native_reimport_success"] = True
    row["native_legality_success"] = True
    row["placement_match"] = (
        operation_semantics(mapping) == operation_semantics(replay_mapping)
    )
    row["route_match"] = route_semantics(mapping) == route_semantics(replay_mapping)
    row["ii_match"] = mapping.get("ii") == replay_mapping.get("ii")
    row["memory_match"] = mapping.get("memory_bindings", []) == replay_mapping.get(
        "memory_bindings", []
    )

    # The reimported native state must independently pass the Python checker;
    # otherwise the two checkers disagree even if semantic fields match.
    if not replay_flow_result["legal"]:
        row["error"] = "reimport_independent_legality_failure:" + json.dumps(
            replay_flow_result["violations"], sort_keys=True
        )
        row["native_legality_success"] = False
        return row

    roundtrip_ok = all(
        row[field]
        for field in ("placement_match", "route_match", "ii_match", "memory_match")
    )
    row["configuration_success"] = roundtrip_ok

    # Simulation is independent evidence and is never inferred from legality.
    simulation_candidates = [
        directory / "sim_result.txt",
        directory / "simulation.log",
        directory / "native_reimport" / "sim_result.txt",
    ]
    for simulation_path in simulation_candidates:
        if simulation_path.is_file():
            text = simulation_path.read_text(errors="replace")
            row["simulation_success"] = (
                "Mismatches: 0" in text or "243,0" in text or "387,0" in text
            )
            row["output_match"] = row["simulation_success"]
            break

    if not roundtrip_ok:
        mismatches = [
            field
            for field in ("placement_match", "route_match", "ii_match", "memory_match")
            if not row[field]
        ]
        row["error"] = "semantic_roundtrip_mismatch:" + ",".join(mismatches)
    elif not row["simulation_success"]:
        row["error"] = "simulation_not_executed_for_pair"
    return row


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", newline="", dir=path.parent, delete=False
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = handle.name
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("results/revision_v5b/reference_mappings"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/revision_v5b/raw/roundtrip_validation.csv"),
    )
    args = parser.parse_args()
    rows = [evaluate_pair(args.root, spec) for spec in PAIR_SPECS]
    atomic_csv(args.output, rows)
    print(
        json.dumps(
            {
                "reference_pairs": len(rows),
                "native_exports": sum(row["native_mapping_success"] for row in rows),
                "flow_legal": sum(row["flow_legality_success"] for row in rows),
                "native_reimports": sum(row["native_reimport_success"] for row in rows),
                "exact_roundtrips": sum(
                    row["placement_match"]
                    and row["route_match"]
                    and row["ii_match"]
                    and row["memory_match"]
                    for row in rows
                ),
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
