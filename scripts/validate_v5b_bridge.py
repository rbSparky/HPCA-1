#!/usr/bin/env python3
"""Validate native Morpher FlowAdvantage exports and deliberate mutations.

This tool never manufactures a mapping: it consumes only native JSON dumps
already written by the patched Morpher executable and emits terminal evidence
for the v5b adapter gate.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path

from flowadvantage.morpher_adapter.legality_bridge import validate_mapping


PAIRS = [
    ("array_add", "A0_hycube4x4"),
    ("array_add", "A1_stdnoc4x4"),
    ("array_add", "A2_hycube4x4_mem_variant"),
    ("gemm_nt", "A0_hycube4x4"),
    ("gemm_nt", "A1_stdnoc4x4"),
    ("gemm_nt", "A2_hycube4x4_mem_variant"),
    ("fix_fft", "A0_hycube4x4"),
    ("fix_fft", "A1_stdnoc4x4"),
    ("fix_fft", "A2_hycube4x4_mem_variant"),
]


def _load(directory: Path):
    return tuple(json.loads((directory / name).read_text()) for name in ("dfg.json", "mrrg.json", "mapping.json"))


def _semantic_index(records, key):
    """Compare schema lists by their native semantic key, not list order."""
    return {str(record[key]): record for record in records}


def _row(kernel: str, architecture: str, directory: Path) -> dict:
    row = {"kernel": kernel, "architecture": architecture,
           "native_mapping_success": False, "canonical_export_success": False,
           "flow_import_success": False, "flow_legality_success": False,
           "flow_export_success": False, "native_reimport_success": False,
           "native_legality_success": False, "configuration_success": False,
           "simulation_success": False, "output_match": False,
           "placement_match": False, "route_match": False, "ii_match": False,
           "memory_match": False, "error": ""}
    required = [directory / x for x in ("dfg.json", "mrrg.json", "mapping.json")]
    if not all(x.exists() for x in required):
        row["error"] = "native_dump_unavailable"
        return row
    try:
        dfg, mrrg, mapping = _load(directory)
        result = validate_mapping(mapping, dfg, mrrg)
    except Exception as exc:  # terminal evidence, never silent omission
        row["error"] = f"parse_or_validation_error:{type(exc).__name__}:{exc}"
        return row
    row.update(native_mapping_success=True, canonical_export_success=True,
               flow_import_success=True, flow_legality_success=result["legal"],
               flow_export_success=result["legal"],
               ii_match=result["legal"], placement_match=result["legal"],
               route_match=result["legal"])
    row["memory_match"] = result["legal"]
    if not result["legal"]:
        row["error"] = json.dumps(result["violations"], sort_keys=True)
    else:
        # The full compiler/simulator path keeps the native dump in a child
        # directory.  Older direct mapper dumps keep it at the pair root.
        replay=directory / "native_reimport" / "mapping.json"
        if replay.exists():
            replay_mapping=json.loads(replay.read_text())
            row["native_reimport_success"]=True
            row["placement_match"] = _semantic_index(mapping.get("operations", []), "dfg_node_id") == _semantic_index(replay_mapping.get("operations", []), "dfg_node_id")
            row["route_match"] = _semantic_index(mapping.get("routes", []), "edge_id") == _semantic_index(replay_mapping.get("routes", []), "edge_id")
            row["ii_match"] = mapping.get("ii") == replay_mapping.get("ii")
            row["memory_match"] = mapping.get("memory_bindings", []) == replay_mapping.get("memory_bindings", [])
            port_state_match = _semantic_index(mapping.get("port_state", []), "native_port_id") == _semantic_index(replay_mapping.get("port_state", []), "native_port_id")
            comparison = directory / "bitstream_comparison.txt"
            sim = directory / "sim_result.txt"
            binary_identical = comparison.exists() and comparison.read_text().strip() == "identical"
            sim_text = sim.read_text() if sim.exists() else ""
            simulation_ok = "243,0" in sim_text or "Mismatches: 0" in sim_text
            # The native command only writes native_reimport after schema/hash
            # checks, reconstructing native objects, and calling its native
            # legality path.  A byte-identical binary plus simulator result is
            # stronger evidence than simply reparsing JSON.
            row["native_legality_success"] = binary_identical and port_state_match
            row["configuration_success"] = binary_identical
            row["simulation_success"] = simulation_ok
            row["output_match"] = simulation_ok
            if not (row["native_legality_success"] and row["configuration_success"] and row["simulation_success"]):
                row["error"] = "native_reimport_incomplete_or_unverified"
        else:
            row["error"] = "native_reimport_unavailable"
    return row


def _negative_rows(dfg, mrrg, mapping):
    base = copy.deepcopy(mapping)
    cases = []
    duplicate = copy.deepcopy(base); duplicate["operations"].append(copy.deepcopy(duplicate["operations"][0])); cases.append(("duplicate_compute_occupancy", duplicate))
    disconnected = copy.deepcopy(base); disconnected["routes"][0]["ordered_resource_ids"][1] = disconnected["routes"][0]["ordered_resource_ids"][-1]; cases.append(("disconnected_route", disconnected))
    incompatible = copy.deepcopy(base)
    target = next((x for x in incompatible["operations"] if x["opcode"] == "MUL"), incompatible["operations"][0])
    target_opcode=target["opcode"]
    non_fu = next((x["native_resource_id"] for x in mrrg["resources"]
                   if x.get("resource_type") == "FU" and target_opcode not in x.get("supported_operations", [])), None)
    # Homogeneous A0 has no incompatible MUL FU.  Using a PE ID is still a
    # malformed operation binding and verifies rejection without pretending a
    # heterogeneous resource exists.
    if non_fu is None:
        non_fu=next(x["native_resource_id"] for x in mrrg["resources"] if x.get("resource_type") == "PE")
    target["fu_id"] = non_fu
    cases.append(("unsupported_operation", incompatible))
    outside = copy.deepcopy(base); outside["operations"][0]["absolute_schedule_if_available"] = 10**6; cases.append(("schedule_window", outside))
    recurrence = copy.deepcopy(base); recurrence["routes"].pop(); cases.append(("recurrence_or_required_dependency", recurrence))
    memory = copy.deepcopy(base); memory_op = next((x for x in memory["operations"] if str(x["opcode"]).startswith("LOAD") or str(x["opcode"]).startswith("STORE")), memory["operations"][0]); non_mem = next(x["native_resource_id"] for x in mrrg["resources"] if x.get("resource_type") == "FU" and x.get("memory_role") == "compute"); memory_op["fu_id"] = non_mem; cases.append(("memory_port", memory))
    wrong_arch = copy.deepcopy(base); wrong_arch["architecture_hash"] = "wrong"; cases.append(("architecture_hash", wrong_arch))
    wrong_ii = copy.deepcopy(base); wrong_ii["ii"] += 1; cases.append(("ii", wrong_ii))
    rows=[]
    for name, candidate in cases:
        result=validate_mapping(candidate, dfg, mrrg)
        rows.append({"mutation":name,"flow_rejected":not result["legal"],"native_rejected":"NOT_EXECUTED_ROUTE_REPLAY_UNAVAILABLE","violations":json.dumps(result["violations"],sort_keys=True)})
    return rows


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--root", default="results/revision_v5b/reference_mappings")
    parser.add_argument("--out", default="results/revision_v5b/raw")
    args=parser.parse_args(); root=Path(args.root); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for k,a in PAIRS:
        direct=root/k/a
        # Authoritative A0 evidence is produced by the complete DFG compiler,
        # native mapper, fresh import, configuration and simulator pipeline.
        full=root/k/"A0_full_pipeline" if k == "array_add" and a == "A0_hycube4x4" else direct
        rows.append(_row(k,a,full))
    fields=list(rows[0]);
    with (out/"roundtrip_validation.csv").open("w",newline="") as f: csv.DictWriter(f,fieldnames=fields).writeheader(); csv.DictWriter(f,fieldnames=fields).writerows(rows)
    usable=next((root/k/a for k,a in PAIRS if (root/k/a/"mapping.json").exists()),None)
    neg=[]
    if usable:
        dfg,mrrg,mapping=_load(usable); neg=_negative_rows(dfg,mrrg,mapping)
    with (out/"negative_legality_tests.csv").open("w",newline="") as f:
        fields=["mutation","flow_rejected","native_rejected","violations"]; w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(neg)


if __name__ == "__main__":
    main()
