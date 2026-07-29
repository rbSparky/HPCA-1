#!/usr/bin/env python3
"""Run the eight v5b illegal-mapping mutations through both legality checkers.

The input is a native Morpher export.  Every mutation, checker result, native
stdout/stderr stream, and final CSV row is written atomically.  A native
rejection is credited only when the patched mapper exits through its structured
import/legality rejection path; a crash or infrastructure error is not a pass.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from flowadvantage.morpher_adapter.legality_bridge import validate_mapping


THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
}


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def load_contract(reference: Path) -> tuple[dict, dict, dict]:
    return tuple(
        json.loads((reference / name).read_text())
        for name in ("dfg.json", "mrrg.json", "mapping.json")
    )


def mutation_cases(
    mapping: dict[str, Any], dfg: dict[str, Any], mrrg: dict[str, Any]
) -> list[tuple[str, str, dict[str, Any]]]:
    """Construct the exact eight prompt-required mutations."""

    resources = {
        value["native_resource_id"]: value for value in mrrg["resources"]
    }
    cases: list[tuple[str, str, dict[str, Any]]] = []

    duplicate = copy.deepcopy(mapping)
    first = duplicate["operations"][0]
    second = next(
        operation
        for operation in duplicate["operations"][1:]
        if operation["native_node_key"] != first["native_node_key"]
    )
    for field in ("pe_id", "pe_x", "pe_y", "fu_id", "modulo_time", "latency"):
        second[field] = first[field]
    cases.append(
        (
            "duplicate_compute_occupancy",
            "duplicate compute occupancy",
            duplicate,
        )
    )

    disconnected = copy.deepcopy(mapping)
    route = next(
        value
        for value in disconnected["routes"]
        if len(value.get("ordered_resource_ids", [])) >= 2
    )
    source_id = route["ordered_resource_ids"][0]
    adjacent = {
        edge["dst"] for edge in mrrg["edges"] if edge.get("src") == source_id
    }
    replacement = next(
        resource_id
        for resource_id, resource in resources.items()
        if resource.get("resource_type") == "port"
        and resource_id not in adjacent
        and resource_id != source_id
    )
    route["ordered_resource_ids"][1] = replacement
    route["ordered_link_ids"][0] = f"{source_id}->{replacement}"
    cases.append(("disconnected_route", "non-contiguous route", disconnected))

    incompatible = copy.deepcopy(mapping)
    mul = next(
        operation
        for operation in incompatible["operations"]
        if str(operation["opcode"]).upper() == "MUL"
    )
    # The frozen A0 is homogeneous, so it contains no physical PE whose FU
    # rejects MUL. Binding a MUL to the PE module rather than its FU is the
    # native contract's unambiguously incompatible-resource mutation.
    mul["fu_id"] = mul["pe_id"]
    cases.append(("incompatible_mul_pe", "unsupported operation", incompatible))

    outside = copy.deepcopy(mapping)
    operation = outside["operations"][0]
    operation["absolute_schedule_if_available"] = max(
        int(operation.get("alap", 0)) + 1, 1_000_000
    )
    cases.append(("schedule_window", "ASAP/ALAP violation", outside))

    recurrence = copy.deepcopy(mapping)
    recurrence_dependencies = {
        (
            str(value["source_node_key"]),
            str(value["destination_node_key"]),
        ): int(value.get("iteration_distance", 0) or 0)
        for value in dfg["dependencies"]
        if int(value.get("iteration_distance", 0) or 0) > 0
    }
    recurrence_route = next(
        value
        for value in recurrence["routes"]
        if (
            str(value["source_node_key"]),
            str(value["destination_node_key"]),
        )
        in recurrence_dependencies
    )
    ii = int(mapping["ii"])
    recurrence_route["ordered_resource_latencies"] = [
        int(value) + 2 * ii
        for value in recurrence_route["ordered_resource_latencies"]
    ]
    recurrence_route["start_time"] += 2 * ii
    recurrence_route["end_time"] += 2 * ii
    cases.append(("recurrence_timing", "recurrence violation", recurrence))

    memory = copy.deepcopy(mapping)
    memory_operation = next(
        operation
        for operation in memory["operations"]
        if str(operation["opcode"]).upper().startswith(("LOAD", "STORE"))
    )
    time_slot = int(memory_operation["modulo_time"])
    compute_fu = next(
        value
        for value in resources.values()
        if value.get("resource_type") == "FU"
        and value.get("memory_role") == "compute"
        and int(value.get("time_slot", -1)) == time_slot
    )
    compute_pe_prefix = str(compute_fu["native_resource_id"]).split(".FU0", 1)[0]
    compute_pe_id = next(
        resource_id
        for resource_id, value in resources.items()
        if value.get("resource_type") == "PE"
        and resource_id.startswith(compute_pe_prefix + ".")
    )
    compute_pe = resources[compute_pe_id]
    memory_operation.update(
        {
            "fu_id": compute_fu["native_resource_id"],
            "pe_id": compute_pe_id,
            "pe_x": compute_pe.get("x", memory_operation.get("pe_x")),
            "pe_y": compute_pe.get("y", memory_operation.get("pe_y")),
        }
    )
    cases.append(("memory_inaccessible_pe", "memory-port violation", memory))

    wrong_architecture = copy.deepcopy(mapping)
    wrong_architecture["architecture_hash"] = "0" * 16
    cases.append(
        (
            "architecture_hash",
            "architecture-hash mismatch",
            wrong_architecture,
        )
    )

    wrong_ii = copy.deepcopy(mapping)
    wrong_ii["ii"] = int(wrong_ii["ii"]) + 1
    cases.append(("ii_mismatch", "II mismatch", wrong_ii))
    return cases


def run_native(
    *,
    image: str,
    evidence: Path,
    dfg_path: str,
    architecture_path: str,
    ii: int,
    timeout: float,
) -> dict[str, Any]:
    command = [
        "docker",
        "run",
        "--rm",
        *sum((["-e", f"{key}={value}"] for key, value in THREAD_ENV.items()), []),
        "-v",
        f"{evidence.resolve()}:/evidence",
        image,
        "/home/user/morpher/Morpher_CGRA_Mapper/build/src/cgra_xml_mapper",
        "-d",
        dfg_path,
        "-x",
        "4",
        "-y",
        "4",
        "-j",
        architecture_path,
        "-i",
        str(ii),
        "-t",
        "HyCUBE_4REG",
        "-m",
        "0",
        "--load-flowadvantage-mapping",
        "/evidence/mapping.json",
        "--dump-flowadvantage-state",
        "/evidence/native_reimport",
    ]
    start = time.monotonic()
    try:
        completed = subprocess.run(
            command, text=True, capture_output=True, timeout=timeout, check=False
        )
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        completed = None
        timed_out = True
        stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else exc.stdout or ""
        stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else exc.stderr or ""
    else:
        stdout, stderr = completed.stdout, completed.stderr
    wall_seconds = time.monotonic() - start
    atomic_text(evidence / "native.stdout.log", stdout)
    atomic_text(evidence / "native.stderr.log", stderr)
    if timed_out:
        return {
            "native_exit_code": None,
            "native_timed_out": True,
            "native_rejected": False,
            "native_rejection_class": "",
            "native_wall_seconds": wall_seconds,
            "native_command": command,
        }
    structured_failure = (
        "FlowAdvantage import FAIL" in stdout
        or "FlowAdvantage native legality FAIL" in stdout
    )
    rejection_class = ""
    for line in stdout.splitlines():
        if "FlowAdvantage import FAIL" in line or "FlowAdvantage native legality FAIL" in line:
            rejection_class = line.rsplit(":", 1)[-1].strip()
    return {
        "native_exit_code": completed.returncode,
        "native_timed_out": False,
        "native_rejected": completed.returncode in (3, 4) and structured_failure,
        "native_rejection_class": rejection_class,
        "native_wall_seconds": wall_seconds,
        "native_command": command,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path(
            "results/revision_v5b/reference_mappings/"
            "gemm_nt/A0_hycube4x4_fixed9_migrated"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/revision_v5b/negative_legality"),
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("results/revision_v5b/raw/negative_legality_tests.csv"),
    )
    parser.add_argument(
        "--image", default="quotientflow-morpher-v5b-native-fixed15"
    )
    parser.add_argument(
        "--dfg-path",
        default=(
            "/home/user/morpher/Morpher_CGRA_Mapper/applications/gemm_nt/"
            "gemm_nt_INNERMOST_LN111_PartPred_DFG.xml"
        ),
    )
    parser.add_argument(
        "--architecture-path",
        default=(
            "/home/user/morpher/Morpher_CGRA_Mapper/json_arch/"
            "hycube_original.json"
        ),
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    dfg, mrrg, mapping = load_contract(args.reference)
    baseline = validate_mapping(mapping, dfg, mrrg)
    if not baseline["legal"]:
        raise RuntimeError(f"reference mapping is illegal: {baseline['violations']}")

    rows: list[dict[str, Any]] = []
    for mutation, expected_class, candidate in mutation_cases(mapping, dfg, mrrg):
        evidence = args.output / mutation
        evidence.mkdir(parents=True, exist_ok=True)
        atomic_json(evidence / "mapping.json", candidate)
        flow = validate_mapping(candidate, dfg, mrrg)
        flow_classes = sorted(
            {str(violation.get("class", "")) for violation in flow["violations"]}
        )
        native = run_native(
            image=args.image,
            evidence=evidence,
            dfg_path=args.dfg_path,
            architecture_path=args.architecture_path,
            ii=int(mapping["ii"]),
            timeout=args.timeout,
        )
        row = {
            "mutation": mutation,
            "reference_directory": str(args.reference),
            "expected_violation": expected_class,
            "flow_rejected": not flow["legal"],
            "flow_expected_violation_observed": expected_class in flow_classes,
            "flow_violation_classes": json.dumps(flow_classes),
            "native_rejected": native["native_rejected"],
            "native_rejection_class": native["native_rejection_class"],
            "native_exit_code": native["native_exit_code"],
            "native_timed_out": native["native_timed_out"],
            "native_wall_seconds": native["native_wall_seconds"],
            "mapping_path": str(evidence / "mapping.json"),
            "stdout_path": str(evidence / "native.stdout.log"),
            "stderr_path": str(evidence / "native.stderr.log"),
        }
        atomic_json(evidence / "result.json", row)
        rows.append(row)

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with tempfile.NamedTemporaryFile(
        mode="w", newline="", dir=args.csv.parent, delete=False
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
        temporary_csv = handle.name
    os.replace(temporary_csv, args.csv)

    failures = [
        row
        for row in rows
        if not (
            row["flow_rejected"]
            and row["flow_expected_violation_observed"]
            and row["native_rejected"]
        )
    ]
    print(
        json.dumps(
            {
                "mutations": len(rows),
                "fully_rejected": len(rows) - len(failures),
                "failures": [row["mutation"] for row in failures],
                "csv": str(args.csv),
            },
            indent=2,
        )
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
