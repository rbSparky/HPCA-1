#!/usr/bin/env python3
"""Add Morpher's source-defined loop-control cells to an SPM allocation.

For the pinned 32-bit toolchain the DFG generator emits:

* ``loopend = MEM_SIZE / 2 - 1``
* ``loopstart = MEM_SIZE - 2``

The formula is implemented in
``Morpher_DFG_Generator/src/common/dfg.cpp::DFG::nameNodes``.  Keeping these
simulator control cells separate from application pointer allocation avoids
polluting architecture DATA_LAYOUT metadata.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def build_simulator_allocation(
    input_csv: Path,
    output_csv: Path,
    provenance_json: Path,
    total_memory_size: int,
    source_reference: str,
) -> dict[str, int]:
    if total_memory_size <= 2 or total_memory_size % 2:
        raise ValueError("total_memory_size must be an even integer greater than two")

    with input_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["var_name", "base_addr"]:
            raise ValueError(
                "input allocation must have exactly var_name,base_addr columns"
            )
        allocation: dict[str, int] = {}
        used_addresses: dict[int, str] = {}
        for row in reader:
            variable = row["var_name"].strip()
            if not variable:
                raise ValueError("empty variable name")
            if variable in allocation:
                raise ValueError(f"duplicate variable {variable!r}")
            try:
                address = int(row["base_addr"])
            except ValueError as error:
                raise ValueError(f"non-integer address for {variable!r}") from error
            if address < 0 or address >= total_memory_size:
                raise ValueError(f"address for {variable!r} is outside memory")
            if address in used_addresses:
                raise ValueError(
                    f"address collision: {variable!r} and "
                    f"{used_addresses[address]!r} both use {address}"
                )
            allocation[variable] = address
            used_addresses[address] = variable

    control = {
        "loopend": total_memory_size // 2 - 1,
        "loopstart": total_memory_size - 2,
    }
    for variable, address in control.items():
        if variable in allocation:
            raise ValueError(f"input allocation already contains {variable!r}")
        if address in used_addresses:
            raise ValueError(
                f"reserved {variable} address {address} collides with "
                f"{used_addresses[address]!r}"
            )
        allocation[variable] = address
        used_addresses[address] = variable

    rows = sorted(allocation.items())
    csv_lines = ["var_name,base_addr"]
    csv_lines.extend(f"{name},{address}" for name, address in rows)
    _atomic_write(output_csv, "\n".join(csv_lines) + "\n")

    provenance: dict[str, Any] = {
        "schema": "morpher_sim_mem_alloc_v1",
        "input_allocation": str(input_csv.resolve()),
        "input_allocation_sha256": _sha256(input_csv),
        "total_memory_size": total_memory_size,
        "source_reference": source_reference,
        "control_addresses": control,
        "formula": {
            "loopend": "total_memory_size / 2 - 1",
            "loopstart": "total_memory_size - 2",
        },
        "output_allocation": str(output_csv.resolve()),
        "records": [
            {"variable": variable, "address": address} for variable, address in rows
        ],
    }
    _atomic_write(
        provenance_json, json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    )
    return allocation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--provenance-json", type=Path, required=True)
    parser.add_argument("--total-memory-size", type=int, required=True)
    parser.add_argument("--source-reference", required=True)
    arguments = parser.parse_args()
    build_simulator_allocation(
        input_csv=arguments.input_csv,
        output_csv=arguments.output_csv,
        provenance_json=arguments.provenance_json,
        total_memory_size=arguments.total_memory_size,
        source_reference=arguments.source_reference,
    )


if __name__ == "__main__":
    main()
