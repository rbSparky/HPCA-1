#!/usr/bin/env python3
"""Derive Morpher's flat SPM allocation from a checked-in banked architecture.

Morpher's application-specific memory architecture files are authoritative for
the variable-to-bank placement and each bank's local addresses.  The generic
``update_mem_alloc.py`` interface, however, expects a flat address space where
bank ``i`` begins at ``i * bank_size``.  This utility performs exactly that
conversion and records enough provenance to audit every emitted address.

It deliberately does not invent variables or pack them heuristically.  Every
DFG memory/outer-loop pointer must already occur in the source architecture's
``DATA_LAYOUT``.  This is important for reproducing the checked-in FFT and GEMM
memory experiments without their original runtime-generated ``mem_alloc.txt``.
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


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _atomic_csv(path: Path, rows: list[tuple[str, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(("var_name", "base_addr"))
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _bank_layouts(architecture: dict[str, Any]) -> list[tuple[str, dict[str, int]]]:
    arch = architecture.get("ARCH")
    if not isinstance(arch, dict):
        raise ValueError("source architecture has no ARCH object")

    layouts: list[tuple[str, dict[str, int]]] = []
    for name, module in arch.items():
        if not name.startswith("SPM_B") or not name.endswith("_WRAPPER"):
            continue
        if not isinstance(module, dict):
            continue
        raw_layout = module.get("DATA_LAYOUT")
        if not isinstance(raw_layout, dict):
            continue
        layout: dict[str, int] = {}
        for variable, address in raw_layout.items():
            if isinstance(address, bool) or not isinstance(address, int):
                raise ValueError(f"{name}.DATA_LAYOUT[{variable!r}] is not an integer")
            if address < 0:
                raise ValueError(f"{name}.DATA_LAYOUT[{variable!r}] is negative")
            layout[str(variable)] = address
        layouts.append((name, layout))

    def bank_index(item: tuple[str, dict[str, int]]) -> int:
        name = item[0]
        middle = name[len("SPM_B") : -len("_WRAPPER")]
        if not middle.isdigit():
            raise ValueError(f"cannot parse bank index from {name!r}")
        return int(middle)

    layouts.sort(key=bank_index)
    if not layouts:
        raise ValueError("source architecture has no banked DATA_LAYOUT modules")
    expected = list(range(len(layouts)))
    observed = [bank_index(item) for item in layouts]
    if observed != expected:
        raise ValueError(f"bank indices are not contiguous: {observed}")
    return layouts


def derive(
    source_architecture: Path,
    bank_size: int,
    output_csv: Path,
    provenance_json: Path,
    required_variables: list[str],
) -> None:
    if bank_size <= 0:
        raise ValueError("bank_size must be positive")
    architecture = json.loads(source_architecture.read_text(encoding="utf-8"))
    layouts = _bank_layouts(architecture)

    converted: dict[str, int] = {}
    records: list[dict[str, Any]] = []
    for bank_index, (module, layout) in enumerate(layouts):
        for variable, local_address in sorted(layout.items()):
            if local_address >= bank_size:
                raise ValueError(
                    f"{module}.{variable} local address {local_address} "
                    f"is outside bank size {bank_size}"
                )
            if variable in converted:
                raise ValueError(f"variable {variable!r} appears in multiple banks")
            flat_address = bank_index * bank_size + local_address
            converted[variable] = flat_address
            records.append(
                {
                    "variable": variable,
                    "source_bank": bank_index,
                    "source_module": module,
                    "source_local_address": local_address,
                    "derived_flat_address": flat_address,
                }
            )

    required = sorted(set(required_variables))
    missing = sorted(set(required) - set(converted))
    if missing:
        raise ValueError(
            "source architecture lacks required DFG variables: " + ", ".join(missing)
        )

    # Keep only semantically required DFG variables.  Loop control addresses are
    # encoded directly as constants in the checked-in PartPred DFGs.
    rows = sorted((variable, converted[variable]) for variable in required)
    _atomic_csv(output_csv, rows)
    _atomic_json(
        provenance_json,
        {
            "schema": "morpher_mem_alloc_derivation_v1",
            "source_architecture": str(source_architecture.resolve()),
            "source_architecture_sha256": _sha256(source_architecture),
            "bank_size": bank_size,
            "bank_count": len(layouts),
            "required_variables": required,
            "emitted_variables": [name for name, _ in rows],
            "records": [
                record for record in records if record["variable"] in set(required)
            ],
            "conversion": "flat_address = source_bank * bank_size + source_local_address",
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-architecture", type=Path, required=True)
    parser.add_argument("--bank-size", type=int, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--provenance-json", type=Path, required=True)
    parser.add_argument(
        "--required-variable",
        action="append",
        default=[],
        help="DFG pointer variable that must exist; repeat for every pointer",
    )
    arguments = parser.parse_args()
    derive(
        source_architecture=arguments.source_architecture,
        bank_size=arguments.bank_size,
        output_csv=arguments.output_csv,
        provenance_json=arguments.provenance_json,
        required_variables=arguments.required_variable,
    )


if __name__ == "__main__":
    main()
