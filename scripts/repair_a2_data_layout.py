#!/usr/bin/env python3
"""Build a kernel-specific A2 memory layout from the native DFG contract.

The HyCUBE memory variant stores a DATA_LAYOUT entry for every base pointer
referenced by a DFG.  The stock array_add architecture only contains the
three array names used by that application, so reusing it for another DFG
causes Morpher's UpdateVariableBaseAddr() to abort.  This utility preserves
the validated hardware description and allocates missing pointer objects in
the two existing scratchpad banks using a deterministic, non-overlapping
first-fit allocator.  Sentinel loop addresses remain fixed at the native
values.  No operation or routing semantics are changed.
"""
from __future__ import annotations

import argparse
import copy
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path


def dfg_pointers(xml_path: Path) -> list[tuple[str, int]]:
    root = ET.parse(xml_path).getroot()
    out: dict[str, int] = {}
    # Morpher emits both attribute and text forms in different DFG versions.
    for elem in root.iter():
        if elem.tag.split("}")[-1] != "BasePointerName":
            continue
        name = (elem.text or elem.attrib.get("name") or "").strip()
        if not name:
            continue
        try:
            size = int(elem.attrib.get("size", "1"))
        except ValueError:
            size = 1
        out[name] = max(1, size)
    # Also accept an attribute on the operation node used by older exports.
    for elem in root.iter():
        name = elem.attrib.get("BasePointerName") or elem.attrib.get("base_pointer_name")
        if name:
            try:
                size = int(elem.attrib.get("size", "1"))
            except ValueError:
                size = 1
            out[name] = max(out.get(name, 1), size)
    return sorted(out.items())


def layout_sections(arch: dict) -> tuple[dict, dict]:
    a = arch["ARCH"]
    b0 = a["SPM_B0_WRAPPER"]["DATA_LAYOUT"]
    b1 = a["SPM_B1_WRAPPER"]["DATA_LAYOUT"]
    return b0, b1


def local_offset(addr: int, bank: int) -> int:
    return addr if bank == 0 else addr - 2048


def allocate(arch_path: Path, dfg_path: Path, output_path: Path) -> dict:
    arch = json.loads(arch_path.read_text())
    pointers = dfg_pointers(dfg_path)
    b0, b1 = layout_sections(arch)
    # Preserve the native sentinel addresses and any existing application
    # layout entries.  These are part of the architecture's memory contract.
    fixed = {"loopend", "loopstart"}
    used = [0, 0]
    for bank, layout in enumerate((b0, b1)):
        for name, addr in layout.items():
            if name in fixed:
                continue
            used[bank] = max(used[bank], local_offset(int(addr), bank) + 1)

    additions = []
    for name, size in pointers:
        if name in fixed or name in b0 or name in b1:
            continue
        # First-fit by current footprint.  Ties are resolved to B0 so the
        # result is independent of Python hash randomization.
        bank = 0 if used[0] <= used[1] else 1
        base = (used[bank] + 3) & ~3
        if base + size > 2047:
            other = 1 - bank
            base = (used[other] + 3) & ~3
            if base + size > 2047:
                raise RuntimeError(f"scratchpad capacity exhausted for {name} size={size}")
            bank = other
        address = base + (2048 if bank else 0)
        (b0 if bank == 0 else b1)[name] = address
        used[bank] = base + size
        additions.append({"name": name, "size": size, "bank": f"B{bank}", "address": address})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(arch, indent=2) + "\n")
    return {
        "architecture": str(arch_path),
        "dfg": str(dfg_path),
        "output": str(output_path),
        "pointers": [{"name": n, "size": s} for n, s in pointers],
        "additions": additions,
        "bank_footprint_bytes": used,
        "capacity_bytes_per_bank": 2048,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--architecture", required=True, type=Path)
    ap.add_argument("--dfg", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--report", type=Path)
    args = ap.parse_args()
    report = allocate(args.architecture, args.dfg, args.output)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
