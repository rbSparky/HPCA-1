from __future__ import annotations

import json
from pathlib import Path

from scripts.repair_a2_data_layout import allocate, dfg_pointers


def _architecture() -> dict:
    return {
        "ARCH": {
            "SPM_B0_WRAPPER": {"DATA_LAYOUT": {"A": 0, "loopend": 2047}},
            "SPM_B1_WRAPPER": {"DATA_LAYOUT": {"B": 2048, "loopstart": 4094}},
        }
    }


def test_repair_handles_morpher_fragment_and_allocates_without_overlap(tmp_path: Path):
    arch = tmp_path / "arch.json"
    dfg = tmp_path / "kernel.xml"
    out = tmp_path / "repaired.json"
    arch.write_text(json.dumps(_architecture()))
    # Morpher's legacy export has two top-level elements and unescaped text;
    # pointer tags remain the stable contract we need to parse.
    dfg.write_text(
        '<MutexBB>\n</MutexBB>\n<DFG count="2">\n'
        '<Node><BasePointerName size="16">x</BasePointerName></Node>\n'
        '<Node text="a & b"><BasePointerName size="4">y</BasePointerName></Node>\n'
        '</DFG>\n'
    )
    assert dfg_pointers(dfg) == [("x", 16), ("y", 4)]
    report = allocate(arch, dfg, out)
    repaired = json.loads(out.read_text())
    b0 = repaired["ARCH"]["SPM_B0_WRAPPER"]["DATA_LAYOUT"]
    b1 = repaired["ARCH"]["SPM_B1_WRAPPER"]["DATA_LAYOUT"]
    assert b0["loopend"] == 2047
    assert b1["loopstart"] == 4094
    addresses = [b0.get("x"), b0.get("y"), b1.get("x"), b1.get("y")]
    addresses = [value for value in addresses if value is not None]
    assert len(addresses) == len(set(addresses))
    assert report["additions"]
