#!/usr/bin/env python3
"""Index validated native export artifacts without claiming local executability.

The registry is deliberately strict: a canonical JSON export is not treated as
an executable native input. Native XML/architecture paths and their recorded
SHA-256 values come from the immutable export manifest; ``native_available``
is true only when both files are present locally and hash-verified.
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from scripts.v5b_pilot_queue_common import sha256_file

FIELDS = ("TOOLCHAIN", "BINARY_SHA256", "DFG_REL", "DFG_SHA256", "ARCH_REL", "ARCH_SHA256", "X", "Y", "INITIAL_II", "PE_TYPE", "METHOD")

def parse_env(path: Path) -> dict[str, str]:
    values = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1); values[key] = value.replace("\\,", ",")
    missing = [key for key in FIELDS if not values.get(key)]
    if missing: raise ValueError(f"{path}: missing fields {missing}")
    for key in ("BINARY_SHA256", "DFG_SHA256", "ARCH_SHA256"):
        if not re.fullmatch(r"[0-9a-f]{64}", values[key]): raise ValueError(f"{path}: invalid {key}")
    return values

def build(root: Path, output: Path) -> dict:
    rows = []
    for manifest in sorted(root.glob("**/manifest.env")):
        pair = manifest.parent
        try: env = parse_env(manifest)
        except ValueError: continue
        docs = {name: pair / f"{name}.json" for name in ("dfg", "mrrg", "mapping")}
        if not all(path.is_file() for path in docs.values()): continue
        for path in docs.values():
            json.loads(path.read_text())
        # Native paths are recorded as provenance. They are executable only if
        # a caller separately stages/hash-verifies the XML and architecture.
        native_dfg = Path(env["TOOLCHAIN"]) / "applications" / Path(env["DFG_REL"]).name
        native_arch = Path(env["TOOLCHAIN"]) / env["ARCH_REL"]
        rows.append({
            "pair": pair.relative_to(root).as_posix(),
            "manifest": manifest.relative_to(root).as_posix(),
            "native_toolchain": env["TOOLCHAIN"], "native_binary_sha256": env["BINARY_SHA256"],
            "native_dfg": env["DFG_REL"], "native_dfg_sha256": env["DFG_SHA256"],
            "native_architecture": env["ARCH_REL"], "native_architecture_sha256": env["ARCH_SHA256"],
            "native_available": bool(native_dfg.is_file() and native_arch.is_file() and sha256_file(native_dfg) == env["DFG_SHA256"] and sha256_file(native_arch) == env["ARCH_SHA256"]),
            "canonical_json": {name: sha256_file(path) for name, path in docs.items()},
            "ii": int(json.loads(docs["mapping"].read_text())["ii"]),
        })
    payload = {"schema": "flowadvantage_native_input_registry_v1", "rows": rows, "native_executable_rows": sum(bool(row["native_available"]) for row in rows)}
    output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload

def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--root", type=Path, default=Path("results/revision_v5b/reference_mappings")); p.add_argument("--output", type=Path, default=Path("results/revision_v5b/native_input_registry.json")); a = p.parse_args(); print(json.dumps(build(a.root, a.output), indent=2))
if __name__ == "__main__": main()
