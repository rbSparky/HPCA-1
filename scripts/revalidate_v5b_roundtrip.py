#!/usr/bin/env python3
"""Reconcile pre-canonicalization round-trip errors without rerunning mapping.

The v8 queues were launched before the native route projection learned to
ignore Morpher's representation-only same-PE register/wait collapse.  This
script never changes an old queue row.  It creates an append-only repair
manifest only when the original serialized mapping, independent FlowAdvantage
legality result, native legality result, and the reimported mapping are all
present and the current canonical projection matches.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_v5b_pilot_worker import _semantic_mapping_projection


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue-root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    repaired: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for queue_root_value in args.queue_root:
        queue_root = queue_root_value.resolve()
        manifest = queue_root / "work_manifest.csv"
        if not manifest.is_file():
            continue
        with manifest.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            if row.get("status") != "ERROR":
                continue
            if "native Morpher reimport changed" not in row.get("error_message", ""):
                continue
            work_id = row["work_id"]
            artifact = queue_root / "artifacts" / work_id / "attempt1"
            mapping_path = artifact / "mapping.json"
            legality_path = artifact / "legality.json"
            roundtrip_path = artifact / "native_reimport" / "export" / "mapping.json"
            validation_path = artifact / "native_reimport" / "validation.json"
            required = (mapping_path, legality_path, roundtrip_path, validation_path)
            if not all(path.is_file() for path in required):
                diagnostics.append({"work_id": work_id, "status": "MISSING_REPAIR_INPUT"})
                continue
            try:
                mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
                replay = json.loads(roundtrip_path.read_text(encoding="utf-8"))
                legality = json.loads(legality_path.read_text(encoding="utf-8"))
                native = json.loads(validation_path.read_text(encoding="utf-8"))
                canonical_match = (
                    _semantic_mapping_projection(mapping)
                    == _semantic_mapping_projection(replay)
                )
                flow_legal = bool(legality.get("legal", legality.get("valid", False)))
                native_legal = bool(native.get("native_legality", False))
                if not (canonical_match and flow_legal and native_legal):
                    diagnostics.append({
                        "work_id": work_id,
                        "status": "REPAIR_REJECTED",
                        "canonical_match": canonical_match,
                        "flow_legal": flow_legal,
                        "native_legal": native_legal,
                    })
                    continue
                repaired_row = dict(row)
                repaired_row.update({
                    "status": "DONE",
                    "success": True,
                    "legal": True,
                    "flowadvantage_legality": True,
                    "morpher_legality": True,
                    "termination": "DONE_REPAIRED_CANONICAL_ROUTE",
                    "minimum_ii": mapping.get("ii", ""),
                    "route_cost": sum(
                        max(0, len(route.get("ordered_resource_ids", [])) - 1)
                        for route in mapping.get("routes", [])
                    ),
                    "operation_count": len(mapping.get("operations", [])),
                    "route_count": len(mapping.get("routes", [])),
                    "repair_source_queue": str(queue_root),
                    "repair_reason": (
                        "old strict route projection rejected Morpher's legal "
                        "same-PE register/wait canonicalization; current projection "
                        "matches endpoints, physical directed links, and timing"
                    ),
                    "old_status": row.get("status", ""),
                    "old_error_message": row.get("error_message", ""),
                    "canonical_validation_path": str(validation_path),
                })
                repaired.append(repaired_row)
            except (OSError, ValueError, KeyError, TypeError) as error:
                diagnostics.append({"work_id": work_id, "status": "REPAIR_PARSE_ERROR", "error": str(error)})
    fields = sorted({key for row in repaired for key in row})
    with (output / "repaired_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or ["work_id", "status"])
        writer.writeheader()
        writer.writerows(repaired)
    with (output / "repair_diagnostics.json").open("w", encoding="utf-8") as handle:
        json.dump({"repaired": len(repaired), "diagnostics": diagnostics}, handle, indent=2, sort_keys=True)
    print(json.dumps({"repaired": len(repaired), "diagnostics": len(diagnostics), "output": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
