#!/usr/bin/env python3
"""Migrate a pre-edge-semantics Morpher dump without changing mapping content.

The first native bridge revision exported every DFG dependency as ``data`` and
attempted to materialize Morpher ``PS`` pseudo edges as MRRG routes.  Morpher's
own PathFinder deliberately excludes PS edges from route estimation.  This
utility recovers the exact native operand type from the DFG XML, annotates every
dependency, and removes only route records corresponding to non-routed PS
edges.  It never synthesizes placements, routes, or resources.

The migration is intentionally provenance preserving: it writes a new output
directory, records SHA-256 hashes of every input, and refuses ambiguous node or
edge matches.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _parse_native_xml(path: Path) -> tuple[ET.Element, ET.Element]:
    # Several checked-in Morpher DFGs omit whitespace between attributes
    # (e.g. ALAP="1"BB="..."). TinyXML accepts this historical syntax; the
    # standard-library parser needs the semantically neutral whitespace repair.
    text = path.read_text(encoding="utf-8")
    text = re.sub(r'"(?=[A-Za-z_][A-Za-z0-9_]*=)', '" ', text)
    wrapper = ET.fromstring(f"<FlowAdvantageWrapper>{text}</FlowAdvantageWrapper>")
    dfg = wrapper.find("DFG")
    if dfg is None:
        raise ValueError(f"{path}: missing DFG element")
    return wrapper, dfg


def _xml_node_candidates(
    native_dfg: ET.Element, exported_nodes: list[dict[str, Any]]
) -> dict[int, list[dict[str, Any]]]:
    by_id: dict[int, list[dict[str, Any]]] = {}
    for node in exported_nodes:
        by_id.setdefault(int(node["dfg_node_id"]), []).append(node)

    resolved: dict[int, list[dict[str, Any]]] = {}
    for xml_node in native_dfg.findall("Node"):
        native_id = int(xml_node.attrib["idx"])
        opcode = (xml_node.findtext("OP") or "").strip()
        asap = int(xml_node.attrib.get("ASAP", 0))
        alap = int(xml_node.attrib.get("ALAP", 0))
        bb = xml_node.attrib.get("BB", "")
        matches = [
            node
            for node in by_id.get(native_id, [])
            if str(node.get("opcode", "")) == opcode
            and int(node.get("asap", 0)) == asap
            and int(node.get("alap", 0)) == alap
            and str(node.get("bb", "")) == bb
        ]
        if len(matches) != 1:
            raise ValueError(
                "ambiguous native node: "
                f"id={native_id} op={opcode} ASAP={asap} ALAP={alap} BB={bb!r}; "
                f"matches={len(matches)}"
            )
        resolved.setdefault(native_id, []).append(
            {"xml": xml_node, "export": matches[0]}
        )
    if sum(map(len, resolved.values())) != len(exported_nodes):
        raise ValueError("native/exported DFG node census mismatch")
    return resolved


def migrate(source: Path, destination: Path, xml_path: Path) -> dict[str, Any]:
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"destination is not empty: {destination}")
    required = [source / name for name in ("dfg.json", "mrrg.json", "mapping.json")]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    dfg_doc = json.loads(required[0].read_text(encoding="utf-8"))
    mrrg_doc = json.loads(required[1].read_text(encoding="utf-8"))
    mapping_doc = json.loads(required[2].read_text(encoding="utf-8"))
    native_document, native_dfg = _parse_native_xml(xml_path)
    resolved = _xml_node_candidates(native_dfg, dfg_doc["nodes"])

    edge_types: dict[tuple[str, str], str] = {}
    for candidates in resolved.values():
        for candidate in candidates:
            source_key = candidate["export"]["native_node_key"]
            for output in candidate["xml"].findall("./Outputs/Output"):
                destination_id = int(output.attrib["idx"])
                destinations = resolved.get(destination_id, [])
                # Resolve duplicate numeric IDs using the already-exported edge.
                matching_exported_edges = [
                    edge
                    for edge in dfg_doc["dependencies"]
                    if edge["source_node_key"] == source_key
                    and edge["destination_node_key"]
                    in {item["export"]["native_node_key"] for item in destinations}
                ]
                if len(matching_exported_edges) != 1:
                    raise ValueError(
                        "ambiguous native edge: "
                        f"{source_key} -> numeric destination {destination_id}; "
                        f"matches={len(matching_exported_edges)}"
                    )
                destination_key = matching_exported_edges[0]["destination_node_key"]
                key = (source_key, destination_key)
                if key in edge_types:
                    raise ValueError(f"duplicate native edge {key}")
                edge_types[key] = output.attrib.get("type", "")

    dependencies = dfg_doc["dependencies"]
    if len(edge_types) != len(dependencies):
        raise ValueError(
            f"native/exported edge census mismatch: {len(edge_types)} != {len(dependencies)}"
        )
    pseudo_edges: set[tuple[str, str]] = set()
    for dependency in dependencies:
        key = (
            dependency["source_node_key"],
            dependency["destination_node_key"],
        )
        native_type = edge_types[key]
        dependency["edge_type"] = native_type
        dependency["requires_route"] = native_type != "PS"
        if native_type == "PS":
            pseudo_edges.add(key)

    mutex_pairs = []
    mutex = native_document.find("MutexBB")
    if mutex is not None:
        for left in mutex.findall("BB1"):
            left_name = left.attrib.get("name", left.attrib.get("Name", ""))
            for right in left.findall("BB2"):
                right_name = right.attrib.get("name", right.attrib.get("Name", ""))
                if left_name and right_name:
                    mutex_pairs.append({"left": left_name, "right": right_name})
    dfg_doc["mutex_basic_blocks"] = mutex_pairs

    # Export the native Port::isMutexNodes operand-port rule explicitly so an
    # independent consumer does not need to infer it from coordinates.
    for resource in mrrg_doc["resources"]:
        if resource.get("resource_type") != "port":
            continue
        resource_id = str(resource.get("native_resource_id", ""))
        port_name = resource_id.rsplit(".", 1)[-1]
        resource["allows_operand_mux"] = (
            port_name == "P"
            or "_P" in port_name
            or "I1" in port_name
            or "I2" in port_name
        )

    old_routes = mapping_doc["routes"]
    invalid_non_pseudo = [
        route
        for route in old_routes
        if route.get("export_status", "ok") != "ok"
        and (
            route["source_node_key"],
            route["destination_node_key"],
        )
        not in pseudo_edges
    ]
    if invalid_non_pseudo:
        raise ValueError(
            f"refusing to hide {len(invalid_non_pseudo)} failed physical routes"
        )
    mapping_doc["routes"] = [
        route
        for route in old_routes
        if (
            route["source_node_key"],
            route["destination_node_key"],
        )
        not in pseudo_edges
    ]

    provenance = {
        "migration": "native_edge_semantics_v1",
        "source_directory": str(source.resolve()),
        "native_dfg_xml": str(xml_path.resolve()),
        "input_sha256": {
            path.name: _sha256(path) for path in required
        },
        "native_dfg_xml_sha256": _sha256(xml_path),
        "pseudo_edges_removed_from_route_universe": len(pseudo_edges),
        "physical_routes_preserved": len(mapping_doc["routes"]),
    }
    dfg_doc.setdefault("metadata", {})["migration"] = provenance
    mapping_doc.setdefault("metadata", {})["migration"] = provenance
    mrrg_doc.setdefault("metadata", {})["migration"] = provenance

    destination.mkdir(parents=True, exist_ok=True)
    _atomic_json(destination / "dfg.json", dfg_doc)
    _atomic_json(destination / "mrrg.json", mrrg_doc)
    _atomic_json(destination / "mapping.json", mapping_doc)
    _atomic_json(destination / "migration_manifest.json", provenance)
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--dfg-xml", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(migrate(args.source, args.destination, args.dfg_xml), indent=2))


if __name__ == "__main__":
    main()
