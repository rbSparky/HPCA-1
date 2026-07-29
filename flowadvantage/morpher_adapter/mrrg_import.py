"""Import Morpher architecture descriptions and native II-expanded MRRGs.

The mapper's native MRRG dump is the authoritative input for FlowAdvantage.
Architecture JSON is retained only for inventory and pre-build inspection: it
does not encode the time-expanded graph, generated memory wrappers, or all
routing ports used by Morpher's mapper.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _read(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def import_architecture_json(path):
    obj = _read(path)
    arch = obj.get("ARCH", obj)
    ops = []
    for section in ("FU", "FU_MEM"):
        if isinstance(arch.get(section), dict):
            ops.extend(arch[section].get("OPS", {}).keys())
    canonical = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    return {
        "format": "morpher_arch_json",
        "raw": obj,
        "supported_ops": sorted(set(ops)),
        "architecture_hash": hashlib.sha256(canonical).hexdigest(),
    }


def import_native_mrrg(path):
    """Parse a ``flowadvantage_morpher_mrrg_v1`` native dump losslessly.

    Resource IDs are never reconstructed from coordinates: Morpher can expose
    several ports/FUs at a PE and modulo phase, and only its native IDs retain
    that distinction.  The returned indexes are intentionally plain Python
    data so model/mapper code can convert them to compact arrays without
    carrying NetworkX objects between workers.
    """
    payload = _read(path)
    if payload.get("schema") != "flowadvantage_morpher_mrrg_v1":
        raise ValueError(f"unsupported native MRRG schema: {payload.get('schema')!r}")
    ii = payload.get("ii")
    if not isinstance(ii, int) or ii <= 0:
        raise ValueError("native MRRG is missing a positive II")
    resources = payload.get("resources")
    edges = payload.get("edges")
    if not isinstance(resources, list) or not isinstance(edges, list):
        raise ValueError("native MRRG requires resource and edge arrays")
    by_id = {}
    for resource in resources:
        resource_id = resource.get("native_resource_id")
        if not isinstance(resource_id, str) or not resource_id:
            raise ValueError("MRRG resource missing native_resource_id")
        if resource_id in by_id:
            # The exporter may encounter a module via multiple traversal roots.
            # Duplicates are acceptable only when their semantic records agree.
            if by_id[resource_id] != resource:
                raise ValueError(f"conflicting duplicate MRRG resource {resource_id}")
            continue
        by_id[resource_id] = resource
    adjacency = {resource_id: [] for resource_id in by_id}
    reverse_adjacency = {resource_id: [] for resource_id in by_id}
    edge_records = []
    seen_edges = set()
    for edge in edges:
        src, dst = edge.get("src"), edge.get("dst")
        if src not in by_id or dst not in by_id:
            raise ValueError(f"MRRG edge references unknown resource: {src!r}->{dst!r}")
        key = (src, dst)
        if key in seen_edges:
            continue
        seen_edges.add(key)
        record = {"src": src, "dst": dst, "capacity": float(edge.get("capacity", 1.0))}
        edge_records.append(record)
        adjacency[src].append(dst)
        reverse_adjacency[dst].append(src)
    for values in adjacency.values():
        values.sort()
    for values in reverse_adjacency.values():
        values.sort()
    compute = [r for r in by_id.values() if r.get("resource_type") in {"PE", "FU", "DataPath"}]
    routing = [r for r in by_id.values() if r.get("resource_type") not in {"PE", "FU", "DataPath"}]
    return {
        "format": "flowadvantage_morpher_mrrg_v1",
        "ii": ii,
        "architecture_hash": payload.get("architecture_hash"),
        "resources": by_id,
        "edges": edge_records,
        "adjacency": adjacency,
        "reverse_adjacency": reverse_adjacency,
        "compute_resource_ids": sorted(r["native_resource_id"] for r in compute),
        "routing_resource_ids": sorted(r["native_resource_id"] for r in routing),
        "metadata": dict(payload.get("metadata", {})),
        "raw": payload,
    }
