"""Independent legality checker for the native Morpher interchange contract.

This module intentionally shares no mapper state with Morpher.  It validates
only serialized DFG, II-expanded MRRG, and mapping documents.  Native Morpher
re-import remains a second, independently implemented checker.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations
from typing import Any, Iterable


def _node_key(record: dict[str, Any]) -> str:
    key = record.get("native_node_key")
    if key is not None:
        return str(key)
    return str(record.get("dfg_node_id"))


def _index_unique(
    items: Iterable[dict[str, Any]], key: str, violations: list[dict[str, Any]]
) -> dict[Any, dict[str, Any]]:
    result: dict[Any, dict[str, Any]] = {}
    for item in items:
        value = item.get(key)
        if value in result:
            # Native module traversal can encounter one port through multiple
            # parent modules. Identical records are a harmless serialization
            # duplicate; contradictory records are not.
            if result[value] != item:
                violations.append(
                    {"class": "conflicting duplicate native identifier", "id": value}
                )
        else:
            result[value] = item
    return result


def _placement_prefix(operation: dict[str, Any]) -> str:
    # Morpher PE full names repeat the PE token
    # (CGRA_Ins.PE_X1...PE_X1...).  The first portion is the stable owner path
    # shared by every FU/port in that PE.
    return str(operation.get("pe_id", "")).split(".PE_")[0]


def _mutex_pairs(dfg: dict[str, Any]) -> set[frozenset[str]]:
    pairs: set[frozenset[str]] = set()
    for record in dfg.get("mutex_basic_blocks", []):
        left, right = str(record.get("left", "")), str(record.get("right", ""))
        if left and right:
            pairs.add(frozenset((left, right)))
    return pairs


def validate_mapping(
    mapping: dict[str, Any], dfg: dict[str, Any], mrrg: dict[str, Any]
) -> dict[str, Any]:
    """Return a structured, exhaustive legality result.

    The checker validates stable identities (including duplicated legacy
    numeric IDs), placement compatibility/capacity, every routed physical
    dependency, directed route continuity, serialized route timing, native
    operand-mux and basic-block mutual exclusion rules, and MRRG conflict sets.
    """

    violations: list[dict[str, Any]] = []
    if mapping.get("schema") != "flowadvantage_morpher_mapping_v1":
        violations.append({"class": "schema mismatch"})
    if mapping.get("ii") != mrrg.get("ii") or mapping.get("ii") != dfg.get("ii"):
        violations.append({"class": "II mismatch"})
    if mapping.get("architecture_hash") != mrrg.get("architecture_hash"):
        violations.append({"class": "architecture-hash mismatch"})
    if mapping.get("dfg_hash") != dfg.get("dfg_hash"):
        violations.append({"class": "DFG-hash mismatch"})

    ii = int(mrrg.get("ii", 0) or 0)
    node_records = dfg.get("nodes", [])
    nodes = _index_unique(
        (dict(record, _stable_key=_node_key(record)) for record in node_records),
        "_stable_key",
        violations,
    )
    resources = _index_unique(
        mrrg.get("resources", []), "native_resource_id", violations
    )
    directed_edges = {
        (edge.get("src"), edge.get("dst")) for edge in mrrg.get("edges", [])
    }

    operations = mapping.get("operations", [])
    operations_by_node: dict[str, dict[str, Any]] = {}
    compute_occupancy: Counter[tuple[Any, Any, Any]] = Counter()
    for operation in operations:
        node_key = _node_key(operation)
        node = nodes.get(node_key)
        if node is None:
            violations.append({"class": "unknown DFG node", "node": node_key})
            continue
        if node_key in operations_by_node:
            violations.append(
                {"class": "duplicate compute occupancy", "node": node_key}
            )
            continue
        operations_by_node[node_key] = operation

        pe_id, fu_id = operation.get("pe_id"), operation.get("fu_id")
        pe, fu = resources.get(pe_id), resources.get(fu_id)
        if pe is None:
            violations.append({"class": "unknown MRRG resource", "resource": pe_id})
            continue
        if pe.get("resource_type") != "PE":
            violations.append({"class": "compute resource type", "resource": pe_id})
        if fu is None:
            violations.append({"class": "unknown MRRG resource", "resource": fu_id})
        else:
            if fu.get("resource_type") != "FU":
                violations.append(
                    {
                        "class": "unsupported operation",
                        "node": node_key,
                        "reason": "not_a_function_unit",
                        "fu": fu_id,
                    }
                )
            opcode = str(node.get("opcode", operation.get("opcode", ""))).upper()
            supported = {str(value).upper() for value in fu.get("supported_operations", [])}
            if supported and opcode not in supported:
                violations.append(
                    {
                        "class": "unsupported operation",
                        "node": node_key,
                        "opcode": opcode,
                        "fu": fu_id,
                    }
                )
            if opcode.startswith(("LOAD", "STORE")) and fu.get("memory_role") != "memory":
                violations.append(
                    {
                        "class": "memory-port violation",
                        "node": node_key,
                        "opcode": opcode,
                        "fu": fu_id,
                    }
                )

        modulo_time = operation.get("modulo_time")
        if not isinstance(modulo_time, int) or not 0 <= modulo_time < ii:
            violations.append({"class": "timing violation", "node": node_key})
        if pe.get("time_slot") != modulo_time:
            violations.append({"class": "timing violation", "node": node_key})
        latency = operation.get("latency")
        if isinstance(latency, int) and ii and latency % ii != modulo_time:
            violations.append(
                {
                    "class": "timing violation",
                    "node": node_key,
                    "latency": latency,
                    "modulo_time": modulo_time,
                }
            )
        absolute = operation.get("absolute_schedule_if_available")
        if absolute is not None:
            asap = int(node.get("asap", operation.get("asap", 0)))
            alap = int(node.get("alap", operation.get("alap", asap)))
            if not asap <= int(absolute) <= alap:
                violations.append(
                    {
                        "class": "ASAP/ALAP violation",
                        "node": node_key,
                        "absolute_schedule": absolute,
                        "asap": asap,
                        "alap": alap,
                    }
                )
        compute_occupancy[(pe_id, fu_id, modulo_time)] += 1

    for slot, count in compute_occupancy.items():
        capacity = int(resources.get(slot[1], {}).get("capacity", 1))
        if count > capacity:
            violations.append(
                {
                    "class": "duplicate compute occupancy",
                    "slot": list(slot),
                    "uses": count,
                    "capacity": capacity,
                }
            )
    missing_operations = sorted(set(nodes) - set(operations_by_node))
    if missing_operations:
        violations.append(
            {"class": "unconsumed operation", "nodes": missing_operations}
        )

    dependencies: dict[tuple[str, str], dict[str, Any]] = {}
    required_dependencies: set[tuple[str, str]] = set()
    for dependency in dfg.get("dependencies", []):
        pair = (
            str(
                dependency.get(
                    "source_node_key", dependency.get("source_node")
                )
            ),
            str(
                dependency.get(
                    "destination_node_key", dependency.get("destination_node")
                )
            ),
        )
        if pair in dependencies:
            violations.append({"class": "duplicate dependency", "edge": list(pair)})
        dependencies[pair] = dependency
        if bool(
            dependency.get(
                "requires_route", dependency.get("edge_type", "data") != "PS"
            )
        ):
            required_dependencies.add(pair)

    routed: set[tuple[str, str]] = set()
    resource_signals: dict[str, set[str]] = defaultdict(set)
    for route in mapping.get("routes", []):
        pair = (
            str(route.get("source_node_key", route.get("source_node"))),
            str(route.get("destination_node_key", route.get("destination_node"))),
        )
        if pair not in required_dependencies:
            violations.append(
                {"class": "wrong route endpoints", "edge": route.get("edge_id")}
            )
            continue
        if pair in routed:
            violations.append({"class": "duplicate route", "edge": route.get("edge_id")})
            continue
        routed.add(pair)
        if route.get("export_status", "ok") != "ok":
            violations.append(
                {
                    "class": "non-contiguous route",
                    "edge": route.get("edge_id"),
                    "export_status": route.get("export_status"),
                }
            )
            continue

        resource_ids = route.get("ordered_resource_ids", [])
        latencies = route.get("ordered_resource_latencies", [])
        if not resource_ids:
            violations.append(
                {"class": "wrong route endpoints", "edge": route.get("edge_id")}
            )
            continue
        if latencies and len(latencies) != len(resource_ids):
            violations.append(
                {"class": "route timing violation", "edge": route.get("edge_id")}
            )
        for resource_id in resource_ids:
            if resource_id not in resources:
                violations.append(
                    {"class": "unknown MRRG resource", "resource": resource_id}
                )
            resource_signals[resource_id].add(pair[0])
        for source_resource, destination_resource in zip(
            resource_ids, resource_ids[1:]
        ):
            if (source_resource, destination_resource) not in directed_edges:
                violations.append(
                    {
                        "class": "non-contiguous route",
                        "link": f"{source_resource}->{destination_resource}",
                    }
                )
        if latencies:
            if route.get("start_time") != latencies[0] or route.get("end_time") != latencies[-1]:
                violations.append(
                    {"class": "route timing violation", "edge": route.get("edge_id")}
                )
            if any(int(right) < int(left) for left, right in zip(latencies, latencies[1:])):
                violations.append(
                    {"class": "route timing violation", "edge": route.get("edge_id")}
                )
            for resource_id, latency_value in zip(resource_ids, latencies):
                phase = resources.get(resource_id, {}).get("time_slot")
                if isinstance(phase, int) and phase >= 0 and ii and int(latency_value) % ii != phase:
                    violations.append(
                        {
                            "class": "route timing violation",
                            "edge": route.get("edge_id"),
                            "resource": resource_id,
                            "latency": latency_value,
                            "time_slot": phase,
                        }
                    )
            destination_node = nodes.get(pair[1], {})
            dependency = dependencies[pair]
            iteration_distance = int(dependency.get("iteration_distance", 0) or 0)
            destination_operation = operations_by_node.get(pair[1])
            if iteration_distance and destination_operation is not None:
                destination_latency = destination_operation.get("latency")
                if (
                    isinstance(destination_latency, int)
                    and max(map(int, latencies))
                    > destination_latency + iteration_distance * ii
                ):
                    violations.append(
                        {
                            "class": "recurrence violation",
                            "edge": route.get("edge_id"),
                            "iteration_distance": iteration_distance,
                            "maximum_route_latency": max(map(int, latencies)),
                            "maximum_legal_latency": (
                                destination_latency + iteration_distance * ii
                            ),
                        }
                    )
            for recurrence_parent_key in destination_node.get(
                "recurrence_parent_keys", []
            ):
                recurrence_parent = operations_by_node.get(str(recurrence_parent_key))
                if recurrence_parent is None:
                    violations.append(
                        {
                            "class": "recurrence violation",
                            "edge": route.get("edge_id"),
                            "reason": "unplaced_recurrence_parent",
                            "recurrence_parent": recurrence_parent_key,
                        }
                    )
                    continue
                parent_latency = recurrence_parent.get("latency")
                if (
                    isinstance(parent_latency, int)
                    and max(map(int, latencies)) > parent_latency + ii
                ):
                    violations.append(
                        {
                            "class": "recurrence violation",
                            "edge": route.get("edge_id"),
                            "recurrence_parent": recurrence_parent_key,
                            "maximum_route_latency": max(map(int, latencies)),
                            "maximum_legal_latency": parent_latency + ii,
                        }
                    )
        source_operation = operations_by_node.get(pair[0])
        destination_operation = operations_by_node.get(pair[1])
        if source_operation is None or destination_operation is None:
            violations.append(
                {"class": "wrong route endpoints", "edge": route.get("edge_id")}
            )
        else:
            source_prefix = _placement_prefix(source_operation)
            destination_prefix = _placement_prefix(destination_operation)
            if source_prefix and source_prefix not in str(resource_ids[0]):
                violations.append(
                    {
                        "class": "wrong route endpoints",
                        "edge": route.get("edge_id"),
                        "end": "source",
                    }
                )
            if destination_prefix and destination_prefix not in str(resource_ids[-1]):
                violations.append(
                    {
                        "class": "wrong route endpoints",
                        "edge": route.get("edge_id"),
                        "end": "destination",
                    }
                )

    for pair in sorted(required_dependencies - routed):
        violations.append({"class": "missing route", "edge": list(pair)})

    mutex_pairs = _mutex_pairs(dfg)
    node_bb = {key: str(value.get("bb", "")) for key, value in nodes.items()}

    def signals_compatible(resource: dict[str, Any], signals: set[str]) -> bool:
        if len(signals) <= int(resource.get("capacity", 1)):
            return True
        if resource.get("allows_operand_mux"):
            return True
        return all(
            frozenset((node_bb.get(left, ""), node_bb.get(right, ""))) in mutex_pairs
            for left, right in combinations(sorted(signals), 2)
        )

    for resource_id, signals in resource_signals.items():
        resource = resources.get(resource_id, {})
        if not signals_compatible(resource, signals):
            violations.append(
                {
                    "class": "routing capacity conflict",
                    "resource": resource_id,
                    "signals": sorted(signals),
                    "capacity": resource.get("capacity", 1),
                }
            )
    checked_conflicts: set[tuple[str, str]] = set()
    for resource_id, resource in resources.items():
        for other_id in resource.get("conflicting_resource_ids", []):
            pair = tuple(sorted((str(resource_id), str(other_id))))
            if pair in checked_conflicts:
                continue
            checked_conflicts.add(pair)
            left_signals = resource_signals.get(pair[0], set())
            right_signals = resource_signals.get(pair[1], set())
            if not left_signals or not right_signals:
                continue
            combined = left_signals | right_signals
            if not signals_compatible(resource, combined):
                violations.append(
                    {
                        "class": "routing conflict-set violation",
                        "resources": list(pair),
                        "signals": sorted(combined),
                    }
                )

    return {
        "legal": not violations,
        "violations": violations,
        "counts": {
            "operations": len(operations),
            "dependencies": len(dependencies),
            "routed_dependencies": len(required_dependencies),
            "pseudo_dependencies": len(dependencies) - len(required_dependencies),
            "routes": len(mapping.get("routes", [])),
        },
    }


def validate_imported_mapping(
    mapping: dict[str, Any], dfg: dict[str, Any], arch: dict[str, Any]
) -> dict[str, Any]:
    """Compatibility wrapper; canonical calls must supply native MRRG JSON."""

    if "resources" in arch:
        return validate_mapping(mapping, dfg, arch)
    ids = set(dfg.get("nodes", {}))
    assigned = {
        int(key)
        for key in mapping.get("assignments", {})
        if str(key).lstrip("-").isdigit()
    }
    legal = assigned.issubset(ids)
    return {
        "legal": legal,
        "violations": [] if legal else [{"class": "unknown DFG node"}],
    }
