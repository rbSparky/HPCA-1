"""Exact Morpher-native mapping primitives used by the real pilot.

This module deliberately operates on Morpher's exported native identifiers.
It does not reconstruct an MRRG from coordinates and it does not translate a
native route into the synthetic QuotientFlow graph.  The objects here are the
contract boundary for a real mapper:

* placement candidates are concrete DataPath/FU/PE/modulo-time assignments;
* routes are directed native-port paths ending at the dependency operand port;
* occupancy implements Morpher's broadcast, operand-mux, basic-block mutex,
  and conflict-port rules; and
* serialization emits the replayable native mapping contract, including the
  complete ``port_state`` required by Morpher configuration generation.

Scheduling is intentionally explicit.  Morpher's ASAP/ALAP values are ordering
information rather than hard absolute windows in its PathFinder mapper.  A
caller must provide a finite absolute latency interval when enumerating
placements; the API never silently invents a scheduling horizon.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from heapq import heappop, heappush
from itertools import combinations
from typing import Any, Iterable, Mapping, Sequence

from .operation_taxonomy import translate_operation


def _node_key(record: Mapping[str, Any]) -> str:
    return str(record.get("native_node_key", record.get("dfg_node_id")))


def _resource_id(record: Mapping[str, Any]) -> str:
    return str(record["native_resource_id"])


def _container(native_id: str) -> str:
    return native_id.rsplit(".", 1)[0]


def _dependency_key(dependency: Mapping[str, Any]) -> tuple[str, str]:
    return (
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


@dataclass(frozen=True, order=True)
class NativeSignal:
    """One native value occupying a port.

    Morpher counts distinct *source* DFG values for physical capacity.  The
    destination and latency remain part of the serialized signal because they
    select fanout/operand configuration and are required for native replay.
    """

    source_key: str
    destination_node: int
    latency: int
    source_node: int

    def as_json(self) -> dict[str, Any]:
        return {
            "source_node": self.source_node,
            "source_node_key": self.source_key,
            "destination_node": self.destination_node,
            "latency": self.latency,
        }


@dataclass(frozen=True)
class NativePlacement:
    node_key: str
    node_id: int
    opcode: str
    pe_id: str
    fu_id: str
    dp_id: str
    modulo_time: int
    latency: int
    operation_latency: int
    pe_x: int | None = None
    pe_y: int | None = None

    @property
    def output_latency(self) -> int:
        return self.latency + self.operation_latency


@dataclass(frozen=True)
class NativeRoute:
    source_key: str
    destination_key: str
    source_node: int
    destination_node: int
    edge_type: str
    resource_ids: tuple[str, ...]
    resource_latencies: tuple[int, ...]

    @property
    def edge_id(self) -> str:
        return f"{self.source_key}->{self.destination_key}"

    def as_json(self) -> dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "source_node": self.source_node,
            "destination_node": self.destination_node,
            "source_node_key": self.source_key,
            "destination_node_key": self.destination_key,
            "ordered_resource_ids": list(self.resource_ids),
            "ordered_resource_latencies": list(self.resource_latencies),
            "ordered_link_ids": [
                f"{left}->{right}"
                for left, right in zip(self.resource_ids, self.resource_ids[1:])
            ],
            "start_time": self.resource_latencies[0],
            "end_time": self.resource_latencies[-1],
            "export_status": "ok",
        }


@dataclass(frozen=True)
class NativeAction:
    placement: NativePlacement
    routes: tuple[NativeRoute, ...]

    def stable_key(self) -> tuple[Any, ...]:
        return (
            self.placement.node_key,
            self.placement.latency,
            self.placement.dp_id,
            tuple(
                (
                    route.source_key,
                    route.destination_key,
                    route.resource_ids,
                    route.resource_latencies,
                )
                for route in self.routes
            ),
        )


class NativeContractError(ValueError):
    """The exported native contract is missing semantics needed for mapping."""


class NativeMorpherProblem:
    """Indexed native DFG/MRRG with exact placement and route endpoints."""

    def __init__(
        self,
        dfg: Mapping[str, Any],
        mrrg: Mapping[str, Any],
        *,
        operation_latencies: Mapping[tuple[str, str], int] | None = None,
        reference_mapping: Mapping[str, Any] | None = None,
    ) -> None:
        if dfg.get("schema") != "flowadvantage_morpher_dfg_v1":
            raise NativeContractError("DFG must use flowadvantage_morpher_dfg_v1")
        if mrrg.get("schema") != "flowadvantage_morpher_mrrg_v1":
            raise NativeContractError("MRRG must use flowadvantage_morpher_mrrg_v1")
        if dfg.get("ii") != mrrg.get("ii"):
            raise NativeContractError("DFG and MRRG II differ")
        self.dfg = dict(dfg)
        self.mrrg = dict(mrrg)
        self.ii = int(mrrg["ii"])
        self.nodes: dict[str, dict[str, Any]] = {}
        for raw in dfg.get("nodes", []):
            record = dict(raw)
            key = _node_key(record)
            if key in self.nodes:
                raise NativeContractError(f"duplicate stable DFG node key {key}")
            self.nodes[key] = record
        self.dependencies: dict[tuple[str, str], dict[str, Any]] = {}
        self.incoming: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.outgoing: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for raw in dfg.get("dependencies", []):
            record = dict(raw)
            key = _dependency_key(record)
            if key in self.dependencies:
                raise NativeContractError(f"duplicate DFG dependency {key}")
            if key[0] not in self.nodes or key[1] not in self.nodes:
                raise NativeContractError(f"dependency references unknown node {key}")
            self.dependencies[key] = record
            self.incoming[key[1]].append(record)
            self.outgoing[key[0]].append(record)
        for values in (*self.incoming.values(), *self.outgoing.values()):
            values.sort(key=_dependency_key)

        self.resources: dict[str, dict[str, Any]] = {}
        for raw in mrrg.get("resources", []):
            record = dict(raw)
            rid = _resource_id(record)
            previous = self.resources.get(rid)
            if previous is not None and previous != record:
                raise NativeContractError(f"conflicting MRRG resource {rid}")
            self.resources[rid] = record
        self.adjacency: dict[str, tuple[str, ...]] = {
            rid: () for rid in self.resources
        }
        mutable_adjacency: dict[str, set[str]] = defaultdict(set)
        for edge in mrrg.get("edges", []):
            src, dst = str(edge.get("src")), str(edge.get("dst"))
            if src not in self.resources or dst not in self.resources:
                raise NativeContractError(f"edge references unknown resource {src}->{dst}")
            mutable_adjacency[src].add(dst)
        self.adjacency = {
            rid: tuple(sorted(mutable_adjacency.get(rid, ())))
            for rid in self.resources
        }
        self.conflicts: dict[str, frozenset[str]] = {}
        for rid, resource in self.resources.items():
            conflicting = {
                str(other)
                for other in resource.get("conflicting_resource_ids", [])
                if str(other) in self.resources
            }
            self.conflicts[rid] = frozenset(conflicting)
        # Native conflict declarations are semantically undirected even if a
        # module exporter emits only one direction.
        symmetric = {rid: set(values) for rid, values in self.conflicts.items()}
        for rid, values in self.conflicts.items():
            for other in values:
                symmetric[other].add(rid)
        self.conflicts = {rid: frozenset(values) for rid, values in symmetric.items()}

        self.pe_by_container: dict[str, dict[str, Any]] = {}
        self.fu_by_container: dict[str, dict[str, Any]] = {}
        self.dp_by_container: dict[str, dict[str, Any]] = {}
        for resource in self.resources.values():
            rid, kind = _resource_id(resource), resource.get("resource_type")
            if kind == "PE":
                self.pe_by_container[_container(rid)] = resource
            elif kind == "FU":
                self.fu_by_container[_container(rid)] = resource
            elif kind == "DP":
                self.dp_by_container[_container(rid)] = resource
        self.dp_records: list[dict[str, Any]] = []
        self.fu_operation_latencies: dict[tuple[str, str], int] = {}
        for resource in self.resources.values():
            if resource.get("resource_type") != "FU":
                continue
            fu_id = _resource_id(resource)
            raw_latencies = resource.get("operation_latencies", {})
            if raw_latencies is None:
                raw_latencies = {}
            if not isinstance(raw_latencies, Mapping):
                raise NativeContractError(
                    f"FU operation_latencies must be an object: {fu_id}"
                )
            for opcode, latency in raw_latencies.items():
                value = int(latency)
                if value < 0:
                    raise NativeContractError(
                        f"negative FU operation latency: {fu_id}:{opcode}"
                    )
                self.fu_operation_latencies[(fu_id, str(opcode).upper())] = value
        for dp_container, dp in sorted(self.dp_by_container.items()):
            fu_container = _container(dp_container)
            pe_container = _container(fu_container)
            fu, pe = self.fu_by_container.get(fu_container), self.pe_by_container.get(
                pe_container
            )
            if fu is None or pe is None:
                raise NativeContractError(
                    f"cannot resolve native DataPath ownership for {dp_container}"
                )
            output_port = f"{dp_container}.T"
            if output_port not in self.resources:
                raise NativeContractError(
                    f"DataPath has no native T output port: {dp_container}"
                )
            self.dp_records.append(
                {
                    "dp_id": _resource_id(dp),
                    "dp_container": dp_container,
                    "fu_id": _resource_id(fu),
                    "pe_id": _resource_id(pe),
                    "x": pe.get("x"),
                    "y": pe.get("y"),
                    "time_slot": int(pe.get("time_slot", -1)),
                    "memory_role": fu.get("memory_role"),
                    "supported_operations": tuple(
                        sorted(str(op).upper() for op in fu.get("supported_operations", []))
                    ),
                    "output_port": output_port,
                    "dp_signature": ".".join(dp_container.split(".")[-2:]),
                }
            )
        self.dp_records.sort(
            key=lambda item: (
                item["time_slot"],
                item["x"] if item["x"] is not None else -1,
                item["y"] if item["y"] is not None else -1,
                item["dp_id"],
            )
        )
        self.dp_by_id = {item["dp_id"]: item for item in self.dp_records}

        self.mutex_pairs = {
            frozenset((str(pair.get("left")), str(pair.get("right"))))
            for pair in dfg.get("mutex_basic_blocks", [])
            if pair.get("left") and pair.get("right")
        }
        explicit = {
            (str(role), str(opcode).upper()): int(value)
            for (role, opcode), value in (operation_latencies or {}).items()
        }
        inferred = (
            self._infer_operation_latencies(reference_mapping)
            if reference_mapping is not None
            else {}
        )
        inferred.update(explicit)
        self.operation_latencies = inferred
        if reference_mapping is None:
            missing_latency_metadata = []
            for dp in self.dp_records:
                for opcode in dp["supported_operations"]:
                    if (
                        (dp["fu_id"], opcode) not in self.fu_operation_latencies
                        and (str(dp["memory_role"] or "compute"), opcode)
                        not in self.operation_latencies
                    ):
                        missing_latency_metadata.append((dp["fu_id"], opcode))
            if missing_latency_metadata:
                first = missing_latency_metadata[0]
                raise NativeContractError(
                    "native FU operation-latency metadata is incomplete "
                    f"({len(missing_latency_metadata)} missing; first={first}). "
                    "Production mapping must use a current native dump. "
                    "Reference-witness fallback is validation-only."
                )

    @classmethod
    def from_native_documents(
        cls,
        dfg: Mapping[str, Any],
        mrrg: Mapping[str, Any],
        **kwargs: Any,
    ) -> "NativeMorpherProblem":
        """Construct from raw JSON payloads (not the compatibility import views)."""

        return cls(dfg, mrrg, **kwargs)

    def _infer_operation_latencies(
        self, reference_mapping: Mapping[str, Any]
    ) -> dict[tuple[str, str], int]:
        """Infer exact role/opcode latency from a native mapped reference.

        This is not a guessed default.  Every inferred value is witnessed by a
        native route start and conflicting witnesses reject the contract.
        Future native dumps can provide ``operation_latencies`` on FU resources,
        which callers may pass directly through ``operation_latencies``.
        """

        operations = {
            _node_key(operation): operation
            for operation in reference_mapping.get("operations", [])
        }
        witnesses: dict[tuple[str, str], set[int]] = defaultdict(set)
        for route in reference_mapping.get("routes", []):
            source_key = str(
                route.get("source_node_key", route.get("source_node"))
            )
            operation = operations.get(source_key)
            if operation is None or route.get("start_time") is None:
                continue
            fu = self.resources.get(str(operation.get("fu_id")), {})
            role = str(fu.get("memory_role", "compute"))
            opcode = str(operation.get("opcode", "")).upper()
            value = int(route["start_time"]) - int(operation["latency"])
            if value < 0:
                raise NativeContractError(
                    f"negative native operation latency for {source_key}"
                )
            witnesses[(role, opcode)].add(value)
        # Sink operations have no outgoing DFG edge, but DataPath::assignNode
        # still records a self-signal on the operation's output T port.  That
        # native witness is equally authoritative.
        by_key = {
            _node_key(operation): operation
            for operation in reference_mapping.get("operations", [])
        }
        for state in reference_mapping.get("port_state", []):
            for signal in state.get("signals", []):
                source_key = str(
                    signal.get("source_node_key", signal.get("source_node"))
                )
                operation = by_key.get(source_key)
                if (
                    operation is None
                    or int(signal.get("destination_node", -1))
                    != int(operation.get("dfg_node_id", -2))
                ):
                    continue
                fu = self.resources.get(str(operation.get("fu_id")), {})
                role = str(fu.get("memory_role", "compute"))
                opcode = str(operation.get("opcode", "")).upper()
                value = int(signal["latency"]) - int(operation["latency"])
                if value >= 0:
                    witnesses[(role, opcode)].add(value)
        result: dict[tuple[str, str], int] = {}
        for key, values in witnesses.items():
            if len(values) != 1:
                raise NativeContractError(
                    f"conflicting native operation-latency witnesses {key}: {values}"
                )
            result[key] = next(iter(values))
        return result

    def operation_latency(
        self,
        opcode: str,
        memory_role: str | None,
        *,
        fu_id: str | None = None,
    ) -> int:
        if fu_id is not None:
            exact_key = (str(fu_id), str(opcode).upper())
            if exact_key in self.fu_operation_latencies:
                return self.fu_operation_latencies[exact_key]
        key = (str(memory_role or "compute"), str(opcode).upper())
        if key not in self.operation_latencies:
            raise NativeContractError(
                "native operation latency is unavailable for "
                f"role={key[0]!r}, opcode={key[1]!r}; provide an exported "
                "FU latency table or a native reference witness"
            )
        return self.operation_latencies[key]

    def operation_order(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                self.nodes,
                key=lambda key: (
                    int(self.nodes[key].get("asap", 0)),
                    -len(self.outgoing.get(key, ())),
                    key,
                ),
            )
        )

    def _corresponding_dp(
        self, candidate: Mapping[str, Any], target_phase: int
    ) -> dict[str, Any]:
        matches = [
            record
            for record in self.dp_records
            if record["x"] == candidate["x"]
            and record["y"] == candidate["y"]
            and record["time_slot"] == target_phase
            and record["dp_signature"] == candidate["dp_signature"]
            and record["memory_role"] == candidate["memory_role"]
        ]
        if len(matches) != 1:
            raise NativeContractError(
                "native time-expanded DataPath correspondence is ambiguous for "
                f"{candidate['dp_id']} at phase {target_phase}: {len(matches)} matches"
            )
        return matches[0]

    def placement_candidates(
        self,
        node_key: str,
        *,
        earliest_latency: int,
        latest_latency: int,
        state: "NativeMappingState | None" = None,
    ) -> tuple[NativePlacement, ...]:
        """Enumerate every legal concrete placement in an explicit time window."""

        if earliest_latency > latest_latency:
            raise ValueError("earliest_latency exceeds latest_latency")
        node = self.nodes[node_key]
        opcode = str(node.get("opcode", "")).upper()
        candidates: list[NativePlacement] = []
        for dp in self.dp_records:
            if opcode not in dp["supported_operations"]:
                continue
            if state is not None and state.compute_occupied(dp["dp_id"]):
                continue
            op_latency = self.operation_latency(
                opcode, dp["memory_role"], fu_id=dp["fu_id"]
            )
            for latency in range(earliest_latency, latest_latency + 1):
                if latency % self.ii != dp["time_slot"]:
                    continue
                candidates.append(
                    NativePlacement(
                        node_key=node_key,
                        node_id=int(node["dfg_node_id"]),
                        opcode=opcode,
                        pe_id=dp["pe_id"],
                        fu_id=dp["fu_id"],
                        dp_id=dp["dp_id"],
                        modulo_time=dp["time_slot"],
                        latency=latency,
                        operation_latency=op_latency,
                        pe_x=dp["x"],
                        pe_y=dp["y"],
                    )
                )
        return tuple(
            sorted(
                candidates,
                key=lambda item: (
                    item.latency,
                    item.pe_x if item.pe_x is not None else -1,
                    item.pe_y if item.pe_y is not None else -1,
                    item.dp_id,
                ),
            )
        )

    def output_port(self, placement: NativePlacement) -> str:
        base = self.dp_by_id[placement.dp_id]
        future = self._corresponding_dp(
            base, placement.output_latency % self.ii
        )
        return str(future["output_port"])

    def operand_port(self, placement: NativePlacement, edge_type: str) -> str:
        operand = str(edge_type).upper()
        if operand not in {"I1", "I2", "P"}:
            raise NativeContractError(
                f"dependency operand {operand!r} has no physical Morpher input port"
            )
        dp_container = self.dp_by_id[placement.dp_id]["dp_container"]
        target = f"{dp_container}.{operand}"
        if target not in self.resources:
            raise NativeContractError(
                f"native operand port {target!r} is absent from the MRRG"
            )
        return target

    def path_latencies(
        self, resource_ids: Sequence[str], start_latency: int
    ) -> tuple[int, ...]:
        if not resource_ids:
            raise ValueError("route cannot be empty")
        latency = int(start_latency)
        first_phase = self.resources[resource_ids[0]].get("time_slot")
        if isinstance(first_phase, int) and latency % self.ii != first_phase:
            raise NativeContractError("route start latency does not match MRRG phase")
        result = [latency]
        for destination in resource_ids[1:]:
            phase = self.resources[destination].get("time_slot")
            if isinstance(phase, int) and phase >= 0:
                latency += (phase - latency) % self.ii
            result.append(latency)
        return tuple(result)

    def enumerate_routes(
        self,
        dependency: Mapping[str, Any],
        source: NativePlacement,
        destination: NativePlacement,
        state: "NativeMappingState",
        *,
        k: int = 4,
        max_expansions: int = 200_000,
    ) -> tuple[NativeRoute, ...]:
        """Enumerate deterministic shortest legal native-port routes.

        The search state includes absolute latency.  A route is accepted only
        when it reaches the exact consumer latency (plus the declared
        loop-carried iteration distance), rather than merely a matching modulo
        phase.
        """

        if k <= 0:
            return ()
        edge_type = str(dependency.get("edge_type", "")).upper()
        if not bool(dependency.get("requires_route", edge_type != "PS")):
            return ()
        source_port = self.output_port(source)
        destination_port = self.operand_port(destination, edge_type)
        start_latency = source.output_latency
        deadline = destination.latency + int(
            dependency.get("iteration_distance", 0) or 0
        ) * self.ii
        if start_latency > deadline:
            return ()
        source_key, destination_key = _dependency_key(dependency)
        destination_node = int(self.nodes[destination_key]["dfg_node_id"])
        signal = NativeSignal(
            source_key=source_key,
            destination_node=destination_node,
            latency=start_latency,
            source_node=int(self.nodes[source_key]["dfg_node_id"]),
        )
        first = self._shortest_temporal_path(
            source_port,
            destination_port,
            start_latency,
            deadline,
            state,
            signal,
            max_expansions=max_expansions,
        )
        if first is None:
            return ()
        accepted: list[tuple[tuple[str, ...], tuple[int, ...]]] = [first]
        candidate_heap: list[
            tuple[int, tuple[str, ...], tuple[int, ...]]
        ] = []
        candidate_seen: set[tuple[str, ...]] = {first[0]}
        # Deterministic Yen enumeration avoids exponential path-prefix copies
        # from the former uniform-cost simple-path search.  Every spur search
        # is a linear BFS over (native resource, absolute latency) states.
        while len(accepted) < k:
            previous_path, previous_latencies = accepted[-1]
            for spur_index in range(len(previous_path) - 1):
                root_path = previous_path[: spur_index + 1]
                root_latencies = previous_latencies[: spur_index + 1]
                blocked_edges = set()
                for accepted_path, _ in accepted:
                    if accepted_path[: spur_index + 1] == root_path:
                        blocked_edges.add(
                            (
                                accepted_path[spur_index],
                                accepted_path[spur_index + 1],
                            )
                        )
                spur = self._shortest_temporal_path(
                    root_path[-1],
                    destination_port,
                    root_latencies[-1],
                    deadline,
                    state,
                    signal,
                    blocked_nodes=frozenset(root_path[:-1]),
                    blocked_edges=frozenset(blocked_edges),
                    max_expansions=max_expansions,
                )
                if spur is None:
                    continue
                spur_path, spur_latencies = spur
                path = root_path[:-1] + spur_path
                latencies = root_latencies[:-1] + spur_latencies
                if len(path) != len(set(path)) or path in candidate_seen:
                    continue
                candidate_seen.add(path)
                heappush(candidate_heap, (len(path), path, latencies))
            if not candidate_heap:
                break
            _, path, latencies = heappop(candidate_heap)
            accepted.append((path, latencies))
        return tuple(
            NativeRoute(
                source_key=source_key,
                destination_key=destination_key,
                source_node=int(self.nodes[source_key]["dfg_node_id"]),
                destination_node=destination_node,
                edge_type=edge_type,
                resource_ids=path,
                resource_latencies=latencies,
            )
            for path, latencies in accepted
        )

    def _shortest_temporal_path(
        self,
        source_port: str,
        destination_port: str,
        start_latency: int,
        deadline: int,
        state: "NativeMappingState",
        signal: NativeSignal,
        *,
        blocked_nodes: frozenset[str] = frozenset(),
        blocked_edges: frozenset[tuple[str, str]] = frozenset(),
        max_expansions: int,
    ) -> tuple[tuple[str, ...], tuple[int, ...]] | None:
        start = (source_port, start_latency)
        queue = deque((start,))
        previous: dict[
            tuple[str, int], tuple[str, int] | None
        ] = {start: None}
        expansions = 0
        goal: tuple[str, int] | None = None
        while queue and expansions < max_expansions:
            current, current_latency = queue.popleft()
            if current == destination_port and current_latency == deadline:
                goal = (current, current_latency)
                break
            expansions += 1
            for nxt in self.adjacency.get(current, ()):
                if nxt in blocked_nodes or (current, nxt) in blocked_edges:
                    continue
                phase = self.resources[nxt].get("time_slot")
                next_latency = current_latency
                if isinstance(phase, int) and phase >= 0:
                    next_latency += (phase - current_latency) % self.ii
                if next_latency > deadline:
                    continue
                next_state = (nxt, next_latency)
                if next_state in previous:
                    continue
                # Prevent a modulo-cycle from becoming the first predecessor
                # chain to a temporal state.  Native routes are simple in the
                # II-expanded resource graph.
                ancestor: tuple[str, int] | None = (current, current_latency)
                repeats_resource = False
                while ancestor is not None:
                    if ancestor[0] == nxt:
                        repeats_resource = True
                        break
                    ancestor = previous[ancestor]
                if repeats_resource:
                    continue
                next_signal = NativeSignal(
                    source_key=signal.source_key,
                    destination_node=signal.destination_node,
                    latency=next_latency,
                    source_node=signal.source_node,
                )
                if not state.can_occupy(nxt, next_signal):
                    continue
                previous[next_state] = (current, current_latency)
                queue.append(next_state)
        if goal is None:
            return None
        states = []
        cursor: tuple[str, int] | None = goal
        while cursor is not None:
            states.append(cursor)
            cursor = previous[cursor]
        states.reverse()
        resources = tuple(resource for resource, _ in states)
        if len(resources) != len(set(resources)):
            # Native replay treats one II-expanded port as one capacity
            # resource.  A repeated resource is a routing cycle, not a legal
            # wait implementation.
            return None
        return resources, tuple(latency for _, latency in states)

    def action_for_placement(
        self,
        placement: NativePlacement,
        state: "NativeMappingState",
        *,
        k_paths: int = 4,
    ) -> NativeAction | None:
        """Route all newly completed incoming dependencies deterministically."""

        actions = self.actions_for_placement(
            placement,
            state,
            k_paths=k_paths,
            max_route_combinations=1,
        )
        return actions[0] if actions else None

    def actions_for_placement(
        self,
        placement: NativePlacement,
        state: "NativeMappingState",
        *,
        k_paths: int = 4,
        max_route_combinations: int | None = None,
        max_route_expansions: int = 1_000_000,
    ) -> tuple[NativeAction, ...]:
        """Enumerate routed actions for one target without child-value censoring.

        Every newly enabled physical dependency is routed.  This includes
        incoming dependencies whose source was already placed and outgoing
        dependencies to an already placed consumer (for recurrence-aware or
        otherwise non-topological stable orders).  Route combinations are
        generated in deterministic lexicographic path order.  A finite
        ``max_route_combinations`` is an explicit search-budget bound, never
        described as an exhaustive action universe.
        """

        initial = state.copy()
        if not initial.place(placement):
            return ()
        enabled: list[dict[str, Any]] = []
        for dependency in self.incoming.get(placement.node_key, ()):
            source_key, destination_key = _dependency_key(dependency)
            if (
                bool(
                    dependency.get(
                        "requires_route", dependency.get("edge_type") != "PS"
                    )
                )
                and source_key in initial.placements
                and (source_key, destination_key) not in initial.routes
            ):
                enabled.append(dependency)
        for dependency in self.outgoing.get(placement.node_key, ()):
            source_key, destination_key = _dependency_key(dependency)
            if (
                bool(
                    dependency.get(
                        "requires_route", dependency.get("edge_type") != "PS"
                    )
                )
                and destination_key in initial.placements
                and (source_key, destination_key) not in initial.routes
            ):
                enabled.append(dependency)
        enabled.sort(key=_dependency_key)
        frontier: list[tuple[NativeMappingState, tuple[NativeRoute, ...]]] = [
            (initial, ())
        ]
        for dependency in enabled:
            next_frontier: list[
                tuple[NativeMappingState, tuple[NativeRoute, ...]]
            ] = []
            source_key, destination_key = _dependency_key(dependency)
            for partial, chosen in frontier:
                source = partial.placements[source_key]
                destination = partial.placements[destination_key]
                alternatives = self.enumerate_routes(
                    dependency,
                    source,
                    destination,
                    partial,
                    k=k_paths,
                    max_expansions=max_route_expansions,
                )
                for route in alternatives:
                    updated = partial.copy()
                    if updated.add_route(route):
                        next_frontier.append((updated, chosen + (route,)))
            next_frontier.sort(
                key=lambda item: tuple(
                    (
                        route.source_key,
                        route.destination_key,
                        route.resource_ids,
                    )
                    for route in item[1]
                )
            )
            if max_route_combinations is not None:
                next_frontier = next_frontier[:max_route_combinations]
            frontier = next_frontier
            if not frontier:
                return ()
        actions = [
            NativeAction(placement=placement, routes=routes)
            for _, routes in frontier
        ]
        actions.sort(key=NativeAction.stable_key)
        if max_route_combinations is not None:
            actions = actions[:max_route_combinations]
        return tuple(actions)


@dataclass
class NativeMappingState:
    """Mutable partial native mapping with exact occupancy semantics."""

    problem: NativeMorpherProblem
    placements: dict[str, NativePlacement] = field(default_factory=dict)
    operation_records: dict[str, dict[str, Any]] = field(default_factory=dict)
    routes: dict[tuple[str, str], NativeRoute] = field(default_factory=dict)
    resource_signals: dict[str, set[NativeSignal]] = field(
        default_factory=lambda: defaultdict(set)
    )
    dp_occupancy: dict[str, str] = field(default_factory=dict)
    memory_bindings: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def copy(self) -> "NativeMappingState":
        return NativeMappingState(
            problem=self.problem,
            placements=dict(self.placements),
            operation_records={
                key: dict(value) for key, value in self.operation_records.items()
            },
            routes=dict(self.routes),
            resource_signals=defaultdict(
                set,
                {
                    key: set(values)
                    for key, values in self.resource_signals.items()
                },
            ),
            dp_occupancy=dict(self.dp_occupancy),
            memory_bindings=[dict(value) for value in self.memory_bindings],
            metadata=dict(self.metadata),
        )

    def compute_occupied(self, dp_id: str) -> bool:
        return dp_id in self.dp_occupancy

    def _source_mutex(self, left: str, right: str) -> bool:
        if left == right:
            return True
        left_bb = str(self.problem.nodes.get(left, {}).get("bb", ""))
        right_bb = str(self.problem.nodes.get(right, {}).get("bb", ""))
        return (
            bool(left_bb)
            and bool(right_bb)
            and frozenset((left_bb, right_bb)) in self.problem.mutex_pairs
        )

    def _signals_compatible(
        self,
        resource_id: str,
        existing: Iterable[NativeSignal],
        candidate: NativeSignal,
    ) -> bool:
        resource = self.problem.resources[resource_id]
        existing_sources = {signal.source_key for signal in existing}
        if candidate.source_key in existing_sources:
            return True  # native broadcast of one value
        if resource.get("allows_operand_mux"):
            return True
        return all(
            self._source_mutex(candidate.source_key, source)
            for source in existing_sources
        )

    def can_occupy(self, resource_id: str, signal: NativeSignal) -> bool:
        if resource_id not in self.problem.resources:
            return False
        if not self._signals_compatible(
            resource_id, self.resource_signals.get(resource_id, ()), signal
        ):
            return False
        for conflict in self.problem.conflicts.get(resource_id, ()):
            if not self._signals_compatible(
                resource_id, self.resource_signals.get(conflict, ()), signal
            ):
                return False
        return True

    def place(
        self,
        placement: NativePlacement,
        *,
        operation_record: Mapping[str, Any] | None = None,
        add_output_signal: bool = True,
    ) -> bool:
        if placement.node_key in self.placements:
            return False
        if placement.dp_id in self.dp_occupancy:
            return False
        node = self.problem.nodes.get(placement.node_key)
        dp = self.problem.dp_by_id.get(placement.dp_id)
        if node is None or dp is None:
            return False
        if placement.opcode not in dp["supported_operations"]:
            return False
        if placement.latency % self.problem.ii != placement.modulo_time:
            return False
        if placement.modulo_time != dp["time_slot"]:
            return False
        self.placements[placement.node_key] = placement
        self.dp_occupancy[placement.dp_id] = placement.node_key
        if operation_record is not None:
            self.operation_records[placement.node_key] = dict(operation_record)
        if add_output_signal:
            output = self.problem.output_port(placement)
            self_signal = NativeSignal(
                source_key=placement.node_key,
                destination_node=placement.node_id,
                latency=placement.output_latency,
                source_node=placement.node_id,
            )
            if not self.can_occupy(output, self_signal):
                del self.placements[placement.node_key]
                del self.dp_occupancy[placement.dp_id]
                self.operation_records.pop(placement.node_key, None)
                return False
            self.resource_signals[output].add(self_signal)
        return True

    def add_route(self, route: NativeRoute) -> bool:
        key = (route.source_key, route.destination_key)
        if key in self.routes:
            return False
        if key not in self.problem.dependencies:
            return False
        if len(route.resource_ids) != len(route.resource_latencies):
            return False
        staged: list[tuple[str, NativeSignal]] = []
        for rid, latency in zip(route.resource_ids, route.resource_latencies):
            signal = NativeSignal(
                source_key=route.source_key,
                destination_node=route.destination_node,
                latency=int(latency),
                source_node=route.source_node,
            )
            if not self.can_occupy(rid, signal):
                return False
            staged.append((rid, signal))
        for left, right in zip(route.resource_ids, route.resource_ids[1:]):
            if right not in self.problem.adjacency.get(left, ()):
                return False
        for rid, signal in staged:
            self.resource_signals[rid].add(signal)
        self.routes[key] = route
        return True

    def apply_action(self, action: NativeAction) -> "NativeMappingState | None":
        """Atomically apply an action to a copy of this state."""

        updated = self.copy()
        if not updated.place(action.placement):
            return None
        for route in action.routes:
            if not updated.add_route(route):
                return None
        return updated

    def stable_key(self) -> tuple[Any, ...]:
        """Exact deterministic state key; no symmetry approximation."""

        return (
            tuple(
                (
                    key,
                    placement.dp_id,
                    placement.latency,
                    placement.operation_latency,
                )
                for key, placement in sorted(self.placements.items())
            ),
            tuple(
                (
                    key,
                    route.resource_ids,
                    route.resource_latencies,
                )
                for key, route in sorted(self.routes.items())
            ),
            tuple(
                (
                    rid,
                    tuple(sorted(signals)),
                )
                for rid, signals in sorted(self.resource_signals.items())
                if signals
            ),
        )

    @classmethod
    def from_mapping(
        cls,
        problem: NativeMorpherProblem,
        mapping: Mapping[str, Any],
    ) -> "NativeMappingState":
        """Import a canonical mapping without losing native ``port_state``."""

        state = cls(
            problem=problem,
            memory_bindings=[
                dict(value) for value in mapping.get("memory_bindings", [])
            ],
            metadata=dict(mapping.get("metadata", {})),
        )
        routes_by_source: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for route in mapping.get("routes", []):
            routes_by_source[
                str(route.get("source_node_key", route.get("source_node")))
            ].append(route)
        for operation in mapping.get("operations", []):
            key = _node_key(operation)
            fu_id = str(operation["fu_id"])
            matching_dp = [
                dp
                for dp in problem.dp_records
                if dp["fu_id"] == fu_id
            ]
            if len(matching_dp) != 1:
                raise NativeContractError(
                    f"operation {key} FU resolves to {len(matching_dp)} DataPaths"
                )
            dp = matching_dp[0]
            witnessed = {
                int(route["start_time"]) - int(operation["latency"])
                for route in routes_by_source.get(key, ())
                if route.get("start_time") is not None
            }
            if len(witnessed) == 1:
                op_latency = next(iter(witnessed))
            else:
                op_latency = problem.operation_latency(
                    str(operation["opcode"]),
                    dp["memory_role"],
                    fu_id=dp["fu_id"],
                )
            placement = NativePlacement(
                node_key=key,
                node_id=int(operation["dfg_node_id"]),
                opcode=str(operation["opcode"]).upper(),
                pe_id=str(operation["pe_id"]),
                fu_id=fu_id,
                dp_id=dp["dp_id"],
                modulo_time=int(operation["modulo_time"]),
                latency=int(operation["latency"]),
                operation_latency=op_latency,
                pe_x=operation.get("pe_x"),
                pe_y=operation.get("pe_y"),
            )
            if not state.place(
                placement, operation_record=operation, add_output_signal=False
            ):
                raise NativeContractError(f"illegal native placement {key}")
        # Port state is authoritative and includes selector/fanout state not
        # recoverable from Morpher's shortened physical route records.
        for port_state in mapping.get("port_state", []):
            rid = str(port_state.get("native_port_id"))
            if rid not in problem.resources:
                raise NativeContractError(f"unknown port_state resource {rid}")
            for raw in port_state.get("signals", []):
                state.resource_signals[rid].add(
                    NativeSignal(
                        source_key=str(
                            raw.get("source_node_key", raw.get("source_node"))
                        ),
                        destination_node=int(raw["destination_node"]),
                        latency=int(raw["latency"]),
                        source_node=int(raw["source_node"]),
                    )
                )
        for raw in mapping.get("routes", []):
            route = NativeRoute(
                source_key=str(
                    raw.get("source_node_key", raw.get("source_node"))
                ),
                destination_key=str(
                    raw.get("destination_node_key", raw.get("destination_node"))
                ),
                source_node=int(raw["source_node"]),
                destination_node=int(raw["destination_node"]),
                edge_type=str(
                    problem.dependencies[
                        (
                            str(
                                raw.get(
                                    "source_node_key", raw.get("source_node")
                                )
                            ),
                            str(
                                raw.get(
                                    "destination_node_key",
                                    raw.get("destination_node"),
                                )
                            ),
                        )
                    ].get("edge_type", "")
                ),
                resource_ids=tuple(map(str, raw.get("ordered_resource_ids", []))),
                resource_latencies=tuple(
                    map(int, raw.get("ordered_resource_latencies", []))
                ),
            )
            state.routes[(route.source_key, route.destination_key)] = route
        return state

    def _operation_json(self, placement: NativePlacement) -> dict[str, Any]:
        existing = self.operation_records.get(placement.node_key)
        if existing is not None:
            record = dict(existing)
            record["dp_id"] = placement.dp_id
            return record
        node = self.problem.nodes[placement.node_key]
        return {
            "dfg_node_id": placement.node_id,
            "native_node_key": placement.node_key,
            "opcode": placement.opcode,
            "capabilities": sorted(
                translate_operation(placement.opcode)["capabilities"]
            ),
            "pe_id": placement.pe_id,
            "pe_x": placement.pe_x,
            "pe_y": placement.pe_y,
            "fu_id": placement.fu_id,
            "dp_id": placement.dp_id,
            "modulo_time": placement.modulo_time,
            "absolute_schedule_if_available": None,
            "latency": placement.latency,
            "asap": int(node.get("asap", 0)),
            "alap": int(node.get("alap", 0)),
            "bb": node.get("bb", ""),
            "has_constant": bool(node.get("has_constant", False)),
            "constant": node.get("constant", 0),
            "fixed_or_flexible": "flowadvantage_selected",
        }

    def to_mapping(self) -> dict[str, Any]:
        """Serialize a replayable canonical native mapping document."""

        operations = [
            self._operation_json(self.placements[key])
            for key in sorted(self.placements)
        ]
        routes = [
            self.routes[key].as_json() for key in sorted(self.routes)
        ]
        port_state = []
        for rid in sorted(self.resource_signals):
            signals = sorted(self.resource_signals[rid])
            if signals:
                port_state.append(
                    {
                        "native_port_id": rid,
                        "signals": [signal.as_json() for signal in signals],
                    }
                )
        return {
            "schema": "flowadvantage_morpher_mapping_v1",
            "toolchain_commit": self.metadata.get(
                "toolchain_commit",
                "9a9dce7aea521f1d5ef33686f57ca84864edb3c9",
            ),
            "architecture_hash": self.problem.mrrg.get("architecture_hash"),
            "dfg_hash": self.problem.dfg.get("dfg_hash"),
            "ii": self.problem.ii,
            "operations": operations,
            "dependencies": [
                dict(self.problem.dependencies[key])
                for key in sorted(self.problem.dependencies)
            ],
            "resources": [
                dict(self.problem.resources[rid])
                for rid in sorted(self.problem.resources)
            ],
            "routes": routes,
            "port_state": port_state,
            "memory_bindings": [dict(value) for value in self.memory_bindings],
            "metadata": {
                **self.metadata,
                "native_source": "FlowAdvantage NativeMappingState",
            },
        }
