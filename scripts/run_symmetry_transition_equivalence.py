#!/usr/bin/env python3
"""Generate exhaustive reviewer evidence for exact transition quotienting."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

import networkx as nx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quotientflow.arch import Architecture, verify_arch_permutation
from quotientflow.dfg import DFG, _annotate
from quotientflow.transition_equivalence import validate_transition_equivalence


def micro_architecture(case: str, ii: int = 2) -> Architecture:
    graph = nx.DiGraph(name=f"review_{case}")
    for x in range(2):
        for y in range(2):
            memory_role = "load" if case == "memory" and x == 0 else "none"
            supports = (
                ("ADD", "MUL", "LOAD")
                if case != "memory" or x == 0
                else ("ADD", "MUL")
            )
            for t in range(ii):
                graph.add_node(
                    (x, y, t), capacity=1.0, supports=supports,
                    memory_role=memory_role, resource_type="compute",
                )
    links = []
    for x in range(2):
        for y in range(2):
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                xx, yy = x + dx, y + dy
                if 0 <= xx < 2 and 0 <= yy < 2:
                    links.append(((x, y), (xx, yy)))
    for x in range(2):
        for y in range(2):
            for t in range(ii):
                graph.add_edge(
                    (x, y,t), (x,y,(t+1)%ii), kind="wait", capacity=1.0,
                    base_cost=1.05, dx=0, dy=0, wrap=False,
                    mutex_group=f"wait:{t}", broadcast_group=None,
                )
    for (x, y), (xx, yy) in links:
        for t in range(ii):
            graph.add_edge(
                (x,y,t), (xx,yy,(t+1)%ii), kind="cardinal", capacity=1.0,
                base_cost=1.0, dx=xx-x, dy=yy-y, wrap=False,
                mutex_group=f"axis:{'x' if xx != x else 'y'}:{t}",
                broadcast_group=f"axis:{'x' if xx != x else 'y'}:{t}",
            )
    edge_list = sorted(graph.edges)
    edge_index = {edge: index for index, edge in enumerate(edge_list)}
    for edge, index in edge_index.items():
        graph.edges[edge]["index"] = index
    arch = Architecture(f"review_{case}", 2, 2, ii, graph, edge_list, edge_index, [])
    transforms = []
    # Reflections preserve axis-labelled mutex/broadcast sets.  Transposition
    # changes those semantic groups; a time rotation violates the fixed modulo
    # coordinate contract.  Both are deliberately proposed so rejection is
    # recorded rather than assumed.
    for reflect_x, reflect_y in ((False, False), (False, True), (True, False)):
        perm = {
            (x,y,t): (1-x if reflect_x else x, 1-y if reflect_y else y, t)
            for x,y,t in graph
        }
        transforms.append(perm)
    transforms.append({(x,y,t): (y,x,t) for x,y,t in graph})
    transforms.append({(x,y,t): (x,y,(t+1)%ii) for x,y,t in graph})
    arch.symmetries = transforms
    return arch


def micro_dfg(case: str) -> DFG:
    graph = nx.DiGraph()
    if case == "memory":
        graph.add_node(0, op="LOAD", memory_role="load")
        graph.add_node(1, op="ADD", memory_role="none")
        graph.add_edge(0, 1, kind="memory", distance=0)
    elif case == "recurrence":
        graph.add_node(0, op="MUL", memory_role="none")
        graph.add_node(1, op="ADD", memory_role="none")
        graph.add_edge(0, 1, kind="recurrence", distance=1)
    else:
        graph.add_node(0, op="MUL", memory_role="none", priority_class=0)
        graph.add_node(1, op="MUL", memory_role="none", priority_class=0)
        graph.add_node(2, op="ADD", memory_role="none", priority_class=1)
        graph.add_edges_from(((0,2,{"kind":"data", "distance":0}), (1,2,{"kind":"data", "distance":0})))
    _annotate(graph, ii=2)
    for node in graph:
        level = graph.nodes[node]["level"]
        graph.nodes[node].update(latency=1, asap=level, alap=level + 1, absolute_schedule=level)
    # _annotate replaces edge kind, restore the semantic cases.
    if case == "memory":
        graph.edges[0,1].update(kind="memory", distance=0)
    elif case == "recurrence":
        graph.edges[0,1].update(kind="recurrence", distance=1)
    else:
        for edge in graph.edges:
            graph.edges[edge].update(kind="data", distance=0)
    identity = {node: node for node in graph}
    permutations = [identity]
    if case == "tied_priority":
        permutations.append({0: 1, 1: 0, 2: 2})
    return DFG(case, 24072026, graph, permutations)


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/hail_mary_hpca1/reviewer_symmetry")
    args = parser.parse_args()
    output = Path(args.output)
    raw = output / "raw"
    cases = ("tied_priority", "recurrence", "memory")
    result_rows, rejection_rows = [], []
    for case in cases:
        dfg, arch = micro_dfg(case), micro_architecture(case)
        result, rejected = validate_transition_equivalence(case, dfg, arch)
        result_rows.append(asdict(result))
        dfg_names = ("identity", "swap_tied_operations")
        arch_names = (
            "identity", "reflect_y", "reflect_x", "transpose_xy", "rotate_modulo_time"
        )
        rejection_rows.extend(
            {
                "case": case,
                **asdict(rejection),
                "dfg_transform_name": dfg_names[rejection.dfg_transform],
                "architecture_transform_name": arch_names[
                    rejection.architecture_transform
                ],
            }
            for rejection in rejected
        )
    write_csv(raw / "transition_equivalence.csv", result_rows, list(result_rows[0]))
    write_csv(
        raw / "rejected_transforms.csv", rejection_rows,
        [
            "case", "dfg_transform", "architecture_transform", "reason",
            "dfg_transform_name", "architecture_transform_name",
        ],
    )
    summary = {
        "schema": "quotientflow_transition_equivalence_v1",
        "cases": len(result_rows),
        "all_passed": all(row["passed"] for row in result_rows),
        "reachable_states": sum(row["reachable_states"] for row in result_rows),
        "checked_state_transforms": sum(row["checked_state_transforms"] for row in result_rows),
        "checked_actions": sum(row["checked_actions"] for row in result_rows),
        "rejected_transforms": len(rejection_rows),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    table = "\n".join(
        "| {case} | {reachable_states} | {admitted_transforms} | {rejected_transforms} | {checked_actions} | {passed} |".format(**row)
        for row in result_rows
    )
    report = f"""# Exact symmetry transition-equivalence validation

This is an exhaustive transition-system check, not merely a graph-isomorphism
or final-outcome comparison.  Only transforms preserving the full semantic
resource contract and the mapper's total operation order are admitted.

| Case | Reachable states | Admitted transforms | Rejected transforms | Checked actions | Pass |
|---|---:|---:|---:|---:|---|
{table}

All admitted transforms were checked for legal-action bijection, invariant
scoring, commuting successors, and equal terminal legality/cost.  The cases
exercise tied priorities, recurrence-distance metadata, memory compatibility,
time-expanded resources, mutex groups, and broadcast annotations.  Temporal or
resource transforms that change those attributes are rejected.

Crucially, the tied-priority branch swap is a valid typed-DFG automorphism but
is rejected because it does not preserve the stable node-ID tie-break in the
sequential operation order.  QuotientFlow must therefore not claim that DFG
transform as an exact search symmetry without changing the transition policy.

Raw evidence:

- `raw/transition_equivalence.csv`
- `raw/rejected_transforms.csv`
- `summary.json`
"""
    (output / "REPORT.md").write_text(report)
    if not summary["all_passed"]:
        raise SystemExit("transition-equivalence validation failed")


if __name__ == "__main__":
    main()
