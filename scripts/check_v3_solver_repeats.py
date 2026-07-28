#!/usr/bin/env python3
"""Independent repeated-solve objective checks for J1."""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quotientflow.flow_advantage import action_id  # noqa: E402
from quotientflow.mapper import enumerate_actions  # noqa: E402
from quotientflow.relaxation import solve_relaxation  # noqa: E402

OUT = ROOT / "results" / "revision_v3"
RAW = OUT / "raw"
CONFIG = {
    "tau": 0.001,
    "solver_primary": "CLARABEL",
    "solver_fallback": "OSQP",
    "max_seconds": 12.0,
}


def main():
    labels = pd.read_csv(RAW / "action_advantage_labels.csv")
    selected = labels[labels.child_feasible].drop_duplicates(
        "state_id"
    ).head(6)
    rows = []
    for source in selected.itertuples():
        with (ROOT / source.state_path).open("rb") as handle:
            sample = pickle.load(handle)
        operation = int(source.operation_id)
        actions, _, _ = enumerate_actions(
            sample["dfg"], sample["arch"], sample["state"], operation, 4
        )
        lookup = {action_id(operation, action): action for action in actions}
        action = lookup[source.action_id]
        repeated_parent = solve_relaxation(
            sample["dfg"], sample["arch"], sample["state"], **CONFIG
        )
        repeated_child = solve_relaxation(
            sample["dfg"], sample["arch"], action.state, **CONFIG
        )
        parent_relative = abs(
            repeated_parent.objective - source.J_parent
        ) / max(1.0, abs(source.J_parent))
        child_relative = abs(
            repeated_child.objective - source.child_objective
        ) / max(1.0, abs(source.child_objective))
        rows.append({
            "state_id": source.state_id,
            "action_id": source.action_id,
            "parent_original": source.J_parent,
            "parent_repeated": repeated_parent.objective,
            "parent_relative_difference": parent_relative,
            "child_original": source.child_objective,
            "child_repeated": repeated_child.objective,
            "child_relative_difference": child_relative,
            "match_within_1e-6": (
                repeated_parent.solved and repeated_child.solved
                and parent_relative <= 1e-6 and child_relative <= 1e-6
            ),
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(RAW / "cache_repeat_checks.csv", index=False)
    print(json.dumps({
        "checks": len(frame),
        "all_match": bool(frame["match_within_1e-6"].all()),
        "max_relative_difference": float(max(
            frame.parent_relative_difference.max(),
            frame.child_relative_difference.max(),
        )),
    }, indent=2))


if __name__ == "__main__":
    main()
