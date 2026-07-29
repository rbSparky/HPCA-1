# Native FlowAdvantage frontier/runtime calibration

This validation-only note records the first clean empty-state calibrations
after the frontier-preserving and reachability-index repairs. It is not a
headline real-kernel quality table.

## Protocol

Each job starts from an empty `NativeMappingState`, constructs the frozen
deterministic length prefix, preserves the complete beam frontier, and then
executes the requested completion scorer. Reference mappings are hash-checked
only for provenance; their placements/routes are never used as initialization.
All outputs are atomic queue records and independent legality is required for
success.

## Results

| calibration | depth | method | status | mapped ops | parent solves | scorer rejections | wall s | interpretation |
|---|---:|---|---|---:|---:|---:|---:|---|
| depth40 length | 0.40 | length | `VALID_MAPPING_FAILURE` | 15/20 | 0 | 0 | 145.531 | valid discrete-search failure; prefix and completion ran normally |
| depth40 dual | 0.40 | dual_linear | `VALID_MAPPING_FAILURE` | 8/20 | 4 | 4 | 431.579 | exact parent relaxations were attempted; three infeasible and one user-limit branch were pruned |
| depth60 dual | 0.60 | dual_linear | `VALID_MAPPING_FAILURE` | 12/20 | 4 | 4 | 214.971 | exact parent relaxations were attempted; all frontier branches were infeasible |

The `VALID_MAPPING_FAILURE` rows are not infrastructure failures and are not
counted as positive real-kernel evidence. They establish that the repaired
queue correctly distinguishes a valid mapping-search failure from a crashed or
stalled worker.

## Runtime diagnosis and repair

The earlier depth-60 run spent roughly five minutes before its first parent
solve, at ~109% CPU and ~1.4 GB RSS, rebuilding native MRRG phase/adjacency and
reverse dictionaries separately for each dependency in
`_reachable_edge_columns`. A SIGUSR1 traceback located the time in graph
preprocessing, not the solver. The worker was stopped operationally and its
logs were preserved; it was not counted as a mapping failure.

Commit `9656eb9` caches the immutable phase/adjacency/reverse topology per
problem and edge ordering and indexes forward timestamps by resource. The
repaired depth-60 run reached four parent solves and published a structured
terminal row in 214.971 seconds. Native relaxation and queue tests pass.

The remaining cost is exact CVXPY canonicalization and numerical relaxation:
the depth-40 dual row reports 132.162 s canonicalization and 173.723 s solver
time across its parent branches. This is a real scalability constraint for the
full FlowAdvantage queue; it is not hidden or attributed to mapping quality.

## Paths

* `results/revision_v5b/pilot_queue_depth40_length_v1/`
* `results/revision_v5b/pilot_queue_depth40_dual_frontier_v1/`
* `results/revision_v5b/pilot_queue_depth60_dual_frontier_v2/`
* `flowadvantage/morpher_adapter/native_relaxation.py`
* `results/revision_v5b/CLUSTER_INCIDENT_20260729.md`
