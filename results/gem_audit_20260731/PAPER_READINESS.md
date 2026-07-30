# Paper-suite readiness and execution plan

## Completion estimate

The frozen evidence contract from `docs/HPCA_PAPER_SUITE_PLAN.md` is **26% complete**.
This is an evidence-completion estimate, not a code-completion estimate.

| Evidence category | Weight | Earned | Current basis |
| --- | ---: | ---: | --- |
| Adapter and functional correctness | 20 | 12.0 | 6/9 legality round trips; 8/8 negative tests; no cycle simulation |
| Principal 12×4 real matrix | 35 | 5.5 | 30 normal Flow cells over ten paired identities; sparse coverage |
| Conventional baselines | 10 | 0.0 | no matched PathFinder and three-seed annealing matrix |
| Lookahead and parent ablations | 10 | 1.0 | synthetic evidence only; no real no-parent/full comparison |
| Portability and scaling | 10 | 0.5 | A3 attempted, but principal jobs fail or time out |
| Runtime and statistics | 5 | 2.0 | atomic timings and reconstructed bootstrap; budgets/caches unmatched |
| Artifact and reporting | 10 | 5.0 | strong hashed raw records; paper tables and one-command reproduction incomplete |

Execution infrastructure is further along—roughly 70%—because native graph
imports, a full legality checker, atomic queues, heartbeat recovery, checkpoint
hashing, and semantic caches exist. The gap between 70% infrastructure and 26%
evidence is the central project risk.

## Mandatory identify → refine → verify loop

No broad queue should be launched until every P0 loop passes on a fixed smoke set.

### P0.1 — relaxation runtime and solver health

**Identify:** profile a successful 4×4 state and the A3 root state, separating
matrix assembly, CVXPY canonicalization, solver time, cache I/O, and peak RSS.

**Refine:** use one sparse parameterized problem template per
DFG/MRRG/II/remaining-shape signature; cache index maps; prune unreachable
variables before construction; warm-start parent and child primal values; avoid
NetworkX objects in the numerical loop. Keep one BLAS thread and no nested pools.

**Verify:** on at least six fixed states require identical feasibility class,
selected action when gaps exceed 1e-8, route cost within 1e-8, objective relative
difference within 1e-6, and residual limits unchanged. Require at least a 2×
top-4 speedup before scaling.

### P0.2 — `user_limit` classification

**Identify:** retain solver iteration counts, termination cause, primal/dual
residuals, and incumbent objective for every limited solve.

**Refine:** add a documented retry hierarchy using the existing solver set and
longer numerical iterations for diagnostic reclassification. Do not tune mapper
scores from test outcomes.

**Verify:** no `user_limit` row is called infeasible without a solver certificate;
repeated cached solutions agree to 1e-6 relative objective tolerance.

### P0.3 — native functional correctness

**Identify:** trace why imported mappings produce configuration but no simulation
record.

**Refine:** connect mapping export to Morpher configuration, simulator input, test
vectors, and reference output comparison. Preserve memory terminals and recurrence
semantics.

**Verify:** at least six round-trip mappings simulate with zero mismatches, and at
least seven of nine reference pairs pass every legality stage; the target remains
9/9.

### P0.4 — II search

The diagnostic queue evaluates fixed reference IIs. Implement and smoke Morpher's
resource/recurrence lower bound and increasing-II search through lower-bound+4.
Every II attempt must be atomic and distinguish valid failure from timeout.

## Minimum defensible pilot after P0

Freeze before execution:

- 10–12 real kernels and all four architectures (40–48 pairs);
- PathFinder and simulated annealing seeds 11/23/37;
- length, dual-linear, frozen proposal, frozen top-4, and true no-parent top-4;
- full relaxed lookahead on the fixed 12-pair subset;
- 30/120/600-second matched budgets;
- A3 held out from all labels/tuning;
- cycle simulation and output matching for every reported successful mapping where supported.

Use balanced round-robin atomic scheduling. Report only identical paired sets.
Generate 10,000-resample paired confidence intervals per architecture and kernel,
plus cold/warm runtime decompositions. LISA must be run where its model is genuinely
compatible and marked unsupported elsewhere.

## Required evidence after the pilot

If the pilot passes, the paper suite still needs at least 24 kernels, eight
architectures, five stochastic seeds, held-out topology/size/resource families,
full ablations, success-budget curves, minimum-II tables, simulator correctness,
and one-command table/figure reproduction. This full suite has not started.

## Deadline assessment

At the audit time approximately 20.5 hours remained until 2026-07-31 23:59 IST.
The prior 33-job diagnostic queue consumed roughly 13 hours while covering only
11 pair identities and three Flow methods. A complete HPCA-grade 24×8 suite is
not achievable by that deadline at the observed runtime. The feasible deadline
target is narrower: fix P0, obtain a balanced matched pilot, and preserve every
artifact needed to expand without regeneration.

## Reusable assets

Do not regenerate the following unless a semantic version changes:

- all revision-v3 checkpoints and frozen selection JSON;
- revision-v4b synthetic raw results and relaxation caches;
- revision-v5b native DFG/MRRG/mapping exports and negative tests;
- terminal remote work-item JSON/log/heartbeat snapshot;
- valid semantic relaxation cache entries whose full keys match source,
  DFG, architecture, II, state, action, solver, regularization, and feature version.

Any optimization that changes matrix structure, feature semantics, or candidate
generation must bump its version. It may reuse native graphs and model weights,
but not incompatible derived cache entries.
