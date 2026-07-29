# FlowAdvantage paper-suite v2 execution protocol

This document freezes the next paper-level execution chunk. It records the
procedure, not results. Any missing, unsupported, timeout, or negative row is
retained and reported; it is never silently converted into a mapping failure.

## Frozen scope

The suite uses Morpher base commit `9a9dce7aea521f1d5ef33686f57ca84864edb3c9`
with the fixed18 deterministic-SA image recorded in
`results/revision_v5b/morpher_patch/PATCH_MANIFEST.md`. The six previously
selected kernels are retained unchanged. Four compiler-derived kernels are
added as a separate frozen group: `fir`, `conv2`, `conv3`, and `mac`. Kernel
selection is completed before reading FlowAdvantage outcomes.

Architectures are A0 (standard HyCUBE 4×4), A1 (standard NoC 4×4), A2
(memory-constrained HyCUBE 4×4), and A3 (held-out 8×8). A3 is never used for
training, checkpoint selection, or threshold selection.

## Mapping protocol

Every FlowAdvantage and dual-linear job starts from an empty native mapping.
The worker first runs the deterministic `LengthActionScorer` beam and stops at
`ceil(0.40 × operation_count)`, clamped to `[1, operation_count−1]`. The
resulting legal prefix is then passed to the selected method, which completes
the remaining operations. The native reference mapping is latency metadata
only; its placements, routes, occupancy, and score are forbidden inputs.

This is a two-stage complete mapping experiment, not an n−1-operation witness
completion. Prefix timing, expansions, generated actions, and routing attempts
are stored with every atomic result.

## Methods and paired cells

The required deterministic methods are `length`, `dual_linear`,
`flow_proposal`, and `flow_top4`. Native PathFinder and seeded simulated
annealing (seeds 11, 23, 37) are conventional baselines. Full relaxed
lookahead is restricted to the frozen balanced subset of
`array_add`, `gemm_nt`, and `fix_fft` on available A0–A2 contracts. All quality
comparisons use identical kernel/architecture/seed/budget cells. No-parent
FlowAdvantage is explicitly unsupported because the available
`residual_gnn_no_dual` checkpoint still adds a parent dual baseline at
inference; using it without a parent solve would be a proxy and is forbidden.

## Staged queue readiness

1. Verify Docker/image/source/checkpoint/config hashes and native export
   schemas.
2. Run adapter round-trip, native legality, independent legality, and cycle
   simulation smokes. Do not launch headline jobs while these are red.
3. Run the empty-state deterministic-prefix smoke twice and require identical
   prefix placement, routes, capacity state, and depth.
4. Run one atomic worker for each required method, including one full-lookahead
   cell, with heartbeat and process-tree timeout handling.
5. Generate the immutable paired manifest. Reject missing contracts, invalid
   MRRG edges, hash mismatches, and unsupported methods before submission.
6. Submit jobs in round-robin kernel/architecture/method order with one CPU
   worker and one GPU0 worker at most. Aggregate only terminal schema-valid
   atomic rows.

The queue may proceed to the next tranche only when every readiness smoke has
an explicit terminal record. A timeout is operational evidence, not a quality
failure. If fewer than the configured paired cells complete normally, report
coverage and stop the affected comparison rather than changing the suite.

## Reporting requirements

For each method and paired cell retain: II attempts, legal status from both
checkers, simulation/output status where available, route cost, routing
failures, backtracks, expansions, parent/child solves, feature/proposal/
relaxation/native-mapper time, cache hits, timeout/error class, source/config/
kernel/architecture/checkpoint hashes, and seed. Report success, minimum II,
route cost, and compile time only on identical paired cells. Use 10,000 paired
bootstrap resamples with seed `24072026`; confidence intervals that include
zero are not called statistically reliable.

No result is claimed in this document. The hard deadline is 31 July 2026
23:59 IST; the final artifact freeze must occur before that deadline.
