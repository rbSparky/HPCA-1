# Gem branch result audit — 2026-07-31

> Paper-suite evidence: **26%** `[█████░░░░░░░░░░░░░░░]`<br>
> Time remaining to the stated 2026-07-31 23:59 IST deadline at audit time: **20.5 hours**

## Reconstructed state

The audited source is branch `gem` at commit
`126b804104b3e5ce45b2cdda5ab5cdd3932f1805`. The selected checkpoint is
`results/revision_v3/checkpoints/residual_gnn_seed_23.pt`, SHA-256
`468a8ffc541d20efcc77333e8492d4a1fcda88a022a506c9013b809fb991cad8`.
The terminal cluster snapshot is
`results/remote_snapshots/20260731_flowadvantage_v2_manifest_v7_terminal`.

All 33 queued FlowAdvantage diagnostic jobs reached a terminal state: 11 `DONE`,
19 `VALID_MAPPING_FAILURE`, and 3 `TIMEOUT`. There were no infrastructure
`ERROR` rows. The queue covers 11 distinct kernel/architecture pairs, five
kernels, four architectures, and three methods. It is not the frozen 12-kernel
by four-architecture paper matrix.

Current source tests pass: **148 passed in 77.44 seconds**.

## Real-mapping results that can be stated

Only normally completed, identical pair identities are compared below. The ten
paired cases have heterogeneous watchdogs and warm-cache ordering, so these are
extended-budget diagnostics—not matched-budget paper results.

| Method | Paired cases | Length successes | Method successes | Success delta | Bootstrap 95% CI | Common-success route-cost delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dual-linear | 10 | 3 | 4 | +10.0 points | [0.0, 30.0] | 0.0% |
| proposal-only | 10 | 3 | 3 | 0.0 points | [0.0, 0.0] | +2.31% |
| top-4 exact reranking | 10 | 3 | 4 | +10.0 points | [0.0, 30.0] | 0.0% |

The additional top-4 success is `array_add/A0_hycube4x4`. The benefit is thus
one kernel on one architecture, and the confidence interval includes zero.
Top-4 matches dual-linear success on this subset; it has not yet demonstrated a
learned advantage over the analytical baseline.

All three `array_add/A3_hycube8x8` methods timed out after 7,200 seconds. The
earlier interpretation that top-4 mapped this A3 case is false. Timeouts are
reported separately and are not mapping failures.

## Runtime inference

The principal bottleneck is optimization, not GNN inference. Successful top-4
jobs took 958.6–3,974.0 seconds, with a 1,012.6-second median. For example,
`trmm/A2` spent 917.8 wall seconds in child relaxation processing while GNN
encoding plus action-head inference took about 0.015 seconds. The A3 timeout
heartbeat reached roughly 11 GB RSS before any recorded parent or child solve.

The real top-4 implementation therefore fails the current practical-runtime
target. The next heavy run should not begin until a paired microbenchmark shows
that parameterized sparse relaxation templates, reachability reduction, and
warm starts preserve numerical results while lowering construction/solve time.

## Solver-status inference

Several `array_cond` and `hpcg` failures contain parent relaxation status
`user_limit`. That is a solver termination status, not proof of mathematical
infeasibility. These rows remain valid mapper failures under the executed
fail-closed policy, but cannot support a claim that the underlying relaxation or
mapping is physically infeasible. A solver-health retry must distinguish:

- certified infeasible;
- iteration/time limited with acceptable residuals;
- numerically failed;
- structurally no legal temporal route.

## Adapter status

The adapter has meaningful implementation evidence: six of nine reference
pairs preserve placement, routes, II, memory bindings, independent legality,
native reimport legality, and configuration generation. All eight deliberately
illegal mutations are rejected by both checkers. However, three reference
pairs lack native dumps and zero of nine round trips have cycle-simulation output
checks. Adapter gate A0/R0 is therefore **RED**, not because the six round trips
failed, but because functional validation and required coverage are incomplete.

## Synthetic evidence retained

Revision-v4b remains the strongest complete controlled result: length maps
41/56 and FlowAdvantage top-4 maps 54/56 (+23.21 points; paired 95% CI
[12.50, 33.93]). On 28 paired cases, top-4 and full relaxed lookahead have
identical success, top-4 removes 89.12% of child solves, and is 5.03x faster.
This validates the controlled mechanism; it is not a substitute for real Morpher
evidence.

## Scientific conclusion today

The direction is still promising, but a paper-level real-kernel advantage is not
established. The current real run proves that the end-to-end mapper and atomic
queue can execute on native graphs. It offers a one-case positive mapping signal
and exposes the real bottleneck. It does not yet provide the breadth, matched
baselines, II search, functional simulation, held-out portability, or runtime
needed for an HPCA claim.

Reproduction tables are under `tables/`; unrounded terminal records are in
`raw/terminal_flow_runs.csv`; input artifact hashes are in
`SOURCE_ARTIFACTS.sha256`.
