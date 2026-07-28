# FlowAdvantage HPCA Paper Suite Plan

Deadline: **2026-07-31 23:59 IST**. This plan is the executable continuation
of the rigor grounding document. It treats tonight as the first chunk of the
larger suite and freezes all decisions that could otherwise become test-set
tuning.

## Fixed strategy

Guarantee the claim-critical 12-kernel × four-architecture suite first, then
expand toward 24 kernels × eight architectures only after the core matrix is
paired and complete. Use eight isolated CPU workers and one serialized GPU0
worker. The progress percentage is weighted by evidence obligations, not raw
job count.

## Phase 0: bridge and environment smokes

Before performance jobs, rebuild the patched Morpher image and run:

1. Native DFG, MRRG, and mapping export.
2. FlowAdvantage import and independent legality.
3. Native re-import, native configuration, and simulator output check.
4. Nine reference pairs: `array_add`, `gemm_nt`, `fix_fft` × A0/A1/A2.
5. Eight deliberate negative legality mutations.
6. Frozen checkpoint load and batched proposal inference on A0–A3.
7. Parent/child relaxation repeatability and residual checks.
8. One complete PathFinder, SA, dual-linear, proposal-only, top-4, and
   full-lookahead mapping smoke.
9. Eight balanced queue-worker smokes with atomic output and recovery.

No headline mapping rows are produced while these checks are red. A precise
unsupported semantic is recorded separately from a format or implementation
failure.

## Phase 1: frozen data and core matrix

Generate the full kernel/architecture census before FlowAdvantage outcomes are
read. The primary list is fixed in
`results/paper_suite_v1/kernel_selection.json`; fallback order is fixed and
cannot be changed from test difficulty.

For every supported pair, run PathFinder; SA seeds 11/23/37; dual-linear;
proposal-only; hybrid top-4; and no-parent top-4. Run compile budgets 30, 120,
and 600 seconds for the inexpensive methods. Run full relaxed lookahead and
top-4 with the same higher research cap on the fixed 12-pair subset.

Jobs are round-robin by kernel class, architecture, seed, budget, and method.
Each job is one isolated process with an immutable manifest, heartbeat,
temporary result, fsync, schema validation, atomic rename, and terminal status.

## Phase 2: ablations and portability

Run top-2/top-4/top-6/full-lookahead, hybrid/no-parent, feature removals,
candidate caps, parent refresh rates, cache cold/warm, kernel-family holdout,
and A3 held-out-size evaluation. Validation-only thresholds are chosen before
frozen test cells; no method or threshold is selected from test outcomes.

## Phase 3: statistics and artifact freeze

Compute paired bootstrap intervals, per-kernel/per-architecture breakdowns,
budget curves, II heatmaps, route-failure plots, top-k solve/quality curves,
and runtime decomposition. Verify every displayed number traces to an atomic
row and hashes the configuration, source, checkpoint, kernel, architecture,
solver, and patch.

## Progress contract

`PAPER_RESULTS.md` is regenerated after every terminal work item. Its fixed
weighted denominator is:

| Evidence obligation | Weight |
|---|---:|
| Adapter and functional correctness | 20% |
| Complete 12×4 principal matrix | 35% |
| Stochastic and conventional baseline coverage | 10% |
| Full-lookahead/top-k and parent ablations | 10% |
| Held-out portability and scaling | 10% |
| Runtime and statistical analysis | 5% |
| Tables, plots, manifests, and artifact audit | 10% |

Only schema-valid evidence advances a section. `BORDERLINE/AMBER` evidence
counts as completed evidence but is visibly marked; `TIMEOUT`, `ERROR`, and
`UNSUPPORTED` do not count as quality results.

## Overnight schedule

00:00–02:00 IST: finish GEMM bridge, rebuild, and contract checks.

02:00–04:00 IST: freeze kernel census, architecture hashes, config, model,
environment, and queue smoke results.

04:00–08:00 IST: launch the balanced PathFinder/SA/dual/proposal/top-4
tranches; maintain the tracker and preserve every atomic result.

July 29 daytime: complete the core matrix and fill retries.

July 30: complete full-lookahead retention, parent ablation, feature/candidate
ablations, and A3 portability.

July 31 morning: statistics, scaling, cold/warm runtime, tables and figures.

July 31 afternoon/evening: re-run only missing claim-critical rows, freeze the
artifact package, tag `v1.0.0-paper-suite-2026-07-31`, and stop tuning.

## Cluster safety

The queue uses absolute remote paths, additive sync, one GPU0 flock, and
separate CPU/GPU pools. Before the overnight run, remove only the authorized
unused caches after active-process checks:

- Hugging Face `openai/gpt-oss-20b` cache (39 GB).
- Duplicate remote-work `gpt-oss-20b` cache (13 GB).
- Hugging Face DeepSeek-R1-Distill-Qwen-14B cache (28 GB).

Qwen/GLM caches are not removed while other active jobs may use them. Deletion
is recorded with path, size, timestamp, and recovered disk space.

## Final package

The final package contains the grounding document, this plan, frozen configs,
environment and toolchain manifests, atomic raw results, tables, plots,
`PAPER_RESULTS.md`, gates, bootstrap output, source/checkpoint/patch hashes,
reproduction scripts, and a compressed handoff archive. No timeout or
negative finding is omitted.
