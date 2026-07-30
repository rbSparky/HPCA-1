# HPCA-1 Hail-Mary Execution and Evidence Ledger

This is the authoritative, self-contained execution ledger for submission **4950**. The hard evidence deadline is **1 August 2026, 10:00 IST**. It records both positive and negative results. A timeout, solver rejection, validator failure, infrastructure error, unsupported cell, or partial mapping is never converted into a mapping failure or success.

<!-- GENERATED_PROGRESS_BEGIN -->
**Paper-ready evidence progress:** `██░░░░░░░░░░░░░░░░░░░░░░░` **6.0%**

**Hard deadline:** 2026-08-01 10:00 IST — **29.59 hours remaining**

**Last refreshed:** 2026-07-30T22:54:26+00:00

| Ledger | NOT_STARTED | RUNNING | VERIFIED | BLOCKED |
|---|---:|---:|---:|---:|
| Reviewer actions | 7 | 1 | 2 | 0 |
| Indexed results | 3 | 1 | 3 | 0 |
<!-- GENERATED_PROGRESS_END -->

## Immutable inputs and scope

- Branch at staging: `gem`; source commit before hail-mary edits: `8a580d6cca85008e9d20bb398a507275f0a0146a`.
- Revised-paper archive SHA-256: `8b454b9c598820973ea035fb01697ba1824f7354464d9e806e860c6b901a371c`.
- Tentative experiment specification SHA-256: `4a80d8bc375bafc90700cf446823fc5e3df278a21ff1867eda42b3b40cfe0fa3`.
- Frozen learned proposal: revision-v3 validation-selected residual GNN, seed 23; its checkpoint hash must be copied from the frozen model manifest into the experiment manifest.
- Morpher commit: `9a9dce7aea521f1d5ef33686f57ca84864edb3c9`.
- Primary goal: a complete defensible minimum real-kernel pilot, not the infeasible literal maximum of the tentative specification.
- A3 is held out until source, model, configs, analysis code, thresholds, and A0-A2 decisions are committed and hashed.
- No real Morpher kernel, reference mapping, A3 outcome, or pilot-test label may affect training, checkpoint choice, candidate generation, II range, or thresholds.

## What already counts—and what does not

The complete revision-v4b synthetic experiment remains valid background evidence: length mapped 41/56 and top-4 mapped 54/56, a paired gain of 23.21 percentage points with 95% interval `[12.50, 33.93]`; top-4 matched full relaxed lookahead on 28 cases while eliminating 89.12% of child solves and running 5.03x faster. These values **do not count as real Morpher evidence**.

Existing real queues are diagnostic until their budgets, prefix, II selection, input identities, and terminal paired coverage are validated. Family-sorted prefixes, unmatched method rows, and jobs exceeding the frozen budget cannot enter headline aggregates.

## Frozen primary experiment contract

### Workload matrix

- Kernels: `array_add`, `array_cond`, `gemm_nt`, `fix_fft`, `hpcg`, `trmm`.
- Core architectures: A0 4x4 HyCUBE, A1 4x4 StdNoC, A2 memory-constrained 4x4 HyCUBE.
- Held-out architecture: A3 8x8 HyCUBE, opened only after the freeze gate.
- Pair count: 18 core plus six held-out = 24.
- Unsupported semantics are established from the pre-outcome adapter census. No observed failure may cause pair replacement.

### Methods and seeds

- Deterministic: `length`, `dual_linear`, frozen `proposal`, frozen `top4`, native PathFinder.
- Stochastic: native simulated annealing, seeds `11`, `23`, `37`.
- Secondary top-k subset: `top1`, `top2`, `top4`, `top8`, `full_lookahead` on `{array_add,array_cond,fix_fft,trmm} x {A0,A2}`.

### Fairness controls

- Headline prefix fraction: `0.0`.
- II search: independently derived `LB..LB+4`, ascending; stop at the first legal complete mapping.
- Per-II cap: 120 seconds; per pair/method/seed end-to-end cap: 600 seconds.
- Secondary top-k/full-lookahead budget: 1,200 seconds for every compared method.
- Wall time includes loading, candidate generation, routing, all relaxation setup/iterations, features, inference, validation, serialization, and process startup.
- Flow methods share operation order, candidate generator, route/action caps, beam/expansion limits, legality, and terminal rules. Persist the action-universe hash before scoring.
- Native conventional methods start from their own empty state and receive the same external total budget.
- Success requires mapper completion plus independent serialized legality and native Morpher reimport legality.

### Terminal status enum

`LEGAL_SUCCESS`, `VALID_MAPPING_FAILURE`, `TIMEOUT`, `SOLVER_REJECTED`, `VALIDATOR_FAILURE`, `ERROR`, `UNSUPPORTED`, `CANCELLED`.

Only `LEGAL_SUCCESS` with both legality fields true has `success=true`. Every atomic result is written to a temporary file, flushed, fsynced, schema-validated, and atomically renamed. Completed results are immutable.

## Admission gates before expensive runs

### G0 — correctness and provenance

- Full regression suite passes.
- Eight corrupted mappings remain rejected by both checkers.
- Zero-prefix creates a genuinely empty legal state.
- Problem-only export produces native DFG and actual II-expanded MRRG without a successful witness.
- External timeout kills the entire process tree.
- Two repeated deterministic smoke runs have identical outcome, mapping hash, route cost, and action-universe hashes.
- Removing reference placements/routes after provenance hashing leaves search unchanged.

Failure blocks paper runs until repaired.

### G1 — optimized-path equivalence and viability

On six balanced jobs, optimized and original execution must have identical legality, outcome, selected action when score gap exceeds `1e-8`, and route cost within `1e-8`; relaxation objective relative difference must be at most `1e-6`.

At least four of six top-4 smoke jobs must terminate normally within 600 seconds. Reusable relaxation structure must yield at least 2x speedup or demonstrate with separate setup/iteration timers that numerical iterations are the irreducible bottleneck. A 4x4 worker must remain below 8 GB RSS. A3 starts alone and must emit regular heartbeats.

### G2 — freeze and A3 unlock

Before any A3 run, commit and hash source, config, checkpoint, feature version, solver settings, candidate caps, route caps, operation order, budgets, pair selection, analysis scripts, and reviewer gates. Record the commit in the manifest. Test outcomes may not change these choices.

## Runtime optimization checklist

- Reuse depth-indexed sparse relaxation templates because stable operation order fixes the remaining operation/dependency set at each depth.
- Parameterize residual capacities, occupied resources, anchored endpoints, and knockout resources.
- Reuse solver workspaces, sparsity patterns, warm starts, and numerical factorization only where the solver actually supports it.
- Cache DFG/MRRG encodings, topology reachability, alternative paths, compatible resources, and architecture statistics by semantic hash.
- Compute state features once; use indexed arrays rather than per-action NetworkX traversals.
- Batch all action-head inference and load the checkpoint once per worker.
- Record matrix build, parameter update, solver setup, solver iterations, dual extraction, scalar features, graph encodings, action head, candidate generation, validation, serialization, CPU time, wall time, and peak RSS separately.
- Persist reusable model weights, embeddings, templates, warm starts, and cache metadata; never reuse across a semantic-key mismatch.

## Reviewer-objection completion map

The machine-readable source of truth is `reviewer_actions.csv`.

| Review | Acceptance requirement | Required evidence |
|---|---|---|
| R1 | Paired end-to-end mapping | Complete zero-prefix 24-pair matrix, status census, paired bootstrap, minimum II, route cost, and quality/time. |
| R2 | Runtime/top-k viability | Before/after profiles, setup versus iteration time, cache statistics, top-k/full-lookahead Pareto. |
| R3 | Prefix causality | Headline 0%; frozen 12-pair 0/20/40 sensitivity with identical frontiers for suffix methods. |
| R4 | Learning reproducibility | Immutable split/checkpoint/config disclosure and held-out synthetic ranking/ablation table; no real-test training. |
| R5 | Dual validity | Direct finite knockouts and reduced tau/solver/tolerance stability on balanced native states, separate from synthetic evidence. |
| R6 | Exact symmetry | Exhaustive legal-action bijection, rank/score/successor equivariance, terminal orbit and best-cost equality. |
| R7 | Closest baseline | Faithful compatible released artifact run or precise dated license/contract incompatibility report; no proxy superiority claim. |
| R8 | Submission readiness | Number 4950, approved disclosure, anonymous metadata/text, embedded fonts, valid US Letter PDF. |

## Secondary studies

### Prefix sensitivity

Use a frozen balanced 12-pair subset at 0%, 20%, and 40%. All Flow methods receive identical prefix frontier hashes at 20/40; prefix construction time remains inside the 600-second budget. A benefit only at 40% is described as suffix completion.

### Learning disclosure

Derive tables from existing immutable revision-v2/v3 labels and checkpoints: action/state counts, splits, kernel/architecture membership, on-policy depth distribution, seeds, dimensions, parameters, optimizer, loss, epochs, selection rule, training time, label status census, normalization, top-k recall, epsilon-optimal top-1, regret, Spearman/Kendall, and end-to-end association. Report residual GNN, residual MLP, direct-Q/action-v2, no-dual, initial-only, length, and dual-linear variants. Do not manufacture a true no-parent result from a model that still adds the parent baseline.

### Dual stability

Use 36 balanced native states: six kernels x A0-A2 x two depth buckets, with eight deterministic knockouts per state. Use 12 preselected states for a reduced tau/solver/tolerance/redundancy sweep. Preserve finite, infeasible, rejected, and fallback outcomes. Report scoreable rate, residual health, positive-dual rate, within-state rank correlation, top-k impact recall, setting disagreement, action overlap, regret changes, setup/iteration time, and limited statistical power.

### Symmetry

Use small exhaustive 2x2/3x3 instances covering tied priorities, chains/diamonds, recurrences, memory-port restrictions, mutex/broadcast/mux semantics, temporal transforms, order-violating graph automorphisms, and resource/capacity/timing violations. Reject every transform that fails the complete total order or native semantics. No truncated group is exact. Any failed assertion disables the paper's quotient quality claim.

### Closest baseline

Audit the official ICCAD-2022 GNN/RL mapper, Rewire, and native LISA artifacts for repository ownership, license, commit, DFG/MRRG semantics, operation scheduling, memory, action atomicity, termination, and legality compatibility. Run only a released, semantically matched artifact. Otherwise publish `BASELINE_COMPATIBILITY.md` and make no numeric superiority claim.

## Execution schedule and hard cutoffs (IST)

| Deadline | Deliverable |
|---|---|
| 31 Jul 07:00 | Tracker, paper staging, zero-prefix/problem export/status fixes, stale-job capture, tests. |
| 31 Jul 11:30 | Relaxation optimization, equivalence/admission smoke, Morpher rebuild, immutable freeze. |
| 31 Jul 19:00 | Complete A0-A2 matrix; symmetry, learning disclosure, and native dual study running in parallel. |
| 1 Aug 02:00 | A3, top-k/full-lookahead subset, native retries, and trace-matched simulation complete. |
| 1 Aug 05:30 | Prefix sensitivity, timing repeats, baseline audit; experimental raw data frozen. |
| 1 Aug 07:30 | Tables, figures, confidence intervals, regression/failure analysis, manuscript results integrated. |
| 1 Aug 09:00 | PDF and artifact integrity/reproduction preflight complete. |
| 1 Aug 09:40 | Final commit/tag/push and verified handoff ZIP complete. |
| 1 Aug 10:00 | Hard stop; only checksum/transfer contingency after 09:40. |

No new experiments begin after 05:30 IST. No numerical manuscript edits occur after 09:00 IST unless correcting a provenance mismatch, in which case the affected claim is removed rather than hand-edited.

## Deadline triage

Drop work only in this order:

1. Extra top-k points beyond 2/4/full.
2. Deterministic timing cells beyond the frozen six.
3. Breadth of the native dual sensitivity sweep.
4. Unsupported LISA/third-party cells.
5. Exact-trace simulations beyond the defensible minimum.

Never drop the zero-prefix core matrix, PathFinder/SA paired baselines, held-out A3 subset, legality, terminal-status accounting, ordered-transition symmetry tests, leakage disclosure, or artifact/PDF audit.

## Statistical and reporting contract

- Primary unit: kernel/architecture pair.
- Bootstrap: 10,000 paired resamples, seed `24072026`.
- Simulated annealing: preserve per-seed outcomes; never compare deterministic single-run output to best-of-three.
- Route cost: common legal successes only.
- Minimum II: lowest legal II in the frozen grid; no success is right-censored, never imputed.
- A confidence interval containing zero is described as inconclusive.
- Every table and figure states its paired denominator and includes or explicitly accounts for non-success statuses.
- Tables and figures are generated from scripts, never manually edited.

## Required final artifacts

- Frozen manifest and config hashes.
- Atomic raw run/II/action/solver/legality/simulation rows and logs.
- Complete failure, timeout, solver-rejection, validator, unsupported, and cancellation census.
- End-to-end, minimum-II, common-success route-cost, runtime/RSS, top-k Pareto, prefix, learning, dual, symmetry, A3, and closest-baseline tables.
- Vector figures plus source CSV.
- Exact-trace simulation evidence where available; unavailable provenance is explicit.
- Updated manuscript and response ledger with claims sourced only from `result_index.csv`.
- Environment, source, model, architecture, kernel, config, and analysis hashes.
- A clean-cache reproduction command.
- `results/HPCA1_HAIL_MARY_HANDOFF_20260801.zip`, its SHA-256, and successful ZIP integrity test.

## Final acceptance checklist

- [ ] Core A0-A2 matrix is at least 90% terminal for every headline method.
- [ ] All missing rows have an explicit operational status.
- [ ] A3 remained locked until the recorded freeze commit.
- [ ] Every successful mapping passed both legality paths.
- [ ] All headline numbers are paired and trace to immutable raw records.
- [ ] Zero-prefix results replace the 40%-suffix headline.
- [ ] Symmetry wording matches the exhaustive result; failed tests remove exactness claims.
- [ ] Learning and dual limitations are explicit.
- [ ] Submission number 4950 and approved AI disclosure pass PDF preflight.
- [ ] Repository commit/tag, pushed branch, archive CRC, and SHA-256 are recorded.

## Exact control commands

```bash
# Refresh progress and deadline clock
python3 scripts/update_hail_mary_tracker.py

# Build and fail-closed preflight the staged manuscript
bash scripts/build_hpca1_paper.sh

# Full repository regression suite
python3 -m pytest -q
```

Experiment launch commands, immutable manifest hash, cluster run IDs, aggregation commands, and the clean-cache reproduction command are appended here only after their implementations and configs are frozen.
