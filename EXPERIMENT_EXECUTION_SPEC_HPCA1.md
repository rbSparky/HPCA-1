# QUOTIENTFLOW Experiment Execution Specification

## 1. Objective

Generate the complete, auditable evidence needed to revise the HPCA paper without fabricating or selectively filtering results. The experiments must answer:

1. Does QUOTIENTFLOW improve legal mapping success, minimum initiation interval (II), or route cost under matched end-to-end budgets?
2. Does the learned residual model improve held-out action ranking over immediate length and the dual-linear baseline?
3. Are residual-capacity duals useful and numerically stable enough to serve as a baseline?
4. Does top-k exact correction retain full-lookahead quality while reducing child solves and compilation time?
5. Does symmetry quotienting preserve the exact transition system on small instances and reduce work on larger instances?
6. Does the method transfer across kernels and to held-out architecture A3?
7. Is the quality gain defensible relative to compilation time and memory?

Do not optimize for producing a positive result. Preserve all failures, regressions, timeouts, rejected labels, solver disagreements, and unsupported cells.

---

## 2. Non-negotiable controls

1. **Freeze before testing.** Commit the source, configuration, model split, architecture set, kernel set, II policy, seeds, budgets, solver settings, candidate caps, route caps, beam width, and analysis script before reading final test or A3 outcomes.
2. **A3 is locked.** A3 must not be used for training, validation, checkpoint selection, threshold selection, hyperparameter tuning, or debugging based on outcomes.
3. **No witness leakage.** Reference mappings may be hash-checked for corpus provenance, but their placements, routes, scores, or II must not select actions, initialize search, choose the II grid, or tune parameters.
4. **Same native action universe.** Length, dual-linear, proposal, top-k, and full lookahead must use the same operation order, candidate generator, route enumeration, action/route caps, legality checks, beam width, and terminal rules. Only scoring and the number of exact child solves may differ.
5. **Headline runs use zero prefix.** Deterministic headline comparisons start from the empty native state. The 20% and 40% length prefixes are secondary sensitivity experiments.
6. **End-to-end budget.** Wall time includes initialization, prefix generation if any, routing enumeration, residual solves, feature construction, GNN inference, canonicalization, serialization, and final validation.
7. **Failure is data.** Never turn timeout, solver rejection, infrastructure error, unsupported configuration, or partial mapping into a valid mapping failure or successful result.
8. **Legality is conjunctive.** A success requires mapper completion, the independent serialized-contract validator, and native reimport validation.
9. **No per-method tuning on test cells.** Global search parameters are fixed from training/validation data only. Do not increase a method's budget, beam width, action cap, or route cap selectively.
10. **Raw data are immutable.** Write one schema-validated result per run through a temporary file and atomic rename. Never overwrite a completed run.

---

## 3. Frozen benchmark suite

### 3.1 Kernels

Full suite:

- `array_add`
- `array_cond`
- `gemm_nt`
- `fix_fft`
- `hpcg`
- `trmm`
- `fir`
- `conv2`
- `conv3`
- `mac`

Minimum acceptance subset, run first:

- `array_add` — simple/easy control
- `array_cond` — conditional or mutex-sensitive behavior
- `gemm_nt` — dense compute and routing
- `fix_fft` — recurrence/communication structure
- `hpcg` — irregular and memory-sensitive structure
- `trmm` — existing strict-probe kernel

Do not change this subset after observing QUOTIENTFLOW outcomes.

### 3.2 Architectures

- `A0`: standard 4x4 HyCUBE
- `A1`: standard 4x4 NoC
- `A2`: memory-constrained 4x4 HyCUBE
- `A3`: held-out 8x8 HyCUBE

The minimum acceptance matrix uses the six-kernel subset on A0-A2 and the same six kernels on held-out A3. The full matrix uses all ten kernels on A0-A3.

### 3.3 II grid

For each `{kernel, architecture}` compute, without using a reference witness:

```text
II_lb = max(RecMII, ResMII, native_architecture_lower_bound)
```

Freeze the primary grid:

```text
II = II_lb, II_lb+1, ..., II_lb+6
```

Clip only if the native system has a documented legal maximum. If no mandatory method finds a legal mapping anywhere in the primary grid, execute the pre-registered extension `II_lb+7 ... II_lb+10` for **all mandatory methods** on that kernel-architecture pair. Do not extend only for a favored method.

Derive minimum mapped II as the lowest II in the frozen grid with a legal success. If no success occurs, mark the observation right-censored above the largest attempted II. Do not impute an II.

### 3.4 Search configuration

Extract the current canonical values for:

- beam width;
- maximum routes per dependence;
- action cap per state;
- expansion cap;
- route-attempt cap;
- deterministic total operation order;
- solver statuses considered usable;
- primal/dual residual health thresholds;
- tie-breaking keys.

If a value is missing, select one global value using only training/validation cells on A0-A2, record the selection procedure, and freeze it. All deterministic methods use the same values.

---

## 4. Methods and seeds

### 4.1 Mandatory headline methods

1. `length`
2. `dual_linear`
3. `proposal`
4. `top4`
5. native `pathfinder`
6. native `simulated_annealing`

### 4.2 Decision-hierarchy subset

Run on the fixed Pareto/full-lookahead subset:

- `top1`
- `top2`
- `top4`
- `top8`
- `full_lookahead`

### 4.3 Simulated-annealing seeds

Use ten fixed seeds:

```text
11, 23, 37, 53, 71, 89, 107, 131, 149, 173
```

The primary stochastic comparison is the distribution/expected performance of one seeded run at the same per-run budget. Do not use “any of ten seeds succeeded” or best-of-ten as the primary result, because that spends ten times the compute. A separate multi-start result is allowed only when total compute is matched explicitly.

### 4.4 Learned-model seeds

Main residual-target model:

```text
101, 211, 307, 401, 503
```

All ablation models: at least three of those seeds. Use all five when feasible.

### 4.5 Deterministic runtime repeats

Repeat deterministic configurations three times on identical inputs. The first run supplies the mapping outcome; all repeats must produce the same status, mapping hash, route cost, and action-set hashes. Use the repeats only to estimate runtime variation. Any outcome variation is a reproducibility failure.

---

## 5. Output schemas

### 5.1 One row per end-to-end run

Required columns:

```text
run_id
experiment_id
kernel
architecture
ii
ii_lb
method
model_variant
model_seed
search_seed
repeat
prefix_fraction
budget_s
status
status_detail
legality_serialized
legality_native_reimport
success
mapped_ops
total_ops
route_cost
wall_s
cpu_s
peak_rss_bytes
expansions
actions_generated
actions_scored
routing_attempts
parent_solves
child_solves
solver_fallbacks
solver_rejections
cache_hits
cache_misses
matrix_build_s
parameter_update_s
solver_setup_s
solver_iteration_s
dual_extract_s
feature_scalar_s
dfg_embedding_s
mrrg_embedding_s
action_head_s
inference_s
canonicalization_s
candidate_generation_s
validator_s
prefix_s
suffix_s
dfg_nodes
dfg_edges
mrrg_nodes
mrrg_edges
root_action_count
mapping_hash
action_universe_hash
source_commit
source_dirty
config_hash
manifest_hash
checkpoint_hash
input_hash
solver_name
solver_version
tau
solver_tolerance
host_id
cpu_model
gpu_model
thread_count
start_utc
end_utc
stdout_path
stderr_path
result_json_path
```

Use the status enum:

```text
LEGAL_SUCCESS
VALID_MAPPING_FAILURE
TIMEOUT
SOLVER_REJECTED
VALIDATOR_FAILURE
ERROR
UNSUPPORTED
CANCELLED
```

`success=true` only for `LEGAL_SUCCESS` with both legality fields true.

### 5.2 One row per held-out action group

```text
fold
split
kernel
architecture
trajectory_id
state_id
depth
depth_fraction
next_operation
action_count
method_or_model
model_seed
oracle_best_action
selected_action
selected_regret
relative_selected_regret
exact_top1
epsilon_optimal_top1
oracle_in_predicted_top1
oracle_in_predicted_top2
oracle_in_predicted_top4
oracle_in_predicted_top8
spearman
kendall
ndcg_4
parent_status
valid_child_labels
rejected_child_labels
fallback_child_labels
state_hash
action_set_hash
checkpoint_hash
```

### 5.3 One row per dual/resource observation

```text
fold
kernel
architecture
state_id
resource_id
resource_kind
free_capacity
dual_value
knockout_status
knockout_delta_j
knockout_infeasible
tau
solver
solver_tolerance
redundancy_mode
primal_residual
dual_residual
solver_iterations
state_scoreable
```

### 5.4 One row per symmetry assertion

```text
instance_family
instance_id
state_id
transform_id
transform_admitted
rejection_reason
preserves_total_rank
preserves_prefix
legal_action_bijection
score_equivariant
successor_equivariant
terminal_invariant
route_cost_invariant
quotient_terminal_hash
nonquotient_terminal_hash
quotient_best_cost
nonquotient_best_cost
atol
rtol
```

Store CSV or Parquet for tables and JSONL for nested diagnostics. Preserve all raw JSON and logs.

---

## 6. Execution order and decision gates

Run in this order. Do not launch the full matrix immediately.

### Gate 0 — Baseline integrity

Required before any new evidence:

- existing 148 regression tests pass;
- all eight deliberately corrupted mappings are rejected by both validators;
- source tree is clean and commit hash recorded;
- a two-cell smoke run writes schema-valid results;
- external timeout kills the entire process tree;
- repeated deterministic runs have identical mapping hashes;
- reference-mapping data removal after provenance checking does not change search behavior;
- deterministic methods expose identical action-universe hashes at sampled states.

If any item fails, fix it before generating paper data.

### Gate 1 — Runtime viability

Complete Experiment E1 below. Proposal and top-4 should complete a nontrivial fraction of the fixed pilot cells under 600 seconds. If top-4 still times out almost everywhere, return the profiling evidence before spending compute on the full matrix. Do not hide the negative result.

### Gate 2 — Ranking viability

Complete the leakage-safe dataset, training, and held-out ranking experiments. If the full residual model does not improve held-out top-k recall/regret over dual-only and length, run only the minimum end-to-end pilot needed to confirm the negative result before scaling.

### Gate 3 — Correctness viability

Symmetry quotienting may be enabled in quality experiments only after all exhaustive small-instance action-bijection and terminal-set tests pass. Otherwise disable quotienting, report the failure, and keep the main mapping-quality experiment quotient-free.

### Gate 4 — Freeze and unlock A3

Only after code, model checkpoints, thresholds, candidate caps, budgets, analysis scripts, and A0-A2 decisions are frozen may A3 be evaluated.

---

## 7. E0 — Infrastructure and action-universe parity

### Purpose

Prove that outcomes can be attributed to scoring rather than different legal actions or broken result handling.

### Procedure

1. Select at least 100 reachable states stratified across:
   - six minimum kernels;
   - A0-A2;
   - five depth bins: 0-20%, 20-40%, 40-60%, 60-80%, 80-100%;
   - low, medium, and high action counts.
2. For every state, enumerate the bounded native legal action set once.
3. Feed the identical serialized action set to length, dual-linear, proposal, top-k, and full-lookahead scorers.
4. Record sorted action IDs and an action-set hash before scoring.
5. Assert that scoring never adds, deletes, or changes an action.
6. Apply a sample of actions through both ordinary and instrumented paths and compare child-state hashes.
7. Remove reference placements/routes after provenance validation and repeat selected runs; behavior must be identical.

### Required output

- `action_universe_parity.csv`
- `reference_isolation_test.json`
- `timeout_process_tree_test.json`
- deterministic reproducibility report

### Pass condition

Zero action-universe mismatches, zero reference-leakage differences, and zero deterministic outcome mismatches.

---

## 8. E1 — Relaxation-path optimization and profiling

### Purpose

Address the current 3.3-4.0x proposal overhead and top-k timeouts without changing semantics.

### Implement these optimizations

1. Precompile sparse placement/flow constraint structure for each `{kernel, architecture, II}`.
2. Parameterize occupancy, residual capacities, anchored endpoints, and optional knockout resources.
3. Reuse parent/child matrices.
4. Warm-start primal and dual variables when supported.
5. Reuse numerical factorization when the solver supports it; otherwise record `unsupported` rather than claiming reuse.
6. Batch all actions in a state for scalar feature computation and neural inference.
7. Compute DFG and MRRG embeddings once per state, not once per action.
8. Cache architecture-only statistics by architecture/input hash.
9. Cache state-independent feature components.
10. Separate solver setup time from numerical iteration time.

### Semantics check

For 100 fixed states, compare the unoptimized and optimized paths:

- objective values;
- dual vectors;
- action scores;
- sorted rankings;
- selected action;
- child state;
- terminal mapping and route cost.

Use tolerance:

```text
atol = max(1e-7, 10 * configured_solver_tolerance)
rtol = 1e-6
```

Mappings, action IDs, legality, and route cost must match exactly. Numerical scores may use tolerance.

### Fixed pilot subset

Kernels:

```text
array_add, array_cond, fix_fft, hpcg, trmm
```

Architectures:

```text
A0, A2
```

IIs:

```text
II_lb+1 and II_lb+3
```

Methods:

```text
dual_linear, proposal, top1, top2, top4, top8
```

Budget: 600 seconds end-to-end. Repeat three times.

### Optimization ablation

Measure:

1. baseline;
2. matrix/parameter reuse only;
3. + warm starts;
4. + factorization reuse where available;
5. + batched feature/model scoring;
6. + graph embedding reuse;
7. + architecture/statistic caches;
8. all optimizations combined.

For each stage report median, P95, cache hit rate, parent/child solve counts, and semantic-equivalence status.

### Required plots

- stacked stage-time bars;
- before/after wall time;
- solver setup versus iteration time;
- cache hit rate and solves avoided;
- optimization speedup distribution.

---

## 9. E2 — Leakage-safe data generation and training

### 9.1 Five-fold kernel holdout

Use these fixed kernel folds:

```text
F0: array_add, trmm
F1: array_cond, hpcg
F2: gemm_nt, fix_fft
F3: fir, conv3
F4: conv2, mac
```

For fold `i`:

- test kernels = `Fi`;
- validation kernels = `F(i+1 mod 5)`;
- training kernels = remaining six kernels;
- training/validation architectures = A0-A2 only;
- A3 is completely excluded.

Every state, action group, and trajectory from a kernel stays in that kernel's fold. This yields out-of-kernel predictions for every kernel. For A3, use the checkpoint from the fold where that kernel was the test kernel, giving simultaneous kernel and architecture holdout.

### 9.2 State collection

Collect states only from the permitted split using a fixed mixture of policies:

- length;
- dual-linear where scoreable;
- randomized length tie-breaking with fixed collection seeds;
- failed-frontier states and successful trajectories.

Do not collect states using a model trained on the same test kernel.

Stratify by:

- five depth bins;
- action-count quartile;
- success/failure trajectory;
- architecture;
- kernel.

Target per available `{kernel, architecture}`:

- minimum 200 state groups;
- target 500 state groups;
- include the complete bounded legal action set for each retained group.

If fewer groups exist, record the actual census. Do not duplicate groups to meet the target.

### 9.3 Label generation

For every retained action:

1. solve the exact child relaxation;
2. record `Q*`, residual target, status, residual health, fallback use, and solve time;
3. preserve rejected/infeasible labels in the census;
4. never replace a rejected child solve with a finite target;
5. compute the group oracle only from valid child labels and report group coverage.

Freeze the label solver contract and hash it.

### 9.4 Model variants

Train the following as separate checkpoints, not relabeled inference modes:

1. **Full residual target:** dual baseline + graph towers + scalar features, predicts residual correction.
2. **Dual-only:** no learned model; use the parent dual-linear score.
3. **Scalar-only residual:** dual baseline + MLP over non-graph scalar features; no DFG/MRRG learned embeddings.
4. **GNN-only residual:** dual baseline + graph/action embeddings; remove handcrafted scalar features except quantities already in the explicit baseline.
5. **No-dual/no-parent:** remove the parent dual baseline, parent objective, dual-derived features, and residual-health features; predict correction relative to immediate length or predict an explicitly documented target.
6. **Direct-Q-star:** full-capacity encoder predicts `Q*` directly, without additive dual baseline.
7. **Residual-target main model:** the proposed model.

Keep parameter budgets comparable where possible and report exact parameter counts.

### 9.5 Training disclosure

Record:

- dimensions of every layer;
- parameter count;
- optimizer and version;
- learning rate and schedule;
- weight decay;
- batch/group construction;
- loss weights;
- epochs and early stopping;
- gradient clipping;
- training seeds;
- checkpoint-selection values;
- GPU/CPU and training time;
- group/action counts by split;
- label success/rejection/fallback rates;
- normalization statistics fitted only on training data.

Use the paper's lexicographic checkpoint rule unless changed before test evaluation:

1. top-4 oracle recall;
2. epsilon-optimal top-1;
3. relative selected regret;
4. exact top-1.

---

## 10. E3 — Held-out action-ranking evaluation

### Purpose

Show whether learned residual correction improves the action ordering that the mapper actually consumes.

### Evaluation set

Use all valid held-out action groups from the five kernel folds on A0-A2. Evaluate A3 only after freezing checkpoints. Macro-average by action group, then by kernel/architecture; do not let groups with thousands of actions dominate.

### Definitions

For group `g`, let:

```text
Q_best = min_a Q*(a)
regret(selected) = Q*(selected) - Q_best
relative_regret = regret / max(abs(Q_best), 1.0)
```

Define epsilon-optimal top-1 as:

```text
relative_regret <= 0.01
```

Report sensitivity at 0.5%, 1%, and 5% as a secondary analysis.

### Required metrics

For every model variant and seed:

- exact top-1 accuracy;
- epsilon-optimal top-1;
- oracle recall at top-1/2/4/8;
- absolute selected regret;
- relative selected regret;
- Spearman correlation per group;
- Kendall correlation per group;
- NDCG@4;
- inference time per state and per action;
- rejected-group and partial-label rates.

Use 10,000 bootstrap resamples over action groups, seed `24072026`, preserving kernel/architecture grouping in the resampling unit.

### End-to-end linkage

For each evaluated end-to-end cell, aggregate ranking quality observed along its trajectory and correlate it with:

- legal success;
- minimum II;
- route cost;
- expansions;
- wall time.

Report association as exploratory; do not claim causality.

---

## 11. E4 — Dual usefulness and numerical stability

### 11.1 Held-out state sample

Use at least 1,000 held-out states across the five folds, balanced across kernels, A0-A2, depth bins, and action-count strata. Use a 250-state subset for the full expensive solver sweep.

### 11.2 Finite resource-knockout test

At the default solver contract, for each state sample up to 64 currently free resources:

- include all positive-dual resources if there are at most 32;
- otherwise include the 16 largest positive duals and 16 randomly sampled positive duals;
- fill the remaining slots with matched zero/near-zero-dual resources;
- include compute resources as a separate category where feasible.

For each resource solve:

```text
DeltaJ(resource) = J(state with resource capacity set to zero) - J(state)
```

Record finite, infeasible, rejected, and fallback outcomes separately. Do not coerce infeasible knockouts into a finite number.

Required analyses:

- within-state Spearman/Kendall correlation between dual and finite `DeltaJ`;
- top-k recall for the largest knockout impacts;
- precision/recall for predicting infeasible knockouts;
- positive-dual rate;
- fraction of high-dual resources with negligible finite impact;
- fraction of zero-dual resources with large finite impact.

### 11.3 Tau/solver/tolerance sweep

Let `tau0` be the current default. Sweep:

```text
tau = 0.01*tau0, 0.1*tau0, tau0, 10*tau0, 100*tau0
```

Use:

- primary solver;
- fallback solver;
- coarse, default, and tight tolerance settings already supported by the implementation.

Run the full Cartesian sweep on the 250-state subset. On all 1,000 states, run the reduced sweep:

```text
tau = 0.1*tau0, tau0, 10*tau0
solver = primary, fallback
solver_tolerance = default
```

Also compare original versus deduplicated/redundancy-reduced constraint rows at `tau0` and default tolerance.

### Required stability metrics

- scoreable-state rate;
- permitted-status rate;
- fallback rate;
- rejected-state rate;
- positive-dual rate;
- primal/dual residual distributions;
- solver objective disagreement;
- dual-vector cosine similarity;
- action-ranking Spearman/Kendall correlation between settings;
- top-1/top-4 action overlap;
- selected-regret change;
- fraction of states whose top-1 action changes;
- solve time and iteration count.

### Comparison baselines

On the same held-out action groups compare:

- immediate length ranking;
- dual-linear ranking;
- exact-child oracle;
- learned residual ranking.

---

## 12. E5 — Paired zero-prefix end-to-end mapping

### 12.1 Primary budget

Use a 600-second external end-to-end wall budget per run. The paper's current probe uses 600 seconds, so this is the primary comparable operating point.

### 12.2 Minimum acceptance matrix

Run first:

```text
6 kernels x 4 architectures x frozen II grid
```

Mandatory methods:

```text
length
dual_linear
proposal
top4
pathfinder
simulated_annealing (10 seeds)
```

Prefix fraction: 0 for all deterministic headline methods. PathFinder and simulated annealing start from their own empty native state.

### 12.3 Full matrix

After Gates 1-4 pass, run:

```text
10 kernels x 4 architectures x frozen II grid
```

with the same mandatory methods and budget.

### 12.4 Primary outcomes

1. **Legal success under 600 seconds** at each fixed II.
2. **Minimum mapped II** within the frozen grid.
3. **Right-censored no-success count** above the grid.
4. **Route cost** only on identical `{kernel, architecture, II}` cells where both methods legally succeed.
5. **Compilation cost:** wall time, CPU time, peak RSS, parent/child solves, expansions, and routing attempts.
6. **Failure census:** valid mapping failure, timeout, solver rejection, validator failure, infrastructure error, unsupported.

### 12.5 Pairing and statistics

Use the `{kernel, architecture}` pair as the primary independent unit. Report:

- raw per-cell outcomes;
- win/tie/loss counts;
- paired difference in success proportion;
- paired minimum-II difference among jointly observed successes;
- paired route-cost difference on common successful II cells;
- median and P95 wall time;
- paired solve-count and expansion differences.

Use 10,000 paired bootstrap resamples with seed `24072026`. Resample whole kernel-architecture units, not individual actions or timing repeats. Include the estimate, 95% interval, and paired count. If the interval includes zero, describe the result as inconclusive rather than improved.

For simulated annealing, report per-seed distributions and use a hierarchical bootstrap over kernel-architecture cells and seeds. Do not compare a deterministic single run with best-of-ten SA without compute matching.

### 12.6 Suggested pre-registered practical-effect criteria

Freeze these before the full run or replace them with author-approved alternatives before seeing test outcomes:

- success-under-budget: at least 5 percentage points absolute improvement with paired CI excluding zero; or
- minimum II: a lower paired mean/median with CI excluding zero and at least several one-II wins; or
- a clear Pareto improvement at a common budget.

Route cost alone is insufficient if success rate decreases.

---

## 13. E6 — Prefix-depth sensitivity

### Purpose

Determine whether the method's benefit persists when it controls early decisions.

### Fixed subset

Kernels: six minimum kernels.

Architectures: A0-A2.

IIs:

```text
II_lb+1 and II_lb+3
```

Methods:

```text
length, dual_linear, proposal, top4
```

Prefix fractions:

```text
0.0, 0.2, 0.4
```

Budget: 600 seconds end-to-end.

### Procedure

1. For 20% and 40%, generate the complete length frontier once per `{kernel, architecture, II, prefix_fraction}`.
2. Save and hash that frontier.
3. Feed the identical frontier to all suffix methods.
4. Include prefix-generation time inside the 600-second budget and also report prefix and suffix time separately.
5. At 0%, start every method from the empty state.
6. Keep candidate caps, beam width, route caps, operation order, and validators unchanged.

### Report

- success under budget;
- minimum II where applicable;
- route cost on common successes;
- mapped operations at timeout/failure;
- expansions and solve counts;
- interaction between method and prefix depth.

The paper's headline must use 0%. A result that appears only at 40% supports suffix completion, not end-to-end mapping.

---

## 14. E7 — Top-k versus full lookahead and quality-time Pareto

### 14.1 Select subset without method outcomes

Use static structural properties only: DFG nodes/edges, MRRG nodes/edges, root action count, and estimated relaxation size. Select a balanced subset containing easy/hard and low/high branching cells.

Default fixed subset:

```text
kernels: array_add, array_cond, fix_fft, trmm
architectures: A0, A2
IIs: II_lb+1, II_lb+3
```

This gives 16 kernel-architecture-II cells.

### 14.2 Methods

```text
length
dual_linear
proposal
top1
top2
top4
top8
full_lookahead
```

### 14.3 Budget ladder

```text
30, 60, 120, 300, 600, 1200 seconds
```

For a separate full-lookahead completion attempt, use 3600 seconds for **all** methods on the subset. Keep the legal action set identical. A timeout is a result.

### 14.4 Required metrics

- success under each budget;
- minimum II where derivable;
- route cost;
- wall time and memory;
- parent and child solves;
- fraction of oracle-best actions retained by top-k;
- selected regret;
- child-solve reduction versus full lookahead;
- expansion reduction/increase;
- solver setup and iteration time;
- model and feature time.

### Required figures

1. quality versus wall-time Pareto plot;
2. success versus budget;
3. top-k oracle recall versus child solves;
4. route cost/minimum II versus compilation time;
5. stage-level runtime by k.

Do not claim top-k retains full-lookahead quality unless the same action groups and completed cells support that comparison.

---

## 15. E8 — Symmetry transition-equivalence and reduction

### 15.1 Generated adversarial instance families

Create small DFG/MRRG instances, generally no more than 8 operations and small 2x2/3x3 architectures, covering:

1. independent equal-opcode operations tied on schedule level and fan-out but separated by stable IDs;
2. symmetric chains and diamonds;
3. recurrence edges and modulo-time constraints;
4. memory operations with symmetric and asymmetric memory ports;
5. mutex-compatible occupancy;
6. broadcasts and same-signal sharing;
7. mux conflicts;
8. temporal rotations/reflections of the architecture;
9. transforms that preserve the graph but not the total operation rank — these must be rejected;
10. transforms that preserve graph/order but violate native resource type, capacity, latency, memory, or timing semantics — these must be rejected.

### 15.2 Per-state assertions

For every reachable state and every candidate transform:

1. verify total operation-rank preservation;
2. verify current prefix and remaining sequence transformation;
3. enumerate the complete legal atomic action set;
4. establish a bijection between actions before and after transformation;
5. compare local scores for length, dual-linear, and proposal within numerical tolerance;
6. apply corresponding actions and verify successor equivariance;
7. verify terminal legality and route-cost invariance.

### 15.3 Exhaustive search comparison

With no beam truncation and no expansion truncation:

- enumerate all reachable states with quotienting off;
- enumerate with quotienting on;
- compare terminal solution orbits;
- compare best legal route cost;
- compare failure/success status;
- compare accepted transform groups.

Pass requires 100% equality of terminal orbit hashes and best costs on every generated instance. Any failure disables the quotient path for paper quality experiments.

### 15.4 Larger-instance reduction

After exactness passes, run quotient on/off for symmetric larger cases and report:

- admitted DFG and architecture group sizes;
- canonicalization time;
- states merged;
- expansions avoided;
- parent and child solves avoided;
- peak memory;
- success and route cost;
- cases where quotient overhead exceeds saved work.

For finite beams, do not require identical heuristic output; report the difference. Exact claims are restricted to exhaustive transition equivalence.

---

## 16. E9 — Closest learned/routing-aware baseline

### Required attempt

Investigate, in order:

1. a usable implementation of the ICCAD 2022 GNN/RL routing-aware CGRA mapper;
2. a usable implementation of Rewire (DAC 2025);
3. another recent executable routing-aware learned or multi-node CGRA mapper with compatible licensing.

### Rules

- Confirm paper identity, repository ownership, license, commit, and benchmark contract.
- Do not use an unrelated repository with a matching generic name.
- Do not claim an exact reproduction if substantial components are reimplemented or substituted.
- Preserve the baseline's native semantics; do not force it into QUOTIENTFLOW's beam interface and still call it the original method.
- Export a common subset only where architecture, DFG, routing, memory, II, and legality semantics can be reconciled.
- Use the same input kernels, architecture subset, external wall budget, and final legality checks where possible.
- Document action atomicity, route semantics, operation scheduling assumptions, search termination, stochastic seeds, and unsupported features.

### If no executable implementation exists

Produce `BASELINE_COMPATIBILITY.md` containing:

- repositories and author pages checked;
- date checked;
- license/source availability;
- exact incompatibilities;
- components that would require a fresh reimplementation;
- why a numeric comparison would not be faithful.

The paper may then provide a precise contract comparison but must not claim superiority.

---

## 17. Required analysis products

Generate these tables from scripts, never by manual editing:

1. full per-cell end-to-end outcome table;
2. success-under-budget summary;
3. minimum-II win/tie/loss table;
4. common-success route-cost table;
5. complete failure census;
6. runtime and memory table;
7. stage-level profiling table;
8. training/split/model disclosure table;
9. held-out action-ranking table;
10. learned-component ablation table;
11. dual knockout/stability table;
12. top-k/full-lookahead table;
13. prefix-sensitivity table;
14. quotient correctness/reduction table;
15. A3 transfer table;
16. closest-baseline compatibility/results table.

Generate these figures in vector PDF plus source CSV:

1. headline success/minimum-II/runtime figure;
2. quality-time Pareto curves;
3. top-k recall and child-solve tradeoff;
4. held-out ranking ablation;
5. dual versus finite-knockout impact;
6. tau/solver/tolerance stability;
7. prefix-depth sensitivity;
8. quotient reductions and overhead;
9. per-kernel/per-architecture regressions.

Every figure must include failures or clearly state its paired denominator.

---

## 18. Result bundle to return

Return one archive with this structure:

```text
quotientflow_results/
  RESULTS_HANDOFF.md
  frozen_manifest.yaml
  environment/
    source_commit.txt
    git_diff.patch
    python_freeze.txt
    system_info.txt
    solver_versions.txt
    hardware_inventory.txt
  configs/
    *.yaml
  splits/
    fold_manifest.yaml
    train_groups.csv
    validation_groups.csv
    test_groups.csv
  checkpoints/
    <fold>/<variant>/<seed>/...
  raw_runs/
    *.json
  logs/
    *.stdout.gz
    *.stderr.gz
  tables/
    end_to_end_runs.parquet
    end_to_end_runs.csv
    action_ranking.parquet
    dual_validation.parquet
    symmetry_assertions.parquet
    failure_census.csv
    training_census.csv
  analysis/
    reproduce_all.py
    statistics.py
    make_tables.py
    make_figures.py
    requirements.txt
  figures/
    *.pdf
    *.csv
  reports/
    ACTION_UNIVERSE_PARITY.md
    RUNTIME_PROFILE.md
    TRAINING_REPORT.md
    DUAL_STABILITY_REPORT.md
    SYMMETRY_EQUIVALENCE_REPORT.md
    BASELINE_COMPATIBILITY.md
```

`RESULTS_HANDOFF.md` must state:

- which gates passed or failed;
- exact experiments completed;
- missing/corrupt runs;
- whether A3 remained locked until freeze;
- primary results with denominators and confidence intervals;
- all regressions and negative results;
- hashes of the manifest, raw table, checkpoints, and analysis scripts;
- one command that regenerates all tables and figures from raw data.

Suggested command contract:

```bash
python analysis/reproduce_all.py \
  --manifest frozen_manifest.yaml \
  --runs tables/end_to_end_runs.parquet \
  --ranking tables/action_ranking.parquet \
  --duals tables/dual_validation.parquet \
  --symmetry tables/symmetry_assertions.parquet \
  --out reproduced/
```

The script must fail on duplicate run IDs, schema violations, missing mandatory cells, hash mismatches, or invalid status/legality combinations.

---

## 19. Minimum evidence needed before revising the paper upward

Do not ask the paper-writing agent to make acceptance-level claims until the returned bundle contains:

1. zero-prefix paired end-to-end results for the mandatory methods;
2. complete failure/timeout census;
3. meaningful held-out ranking improvement over dual-only and length, or an honest negative result;
4. top-k completion on nontrivial cells and a quality-time curve;
5. direct dual knockout and stability measurements;
6. complete training/split/checkpoint disclosure and ablations;
7. 100% passing exhaustive symmetry equivalence tests before quotient claims;
8. held-out A3 results generated only after freeze;
9. a closest-baseline result or a documented incompatibility report;
10. raw data and reproducible table/figure scripts.

