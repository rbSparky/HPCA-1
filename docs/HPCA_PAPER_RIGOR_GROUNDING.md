# HPCA Paper-Rigor Grounding for FlowAdvantage

Status: working evaluation contract, frozen before the real pilot.

Source archive: `HPCA.zip` (17 unique PDFs; the Cambricon-DG copy was
duplicated, and the two VeloxGNN files were retained as separate byte-level
versions for audit). The archive was converted to text and inspected for
experimental setup, baseline, validation, ablation, scaling, simulator,
synthesis, and artifact sections on 2026-07-29 IST.

## What accepted HPCA papers establish

The papers do not rely on one attractive average number. They establish a
chain from a precise mechanism to a reproducible implementation, a diverse
workload/architecture matrix, strong matched baselines, and an explanation of
where the gain comes from.

### Directly relevant precedent: LISA

LISA is the closest methodological precedent. It evaluates 12 PolyBench DFGs
over six accelerator organizations and 71 architecture/application
combinations. It compares against ILP and simulated annealing, repeats SA
three times and reports the median, evaluates mapping II and compilation
time, tests architecture portability, measures GNN-label accuracy, and runs
component-effectiveness experiments. Its artifact includes source, scripts,
benchmarks, Docker, result files, and expected reproduction tolerances. Its
mapping experiments use generous target-II time limits, and failed mappings
are explicitly represented rather than silently removed.

This gives FlowAdvantage a minimum compiler-paper standard: a real DFG and
MRRG matrix, at least two strong conventional mappers, repeated stochastic
baselines, matched budgets, II and compile-time outcomes, portability holds,
component ablations, and an executable artifact.

### Hardware-evaluation precedent

HyGCN, SGCN, GCNAX, ReGNN, and related accelerator papers combine several
real datasets or workload families with multiple model configurations,
cycle-accurate simulation, memory-system modeling, and (when the contribution
changes hardware) RTL synthesis, area, power, and critical-path reporting.
SGCN explicitly validates simulator timing against HDL and reports nine
datasets, ablations, layer/cache sensitivity, and a breakdown of energy.
HyGCN reports multiple datasets and GNN models, PyG CPU/GPU baselines,
cycle-accurate execution, Ramulator, RTL synthesis, PrimeTime, and CACTI.

FlowAdvantage does not propose new hardware, so it must not invent an area or
power claim. The analogous consequences are minimum II, steady-state cycles,
PE/routing/register/memory utilization, route cost, routing failures, and
compiler wall time. If an existing Morpher energy model is not validated for a
configuration, energy is marked unavailable instead of extrapolated.

### Broader GNN-paper precedent

AutoGNN, BeaconGNN, Buffalo, Celeritas, Mithril, PruneGNN, VeloxGNN, and the
scaling paper consistently report: named real datasets with sizes and
structural statistics; multiple GNN models or data regimes; end-to-end time
rather than only kernel time; memory/transfer effects; scaling curves;
sensitivity or ablation studies; and a clear hardware/software baseline.
Claims are tied to the exact machine, software version, precision, epochs,
memory size, and measurement method. Robustness datasets are held out from
design choices where possible.

## FlowAdvantage evidence contract

### Claims and required evidence

| Claim | Required evidence |
|---|---|
| Native integration is correct | Native DFG/MRRG/mapping export, FlowAdvantage import, independent legality, native re-import, matching II/placement/routes, and cycle-simulation output checks. |
| Top-4 retains relaxed lookahead | Fixed paired top-4/full-lookahead subset, identical candidate universe, exact child-solve counts, route cost, II, success, and wall time. |
| FlowAdvantage improves mapping | PathFinder and three-seed SA on identical kernel/architecture/II/budget cells; success, minimum II, route cost, failures, expansions, and compile time. |
| The proposal is portable | A3 held out from training/label generation/tuning; report A0–A2 versus A3 per-kernel and aggregate retention. |
| Parent relaxation is needed or avoidable | Hybrid and true no-parent proposal under identical candidates, child correction, budgets, and paired pairs. |
| Runtime is practical | Cold/warm cache decomposition, feature/GNN/parent/child/routing time, CPU time versus wall time, peak RSS, cache hits, and worker concurrency. |
| The result is reproducible | Atomic raw rows, config/source/checkpoint/architecture/kernel hashes, solver versions, seeds, environment manifest, queue logs, and exact table/figure generators. |

### Frozen core matrix

The first defensible matrix is 12 kernels × four architectures: A0 HyCUBE,
A1 standard NoC, A2 constrained-memory HyCUBE, and held-out A3 8×8 HyCUBE.
The frozen kernel order is `array_add`, `array_cond`, `hpcg`, `trmm`,
`gemm_nt`, `fix_fft`, `fir`, `conv2`, `conv3`, `mac`, `matrixmultiply`, and
`dct`. Unsupported or unextractable kernels are replaced only from a frozen
fallback list, using semantic failure evidence recorded before FlowAdvantage
outcomes are inspected.

Mandatory methods are PathFinder, three-seed simulated annealing, dual-linear,
frozen proposal-only, frozen top-4 exact reranking, and the frozen no-parent
top-4 variant. Full relaxed lookahead is evaluated on the fixed
`array_add`/`gemm_nt`/`fix_fft` × A0–A3 subset.

### Metrics

Every terminal row records legal success, valid mapping failure, budget
exhaustion, infrastructure timeout, error, unsupported status, minimum II,
lower-bound II, route cost, routing failures, backtracks, expansions,
generated actions, parent/child solves, feature time, proposal time,
relaxation time, native mapper time, simulation time, total wall time, CPU
time, cache hits/misses, peak RSS, legality status, and output-match status.

Success-under-budget and algorithmic mapping success are reported separately.
Infrastructure timeout is never silently recoded as an empirical mapping
failure. All aggregate comparisons use identical paired terminal sets.

### Statistical and borderline policy

Use 10,000 deterministic paired-bootstrap resamples with seed 24072026 for
success, II, route cost, failed routes, and compile time. Report the point
estimate, 95% interval, paired count, and per-family/per-architecture values.
If an estimate is close to a gate or its interval crosses a gate, label it
`BORDERLINE/AMBER`; preserve the numerical value and uncertainty. “Borderline”
does not mean invalid, and it never licenses changing a fixed threshold.

### Ablations required for a paper claim

The claim-critical ablations are: proposal-only versus top-2/top-4/top-6;
full relaxed lookahead; hybrid versus no-parent; no on-policy training; no
architecture normalization; no alternative-path features; no residual
correction; candidate caps 12/16/20/24; parent refresh frequency; A3 holdout;
kernel-family holdout; and cold versus warm caches. Small action-level
ablations may use the full labeled state set, while mapping ablations use a
fixed balanced subset and are never tuned on test outcomes.

### What would be insufficient

The following cannot support a paper headline: synthetic-only mapping,
prediction Spearman without mapping outcomes, one easy kernel, one
architecture, one stochastic seed, best-of-seed reporting, unequal method
coverage, missing timeouts, a manually reconstructed MRRG, parser-only
round-trips, a binary that was not regenerated by native Morpher, or a
simulator result with mismatched trace/allocation semantics.

## Acceptance checklist

- [ ] Native adapter gate passes with structured negative tests.
- [ ] At least ten supported real kernels and all four architectures.
- [ ] PathFinder, three-seed SA, and FlowAdvantage top-4 have paired core rows.
- [ ] Full-lookahead retention and parent/no-parent results are paired.
- [ ] A3 portability is evaluated without retraining or label regeneration.
- [ ] Cycle simulation validates at least six pairs with matching output.
- [ ] All failures, unsupported cases, timeouts, and retries are visible.
- [ ] Bootstrap intervals, ablations, scaling, runtime decomposition, tables,
      figures, raw manifests, hashes, and reproduction scripts are complete.
