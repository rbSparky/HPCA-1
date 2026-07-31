# Hail-Mary configuration deviations

This file is frozen before the new headline test matrix is executed. It records
differences from the tentative `EXPERIMENT_EXECUTION_SPEC_HPCA1.md`; none are
responses to observed Hail-Mary test outcomes.

| Item | Tentative maximum | Frozen rescue value | Reason |
|---|---:|---:|---|
| Deadline | 31 July 23:59 IST | 1 August 10:00 IST | Author-provided extension. |
| Pilot kernels | 10 | 6 predeclared minimum kernels | Complete paired coverage is more defensible than an incomplete broad matrix. |
| II primary grid | `LB..LB+6`, optional `+10` | `LB..LB+4` | Matches the previously declared real-pilot search contract and bounds each paired method to 600 seconds. |
| Simulated-annealing seeds | 10 | 11, 23, 37 | Matches the current manuscript and existing deterministic Morpher seed patch; all seed rows remain separate. |
| Learned-model seeds | 5 new | frozen revision-v3 seed 23 headline; existing seeds 11/23/37 disclosed | The real evaluation is zero-shot and must not be tuned or retrained on pilot-test outcomes. |
| Held-out label states | 1,000 target | existing complete synthetic corpus plus a bounded balanced native pilot | The maximum sweep cannot complete before the deadline; the smaller native study is labeled limited-power. |
| Deterministic timing repeats | every cell | six balanced cells | Mapping outcomes remain deterministic; repeats estimate timing variation without displacing paired coverage. |

Timeouts, unsupported cells, solver rejections, and validation failures remain
distinct from valid mapping failure. No gate or threshold is changed here.

## Runtime-engineering amendments (recorded before the restarted queue)

| Item | Original execution value | Relaunch value | Reason |
|---|---:|---:|---|
| Flow work-item wall limit | 600 s | 3,600 s | The first native root relaxation was demonstrably progressing inside Clarabel; the shorter limit would censor valid work. The terminal status remains `TIMEOUT` only after the longer limit. |
| Relaxation solver request limit | 120 s | 600 s | Avoid prematurely terminating a large but resource-bounded native root solve; solver status and wall time remain recorded separately. |
| Native relaxation formulation | dense CVXPY masks/max epigraphs | sparse indexed capacity operators and paired outgoing/incoming inequalities | Exact algebraic equivalence; verified by native relaxation tests and documented in the runtime log. |
| Problem export workers | 2 | 4 per export wave | Host has 8 CPUs and each native export is single-threaded; no semantic change. |
| Export scheduling | all pairs in one wave | resumable waves with selective kernel/architecture filters | Prevents a known incompatible memory-architecture cell from blocking unrelated exports; unsupported cells are retained with stderr. |

### Native schedule interval padding

The restarted native pilot sets `relaxation_extra_ii_periods=0` explicitly in
each immutable work item. Morpher exports finite ASAP/ALAP latency bounds, so
adding an artificial extra II period only creates placement variables outside
the exported schedule contract and dominates the root relaxation size. The
general library default remains unchanged for callers that need padding; this
pilot-only interval choice changes neither the native legality checker nor the
relaxation objective, and is kept in the immutable manifest/config hash.

The pilot also permits an explicit one-period fallback only when the zero-pad
interval proves that a dependency has no legal temporal corridor. This avoids
declaring a structurally feasible Morpher state infeasible while retaining the
fast zero-pad path for cases where the exported ASAP/ALAP interval is complete.
Fallback use is counted per work item (`schedule_padding_fallbacks`) and is not
silently merged with the zero-pad timing configuration.
# 2026-07-31 runtime rescue additions

## Explicit Clarabel internal parallelism benchmark

The frozen baseline remains `relaxation_max_threads=1`. A separate, append-only
benchmark uses `relaxation_max_threads=8` with `OMP_NUM_THREADS=1` and two outer
workers. This changes only solver parallelism; it does not change the objective,
constraints, tolerances, candidate universe, or legality rules. The benchmark
heartbeat/logs are preserved under the cluster `bench_mt8` directory and are not
included in headline aggregates until numerical-equivalence checks pass.

## Static-root proposal variant

Because dynamic parent re-solving remained the dominant wall-time cost, an
explicit `*_static` variant was added. It solves the exact empty-root
relaxation once per semantic problem/II and reuses its dual context while the
discrete mapper advances. It is reported separately from the frozen dynamic
methods and is not substituted silently. Static-root rows carry
`static_parent=true`, a distinct method name, and a shared semantic cache.

## Morpher route canonicalization

The stable GCC7 Morpher importer can legally collapse a redundant directed
route detour while preserving II, operation placement, endpoints/timing, and
native legality. Strict `route_match` remains recorded as false in validation;
`native_contract_match` is true only when II and placement are unchanged, and
`native_route_canonicalized` is reported separately. No such row is presented
as byte-for-byte route equality.

## A2 rest-kernel memory-layout repair

The first native export attempt for `fix_fft`, `gemm_nt`, `hpcg`, and `trmm` on
`A2_hycube4x4_mem_variant` aborted in
`PathFinderMapper::UpdateVariableBaseAddr`. The DFGs did contain
`BasePointerName` tags; the failure was the stronger native contract that
*every* pointer name must also occur in the selected architecture's
`SPM_B0_WRAPPER`/`SPM_B1_WRAPPER.DATA_LAYOUT`. The historical array_add JSON
only declared `A`, `B`, `C`, `loopstart`, and `loopend`.

This was repaired rather than marked unsupported. The committed
`scripts/repair_a2_data_layout.py` extracts the native pointer names and byte
sizes, preserves the validated A2 topology and loop sentinels, and allocates
missing objects deterministically and without overlap across the two existing
2048-byte scratchpad banks. Repaired layouts and exports are in
`results/hail_mary_hpca1/a2_repaired_arch/` and
`results/hail_mary_hpca1/a2_repaired_exports/`; all four exports pass the
native Morpher problem-export command at their actual lower-bound II. The
original abort logs remain preserved as failure provenance, while the clean
queue uses the repaired per-kernel native architecture paths.

## Remote scientific-runtime repair

The cluster environment initially reported NumPy `2.3.5` with SciPy `1.14.1`,
whose supported NumPy range is `<2.3`; every worker emitted the compatibility
warning. Before the cache-warm repaired queue, NumPy was pinned to the
compatible `2.2.6` build in the existing environment. No solver, mapper,
model, architecture, or result semantics changed; subsequent workers use the
corrected numerical ABI.
