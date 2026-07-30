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
