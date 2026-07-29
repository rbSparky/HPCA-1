# Real FlowAdvantage probe (non-headline calibration)

This document records the first completed, atomic, compiler-generated native
contract probe. It is not the balanced paper suite and is not used to tune a
model or a gate.

## Frozen inputs

- Kernel: `trmm`
- Architectures: `A0_hycube4x4`, `A2_hycube4x4_mem_variant`
- Initialization: deterministic length prefix, depth fraction `0.4`
- Beam: width `4`, action limit `24`, max expansions `5000`
- Relaxation: native fractional residual, `tau=0.001`, Clarabel primary,
  OSQP fallback, 120-second per relaxation limit
- Worker: one spawned process, BLAS/OpenMP thread limits set to one,
  600-second atomic wall limit

## Terminal rows

| kernel | architecture | method | status | mapped ops | route cost | parent solves | child solves | worker wall (s) |
|---|---|---|---|---:|---:|---:|---:|---:|
| trmm | A0_hycube4x4 | length | DONE | 11/11 | 71.0 | 0 | 0 | 117.44 |
| trmm | A2_hycube4x4_mem_variant | length | DONE | 11/11 | 71.0 | 0 | 0 | 146.61 |
| trmm | A0_hycube4x4 | dual_linear | TIMEOUT | 6/11 | — | 4 | 0 | 601.33 |
| trmm | A2_hycube4x4_mem_variant | dual_linear | TIMEOUT | 6/11 | — | 4 | 0 | 602.22 |
| trmm | A0_hycube4x4 | flow_proposal | DONE | 11/11 | 71.0 | 22 | 0 | 468.48 |
| trmm | A2_hycube4x4_mem_variant | flow_proposal | DONE | 11/11 | 71.0 | 22 | 0 | 481.49 |
| trmm | A0_hycube4x4 | flow_top4 | TIMEOUT | 5/11 | 0 | 0 | 602.12 |
| trmm | A2_hycube4x4_mem_variant | flow_top4 | TIMEOUT | 5/11 | 0 | 0 | 602.00 |

Raw atomic manifests and heartbeats are preserved under:

```text
results/paper_suite_v2_length_probe_v1/run/
results/paper_suite_v2_flow_probe_v1/run/
```

## Interpretation

Both length and proposal-only completed legal mappings with identical route
cost on this pair. The proposal-only method used 22 parent relaxations and was
approximately 3.9x the length wall time on A0 and 3.3x on A2. The dual-linear
rows reached the hard boundary after four parent solves and are operational
timeouts, not mapping failures. Top-4 exact correction also reached the hard
boundary before a complete result; no top-4 quality claim is made from these
rows.

The probe therefore validates atomic execution and exposes the dominant real
contract cost (repeated native CVXPY parent/child solves), but it does not
establish real-kernel mapping utility. The next optimization target is solver
reuse/compilation cost, followed by a balanced paired run.
