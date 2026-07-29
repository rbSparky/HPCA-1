# Revision-v5b negative legality validation

The eight required illegal mapping mutations were generated from the native
`gemm_nt` A0 export at II=8. This reference was selected because it contains
MUL, memory, and loop-carried dependencies in one native mapping. Each mutated
mapping was checked independently by:

1. `flowadvantage.morpher_adapter.legality_bridge.validate_mapping`, operating
   only on serialized DFG/MRRG/mapping contracts; and
2. Morpher fixed15, reconstructing a fresh native MRRG and executing
   `ImportFlowAdvantageMapping` plus `ValidateFlowAdvantageMapping`.

All 8/8 mutations were rejected by both checkers through structured validation
paths. No crash, timeout, or nonzero infrastructure exit was credited as a
rejection.

| Mutation | Independent expected violation observed | Native rejection |
|---|---:|---:|
| Duplicate compute occupancy | yes | yes |
| Disconnected directed route | yes | yes |
| MUL bound to incompatible resource | yes | yes |
| Schedule outside ASAP/ALAP | yes | yes |
| Loop-carried recurrence overrun | yes | yes |
| Memory operation on inaccessible compute PE | yes | yes |
| Wrong architecture hash | yes | yes |
| Wrong II | yes | yes |

Raw, unrounded results are in
`raw/negative_legality_tests.csv`. Each mutation has an atomic evidence
directory under `negative_legality/` containing the exact mutated mapping,
native stdout, native stderr, and `result.json`.

The authoritative command is:

```bash
/home/rishabh/miniconda/envs/taugat_pyg/bin/python \
  scripts/run_v5b_negative_legality.py
```
