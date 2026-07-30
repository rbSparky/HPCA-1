# Exact symmetry transition-equivalence validation

This is an exhaustive transition-system check, not merely a graph-isomorphism
or final-outcome comparison.  Only transforms preserving the full semantic
resource contract and the mapper's total operation order are admitted.

| Case | Reachable states | Admitted transforms | Rejected transforms | Checked actions | Pass |
|---|---:|---:|---:|---:|---|
| tied_priority | 1409 | 3 | 7 | 4224 | True |
| recurrence | 53 | 3 | 2 | 156 | True |
| memory | 129 | 2 | 3 | 256 | True |

All admitted transforms were checked for legal-action bijection, invariant
scoring, commuting successors, and equal terminal legality/cost.  The cases
exercise tied priorities, recurrence-distance metadata, memory compatibility,
time-expanded resources, mutex groups, and broadcast annotations.  Temporal or
resource transforms that change those attributes are rejected.

Crucially, the tied-priority branch swap is a valid typed-DFG automorphism but
is rejected because it does not preserve the stable node-ID tie-break in the
sequential operation order.  QuotientFlow must therefore not claim that DFG
transform as an exact search symmetry without changing the transition policy.

Raw evidence:

- `raw/transition_equivalence.csv`
- `raw/rejected_transforms.csv`
- `summary.json`
