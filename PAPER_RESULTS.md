# FlowAdvantage Paper Results Index

Generated: `2026-07-29T01:35:46.751698+05:30`  
Deadline: `2026-07-31 23:59 IST`  
Hours remaining: **70.39**

## Paper-readiness progress

**[██░░░░░░░░░░░░░░░░░░░░░░] 10.0%** (fixed claim-critical weighted denominator)

| Evidence obligation | Completion |
|---|---:|
| Adapter and functional correctness | 0.0% |
| Complete 12×4 principal matrix | 0.0% |
| Conventional baseline coverage | 0.0% |
| Lookahead and parent ablations | 0.0% |
| Portability and scaling | 0.0% |
| Runtime and statistics | 0.0% |
| Artifacts and reporting | 100.0% |

## Queue status

- Registered work items: `1164`
- Terminal work items: `0`
- Quality-terminal items: `0`
- Status counts: `PENDING=1164`

Timeouts, errors, and unsupported rows are retained and excluded from quality
aggregates. Borderline values remain valid evidence and are marked AMBER in
the gate tables; no threshold is changed after observing results.

## Artifact index

- Grounding: [docs/HPCA_PAPER_RIGOR_GROUNDING.md](docs/HPCA_PAPER_RIGOR_GROUNDING.md)
- Execution plan: [docs/HPCA_PAPER_SUITE_PLAN.md](docs/HPCA_PAPER_SUITE_PLAN.md)
- Frozen config: [configs/paper_suite_v1.yaml](configs/paper_suite_v1.yaml)
- Atomic manifests: `results/paper_suite_v1/work_manifest.csv`, `results/paper_suite_v1/work_items/`
- Raw results: `results/paper_suite_v1/raw/`
- Tables: `results/paper_suite_v1/tables/`
- Plots: `results/paper_suite_v1/plots/`
- Checkpoints and logs: `results/paper_suite_v1/checkpoints/`, `results/paper_suite_v1/logs/`

## Current interpretation

The real Morpher bridge is not considered paper-ready until native legality,
independent legality, route/placement/II round-trip, and cycle-simulation
checks pass on the frozen reference corpus. A mapping timeout is operational
evidence, not a scientific mapping failure.
