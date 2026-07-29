# FlowAdvantage Paper Results Index

Generated: `2026-07-29T12:52:00Z`  
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

## Queue status (current)

- The paper-suite v2 manifest is intentionally not launched: native
  executable-input provenance is still being completed, and no native
  baseline row is claimed from canonical JSON alone.
- Latest immutable cluster status: `9 DONE`, `7 RUNNING`, `14 QUEUED`,
  `0 ERROR`, `0 TIMEOUT`. Completed exports are pulled only after schema and
  independent legality validation.
- Local end-to-end calibration (not headline paper evidence): the empty-state
  40% length-prefix job completed as a structured `VALID_MAPPING_FAILURE`
  after mapping 15/20 operations (145.5 s); the matching dual-linear job
  completed as a structured valid failure after four parent-relaxation
  attempts (427.2 s). Both are retained under
  `results/revision_v5b/pilot_queue_depth40_{length_v1,dual_frontier_v1}`.
- The 60% dual-linear calibration likewise completed as a structured valid
  failure after the frontier-preserving repair; it did not produce an
  infrastructure error.

Timeouts, errors, and unsupported rows are retained and excluded from quality
aggregates. Borderline values remain valid evidence and are marked AMBER in
the gate tables; no threshold is changed after observing results.

## Artifact index

- Grounding: [docs/HPCA_PAPER_RIGOR_GROUNDING.md](docs/HPCA_PAPER_RIGOR_GROUNDING.md)
- Execution plan: [docs/HPCA_PAPER_SUITE_PLAN.md](docs/HPCA_PAPER_SUITE_PLAN.md)
- Frozen config: [configs/paper_suite_v2.yaml](configs/paper_suite_v2.yaml)
- Native export provenance: `results/revision_v5b/native_input_registry.json`
- Atomic calibration manifests: `results/revision_v5b/pilot_queue_depth40_{length_v1,dual_frontier_v1}/`
- Validation protocol: `results/revision_v5b/PAPER_MAPPING_PROTOCOL.md`
- Paper queue launch remains gated on complete native-input/toolchain
  provenance and the required baseline smoke.

## Current interpretation

The real Morpher bridge is not considered paper-ready until native legality,
independent legality, route/placement/II round-trip, and cycle-simulation
checks pass on the frozen reference corpus. A mapping timeout is operational
evidence, not a scientific mapping failure.
