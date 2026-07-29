# Native Morpher FlowAdvantage proposal

## Frozen artifact

- Model: revision-v3 `residual_gnn`, seed 23 (validation-selected hybrid)
- Checkpoint: `results/revision_v3/checkpoints/residual_gnn_seed_23.pt`
- SHA-256: `468a8ffc541d20efcc77333e8492d4a1fcda88a022a506c9013b809fb991cad8`
- Weights changed: no
- Morpher or pilot labels used: no
- Architecture identity feature used: no

The checkpoint is a hybrid proposal. It predicts the finite-perturbation
correction to the analytical parent-dual baseline. Native inference therefore
requires an exact `flowadvantage_native_parent_relaxation_v1` context. Missing
parent objectives, duals, or slacks are errors; the implementation never
zero-fills solver-derived fields.

## Native graph projection

`flowadvantage/morpher_adapter/native_proposal.py` encodes every stable native
DFG node, every stable resource in the actual II-expanded MRRG, and every
directed native MRRG edge. It does not reconstruct a coordinate-only mesh.
Coordinates, phase, boundary position, degree, occupancy, resource duals and
directed edge attributes are architecture-normalized.

The frozen DFG input has two compute capability channels. Native operations
are projected by declared capabilities:

- multiply or multiply-accumulate -> multiply-compatible channel;
- general integer/floating ALU, shift, logic, compare or select/mux ->
  non-multiply ALU-compatible channel;
- load, store, constant and predicate -> explicit fixed-terminal zero channel;
- unrecognized compute semantics -> unsupported.

The scorer fails before inference when unsupported compute operations exceed
10%. No unknown opcode is replaced with ADD or MUL.

The training model has three phase channels. For arbitrary native II, normalized
phase is projected by piecewise-linear interpolation over the three training
knots. This reproduces the original one-hot representation exactly at II=3 and
does not encode an architecture name.

## Cached inference

The following are cached:

- immutable DFG/MRRG indexes and directed graph tensors by native hashes and II;
- static bounded topology statistics;
- DFG and MRRG encoder outputs by exact partial-state hash and exact
  parent-relaxation semantic hash;
- state-level residual connectivity, demand exposure and slack summaries.

Every candidate action from one state is passed through the action head in one
batched forward pass. The scorer records compatibility, parent-context,
features, both encoders, action head and total wall time separately.

## Scientific boundary

This component ranks native actions only. Exact child relaxation remains behind
the separate `ExactChildActionEvaluator` interface, and the scorer contains no
child-value proxy or fallback. A pilot result must still use native child
relaxation for top-4 reranking and native/independent legality checking for the
complete mapping.

## Authoritative fixed17 smoke

`tests/test_morpher_native_proposal.py` uses:

`results/revision_v5b/reference_mappings/array_add/A0_hycube4x4_standard_fixed17_smoke`

It verifies:

- checkpoint SHA/model type/seed/feature schema;
- capability compatibility and the >10% unsupported fail-closed rule;
- exact parent state/hash checking;
- finite deterministic scores;
- state/encoder cache reuse;
- batched action-head inference;
- mapper scorer-protocol integration;
- operational cold/warm inference bounds.

The deterministic parent values in this unit test exercise the inference
contract only and are not reported as experimental relaxation evidence.
