# FlowAdvantage native bridge patch manifest

Base Morpher commit: `9a9dce7aea521f1d5ef33686f57ca84864edb3c9`.

Validated bridge image: `quotientflow-morpher-v5b-native-fixed11@sha256:629964cd149c9e8879b026bf4c0164ea4d54f3acd733618452db6d5fcf62124d`.

Current legality-validation build: `quotientflow-morpher-v5b-native-fixed15@sha256:ad69877bd76b754f3041422916f1a15bd9cb07b1a520fb7cbe4103a08f64cea4`.

Exact source diff: `flowadvantage_bridge.patch`

Patch SHA-256: `22e4311a1c7b3b021bc5047aa962b9fa6ce248b5c91f33e4168cf5bf1556db4b`.

The patch adds dump/load flags, actual expanded MRRG export, stable keyed
DFG identities (including duplicate legacy numeric IDs), native operand-edge
semantics, canonical native-ID mapping export, complete source and destination
port resources, explicit operand-mux/conflict metadata, fresh-MRRG external
import, semantic hash guards, directed route validation, native mutex-aware
capacity validation, and imported binary generation.

It does not change PathFinder, SA, LISA, architecture semantics, or the native
mapping objective. The validated `gemm_nt` round trip preserves all 52
placements, all 72 physical routes, and II=8 exactly; its single PS pseudo
dependency remains in the DFG contract and is correctly excluded from the
physical route universe.

The compatibility build also removes two undefined/crashing behaviors exposed
by Morpher's own checked-in `fix_fft` DFG: a missing predicate `NPB` attribute
now deterministically defaults to the native non-negated value `0`, and a
candidate whose recurrence anchor is temporarily unmapped by backtracking
defers the check instead of dereferencing a null placement. External mapping
import then verifies both the original recurrence-anchor inequality and every
loop-carried dependency's destination-latency bound against the complete
ordered route latency sequence.
