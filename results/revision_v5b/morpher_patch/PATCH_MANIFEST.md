# FlowAdvantage native bridge patch manifest

Base Morpher commit: `9a9dce7aea521f1d5ef33686f57ca84864edb3c9`.

Validated bridge image: `quotientflow-morpher-v5b-native-fixed11@sha256:629964cd149c9e8879b026bf4c0164ea4d54f3acd733618452db6d5fcf62124d`.

Current compatibility build: `quotientflow-morpher-v5b-native-fixed13@sha256:e74612d9705b4c4ee04ccd8ea715be101dc00db23b0d2cd7b868ba77d756bc3c`.

Exact source diff: `flowadvantage_bridge.patch`

Patch SHA-256: `97022329d617deb612d46a5a3bc98d8901cb596a16ce665e249da9ae7572844b`.

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
candidate whose recurrence anchor is temporarily unmapped by backtracking is
conservatively rejected instead of dereferencing a null placement.
