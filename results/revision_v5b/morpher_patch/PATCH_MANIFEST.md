# FlowAdvantage native bridge patch manifest

Base Morpher commit: `9a9dce7aea521f1d5ef33686f57ca84864edb3c9`.

Validated image: `quotientflow-morpher-v5b-native-fixed11@sha256:629964cd149c9e8879b026bf4c0164ea4d54f3acd733618452db6d5fcf62124d`.

Exact source diff: `flowadvantage_bridge.patch`

Patch SHA-256: `aa9a839517d530826385b1b75187d9144e090cdfbf7cbf39a3b74345504f1925`.

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
