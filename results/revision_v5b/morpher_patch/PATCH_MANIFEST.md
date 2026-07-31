# FlowAdvantage native bridge patch manifest

Base Morpher commit: `9a9dce7aea521f1d5ef33686f57ca84864edb3c9`.

Validated bridge image: `quotientflow-morpher-v5b-native-fixed11@sha256:629964cd149c9e8879b026bf4c0164ea4d54f3acd733618452db6d5fcf62124d`.

Current legality-validation build: `quotientflow-morpher-v5b-native-fixed15@sha256:ad69877bd76b754f3041422916f1a15bd9cb07b1a520fb7cbe4103a08f64cea4`.

Self-recurrence/fanout validation build:
`quotientflow-morpher-v5b-native-fixed16@sha256:6b4397f60624fd426226362858ae543bdd92a521c52f80fb37b207e5d73d4d02`.

Memory round-trip validation build:
`quotientflow-morpher-v5b-native-fixed17@sha256:4581c2c9a868ffd129609c5b962417e1d8ff3e84cbe2890a6d1364101e901957`.

Deterministic simulated-annealing build:
`quotientflow-morpher-v5b-native-fixed18-sa@sha256:46d19ce00d59b279c4d1e0e7bdcc09072eb13c1ae1d28a33e7bbb1dce65eb33f`.

Exact source diff: `flowadvantage_bridge.patch`

Patch SHA-256: `dc4b4d82a6c127279a682861e2306f6e9ca68b7e3df77f37a50c0655cc1c041b`.

Self-recurrence/fanout fix: `self_recurrence_fanout_fix.patch`

Fix SHA-256: `1b887fb218818979535088a4c2c57c109ec1fb0514f456ac22b5f1b647a1ecd8`.

Memory import identity fix: `memory_import_identity_fix.patch`

Fix SHA-256: `69d6eb2f58e509852d19018b946955805053ed81883360c81a4c24b0e8c8276e`.

Simulator defined-return fix: `hycube_simulator_defined_returns.patch`

Fix SHA-256: `afdd1c71fe7d1062a4564107c2c342192dc4b391cefd4b205dd1bd73ffa8a131`.

The patch adds dump/load flags, actual expanded MRRG export, stable keyed
DFG identities (including duplicate legacy numeric IDs), native operand-edge
semantics, canonical native-ID mapping export, complete source and destination
port resources, explicit operand-mux/conflict metadata, fresh-MRRG external
import, semantic hash guards, directed route validation, native mutex-aware
capacity validation, exact per-FU operation-latency tables, and imported
binary generation.

The bridge patch does not change SA, LISA, architecture semantics, or the
native mapping objective. The separate fixed16 compatibility correction
changes only PathFinder/Heuristic route bookkeeping for a previously
unrepresentable combination: a DFG self recurrence plus another fanout from
the same source. Morpher overloads `(port, src_id)` as both its T-output
sentinel and a real self-edge destination. The correction distinguishes these
cases by the native port name: T retains sentinel behavior, while a non-T port
receives the additional fanout destination. Fixed16 passed the unchanged
20-operation/23-route array smoke with zero independent-legality violations.

Memory-aware native mapping mutates GEP and outer-loop constants using the
selected architecture's variable base addresses before export. Fixed17 applies
that same native `UpdateVariableBaseAddr()` transformation before external
stable-key resolution. This is an ordering correction, not a new identity
heuristic. On the exact A2 GEMM artifact it preserves 52/52 placements, 72/72
ordered routes, and II=8 across export/reimport; both serialized states pass
the independent legality checker with zero violations and the imported state
passes Morpher's native legality checker.

The validated `gemm_nt` round trip preserves all 52
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

## Lossless external-route round trip (Hail Mary continuation)

`lossless_route_roundtrip.patch` adds an explicit `CGRA`-owned map of imported
route records. Morpher's live `routingPorts` relation intentionally retains
only terminal ownership; reconstructing a path from that relation can omit
register/wait resources even though the native legality checker accepts the
mapping. On an external FlowAdvantage import, the bridge now retains each
validated ordered native route record and emits that exact record on the next
state dump. Native-generated mappings still use the original graph traversal.
This changes serialization fidelity only; it does not change placement,
routing, conflict, latency, or simulator semantics.

## Deterministic simulated annealing seed

`simulated_annealing_seed.patch` documents the pinned-source changes. The
post-patch source digests are:

* `include/morpher/mapper/SimulatedAnnealingMapper.h`: `1d0e204a35df263a2c58b3ccb0ccdb263c427564144d42d7d591c00291586be4`
* `src/mapper/SimulatedAnnealingMapper.cpp`: `2cc3159b1b413087cfe9f68270fe96a2427eddc992086822b1685efb55730b03`
* `include/morpher/util/util.h`: `87c25101c4d85f0f3cbc794795c81bbe8e034bc82b14ca37f2e0caa0f9deaa7c`
* `src/CGRA_xml_compiler.cpp`: `13685c3d1c4821f5c6be5312974cdd80e477f1b50ab00114e3a3d166e994b4d5`

`--seed N` is accepted for native simulated annealing (`-m 1`); `N=0` or
omission preserves the pre-patch random-device/time behavior. When present,
acceptance draws, node selection, parent selection, and candidate-destination
shuffles all consume the mapper-owned `std::mt19937` stream. No SA objective,
temperature schedule, or architecture behavior changed.

The fixed18 smoke mapped the native 20-operation `array_add` DFG at II=4
twice with `--seed 11 -r 100`. Both runs produced byte-identical
`mapping.json` files (SHA-256
`5a88d356b6045cde086097c58ed8a530427d4d7a51e08a0ac4bc201ff3b601af`)
and identical operation placements, ordered routes, memory bindings, and II.
