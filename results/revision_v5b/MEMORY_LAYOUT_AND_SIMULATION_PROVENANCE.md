# Morpher memory-layout and simulation provenance

## Resolved A2 mapping-layout blocker

The checked-in `gemm_nt` and `fix_fft` DFGs do have authoritative memory
layouts at the pinned Morpher mapper submodule commit
`706b0b4ebc3a5569dfdd7b7578803e1d7ced9505`:

* `Morpher_CGRA_Mapper/json_arch/gemm_various_mem_archs/`
  `stdnoc_mem_dual_port_two_banked.json`
* `Morpher_CGRA_Mapper/json_arch/fft_various_mem_archs/`
  `stdnoc_mem_dual_port_two_banked.json`

These files contain every pointer name used by the corresponding checked-in
DFG, including the compiler-versioned scalar names (`mul12`, `add49`,
`shl29`, `conv31252`, and `conv.i241`). They therefore supersede the earlier
classification that the memory layouts were unavailable.

`scripts/derive_morpher_mem_alloc.py` converts the source architecture's
bank-local addresses to the flat address convention consumed by Morpher's
official `update_mem_alloc.py`:

```text
flat_address = source_bank * 2048 + source_local_address
```

The converter rejects missing DFG variables, duplicate variables across
banks, noninteger addresses, noncontiguous bank indices, and out-of-bank
addresses. It records the source SHA-256 and every conversion in:

* `reference_mappings/fix_fft/A2_hycube4x4_mem_variant_fixed14_generated/`
  `memory_layout_provenance.json`
* `reference_mappings/gemm_nt/A2_hycube4x4_mem_variant_fixed14_generated/`
  `memory_layout_provenance.json`

The resulting `architecture_exact_layout.json` files are generated from the
frozen A2 `hycube_original_updatemem.json` template with the upstream
`update_mem_alloc.py`. No variable, bank assignment, local address, or value
is inferred from a mapping outcome.

## Functional trace provenance

The pinned DFG-generator checkout no longer contains the FFT reference source
and traces, but its Git object database retains the upstream artifact at:

```text
Morpher_DFG_Generator commit:
4f3da1acd89c475b101c76f772351330c2f95bf5
path:
benchmarks/morpher_benchmarks/fft/
```

The full upstream directory (source, generated DFG diagnostics, memory
allocation, and 127 traces) is preserved byte-for-byte under:

```text
reference_mappings/fix_fft/source_provenance_4f3da1a/
```

The compiler-versioned SSA names in that historical trace corpus differ from
the pinned mapper DFG as follows:

```text
conv.i238 -> conv.i241
conv30249 -> conv31252
l.1250    -> l.1253
shl28     -> shl29
```

The mapping is explicit in `ssa_name_map_to_pinned_dfg.json`. The names match
the same semantic values in the upstream source/DFG diagnostics; stable
variable names (`fr`, `fi`, `conv.i`, and `manupa1`) remain unchanged.

`scripts/translate_morpher_trace_names.py` applies only this name
substitution. It copies offsets and pre/post values verbatim, requires every
translated name to occur in the authoritative target allocation, and writes
per-file source/output hashes to `trace_translation_manifest.json`.

Functional simulation may be claimed only if the fixed A2 mapping emits a
native HyCUBE bitstream and the translated upstream traces produce zero
mismatches. The presence of the source/traces or a successful mapping alone
is not counted as simulation validation.

No corresponding source-and-trace corpus for the exact checked-in `gemm_nt`
DFG was found in the pinned repository or its reachable mapper/generator
history. GEMM mapping and legality validation remain valid, but functional
cycle-simulation correctness must remain unassessed unless an exact
source/trace artifact is recovered.

## Native A2 mapping and round trip

The memory-layout repair has now been exercised by the authoritative GCC 7
mapper binary on the cluster:

| Kernel | II | Operations | Physical routes | Independent legality |
|---|---:|---:|---:|---|
| `gemm_nt` | 8 | 52 | 72 | legal, zero violations |
| `fix_fft` | 5 | 58 | 77 | legal, zero violations |

The FFT DFG exposed an upstream `routingPorts` bookkeeping collision. Morpher
uses `(T_port, source_node_id)` as an operation-output sentinel, but a real
self-recurrence also uses `source_node_id` as its route destination. A second
fanout through the same non-T routing port was therefore mistaken for the
sentinel and aborted. `morpher_patch/self_recurrence_fanout_fix.patch`
distinguishes the two cases using the native port type; it does not alter
candidate costs, routing capacity, or the mapping objective.

Memory-aware export also records load/store constants after native
`UpdateVariableBaseAddr()`, while the external importer previously resolved
node keys before that native update. The importer now applies the same update
before identity resolution
(`morpher_patch/memory_import_identity_fix.patch`). Fixed17 preserves:

* GEMM: 52/52 placements, 72/72 ordered routes, II 8;
* FFT: 58/58 placements, 77/77 ordered routes, II 5.

Both reimports pass Morpher native legality and the independent checker.

Cluster artifacts:

* `cluster/gemm_nt_A2_cluster_reference/`
* `cluster/fix_fft_A2_fixed16_cluster_reference/`

## Cycle-simulator diagnosis

The upstream simulator had four non-void functions that fell through without
returning. On the host compiler this manifested as a stack-smash in
`CGRA::parseCMEM`; AddressSanitizer localized the undefined behavior.
`morpher_patch/hycube_simulator_defined_returns.patch` supplies defined
returns and converts the impossible crossbar-selector default into a
structured exception.

With that runtime defect repaired, both the native FFT bitstream and the
round-tripped FFT bitstream execute but do not reach the loop-end memory cell
when driven by the historical trace. Both arrangements supported by the
simulator (`-t 1` and `-t 2`) were tested. This is consistent with the
remaining provenance mismatch: the trace comes from an older generator
commit, while the mapped PartPred DFG is from the pinned mapper checkout.
Renaming SSA variables is sufficient for allocation parsing but does not prove
compiler-version execution equivalence.

Accordingly:

* real A2 mapping, legality, export, and reimport are validated;
* the historical trace timeout is not counted as a mapping failure;
* FFT functional correctness remains unassessed until traces are generated
  from the exact pinned DFG/source pipeline;
* no mismatch count or passing simulation is fabricated.
