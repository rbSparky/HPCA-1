# A2 rest-kernel export repair

The initial A2 failures were native contract failures, not absent DFG metadata. `PathFinderMapper::UpdateVariableBaseAddr()` requires every DFG `BasePointerName` to have an entry in the selected architecture `DATA_LAYOUT`. The historical array_add memory JSON only declared its own pointers.

The repair uses `scripts/repair_a2_data_layout.py`: it preserves the validated A2 topology and the native loop sentinels (`loopend=2047`, `loopstart=4094`), extracts each pointer name and byte size from the native DFG, and allocates missing objects deterministically into the two existing 2048-byte scratchpad banks without overlap. No synthetic MRRG or arbitrary opcode mapping is used.

| kernel | added pointers | bank footprints | export |
|---|---|---|---|
| `fix_fft` | conv.i(4B,B1@2052), conv.i241(4B,B1@2056), conv31252(4B,B1@2060), fi(128B,B1@2064), fr(128B,B0@84), l.1253(4B,B1@2192), manupa1(2B,B1@2196), shl29(4B,B1@2200) | B0=212B, B1=156B | PASS at II 4 |
| `gemm_nt` | ALPHA(4B,B1@2052), add49(4B,B1@2056), mul(4B,B1@2060), mul12(4B,B1@2064), shr(4B,B1@2068) | B0=81B, B1=24B | PASS at II 6 |
| `hpcg` | manupa1(4B,B1@2052), manupa11(4B,B1@2056), x(16B,B1@2060) | B0=81B, B1=28B | PASS at II 4 |
| `trmm` | manupa4(4B,B1@2052) | B0=81B, B1=8B | PASS at II 4 |

The original abort logs remain immutable provenance in the prior export attempt directories. The repaired exports are used by the fresh queue; no failed attempt is reclassified as a mapping result.
