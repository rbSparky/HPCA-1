# Pinned Morpher fixed14 cluster toolchain

Date: 2026-07-29  
Host alias: `mll5090`  
Host: Ubuntu 24.04.2, Linux 6.8.0-84, AMD Threadripper PRO 7965WX
(24 physical cores / 48 threads), 125 GiB RAM.

## Current authoritative runtime

The cluster has no Docker, Podman, Apptainer, Singularity, or root access.
Recompiling the legacy mapper with host GCC 13 is not an equivalent runtime:
although it builds, the array-add native smoke crashes in legacy undefined
behavior before parsing completes. That build remains recorded only as a
portability diagnostic.

The current authoritative runtime uses the exact GCC-7 executable extracted
from the locally validated fixed17 Docker image, on the cluster host's
backward-compatible glibc/libstdc++ runtime. Fixed17 accumulates the validated
native JSON bridge, self-recurrence/fanout disambiguation, memory-import
identity ordering, and native FU operation-latency export:

```text
Morpher base commit:
  9a9dce7aea521f1d5ef33686f57ca84864edb3c9
Bridge patch SHA-256:
  dc4b4d82a6c127279a682861e2306f6e9ca68b7e3df77f37a50c0655cc1c041b
Self-recurrence/fanout patch SHA-256:
  1b887fb218818979535088a4c2c57c109ec1fb0514f456ac22b5f1b647a1ecd8
Memory-import identity patch SHA-256:
  69d6eb2f58e509852d19018b946955805053ed81883360c81a4c24b0e8c8276e
Docker image:
  quotientflow-morpher-v5b-native-fixed17@
  sha256:4581c2c9a868ffd129609c5b962417e1d8ff3e84cbe2890a6d1364101e901957
Exact mapper binary SHA-256:
  2b49863713eb22523c2331058e761e00b6240365d749a6c01f86fa789436ec6f
Remote immutable prefix:
  /home/Rishabh@MLL-5090/remote-work/HPCA/.cluster_toolchains/
  morpher-fixed17-gcc7-2b49863713eb
```

The earlier fixed14 prefix and every result produced with it remain immutable;
they are not silently reinterpreted as fixed17 results.

The source archive and binary were copied to content-addressed, append-only
paths under `.cluster_inputs`. Installation was executed through
`tools/cluster_queue.sh`; it refuses to overwrite an existing prefix, verifies
all checksums, runs the native `--help` smoke, writes a manifest, and atomically
renames the completed prefix.

## Exact commands

The source archive was made from the fixed14 tree, excluding only `.git` and
build outputs:

```bash
tar -C results/revision_v5b/morpher_patch \
  --exclude='Morpher_CGRA_Mapper/.git' \
  --exclude='Morpher_CGRA_Mapper/build' \
  --sort=name --mtime='UTC 2026-07-29' \
  --owner=0 --group=0 --numeric-owner \
  -czf /tmp/qf_morpher_fixed14_888f3213.tar.gz \
  Morpher_CGRA_Mapper
```

The exact executable is extracted from the selected validated image:

```bash
cid=$(docker create quotientflow-morpher-v5b-native-fixed17)
docker cp \
  "$cid:/home/user/morpher/Morpher_CGRA_Mapper/build/src/cgra_xml_mapper" \
  /tmp/cgra_xml_mapper_fixed17_gcc7
docker rm "$cid"
```

The current remote installation job was:

```bash
RESOURCE_POOL=cpu bash tools/cluster_queue.sh submit \
  morpher_fixed17_install_gcc7 -- \
  env MORPHER_VARIANT=fixed17 \
  BRIDGE_PATCH_SHA256=69d6eb2f58e509852d19018b946955805053ed81883360c81a4c24b0e8c8276e \
  SOURCE_DOCKER_IMAGE=quotientflow-morpher-v5b-native-fixed17@sha256:4581c2c9a868ffd129609c5b962417e1d8ff3e84cbe2890a6d1364101e901957 \
  bash tools/install_cluster_morpher_binary_v5b.sh \
  /home/Rishabh@MLL-5090/remote-work/HPCA/.cluster_inputs/qf_morpher_fixed17_28a220b0.tar.gz \
  28a220b094046a7083cc29dfb546171b5094f6bcb9d07db388b4f39b68b8355d \
  28a220b094046a7083cc29dfb546171b5094f6bcb9d07db388b4f39b68b8355d \
  /home/Rishabh@MLL-5090/remote-work/HPCA/.cluster_inputs/cgra_xml_mapper_fixed17_2b498637 \
  2b49863713eb22523c2331058e761e00b6240365d749a6c01f86fa789436ec6f
```

Queue run: `20260729T083454Z_morpher_fixed17_install_gcc7`.

## Native dump smoke

Fixed17 completed the native array-add PathFinder mapping and emitted the
three bridge contracts. All 64 FU records contained complete latency maps,
and each latency-map opcode set exactly matched the FU supported-opcode set:

```text
II: 4
operations: 20
routes: 23
wall time: 7.947608 s
dfg.json SHA-256:
  498762341449ad1507b14dcb19954cb7dbf10ddb73f1df208c5ce43180c76bd8
mrrg.json SHA-256:
  cacd7a0ec93a43b05db4d4ed2f2d31040d99ae9b5061464026228760ac8b3f7e
mapping.json SHA-256:
  45db4456b5cdafb8772554f39b891f397edec73ebed3489353134f0bd9a70ffc
```

Queue run: `20260729T083534Z_morpher_fixed17_gcc7_latency_smoke`.

The original fixed14 smoke remains at
`results/revision_v5b/cluster/smoke_array_add_fixed14_gcc7/`.

Thread limits applied by the queue:

```text
OMP_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1
```

## Cluster execution contract

- `tools/cluster_queue.sh` performs additive source sync and append-only job
  creation.
- CPU jobs use deterministic per-slot `flock`s; no nested worker pool is
  created.
- `tools/run_cluster_morpher_export_v5b.sh` creates one isolated output tree,
  updates a 15-second heartbeat, writes native stdout/stderr separately,
  validates all three JSON products, and atomically renames only a successful
  result.
- Source, binary, DFG, architecture, and output hashes are recorded per run.
- Existing run or output directories are never overwritten or deleted.

This runtime is suitable for parallel independent native-mapper work items.
Native Morpher search itself is CPU-bound; GPU0 is reserved for FlowAdvantage
inference/model workloads.

## Representative throughput benchmark

The checked-in `fix_fft_INNERMOST_LN121_DFG.xml` on A0 HyCUBE completed on the
cluster in 315.555 seconds (99% CPU, 59,784 KiB peak RSS), producing II 9 with
47 placements and 65 ordered routes. The matching local fixed14 run required
approximately 507 seconds, so this representative single native search was
1.61x faster on the cluster. Independent queue slots allow additional native
searches to make progress concurrently.

Queue run:
`20260729T081005Z_morpher_fixed14_fixfft_a0_benchmark_correct`.

Pulled output:
`results/revision_v5b/cluster/fix_fft_A0_cluster_benchmark/`.

## First parallel A2 memory result

The A2 GEMM reference was generated from Morpher's checked-in two-bank GEMM
layout, using the deterministic flat-address derivation and preserving its
complete provenance. It completed in 120.214 seconds:

```text
kernel: gemm_nt
DFG SHA-256:
  1828440d7ea21f8b0171570e6eeab0fdbde79709e5107241cf770f0043b63a4b
generated architecture SHA-256:
  3442028454236f4e0527e7b11d7e305ea6c439dcc2439cce28c8b91705b414cc
II: 8
operations: 52
DFG dependencies: 73
physical routes: 72
pseudo PS dependencies: 1
independent FlowAdvantage legality: PASS, zero violations
```

Queue run: `20260729T081916Z_morpher_fixed14_gemm_a2_reference_v2`.

Pulled output:
`results/revision_v5b/cluster/gemm_nt_A2_cluster_reference/`.

The corresponding FFT A2 attempts are retained as input-compatibility
diagnostics rather than mapping failures. The legacy PartPred FFT DFG aborts
inside native `assignPath`; the otherwise mapped sample FFT DFG contains the
official memory variables but also at least one native memory operation with
no `BasePointerName`, causing `UpdateVariableBaseAddr` to reject it before
search. No missing pointer identity was inferred or fabricated.

Fixed16 subsequently corrected the native self-recurrence/fanout sentinel
collision, allowing the actual PartPred FFT DFG to map with its official
two-bank layout in 60.239 seconds at II 5 (58 operations, 90 dependencies,
77 physical routes and 13 pseudo dependencies). Independent FlowAdvantage
legality passed with zero violations.

Fixed17 then corrected memory-import identity ordering and exported FU latency
tables. Native fixed17 reimport produced exact semantic round trips for both
A2 memory references:

| Kernel | II | Placements | Routes | Placement match | Route match | Independent legality |
|---|---:|---:|---:|---|---|---|
| `gemm_nt` | 8 | 52 | 72 | exact | exact | PASS |
| `fix_fft` | 5 | 58 | 77 | exact | exact | PASS |

The fixed17 reimport jobs took 29.586 and 23.401 seconds respectively.
All 128 GEMM FU records and all 80 FFT FU records carried latency maps whose
opcode sets exactly matched native operation support.

Queue runs:

```text
20260729T083646Z_morpher_fixed17_gemm_a2_native_reimport
20260729T083803Z_morpher_fixed17_fixfft_a2_native_reimport
```

The exact reimport interface is implemented by
`tools/run_cluster_morpher_reimport_v5b.sh`; it requires native import PASS,
then compares II, every operation placement/time assignment, and every ordered
route resource/link/latency sequence before atomically exposing a result.
