# Pinned Morpher fixed14 cluster toolchain

Date: 2026-07-29  
Host alias: `mll5090`  
Host: Ubuntu 24.04.2, Linux 6.8.0-84, AMD Threadripper PRO 7965WX
(24 physical cores / 48 threads), 125 GiB RAM.

## Authoritative runtime

The cluster has no Docker, Podman, Apptainer, Singularity, or root access.
Recompiling the legacy mapper with host GCC 13 is not an equivalent runtime:
although it builds, the array-add native smoke crashes in legacy undefined
behavior before parsing completes. That build remains recorded only as a
portability diagnostic.

The authoritative runtime therefore uses the exact GCC-7 executable extracted
from the locally validated Docker image, on the cluster host's backward-
compatible glibc/libstdc++ runtime:

```text
Morpher base commit:
  9a9dce7aea521f1d5ef33686f57ca84864edb3c9
Bridge patch SHA-256:
  4894054e67eb43f16ad0a40bcbf9783305b67612925a3ea5b0d7acdfd11fad54
Source-tree SHA-256:
  888f3213f34ef66c6ff9e96b66515d0d3bbc6b11020c325629dbe38a1c9a494e
Source archive SHA-256:
  c5089d5dde28920b66d3e089fd2eae7e2686fc4dcd90e02cc9ae43b666f67a5b
Docker image:
  quotientflow-morpher-v5b-native-fixed14@
  sha256:f1dee9de88d0935ccb1825200040e090abf78ff76140bb0e8d494c00862c9088
Exact mapper binary SHA-256:
  af402f1b7f2077f074ec983fb1965f1cfb530465303bd891bff3a495da8c5c87
Remote immutable prefix:
  /home/Rishabh@MLL-5090/remote-work/HPCA/.cluster_toolchains/
  morpher-fixed14-gcc7-af402f1b7f20
```

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

The exact executable was extracted from the validated image:

```bash
cid=$(docker create quotientflow-morpher-v5b-native-fixed14)
docker cp \
  "$cid:/home/user/morpher/Morpher_CGRA_Mapper/build/src/cgra_xml_mapper" \
  /tmp/cgra_xml_mapper_fixed14_gcc7
docker rm "$cid"
```

The remote installation job was:

```bash
RESOURCE_POOL=cpu bash tools/cluster_queue.sh submit \
  morpher_fixed14_install_gcc7 -- \
  bash tools/install_cluster_morpher_binary_v5b.sh \
  /home/Rishabh@MLL-5090/remote-work/HPCA/.cluster_inputs/qf_morpher_fixed14_888f3213.tar.gz \
  c5089d5dde28920b66d3e089fd2eae7e2686fc4dcd90e02cc9ae43b666f67a5b \
  888f3213f34ef66c6ff9e96b66515d0d3bbc6b11020c325629dbe38a1c9a494e \
  /home/Rishabh@MLL-5090/remote-work/HPCA/.cluster_inputs/cgra_xml_mapper_fixed14_gcc7_8d1de1c0 \
  af402f1b7f2077f074ec983fb1965f1cfb530465303bd891bff3a495da8c5c87
```

Queue run: `20260729T080652Z_morpher_fixed14_install_gcc7`.

## Native dump smoke

The authoritative runtime completed the native array-add PathFinder mapping and
emitted the three bridge contracts:

```text
II: 4
operations: 20
routes: 23
wall time: 7.897727 s
dfg.json SHA-256:
  498762341449ad1507b14dcb19954cb7dbf10ddb73f1df208c5ce43180c76bd8
mrrg.json SHA-256:
  cacd7a0ec93a43b05db4d4ed2f2d31040d99ae9b5061464026228760ac8b3f7e
mapping.json SHA-256:
  2a9293ece5da59f9a08c95d1accb1442d62c9c9c2bbd1a8d07cf40620c4b2a56
```

Queue run: `20260729T080713Z_morpher_fixed14_gcc7_array_smoke`.

Pulled immutable output:
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
