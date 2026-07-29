# Non-destructive cluster queue

`tools/cluster_queue.sh` is the paper-experiment interface for the `mll5090`
host. It complements the legacy `remote.sh`; it does not modify it.

The helper uses the CUDA-verified remote environment
`/home/Rishabh@MLL-5090/envs/gpu-test`, binds submitted work to GPU0, limits
CPU thread pools to one per job, and serializes GPU0 jobs with a remote
`flock`. This prevents overlapping experiments from fighting over GPU memory
or BLAS threads.

Every submitted run is append-only under the remote project's
`.cluster_runs/<UTC-run-id>/`. It contains a JSON argv record, executable
reconstructed command, start/end status, environment manifest, and separate
stdout/stderr logs. The command source is synchronized additively; no
`--delete` synchronization is used. Generated results are excluded from source
sync and can only be copied back through a new local
`results/remote_runs/<run-id>/` directory. A pull refuses to overwrite one.

```bash
# Read-only capacity/status inspection.
bash tools/cluster_queue.sh health
bash tools/cluster_queue.sh status

# Additive source sync (no deletion).
bash tools/cluster_queue.sh sync

# Submit one GPU0-serialized, reproducible run.
bash tools/cluster_queue.sh submit flow_eval_seed11 -- \
  python scripts/evaluate_v3_topk.py --seed 11

# Inspect and retrieve a completed run.
bash tools/cluster_queue.sh logs <UTC-run-id>
bash tools/cluster_queue.sh pull <UTC-run-id>
```

The queue intentionally has no destructive cleanup or cancellation operation.
If an experiment must be stopped, preserve its logs and record the reason in
the corresponding revision's raw execution manifest before manually stopping
the remote process.

The guide's original `gpu-cu128` environment currently has a CUDA 13 wheel
that the installed 575.51.03 driver cannot initialize. `gpu-test` instead
provides PyTorch 2.11.0+cu128 and has passed `scripts/doctor.py` on GPU0.

## Pinned Morpher v5b runtime

The cluster does not provide Docker or root access. The validated fixed14
native mapper is installed as an immutable, content-addressed prefix:

```text
/home/Rishabh@MLL-5090/remote-work/HPCA/.cluster_toolchains/
morpher-fixed14-gcc7-af402f1b7f20
```

Use `RESOURCE_POOL=cpu` because native Morpher search is CPU-bound. For
example, one atomic native dump is submitted with:

```bash
RESOURCE_POOL=cpu bash tools/cluster_queue.sh submit NAME -- \
  bash tools/run_cluster_morpher_export_v5b.sh \
  /home/Rishabh@MLL-5090/remote-work/HPCA/.cluster_toolchains/morpher-fixed14-gcc7-af402f1b7f20 \
  applications/sample_xmls/fix_fft_INNERMOST_LN121_DFG.xml \
  json_arch/hycube_original.json \
  4 4 0 HyCUBE_4REG 0 \
  /home/Rishabh@MLL-5090/remote-work/HPCA/.cluster_outputs/UNIQUE_OUTPUT_NAME
```

The runner refuses to overwrite output, preserves native stdout/stderr,
updates a 15-second heartbeat, validates the three JSON contract files, and
atomically exposes a successful output. Exact setup provenance and smoke
evidence are recorded in
`results/revision_v5b/cluster/CLUSTER_MORPHER_SETUP.md`.
