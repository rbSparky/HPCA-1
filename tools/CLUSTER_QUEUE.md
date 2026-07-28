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
