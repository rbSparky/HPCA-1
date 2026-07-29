# Cluster provenance incident — 2026-07-29

## Summary

At approximately 17:03 IST, a profiling task invoked the legacy
`tools/remote.sh run` helper. Its source synchronization used
`rsync --delete` against the shared remote project root. The local checkout
does not contain the cluster's append-only hidden directories, so the mirror
operation removed:

- `.cluster_runs`;
- `.cluster_outputs`;
- `.cluster_inputs`;
- `.cluster_queue`;
- most files below `.cluster_toolchains`.

This is an execution-infrastructure failure. It is not a mapping failure and
none of the interrupted work is counted as scientific evidence.

## Preserved evidence

Outputs pulled to the local repository before the incident remain intact,
including the fixed17 latency smoke, native proposal GPU smoke, array-cond A0
reference, and trmm A0 reference. Both new reference mappings pass the
independent FlowAdvantage legality checker.

The remote native processes whose files had already been opened remained
alive temporarily, but their parent output directories had been unlinked.
They could not publish atomic outputs and therefore cannot yield valid result
rows.

## Corrective action

`tools/remote.sh` source synchronization is now additive: `--delete` was
removed, and every `.cluster_*` directory is explicitly excluded. The
paper-suite path must use `tools/cluster_queue.sh`; the legacy helper is not
an experiment scheduler.

The cluster environment and GPU remained healthy. Recovery requires
reinstalling the fixed toolchain from the locally preserved source/image,
recreating append-only queue directories, re-uploading hash-verified inputs,
and resubmitting each missing atomic work item under a new run ID.

## Scientific handling

- No missing or interrupted row is classified as a valid mapping failure.
- Previously pulled atomic outputs are retained with their original hashes.
- Recovered jobs receive new run IDs; deleted queue metadata is not
  reconstructed or fabricated.
- This incident and the recovery wall time remain part of the artifact log.
