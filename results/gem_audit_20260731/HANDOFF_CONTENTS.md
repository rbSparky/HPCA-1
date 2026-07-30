# Handoff contents

The handoff contains the tracked repository at the recorded gem commit plus the
latest audit, terminal remote queue snapshot, selected frozen model/checkpoints,
synthetic v4b evidence, adapter-v5b reports/raw tables, and the local length
baseline/retry manifests needed to reproduce paired comparisons.

Large reproducible or environment-specific objects are deliberately excluded:

- `results/revision_v5b/reference_mappings/` (hundreds of MB; native exports are
  represented by registries, hashes, and selected audit records);
- `results/revision_v5b/morpher_patch/` build/check-out payloads (tracked source
  patch files remain included where Git tracks them);
- bulk relaxation caches (their version/hash metadata and selected model weights
  are included);
- previous handoff archives, `__pycache__`, and `.pytest_cache`.

The archive checksum and a successful `unzip -t` verification are recorded next
to the archive in the results directory.
