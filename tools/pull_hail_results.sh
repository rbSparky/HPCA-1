#!/usr/bin/env bash
set -euo pipefail

# Append-only transfer for the hail-mary atomic queues.  Do not use the
# repository-wide ``remote.sh down`` here: native MRRG exports are large and
# several pre-existing directories are root-owned in the working tree.  Queue
# work-items, manifests, heartbeats and logs are the auditable evidence.
HOST="${HOST:-mll5090}"
REMOTE="${REMOTE:-/home/Rishabh@MLL-5090/remote-work/HPCA/results/hail_mary_hpca1/}"
LOCAL="${LOCAL:-results/hail_mary_hpca1/}"
mkdir -p "${LOCAL}"
rsync -az --no-times --omit-dir-times \
  --exclude 'cache/***' \
  --exclude 'reference_mappings/***' \
  --exclude 'problem_exports/***' \
  --exclude 'morpher_patch/***' \
  --exclude 'native_inputs/***' \
  -e ssh "${HOST}:${REMOTE}" "${LOCAL}"
echo "pulled atomic hail results into ${LOCAL}"
