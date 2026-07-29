#!/usr/bin/env bash
# Append-only remote execution helper for reproducible paper experiments.
# It intentionally never deletes a remote file and never writes into local
# results directories. Jobs use a GPU-specific remote flock to serialize GPU0.
set -euo pipefail

HOST="${HOST:-mll5090}"
REMOTE_HOME="${REMOTE_HOME:-/home/Rishabh@MLL-5090}"
REMOTE_BASE="${REMOTE_BASE:-${REMOTE_HOME}/remote-work}"
PROJECT_NAME="${PROJECT_NAME:-HPCA}"
REMOTE_DIR="${REMOTE_DIR:-${REMOTE_BASE}/${PROJECT_NAME}}"
RUN_ROOT="${RUN_ROOT:-${REMOTE_DIR}/.cluster_runs}"
MAMBA_ENV="${MAMBA_ENV:-/home/Rishabh@MLL-5090/envs/gpu-test}"
GPU="${GPU:-0}"
RESOURCE_POOL="${RESOURCE_POOL:-gpu0}"

die() { echo "error: $*" >&2; exit 2; }

require_name() {
  [[ "${1:-}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || die "run name must be alphanumeric plus . _ -"
}

sync_source() {
  # Source-only, additive synchronization. Generated artifacts stay remote.
  rsync -az \
    --exclude '.git/' --exclude '__pycache__/' --exclude '.venv/' --exclude '.mamba/' \
    --exclude '/results/' --exclude '/outputs/' --exclude '/logs/' --exclude '/data/' \
    --exclude '/.cluster_runs/' --exclude '/.cluster_outputs/' \
    --exclude '/.cluster_toolchains/' --exclude '/.cluster_inputs/' \
    --exclude '/.cluster_queue/' --exclude '/.remote_jobs/' \
    -e ssh ./ "${HOST}:${REMOTE_DIR}/"
}

remote_bash() {
  local forwarded=""
  local name quoted
  for name in RUN_ID RUN_NAME ENCODED_CMD REMOTE_DIR RUN_ROOT MAMBA_ENV GPU RESOURCE_POOL CPU_SLOT; do
    if [[ -v "$name" ]]; then
      printf -v quoted '%q' "${!name}"
      forwarded+="${name}=${quoted} "
    fi
  done
  ssh "${HOST}" "export PATH=\"\$HOME/bin:\$PATH\"; ${forwarded}bash -s"
}

health() {
  remote_bash <<'REMOTE'
set -euo pipefail
echo '== host =='
hostname; date -Iseconds; uptime
echo '== gpu =='
nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.used,utilization.gpu --format=csv,noheader
echo '== gpu processes =='
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader || true
echo '== environments =='
micromamba env list || true
REMOTE
}

submit() {
  local name="${1:-}"; shift || true
  require_name "$name"
  [[ "${1:-}" == "--" ]] || die "usage: submit NAME -- command [args...]"
  shift
  (( $# > 0 )) || die "missing command"
  local run_id="$(date -u +%Y%m%dT%H%M%SZ)_${name}"
  local encoded
  encoded="$(printf '%s\0' "$@" | base64 -w0)"
  sync_source
  RUN_ID="$run_id" RUN_NAME="$name" ENCODED_CMD="$encoded" REMOTE_DIR="$REMOTE_DIR" RUN_ROOT="$RUN_ROOT" MAMBA_ENV="$MAMBA_ENV" GPU="$GPU" RESOURCE_POOL="$RESOURCE_POOL" remote_bash <<'REMOTE'
set -euo pipefail
run_dir="${RUN_ROOT}/${RUN_ID}"
mkdir -p "$run_dir"
[[ ! -e "$run_dir/manifest.env" ]] || { echo "immutable run already exists: $RUN_ID" >&2; exit 3; }
printf 'RUN_ID=%q\nNAME=%q\nSUBMITTED_AT=%q\nGPU=%q\nENV=%q\nREMOTE_DIR=%q\n' \
  "$RUN_ID" "$RUN_NAME" "$(date -Iseconds)" "$GPU" "$MAMBA_ENV" "$REMOTE_DIR" > "$run_dir/manifest.env"
printf 'RESOURCE_POOL=%q\n' "$RESOURCE_POOL" >> "$run_dir/manifest.env"
printf 'CPU_SLOT=%q\n' "${CPU_SLOT:-auto}" >> "$run_dir/manifest.env"
python3 - "$ENCODED_CMD" "$run_dir/argv.json" <<'PY'
import base64, json, sys
parts = base64.b64decode(sys.argv[1]).split(b'\0')[:-1]
with open(sys.argv[2], 'w', encoding='utf-8') as f:
    json.dump([p.decode('utf-8') for p in parts], f, indent=2)
PY
printf 'QUEUED %s\n' "$(date -Iseconds)" > "$run_dir/status"
nohup bash -s -- "$run_dir" "$REMOTE_DIR" "$MAMBA_ENV" "$GPU" > "$run_dir/launcher.log" 2>&1 <<'WORKER' &
set -euo pipefail
run_dir="$1"; project="$2"; env_path="$3"; gpu="$4"; pool="${RESOURCE_POOL:-gpu0}"
if [[ "$pool" == cpu ]]; then
  # Deterministic eight-slot CPU pool. Jobs sharing a slot serialize; the
  # eight slot locks permit independent workers without a nested process pool.
  if [[ -n "${CPU_SLOT:-}" ]]; then
    [[ "$CPU_SLOT" =~ ^[0-7]$ ]] || {
      echo "CPU_SLOT must be an integer from 0 through 7" >&2
      exit 2
    }
    slot="$CPU_SLOT"
  else
    slot=$(cksum <<< "$(basename "$run_dir")" | awk '{print $1 % 8}')
  fi
  lock="$HOME/.quotientflow_cpu${slot}.lock"
  gpu_env=""
else
  lock="$HOME/.quotientflow_gpu${gpu}.lock"
  gpu_env="CUDA_VISIBLE_DEVICES=$gpu"
fi
exec 9>"$lock"
flock 9
printf 'RUNNING %s\n' "$(date -Iseconds)" > "$run_dir/status"
python3 - "$run_dir/argv.json" "$run_dir/command.sh" <<'PY'
import json, shlex, sys
args=json.load(open(sys.argv[1], encoding='utf-8'))
open(sys.argv[2], 'w', encoding='utf-8').write('#!/usr/bin/env bash\nexec ' + ' '.join(map(shlex.quote,args))+'\n')
PY
chmod 700 "$run_dir/command.sh"
set +e
(
  cd "$project"
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
  if [[ -n "$gpu_env" ]]; then export "$gpu_env"; fi
  micromamba run -p "$env_path" "$run_dir/command.sh"
) > "$run_dir/stdout.log" 2> "$run_dir/stderr.log"
rc=$?
set -e
printf 'EXIT_CODE=%s\nENDED_AT=%s\n' "$rc" "$(date -Iseconds)" >> "$run_dir/manifest.env"
if (( rc == 0 )); then printf 'DONE %s\n' "$(date -Iseconds)" > "$run_dir/status"; else printf 'ERROR %s\n' "$(date -Iseconds)" > "$run_dir/status"; fi
WORKER
echo "$RUN_ID"
REMOTE
}

status() {
  remote_bash <<REMOTE
set -euo pipefail
root='${RUN_ROOT}'
[[ -d "\$root" ]] || { echo 'no cluster runs'; exit 0; }
for d in "\$root"/*; do [[ -d "\$d" ]] || continue; printf '%s ' "\$(basename "\$d")"; cat "\$d/status" 2>/dev/null || echo UNKNOWN; done
REMOTE
}

logs() {
  local id="${1:-}"; require_name "$id"
  ssh "$HOST" "tail -n 120 '${RUN_ROOT}/${id}/stdout.log' 2>/dev/null; tail -n 120 '${RUN_ROOT}/${id}/stderr.log' 2>/dev/null"
}

pull() {
  local id="${1:-}"; require_name "$id"
  local dest="results/remote_runs/${id}"
  [[ ! -e "$dest" ]] || die "refusing to overwrite $dest"
  mkdir -p "$(dirname "$dest")"
  rsync -az -e ssh "${HOST}:${RUN_ROOT}/${id}/" "$dest/"
  echo "$dest"
}

usage() {
  cat <<'USAGE'
Usage:
  tools/cluster_queue.sh health
  tools/cluster_queue.sh sync
  RESOURCE_POOL=cpu tools/cluster_queue.sh submit NAME -- COMMAND [ARGS...]
  RESOURCE_POOL=gpu0 tools/cluster_queue.sh submit NAME -- COMMAND [ARGS...]
  tools/cluster_queue.sh status
  tools/cluster_queue.sh logs RUN_ID
  tools/cluster_queue.sh pull RUN_ID

`submit` is append-only and serializes GPU0 jobs through a remote flock.
It never deletes remote files; `pull` refuses to overwrite local artifacts.
CPU jobs may set `CPU_SLOT=0` through `CPU_SLOT=7` to select an explicit
independent lock; otherwise a deterministic run-ID hash selects the slot.
USAGE
}

case "${1:-}" in
  health) health ;;
  sync) sync_source ;;
  submit) shift; submit "$@" ;;
  status) status ;;
  logs) shift; logs "$@" ;;
  pull) shift; pull "$@" ;;
  *) usage; exit 2 ;;
esac
