#!/usr/bin/env bash
set -euo pipefail

HOST="${HOST:-mll5090}"
REMOTE_BASE="${REMOTE_BASE:-~/remote-work}"
PROJECT_NAME="${PROJECT_NAME:-$(basename "$(pwd)")}"
REMOTE_DIR="${REMOTE_DIR:-${REMOTE_BASE}/${PROJECT_NAME}}"
JOB_DIR="${JOB_DIR:-.remote_jobs}"

# Optional: set your micromamba env name (remote)
MAMBA_ENV="${MAMBA_ENV:-base}"
# Local home used for path rewrite (handles unquoted ~ expansions)
LOCAL_HOME="${LOCAL_HOME:-$HOME}"

sync_up() {
  rsync -az --delete \
    --exclude '.git/' --exclude '__pycache__/' --exclude '.venv/' --exclude '.mamba/' \
    --exclude '/outputs/' --exclude '/logs/' --exclude '/data/' --exclude '/results/' --exclude '/results_repro/' --exclude '/results_verified/' --exclude '/ogb_data/' --exclude "${JOB_DIR}/" \
    --exclude 'celebA dataset.zip' --exclude 'celebA dataset.zip:Zone.Identifier' \
    -e ssh \
    ./ "${HOST}:${REMOTE_DIR}/"
}

sync_down() {
  # Pull back common outputs (customize as needed)
  rsync -az -e ssh \
    "${HOST}:${REMOTE_DIR}/" ./ \
    --include 'logs/***' --include 'outputs/***' --include 'results/***' --include 'results_verified/***' --include 'checkpoints/***' \
    --include 'figures/' --include 'figures/***' \
    --include 'paper/' --include 'paper/cvpr/' --include 'paper/cvpr/tables/' --include 'paper/cvpr/tables/***' \
    --exclude '*'
}

remote_run() {
  local cmd="$*"
  cmd="$(rewrite_home "$cmd")"
  # micromamba activation can vary by setup; prefer `micromamba run` for non-interactive shells.
  ssh "${HOST}" "cd ${REMOTE_DIR} && \
    export PATH=\"\$HOME/bin:\$PATH\" && \
    if command -v micromamba >/dev/null 2>&1; then \
      if [[ \"${MAMBA_ENV}\" == /* || \"${MAMBA_ENV}\" == ~* ]]; then \
        micromamba run -p \"${MAMBA_ENV}\" ${cmd}; \
      else \
        micromamba run -n \"${MAMBA_ENV}\" ${cmd}; \
      fi; \
    else \
      ${cmd}; \
    fi"
}

rewrite_home() {
  local cmd="$*"
  local local_home="${LOCAL_HOME%/}"
  local placeholder='$HOME'
  local placeholder_slash='$HOME/'

  if [[ -n "${local_home}" ]]; then
    cmd="${cmd//${local_home}\//${placeholder_slash}}"
    if [[ "${cmd}" == "${local_home}" ]]; then
      cmd="${placeholder}"
    fi
  fi

  printf '%s' "${cmd}"
}

batch_submit() {
  local cmd="$*"
  cmd="$(rewrite_home "$cmd")"
  if [[ -z "${cmd}" ]]; then
    echo "error: missing command for batch submit" >&2
    exit 2
  fi
  sync_up
  ssh "${HOST}" "cd ${REMOTE_DIR} && \
    export PATH=\"\$HOME/bin:\$PATH\" && \
    mkdir -p ${JOB_DIR} && \
    job_id=\$(date +%Y%m%d_%H%M%S)_\$RANDOM && \
    log=\"${JOB_DIR}/\${job_id}.log\" && \
    pidfile=\"${JOB_DIR}/\${job_id}.pid\" && \
    cmdfile=\"${JOB_DIR}/\${job_id}.cmd\" && \
    startfile=\"${JOB_DIR}/\${job_id}.start\" && \
    echo \"${cmd}\" > \"\${cmdfile}\" && \
    date -Iseconds > \"\${startfile}\" && \
    if command -v micromamba >/dev/null 2>&1; then \
      if [[ \"${MAMBA_ENV}\" == /* || \"${MAMBA_ENV}\" == ~* ]]; then \
        nohup micromamba run -p \"${MAMBA_ENV}\" ${cmd} > \"\${log}\" 2>&1 & \
      else \
        nohup micromamba run -n \"${MAMBA_ENV}\" ${cmd} > \"\${log}\" 2>&1 & \
      fi; \
    else \
      nohup ${cmd} > \"\${log}\" 2>&1 & \
    fi; \
    echo \$! > \"\${pidfile}\" && \
    echo \"\${job_id}\""
}

batch_list() {
  ssh "${HOST}" "cd ${REMOTE_DIR} && \
    if [[ ! -d ${JOB_DIR} ]]; then \
      echo \"No jobs found\"; exit 0; \
    fi; \
    shopt -s nullglob; \
    files=( ${JOB_DIR}/*.pid ); \
    if (( \${#files[@]} == 0 )); then \
      echo \"No jobs found\"; exit 0; \
    fi; \
    printf \"%-20s %-8s %-8s %s\\n\" \"JOB_ID\" \"STATUS\" \"PID\" \"COMMAND\"; \
    for pidfile in \"\${files[@]}\"; do \
      job_id=\$(basename \"\${pidfile}\" .pid); \
      pid=\$(cat \"\${pidfile}\"); \
      cmd=\$(cat \"${JOB_DIR}/\${job_id}.cmd\"); \
      status=\"stopped\"; \
      if ps -p \"\${pid}\" >/dev/null 2>&1; then status=\"running\"; fi; \
      printf \"%-20s %-8s %-8s %s\\n\" \"\${job_id}\" \"\${status}\" \"\${pid}\" \"\${cmd}\"; \
    done"
}

batch_stop() {
  local job_id="${1:-}"
  if [[ -z "${job_id}" ]]; then
    echo "error: missing job id for batch stop" >&2
    exit 2
  fi
  ssh "${HOST}" "cd ${REMOTE_DIR} && \
    pidfile=\"${JOB_DIR}/${job_id}.pid\" && \
    if [[ ! -f \"\${pidfile}\" ]]; then \
      echo \"Job not found: ${job_id}\"; exit 1; \
    fi; \
    pid=\$(cat \"\${pidfile}\"); \
    if ps -p \"\${pid}\" >/dev/null 2>&1; then \
      kill \"\${pid}\"; \
      sleep 1; \
      if ps -p \"\${pid}\" >/dev/null 2>&1; then kill -9 \"\${pid}\"; fi; \
    fi; \
    echo \"Stopped job ${job_id} (pid \${pid})\""
}

usage() {
  cat <<USAGE
Usage:
  tools/remote.sh up
  tools/remote.sh run <command...>
  tools/remote.sh down
  tools/remote.sh batch submit <command...>
  tools/remote.sh batch list
  tools/remote.sh batch stop <job_id>

Env overrides:
  HOST=mll5090
  REMOTE_BASE=~/remote-work
  PROJECT_NAME=<name>
  MAMBA_ENV=base  # name or full path; path uses `micromamba run -p`
  JOB_DIR=.remote_jobs
  LOCAL_HOME=\$HOME  # rewrites local home paths to remote \$HOME
USAGE
}

case "${1:-}" in
  up) shift; sync_up ;;
  run) shift; sync_up; remote_run "$@";;
  down) shift; sync_down ;;
  batch)
    shift
    case "${1:-}" in
      submit) shift; batch_submit "$@";;
      list) shift; batch_list;;
      stop) shift; batch_stop "$@";;
      *) usage; exit 2 ;;
    esac
    ;;
  *) usage; exit 2 ;;
esac
