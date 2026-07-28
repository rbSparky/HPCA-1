# AGENTS.md 

## Goal
You (the agent) will edit files locally in this repository, but **run all compute/tests on the remote 5090 server** via SSH, using the helper script `tools/remote.sh`.

This repo is developed from WSL2. The remote host is reachable only on the college Wi-Fi.

## Remote target
- SSH host alias: `mll5090`
- Underlying login:
  - Username: `Rishabh@MLL-5090`
  - Server IP: `10.10.16.51`
- Remote workspace root (default): `~/remote-work/<project_name>`

Assume SSH keys are already configured for passwordless login (`ssh mll5090` works).

## Non-negotiable workflow rule
**Never run training, GPU code, or heavy tests locally.**
Always run via:
- `tools/remote.sh run <command...>`

This command automatically:
1) rsyncs the current repo to the remote directory
2) runs the command on the server in that remote directory

If you need outputs/logs, you may pull them via:
- `tools/remote.sh down`

## How to run commands
Use these patterns:

### Quick sanity checks
the cluster just has micromamba and not sudo access
usually to use the cluster for my tasks i do 
eval "$(micromamba shell hook --shell bash)"
micromamba activate /home/Rishabh@MLL-5090/envs/gpu-cu128 (this is the main env i use it has torch setup and gpu support
to verify: python ~/5090_smoke_test.py
this is the sanity test please verify output

### Tests
- `tools/remote.sh run python -m pytest -q`
- `tools/remote.sh run python -m unittest -q`

### Training / scripts
- `tools/remote.sh run python train.py --config configs/foo.yaml`

## Micromamba / environment
Remote machine has micromamba and python, but no sudo.
The helper script attempts:
- `eval "$(micromamba shell hook --shell bash)"`
- `micromamba activate $MAMBA_ENV`

Default is `MAMBA_ENV=base`. If imports fail, discover the correct env by running:
- `tools/remote.sh run micromamba env list`
Then set env for future commands by prefixing:
- `MAMBA_ENV=<envname> tools/remote.sh run <command...>`

If you need to install packages in remote, do it with micromamba in the env that i told you in. 

## Sync policy
- Source of truth is **local repo**.
- Remote directory is a mirror created by rsync.
- Avoid syncing huge data artifacts unless required; prefer to write outputs under `outputs/` or `logs/`.

If you must exclude additional paths, update `tools/remote.sh` rsync excludes.

## Safety / network approvals (Codex)
SSH/rsync require network access. If your environment blocks network in the sandbox, request approval or instruct the user to enable network for the workspace sandbox.

## What to do at the start of any task
1) Read this file.
2) Confirm `tools/remote.sh` exists; if not, create it (ask user before creating scripts if operating in a restrictive approval mode).
3) Run `tools/remote.sh run nvidia-smi` to verify remote execution works.
4) Proceed with code changes; after each change, run the relevant remote tests/commands.


## your task
make new python file to download mnist and to train a small model on it
run training in server and generate training accuracy plot
then get the result plot in my local folder so i can check 
