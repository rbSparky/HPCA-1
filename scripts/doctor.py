#!/usr/bin/env python
"""Capture the exact host and active-environment records."""

from __future__ import annotations

import json
import os
import platform
import subprocess
from pathlib import Path

import psutil
import torch


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "system"


def command(args):
    try:
        return subprocess.run(args, text=True, capture_output=True, check=False).stdout
    except FileNotFoundError as exc:
        return f"unavailable: {exc}\n"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    smi = command(["nvidia-smi"])
    (OUT / "nvidia_smi.txt").write_text(smi)
    env_name = os.environ.get("CONDA_DEFAULT_ENV", "")
    if env_name:
        exported = command(["conda", "env", "export", "-n", env_name])
    else:
        exported = command(["conda", "env", "export", "-p", os.path.dirname(os.path.dirname(os.sys.executable))])
    (OUT / "conda_env.yml").write_text(exported)
    (OUT / "pip_freeze.txt").write_text(command([os.sys.executable, "-m", "pip", "freeze"]))
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
    vram = (
        torch.cuda.get_device_properties(0).total_memory / 1024**3
        if torch.cuda.is_available()
        else 0.0
    )
    query = command(
        [
            "nvidia-smi",
            "--query-gpu=driver_version",
            "--format=csv,noheader",
        ]
    ).strip()
    record = {
        "os": f"{platform.platform()} ({platform.version()})",
        "cpu": platform.processor() or command(["lscpu"]).split("Model name:")[-1].splitlines()[0].strip(),
        "ram_gb": round(psutil.virtual_memory().total / 1024**3, 3),
        "gpu": gpu,
        "vram_gb": round(vram, 3),
        "nvidia_driver": query,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda or "",
        "cuda_available": bool(torch.cuda.is_available()),
        "python": platform.python_version(),
        "executable": os.sys.executable,
        "conda_environment": env_name or "prefix:" + os.path.dirname(os.path.dirname(os.sys.executable)),
    }
    if torch.cuda.is_available():
        x = torch.randn(2048, 2048, device="cuda")
        y = x @ x
        torch.cuda.synchronize()
        record["gpu_smoke"] = float(y[0, 0])
    (OUT / "system.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
