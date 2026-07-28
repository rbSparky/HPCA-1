#!/usr/bin/env python3
"""Run preflight checks required before launching paper-suite workers."""
from __future__ import annotations

import csv
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "paper_suite_v1"


def check(name: str, command: list[str] | None = None, paths: list[Path] | None = None) -> dict:
    started = time.time()
    if command is not None:
        proc = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
        ok = proc.returncode == 0
        detail = (proc.stdout + proc.stderr)[-4000:]
        return {"check": name, "status": "PASS" if ok else "FAIL", "seconds": round(time.time() - started, 3), "detail": detail, "returncode": proc.returncode}
    missing = [str(p) for p in (paths or []) if not p.exists()]
    return {"check": name, "status": "PASS" if not missing else "FAIL", "seconds": round(time.time() - started, 3), "detail": "all paths present" if not missing else "missing: " + "; ".join(missing), "returncode": 0 if not missing else 1}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [
        check("python_adapter_tests", ["python3", "-m", "pytest", "-q", "tests/test_morpher_adapter.py"]),
        check("queue_shell_syntax", ["bash", "-n", "tools/cluster_queue.sh"]),
        check("manifest_integrity", ["python3", "-c", "import csv; p='results/paper_suite_v1/work_manifest.csv'; rows=list(csv.DictReader(open(p))); assert len(rows)==1164; assert len({r['work_id'] for r in rows})==1164"]),
        check("native_docker_image", ["docker", "image", "inspect", "quotientflow-morpher-v5b-native"]),
        check("frozen_checkpoint", paths=[ROOT / "results/revision_v3/checkpoints/residual_gnn_seed_23.pt", ROOT / "results/revision_v3/selected_flow_model.json"]),
        check("array_add_native_evidence", paths=[ROOT / "results/revision_v5b/reference_mappings/array_add/A0_full_pipeline/native_dump/dfg.json", ROOT / "results/revision_v5b/reference_mappings/array_add/A0_full_pipeline/native_reimport/mapping.json", ROOT / "results/revision_v5b/reference_mappings/array_add/A0_full_pipeline/sim_result.txt"]),
        check("remote_queue_health", ["bash", "tools/cluster_queue.sh", "health"]),
        check("remote_gpu_smoke_artifact", paths=[ROOT / "results/remote_runs/20260728T181724Z_gpu0_doctor_fixed/stdout.log"]),
    ]
    path = OUT / "raw" / "smoke_checks.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["check", "status", "seconds", "detail", "returncode"])
        writer.writeheader(); writer.writerows(rows)
    summary = {"schema": "flowadvantage_paper_smoke_v1", "generated_at": time.time(), "passed": sum(r["status"] == "PASS" for r in rows), "total": len(rows), "checks": rows}
    (OUT / "raw" / "smoke_checks.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": summary["passed"], "total": summary["total"], "csv": str(path)}, indent=2))
    return 0 if summary["passed"] == summary["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
