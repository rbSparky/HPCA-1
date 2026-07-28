#!/usr/bin/env python3
"""Durable, atomic, single-process FlowAdvantage queue.

Unlike v4's ordered ``pool.map`` batch, each result becomes visible immediately.
The queue owns process-tree termination and never aggregates worker returns.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import time
import concurrent.futures
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
              "BLIS_NUM_THREADS", "TORCH_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import pandas as pd
import psutil

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/revision_v4b"
MANIFEST = OUT / "work_manifest.csv"
ITEMS, LOGS = OUT / "work_items", OUT / "logs"
FIELDS = ["work_id", "instance_id", "method", "budget", "attempt", "status",
          "worker_pid", "start_time", "last_heartbeat", "end_time", "wall_seconds",
          "cpu_seconds", "peak_rss_mb", "exit_code", "timeout_seconds", "result_path",
          "stdout_path", "stderr_path", "error_type", "error_message", "instance_path",
          "architecture", "dfg_family", "seed"]


def write_manifest(rows: list[dict]) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    temp = MANIFEST.with_suffix(".tmp")
    with temp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows); handle.flush(); os.fsync(handle.fileno())
    os.replace(temp, MANIFEST)


def read_manifest() -> list[dict]:
    return list(csv.DictReader(MANIFEST.open())) if MANIFEST.exists() else []


def make_manifest(stage: str) -> list[dict]:
    instances = pd.read_csv(ROOT / "results/revision_v4/raw/stress_v4_instances.csv")
    methods = {"length": ["length"], "cheap": ["revision_v1_pred_link", "revision_v2_action_gnn", "dual_linear", "residual_gnn_no_correction"], "top4": ["selected_v3_top4"], "parent": ["hybrid_top4", "noparent_top4"], "full": ["full_relaxed_lookahead"]}[stage]
    rows=[]
    # deterministic round-robin by hashed order, not family-sorted prefix.
    # A deterministic shuffled order prevents the old family-sorted easy prefix.
    instances=instances.sample(frac=1, random_state=24072026).reset_index(drop=True)
    if stage == "full":
        # Fixed balanced K8 subset: one deterministic seed per family ×
        # architecture, selected before any v4b outcomes are inspected.
        instances = (instances.sort_values(["dfg_family", "architecture", "seed"])
                      .groupby(["dfg_family", "architecture"], sort=True,
                               as_index=False).head(1).reset_index(drop=True))
    for m in methods:
        for row in instances.itertuples():
            work_id=f"{row.instance_id}__medium__{m}".replace("/", "_")
            timeout={"length":60,"revision_v1_pred_link":90,"revision_v2_action_gnn":120,
                "dual_linear":180,"residual_gnn_no_correction":900,"selected_v3_top4":3600,
                "hybrid_top4":3600,"noparent_top4":2400,"full_relaxed_lookahead":7200}[m]
            rows.append({"work_id":work_id,"instance_id":row.instance_id,"method":m,"budget":"medium","attempt":0,"status":"PENDING","worker_pid":"","start_time":"","last_heartbeat":"","end_time":"","wall_seconds":"","cpu_seconds":"","peak_rss_mb":"","exit_code":"","timeout_seconds":timeout,"result_path":str((ITEMS/f"{work_id}.json").relative_to(ROOT)),"stdout_path":str((LOGS/f"{work_id}.stdout.log").relative_to(ROOT)),"stderr_path":str((LOGS/f"{work_id}.stderr.log").relative_to(ROOT)),"error_type":"","error_message":"","instance_path":row.instance_path,"architecture":row.architecture,"dfg_family":row.dfg_family,"seed":row.seed})
    return rows


def kill_tree(proc: subprocess.Popen) -> None:
    try:
        root=psutil.Process(proc.pid); descendants=root.children(recursive=True)
        for p in descendants: p.terminate()
        root.terminate()
        _, alive=psutil.wait_procs(descendants+[root], timeout=5)
        for p in alive: p.kill()
    except psutil.NoSuchProcess: pass


def run_item(row: dict) -> dict:
    ITEMS.mkdir(parents=True, exist_ok=True); LOGS.mkdir(parents=True, exist_ok=True)
    row["attempt"] = int(row.get("attempt") or 0) + 1; row["status"]="RUNNING"
    row["start_time"]=row["last_heartbeat"]=str(time.time())
    spec={**row,"heartbeat_path":str(ITEMS/f"{row['work_id']}.heartbeat.json"),"mapper_timeout_seconds":max(5, int(float(row["timeout_seconds"]))-10)}
    spec_path=ITEMS/f"{row['work_id']}.spec.json"; spec_path.write_text(json.dumps(spec))
    env=os.environ.copy(); env["PYTHONPATH"]=str(ROOT)
    for name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS","BLIS_NUM_THREADS","TORCH_NUM_THREADS"): env[name]="1"
    out=(ROOT/row["stdout_path"]).open("a"); err=(ROOT/row["stderr_path"]).open("a")
    proc=subprocess.Popen([sys.executable,"-m","scripts.run_v4b_worker","--spec",str(spec_path)],cwd=ROOT,env=env,stdout=out,stderr=err,start_new_session=True)
    row["worker_pid"]=proc.pid; peak=0.; cpu=0.; timed=False
    deadline=time.monotonic()+float(row["timeout_seconds"])
    while proc.poll() is None:
        try:
            p=psutil.Process(proc.pid); mem=p.memory_info().rss/1024**2; c=p.cpu_times(); peak=max(peak,mem); cpu=c.user+c.system
        except psutil.NoSuchProcess: pass
        hb=ITEMS/f"{row['work_id']}.heartbeat.json"
        if hb.exists():
            try: row["last_heartbeat"]=str(json.loads(hb.read_text())["timestamp"])
            except Exception: pass
        if time.monotonic()>deadline:
            timed=True; kill_tree(proc); break
        time.sleep(1)
    out.close(); err.close(); row["end_time"]=str(time.time()); row["wall_seconds"]=float(row["end_time"])-float(row["start_time"]); row["cpu_seconds"]=cpu; row["peak_rss_mb"]=peak; row["exit_code"]=proc.poll()
    result=ROOT/row["result_path"]
    if timed:
        row.update(status="TIMEOUT", error_type="HardWallTimeout", error_message="process tree terminated after configured limit")
    elif result.exists():
        payload=json.loads(result.read_text())
        if "row" in payload: row["status"]="DONE"
        else: row.update(status="ERROR",error_type=payload.get("error_type","WorkerError"),error_message=payload.get("error_message",""))
    else: row.update(status="ERROR",error_type="NoAtomicResult",error_message="worker exited without result")
    return row


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--stage",choices=["length","cheap","top4","parent","full"],required=True); parser.add_argument("--limit",type=int); parser.add_argument("--concurrency",type=int,default=1); args=parser.parse_args()
    rows=read_manifest()
    required=make_manifest(args.stage)
    existing={r["work_id"] for r in rows}
    additions=[r for r in required if r["work_id"] not in existing]
    if additions:
        rows.extend(additions); write_manifest(rows)
    stage_methods={"length":{"length"},"cheap":{"revision_v1_pred_link","revision_v2_action_gnn","dual_linear","residual_gnn_no_correction"},"top4":{"selected_v3_top4"},"parent":{"hybrid_top4","noparent_top4"},"full":{"full_relaxed_lookahead"}}[args.stage]
    selected=[r for r in rows if r["method"] in stage_methods and r["status"] in ("PENDING","RETRYABLE") and int(r.get("attempt") or 0)<2]
    if args.limit: selected=selected[:args.limit]
    def finish(result):
        index=next(i for i,r in enumerate(rows) if r["work_id"]==result["work_id"])
        rows[index]=result; write_manifest(rows)
        print(result["work_id"],result["status"],flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1,min(2,args.concurrency))) as pool:
        futures = [pool.submit(run_item, row) for row in selected]
        # Consume completion order, never submission order: a single hard
        # instance cannot hide already-finished results behind it.
        for future in concurrent.futures.as_completed(futures):
            finish(future.result())
    print(pd.DataFrame(rows).status.value_counts().to_dict())


if __name__=="__main__": main()
