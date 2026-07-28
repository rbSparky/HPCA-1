#!/usr/bin/env python3
"""Recover stale v4b work without deleting logs or result files."""
from pathlib import Path
import csv, time
import psutil

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'results/revision_v4b'; manifest=OUT/'work_manifest.csv'
rows=list(csv.DictReader(manifest.open()))
for row in rows:
    if row['status']=='ERROR' and row.get('error_type')=='AttributeError' and int(row.get('attempt') or 0)<2:
        row['status']='RETRYABLE'; continue
    if row['status']!='RUNNING': continue
    live=False
    try: live=bool(row['worker_pid']) and psutil.pid_exists(int(row['worker_pid']))
    except ValueError: pass
    stale=(time.time()-float(row.get('last_heartbeat') or 0))>40
    if not live or stale:
        row['status']='RETRYABLE' if int(row.get('attempt') or 0)<2 else 'TIMEOUT'
        row['error_type']='StaleWorkItem'; row['error_message']='process absent or heartbeat older than 40 seconds'
tmp=manifest.with_suffix('.tmp')
with tmp.open('w',newline='') as h:
 w=csv.DictWriter(h,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows); h.flush(); __import__('os').fsync(h.fileno())
tmp.replace(manifest)
print({s:sum(r['status']==s for r in rows) for s in sorted({r['status'] for r in rows})})
