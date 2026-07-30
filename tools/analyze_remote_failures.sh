#!/usr/bin/env bash
ssh mll5090 "python3 -c \"
import csv, json
from pathlib import Path

manifest_path = Path('/home/Rishabh@MLL-5090/remote-work/HPCA/results/paper_suite_v2_manifest_v7_remote/work_manifest.csv')
with open(manifest_path) as f:
    rows = list(csv.DictReader(f))

failures = [r for r in rows if r['status'] in ('VALID_MAPPING_FAILURE', 'TIMEOUT', 'ERROR')]
print('Total failures/timeouts:', len(failures))

by_status = {}
for r in failures:
    st = r['status']
    by_status.setdefault(st, []).append(r)

print('=== BREAKDOWN BY STATUS ===')
for st, items in by_status.items():
    print(st + ':', len(items))

print('=== DETAILED ITEM ANALYSIS ===')
for r in failures:
    res_path = Path(r['result_path'])
    reason = r.get('error_message') or r.get('error_type') or 'No manifest error'
    if res_path.exists():
        try:
            with open(res_path) as jf:
                data = json.load(jf)
                reason = data.get('reason') or data.get('error_message') or data.get('failure_reason') or reason
        except Exception as e:
            reason = str(e)

    k = r['kernel']
    a = r['architecture']
    m = r['method']
    st = r['status']
    print(st + ' | ' + k + ' | ' + a + ' | ' + m + ' | Reason: ' + str(reason))
\""
