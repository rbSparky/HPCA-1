#!/usr/bin/env bash
ssh mll5090 "python3 -c \"
import csv
with open('/home/Rishabh@MLL-5090/remote-work/HPCA/results/paper_suite_v2_manifest_v7_remote/work_manifest.csv') as f:
    rows = list(csv.DictReader(f))
counts = {}
for r in rows:
    st = r['status']
    counts[st] = counts.get(st, 0) + 1
print('STATUS_COUNTS:', counts)
for r in rows:
    st = r['status']
    if st in ('RUNNING', 'PENDING'):
        print(st + ':', r['kernel'], '@', r['architecture'], '(', r['method'], ')')
\""
