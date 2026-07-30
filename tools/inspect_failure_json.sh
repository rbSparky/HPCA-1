#!/usr/bin/env bash
ssh mll5090 "python3 -c \"
import json, glob
files = glob.glob('/home/Rishabh@MLL-5090/remote-work/HPCA/results/paper_suite_v2_manifest_v7_remote/work_items/array_cond*.json')
for f in files[:3]:
    print('=== FILE:', f)
    data = json.load(open(f))
    print(json.dumps(data, indent=2))
\""
