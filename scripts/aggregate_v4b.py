#!/usr/bin/env python3
"""Aggregate completed atomic work JSON files without altering them."""
from pathlib import Path
import json, pandas as pd
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'results/revision_v4b'; ITEMS=OUT/'work_items'; RAW=OUT/'raw'
rows=[]
manifest=pd.read_csv(OUT/'work_manifest.csv')
done_ids=set(manifest.loc[manifest.status=='DONE','work_id'])
for p in ITEMS.glob('*.json'):
 if p.name.endswith(('.spec.json','.heartbeat.json')): continue
 try:
  x=json.loads(p.read_text())
  if 'row' in x and x.get('work_id') in done_ids: rows.append(x['row'])
 except Exception: pass
d=pd.DataFrame(rows)
if len(d):
 for method, group in d.groupby('method'):
  name={'length':'length_medium_runs','selected_v3_top4':'selected_top4_runs','hybrid_top4':'parent_ablation_runs','noparent_top4':'parent_ablation_runs','full_relaxed_lookahead':'full_lookahead_runs'}.get(method,'cheap_medium_runs')
  path=RAW/f'{name}.csv'; old=pd.DataFrame()
  pd.concat([old,group],ignore_index=True).drop_duplicates(['instance_id','method','budget'],keep='last').to_csv(path,index=False)
cheap=d[~d.method.isin(['length','selected_v3_top4','hybrid_top4','noparent_top4','full_relaxed_lookahead'])]
cheap.to_csv(RAW/'cheap_medium_runs.csv',index=False)
print(d.groupby('method').size().to_dict() if len(d) else {})
