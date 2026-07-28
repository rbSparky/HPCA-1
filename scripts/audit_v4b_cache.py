#!/usr/bin/env python3
"""Read-only cache integrity audit for v3/v4 caches."""
from pathlib import Path
import pickle, time
import pandas as pd
import quotientflow.relaxation  # register cached dataclass for pickle
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'results/revision_v4b'; RAW=OUT/'raw'
targets={'v3_mapper_parents':ROOT/'results/revision_v3/cache/mapper_parents','v3_mapper_children':ROOT/'results/revision_v3/cache/mapper_children','v3_parents':ROOT/'results/revision_v3/cache/parents','v3_children':ROOT/'results/revision_v3/cache/children'}
rows=[]
for name,d in targets.items():
 files=list(d.glob('*.pkl')) if d.exists() else []; valid=corrupt=0; reads=[]
 for p in files[:min(len(files),200)]:
  t=time.perf_counter()
  try:
   with p.open('rb') as h: pickle.load(h); valid+=1
  except Exception: corrupt+=1
  reads.append((time.perf_counter()-t)*1000)
 rows.append({'cache_type':name,'entries':len(files),'total_size_mb':sum(p.stat().st_size for p in files)/1024**2,'valid_entries_sampled':valid,'corrupt_entries_sampled':corrupt,'duplicate_keys':len(files)-len({p.stem for p in files}),'hit_rate':'unknown_historical','median_read_ms':float(pd.Series(reads).median()) if reads else 0.,'median_write_ms':'not_measured_read_only','semantic_key_note':'v3 key includes DFG, architecture, partial state, action where applicable, relaxation config'})
pd.DataFrame(rows).to_csv(RAW/'cache_audit.csv',index=False)
print(pd.DataFrame(rows).to_string(index=False))
