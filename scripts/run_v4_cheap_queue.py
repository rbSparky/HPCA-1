#!/usr/bin/env python3
"""Complete frozen-v3 cheap mapper coverage in the v4 round-robin queue."""
from pathlib import Path
import concurrent.futures, os, pickle
import pandas as pd
from scripts.run_v3_mappers import run_instance

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/revision_v4'; RAW=OUT/'raw'
v3=pd.read_csv(ROOT/'results/revision_v3/raw/stress_mapper_runs.csv')
existing=v3[v3.method.isin(['length','revision_v1_pred_link','revision_v2_action_gnn','dual_linear','residual_gnn_no_correction'])].copy()
inst=pd.read_csv(ROOT/'results/revision_v3/raw/stress_instances.csv')
def work(path): return run_instance(path, False)
def main():
    all_rows=[]
    tasks=[]
    for r in inst.itertuples(): tasks.append((r.instance_path,False))
    # run_instance reads these overrides in each child process
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
      for rows in pool.map(work,[x[0] for x in tasks]):
        all_rows.extend(rows)
    fresh=pd.DataFrame(all_rows)
    combined=pd.concat([existing,fresh],ignore_index=True)
    combined=combined.drop_duplicates(['instance_id','budget','method'],keep='last')
    combined.to_csv(RAW/'stress_v4_cheap_runs.csv',index=False)
    print({'rows':len(combined),'instances':combined.instance_id.nunique(),'methods':combined.method.nunique(),'budgets':combined.budget.nunique()})
if __name__=='__main__': main()
