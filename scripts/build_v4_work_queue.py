#!/usr/bin/env python3
"""Build an immutable round-robin v4 queue from the frozen v3 instances."""
from pathlib import Path
import json
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/revision_v4'
inst=pd.read_csv(ROOT/'results/revision_v3/raw/stress_instances.csv')
run=pd.read_csv(ROOT/'results/revision_v3/raw/stress_mapper_runs.csv')
methods=['length','revision_v1_pred_link','revision_v2_action_gnn','dual_linear','residual_gnn_no_correction']
budgets=['tight','medium','wide']
done=set(zip(run.instance_id,run.budget,run.method))
items=[]
for family in sorted(inst.dfg_family.unique()):
  for arch in sorted(inst.architecture.unique()):
    for seed in sorted(inst.seed.unique()):
      sub=inst[(inst.dfg_family==family)&(inst.architecture==arch)&(inst.seed==seed)]
      for row in sub.itertuples():
        for budget in budgets:
          for method in methods:
            key=(row.instance_id,budget,method)
            items.append({'queue_key':'|'.join(map(str,key)),'instance_id':row.instance_id,'instance_path':row.instance_path,'dfg_family':family,'architecture':arch,'seed':seed,'budget':budget,'method':method,'already_observed':key in done})
df=pd.DataFrame(items)
df['_h']=pd.util.hash_pandas_object(df[['instance_id','budget','method']],index=False).astype('uint64') ^ 24072026
df=df.sort_values(['method','budget','_h'],kind='stable').drop(columns='_h').reset_index(drop=True)
df.insert(0,'queue_index',range(len(df)))
df.to_csv(OUT/'raw/v4_work_queue.csv',index=False)
(OUT/'raw/v4_work_queue.json').write_text(json.dumps({'seed':24072026,'items':len(df),'methods':methods,'budgets':budgets},indent=2)+'\n')
print(df.head(20).to_string(index=False)); print('missing',int((~df.already_observed).sum()))
