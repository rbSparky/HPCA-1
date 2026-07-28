#!/usr/bin/env python3
"""Audit candidate-cap recall using v3 full-action audit states."""
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'results/revision_v4'; RAW=OUT/'raw'
d=pd.read_csv(ROOT/'results/revision_v3/raw/action_advantage_labels.csv')
d=d[d.full_action_audit_state.astype(bool)]
rows=[]
for sid,g in d.groupby('state_id',sort=True):
 g=g.copy().sort_values(['immediate_cost','action_id'],kind='stable').reset_index(drop=True)
 best=float(g.delta_star.min()); eps=float(max(1e-4,.01*max(g.delta_star.max()-g.delta_star.min(),1.0)))
 order=[]
 for col,asc in [('immediate_cost',True),('dual_baseline',True),('new_link_fraction',False),('route_length_normalized',True)]:
  order.extend(g.sort_values([col,'action_id'],ascending=[asc,True]).index.tolist()[:6])
 order.extend(g.index.tolist()); seen=[]
 for i in order:
  if i not in seen: seen.append(i)
 for cap in (12,16,20,24):
  keep=seen[:cap]; kg=g.iloc[keep]
  rows.append({'state_id':sid,'split':g.split.iloc[0],'architecture':g.architecture.iloc[0],'dfg_family':g.dfg_family.iloc[0],'cap':cap,'actions_available':len(g),'actions_retained':len(kg),'true_best_recall':bool((kg.delta_star<=best+1e-9).any()),'epsilon_recall':bool((kg.delta_star<=best+eps).any()),'top4_set_recall':len(set(g.nsmallest(4,'delta_star').action_id)&set(kg.action_id))/min(4,len(g)),'best_retained_regret':float(kg.delta_star.min()-best)})
pd.DataFrame(rows).to_csv(RAW/'candidate_audit.csv',index=False)
print(pd.DataFrame(rows).groupby('cap')[['true_best_recall','epsilon_recall','top4_set_recall']].mean())
