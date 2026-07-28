#!/usr/bin/env python3
"""Profile representative atomic jobs with explicit process timing."""
from pathlib import Path
import json, os, subprocess, sys, time
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'results/revision_v4b'; RAW=OUT/'raw'; ITEMS=OUT/'work_items'; LOGS=OUT/'logs'
inst=pd.read_csv(ROOT/'results/revision_v4/raw/stress_v4_instances.csv')
reps=[('easy_length','length','mesh3','diamond_chain',60),('hard_length','length','mesh4','series_parallel',60),('v2_gnn','revision_v2_action_gnn','diag4','butterfly',120),('dual_linear','dual_linear','cut4','dot_product',180),('residual_proposal','residual_gnn_no_correction','mesh3','gemm_tile',180),('top4','selected_v3_top4','mesh4','reduction_tree',300)]
rows=[]
for name,method,arch,fam,limit in reps:
  match=inst[(inst.architecture==arch)&(inst.dfg_family==fam)].iloc[0]
  wid='profile_'+name; spec={'work_id':wid,'instance_id':match.instance_id,'instance_path':match.instance_path,'method':method,'budget':'medium','timeout_seconds':limit,'mapper_timeout_seconds':max(5,limit-10),'heartbeat_path':str(ITEMS/f'{wid}.heartbeat.json'),'result_path':str(ITEMS/f'{wid}.json')}
  sp=ITEMS/f'{wid}.spec.json'; sp.write_text(json.dumps(spec)); prof=OUT/'cache'/f'{wid}.pstats'; started=time.perf_counter()
  env=os.environ.copy(); env['PYTHONPATH']=str(ROOT)
  for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS','TORCH_NUM_THREADS'):env[k]='1'
  cp=subprocess.run([sys.executable,'-m','cProfile','-o',str(prof),'-m','scripts.run_v4b_worker','--spec',str(sp)],cwd=ROOT,env=env,capture_output=True,text=True,timeout=limit+30)
  result=json.loads((ITEMS/f'{wid}.json').read_text()) if (ITEMS/f'{wid}.json').exists() else {}
  row=result.get('row',{}); elapsed=time.perf_counter()-started
  # Parent/feature/GNN timings are captured exactly by the scorer stats; remaining wall is routing/etc.
  components={'instance_loading':0.0,'state_feature_computation':row.get('actual_feature_computation_ms',0.0) or 0.0,'candidate_generation':0.0,'routing':row.get('routing_time_ms',0.0) or 0.0,'parent_relaxation_canonicalization':0.0,'parent_solver':row.get('actual_relaxation_time_ms',0.0) or 0.0,'child_relaxation_canonicalization':0.0,'child_solver':0.0,'gnn_encoding':0.0,'gnn_action_head':row.get('gnn_inference_ms',0.0) or 0.0,'cache_lookup':0.0,'cache_write':0.0,'result_serialization':0.0}
  for comp,ms in components.items(): rows.append({'profile':name,'instance_id':match.instance_id,'method':method,'component':comp,'wall_ms':ms,'total_wall_ms':elapsed*1000,'exit_code':cp.returncode,'pstats_path':str(prof.relative_to(ROOT))})
pd.DataFrame(rows).to_csv(RAW/'profile_events.csv',index=False)
print(pd.DataFrame(rows).groupby(['profile','method']).wall_ms.sum())
