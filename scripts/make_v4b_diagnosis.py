#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'results/revision_v4b'; RAW=OUT/'raw'
v3=pd.read_csv(ROOT/'results/revision_v4/raw/stress_mapper_runs.csv'); inst=pd.read_csv(ROOT/'results/revision_v4/raw/stress_v4_instances.csv')
seen=set(v3.instance_id); rows=[]
for r in inst.itertuples():
 rows.append({'work_id':f'{r.instance_id}__medium__length','instance_id':r.instance_id,'method':'length','budget':'medium','architecture':r.architecture,'dfg_family':r.dfg_family,'seed':r.seed,'previous_status':'COMPLETED_PREFIX' if r.instance_id in seen else 'UNEXECUTED_BATCH','last_output_timestamp':'unknown_batch_output','last_heartbeat_timestamp':'not_implemented_v4','pid_if_known':'unknown','cpu_seconds_if_known':'unknown','wall_seconds_if_known':'unknown','peak_rss_mb_if_known':'unknown','last_completed_stage':'batch_return_only','cache_hits':'not_logged','cache_misses':'not_logged','exception':'','stdout_path':'not_created_v4','stderr_path':'not_created_v4'})
pd.DataFrame(rows).to_csv(RAW/'stalled_work_items.csv',index=False)
events=pd.read_csv(RAW/'profile_events.csv'); summ=events.groupby(['profile','method'],as_index=False).agg(total_wall_ms=('total_wall_ms','first'),accounted_ms=('wall_ms','sum'),exit_code=('exit_code','first'))
summ['dominant_component']=summ.apply(lambda r: events[(events.profile==r.profile)&(events.method==r.method)].sort_values('wall_ms',ascending=False).iloc[0].component,axis=1)
summ.to_csv(RAW/'process_diagnostics.csv',index=False)
(OUT/'STALL_DIAGNOSIS.md').write_text('''# v4 stall diagnosis

## Root cause

The prior run was **computing slowly and making results invisible at batch
granularity**, not deadlocked. `run_v4_cheap_queue.py` used ordered
`ProcessPoolExecutor.map` over 56 oversized tasks; each task ran five methods
at three budgets and returned one unbounded Python list only at task completion.
The parent therefore could wait behind an early expensive task even if later
workers had progressed. No per-item output or heartbeat existed.

The profiling evidence identifies a second dominant cost: `build_state_context`
performs NetworkX all-pairs shortest paths, edge betweenness, bridge discovery,
and per-dependency shortest paths. It was repeatedly invoked for mapper states
and consumed about 22 seconds in representative dual/top-4 jobs. The old outer
four-process pool also allowed the FlowScorer's inner ThreadPoolExecutor (up to
eight child solves) while BLAS/OpenMP limits were unset, creating oversubscription.

## Ruled out

- Cached relaxation files are valid; sampled cache reads are ~0.03--0.05 ms.
- No global file lock or cache corruption was found.
- No retry loop or invalid-instance exception was observed in profile jobs.

## Repair

`run_v4b_worker.py` owns one instance/method/budget, writes a 5-second
heartbeat and atomically publishes a JSON row. `run_v4b_queue.py` starts one
spawned subprocess at a time, has hard process-tree timeouts, immediately
updates a durable CSV manifest, and never uses nested multiprocessing.
''')
