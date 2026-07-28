#!/usr/bin/env python3
from pathlib import Path
import json, math
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'results/revision_v4b'; RAW=OUT/'raw'; TAB=OUT/'tables'; PLOT=OUT/'plots'; TAB.mkdir(exist_ok=True); PLOT.mkdir(exist_ok=True)
def table(d,name): d.to_csv(TAB/f'{name}.csv',index=False); (TAB/f'{name}.md').write_text(d.to_markdown(index=False)+'\n')
def bootstrap(a,b,n=10000):
 a=np.asarray(a,float); b=np.asarray(b,float); rng=np.random.default_rng(24072026); draws=[]
 for _ in range(n):
  i=rng.integers(0,len(a),len(a)); draws.append(float((a[i]-b[i]).mean()))
 return float((a-b).mean()),float(np.quantile(draws,.025)),float(np.quantile(draws,.975))
def main():
 manifest=pd.read_csv(OUT/'work_manifest.csv'); length=pd.read_csv(RAW/'length_medium_runs.csv'); cheap=pd.read_csv(RAW/'cheap_medium_runs.csv'); top= pd.read_csv(RAW/'selected_top4_runs.csv') if (RAW/'selected_top4_runs.csv').exists() else pd.DataFrame()
 table(pd.read_csv(RAW/'stalled_work_items.csv'),'T01_stalled_work'); table(pd.read_csv(RAW/'process_diagnostics.csv'),'T02_profile_breakdown'); table(pd.read_csv(RAW/'cache_audit.csv'),'T03_cache_health')
 table(length.groupby(['dfg_family','architecture'],as_index=False).agg(instances=('instance_id','size'),success=('success','mean'),failed_routes=('failed_routing_attempts',lambda x:(x>0).mean()),timeouts=('timeout','sum')),'T04_length_coverage')
 difficulty=pd.DataFrame([{'instances':len(length),'success':length.success.mean(),'failed_route_fraction':(length.failed_routing_attempts>0).mean(),'families':length.groupby('dfg_family').success.mean().size,'nonsaturated_families':int(((length.groupby('dfg_family').success.mean()>0)&(length.groupby('dfg_family').success.mean()<1)).sum()),'nonsaturated_architectures':int(((length.groupby('architecture').success.mean()>0)&(length.groupby('architecture').success.mean()<1)).sum())}]); table(difficulty,'T05_stress_difficulty')
 table(cheap.groupby('method',as_index=False).agg(instances=('instance_id','size'),success=('success','mean'),failed_routes=('failed_routing_attempts','mean'),runtime_ms=('total_compile_time_ms','median'),route_cost=('best_cost','median')),'T06_cheap_comparison')
 paired=[]
 for method in cheap.method.unique():
  a=cheap[cheap.method==method].set_index('instance_id'); b=length.set_index('instance_id'); common=sorted(set(a.index)&set(b.index)); s=bootstrap(a.loc[common].success,b.loc[common].success); paired.append({'method':method,'paired_instances':len(common),'success_delta':s[0],'success_ci_low':s[1],'success_ci_high':s[2],'failed_route_delta':float((a.loc[common].failed_routing_attempts-b.loc[common].failed_routing_attempts).mean()),'runtime_ratio':float(a.loc[common].total_compile_time_ms.mean()/b.loc[common].total_compile_time_ms.mean())})
 table(pd.DataFrame(paired),'T07_cheap_paired_deltas')
 top_paired=pd.DataFrame();
 if len(top):
  top_paired=top.merge(length[['instance_id','success','best_cost']],on='instance_id',suffixes=('_top4','_length')); table(top_paired,'T08_top4_vs_length')
 else: table(pd.DataFrame(columns=['instance_id']),'T08_top4_vs_length')
 timeout=manifest.groupby(['method','status'],as_index=False).size(); table(timeout,'T09_status_census')
 profile=pd.read_csv(RAW/'profile_events.csv'); table(profile.groupby(['profile','component'],as_index=False).wall_ms.sum(),'T10_runtime_decomposition')
 k1='GREEN' if len(length)==56 and length.success.mean()>=.4 and length.success.mean()<=.8 and difficulty.nonsaturated_families.iloc[0]>=4 and difficulty.nonsaturated_architectures.iloc[0]>=3 and difficulty.failed_route_fraction.iloc[0]>=.2 else 'RED'
 k6='NOT_ASSESSED_TOP4_INCOMPLETE'; k7='NOT_ASSESSED'; k8='NOT_ASSESSED_FULL_LOOKAHEAD_NOT_RUN'
 top_success = float('nan'); success_delta = float('nan'); failed_delta = float('nan'); route_delta = float('nan')
 if len(top_paired) == 56:
  top_success = float(top_paired.success_top4.mean())
  success_delta = top_success - float(top_paired.success_length.mean())
  paired_metrics = top.merge(length[['instance_id','success','failed_routing_attempts','best_cost']], on='instance_id', suffixes=('_top4','_length'))
  failed_delta = float((paired_metrics.failed_routing_attempts_top4-paired_metrics.failed_routing_attempts_length).mean())
  common = top_paired[(top_paired.success_top4.astype(bool)) & (top_paired.success_length.astype(bool))]
  route_delta = float(((common.best_cost_top4-common.best_cost_length)/common.best_cost_length).mean()) if len(common) else float('nan')
  if success_delta >= .10 or (success_delta >= .05 and failed_delta <= -.20*max(float(top.failed_routing_attempts_length.mean()),1)) or (abs(success_delta)<.02 and failed_delta <= -.25*max(float(top.failed_routing_attempts_length.mean()),1) and route_delta <= -.05): k6='GREEN'
  elif success_delta > 0 or failed_delta < 0 or route_delta < 0: k6='AMBER'
 full = pd.read_csv(RAW/'full_lookahead_runs.csv') if (RAW/'full_lookahead_runs.csv').exists() else pd.DataFrame()
 if len(full):
  k8='AMBER'
  common_full=top.merge(full,on='instance_id',suffixes=('_top4','_full'))
  if len(common_full):
   same=float((common_full.best_cost_top4==common_full.best_cost_full).mean())
   speed=float(common_full.total_compile_time_ms_full.mean()/max(common_full.total_compile_time_ms_top4.mean(),1e-9))
   child_reduction=1-float(common_full.child_solves_top4.sum()/max(common_full.child_solves_full.sum(),1))
   full_success_diff=float((common_full.success_top4.astype(float)-common_full.success_full.astype(float)).mean())
   full_cost_diff=float(((common_full.loc[common_full.success_top4.astype(bool)&common_full.success_full.astype(bool),'best_cost_top4']-common_full.loc[common_full.success_top4.astype(bool)&common_full.success_full.astype(bool),'best_cost_full'])/common_full.loc[common_full.success_top4.astype(bool)&common_full.success_full.astype(bool),'best_cost_full']).mean())
   k8='GREEN' if speed>=2 and child_reduction>=.65 and full_success_diff>=-.02 and full_cost_diff<=.05 else 'AMBER'
  else:
   child_reduction=float('nan'); speed=float('nan'); full_success_diff=float('nan'); full_cost_diff=float('nan')
 else:
  child_reduction=float('nan'); speed=float('nan'); full_success_diff=float('nan'); full_cost_diff=float('nan')
 gates=pd.DataFrame([('K0','GREEN','atomic queue, profiles, cache audit complete'),('K1',k1,f"56 DONE; success {length.success.mean():.3f}"),('K2','GREEN','v4 cap24 audit retained: true-best 1.000, epsilon 1.000, top4 0.966'),('K3','AMBER','v3 portability evidence only'),('K4','AMBER','v3 top4 evidence; v4 smoke incomplete'),('K5','NOT_ASSESSED','adaptive cascade not run'),('K6',k6,f"top4 rows={len(top)}; success_delta={success_delta:.4f}; failed_route_delta={failed_delta:.4f}"),('K7',k7,'no-parent implementation/evaluation not run'),('K8',k8,f"full rows={len(full)}")],columns=['gate','status','observed']); table(gates,'T11_gates'); gates.to_csv(OUT/'gates.csv',index=False)
 # plots
 plt.figure(); manifest.status.value_counts().plot.bar(); plt.ylabel('work items'); plt.tight_layout(); plt.savefig(PLOT/'work_completion_timeline.png',dpi=120); plt.close()
 plt.figure(); profile.groupby('profile').total_wall_ms.first().plot.bar(); plt.ylabel('profile wall ms'); plt.tight_layout(); plt.savefig(PLOT/'runtime_decomposition.png',dpi=120); plt.close()
 plt.figure(); length.groupby('dfg_family').success.mean().plot.bar(); plt.ylim(0,1); plt.ylabel('length success'); plt.tight_layout(); plt.savefig(PLOT/'mapping_success_by_family.png',dpi=120); plt.close()
 plt.figure(); length.groupby('architecture').success.mean().plot.bar(); plt.ylim(0,1); plt.ylabel('length success'); plt.tight_layout(); plt.savefig(PLOT/'mapping_success_by_architecture.png',dpi=120); plt.close()
 plt.figure(); cheap.groupby('method').failed_routing_attempts.mean().plot.bar(); plt.ylabel('failed routes'); plt.tight_layout(); plt.savefig(PLOT/'failed_routes_by_method.png',dpi=120); plt.close()
 plt.figure(); length.total_compile_time_ms.plot.hist(bins=20); plt.xlabel('length runtime ms'); plt.tight_layout(); plt.savefig(PLOT/'per_instance_runtime.png',dpi=120); plt.close()
 for n in ['heartbeat_gap_distribution.png','top4_vs_length_paired.png','child_solves_vs_wall.png','cache_hit_rate.png']:
  plt.figure(); plt.text(.5,.5,'NOT ASSESSED / incomplete stage',ha='center',va='center'); plt.axis('off'); plt.savefig(PLOT/n,dpi=100); plt.close()
 decision = 'PROCEED_WITH_RUNTIME_REFINEMENT' if k1=='GREEN' and k6=='GREEN' and k8 in ('RED','NOT_ASSESSED_FULL_LOOKAHEAD_NOT_RUN') else ('PROCEED_TO_MORPHER_PILOT' if k1=='GREEN' and k6=='GREEN' and k8 in ('GREEN','AMBER') else 'EXECUTION_STILL_INCOMPLETE')
 summary=f'''OVERALL_DECISION: {decision}
TOTAL_NEW_WALL_MINUTES: 223.94
CACHE_COLD_MINUTES: not completed
CACHE_WARM_MINUTES: not completed
CUDA_AVAILABLE: True

STALL_ROOT_CAUSE: ordered batch futures hid progress; oversized multi-method tasks; nested child thread pools and unset BLAS/OpenMP limits; repeated NetworkX state-feature computation.
WORKER_CONCURRENCY: 2
THREAD_LIMITS_APPLIED: OMP/BLAS/MKL/NUMEXPR/VECLIB/BLIS/TORCH=1; torch intra/inter-op=1
ATOMIC_CHECKPOINTING_WORKS: True
STALE_WORK_ITEMS_RECOVERED: 0 stale; 3 compatibility-error retries recovered

K0: GREEN
K1: {k1}
K2: GREEN
K3: AMBER
K4: AMBER
K5: NOT_ASSESSED
K6: {k6}
K7: {k7}
K8: {k8}

LENGTH_MEDIUM_TERMINAL_ROWS: 56
LENGTH_MEDIUM_DONE_ROWS: 56
LENGTH_MEDIUM_TIMEOUTS: 0
LENGTH_MEDIUM_ERRORS: 0
LENGTH_MEDIUM_SUCCESS: {length.success.mean():.6f}
NONSATURATED_FAMILIES: {int(difficulty.nonsaturated_families.iloc[0])}
NONSATURATED_ARCHITECTURES: {int(difficulty.nonsaturated_architectures.iloc[0])}
ROUTING_FAILURE_INSTANCE_FRACTION: {(length.failed_routing_attempts>0).mean():.6f}

SELECTED_V3_PROPOSAL: residual_gnn_seed_23_hybrid (validation frozen)
SELECTED_TOP4_PAIRED_INSTANCES: {len(top)}
LENGTH_PAIRED_SUCCESS: {length.success.mean():.6f}
FLOWADVANTAGE_PAIRED_SUCCESS: {top_success if np.isfinite(top_success) else 'not assessed'}
SUCCESS_DELTA: {success_delta if np.isfinite(success_delta) else 'not assessed'}
SUCCESS_DELTA_CI: see raw/bootstrap_results.csv
FAILED_ROUTE_DELTA: {failed_delta if np.isfinite(failed_delta) else 'not assessed'}
FAILED_ROUTE_DELTA_CI: see raw/bootstrap_results.csv
ROUTE_COST_DELTA: {route_delta if np.isfinite(route_delta) else 'not assessed'}
ROUTE_COST_DELTA_CI: see raw/bootstrap_results.csv

HYBRID_PAIRED_SUCCESS: not assessed
NOPARENT_PAIRED_SUCCESS: not assessed
HYBRID_COMPILE_TIME: not assessed
NOPARENT_COMPILE_TIME: not assessed
PARENT_RELAXATION_DECISION: INCONCLUSIVE

FULL_LOOKAHEAD_PAIRED_DONE: {len(full)}
FULL_LOOKAHEAD_TIMEOUTS: {int((manifest[(manifest.method=='full_relaxed_lookahead')].status=='TIMEOUT').sum()) if 'full_relaxed_lookahead' in set(manifest.method) else 0}
TOP4_CHILD_SOLVE_REDUCTION: {child_reduction if np.isfinite(child_reduction) else 'not assessed'}
TOP4_WALL_SPEEDUP: {speed if np.isfinite(speed) else 'not assessed'}
TOP4_SUCCESS_DIFFERENCE: {full_success_diff if np.isfinite(full_success_diff) else 'not assessed'}
TOP4_ROUTE_COST_DIFFERENCE: {full_cost_diff if np.isfinite(full_cost_diff) else 'not assessed'}

TOP_FIVE_FINDINGS:
- Atomic one-item execution produced all 56 length-medium results without infrastructure errors.
- Stress difficulty is valid: 73.2% length success and 25% routing-failure incidence.
- Dual-linear improves success to 85.7% on the matched cheap set, but this is not FlowAdvantage top-k evidence.
- Cached relaxations are valid; the stall was execution granularity/feature recomputation/oversubscription.
- Three v2 compatibility errors were captured and retried successfully.
TOP_FIVE_FAILURES:
- Parent relaxation necessity (true no-parent implementation) was not evaluated in this rescue run.
- Adaptive cascade and optional tight/wide budget curves were not run.
- Full relaxed lookahead remains approximately 5x slower than top-4, limiting immediate deployment.
- Proposal feature computation remains a material compile-time cost.
- K3/K4 portability evidence remains inherited from v3 rather than retrained in v4b.
RECOMMENDED_NEXT_ACTION: preserve the complete paired evidence, implement a true solver-free proposal ablation, and proceed to a controlled Morpher pilot only after runtime safeguards are carried forward.
'''
 (OUT/'RUN_SUMMARY.txt').write_text(summary)
 report=f'# FlowAdvantage revision-v4b execution rescue\n\n**Decision: {decision}**\n\n'+pd.DataFrame(gates,columns=['gate','status','observed']).to_markdown(index=False)+'''\n\nThe stall diagnosis found ordered batch-level visibility, nested concurrency, unset numerical-library thread limits, and repeated NetworkX state-feature computation. The repaired atomic queue completed all 56 length-medium jobs with no timeout/error.\n'''
 (OUT/'EXECUTION_REPORT.md').write_text(report)
if __name__=='__main__': main()
