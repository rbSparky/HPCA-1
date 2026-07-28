#!/usr/bin/env python3
"""Materialize honest v5b tables/plots from atomic bridge evidence."""
from __future__ import annotations
import csv, json
from pathlib import Path
import matplotlib.pyplot as plt

ROOT=Path("results/revision_v5b")
RAW=ROOT/"raw"; TABLES=ROOT/"tables"; PLOTS=ROOT/"plots"
PAIRS=[
 ("array_add","A0_hycube4x4","DONE","native dump, import, binary; simulator assertion"),
 ("array_add","A1_stdnoc4x4","ERROR","unbounded native mapper run ended in Module::getInternalPort assertion"),
 ("array_add","A2_hycube4x4_mem_variant","ERROR","native mapper abort: thrown int"),
 ("gemm_nt","A0_hycube4x4","TIMEOUT","native mapper exceeded bounded reference attempt"),
 ("gemm_nt","A1_stdnoc4x4","TIMEOUT","native mapper exceeded bounded reference attempt"),
 ("gemm_nt","A2_hycube4x4_mem_variant","ERROR","native mapper abort: thrown int"),
 ("fix_fft","A0_hycube4x4","ERROR","DFG parser assertion npb == 0 || npb == 1"),
 ("fix_fft","A1_stdnoc4x4","ERROR","DFG parser assertion npb == 0 || npb == 1"),
 ("fix_fft","A2_hycube4x4_mem_variant","ERROR","DFG parser assertion npb == 0 || npb == 1"),]

def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True,exist_ok=True)
    if fields is None: fields=list(rows[0]) if rows else []
    with path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore"); w.writeheader(); w.writerows(rows)

def markdown(path, title, rows, fields=None):
    fields=fields or (list(rows[0]) if rows else [])
    lines=[f"# {title}", "", "|"+"|".join(fields)+"|", "|"+"|".join(["---"]*len(fields))+"|"]
    for row in rows: lines.append("|"+"|".join(str(row.get(x,"")) for x in fields)+"|")
    path.write_text("\n".join(lines)+"\n")

def table(name,title,rows,fields=None):
    write_csv(TABLES/f"{name}.csv",rows,fields); markdown(TABLES/f"{name}.md",title,rows,fields)

def note_plot(name,title,text):
    fig,ax=plt.subplots(figsize=(7,3)); ax.axis("off"); ax.set_title(title); ax.text(.5,.5,text,ha="center",va="center",wrap=True); fig.tight_layout(); fig.savefig(PLOTS/name,dpi=160); plt.close(fig)

def main():
    RAW.mkdir(exist_ok=True); TABLES.mkdir(exist_ok=True); PLOTS.mkdir(exist_ok=True)
    attempts=[]
    for k,a,status,reason in PAIRS:
        attempts.append({"kernel":k,"architecture":a,"method":"native_pathfinder_reference","seed":11,"II":"auto","lower_bound_II":"not_recorded","status":status,"legal":status=="DONE","FlowAdvantage_legality":status=="DONE","Morpher_legality":False,"simulation_completed":False,"output_match":False,"route_cost":"","routing_failures":"","backtracks":"","expansions":"","generated_actions":"","parent_solves":0,"child_solves":0,"compile_wall_seconds":"","feature_seconds":0,"proposal_seconds":0,"relaxation_seconds":0,"native_mapper_seconds":"","simulation_seconds":"","reason":reason})
    write_csv(RAW/"ii_attempts.csv",attempts)
    write_csv(RAW/"pilot_runs.csv",attempts)
    errors=[{"kernel":k,"architecture":a,"stage":"native_reference_mapping","status":s,"error":r} for k,a,s,r in PAIRS if s!="DONE"]
    errors.append({"kernel":"array_add","architecture":"A0_hycube4x4","stage":"cycle_simulation","status":"ERROR","error":"CGRATile::store assertion op2 % 4 == 0 with imported binary and available test trace"})
    write_csv(RAW/"adapter_errors.csv",errors)
    write_csv(RAW/"bootstrap_results.csv",[],["comparison","metric","point_estimate","ci_low","ci_high","paired_instances","note"])
    kc=[]
    for k in ["array_add","array_cond","hpcg","trmm","gemm_nt","fix_fft"]:
        kc.append({"kernel":k,"DFG_nodes":20 if k=="array_add" else "NOT_EXPORTED","DFG_edges":23 if k=="array_add" else "NOT_EXPORTED","recurrence_edges":"NOT_EXPORTED","ASAP_ALAP_span":"NOT_EXPORTED","operation_histogram":"native JSON only for array_add" if k=="array_add" else "NOT_ASSESSED","memory_operations":"NOT_ASSESSED","unsupported_compute_fraction":"NOT_ASSESSED","zero_shot_status":"BLOCKED_A0"})
    write_csv(RAW/"kernel_census.csv",kc)
    ac=[]
    for a in ["A0_hycube4x4","A1_stdnoc4x4","A2_hycube4x4_mem_variant","A3_hycube8x8"]:
        ac.append({"architecture":a,"ii":4 if a=="A0_hycube4x4" else "NOT_EXPORTED","native_resources":2522 if a=="A0_hycube4x4" else "NOT_EXPORTED","directed_edges":4656 if a=="A0_hycube4x4" else "NOT_EXPORTED","status":"DUMPED" if a=="A0_hycube4x4" else "NOT_ASSESSED"})
    write_csv(RAW/"architecture_census.csv",ac)
    roundtrip=list(csv.DictReader((RAW/"roundtrip_validation.csv").open()))
    negative=list(csv.DictReader((RAW/"negative_legality_tests.csv").open()))
    table("T01_native_contract","Native contract map",list(csv.DictReader((RAW/"contract_symbols.csv").open())))
    table("T02_roundtrip","Nine-pair round-trip validation",roundtrip)
    table("T03_negative","Negative legality tests",negative)
    table("T04_kernels","Kernel characteristics",kc)
    table("T05_architectures","Architecture/MRRG characteristics",ac)
    coverage=[{"method":"native PathFinder reference","terminal_pairs":9,"normal_pairs":1,"timeout_pairs":2,"error_pairs":6},{"method":"FlowAdvantage proposal/top4","terminal_pairs":0,"normal_pairs":0,"timeout_pairs":0,"error_pairs":0}]
    table("T06_coverage","Pilot coverage",coverage)
    na=[{"method":x,"status":"NOT_ASSESSED_A0_RED"} for x in ["PathFinder","simulated annealing","LISA","dual-linear","FlowAdvantage proposal","FlowAdvantage top4","full relaxed lookahead"]]
    for n,t in [("T07_success","Success by method"),("T08_minimum_ii","Minimum II by method"),("T09_routing","Routing failures and route cost"),("T10_compile","Compile-time comparison"),("T11_top4_full","Top-4 versus full relaxed lookahead"),("T12_a3","Held-out A3 portability")]: table(n,t,na)
    gates=[
      {"gate":"A0","status":"RED","evidence":"1/9 native dump+Flow legality+structural reimport; 0/9 native legality/config+simulation"},
      {"gate":"R0","status":"RED","evidence":"A0 is RED"},
      {"gate":"R1","status":"NOT_ASSESSED","evidence":"Blocked by R0"},
      {"gate":"R2","status":"NOT_ASSESSED","evidence":"No paired real mapping pilot"},
      {"gate":"R3","status":"NOT_ASSESSED","evidence":"No full relaxed-lookahead subset"},
      {"gate":"R4","status":"NOT_ASSESSED","evidence":"No A3 pilot"},
      {"gate":"R5","status":"NOT_ASSESSED","evidence":"No paired pilot"},]
    table("T13_gates","A0 and R0–R5 gates",gates)
    write_csv(ROOT/"gates.csv",gates)
    status_vals={"DONE":1,"TIMEOUT":0,"ERROR":-1}
    fig,ax=plt.subplots(figsize=(7,3)); ax.imshow([[status_vals[s] for _,_,s,_ in PAIRS[:3]],[status_vals[s] for _,_,s,_ in PAIRS[3:6]],[status_vals[s] for _,_,s,_ in PAIRS[6:]]],cmap="RdYlGn",vmin=-1,vmax=1); ax.set_xticks(range(3),[x[1] for x in PAIRS[:3]],rotation=20); ax.set_yticks(range(3),["array_add","gemm_nt","fix_fft"]); ax.set_title("Adapter validation terminal matrix (green=DONE, yellow=timeout, red=error)"); fig.tight_layout(); fig.savefig(PLOTS/"F01_adapter_validation_matrix.png",dpi=160); plt.close(fig)
    note_plot("F02_mapping_success.png","Mapping success by method","Not assessed: A0 adapter gate is RED.")
    note_plot("F03_minimum_ii.png","Minimum-II heatmap","Not assessed: no paired real pilot was run.")
    note_plot("F04_routing_failures.png","Routing failures by method","Not assessed: no FlowAdvantage native mapping run.")
    note_plot("F05_compile_time.png","Compile-time distribution","Only native reference attempts completed; no comparable FlowAdvantage runtime.")
    note_plot("F06_top4_full.png","Top-4/full-lookahead comparison","Not assessed: full relaxed lookahead is blocked by A0.")
    note_plot("F07_a3.png","Held-out architecture comparison","Not assessed: A3 was never used for training or pilot mapping.")

if __name__=="__main__": main()
