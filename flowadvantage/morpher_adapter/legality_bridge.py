"""Independent validator for ``flowadvantage_morpher_mapping_v1`` documents.

The validator deliberately consumes the native II-expanded MRRG dump rather
than an architecture description.  It is independent of Morpher's mapper, but
uses the same stable resource IDs exported by the native bridge.
"""
from collections import Counter

def _index(items, key):
    return {x[key]: x for x in items if key in x}

def validate_mapping(mapping, dfg, mrrg):
    violations=[]
    if mapping.get("schema") != "flowadvantage_morpher_mapping_v1":
        violations.append({"class":"schema mismatch"})
    if mapping.get("ii") != mrrg.get("ii"):
        violations.append({"class":"II mismatch"})
    if mapping.get("architecture_hash") != mrrg.get("architecture_hash"):
        violations.append({"class":"architecture-hash mismatch"})
    if mapping.get("dfg_hash") != dfg.get("dfg_hash"):
        violations.append({"class":"DFG-hash mismatch"})
    nodes=_index(dfg.get("nodes",[]),"dfg_node_id")
    resources=_index(mrrg.get("resources",[]),"native_resource_id")
    edges={(e.get("src"),e.get("dst")) for e in mrrg.get("edges",[])}
    ops=mapping.get("operations",[]); seen=set(); occupancy=Counter()
    for op in ops:
        nid=op.get("dfg_node_id")
        if nid not in nodes: violations.append({"class":"unknown DFG node","node":nid}); continue
        if nid in seen: violations.append({"class":"duplicate compute occupancy","node":nid})
        seen.add(nid)
        pe=resources.get(op.get("pe_id"))
        if not pe: violations.append({"class":"unknown MRRG resource","resource":op.get("pe_id")}); continue
        if pe.get("resource_type") != "PE":
            violations.append({"class":"compute resource type","resource":op.get("pe_id")})
        if pe.get("time_slot") != op.get("modulo_time"): violations.append({"class":"timing violation","node":nid})
        fu=resources.get(op.get("fu_id"))
        if not fu:
            violations.append({"class":"unknown MRRG resource","resource":op.get("fu_id")})
        else:
            if fu.get("resource_type") != "FU":
                violations.append({"class":"unsupported operation","node":nid,"opcode":nodes[nid].get("opcode"),"fu":op.get("fu_id"),"reason":"not_a_function_unit"})
            supported=set(fu.get("supported_operations", []))
            opcode=str(nodes[nid].get("opcode", op.get("opcode", ""))).upper()
            if supported and opcode not in supported:
                violations.append({"class":"unsupported operation","node":nid,"opcode":opcode,"fu":op.get("fu_id")})
        # Morpher's exported PE time is a resource phase.  Its native mapper
        # may add datapath latency before interpreting ASAP/ALAP, so comparing
        # that phase directly with the XML ASAP is unsound.  If an external
        # producer supplies an absolute schedule, it is checked explicitly;
        # otherwise placement-time legality remains owned by native import.
        if op.get("absolute_schedule_if_available") is not None:
            absolute=int(op["absolute_schedule_if_available"])
            asap=int(nodes[nid].get("asap", op.get("asap", 0)))
            alap=int(nodes[nid].get("alap", op.get("alap", asap)))
            if not asap <= absolute <= alap:
                violations.append({"class":"ASAP/ALAP violation","node":nid,"absolute_schedule":absolute,"asap":asap,"alap":alap})
        occupancy[(op.get("pe_id"),op.get("fu_id"),op.get("modulo_time"))]+=1
    for slot,n in occupancy.items():
        if n>1: violations.append({"class":"duplicate compute occupancy","slot":str(slot)})
    if set(nodes)-seen: violations.append({"class":"unconsumed operation","nodes":sorted(set(nodes)-seen)})
    dep={(x.get("source_node"),x.get("destination_node")):x for x in dfg.get("dependencies",[])}
    routed=set()
    used={}
    for route in mapping.get("routes",[]):
        pair=(route.get("source_node"),route.get("destination_node")); routed.add(pair)
        if pair not in dep: violations.append({"class":"wrong route endpoints","edge":route.get("edge_id")}); continue
        ids=route.get("ordered_resource_ids",[])
        if not ids: violations.append({"class":"wrong route endpoints","edge":route.get("edge_id")}); continue
        for rid in ids:
            if rid not in resources: violations.append({"class":"unknown MRRG resource","resource":rid})
            used.setdefault(rid,set()).add(pair[0])
        for a,b in zip(ids,ids[1:]):
            if (a,b) not in edges: violations.append({"class":"non-contiguous route","link":f"{a}->{b}"})
        # Route endpoints are represented by the first/last native port.  The
        # exact mapper-owned port IDs vary by FU, so verify endpoint PE/time
        # affinity via the stable placement records instead of name heuristics.
        src_op=next((x for x in ops if x.get("dfg_node_id")==pair[0]), None)
        dst_op=next((x for x in ops if x.get("dfg_node_id")==pair[1]), None)
        if not src_op or not dst_op:
            violations.append({"class":"wrong route endpoints","edge":route.get("edge_id")})
        else:
            src_pe=str(src_op.get("pe_id", "")).split(".PE_")[0]
            dst_pe=str(dst_op.get("pe_id", "")).split(".PE_")[0]
            if src_pe and src_pe not in ids[0]:
                violations.append({"class":"wrong route endpoints","edge":route.get("edge_id"),"end":"source"})
            if dst_pe and dst_pe not in ids[-1]:
                violations.append({"class":"wrong route endpoints","edge":route.get("edge_id"),"end":"destination"})
    for pair in dep:
        if pair not in routed: violations.append({"class":"missing route","edge":pair})
    for rid,signals in used.items():
        n=len(signals)
        if n>resources.get(rid,{}).get("capacity",1): violations.append({"class":"routing capacity conflict","resource":rid,"uses":n})
    return {"legal":not violations,"violations":violations,"counts":{"operations":len(ops),"dependencies":len(dep),"routes":len(mapping.get("routes",[]))}}

def validate_imported_mapping(mapping, dfg, arch):
    # Compatibility wrapper for old callers; full validation requires mrrg.
    if "resources" in arch: return validate_mapping(mapping, dfg, arch)
    ids=set(dfg.get("nodes",{})); assigned=set(int(k) for k in mapping.get("assignments",{}) if str(k).lstrip('-').isdigit())
    return {"legal":assigned.issubset(ids),"violations":[] if assigned.issubset(ids) else [{"class":"unknown DFG node"}]}
