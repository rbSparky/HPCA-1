import xml.etree.ElementTree as ET
import re
import json
from pathlib import Path
from .operation_taxonomy import translate_operation

def import_dfg_xml(path):
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError:
        # Morpher sample files prepend MutexBB/RecParents sections and are
        # intentionally XML fragments rather than one well-formed document.
        raw = open(path, encoding="utf-8").read(); m = re.search(r"<DFG\b[\s\S]*</DFG>", raw)
        if not m: raise
        fragment = re.sub(r'("[^"]*")(?=[A-Za-z_][\w-]*=)', r'\1 ', m.group(0))
        root = ET.fromstring(fragment)
    dfg = root.find("DFG")
    if dfg is None: dfg = root
    nodes = {}
    for n in dfg.findall("Node"):
        idx = int(n.attrib["idx"]); op = (n.findtext("OP") or "").strip()
        outs = [{"dst": int(x.attrib["idx"]), "nextiter": int(x.attrib.get("nextiter", "0")), "type": x.attrib.get("type", "")} for x in n.findall("./Outputs/Output")]
        ins = [int(x.attrib["idx"]) for x in n.findall("./Inputs/Input")]
        rec = [int(x.attrib["idx"]) for x in n.findall("./RecParents/RecParent")]
        nodes[idx] = {"id": idx, "op": op, "operation": translate_operation(op), "asap": int(n.attrib.get("ASAP", "0")), "alap": int(n.attrib.get("ALAP", "0")), "bb": n.attrib.get("BB", ""), "const": n.attrib.get("CONST"), "inputs": ins, "outputs": outs, "rec_parents": rec}
    edges = []
    for src in nodes.values():
        for out in src["outputs"]:
            if out["dst"] in nodes: edges.append({"src": src["id"], "dst": out["dst"], "distance": out["nextiter"], "type": out["type"]})
    return {"format": "morpher_xml", "nodes": nodes, "edges": edges, "count": len(nodes), "unsupported_compute_nodes": sum(not x["operation"]["supported"] for x in nodes.values())}


def import_native_dfg(path):
    """Import Morpher's native JSON DFG dump without losing dependence data."""
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema") != "flowadvantage_morpher_dfg_v1":
        raise ValueError(f"unsupported native DFG schema: {payload.get('schema')!r}")
    nodes = {}
    for node in payload.get("nodes", []):
        node_id = node.get("dfg_node_id")
        if not isinstance(node_id, int) or node_id in nodes:
            raise ValueError(f"invalid or duplicate DFG node ID: {node_id!r}")
        record = dict(node)
        record["operation"] = translate_operation(record.get("opcode", ""))
        nodes[node_id] = record
    edges = []
    for edge in payload.get("dependencies", []):
        source, destination = edge.get("source_node"), edge.get("destination_node")
        if source not in nodes or destination not in nodes:
            raise ValueError(f"dependency references unknown node: {source!r}->{destination!r}")
        edges.append({
            "src": source,
            "dst": destination,
            "distance": int(edge.get("iteration_distance", 0)),
            "type": edge.get("edge_type", "data"),
            "source_latency": edge.get("source_latency"),
            "required_time_difference": edge.get("required_time_difference"),
        })
    return {
        "format": "flowadvantage_morpher_dfg_v1",
        "ii": payload.get("ii"),
        "dfg_hash": payload.get("dfg_hash"),
        "nodes": nodes,
        "edges": edges,
        "count": len(nodes),
        "unsupported_compute_nodes": sum(not node["operation"]["supported"] for node in nodes.values()),
    }
