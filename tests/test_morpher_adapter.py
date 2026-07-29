import json, tempfile
from pathlib import Path
from flowadvantage.morpher_adapter.dfg_import import import_dfg_xml
from flowadvantage.morpher_adapter.mrrg_import import import_architecture_json
from flowadvantage.morpher_adapter.operation_taxonomy import translate_operation
from flowadvantage.morpher_adapter.mapping_export import export_mapping, export_flowadvantage_mapping
from flowadvantage.morpher_adapter.legality_bridge import validate_imported_mapping, validate_mapping

def _fixture(tmp_path):
    p=tmp_path/'x.xml'; p.write_text('<Root><DFG count="2"><Node idx="1" ASAP="0" ALAP="1"><OP>ADD</OP><Inputs/><Outputs><Output idx="2" nextiter="1" type="I1"/></Outputs><RecParents/></Node><Node idx="2" ASAP="1" ALAP="2"><OP>MUL</OP><Inputs><Input idx="1"/></Inputs><Outputs/><RecParents><RecParent idx="1"/></RecParents></Node></DFG></Root>')
    a=tmp_path/'a.json'; a.write_text(json.dumps({'ARCH':{'FU':{'OPS':{'ADD':1,'MUL':1}}}})); return p,a

def test_node_count(tmp_path): assert import_dfg_xml(_fixture(tmp_path)[0])['count']==2
def test_edge_count(tmp_path): assert len(import_dfg_xml(_fixture(tmp_path)[0])['edges'])==1
def test_direction(tmp_path): assert import_dfg_xml(_fixture(tmp_path)[0])['edges'][0]['src']==1
def test_distance(tmp_path): assert import_dfg_xml(_fixture(tmp_path)[0])['edges'][0]['distance']==1
def test_latency_window(tmp_path): assert import_dfg_xml(_fixture(tmp_path)[0])['nodes'][2]['alap']==2
def test_rec_parent(tmp_path): assert import_dfg_xml(_fixture(tmp_path)[0])['nodes'][2]['rec_parents']==[1]
def test_histogram(tmp_path): assert [n['op'] for n in import_dfg_xml(_fixture(tmp_path)[0])['nodes'].values()]==['ADD','MUL']
def test_cap_add(): assert 'integer_alu' in translate_operation('ADD')['capabilities']
def test_cap_mul(): assert 'multiply' in translate_operation('MUL')['capabilities']
def test_unknown_explicit(): assert translate_operation('STATEFUL')['supported'] is False
def test_load_cap(): assert 'load' in translate_operation('OLOAD')['capabilities']
def test_store_cap(): assert 'store' in translate_operation('OSTORE')['capabilities']
def test_arch_ops(tmp_path): assert set(import_architecture_json(_fixture(tmp_path)[1])['supported_ops'])=={'ADD','MUL'}
def test_arch_hash(tmp_path): assert len(import_architecture_json(_fixture(tmp_path)[1])['architecture_hash'])==64
def test_export_shape(): assert export_mapping({'assignments':{'1':{'pe':0}}})['format']=='morpher_mapping_v1'
def test_export_roundtrip(tmp_path):
    p=tmp_path/'m.json'; export_mapping({'assignments':{'1':{'pe':0}}},p); assert json.loads(p.read_text())['assignments']['1']['pe']==0
def test_legality_known(tmp_path):
    d=import_dfg_xml(_fixture(tmp_path)[0]); assert validate_imported_mapping({'assignments':{'1':{}}},d,{})['legal']
def test_legality_unknown(tmp_path):
    d=import_dfg_xml(_fixture(tmp_path)[0]); assert not validate_imported_mapping({'assignments':{'9':{}}},d,{})['legal']
def test_inputs(tmp_path): assert import_dfg_xml(_fixture(tmp_path)[0])['nodes'][2]['inputs']==[1]
def test_bb_default(tmp_path): assert import_dfg_xml(_fixture(tmp_path)[0])['nodes'][1]['bb']==''
def test_unsupported_count(tmp_path): assert import_dfg_xml(_fixture(tmp_path)[0])['unsupported_compute_nodes']==0

def test_canonical_mapping_export_shape():
    out=export_flowadvantage_mapping({'architecture_hash':'a','dfg_hash':'d','ii':2,'operations':[],'routes':[],'port_state':[]})
    assert out['schema']=='flowadvantage_morpher_mapping_v1' and out['memory_bindings']==[]

def test_canonical_mapping_export_requires_routes():
    try: export_flowadvantage_mapping({'architecture_hash':'a','dfg_hash':'d','ii':2,'operations':[]})
    except ValueError: return
    assert False


def _native_contract_fixture(*, operand_mux=False):
    nodes = [
        {"dfg_node_id": 7, "native_node_key": "7|ADD|0", "opcode": "ADD", "asap": 0, "alap": 1, "bb": "left"},
        {"dfg_node_id": 7, "native_node_key": "7|MUL|1", "opcode": "MUL", "asap": 1, "alap": 2, "bb": "right"},
    ]
    dfg = {
        "schema": "flowadvantage_morpher_dfg_v1",
        "ii": 2,
        "dfg_hash": "dfg",
        "nodes": nodes,
        "dependencies": [
            {
                "source_node": 7,
                "destination_node": 7,
                "source_node_key": "7|ADD|0",
                "destination_node_key": "7|MUL|1",
                "edge_type": "I1",
                "requires_route": True,
            }
        ],
        "mutex_basic_blocks": [],
    }
    resources = [
        {"native_resource_id": "C.PE0", "resource_type": "PE", "time_slot": 0, "capacity": 1},
        {"native_resource_id": "C.PE1", "resource_type": "PE", "time_slot": 1, "capacity": 1},
        {"native_resource_id": "C.PE0.FU", "resource_type": "FU", "time_slot": 0, "capacity": 1, "supported_operations": ["ADD"]},
        {"native_resource_id": "C.PE1.FU", "resource_type": "FU", "time_slot": 1, "capacity": 1, "supported_operations": ["MUL"]},
        {"native_resource_id": "C.PE0.T", "resource_type": "port", "time_slot": 0, "capacity": 1, "allows_operand_mux": False},
        {"native_resource_id": "C.PE1.I1", "resource_type": "port", "time_slot": 1, "capacity": 1, "allows_operand_mux": operand_mux},
    ]
    mrrg = {
        "schema": "flowadvantage_morpher_mrrg_v1",
        "ii": 2,
        "architecture_hash": "arch",
        "resources": resources,
        "edges": [{"src": "C.PE0.T", "dst": "C.PE1.I1", "capacity": 1}],
    }
    mapping = {
        "schema": "flowadvantage_morpher_mapping_v1",
        "ii": 2,
        "architecture_hash": "arch",
        "dfg_hash": "dfg",
        "operations": [
            {"dfg_node_id": 7, "native_node_key": "7|ADD|0", "opcode": "ADD", "pe_id": "C.PE0", "fu_id": "C.PE0.FU", "modulo_time": 0, "latency": 0},
            {"dfg_node_id": 7, "native_node_key": "7|MUL|1", "opcode": "MUL", "pe_id": "C.PE1", "fu_id": "C.PE1.FU", "modulo_time": 1, "latency": 1},
        ],
        "routes": [
            {
                "edge_id": "edge",
                "source_node": 7,
                "destination_node": 7,
                "source_node_key": "7|ADD|0",
                "destination_node_key": "7|MUL|1",
                "ordered_resource_ids": ["C.PE0.T", "C.PE1.I1"],
                "ordered_resource_latencies": [0, 1],
                "start_time": 0,
                "end_time": 1,
                "export_status": "ok",
            }
        ],
    }
    return mapping, dfg, mrrg


def test_native_legality_accepts_stable_duplicate_numeric_ids():
    mapping, dfg, mrrg = _native_contract_fixture()
    result = validate_mapping(mapping, dfg, mrrg)
    assert result["legal"], result["violations"]


def test_native_legality_does_not_require_ps_route():
    mapping, dfg, mrrg = _native_contract_fixture()
    dfg["dependencies"][0]["edge_type"] = "PS"
    dfg["dependencies"][0]["requires_route"] = False
    mapping["routes"] = []
    result = validate_mapping(mapping, dfg, mrrg)
    assert result["legal"], result["violations"]
    assert result["counts"]["pseudo_dependencies"] == 1


def test_native_legality_rejects_missing_physical_route():
    mapping, dfg, mrrg = _native_contract_fixture()
    mapping["routes"] = []
    result = validate_mapping(mapping, dfg, mrrg)
    assert not result["legal"]
    assert any(v["class"] == "missing route" for v in result["violations"])


def test_native_legality_rejects_disconnected_route():
    mapping, dfg, mrrg = _native_contract_fixture()
    mrrg["edges"] = []
    result = validate_mapping(mapping, dfg, mrrg)
    assert not result["legal"]
    assert any(v["class"] == "non-contiguous route" for v in result["violations"])


def test_native_legality_rejects_route_phase_violation():
    mapping, dfg, mrrg = _native_contract_fixture()
    mapping["routes"][0]["ordered_resource_latencies"] = [0, 2]
    mapping["routes"][0]["end_time"] = 2
    result = validate_mapping(mapping, dfg, mrrg)
    assert not result["legal"]
    assert any(v["class"] == "route timing violation" for v in result["violations"])


def test_native_legality_rejects_hash_mismatch():
    mapping, dfg, mrrg = _native_contract_fixture()
    mapping["dfg_hash"] = "wrong"
    result = validate_mapping(mapping, dfg, mrrg)
    assert not result["legal"]
    assert any(v["class"] == "DFG-hash mismatch" for v in result["violations"])


def test_native_legality_rejects_recurrence_route_overrun():
    mapping, dfg, mrrg = _native_contract_fixture()
    dfg["nodes"][1]["recurrence_parent_keys"] = ["7|ADD|0"]
    mapping["routes"][0]["ordered_resource_latencies"] = [0, 3]
    mapping["routes"][0]["end_time"] = 3
    result = validate_mapping(mapping, dfg, mrrg)
    assert not result["legal"]
    assert any(v["class"] == "recurrence violation" for v in result["violations"])


def test_native_legality_rejects_loop_carried_edge_overrun():
    mapping, dfg, mrrg = _native_contract_fixture()
    dfg["dependencies"][0]["iteration_distance"] = 1
    mapping["routes"][0]["ordered_resource_latencies"] = [4, 5]
    mapping["routes"][0]["start_time"] = 4
    mapping["routes"][0]["end_time"] = 5
    result = validate_mapping(mapping, dfg, mrrg)
    assert not result["legal"]
    assert any(v["class"] == "recurrence violation" for v in result["violations"])


def test_native_legality_rejects_memory_operation_on_compute_fu():
    mapping, dfg, mrrg = _native_contract_fixture()
    dfg["nodes"][0]["opcode"] = "LOAD"
    mapping["operations"][0]["opcode"] = "LOAD"
    mrrg["resources"][2]["supported_operations"].append("LOAD")
    mrrg["resources"][2]["memory_role"] = "compute"
    result = validate_mapping(mapping, dfg, mrrg)
    assert not result["legal"]
    assert any(v["class"] == "memory-port violation" for v in result["violations"])
