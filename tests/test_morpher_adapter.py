import json, tempfile
from pathlib import Path
from flowadvantage.morpher_adapter.dfg_import import import_dfg_xml
from flowadvantage.morpher_adapter.mrrg_import import import_architecture_json
from flowadvantage.morpher_adapter.operation_taxonomy import translate_operation
from flowadvantage.morpher_adapter.mapping_export import export_mapping, export_flowadvantage_mapping
from flowadvantage.morpher_adapter.legality_bridge import validate_imported_mapping

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
    out=export_flowadvantage_mapping({'architecture_hash':'a','dfg_hash':'d','ii':2,'operations':[],'routes':[]})
    assert out['schema']=='flowadvantage_morpher_mapping_v1' and out['memory_bindings']==[]

def test_canonical_mapping_export_requires_routes():
    try: export_flowadvantage_mapping({'architecture_hash':'a','dfg_hash':'d','ii':2,'operations':[]})
    except ValueError: return
    assert False
