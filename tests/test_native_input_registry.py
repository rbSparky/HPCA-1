import json
from pathlib import Path
from scripts.build_native_input_registry import build

def test_registry_requires_manifest_and_canonical_contract(tmp_path):
    pair = tmp_path / "k" / "a"; pair.mkdir(parents=True)
    (pair / "manifest.env").write_text("\n".join([
        "TOOLCHAIN=/tool", "BINARY_SHA256=" + "a"*64,
        "DFG_REL=x.xml", "DFG_SHA256=" + "b"*64,
        "ARCH_REL=json_arch/a.json", "ARCH_SHA256=" + "c"*64,
        "X=4", "Y=4", "INITIAL_II=0", "PE_TYPE=STDNOC_4REG", "METHOD=0", ""]))
    for name, schema in (("dfg", "dfg"), ("mrrg", "mrrg"), ("mapping", "mapping")):
        (pair / f"{name}.json").write_text(json.dumps({"schema": schema, "ii": 4}))
    out = tmp_path / "registry.json"; payload = build(tmp_path, out)
    assert payload["schema"] == "flowadvantage_native_input_registry_v1"
    assert len(payload["rows"]) == 1
    assert payload["native_executable_rows"] == 0

def test_registry_does_not_index_partial_pair(tmp_path):
    pair = tmp_path / "k" / "a"; pair.mkdir(parents=True)
    (pair / "manifest.env").write_text("TOOLCHAIN=/tool\n")
    payload = build(tmp_path, tmp_path / "r.json")
    assert payload["rows"] == []


def test_registry_indexes_memory_variant_with_hash_addressed_staged_inputs(tmp_path):
    pair = tmp_path / "k" / "a"; pair.mkdir(parents=True)
    staged = tmp_path / "staged"; staged.mkdir()
    dfg = staged / "dfg.xml"; dfg.write_text("<dfg/>")
    arch = staged / "architecture_exact_layout.json"; arch.write_text(json.dumps({"x": 4}))
    import hashlib
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    (pair / "manifest.env").write_text("\n".join([
        "TOOLCHAIN=/remote/tool", "BINARY_SHA256=" + "a"*64,
        "DFG=" + str(dfg), "DFG_SHA256=" + digest(dfg),
        "ARCH_SHA256=" + digest(arch),
        "X=4", "Y=4", "INITIAL_II=0", "PE_TYPE=HY_CUBE", "METHOD=0", ""]))
    for name, schema in (("dfg", "dfg"), ("mrrg", "mrrg"), ("mapping", "mapping")):
        (pair / f"{name}.json").write_text(json.dumps({"schema": schema, "ii": 4}))
    payload = build(tmp_path, tmp_path / "r.json", staged)
    assert payload["staged_input_rows"] == 1
    assert payload["native_executable_rows"] == 0
    assert payload["rows"][0]["staged_inputs_available"] is True
    assert payload["rows"][0]["native_available"] is False
