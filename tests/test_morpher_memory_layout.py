import csv
import importlib.util
import json
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "derive_morpher_mem_alloc.py"
SPEC = importlib.util.spec_from_file_location("derive_morpher_mem_alloc", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

TRACE_MODULE_PATH = Path(__file__).parents[1] / "scripts" / "translate_morpher_trace_names.py"
TRACE_SPEC = importlib.util.spec_from_file_location(
    "translate_morpher_trace_names", TRACE_MODULE_PATH
)
assert TRACE_SPEC is not None and TRACE_SPEC.loader is not None
TRACE_MODULE = importlib.util.module_from_spec(TRACE_SPEC)
TRACE_SPEC.loader.exec_module(TRACE_MODULE)

SIM_ALLOC_MODULE_PATH = (
    Path(__file__).parents[1] / "scripts" / "make_morpher_sim_mem_alloc.py"
)
SIM_ALLOC_SPEC = importlib.util.spec_from_file_location(
    "make_morpher_sim_mem_alloc", SIM_ALLOC_MODULE_PATH
)
assert SIM_ALLOC_SPEC is not None and SIM_ALLOC_SPEC.loader is not None
SIM_ALLOC_MODULE = importlib.util.module_from_spec(SIM_ALLOC_SPEC)
SIM_ALLOC_SPEC.loader.exec_module(SIM_ALLOC_MODULE)

CYCLE_MODULE_PATH = (
    Path(__file__).parents[1] / "scripts" / "run_morpher_cycle_validation.py"
)
CYCLE_SPEC = importlib.util.spec_from_file_location(
    "run_morpher_cycle_validation", CYCLE_MODULE_PATH
)
assert CYCLE_SPEC is not None and CYCLE_SPEC.loader is not None
CYCLE_MODULE = importlib.util.module_from_spec(CYCLE_SPEC)
CYCLE_SPEC.loader.exec_module(CYCLE_MODULE)


def _architecture(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "ARCH": {
                    "SPM_B0_WRAPPER": {"DATA_LAYOUT": {"A": 0, "alpha": 128}},
                    "SPM_B1_WRAPPER": {"DATA_LAYOUT": {"B": 64}},
                }
            }
        ),
        encoding="utf-8",
    )


def test_mem_alloc_derivation_preserves_bank_and_local_address(tmp_path):
    architecture = tmp_path / "source.json"
    allocation = tmp_path / "allocation.csv"
    provenance = tmp_path / "provenance.json"
    _architecture(architecture)

    MODULE.derive(
        architecture,
        bank_size=2048,
        output_csv=allocation,
        provenance_json=provenance,
        required_variables=["A", "B", "alpha"],
    )

    with allocation.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows == [
        {"var_name": "A", "base_addr": "0"},
        {"var_name": "B", "base_addr": "2112"},
        {"var_name": "alpha", "base_addr": "128"},
    ]
    record_by_name = {
        record["variable"]: record
        for record in json.loads(provenance.read_text(encoding="utf-8"))["records"]
    }
    assert record_by_name["B"]["source_bank"] == 1
    assert record_by_name["B"]["source_local_address"] == 64
    assert record_by_name["B"]["derived_flat_address"] == 2112


def test_mem_alloc_derivation_rejects_missing_required_variable(tmp_path):
    architecture = tmp_path / "source.json"
    _architecture(architecture)
    with pytest.raises(ValueError, match="lacks required DFG variables: missing"):
        MODULE.derive(
            architecture,
            bank_size=2048,
            output_csv=tmp_path / "allocation.csv",
            provenance_json=tmp_path / "provenance.json",
            required_variables=["missing"],
        )


def test_mem_alloc_derivation_rejects_duplicate_variable_across_banks(tmp_path):
    architecture = tmp_path / "source.json"
    architecture.write_text(
        json.dumps(
            {
                "ARCH": {
                    "SPM_B0_WRAPPER": {"DATA_LAYOUT": {"A": 0}},
                    "SPM_B1_WRAPPER": {"DATA_LAYOUT": {"A": 0}},
                }
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="appears in multiple banks"):
        MODULE.derive(
            architecture,
            bank_size=2048,
            output_csv=tmp_path / "allocation.csv",
            provenance_json=tmp_path / "provenance.json",
            required_variables=["A"],
        )


def test_trace_translation_changes_only_explicit_names(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "trace_0.txt").write_text(
        "var_name,offset,pre-run-data,post-run-data\n"
        "old,0,12,34\n"
        "stable,1,-5,8\n",
        encoding="utf-8",
    )
    name_map = tmp_path / "names.json"
    name_map.write_text('{"old": "new"}\n', encoding="utf-8")
    allocation = tmp_path / "allocation.csv"
    allocation.write_text(
        "var_name,base_addr\nnew,0\nstable,4\n", encoding="utf-8"
    )
    output = tmp_path / "translated"
    provenance = tmp_path / "translation.json"

    TRACE_MODULE.translate_corpus(
        source, output, name_map, allocation, provenance
    )

    assert (output / "trace_0.txt").read_text(encoding="utf-8") == (
        "var_name,offset,pre-run-data,post-run-data\n"
        "new,0,12,34\n"
        "stable,1,-5,8\n"
    )
    manifest = json.loads(provenance.read_text(encoding="utf-8"))
    assert manifest["numeric_value_transformation"] == "none"
    assert manifest["offset_transformation"] == "none"


def test_trace_translation_rejects_unallocated_target_name(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "trace_0.txt").write_text(
        "var_name,offset,pre-run-data,post-run-data\nold,0,12,34\n",
        encoding="utf-8",
    )
    name_map = tmp_path / "names.json"
    name_map.write_text('{"old": "missing"}\n', encoding="utf-8")
    allocation = tmp_path / "allocation.csv"
    allocation.write_text("var_name,base_addr\nA,0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="does not occur in the target allocation"):
        TRACE_MODULE.translate_corpus(
            source,
            tmp_path / "translated",
            name_map,
            allocation,
            tmp_path / "translation.json",
        )


def test_simulator_allocation_adds_source_defined_control_cells(tmp_path):
    source = tmp_path / "data.csv"
    source.write_text("var_name,base_addr\nA,0\nB,2176\n", encoding="utf-8")
    allocation = SIM_ALLOC_MODULE.build_simulator_allocation(
        input_csv=source,
        output_csv=tmp_path / "sim.csv",
        provenance_json=tmp_path / "provenance.json",
        total_memory_size=4096,
        source_reference="Morpher_DFG_Generator/src/common/dfg.cpp:11440-11446",
    )
    assert allocation["loopend"] == 2047
    assert allocation["loopstart"] == 4094
    assert (tmp_path / "sim.csv").read_text(encoding="utf-8").splitlines() == [
        "var_name,base_addr",
        "A,0",
        "B,2176",
        "loopend,2047",
        "loopstart,4094",
    ]


def test_simulator_allocation_rejects_reserved_address_collision(tmp_path):
    source = tmp_path / "data.csv"
    source.write_text("var_name,base_addr\nA,2047\n", encoding="utf-8")
    with pytest.raises(ValueError, match="reserved loopend address 2047 collides"):
        SIM_ALLOC_MODULE.build_simulator_allocation(
            input_csv=source,
            output_csv=tmp_path / "sim.csv",
            provenance_json=tmp_path / "provenance.json",
            total_memory_size=4096,
            source_reference="pinned source",
        )


def test_cycle_validation_preserves_atomic_per_trace_results(tmp_path):
    simulator = tmp_path / "simulator"
    simulator.write_text(
        "#!/usr/bin/env python3\n"
        "from pathlib import Path\n"
        "Path('sim_result.txt').write_text('3,0\\n', encoding='utf-8')\n"
        "print('Simulation Result: Matches::3, Mismatches::0')\n",
        encoding="utf-8",
    )
    simulator.chmod(0o755)
    bitstream = tmp_path / "mapping.bin"
    bitstream.write_bytes(b"configuration")
    allocation = tmp_path / "allocation.csv"
    allocation.write_text(
        "var_name,base_addr\nloopend,2047\nloopstart,4094\n",
        encoding="utf-8",
    )
    trace = tmp_path / "trace_0.txt"
    trace.write_text(
        "var_name,offset,pre-run-data,post-run-data\n"
        "loopstart,0,1,0\nloopend,0,0,1\n",
        encoding="utf-8",
    )
    output = tmp_path / "cycle_results"
    summary = CYCLE_MODULE.run_corpus(
        simulator=simulator,
        bitstream=bitstream,
        allocation=allocation,
        traces=[trace],
        output_directory=output,
        x_dimension=4,
        y_dimension=4,
        total_memory_size=4096,
        memory_arrangement=2,
        timeout_seconds=5,
    )
    assert summary["matched_trace_count"] == 1
    assert summary["total_matches"] == 3
    assert summary["total_mismatches"] == 0
    assert (output / "0000_trace_0" / "result.json").is_file()
    with pytest.raises(FileExistsError):
        CYCLE_MODULE.run_corpus(
            simulator=simulator,
            bitstream=bitstream,
            allocation=allocation,
            traces=[trace],
            output_directory=output,
            x_dimension=4,
            y_dimension=4,
            total_memory_size=4096,
            memory_arrangement=2,
            timeout_seconds=5,
        )
