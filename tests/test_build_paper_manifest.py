import json
from pathlib import Path

from scripts.build_paper_work_manifest import build


def test_manifest_uses_only_real_native_contracts(tmp_path):
    class Args:
        output = tmp_path / "out"
        reference_root = Path("results/revision_v5b/reference_mappings")
        checkpoint = Path("results/revision_v3/checkpoints/residual_gnn_seed_23.pt")
        device = "cpu"
        launch = "none"

    jobs_path, metadata_path = build(Args)
    payload = json.loads(jobs_path.read_text())
    metadata = json.loads(metadata_path.read_text())
    assert payload["jobs"]
    assert all(row["dfg_hash"] and row["architecture_hash"] and row["reference_mapping_hash"] for row in payload["jobs"])
    assert all(Path(row["dfg_path"]).is_file() for row in payload["jobs"])
    assert all(row["evaluation_mode"] == "deterministic_length_prefix_then_complete" for row in payload["jobs"])
    assert all(row["initialization_policy"] == "deterministic_length_prefix" for row in payload["jobs"])
    assert all(row["full_end_to_end"] is True for row in payload["jobs"])
    assert metadata["blockers"]  # current corpus is intentionally incomplete
    assert "native_pathfinder" in metadata["unsupported_methods"]


def test_manifest_refuses_nonempty_output(tmp_path):
    class Args:
        output = tmp_path / "out"
        reference_root = Path("results/revision_v5b/reference_mappings")
        checkpoint = Path("results/revision_v3/checkpoints/residual_gnn_seed_23.pt")
        device = "cpu"
        launch = "none"

    build(Args)
    (Args.output / "sentinel").write_text("preserve")
    try:
        build(Args)
    except SystemExit as error:
        assert "refusing to overwrite" in str(error)
    else:
        raise AssertionError("manifest generator overwrote immutable output")
