from pathlib import Path


def test_native_worker_contains_seeded_sa_command_path():
    source = Path("scripts/run_v5b_pilot_worker.py").read_text()
    assert "SA_METHODS" in source
    assert '"1" if is_sa' in source
    assert '"--seed", str(int(spec["seed"]))' in source
    assert '"-r"' in source


def test_pathfinder_does_not_require_entropy_seed():
    source = Path("scripts/run_v5b_pilot_worker.py").read_text()
    # Seed is appended only under the SA branch; native PathFinder remains
    # deterministic under its own native defaults without entropy access.
    block = source[source.index("is_sa ="):source.index("progress[\"stage\"]")]
    assert "if is_sa:" in block
    assert "random_device" not in block


def test_native_worker_freezes_the_inclusive_ii_grid():
    source = Path("scripts/run_v5b_pilot_worker.py").read_text()
    block = source[source.index("command = ["):source.index("progress[\"stage\"]")]
    assert '"--max_II"' in block
    # Morpher stops before max_II, so LB..LB+4 uses an exclusive LB+5 bound.
    assert 'int(spec.get("ii_delta_max", 4)) + 1' in block
