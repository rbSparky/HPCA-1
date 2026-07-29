from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results/revision_v5b/morpher_patch/Morpher_CGRA_Mapper"


def test_sa_exposes_explicit_seed_and_preserves_unseeded_mode():
    header = (SRC / "include/morpher/mapper/SimulatedAnnealingMapper.h").read_text()
    compiler = (SRC / "src/CGRA_xml_compiler.cpp").read_text()
    util = (SRC / "include/morpher/util/util.h").read_text()
    assert "setRandomSeed" in header
    assert "--seed" in util
    assert "has_random_seed" in compiler
    assert "SA deterministic seed" in compiler
    assert "if (seeded)" in header


def test_all_simulated_annealing_random_sources_are_seed_gated():
    source = (SRC / "src/mapper/SimulatedAnnealingMapper.cpp").read_text()
    assert "random_device" in source  # native behavior when no seed supplied
    assert "if (seeded)" in source
    assert "shuffle(candidateDests.begin(), candidateDests.end(), rng)" in source
    assert "std::default_random_engine(seed)" in source  # unseeded compatibility
