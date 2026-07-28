# QuotientFlow validation prototype

This repository implements the deterministic one-hour QuotientFlow falsification
suite specified in `AGENTS.md`.

The local official run reuses the existing `taugat_pyg` Python 3.11 environment
to avoid duplicating several gigabytes of CUDA packages:

```bash
conda activate taugat_pyg
python scripts/doctor.py
python -m pytest -q
python scripts/run_quick_suite.py --config configs/quick.yaml
```

The suite is resumable. Generated instances and oracle targets are cached under
`results/cache`, while all run outputs are written under `results/quick_v1`.
See `results/quick_v1/REPORT.md` and `RUN_SUMMARY.txt` after execution.

The immutable baseline missed the primary mapper and quotient gates. The
validation-only, failure-driven revision prescribed by `AGENTS.md` is recorded
separately under `results/revision_v1`; its frozen outcome is **PROCEED WITH
REFINEMENT**. See `results/revision_v1/REPORT.md` and
`results/revision_v1/RUN_SUMMARY.txt`. Reproduce the selected follow-up with:

```bash
python scripts/run_revision2_scarcity_suite.py
python scripts/run_symmetry_revision.py
python scripts/make_revision_report.py
python scripts/audit_artifacts.py
```

The action-conditioned relaxed-lookahead follow-up is preserved separately
under `results/revision_v2`; it does not overwrite either earlier campaign.
Activate and inspect/reproduce it with:

```bash
source /home/rishabh/miniconda/etc/profile.d/conda.sh
conda activate taugat_pyg
python scripts/generate_action_lookahead_labels.py
python scripts/run_action_models_v2.py
python scripts/run_action_mapper_v2.py
python scripts/resume_oracle_mapper_v2.py
python scripts/retry_oracle_timeouts_v2.py
python scripts/run_action_orbits_v2.py
python scripts/make_revision_v2_report.py
PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q
python scripts/audit_revision_v2.py
```

The deterministic solve caches make reruns resumable. The compact handoff zip
intentionally excludes `results/revision_v2/cache`; copy the cache directory
separately only when warm-start continuity is needed.
