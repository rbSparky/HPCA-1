# HPCA-1 paper staging report

## Outcome

The supplied `revised_paper-hpca1.zip` was path-validated and extracted into the new, non-overwriting directory `paper/hpca1_hail_mary/revised_paper/`. Earlier paper and result artifacts were not modified.

Submission number **4950** replaced the sole rendered draft value. The author-approved AI-use appendix remained exactly unchanged: both the source archive and working copy hash to `38e42b03ae9cfce721c053f20c17c2cb14d564acae857372e2e5e73c79703d5a` for the appendix section.

## Commands executed

```bash
python3 scripts/update_hail_mary_tracker.py
bash scripts/build_hpca1_paper.sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  /home/rishabh/miniconda/envs/taugat_pyg/bin/python -m pytest -q
```

The preflight was run twice. Both builds produced PDF SHA-256:

```text
a680647a2f91252ef506968e6f6ab7f42b690c702d6ddd881d245e7f0aff7ce5
```

## Verified checks

- Three-pass `pdflatex` build: PASS.
- Submission number present in source and rendered PDF: PASS.
- Rendered `\draftvalue` uses: 0.
- Approved AI disclosure unchanged: PASS.
- PDF integrity (`qpdf --check`): PASS.
- Page count/size: 10 pages, US Letter.
- Font embedding: 25/25 rows embedded, Type 1.
- Undefined references/citations: 0 on final pass.
- Overfull boxes: 0.
- PDF author metadata: blank.
- Visible repository/user-path anonymity scan: PASS.
- Prohibited draft-text scan: PASS.
- Rendered page-1 inspection: PASS; the banner visibly reads `HPCA 2027 Submission #4950` without clipping or draft markup.
- Full repository tests: **156 passed in 84.04 seconds**.
- Negative preflight mutation (wrong submission number): correctly rejected.
- Negative tracker mutation (weights not summing to 100): correctly rejected.

## Remaining evidence

Paper formatting and author-controlled compliance are complete. Empirical claims must remain narrow until the remaining actions in `reviewer_actions.csv` are marked `VERIFIED` from immutable indexed results. The staged manuscript intentionally retains the earlier conservative evidence language.
