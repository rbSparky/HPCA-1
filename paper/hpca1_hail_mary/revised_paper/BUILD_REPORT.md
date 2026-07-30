# Build Report — HPCA submission 4950

- **Build command:** `bash scripts/build_hpca1_paper.sh`
- **Root TeX file:** `paper/hpca1_hail_mary/revised_paper/main.tex`
- **Output PDF:** `paper/hpca1_hail_mary/revised_paper/main.pdf`
- **Compiler:** pdfTeX 3.141592653-2.6-1.40.25; LaTeX2e 2023-11-01 patch 1
- **Reproducibility environment:** `LC_ALL=C`, `TZ=UTC`, `SOURCE_DATE_EPOCH=1785456000`
- **Total page count:** 10
- **Page size:** US Letter, 612 x 792 points
- **TeX SHA-256:** `3dee88020d84bc1b5f2c31ecfc15fb31647548cfd6702fd09266965636f7ee9d`
- **PDF SHA-256:** `a680647a2f91252ef506968e6f6ab7f42b690c702d6ddd881d245e7f0aff7ce5`
- **Preflight JSON SHA-256:** `01710adfcf11fe1335ee32f80fadf6ab691f17590082bdf09f60848f79b29939`

## Build diagnostics

- Three `pdflatex -interaction=nonstopmode -halt-on-error` passes completed.
- Two underfull vertical-box diagnostics remain; no content is clipped.
- No overfull boxes.
- No undefined references or citations on the final pass.
- No missing files or compilation errors.

## PDF and format checks

- Submission number 4950 appears in source and rendered PDF.
- Zero rendered `\draftvalue{...}` uses remain.
- PDF opens, is unencrypted, and passes `qpdf --check`.
- All 25 reported font rows are embedded Type 1 fonts.
- No visible `AUTHOR INPUT`, `placeholder`, `TBD`, `TODO`, `dummy value`, `fake value`, or `#XXX` text.
- `Title`, `Subject`, `Keywords`, and `Author` metadata are blank.
- No author repository, user path, or `rbSparky` identifier was found in rendered text.
- The AI-use appendix SHA-256 is `38e42b03ae9cfce721c053f20c17c2cb14d564acae857372e2e5e73c79703d5a`, identical to the supplied source archive.

## Final preflight status

**PAPER FORMAT PASS; EMPIRICAL EVIDENCE PENDING.** Submission-number, PDF-format, font, draft-value, disclosure, and available anonymity checks pass. The zero-prefix paired matrix and the remaining reviewer-mandated evidence tracked in `results/hail_mary_hpca1/HAIL_MARY.md` must complete before empirical claims are strengthened.
