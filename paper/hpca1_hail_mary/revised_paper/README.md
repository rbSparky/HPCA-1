# QuotientFlow HPCA review-driven revision

## Build

```bash
make
```

Equivalent command:

```bash
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

## Revision status

This non-overwriting hail-mary working copy derives from `revised_paper-hpca1.zip`. It removes visible figure draft boxes, adds two conceptual vector schematics, narrows unsupported empirical claims, restricts symmetry quotienting to the complete ordered transition contract, expands related work, and retains the author-approved HPCA AI-use appendix verbatim. The author supplied HotCRP submission number 4950; the reproducible preflight verifies the rendered banner and absence of unresolved draft values. New experiments identified in `AUTHOR_ACTIONS.md` remain required before empirical claims can be strengthened.

Run the repository-level reproducible build and preflight:

```bash
bash scripts/build_hpca1_paper.sh
```
