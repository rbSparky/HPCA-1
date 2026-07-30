# Revision Report

## Executive summary

This revision performs the strongest defensible source-only repair available from the supplied paper archive and review. It does not invent measurements or claim acceptance-level evidence. The manuscript now compiles cleanly, contains no visible figure draft boxes, includes two legible conceptual vector figures, adds the HPCA AI-use appendix, and passes PDF opening/font/metadata checks. The actual HotCRP submission number remains an explicit red author action.

The technical revision is material. It reclassifies the existing 40% experiment as a suffix-completion probe, weakens dual claims to local solver-dependent approximations, states that strong primal convexity does not imply unique multipliers, conditions symmetry reduction on the complete total operation order and score/transition equivariance, and distinguishes exhaustive exactness from finite-beam dominance reduction. It also adds the closest CGRA and learning-to-strong-branch comparisons.

The review's stated ceiling without new experiments is score 3. This revision reaches that ceiling but cannot support the target score 6 because the core paired evaluation, ranking/dual validation, runtime Pareto evidence, quotient equivalence tests, and closest-baseline experiment remain unavailable.

## Score-changing revisions

1. **Correctness specification:** Replaced graph-only symmetry claims with an order-safe weighted-bisimulation contract. Non-rank-preserving DFG automorphisms are excluded, potentially leaving only the identity.
2. **Causal validity:** The current 40% length-prefix result is now explicitly a suffix experiment; the frozen headline protocol requires zero prefix.
3. **Mathematical validity:** Capacity multipliers are described as local approximations. The paper no longer claims that strong primal convexity guarantees stable or unique dual rankings.
4. **Claim-evidence alignment:** Abstract, contributions, discussion, limitations, and conclusion now state exactly what the 148 tests and two-cell probe support.
5. **Related work:** Added the ICCAD 2022 GNN/RL mapper, Rewire (DAC 2025), Gasse et al. strong-branching imitation, and exact symmetry-reduction literature.
6. **Compliance and presentation:** Replaced two conceptual figure boxes with actual grayscale-safe TikZ schematics; removed the two unsupported results/ablation boxes rather than fabricating evidence; added AI-use disclosure; retained anonymous metadata and embedded Type 1 fonts.

## Issue-by-issue response

### R1 - Paired end-to-end evidence: PARTIAL

Claims were narrowed and the required paired protocol was made explicit, including success-under-budget and zero-prefix headline runs. No new mapping matrix was available, so the evidence blocker remains.

### R2 - Runtime and timeout behavior: PARTIAL

The manuscript now frames the relaxation path as the dominant measured obstacle and specifies matrix reuse, warm starts, factorization reuse, batching, embedding reuse, and stage-level profiling. No implementation repository or new runtime data was supplied.

### R3 - 40% length prefix: PARTIAL

The probe is now consistently called suffix completion. The headline protocol sets prefix depth to zero and reserves 0/20/40% for sensitivity. Existing results cannot be converted into zero-prefix evidence.

### R4 - Learning reproducibility: PARTIAL

A complete disclosure and ablation contract was added. The source package contains no training logs, configurations, checkpoints, or split manifests from which the missing values could be recovered.

### R5 - Dual usefulness and stability: PARTIAL

Categorical scarcity language and the unsupported dual-stability claim were removed. The paper now states multiplier nonuniqueness and requires knockout, solver, tau, and tolerance studies. No such data were supplied.

### R6 - Symmetry correctness: PARTIAL

The manuscript-level correctness ambiguity was repaired. Exactness is conditioned on total-order preservation, legal-action bijection, successor and score equivariance, and terminal invariance. Finite beam search is described as a dominance reduction rather than guaranteed identical output. Exhaustive implementation tests are still required.

### R7 - Related work and closest baseline: PARTIAL

Verified references and precise positioning were added. A common-contract baseline run remains absent.

### R8 - Submission readiness: PARTIAL

All visible figure draft boxes and `#XXX` text were removed. The paper now has two completed conceptual figures, no visible prohibited draft terms, blank author metadata, and an AI-use appendix. The actual submission number was not provided and remains a red draft value, so the PDF is not submission-ready.

## Files modified

- `main.tex`: technical, evaluation, related-work, figure, compliance, and disclosure revisions.
- `refs.bib`: added verified metadata for the closest learned CGRA, Rewire, learning-to-branch, and symmetry references.
- `README.md`: updated build and revision status.
- `main.pdf`: rebuilt manuscript.

## Claims narrowed, removed, or strengthened

- **Narrowed:** dual variables “quantify scarcity” -> capacity multipliers provide a local solver-dependent approximation.
- **Removed:** strong convexity “stabilizes the extracted dual ranking” as a guaranteed statement.
- **Narrowed:** current experiment “empty-state-to-completion” -> 40% length-prefix suffix completion.
- **Narrowed:** exact finite-beam quotienting preserves identical output -> exhaustive weighted bisimulation plus finite-beam dominance reduction under equivariance.
- **Removed:** unsupported implied headline and ablation figures.
- **Strengthened:** novelty statement now isolates residual-dual decomposition, exact-child labels, top-k correction, and native atomic action contract.

## Experiments and analyses still required

See `AUTHOR_ACTIONS.md`. The critical path is: actual submission number; zero-prefix paired end-to-end matrix; runtime optimization and Pareto curves; learning reproducibility and ablations; dual knockout/stability study; exhaustive quotient equivalence; and a closest learned/routing-aware baseline.

## Compliance checks

- Build: PASS.
- Total PDF pages: 10.
- Content pages before references: 9 (limit: 11).
- Paper size: US Letter.
- Fonts: embedded Type 1.
- Undefined references/citations: none.
- Overfull boxes: none.
- Visible prohibited draft terms: none.
- Author metadata: blank.
- Anonymity: PASS on available source/PDF checks.
- AI-use appendix: present.
- Actual submission number: unresolved author action.
- Red draft values: 1.

## Predicted score

- **Target:** 6/7.
- **Verified predicted score:** 3/7 (weak reject).
- **Rationale:** The source-only revision repairs correctness wording, causal interpretation, novelty positioning, and visual draft defects, but the primary contribution still lacks decision-relevant end-to-end evidence. The review explicitly set the no-new-experiment ceiling at 3.
