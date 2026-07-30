# Author Actions

Only actions requiring new evidence, unavailable implementation/data, or author approval are listed.

## P0 - Insert the actual HPCA submission number — COMPLETED

- **Manuscript location:** `main.tex:24`, title banner on page 1.
- **Why it matters:** HPCA requires the confidentiality banner to contain the actual submission number. The revision intentionally renders `AUTHOR INPUT` in red rather than inventing a number.
- **Procedure:** Replace `\newcommand{\hpcasubmissionnumber}{\draftvalue{AUTHOR INPUT}}` with `\newcommand{\hpcasubmissionnumber}{<HotCRP number>}`. Rebuild with `make` and confirm that `\draftvalue` has zero uses.
- **Expected artifact:** Final anonymous PDF with the correct banner.
- **Completion:** The author supplied submission number 4950. It is present in `main.tex`; the three-pass build and automated preflight verify the rendered banner, anonymous metadata, and zero unresolved `\draftvalue` uses. `DRAFT_VALUE_LEDGER.csv`, `BUILD_REPORT.md`, and `REVIEW_RESPONSE.yaml` were updated.

## P0 - Run the paired end-to-end mapping matrix

- **Manuscript locations:** Abstract; Sections VI-VIII; Tables II-VI.
- **Why it matters:** No current evidence shows improvement in legal success, minimum II, route cost, or quality versus compilation time.
- **Procedure:** Freeze at least six structurally varied kernels on A0-A2 plus a held-out A3 subset. Run length, dual-linear, proposal, top-4, native PathFinder, and simulated annealing under identical end-to-end wall budgets and terminal-status rules. Deterministic headline methods must start at prefix depth 0. Record every failure, timeout, and error rather than filtering them. Repeat deterministic wall-time measurements and use enough annealing seeds to estimate variation.
- **Required output schema:** one row per `{kernel, architecture, II, method, seed, budget}` with `status, legality, success, mapped_ops, minimum_mapped_ii, route_cost, wall_s, peak_rss, expansions, actions, routing_attempts, parent_solves, child_solves, solver_setup_s, solver_iter_s, feature_s, inference_s, canonicalization_s, validator_s, source_hash, config_hash, checkpoint_hash`.
- **Expected artifacts:** frozen manifest, per-cell CSV/JSON, paired analysis table, confidence intervals, quality-versus-time plots, and complete failure census.
- **Update afterward:** Abstract, contribution 4, Section VI, Section VII, conclusion, and all headline result tables/figures.

## P1 - Implement and measure relaxation-path optimizations

- **Manuscript locations:** Sections IV-C, VIII-C, and the strict probe discussion.
- **Why it matters:** Proposal is currently 3.28-3.99x slower than length; dual-linear and top-4 time out.
- **Procedure:** Add parameterized matrix reuse, warm starts, factorization reuse where the solver permits it, batched action scoring, DFG/MRRG embedding reuse, and architecture-statistic caching. Measure solver setup separately from numerical iterations. Run proposal and top-1/2/4/8 to completion on a representative subset and compare with full lookahead where feasible.
- **Expected artifacts:** stage-level profiles, cache statistics, solve counts, and quality-versus-time Pareto curves.
- **Update afterward:** Table VI, Section VII-C/D, Section VIII-C, and abstract runtime statement.

## P1 - Complete zero-prefix and prefix-sensitivity evaluation

- **Manuscript locations:** Section V-B and the strict-probe subsection.
- **Why it matters:** The current 40% length prefix prevents the learned method from controlling the early decisions that motivate the paper.
- **Procedure:** Run deterministic methods at prefix depths 0%, 20%, and 40% with identical beam, route/action caps, and end-to-end budgets. Use 0% for the headline comparison. Conventional baselines must start from their own empty native state.
- **Expected artifacts:** per-cell prefix-sensitivity table and plot, with success-under-budget and conditional route cost.
- **Update afterward:** Initialization protocol, methods table, results, and limitations.

## P1 - Freeze and disclose the learning pipeline

- **Manuscript locations:** Section IV-A/B and Section VI-D.
- **Why it matters:** Model dimensions, data construction, leakage-safe splits, optimization settings, seeds, label rejection, and held-out ranking quality are unavailable.
- **Procedure:** Split at the DFG-instance/trajectory level. Report train/validation/test group and action counts, kernel/architecture membership, model dimensions and parameters, optimizer and schedule, batch construction, epochs, seeds, checkpoint identity, training time, and child-label success/rejection/fallback rates. Evaluate dual-only, scalar-only, GNN-only, no-dual, direct-Q*, and residual-target variants.
- **Required metrics:** top-1 and top-k oracle recall, epsilon-optimal top-1, selected regret, relative regret, Spearman/Kendall rank correlation, calibration where applicable, and end-to-end outcome correlation.
- **Expected artifacts:** frozen split manifest, training config, checkpoints, logs, held-out ranking table, and ablation table.
- **Update afterward:** Section IV, Section VI-D, results, related work, and reproducibility appendix/artifact documentation.

## P1 - Validate dual usefulness and numerical stability

- **Manuscript locations:** Sections II-C, III-A/B, VI-D, and VIII-A.
- **Why it matters:** Strong primal convexity does not ensure unique capacity multipliers, and the native action is a finite capacity removal.
- **Procedure:** On held-out states, compare positive multipliers against finite resource-knockout differences `J(s | b_l=0)-J(s)`. Sweep tau, primary/fallback solver, feasibility/optimality tolerances, and redundant-constraint handling. Compare dual-linear rankings against immediate length and exact child regret.
- **Required metrics:** rank correlation, top-k recall, positive-dual rate, scoreable-state rate, fallback rate, rejected-state rate, solver disagreement, and ranking stability across settings.
- **Expected artifacts:** dual-validation CSV, sensitivity plots, and numerical-health summary.
- **Update afterward:** Abstract, Sections II-C and III-A/B, results, and limitations.

## P0 - Execute symmetry transition-equivalence tests

- **Manuscript locations:** Section IV-E.
- **Why it matters:** The revision fixes the specification by requiring total-order and score equivariance, but no implementation or exhaustive evidence was supplied.
- **Procedure:** For generated small symmetric DFG/MRRG instances, enumerate every reachable state with quotienting disabled and enabled. For each admitted transform and state, verify a bijection between legal atomic actions, identical local scores within numerical tolerance, equivariant successors, invariant terminal legality, and invariant route cost. Include tied schedule/fan-out priorities, recurrence edges, memory restrictions, mutex/broadcast semantics, and temporal architecture transforms. DFG transforms that do not preserve total rank must be rejected.
- **Expected artifacts:** action-bijection log, adversarial test suite, quotient/nonquotient terminal-set comparison, best-cost equality report, and accepted-group census.
- **Update afterward:** Section IV-E, correctness table, quotient results, and `REVIEW_RESPONSE.yaml` R6 status.

## P1 - Add a closest learned/routing-aware baseline

- **Manuscript locations:** Related Work and headline evaluation.
- **Why it matters:** Rewire and the ICCAD 2022 GNN/RL mapper are closer than the current internal baselines.
- **Procedure:** Port the closest available method to a frozen common subset where legal and licensing contracts permit. Document differences in action atomicity, routing semantics, architecture assumptions, termination, and budget. When execution is impossible, provide a precise contract comparison and make no superiority claim.
- **Expected artifacts:** baseline implementation/configuration or documented incompatibility report, plus common-subset quality and compilation-time results.
- **Update afterward:** Related Work, methodology, results, and novelty statement.

## P0 - Author review of AI-use disclosure and anonymity — COMPLETED

- **Manuscript location:** Appendix A after references.
- **Why it matters:** HPCA requires disclosure of AI-generated content. The text must accurately reflect the authors' actual use and institutional policy.
- **Procedure:** Review and amend the appendix. Confirm that no acknowledgments, repository identities, paths, or PDF metadata reveal author identity.
- **Expected artifact:** author-approved disclosure and anonymous PDF.
- **Completion:** The author approved the existing disclosure in the QuotientFlow coding chat. The appendix text remains byte-for-byte unchanged from the supplied archive, and the automated PDF preflight checks anonymity-sensitive text and metadata.
