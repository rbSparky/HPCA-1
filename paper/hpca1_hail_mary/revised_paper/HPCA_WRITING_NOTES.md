# HPCA paper-writing notes grounded in the supplied corpus

## Corpus inspected

The supplied folder contains 18 PDFs (17 distinct binary files), spanning HPCA papers on GNN acceleration, systems, algorithm--architecture co-design, and LISA, the closest paper to this work. I extracted all PDFs to layout-preserving text, inspected every first page, inspected a mid-paper design/evaluation page from every distinct PDF, and read LISA in detail.

## Consistent paper flow

1. **Opening context is brief and architectural.** The first paragraph establishes the platform or workload importance. The next paragraph identifies a concrete system/compiler bottleneck. Accepted papers do not spend the introduction on implementation history.
2. **The gap is stated as a conflict between desirable properties.** Typical examples are quality versus compilation time, scalability versus fidelity, or generality versus architecture-specific optimization.
3. **A single key observation bridges motivation to design.** The strongest papers show a measurement, counterexample, or structural argument before naming the mechanism.
4. **The method overview appears before the contribution list.** The reader should understand the central mechanism, what it consumes, and what it outputs before seeing bullets.
5. **Contributions are concrete nouns and verified capabilities.** They enumerate a formulation, mechanism, implementation/integration, and evaluation. They do not list coding tasks or intermediate versions.
6. **Background is selective.** It defines only the architecture, execution model, metric, or algorithmic primitive required to understand the new idea. General textbook material is compressed.
7. **Design sections are invariant-driven.** Each component states what must remain legal/correct, then explains the mechanism and its cost. Notation is introduced once and reused.
8. **Evaluation is question-driven.** A typical order is: methodology and baselines; end-to-end result; mechanism validation; ablations; sensitivity/scaling; overhead; limitations. Results are compared on identical inputs and budgets.
9. **Related work is late.** It positions the paper after the design and evaluation are understood.
10. **The conclusion is one compact paragraph.** It restates the problem, mechanism, and supported result without adding new claims.

## Formality and technical style

- Use direct technical claims with explicit scope: “on the paired cells,” “under the fixed budget,” “for legal terminal mappings.”
- Define every symbol before first use. Keep one name per object: DFG node, MRRG resource, partial state, action, relaxation, dual price, residual advantage.
- Distinguish feasibility, legality, mapping failure, timeout, and infrastructure error. Never aggregate them into one failure rate.
- Separate a first-order dual estimate from the exact child-relaxation objective. State precisely what the model predicts.
- State deterministic tie-breaking and candidate-universe equivalence when comparing search policies.
- Use tables for setup and exact values; use figures for trends, breakdowns, and intuition.
- Captions should be self-contained and state the takeaway or the question answered.
- Avoid promotional adjectives unless a quantitative result immediately supports them.

## What a competitive HPCA version definitely needs

- A clear architectural/compiler problem with relevance beyond one codebase.
- A precise formulation of the residual mapping problem and a clean derivation of the action score.
- A native end-to-end integration with legality and output validation.
- Conventional strong baselines (PathFinder and seeded simulated annealing here), not only simplified internal baselines.
- Paired coverage across kernels, architectures, seeds, and equal budgets.
- Mapping success, minimum II, route cost, compilation time, and coverage/status accounting.
- An ablation separating parent duals, learned residual correction, top-k exact child solves, and exact quotienting.
- Held-out architecture evaluation and scaling.
- Statistical intervals for paired aggregate deltas.
- A runtime breakdown that explains solver, feature, model, routing, and native-tool overheads.

## Do not do

- Do not narrate coding phases, patch numbers, old method versions, debug incidents, or abandoned approaches.
- Do not refer to internal filenames, queue names, machine paths, or commit hashes in the main paper.
- Do not call a timeout a mapping failure or infer quality from nonterminal rows.
- Do not use a reference mapping as hidden input to the learned mapper.
- Do not claim an ablation when the compared methods do not share the same candidate universe, budget, and terminal paired cells.
- Do not report a proxy “no-parent” model as a true no-parent result.
- Do not claim portability from an architecture used in training or threshold selection.
- Do not introduce new jargon for existing CGRA concepts.
- Do not use “et al.” in the bibliography; the HPCA template explicitly requires all authors.

## Draft-specific grounding decision

The supplied handoff supports the formulation, native integration design, 148 passing regression tests, dual independent/native negative legality checks on eight mutations, an incomplete 13-row length baseline matrix, and a strict two-architecture `trmm` probe. It does not contain the complete paired headline matrix, learned-model test metrics, quotienting results, or statistically supported quality gains. The first draft therefore reports only available evidence and leaves explicit figure space for the missing paired results rather than fabricating them.
