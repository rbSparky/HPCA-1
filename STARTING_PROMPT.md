You are the primary coding and experimental agent for a new research prototype.

Your repository contains `AGENTS.md`. Read it completely before doing anything else. It defines the scientific hypothesis, exact one-hour experiment, architectures, DFG families, optimization formulation, GNN, mapper, symmetry rules, output tables, and pass/fail gates.

Your task is to implement and execute the **QuotientFlow one-hour validation suite** from scratch.

Context, because you have no prior conversation:

- The research direction extends LISA-style learned CGRA mapping.
- Instead of predicting handcrafted placement labels, we want to predict routing-resource shadow prices from a regularized fractional residual mapping problem.
- Prices are conditioned on a valid partial mapping state; do not evaluate only at the symmetric root state.
- A deterministic beam mapper uses oracle or predicted prices to avoid globally scarce routing resources.
- Exact architecture/DFG automorphisms are used to quotient equivalent partial mapping states.
- The immediate goal is falsification: determine whether oracle prices help, whether a small GNN can predict them, and whether exact quotienting reduces search without changing mapping quality.
- The completed quick experiment should take approximately one hour after setup.
- The long-term system must be capable of expansion into a strong HPCA evaluation, but do not begin by integrating a full external CGRA toolchain.

Execute the following:

1. Inspect the machine, NVIDIA driver, GPU, RAM, OS, Python, and available conda/mamba tools.
2. Create a conda environment named `quotientflow` with Python 3.11.
3. Install scientific dependencies.
4. Inside that conda environment, install a current official CUDA-enabled PyTorch wheel compatible with the installed NVIDIA driver. Use the official PyTorch installation selector rather than guessing a stale wheel. Verify `torch.cuda.is_available()` and run a GPU matrix-multiplication smoke test.
5. Save complete environment and hardware records in the paths required by `AGENTS.md`.
6. Implement the repository structure and all mandatory unit tests.
7. Implement the exact quick architectures and seven DFG families.
8. Implement deterministic partial-state construction.
9. Implement the regularized fractional residual placement/routing optimization with CVXPY and extract routing-capacity dual prices.
10. Implement solver-health checks and finite-link-knockout tests.
11. Implement exact verified automorphisms and canonical state quotienting. Never truncate a group and call it exact.
12. Implement the pure-PyTorch two-tower GNN described in `AGENTS.md`; do not add PyTorch Geometric.
13. Train all three required model seeds.
14. Implement the independent legality checker and all six mapper variants.
15. Run:
   `python scripts/run_quick_suite.py --config configs/quick.yaml`
16. Enforce the wall-clock budget and preserve partial results if the hard stop is approached.
17. Generate every required raw CSV, table, plot, gate assessment, `REPORT.md`, and `RUN_SUMMARY.txt`.
18. Do not alter gate thresholds after observing results.
19. Do not hide negative findings. If a gate fails, diagnose it using the failure-driven rules in `AGENTS.md`.
20. Make only minimal validation-set hyperparameter changes when necessary. Record every deviation in `config_deviations.md`; never tune on test data.

Priority order if time is constrained:

1. Correctness, legality, and solver residuals.
2. Oracle-dual versus length-only search.
3. Exact quotient correctness.
4. GNN prediction.
5. Full integrated predicted mapper.
6. Optional plots or secondary diagnostics.

Do not return a prose-only answer. Finish by returning:

- repository path;
- exact environment activation command;
- exact experiment command;
- total runtime;
- CUDA status;
- gate table G0–G6;
- overall decision;
- paths to `REPORT.md`, `RUN_SUMMARY.txt`, all tables, raw results, and plots;
- concise list of implementation compromises;
- concise recommendation for the next iteration.

The result is useful even when gates fail. Scientific honesty and reproducibility are mandatory.
