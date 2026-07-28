#!/usr/bin/env python3
"""Package the validation-selected post-baseline revision without rewriting baseline data."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results" / "quick_v1"
REV = ROOT / "results" / "revision_v1"
TABLES = REV / "tables"
PLOTS = REV / "plots"


def markdown_table(frame: pd.DataFrame) -> str:
    def cell(value: object) -> str:
        if pd.isna(value):
            return ""
        return str(value).replace("|", "\\|").replace("\n", " ")

    header = "| " + " | ".join(map(str, frame.columns)) + " |"
    rule = "| " + " | ".join("---" for _ in frame.columns) + " |"
    rows = [
        "| " + " | ".join(cell(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    ]
    return "\n".join([header, rule, *rows])


def write_table(frame: pd.DataFrame, stem: str) -> None:
    frame.to_csv(TABLES / f"{stem}.csv", index=False)
    (TABLES / f"{stem}.md").write_text(markdown_table(frame) + "\n")


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)

    mapper = pd.read_csv(REV / "mapper_scarcity_summary.csv")
    symmetry_baseline = pd.read_csv(REV / "symmetry_anchor1.csv")
    optimized_path = REV / "symmetry_optimized.csv"
    symmetry = (
        pd.read_csv(optimized_path)
        if optimized_path.exists()
        else symmetry_baseline
    )
    validation = pd.read_csv(REV / "validation_scarcity_ablation_summary.csv")
    baseline_gates = pd.read_csv(BASE / "tables" / "T08_gates.csv")
    mapper_gate = json.loads((REV / "gate_scarcity_assessment.json").read_text())
    symmetry_gate = json.loads((REV / "symmetry_anchor1_assessment.json").read_text())
    optimized_assessment_path = REV / "symmetry_optimized_assessment.json"
    optimized_assessment = (
        json.loads(optimized_assessment_path.read_text())
        if optimized_assessment_path.exists()
        else None
    )
    if optimized_assessment is not None:
        symmetry_gate["canonicalization_fraction"] = optimized_assessment[
            "new_canonicalization_fraction"
        ]
        symmetry_gate["geomean_expansion_reduction"] = optimized_assessment[
            "geomean_expansion_reduction"
        ]

    write_table(validation, "R01_validation_scarcity_selection")
    write_table(mapper, "R02_frozen_mapper")
    write_table(symmetry, "R03_exact_quotient")

    revised = baseline_gates.copy()
    revised.loc[revised.gate == "G3", ["status", "observed_value", "notes"]] = [
        "AMBER",
        "7.14pt/0%; asym 7.14pt",
        "validation-selected scarcity integration; 0 timeouts; 0 legality failures",
    ]
    revised.loc[revised.gate == "G4", ["status", "observed_value", "notes"]] = [
        "AMBER",
        "1.181x/0.061",
        "12/12 exact matches; best family 1.309x; optimized overhead 6.09%",
    ]
    revised.loc[revised.gate == "G6", ["status", "observed_value", "notes"]] = [
        "GREEN",
        "3 qualifying splits",
        "100% oracle success-gain retention on seen/size/asym; no success loss",
    ]
    write_table(revised, "R04_revised_gates")

    plot_methods = ["length", "oracle_dual", "pred_dual_quotient"]
    success = mapper[mapper.method.isin(plot_methods)].pivot(
        index="split", columns="method", values="success_rate"
    )
    success = success[plot_methods]
    ax = success.plot(kind="bar", figsize=(9, 4), ylim=(0.75, 1.02))
    ax.set_ylabel("Legal mapping success rate")
    ax.set_xlabel("")
    ax.legend(title="")
    plt.tight_layout()
    plt.savefig(PLOTS / "revision_mapping_success.png", dpi=180)
    plt.close()

    family = (
        symmetry.groupby("dfg_family", as_index=False)["expansion_reduction"]
        .apply(lambda x: x.prod() ** (1.0 / len(x)))
        .rename(columns={"expansion_reduction": "geomean_reduction"})
    )
    ax = family.plot(
        kind="bar", x="dfg_family", y="geomean_reduction", legend=False, figsize=(7, 4)
    )
    ax.axhline(1.0, color="black", linewidth=0.8)
    ax.axhline(1.5, color="tab:green", linestyle="--", linewidth=0.8)
    ax.set_ylabel("Expansion reduction (x)")
    ax.set_xlabel("")
    plt.tight_layout()
    plt.savefig(PLOTS / "revision_quotient_reduction.png", dpi=180)
    plt.close()

    gates_md = markdown_table(revised[["gate", "status", "observed_value", "notes"]])
    mapper_md = markdown_table(
        mapper[mapper.method.isin(["length", "oracle_dual", "pred_dual_quotient"])][
            [
                "split",
                "method",
                "success_rate",
                "success_delta_points",
                "median_expansions",
                "timeouts",
                "legality_failures",
            ]
        ]
    )
    family_md = markdown_table(family)

    report = f"""# QuotientFlow post-baseline revision report

## Status

**Overall decision: PROCEED WITH REFINEMENT.**

This report is an auditable, post-baseline failure-driven revision. The original
one-hour result remains unchanged in [`../quick_v1/REPORT.md`](../quick_v1/REPORT.md)
and reported `STOP/REFORMULATE`. No gate threshold was changed. Mapper choices
were selected only on the validation split and then frozen before the four test
splits were evaluated.

## Exact commands

Baseline:

`python scripts/run_quick_suite.py --config configs/quick.yaml`

Frozen mapper revision:

`python scripts/run_revision2_scarcity_suite.py`

Exact quotient revision:

`python scripts/run_symmetry_revision.py`

Packaging:

`python scripts/make_revision_report.py`

Activate the reused low-space environment with:

`source /home/rishabh/miniconda/etc/profile.d/conda.sh && conda activate taugat_pyg`

## Runtime accounting

The completed official quick campaign took **58.048949 minutes**, including its
documented uncensoring retries. The subsequent failure-driven mapper/quotient
campaign occupied approximately **15.766 minutes** of elapsed experiment time
(09:12:28--09:28:14 UTC, reconstructed from artifact timestamps); the two final
frozen runs themselves recorded **1.609956 minutes**. Thus the official suite
plus the authorized empirical refinement used approximately **73.815 minutes**.
Report packaging and verification are not counted as experiment runtime.

## Why the revision was permitted

`AGENTS.md` section 20 prescribes validation-only investigation when G3 or G4
fails. The mapper study tested sum, maximum, top-2, dynamic price refresh, price
weights, and alternative-path scarcity. The selected mapper uses beam width 4,
oracle/predicted price weight 0.5, and weights each link price by the inverse of
the number of residual alternative paths (capped at four). All choices and
rejected trials are retained in the validation ablation CSVs.

The quotient study reduced the deterministic partial-state anchor count from the
baseline 25% state to one anchor. It still enumerates the full verified DFG and
architecture groups; no group is truncated or described as exact when it is not.

## Revised gates

{gates_md}

G0--G2 and G5 are inherited from the completed baseline because neither data,
relaxation, knockout probes, nor trained models changed. G3, G4, and G6 are
reassessed on the frozen revisions.

## Frozen mapper results

{mapper_md}

Oracle guidance improved legal success by 14.29 points on seen tests and 7.14
points on both unseen-size and unseen-asymmetric tests. Across all 56 instances,
the paired improvement was 7.14 points, narrowly below G3's 8-point GREEN
threshold. The asymmetric improvement was also 7.14 points, below its required
10 points. Therefore G3 remains **AMBER**, not GREEN.

The predicted quotient mapper exactly retained the positive oracle success gains
on three splits and never reduced success. It therefore satisfies G6 and is
**GREEN**. All 336 frozen mapper runs completed with zero timeouts and zero
independent-legality failures.

## Exact quotient results

{family_md}

All 12 paired exactness cases matched success, best cost, mapped count, and
legality. The geometric-mean reduction was
{symmetry_gate["geomean_expansion_reduction"]:.3f}x and canonicalization consumed
{100.0 * symmetry_gate["canonicalization_fraction"]:.1f}% of quotient runtime.
The optimized exact-key implementation reproduced every success, cost, and
expansion count from the preserved run while reducing canonicalization work by
11.97x and total quotient runtime by 1.75x. This leaves G4 **AMBER** because its
expansion reduction is below 1.5x, not because of overhead. A separate beam-512 diagnostic on
diamond-chain yielded only 1.023x reduction (3156 versus 3085 expansions) while
increasing quotient runtime, so beam widening was rejected rather than selected
on a favorable anecdote.

## What failed

- G3 missed GREEN by one recovered mapping in the 56-instance aggregate and by
  one mapping on the 14-instance asymmetric split. Expansion counts did not
  improve because successful beam-4 searches terminate at essentially the same
  depth.
- G4 remains below the 1.5x geometric-mean and 2x per-family GREEN thresholds,
  although canonicalization overhead is now comfortably within GREEN's limit.
- G5 remains AMBER: rank correlations are weak even though critical-link AUROC,
  action preservation, and the integrated mapper outcome are useful.
- One of 252 baseline relaxation samples was genuinely infeasible; the remaining
  251 passed stringent residual checks, leaving G1 GREEN.

These are empirical/methodological limitations, not timeout classifications.

## Scientific interpretation

The optimization signal is valid: G1 and the finite-link-knockout G2 test are
GREEN, and scarcity-aware oracle prices improve legal mapping success. The raw
sum integration was the problem, not evidence that duals were meaningless.

Learned price magnitudes/ranks remain mediocre in isolation (G5 AMBER), but their
decision-relevant ordering is sufficient for the frozen integrated mapper to
retain all observed oracle success gains (G6 GREEN). This discrepancy argues for
training on ranks, criticality, or action regret rather than only price magnitude.

Exact quotienting is correct and canonicalization is now inexpensive. Its
1.181x expansion reduction is still too small for a headline claim.
Stabilizer-aware action-orbit generation may further reduce generated work, but
cannot manufacture additional quotient classes where exact symmetry is absent.

## Recommended next iteration

Keep the fractional residual dual signal, scarcity-aware mapper, and optimized
exact key. On validation only, train a rank/critical-link head with residual
alternative-path count and occupancy features, then freeze it for a larger test.
Evaluate stabilizer-aware action orbits as a secondary runtime optimization.
Only then integrate Morpher-v2 or CGRA-ME and expand to real kernels.

## Can this support an HPCA paper?

Not yet. The pilot now justifies continued work, but it does not meet the
paper-grade thresholds in `AGENTS.md`: real toolchain integration, at least 30
kernels, stronger compile-budget gains, 1.5x exact-quotient reduction, and
cycle-accurate validation are still required.

## Artifact index

- Baseline report and complete mandatory artifacts: `results/quick_v1/`
- Frozen revised mappings: `results/revision_v1/mappings_scarcity.csv`
- Validation-only ablations: `results/revision_v1/validation_*`
- Exact quotient pairs: `results/revision_v1/symmetry_anchor1.csv`
- Optimized exact quotient confirmation: `results/revision_v1/symmetry_optimized.csv`
- Revision tables: `results/revision_v1/tables/`
- Revision plots: `results/revision_v1/plots/`
"""
    (REV / "REPORT.md").write_text(report)

    summary = f"""OVERALL_DECISION: PROCEED WITH REFINEMENT
TOTAL_WALL_MINUTES: 58.048949 official baseline; approximately 73.815 including the 15.766-minute post-baseline refinement campaign (final frozen revision runs: 1.609956)
CUDA_AVAILABLE: True
G0: GREEN
G1: GREEN
G2: GREEN
G3: AMBER
G4: AMBER
G5: AMBER
G6: GREEN
TOP_THREE_FINDINGS: Oracle success +7.14 points overall; predicted mapper retains oracle gains on 3 splits; optimized exact canonicalization overhead is 6.09% with identical search results
TOP_THREE_FAILURES: G3 misses GREEN by one aggregate/asymmetric success; G4 expansion reduction is only 1.181x; G5 price-rank correlation remains weak
NEXT_ACTION: Train rank/criticality guidance with scarcity and occupancy features, then evaluate stabilizer-aware action orbits before external-toolchain scaling.
"""
    (REV / "RUN_SUMMARY.txt").write_text(summary)


if __name__ == "__main__":
    main()
