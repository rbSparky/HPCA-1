#!/usr/bin/env python3
"""Write the bounded closest-baseline compatibility audit.

The records intentionally distinguish an available publication from a runnable,
semantically matched artifact.  No published number is converted into a local
result and no unavailable baseline is represented by an internal proxy.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results/hail_mary_hpca1/reviewer_baselines"
AUDIT_TIME = "2026-07-30T23:00:57Z"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict], columns: list[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append(
            "| " + " | ".join(str(row.get(column, "")).replace("|", "\\|") for column in columns) + " |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    artifacts = [
        {
            "baseline": "LISA",
            "publication": "LISA: Graph Neural Network based Portable Mapping on Spatial Accelerators (HPCA 2022)",
            "publication_url": "https://www.comp.nus.edu.sg/~tulika/HPCA_LISA_2022.pdf",
            "official_repository": "https://github.com/ecolab-nus/lisa",
            "artifact_commit": "410695957206ee4b3a4f131e5df3eea3044ec16c",
            "license": "MIT for top-level lisa repository; lisa_gnn_model repository has no detected LICENSE metadata",
            "code_found": "yes",
            "released_weights": "cgra_me_3_3 and cgra_me_4_4 only at lisa_gnn_model commit f98164aedc24645a5453d4233fc2b8dd6c619ead",
            "local_build_status": "mapper-side LISA C++ exists in pinned Morpher; external lisa_gnn directory/model runner is absent from that checkout",
            "semantic_match": "no",
            "classification": "OFFICIAL_ARTIFACT_PRESENT_NOT_SEMANTICALLY_MATCHED",
            "reason": "released weights and labels are CGRA-ME architecture-specific; A0-A3 use Morpher native MRRG/resource contracts and do not match either released checkpoint",
            "allowed_paper_use": "related-work/contract comparison; execute only after architecture-specific training on a development-only corpus",
            "search_evidence": "official README, git tree, submodule commits, released checkpoint paths, and pinned local source inspection",
        },
        {
            "baseline": "ICCAD22 GNN/RL mapper",
            "publication": "Towards High-Quality CGRA Mapping with Graph Neural Networks and Reinforcement Learning (ICCAD 2022)",
            "publication_url": "https://doi.org/10.1145/3508352.3549458",
            "official_repository": "not found",
            "artifact_commit": "",
            "license": "not available",
            "code_found": "no",
            "released_weights": "not found",
            "local_build_status": "not attempted: no official executable artifact located",
            "semantic_match": "paper-level method is related but no executable contract is available",
            "classification": "NO_OFFICIAL_EXECUTABLE_ARTIFACT_FOUND",
            "reason": "bounded GitHub repository/code-title/DOI/author search found the ACM publication record but no author-linked source, environment, weights, or parser contract",
            "allowed_paper_use": "related-work/algorithm contract comparison only; no local quality or runtime number",
            "search_evidence": "Crossref DOI metadata, ACM/SIGDA publication page, GitHub searches by exact title, DOI, author names, CGRA+GNN/RL",
        },
        {
            "baseline": "Rewire",
            "publication": "Rewire: Advancing CGRA Mapping Through a Consolidated Routing Paradigm (DAC 2025)",
            "publication_url": "https://www.comp.nus.edu.sg/~tulika/DAC25.pdf",
            "official_repository": "not found",
            "artifact_commit": "",
            "license": "not available",
            "code_found": "no",
            "released_weights": "not applicable/not found",
            "local_build_status": "not attempted: no official executable artifact located",
            "semantic_match": "no runnable match",
            "classification": "NO_OFFICIAL_EXECUTABLE_ARTIFACT_FOUND",
            "reason": "paper consumes an invalid PF* initial mapping and repairs multi-node clusters using consolidated propagation; FlowAdvantage starts from an empty/partial legal state and chooses atomic placement+routing actions, so a substitute implementation would not be faithful",
            "allowed_paper_use": "related-work/contract comparison only; no local superiority claim",
            "search_evidence": "author-hosted paper, Crossref DOI metadata, GitHub searches by exact title, DOI, authors, Rewire+CGRA",
        },
    ]
    fields = list(artifacts[0])
    write_csv(output / "artifact_audit.csv", fields, artifacts)

    contracts = [
        {
            "dimension": "action atomicity",
            "FlowAdvantage": "one operation + PE/time + completed-dependency routes; learned proposal then exact child reranking",
            "LISA": "label-guided Morpher/CGRA-ME mapping order and placement preferences",
            "ICCAD22_GNN_RL": "joint placement distribution with routing connectivity reward, per publication abstract",
            "Rewire": "multi-node cluster placement and routing repair",
        },
        {
            "dimension": "starting state",
            "FlowAdvantage": "empty legal state for headline; controlled legal prefixes for sensitivity",
            "LISA": "native mapper state with learned labels",
            "ICCAD22_GNN_RL": "publication-specific reduced resource graph mapping state",
            "Rewire": "invalid initial PF* mapping",
        },
        {
            "dimension": "routing semantics",
            "FlowAdvantage": "actual Morpher II-expanded MRRG and independently checked ordered routes",
            "LISA": "released checkpoints target CGRA-ME architectures; Morpher integration needs its external inference assets",
            "ICCAD22_GNN_RL": "reduced resource graph; exact public parser/resource contract unavailable",
            "Rewire": "simultaneous forward/backward propagation and intersection; private PF* implementation",
        },
        {
            "dimension": "portability requirement",
            "FlowAdvantage": "frozen synthetic-trained checkpoint; A3 held out; no real per-architecture labels",
            "LISA": "official workflow generates architecture-specific labels and trains a model for each new accelerator",
            "ICCAD22_GNN_RL": "unknown without released artifact",
            "Rewire": "non-learned but requires its multi-node mapper and PF* initial mapping implementation",
        },
        {
            "dimension": "fair local status",
            "FlowAdvantage": "executed in the paired matrix",
            "LISA": "unsupported for A0-A3 with released weights",
            "ICCAD22_GNN_RL": "artifact unavailable",
            "Rewire": "artifact unavailable",
        },
    ]
    write_csv(output / "contract_comparison.csv", list(contracts[0]), contracts)

    searches = [
        {
            "timestamp_utc": AUDIT_TIME,
            "service": "GitHub REST via authenticated gh CLI",
            "query": "CGRA GNN; LISA CGRA; exact title/DOI/author searches for ICCAD22; Rewire CGRA/DAC25/author searches",
            "result": "ecolab-nus/lisa located; no repository located for the ICCAD22 paper or Rewire",
        },
        {
            "timestamp_utc": AUDIT_TIME,
            "service": "Crossref API",
            "query": "10.1145/3508352.3549458; 10.1109/DAC63849.2025.11133240",
            "result": "publication metadata located; metadata exposes no source artifact",
        },
        {
            "timestamp_utc": AUDIT_TIME,
            "service": "local pinned Morpher checkout",
            "query": "LISAController, call_gnn.sh, lisa_gnn/model/checkpoint inventory",
            "result": "C++ integration and call_gnn.sh exist; required external lisa_gnn directory is absent",
        },
    ]
    write_csv(output / "search_log.csv", list(searches[0]), searches)

    local_evidence_paths = [
        ROOT / "scripts/generate_reviewer_baseline_audit.py",
        ROOT / "results/revision_v5/TOOLCHAIN_DECISION.md",
        ROOT / "results/revision_v5/toolchain/morpher/.gitmodules",
        ROOT / "results/revision_v5b/morpher_patch/Morpher_CGRA_Mapper/call_gnn.sh",
        ROOT / "results/revision_v5b/morpher_patch/Morpher_CGRA_Mapper/src/lisa/LISAController.cpp",
        ROOT / "results/revision_v5b/morpher_patch/Morpher_CGRA_Mapper/src/lisa/LISAMapper.cpp",
    ]
    local_evidence = [
        {
            "path": str(path.relative_to(ROOT)),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in local_evidence_paths
    ]
    write_csv(output / "local_evidence_manifest.csv", list(local_evidence[0]), local_evidence)

    report = f"""# Closest learned/routing-aware baseline compatibility audit

Audit frozen at **{AUDIT_TIME}**. This was a bounded official-artifact search, not a claim that no private or future implementation exists.

## Decision table

{markdown_table(artifacts, ['baseline', 'code_found', 'classification', 'semantic_match', 'allowed_paper_use'])}

## LISA

The official [LISA repository](https://github.com/ecolab-nus/lisa) is available under MIT at commit `410695957206ee4b3a4f131e5df3eea3044ec16c`. Its pinned `lisa_gnn` and `lisa_gnn_model` submodules are `114b6941544e13132d7220aee22522488b118598` and `f98164aedc24645a5453d4233fc2b8dd6c619ead`. The latter releases small checkpoints only for `cgra_me_3_3` and `cgra_me_4_4`.

The official portability procedure is architecture-specific: generate mapping labels for the new accelerator, filter them, and train models named for that architecture. The README warns that label generation can take hours per label. The pinned Morpher source includes `LISAController::callGNNInference`, which requires `$LISA_DIR`, invokes `morpher_mapper/call_gnn.sh`, writes to `../lisa_gnn/data/infer`, and reads architecture-named output labels. The pinned Morpher experiment checkout does not contain that external `lisa_gnn` directory.

Therefore the released LISA weights are not a semantically matched baseline for A0–A3. Copying a `cgra_me_4_4` checkpoint or reporting the native `-m 2` path without its actual inference assets would be invalid. A proper port requires development-only label generation and architecture-specific training, after which it must be reported as such rather than as zero-shot portability.

## ICCAD 2022 GNN/RL mapper

The publication is identifiable as DOI [`10.1145/3508352.3549458`](https://doi.org/10.1145/3508352.3549458), by Yan Zhuang, Zhihao Zhang, and Dajiang Liu. Its public abstract describes GNN policy-gradient placement over a reduced resource graph with routing-connectivity reward. Exact-title, DOI, author, and `CGRA GNN RL mapper` searches found no official repository, environment, checkpoint, license, or import/export contract.

It cannot be faithfully reproduced before the deadline from the paper alone, and an internal imitation would not be the published baseline. It is included as the closest learned-method contract comparison; no local performance number or superiority claim is allowed.

## Rewire

The author-hosted [DAC 2025 paper](https://www.comp.nus.edu.sg/~tulika/DAC25.pdf) defines a materially different multi-node repair mapper. It begins with an invalid mapping from the authors' fine-tuned roughly 3K-line PF* implementation, propagates routing information jointly, intersects candidate sets, and remaps clusters of up to 15 nodes. The paper evaluates proprietary/fine-tuned PF*, SA, and Rewire implementations. The bounded source search found no official repository or executable artifact.

FlowAdvantage's atomic legal-state action interface cannot substitute for Rewire's multi-node invalid-state repair interface. Reimplementation under the deadline would change both the action and termination contracts. Rewire is therefore documented as related routing-aware work, not included as a normalized local result.

## Contract comparison

{markdown_table(contracts, ['dimension', 'FlowAdvantage', 'LISA', 'ICCAD22_GNN_RL', 'Rewire'])}

## What the paper may claim

- PathFinder and simulated annealing are the complete matched conventional baselines in the current Morpher contract.
- LISA has an official artifact, but its released checkpoints do not match A0–A3 and its portability procedure requires new architecture-specific labels/training.
- No runnable official artifact was located for the ICCAD22 GNN/RL mapper or Rewire during the bounded audit.
- FlowAdvantage results must not be described as outperforming any of those three methods unless a future common-contract execution is completed.
- The absence of a runnable matched artifact is an evaluation limitation, not evidence of superiority.

Machine-readable evidence is in `artifact_audit.csv`, `contract_comparison.csv`, and `search_log.csv`.
"""
    (output / "BASELINE_COMPATIBILITY.md").write_text(report, encoding="utf-8")

    checksum_paths = sorted(path for path in output.iterdir() if path.is_file())
    (output / "SHA256SUMS.txt").write_text(
        "".join(
            f"{sha256(path)}  {path.name}\n"
            for path in checksum_paths if path.name != "SHA256SUMS.txt"
        ),
        encoding="utf-8",
    )
    print(f"wrote {len(artifacts)} baseline records to {output}")


if __name__ == "__main__":
    main()
