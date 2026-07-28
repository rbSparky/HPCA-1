"""Table, plot, and report output helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def markdown_text(df: pd.DataFrame) -> str:
    columns = [str(c) for c in df.columns]
    rows = [columns, ["---"] * len(columns)]
    for row in df.itertuples(index=False, name=None):
        rows.append(
            [
                "" if pd.isna(value) else str(value).replace("|", "\\|")
                for value in row
            ]
        )
    return "\n".join("| " + " | ".join(row) + " |" for row in rows) + "\n"


def write_table(df: pd.DataFrame, directory: Path, name: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    df.to_csv(directory / f"{name}.csv", index=False)
    (directory / f"{name}.md").write_text(markdown_text(df))


def make_plots(output: Path, knockouts, predictions, mappings):
    plots = output / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    paths = []

    fig, ax = plt.subplots(figsize=(6, 4))
    finite = knockouts[np.isfinite(knockouts["impact"])] if len(knockouts) else knockouts
    ax.scatter(finite.get("dual_price", []), finite.get("impact", []), s=9, alpha=0.45)
    ax.set(xlabel="Oracle dual price", ylabel="Finite knockout impact", title="Dual price versus knockout impact")
    fig.tight_layout()
    path = plots / "dual_vs_knockout.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(6, 4))
    if len(predictions):
        chosen = predictions[predictions["selected_seed"] == True]  # noqa: E712
        ax.scatter(chosen["target_rank"], chosen["predicted_rank"], s=6, alpha=0.3)
    ax.set(xlabel="Oracle within-graph rank", ylabel="Predicted within-graph rank", title="Oracle versus predicted dual rank")
    fig.tight_layout()
    path = plots / "oracle_vs_predicted_rank.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(10, 4))
    if len(mappings):
        pivot = mappings.groupby(["split", "method"])["success"].mean().unstack(fill_value=0)
        pivot.plot(kind="bar", ax=ax)
    ax.set(ylabel="Legal mapping success rate", title="Mapping success by method and split")
    ax.legend(fontsize=7)
    fig.tight_layout()
    path = plots / "mapping_success.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(10, 4))
    if len(mappings):
        data = mappings.pivot_table(index="split", columns="method", values="expansions", aggfunc="median")
        data.plot(kind="bar", ax=ax)
    ax.set(ylabel="Median expansions", title="Expansion count by method and split")
    ax.legend(fontsize=7)
    fig.tight_layout()
    path = plots / "mapping_expansions.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)
    return paths
