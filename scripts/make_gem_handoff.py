#!/usr/bin/env python3
"""Build a compact, verified gem-branch handoff with reusable experiment state."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import zipfile
from pathlib import Path


EXTRA_PATHS = (
    "results/gem_audit_20260731",
    "results/remote_snapshots/20260731_flowadvantage_v2_manifest_v7_terminal",
    "results/revision_v3/checkpoints",
    "results/revision_v3/selected_flow_model.json",
    "results/revision_v4b/REPORT.md",
    "results/revision_v4b/RUN_SUMMARY.txt",
    "results/revision_v4b/gates.csv",
    "results/revision_v4b/raw",
    "results/revision_v4b/tables",
    "results/revision_v4b/plots",
    "results/revision_v5b/REPORT.md",
    "results/revision_v5b/RUN_SUMMARY.txt",
    "results/revision_v5b/gates.csv",
    "results/revision_v5b/raw",
    "results/revision_v5b/tables",
    "results/revision_v5b/plots",
    "results/revision_v5b/cache",
    "results/revision_v5b/native_input_registry.json",
    "results/revision_v5b/native_input_registry_v2.json",
    "results/revision_v5b/native_input_registry_v3.json",
    "results/revision_v5b/native_input_registry_v4.json",
    "results/paper_suite_v2_length_long_v1",
    "results/paper_suite_v2_length_ext_v3",
    "results/paper_suite_v2_length_ext_v4",
)

SKIP_PARTS = {"__pycache__", ".pytest_cache"}
SKIP_SUFFIXES = {".pyc", ".zip"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def files_under(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.exists():
        return []
    return [candidate for candidate in path.rglob("*") if candidate.is_file()]


def allowed(path: Path) -> bool:
    return not (SKIP_PARTS.intersection(path.parts) or path.suffix in SKIP_SUFFIXES or path.name.endswith("Zone.Identifier"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=root).split(b"\0")
    candidates = {root / item.decode() for item in tracked if item}
    for relative in EXTRA_PATHS:
        candidates.update(files_under(root / relative))
    files = sorted(path for path in candidates if path.exists() and allowed(path) and path.resolve() != output)

    temporary = output.with_suffix(output.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as archive:
        for path in files:
            archive.write(path, path.relative_to(root).as_posix())
        archive.writestr(
            "HANDOFF_BUILD.txt",
            "source_commit=" + subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip() + "\n"
            + f"file_count={len(files)}\n"
            + "builder=scripts/make_gem_handoff.py\n",
        )
    temporary.replace(output)

    with zipfile.ZipFile(output) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise RuntimeError(f"archive CRC validation failed at {bad}")
        archive_count = len(archive.infolist())
    checksum = sha256(output)
    output.with_suffix(output.suffix + ".sha256").write_text(f"{checksum}  {output.name}\n", encoding="utf-8")
    output.with_suffix(output.suffix + ".verified.txt").write_text(
        f"zip_test=PASS\nentries={archive_count}\nsha256={checksum}\n", encoding="utf-8"
    )
    print(f"archive={output}")
    print(f"entries={archive_count}")
    print(f"bytes={output.stat().st_size}")
    print(f"sha256={checksum}")


if __name__ == "__main__":
    main()
