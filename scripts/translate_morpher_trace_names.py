#!/usr/bin/env python3
"""Translate versioned SSA names in an upstream Morpher memory trace corpus.

This is intentionally narrower than a generic trace converter.  The caller
must provide an explicit, complete old-to-new name map and an authoritative
target memory allocation.  Numeric offsets and pre/post values are copied
verbatim.  The resulting manifest hashes every source and output, making the
translation suitable for functional simulator evidence without regenerating
or inventing test values.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


CONTROL_NAMES = {"loopstart", "loopend", "storestart"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _allocation_names(path: Path) -> set[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["var_name", "base_addr"]:
            raise ValueError(f"unexpected allocation header: {reader.fieldnames}")
        return {row["var_name"] for row in reader}


def translate_corpus(
    source_directory: Path,
    output_directory: Path,
    name_map_path: Path,
    target_allocation: Path,
    provenance_path: Path,
) -> None:
    name_map = json.loads(name_map_path.read_text(encoding="utf-8"))
    if not isinstance(name_map, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in name_map.items()
    ):
        raise ValueError("name map must be a JSON object of string-to-string entries")
    if len(set(name_map.values())) != len(name_map):
        raise ValueError("name map is not injective")

    target_names = _allocation_names(target_allocation)
    source_paths = sorted(source_directory.glob("*.txt"))
    if not source_paths:
        raise ValueError(f"no trace files in {source_directory}")
    if output_directory.exists():
        raise FileExistsError(f"refusing to overwrite {output_directory}")

    temporary_directory = Path(
        tempfile.mkdtemp(
            dir=output_directory.parent,
            prefix=f".{output_directory.name}.",
            suffix=".tmp",
        )
    )
    records: list[dict[str, Any]] = []
    try:
        observed_source_names: set[str] = set()
        observed_target_names: set[str] = set()
        for source_path in source_paths:
            output_path = temporary_directory / source_path.name
            with source_path.open(newline="", encoding="utf-8") as source, output_path.open(
                "w", newline="", encoding="utf-8"
            ) as output:
                reader = csv.DictReader(source)
                expected_header = [
                    "var_name",
                    "offset",
                    "pre-run-data",
                    "post-run-data",
                ]
                if reader.fieldnames != expected_header:
                    raise ValueError(
                        f"{source_path} has unexpected trace header {reader.fieldnames}"
                    )
                writer = csv.DictWriter(
                    output, fieldnames=expected_header, lineterminator="\n"
                )
                writer.writeheader()
                row_count = 0
                for row in reader:
                    source_name = row["var_name"]
                    observed_source_names.add(source_name)
                    target_name = name_map.get(source_name, source_name)
                    if (
                        target_name not in target_names
                        and target_name not in CONTROL_NAMES
                    ):
                        raise ValueError(
                            f"{source_path}: translated variable {target_name!r} "
                            "does not occur in the target allocation"
                        )
                    observed_target_names.add(target_name)
                    row["var_name"] = target_name
                    writer.writerow(row)
                    row_count += 1
                output.flush()
                os.fsync(output.fileno())
            records.append(
                {
                    "source_path": str(source_path.resolve()),
                    "source_sha256": _sha256(source_path),
                    "output_file": output_path.name,
                    "output_sha256": _sha256(output_path),
                    "rows": row_count,
                }
            )

        unnecessary = sorted(set(name_map) - observed_source_names)
        if unnecessary:
            raise ValueError(
                "name map contains names absent from the corpus: " + ", ".join(unnecessary)
            )
        os.replace(temporary_directory, output_directory)
    except BaseException:
        # Preserve a failed translation tree for diagnosis instead of deleting it.
        failed = temporary_directory.with_name(temporary_directory.name + ".failed")
        if temporary_directory.exists():
            os.replace(temporary_directory, failed)
        raise

    manifest = {
        "schema": "morpher_trace_name_translation_v1",
        "source_directory": str(source_directory.resolve()),
        "source_commit": "4f3da1acd89c475b101c76f772351330c2f95bf5",
        "name_map_path": str(name_map_path.resolve()),
        "name_map_sha256": _sha256(name_map_path),
        "target_allocation": str(target_allocation.resolve()),
        "target_allocation_sha256": _sha256(target_allocation),
        "numeric_value_transformation": "none",
        "offset_transformation": "none",
        "translated_files": len(records),
        "files": records,
    }
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    with provenance_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--name-map", type=Path, required=True)
    parser.add_argument("--target-allocation", type=Path, required=True)
    parser.add_argument("--provenance-json", type=Path, required=True)
    arguments = parser.parse_args()
    translate_corpus(
        source_directory=arguments.source_directory,
        output_directory=arguments.output_directory,
        name_map_path=arguments.name_map,
        target_allocation=arguments.target_allocation,
        provenance_path=arguments.provenance_json,
    )


if __name__ == "__main__":
    main()
