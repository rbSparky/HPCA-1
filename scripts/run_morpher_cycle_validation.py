#!/usr/bin/env python3
"""Run Morpher's cycle simulator over a deterministic trace corpus.

Each trace executes in an isolated temporary working directory because the
upstream simulator writes a fixed ``sim_result.txt`` filename.  Per-trace
stdout, stderr, exit status, timing, input hashes, and match counts are
preserved.  The final manifest is installed atomically only after every
selected trace reaches a terminal state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any


RESULT_PATTERN = re.compile(r"^\s*(\d+)\s*,\s*(\d+)\s*$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def run_corpus(
    simulator: Path,
    bitstream: Path,
    allocation: Path,
    traces: list[Path],
    output_directory: Path,
    x_dimension: int,
    y_dimension: int,
    total_memory_size: int,
    memory_arrangement: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    for path in (simulator, bitstream, allocation, *traces):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not os.access(simulator, os.X_OK):
        raise PermissionError(f"simulator is not executable: {simulator}")
    if output_directory.exists():
        raise FileExistsError(f"refusing to overwrite {output_directory}")
    if len({path.resolve() for path in traces}) != len(traces):
        raise ValueError("trace list contains duplicates")

    staging = output_directory.with_name(
        f".{output_directory.name}.tmp.{os.getpid()}"
    )
    staging.mkdir(parents=True)
    records: list[dict[str, Any]] = []
    try:
        for index, trace in enumerate(traces):
            trace_directory = staging / f"{index:04d}_{trace.stem}"
            trace_directory.mkdir()
            command = [
                str(simulator.resolve()),
                "-x",
                str(x_dimension),
                "-y",
                str(y_dimension),
                "-c",
                str(bitstream.resolve()),
                "-d",
                str(trace.resolve()),
                "-a",
                str(allocation.resolve()),
                "-m",
                str(total_memory_size),
                "-t",
                str(memory_arrangement),
            ]
            started = time.perf_counter()
            status = "ERROR"
            matches = None
            mismatches = None
            error = ""
            try:
                completed = subprocess.run(
                    command,
                    cwd=trace_directory,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=timeout_seconds,
                    check=False,
                )
                return_code = completed.returncode
                stdout = completed.stdout
                stderr = completed.stderr
                result_path = trace_directory / "sim_result.txt"
                if return_code == 0 and result_path.is_file():
                    match = RESULT_PATTERN.match(
                        result_path.read_text(encoding="utf-8")
                    )
                    if match:
                        matches, mismatches = map(int, match.groups())
                        status = "MATCH" if mismatches == 0 else "MISMATCH"
                    else:
                        error = "invalid sim_result.txt format"
                elif return_code != 0:
                    error = f"simulator exit code {return_code}"
                else:
                    error = "simulator did not produce sim_result.txt"
            except subprocess.TimeoutExpired as exception:
                return_code = None
                stdout = (
                    exception.stdout.decode()
                    if isinstance(exception.stdout, bytes)
                    else exception.stdout or ""
                )
                stderr = (
                    exception.stderr.decode()
                    if isinstance(exception.stderr, bytes)
                    else exception.stderr or ""
                )
                status = "TIMEOUT"
                error = f"exceeded {timeout_seconds} seconds"
            elapsed = time.perf_counter() - started
            (trace_directory / "stdout.log").write_text(stdout, encoding="utf-8")
            (trace_directory / "stderr.log").write_text(stderr, encoding="utf-8")
            record = {
                "trace": str(trace.resolve()),
                "trace_sha256": _sha256(trace),
                "status": status,
                "matches": matches,
                "mismatches": mismatches,
                "return_code": return_code,
                "wall_seconds": elapsed,
                "error": error,
                "command": command,
            }
            _atomic_json(trace_directory / "result.json", record)
            records.append(record)

        summary = {
            "schema": "morpher_cycle_validation_v1",
            "simulator": str(simulator.resolve()),
            "simulator_sha256": _sha256(simulator),
            "bitstream": str(bitstream.resolve()),
            "bitstream_sha256": _sha256(bitstream),
            "allocation": str(allocation.resolve()),
            "allocation_sha256": _sha256(allocation),
            "dimensions": [x_dimension, y_dimension],
            "total_memory_size": total_memory_size,
            "memory_arrangement": memory_arrangement,
            "timeout_seconds_per_trace": timeout_seconds,
            "trace_count": len(records),
            "matched_trace_count": sum(r["status"] == "MATCH" for r in records),
            "mismatched_trace_count": sum(
                r["status"] == "MISMATCH" for r in records
            ),
            "timeout_trace_count": sum(r["status"] == "TIMEOUT" for r in records),
            "error_trace_count": sum(r["status"] == "ERROR" for r in records),
            "total_matches": sum(int(r["matches"] or 0) for r in records),
            "total_mismatches": sum(int(r["mismatches"] or 0) for r in records),
            "records": records,
        }
        _atomic_json(staging / "summary.json", summary)
        staging.rename(output_directory)
        return summary
    except BaseException:
        failed = output_directory.with_name(
            f"{output_directory.name}.failed.{os.getpid()}"
        )
        if staging.exists():
            shutil.move(str(staging), failed)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulator", type=Path, required=True)
    parser.add_argument("--bitstream", type=Path, required=True)
    parser.add_argument("--allocation", type=Path, required=True)
    parser.add_argument("--trace", type=Path, action="append", required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--x", type=int, default=4)
    parser.add_argument("--y", type=int, default=4)
    parser.add_argument("--total-memory-size", type=int, default=4096)
    parser.add_argument(
        "--memory-arrangement",
        type=int,
        choices=(1, 2),
        default=1,
        help="Morpher simulator convention: 1=left memory, 2=both sides",
    )
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    arguments = parser.parse_args()
    summary = run_corpus(
        simulator=arguments.simulator,
        bitstream=arguments.bitstream,
        allocation=arguments.allocation,
        traces=arguments.trace,
        output_directory=arguments.output_directory,
        x_dimension=arguments.x,
        y_dimension=arguments.y,
        total_memory_size=arguments.total_memory_size,
        memory_arrangement=arguments.memory_arrangement,
        timeout_seconds=arguments.timeout_seconds,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
