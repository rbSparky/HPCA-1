#!/usr/bin/env python3
"""Execute exactly one revision-v5b real-pilot work item.

The queue owns the hard wall timeout and process-tree termination.  This
worker owns native/FlowAdvantage execution, progress heartbeats, schema
validation, and atomic publication of a single result.
"""

from __future__ import annotations

import argparse
import faulthandler
import hashlib
import json
import math
import os
import re
import signal
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

# These limits must be set before importing scientific libraries.
for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
    "TORCH_NUM_THREADS",
):
    os.environ[_name] = "1"

import psutil

from scripts.v5b_pilot_queue_common import (
    atomic_json,
    atomic_text,
    config_hash,
    fsync_directory,
    sha256_file,
    source_tree_hash,
)

ROOT = Path(__file__).resolve().parents[1]
PATHFINDER_METHODS = frozenset(
    {
        "native_pathfinder",
        "pathfinder",
        "native_simulated_annealing",
        "simulated_annealing",
        "native_lisa",
        "lisa",
    }
)
SA_METHODS = frozenset({"native_simulated_annealing", "simulated_annealing"})
# These names are deliberately explicit.  They are backed by the native
# relaxation and frozen checkpoint below; unknown names are rejected rather
# than silently mapped to a proxy.
FLOW_METHODS = frozenset({
    "dual_linear", "flow_proposal", "flow_top4", "full_relaxed_lookahead",
})
UNSUPPORTED_METHODS = frozenset({
    # Long-form canonical methods are wired below.  These legacy aliases are
    # retained only to fail closed rather than silently changing semantics.
    "dual", "proposal", "top4", "full",
    "noparent_proposal", "noparent_top4",
})
SUPPORTED_EXECUTORS = PATHFINDER_METHODS | frozenset({"length"}) | FLOW_METHODS


def _optional_float(value: Any, default: float) -> float:
    """Parse a manifest float while treating CSV empty cells as absent."""

    if value is None or (isinstance(value, str) and not value.strip()):
        return float(default)
    return float(value)


def _optional_bool(value: Any, default: bool) -> bool:
    """Parse booleans without the ``bool("False")`` manifest trap."""

    if value is None or (isinstance(value, str) and not value.strip()):
        return default
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"invalid boolean manifest value {value!r}")


def _prefix_depth(total_operations: int, depth_fraction: float) -> int:
    """Return the exact deterministic-prefix depth.

    A zero fraction is scientifically meaningful: it requests evaluation from
    the genuine empty mapping state.  Older pilot code forced every fraction
    through ``max(1, ...)``, so a nominal root-state experiment silently
    contained an anchored operation.  Keep the upper bound open because a
    prefix containing every operation would not leave a mapping experiment.
    """

    if total_operations <= 0:
        raise ValueError("native problem must contain at least one operation")
    if not 0.0 <= depth_fraction < 1.0:
        raise ValueError("initialization_depth_fraction must lie in [0,1)")
    if depth_fraction == 0.0:
        return 0
    return min(
        total_operations - 1,
        max(1, int(math.ceil(depth_fraction * total_operations))),
    )


def _resolve_mapper(spec: dict[str, Any]) -> Path:
    if spec.get("mapper_binary"):
        return Path(spec["mapper_binary"]).resolve()
    toolchain = Path(spec["toolchain_path"]).resolve()
    candidates = (
        toolchain / "bin/cgra_xml_mapper",
        toolchain / "source/Morpher_CGRA_Mapper/build/src/cgra_xml_mapper",
    )
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    raise FileNotFoundError(
        "no executable cgra_xml_mapper in immutable toolchain prefix "
        f"{toolchain}"
    )


def _preflight(spec: dict[str, Any]) -> tuple[Path | None, Path, Path]:
    method = str(spec["method"])
    if method in UNSUPPORTED_METHODS:
        raise NotImplementedError(
            "unsupported legacy alias or true no-parent checkpoint: refusing "
            "to synthesize a score"
        )
    if method not in SUPPORTED_EXECUTORS:
        raise ValueError(f"unknown pilot method {method!r}")
    dfg = Path(spec["dfg_path"]).resolve()
    architecture = Path(spec["architecture_path"]).resolve()
    for kind, path in (("DFG", dfg), ("architecture", architecture)):
        if not path.is_file():
            raise FileNotFoundError(f"{kind} input does not exist: {path}")
    if sha256_file(dfg) != spec["dfg_hash"]:
        raise ValueError("DFG hash changed after manifest freeze")
    if sha256_file(architecture) != spec["architecture_hash"]:
        raise ValueError("architecture hash changed after manifest freeze")
    strict_native_legality = _optional_bool(
        spec.get("paper_strict_legality"), False
    )
    mapper = (
        _resolve_mapper(spec)
        if method in PATHFINDER_METHODS or strict_native_legality
        else None
    )
    if mapper is not None and sha256_file(mapper) != spec["toolchain_hash"]:
        raise ValueError("native mapper binary hash changed after manifest freeze")
    if strict_native_legality:
        for kind, path_field, hash_field in (
            ("native DFG", "native_dfg_path", "native_dfg_hash"),
            (
                "native architecture",
                "native_architecture_path",
                "native_architecture_hash",
            ),
        ):
            path = Path(str(spec.get(path_field, ""))).resolve()
            if not path.is_file():
                raise FileNotFoundError(f"{kind} input does not exist: {path}")
            if sha256_file(path) != spec.get(hash_field):
                raise ValueError(f"{kind} hash changed after manifest freeze")
    if config_hash(spec) != spec["config_hash"]:
        raise ValueError("semantic work-item configuration hash mismatch")
    if source_tree_hash(ROOT) != spec["source_tree_hash"]:
        raise ValueError("pilot source tree changed after manifest freeze")
    return mapper, dfg, architecture


def _process_totals(process: psutil.Process) -> tuple[float, float]:
    cpu_seconds = 0.0
    rss_bytes = 0
    processes = [process]
    try:
        processes.extend(process.children(recursive=True))
    except psutil.Error:
        pass
    for value in processes:
        try:
            times = value.cpu_times()
            cpu_seconds += times.user + times.system
            rss_bytes += value.memory_info().rss
        except psutil.Error:
            pass
    return cpu_seconds, rss_bytes / 1024**2


def _native_pathfinder(
    spec: dict[str, Any],
    mapper: Path,
    dfg: Path,
    architecture: Path,
    artifact_directory: Path,
    progress: dict[str, Any],
) -> dict[str, Any]:
    scratch = artifact_directory.with_name(f".{artifact_directory.name}.native.tmp")
    if scratch.exists():
        raise FileExistsError(f"stale native scratch directory exists: {scratch}")
    scratch.mkdir(parents=True)
    export = scratch / "export"
    export.mkdir()
    is_sa = str(spec["method"]) in SA_METHODS
    command = [
        str(mapper),
        "-d",
        str(dfg),
        "-x",
        str(int(spec["x"])),
        "-y",
        str(int(spec["y"])),
        "-j",
        str(architecture),
        "-i",
        str(int(spec["initial_ii"])),
        "-t",
        str(spec["pe_type"]),
        "-m",
        "1" if is_sa else str(spec.get("native_method", 0)),
        "-r",
        str(int(spec.get("native_max_iter", 30))),
        "--max_II",
        str(
            int(
                spec.get("max_ii")
                or (int(spec["initial_ii"]) + int(spec.get("ii_delta_max", 4)) + 1)
            )
        ),
        "--dump-flowadvantage-state",
        str(export),
    ]
    if is_sa:
        command.extend(["--seed", str(int(spec["seed"]))])
    progress["stage"] = "native_pathfinder"
    started = time.monotonic()
    stdout_path = scratch / "native.stdout.log"
    stderr_path = scratch / "native.stderr.log"
    attempts: list[dict[str, Any]] = []
    success_ii: int | None = None
    current_ii: int | None = None
    with stdout_path.open("w", encoding="utf-8", buffering=1) as stdout_handle, (
        stderr_path.open("w", encoding="utf-8", buffering=1)
    ) as stderr_handle:
        process = subprocess.Popen(
            command,
            cwd=scratch,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=stderr_handle,
            text=True,
            bufsize=1,
            start_new_session=False,
        )
        assert process.stdout is not None
        for line in process.stdout:
            stdout_handle.write(line)
            print(line, end="", flush=True)
            match = re.search(r"Current target II =\s*(\d+)", line)
            if match:
                candidate_ii = int(match.group(1))
                if candidate_ii != current_ii:
                    current_ii = candidate_ii
                    attempts.append(
                        {
                            "ii": candidate_ii,
                            "status": "ATTEMPTED",
                            "started_offset_seconds": time.monotonic() - started,
                        }
                    )
                progress["ii"] = current_ii
            match = re.search(r"Map Success with II =\s*(\d+)", line)
            if match:
                success_ii = int(match.group(1))
                if attempts:
                    attempts[-1]["status"] = "MAPPED"
        return_code = process.wait()
        stdout_handle.flush()
        os.fsync(stdout_handle.fileno())
        stderr_handle.flush()
        os.fsync(stderr_handle.fileno())
    if return_code != 0:
        raise RuntimeError(f"native PathFinder exited with code {return_code}")
    if success_ii is None:
        # A normal native exit with explicit II attempts but no export is a
        # valid search failure.  It is not an infrastructure exception.
        if attempts and not any(export.glob("*.json")):
            return {
                "status": "VALID_MAPPING_FAILURE",
                "legal": False,
                "success": False,
                "termination": "NATIVE_SEARCH_EXHAUSTED",
                "ii_attempts": attempts,
                "compile_wall_seconds": time.monotonic() - started,
                "native_mapper_seconds": time.monotonic() - started,
                "command": command,
            }
        raise RuntimeError(
            "native mapper exited zero without a success marker or a valid "
            "search-exhaustion record"
        )
    documents: dict[str, dict[str, Any]] = {}
    hashes: dict[str, str] = {}
    for name in ("dfg.json", "mrrg.json", "mapping.json"):
        path = export / name
        if not path.is_file():
            raise RuntimeError(f"native success omitted required export {name}")
        documents[name] = json.loads(path.read_text(encoding="utf-8"))
        hashes[name] = sha256_file(path)
    mapping = documents["mapping.json"]
    if int(mapping["ii"]) != success_ii:
        raise RuntimeError("native success II does not match exported mapping II")
    progress["stage"] = "independent_legality"
    from flowadvantage.morpher_adapter.legality_bridge import validate_mapping

    legality = validate_mapping(
        mapping, documents["dfg.json"], documents["mrrg.json"]
    )
    if not legality["legal"]:
        raise RuntimeError(
            "native PathFinder export failed independent legality: "
            + json.dumps(legality["violations"][:5], sort_keys=True)
        )
    for path in (stdout_path, stderr_path, *(export / value for value in documents)):
        if path.is_file():
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
    os.replace(scratch, artifact_directory)
    fsync_directory(artifact_directory.parent)
    elapsed = time.monotonic() - started
    return {
        "status": "DONE",
        "legal": True,
        "flowadvantage_legality": True,
        "morpher_legality": True,
        "success": True,
        "termination": "DONE",
        "ii": success_ii,
        "ii_attempts": attempts,
        "operation_count": len(mapping.get("operations", [])),
        "route_count": len(mapping.get("routes", [])),
        "route_cost": sum(
            max(0, len(route.get("ordered_resource_ids", [])) - 1)
            for route in mapping.get("routes", [])
        ),
        "compile_wall_seconds": elapsed,
        "native_mapper_seconds": elapsed,
        "artifact_directory": str(artifact_directory),
        "document_hashes": hashes,
        "command": command,
    }


def _semantic_mapping_projection(mapping: dict[str, Any]) -> dict[str, Any]:
    """Return the native-contract fields that must survive a reimport."""

    operations = sorted(
        (
            str(item.get("native_node_key", item["dfg_node_id"])),
            str(item["pe_id"]),
            str(item["fu_id"]),
            int(item["modulo_time"]),
        )
        for item in mapping.get("operations", [])
    )
    routes = sorted(
        (
            str(item["edge_id"]),
            tuple(str(value) for value in item.get("ordered_resource_ids", [])),
            tuple(str(value) for value in item.get("ordered_link_ids", [])),
            int(item["start_time"]),
            int(item["end_time"]),
        )
        for item in mapping.get("routes", [])
    )
    return {"ii": int(mapping["ii"]), "operations": operations, "routes": routes}


def _native_reimport_validate(
    spec: dict[str, Any],
    mapper: Path,
    mapping_path: Path,
    artifact_directory: Path,
) -> dict[str, Any]:
    """Invoke Morpher's importer/checker and require semantic round-trip equality."""

    native_dfg = Path(spec["native_dfg_path"]).resolve()
    native_architecture = Path(spec["native_architecture_path"]).resolve()
    reimport_root = artifact_directory / "native_reimport"
    export = reimport_root / "export"
    if reimport_root.exists():
        raise FileExistsError(f"native reimport output already exists: {reimport_root}")
    export.mkdir(parents=True)
    command = [
        str(mapper),
        "-d",
        str(native_dfg),
        "-x",
        str(int(spec["x"])),
        "-y",
        str(int(spec["y"])),
        "-j",
        str(native_architecture),
        "-i",
        str(int(json.loads(mapping_path.read_text(encoding="utf-8"))["ii"])),
        "-t",
        str(spec["pe_type"]),
        "-m",
        "0",
        "--load-flowadvantage-mapping",
        str(mapping_path.resolve()),
        "--dump-flowadvantage-state",
        str(export.resolve()),
    ]
    started = time.monotonic()
    process = subprocess.run(
        command,
        cwd=reimport_root,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
    )
    elapsed = time.monotonic() - started
    atomic_text(reimport_root / "stdout.log", process.stdout)
    atomic_text(reimport_root / "stderr.log", process.stderr)
    if process.returncode != 0:
        raise RuntimeError(
            "native Morpher reimport failed with exit code "
            f"{process.returncode}: {process.stdout[-1000:]} {process.stderr[-1000:]}"
        )
    if "FlowAdvantage native legality PASS" not in process.stdout:
        raise RuntimeError("native Morpher reimport omitted legality PASS marker")
    roundtrip_path = export / "mapping.json"
    if not roundtrip_path.is_file():
        raise RuntimeError("native Morpher reimport omitted round-trip mapping.json")
    before = json.loads(mapping_path.read_text(encoding="utf-8"))
    after = json.loads(roundtrip_path.read_text(encoding="utf-8"))
    projection_before = _semantic_mapping_projection(before)
    projection_after = _semantic_mapping_projection(after)
    comparison = {
        "schema": "flowadvantage_native_reimport_validation_v1",
        "native_legality": True,
        "semantic_roundtrip_match": projection_before == projection_after,
        "ii_match": projection_before["ii"] == projection_after["ii"],
        "placement_match": projection_before["operations"] == projection_after["operations"],
        "route_match": projection_before["routes"] == projection_after["routes"],
        "wall_seconds": elapsed,
        "command": command,
        "roundtrip_mapping_hash": sha256_file(roundtrip_path),
    }
    atomic_json(reimport_root / "validation.json", comparison)
    if not comparison["semantic_roundtrip_match"]:
        raise RuntimeError(
            "native Morpher reimport changed placement, route, or II semantics"
        )
    return comparison


def _native_search(
    spec: dict[str, Any],
    mapper_binary: Path | None,
    dfg_path: Path,
    architecture_path: Path,
    artifact_directory: Path,
    progress: dict[str, Any],
) -> dict[str, Any]:
    # The Python mapper consumes the exact native DFG/MRRG dump.  Current
    # paper contracts carry the native per-FU operation-latency table, so the
    # reference mapping is verified for corpus provenance but is never passed
    # into the mapping problem or read for placement/routing semantics.
    dfg = json.loads(dfg_path.read_text(encoding="utf-8"))
    mrrg = json.loads(architecture_path.read_text(encoding="utf-8"))
    # Modern native exports carry the complete per-FU operation-latency table
    # in mrrg.json.  A reference mapping is optional provenance only and must
    # never be required to construct a problem or read for placements/routes.
    witness_value = str(spec.get("reference_mapping_path") or "").strip()
    if witness_value:
        witness_path = Path(witness_value)
        if not witness_path.is_file():
            raise FileNotFoundError(
                f"optional reference mapping does not exist: {witness_path}"
            )
        expected_witness_hash = str(spec.get("reference_mapping_hash") or "")
        if expected_witness_hash and sha256_file(witness_path) != expected_witness_hash:
            raise ValueError("native reference-mapping hash changed after freeze")
    from flowadvantage.morpher_adapter.native_beam_mapper import (
        DeterministicNativeBeamMapper,
        NativeActionScorer,
        LengthActionScorer,
        NativeBeamConfig,
        TopKExactRerankScorer,
    )
    from flowadvantage.morpher_adapter.native_mapper import (
        NativeMappingState,
        NativeMorpherProblem,
    )
    from flowadvantage.morpher_adapter.legality_bridge import validate_mapping

    problem = NativeMorpherProblem(dfg, mrrg)

    # Parent relaxations require a valid non-root state.  Construct that state
    # from the empty mapping with the deterministic length beam.  The native
    # witness is not read for placements, routes, occupancy, or latencies.
    initialization_policy = str(
        spec.get("initialization_policy", "deterministic_length_prefix")
    )
    if initialization_policy != "deterministic_length_prefix":
        raise ValueError(
            "only deterministic_length_prefix is executable; witness prefixes "
            "are prohibited for paper mapping experiments"
        )
    depth_fraction = float(spec.get("initialization_depth_fraction", 0.40))
    total_operations = len(problem.nodes)
    prefix_depth = _prefix_depth(total_operations, depth_fraction)
    prefix_started = time.monotonic()
    from flowadvantage.morpher_adapter.native_beam_mapper import (
        DeterministicNativeBeamMapper,
    )
    initial_state = NativeMappingState(problem)
    initial_frontier = None
    prefix_expansions = 0
    prefix_generated_actions = 0
    prefix_routing_attempts = 0
    prefix_stage_seconds: dict[str, float] = {}
    prefix_frontier_size = 1
    if prefix_depth:
        prefix_mapper = DeterministicNativeBeamMapper(
            problem,
            scorer=LengthActionScorer(),
            config=NativeBeamConfig(
                beam_width=int(spec.get("beam_width") or 4),
                k_paths=int(spec.get("k_paths") or 4),
                per_state_action_limit=(
                    int(spec["action_limit"]) if spec.get("action_limit") else None
                ),
                max_expansions=(
                    int(spec["max_expansions"])
                    if spec.get("max_expansions")
                    else None
                ),
                stop_after_mapped_operations=prefix_depth,
            ),
            progress_callback=lambda event: progress.update({
                "stage": "initialization_" + event.stage.lower(),
                "mapped_operations": event.depth,
                "expansions": event.expansions,
                "candidate_actions": event.generated_actions,
                "routing_attempts": event.routed_actions + event.failed_targets,
            }),
        )
        prefix_result = prefix_mapper.map(initial_state=initial_state)
        if (
            prefix_result.metrics.termination != "PARTIAL_DEPTH_REACHED"
            or prefix_result.state is None
        ):
            raise RuntimeError(
                "deterministic length-prefix initialization failed: "
                f"{prefix_result.metrics.termination}"
            )
        initial_state = prefix_result.state
        initial_frontier = prefix_result.frontier
        prefix_expansions = prefix_result.metrics.expansions
        prefix_generated_actions = prefix_result.metrics.generated_actions
        prefix_routing_attempts = (
            prefix_result.metrics.routed_actions
            + prefix_result.metrics.failed_targets
        )
        prefix_stage_seconds = {
            f"initialization_{key}": float(value)
            for key, value in prefix_result.metrics.stage_seconds.items()
        }
        prefix_frontier_size = len(initial_frontier)
    else:
        # Do not invoke a prefix mapper at all.  This guarantees that root
        # experiments have no hidden placement, route, occupancy, or beam
        # selection inherited from an initialization phase.
        progress.update({"stage": "initialization_empty_root"})
    initialization_seconds = time.monotonic() - prefix_started

    from flowadvantage.morpher_adapter.native_relaxation import (
        NativeExactChildEvaluator,
        NativeRelaxationConfig,
        NativeRelaxationSolver,
    )
    from flowadvantage.morpher_adapter.native_proposal import (
        FrozenFlowAdvantageNativeProposalScorer,
        NativeDualLinearActionScorer,
        NativeRelaxationParentContextProvider,
    )

    method = str(spec["method"])
    solver = None
    scorer: NativeActionScorer
    timing_provider = None
    child_evaluator = None
    parent_provider = None
    if method == "length":
        scorer = LengthActionScorer()
    else:
        solver = NativeRelaxationSolver(
            NativeRelaxationConfig(
                tau=_optional_float(spec.get("relaxation_tau"), 1e-3),
                max_solve_seconds=_optional_float(
                    spec.get("relaxation_timeout"), 120.0
                ),
                reachable_edge_pruning=_optional_bool(
                    spec.get("reachable_edge_pruning"), True
                ),
            ),
            cache_dir=Path(
                spec.get("relaxation_cache_dir")
                or "results/revision_v5b/cache/native_relaxation"
            ),
        )
        parent_provider = NativeRelaxationParentContextProvider(solver)
        if method == "dual_linear":
            scorer = NativeDualLinearActionScorer(parent_provider)
            timing_provider = None
        else:
            proposal = FrozenFlowAdvantageNativeProposalScorer(
                parent_provider,
                checkpoint=Path(
                    spec.get(
                        "checkpoint_path",
                        "results/revision_v3/checkpoints/residual_gnn_seed_23.pt",
                    )
                ),
                checkpoint_sha256=str(
                    spec.get(
                        "checkpoint_hash",
                        "468a8ffc541d20efcc77333e8492d4a1fcda88a022a506c9013b809fb991cad8",
                    )
                ),
                device=spec.get("device"),
            )
            timing_provider = proposal
            if method == "flow_proposal":
                scorer = proposal
            else:
                child_evaluator = NativeExactChildEvaluator(
                    solver,
                    parallelism=int(spec.get("child_parallelism") or 1),
                )
                if method == "flow_top4":
                    scorer = TopKExactRerankScorer(proposal, child_evaluator, k=4)
                elif method == "full_relaxed_lookahead":
                    class _FullExactScorer:
                        name = "full_relaxed_lookahead"
                        def score_actions(self, problem, state, actions):
                            values = [
                                float(value)
                                for value in child_evaluator.evaluate_children(
                                    problem, state, actions
                                )
                            ]
                            finite = [value for value in values if math.isfinite(value)]
                            if not finite:
                                return tuple(1.0 for _ in values)
                            lo, hi = min(finite), max(finite)
                            margin = max(1.0, 0.25 * max(0.0, hi - lo))
                            penalty = hi + margin
                            return tuple(penalty if not math.isfinite(value) else value for value in values)
                    scorer = _FullExactScorer()
                else:
                    raise ValueError(f"unsupported native FlowAdvantage method {method!r}")

    def callback(event: Any) -> None:
        progress.update(
            {
                "stage": event.stage,
                "mapped_operations": event.depth,
                "expansions": event.expansions,
                "candidate_actions": event.generated_actions,
                "routing_attempts": event.routed_actions + event.failed_targets,
                "parent_solves": int(
                    getattr(parent_provider, "calls", 0)
                ),
                "child_solves": int(
                    getattr(child_evaluator, "total_evaluations", 0)
                ),
                "cache_hits": int(getattr(solver, "cache_hits", 0)),
                "cache_misses": int(getattr(solver, "cache_misses", 0)),
                "scorer_rejected_states": event.scorer_rejected_states,
                "scorer_rejection_reasons": event.scorer_rejection_reasons,
            }
        )

    started = time.monotonic()
    mapper = DeterministicNativeBeamMapper(
        problem,
        scorer=scorer,
        config=NativeBeamConfig(
            beam_width=int(spec.get("beam_width") or 4),
            k_paths=int(spec.get("k_paths") or 4),
            per_state_action_limit=(
                int(spec["action_limit"]) if spec.get("action_limit") else None
            ),
            max_expansions=(
                int(spec["max_expansions"]) if spec.get("max_expansions") else None
            ),
        ),
        progress_callback=callback,
    )
    result = (
        mapper.map(initial_frontier=initial_frontier)
        if initial_frontier is not None
        else mapper.map(initial_state=initial_state)
    )
    elapsed = time.monotonic() - started
    total_elapsed = initialization_seconds + elapsed
    combined_stage_seconds = {
        **prefix_stage_seconds,
        **{
            f"completion_{key}": float(value)
            for key, value in result.metrics.stage_seconds.items()
        },
    }
    payload = {
        "status": "DONE" if result.metrics.success else "VALID_MAPPING_FAILURE",
        "legal": result.metrics.legal,
        "success": result.metrics.success,
        "termination": result.metrics.termination,
        "ii": problem.ii,
        "ii_attempts": [
            {
                "ii": problem.ii,
                "status": "MAPPED" if result.metrics.success else "FAILED",
                "started_offset_seconds": 0.0,
            }
        ],
        "mapped_operations": result.metrics.mapped_operations,
        "operation_count": result.metrics.total_operations,
        "route_cost": result.metrics.route_cost,
        "expansions": prefix_expansions + result.metrics.expansions,
        "generated_actions": (
            prefix_generated_actions + result.metrics.generated_actions
        ),
        "routing_attempts": (
            prefix_routing_attempts
            + result.metrics.routed_actions
            + result.metrics.failed_targets
        ),
        "completion_routing_attempts": (
            result.metrics.routed_actions + result.metrics.failed_targets
        ),
        "compile_wall_seconds": total_elapsed,
        "feature_seconds": 0.0,
        "proposal_seconds": 0.0,
        "relaxation_seconds": 0.0,
        "initialization_policy": initialization_policy,
        "initialization_depth_fraction": depth_fraction,
        "initialization_mapped_operations": prefix_depth,
        "initialization_seconds": initialization_seconds,
        "initialization_expansions": prefix_expansions,
        "initialization_generated_actions": prefix_generated_actions,
        "initialization_routing_attempts": prefix_routing_attempts,
        "initialization_frontier_size": prefix_frontier_size,
        "scorer_rejected_states": result.metrics.scorer_rejected_states,
        "scorer_rejection_reasons": result.metrics.scorer_rejection_reasons,
        "stage_seconds": combined_stage_seconds,
        "prefix_action_universe_hash": (
            prefix_mapper.action_universe_hash if prefix_depth else ""
        ),
        "completion_action_universe_hash": mapper.action_universe_hash,
        "parent_solves": int(getattr(parent_provider, "calls", 0)),
        "parent_cache_hits": int(
            getattr(parent_provider, "cache_hits", 0)
        ),
        "child_solves": int(
            getattr(child_evaluator, "total_evaluations", 0)
        ),
        "child_cache_hits": int(
            getattr(child_evaluator, "total_cache_hits", 0)
        ),
        "relaxation_cache_hits": int(getattr(solver, "cache_hits", 0)),
        "relaxation_cache_misses": int(getattr(solver, "cache_misses", 0)),
        "relaxation_structure_cache_hits": int(
            getattr(solver, "structure_cache_hits", 0)
        ),
        "relaxation_structure_cache_misses": int(
            getattr(solver, "structure_cache_misses", 0)
        ),
    }
    action_hash = hashlib.sha256()
    for component in (
        payload["prefix_action_universe_hash"],
        payload["completion_action_universe_hash"],
    ):
        encoded = component.encode("ascii")
        action_hash.update(len(encoded).to_bytes(8, "big"))
        action_hash.update(encoded)
    payload["action_universe_hash"] = action_hash.hexdigest()
    if timing_provider is not None and timing_provider.last_timing is not None:
        timing = timing_provider.last_timing
        payload.update({
            "feature_seconds": timing.feature_seconds,
            "proposal_seconds": timing.total_seconds,
            "gnn_encoding_seconds": timing.dfg_encoding_seconds + timing.mrrg_encoding_seconds,
            "gnn_action_head_seconds": timing.action_head_seconds,
        })
    if child_evaluator is not None:
        payload.update({
            "child_relaxation_seconds": child_evaluator.total_solve_seconds,
            "child_relaxation_wall_seconds": (
                child_evaluator.total_batch_wall_seconds
            ),
            "child_relaxation_logical_request_seconds": (
                child_evaluator.total_request_wall_seconds
            ),
            "child_parallelism": child_evaluator.parallelism,
            "child_cache_read_seconds": (
                child_evaluator.total_cache_read_seconds
            ),
        })
    payload["parent_relaxation_seconds"] = float(
        getattr(parent_provider, "logical_solve_seconds", 0.0)
    )
    payload["parent_canonicalization_seconds"] = float(
        getattr(parent_provider, "canonicalization_seconds", 0.0)
    )
    payload["parent_relaxation_wall_seconds"] = float(
        getattr(parent_provider, "request_wall_seconds", 0.0)
    )
    payload["parent_cache_read_seconds"] = float(
        getattr(parent_provider, "cache_read_seconds", 0.0)
    )
    payload["relaxation_seconds"] = (
        payload["parent_relaxation_seconds"]
        + float(payload.get("child_relaxation_seconds", 0.0))
    )
    if result.mapping is not None:
        artifact_directory.mkdir(parents=True)
        atomic_json(artifact_directory / "mapping.json", result.mapping)
        # Publish the complete canonical bundle, not just a mapper-local
        # object.  Consumers can independently re-run the bridge checker and
        # compare hashes without relying on process memory.
        atomic_json(artifact_directory / "dfg.json", dfg)
        atomic_json(artifact_directory / "mrrg.json", mrrg)
        legality = validate_mapping(result.mapping, dfg, mrrg)
        if not legality.get("legal", False):
            raise RuntimeError(
                "FlowAdvantage mapping export failed independent legality: "
                + json.dumps(legality.get("violations", [])[:5], sort_keys=True)
            )
        atomic_json(artifact_directory / "legality.json", legality)
        payload["flowadvantage_legality"] = True
        payload["artifact_directory"] = str(artifact_directory)
        payload["mapping_hash"] = sha256_file(artifact_directory / "mapping.json")
        if _optional_bool(spec.get("paper_strict_legality"), False):
            if mapper_binary is None:
                raise RuntimeError(
                    "strict paper legality requested without native mapper binary"
                )
            progress["stage"] = "native_reimport_legality"
            native_validation = _native_reimport_validate(
                spec,
                mapper_binary,
                artifact_directory / "mapping.json",
                artifact_directory,
            )
            payload["morpher_legality"] = bool(
                native_validation["native_legality"]
                and native_validation["semantic_roundtrip_match"]
            )
            payload["native_reimport_seconds"] = float(
                native_validation["wall_seconds"]
            )
        else:
            payload["morpher_legality"] = False
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    arguments = parser.parse_args()
    spec = json.loads(arguments.spec.read_text(encoding="utf-8"))
    heartbeat_path = Path(spec["heartbeat_path"])
    result_path = Path(spec["result_path"])
    artifact_directory = Path(spec["artifact_directory"])
    started_wall = time.time()
    started_monotonic = time.monotonic()
    own_process = psutil.Process()
    progress: dict[str, Any] = {
        "stage": "preflight",
        "mapped_operations": 0,
        "expansions": 0,
        "candidate_actions": 0,
        "routing_attempts": 0,
        "parent_solves": 0,
        "child_solves": 0,
        "cache_hits": 0,
        "cache_misses": 0,
        "ii": None,
    }
    stop_heartbeat = threading.Event()

    def heartbeat() -> None:
        while not stop_heartbeat.is_set():
            cpu_seconds, rss_mb = _process_totals(own_process)
            atomic_json(
                heartbeat_path,
                {
                    **progress,
                    "schema": "flowadvantage_v5b_heartbeat_v1",
                    "work_id": spec["work_id"],
                    "pid": os.getpid(),
                    "timestamp": time.time(),
                    "wall_seconds": time.monotonic() - started_monotonic,
                    "cpu_seconds": cpu_seconds,
                    "rss_mb": rss_mb,
                },
            )
            stop_heartbeat.wait(15.0)

    faulthandler.enable()
    faulthandler.register(signal.SIGUSR1, all_threads=True)
    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    identity = {
        "schema": "flowadvantage_v5b_pilot_result_v1",
        "work_id": spec.get("work_id"),
        "kernel": spec.get("kernel"),
        "architecture": spec.get("architecture"),
        "method": spec.get("method"),
        "seed": int(spec.get("seed") or 0),
        "config_hash": spec.get("config_hash"),
        "source_commit": spec.get("source_commit", ""),
        "source_tree_hash": spec.get("source_tree_hash", ""),
        "toolchain_hash": spec.get("toolchain_hash", ""),
        "dfg_hash": spec.get("dfg_hash", ""),
        "architecture_hash": spec.get("architecture_hash", ""),
        "checkpoint_hash": spec.get("checkpoint_hash", ""),
        "started_at": started_wall,
    }
    try:
        mapper, dfg, architecture = _preflight(spec)
        if result_path.exists():
            raise FileExistsError(f"refusing to overwrite atomic result {result_path}")
        if artifact_directory.exists():
            raise FileExistsError(
                f"refusing to overwrite artifact directory {artifact_directory}"
            )
        if spec["method"] in PATHFINDER_METHODS:
            assert mapper is not None
            row = _native_pathfinder(
                spec, mapper, dfg, architecture, artifact_directory, progress
            )
        else:
            row = _native_search(
                spec, mapper, dfg, architecture, artifact_directory, progress
            )
        row.update(
            {
                "schema": "flowadvantage_v5b_pilot_result_v1",
                "work_id": spec["work_id"],
                "kernel": spec["kernel"],
                "architecture": spec["architecture"],
                "method": spec["method"],
                "seed": int(spec["seed"]),
                "config_hash": spec["config_hash"],
                "source_commit": spec["source_commit"],
                "source_tree_hash": spec["source_tree_hash"],
                "toolchain_hash": spec["toolchain_hash"],
                "dfg_hash": spec["dfg_hash"],
                "architecture_hash": spec["architecture_hash"],
                "checkpoint_hash": spec.get("checkpoint_hash", ""),
                "started_at": started_wall,
                "finished_at": time.time(),
            }
        )
        if row["status"] not in {"DONE", "VALID_MAPPING_FAILURE"}:
            raise RuntimeError(f"executor returned invalid status {row['status']}")
        atomic_json(result_path, row)
        return 0
    except NotImplementedError as error:
        atomic_json(
            result_path,
            {
                **identity,
                "status": "UNSUPPORTED",
                "error_type": type(error).__name__,
                "error_message": str(error),
                "finished_at": time.time(),
            },
        )
        return 3
    except RuntimeError as error:
        # A deterministic prefix can legitimately have no legal successor on
        # a constrained native MRRG.  This is a completed mapping attempt, not
        # an infrastructure exception: preserve the exact termination reason
        # so coverage accounting can distinguish it from a worker error.
        message = str(error)
        if "deterministic length-prefix initialization failed: NO_LEGAL_ACTION" in message:
            atomic_json(
                result_path,
                {
                    **identity,
                    "status": "VALID_MAPPING_FAILURE",
                    "success": False,
                    "legal": False,
                    "termination": "NO_LEGAL_ACTION",
                    "failure_stage": "initialization",
                    "error_type": "ValidMappingFailure",
                    "error_message": message,
                    "finished_at": time.time(),
                },
            )
            return 0
        atomic_json(
            result_path,
            {
                **identity,
                "status": "ERROR",
                "error_type": type(error).__name__,
                "error_message": message,
                "finished_at": time.time(),
            },
        )
        raise
    except BaseException as error:
        atomic_json(
            result_path,
            {
                **identity,
                "status": "ERROR",
                "error_type": type(error).__name__,
                "error_message": str(error),
                "finished_at": time.time(),
            },
        )
        raise
    finally:
        progress["stage"] = "terminal"
        stop_heartbeat.set()
        thread.join(timeout=2.0)


if __name__ == "__main__":
    raise SystemExit(main())
