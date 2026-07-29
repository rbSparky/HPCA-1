"""Frozen FlowAdvantage proposal inference on Morpher's native DFG/MRRG.

This module is the zero-shot model boundary used by the real-kernel mapper.
It deliberately does not implement a child-relaxation proxy.  The selected
revision-v3 model is a *hybrid* proposal: a parent residual relaxation supplies
the analytical dual baseline and the frozen GNN predicts its finite-
perturbation correction.  Consequently, missing parent values are an error;
they are never replaced by zeros.

Morpher's complete native, II-expanded MRRG is encoded.  Native IDs and
directed edges remain authoritative; coordinates and phases are only
architecture-normalized attributes.  The frozen model was trained with
three modulo-phase channels, so arbitrary IIs are projected by piecewise-
linear interpolation on the normalized phase axis.  This exactly reproduces
the training one-hot representation for II=3 and avoids an architecture-name
or phase-number lookup table.
"""

from __future__ import annotations

from collections import OrderedDict, defaultdict, deque
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

import networkx as nx
import numpy as np
import torch

# Each atomic mapping worker is already the unit of parallelism.  Constraining
# PyTorch here prevents its intra-op pool from competing with CVXPY and other
# queue workers.  The environment is set by the worker before this module is
# imported; the explicit calls make the runtime contract verifiable.
torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", "1")))
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    if torch.get_num_interop_threads() != 1:
        raise

from quotientflow.flow_model import ResidualGNN, StandardScaler

from .native_beam_mapper import NativeActionScorer
from .native_mapper import (
    NativeAction,
    NativeMappingState,
    NativeMorpherProblem,
    NativeSignal,
)
from .operation_taxonomy import translate_operation


NATIVE_PROPOSAL_SCHEMA = "flowadvantage_native_proposal_v1"
PARENT_CONTEXT_SCHEMA = "flowadvantage_native_parent_relaxation_v1"
SELECTED_CHECKPOINT_SHA256 = (
    "468a8ffc541d20efcc77333e8492d4a1fcda88a022a506c9013b809fb991cad8"
)
SELECTED_MODEL_TYPE = "residual_gnn"
SELECTED_MODEL_SEED = 23
DEFAULT_CHECKPOINT = (
    Path(__file__).resolve().parents[2]
    / "results/revision_v3/checkpoints/residual_gnn_seed_23.pt"
)


class NativeProposalCompatibilityError(ValueError):
    """The native contract cannot be represented by the frozen model."""


class NativeNoParentUnavailable(NotImplementedError):
    """Raised when no solver-free checkpoint is available.

    The frozen revision-v3 checkpoint was trained with parent dual/objective
    inputs. Supplying zeros or reusing parent prices would silently turn it
    into a proxy, so the real pilot must fail closed until a true no-parent
    model is trained on declared development data.
    """


@dataclass(frozen=True)
class NativeParentRelaxationContext:
    """Exact parent-relaxation values consumed by the selected hybrid model.

    Routing duals are indexed by native MRRG *resource* IDs because Morpher
    enforces capacity on ports/resources rather than on an inferred physical
    coordinate edge.  Compute duals are indexed by native DataPath IDs.
    """

    state_hash: str
    architecture_hash: str
    dfg_hash: str
    ii: int
    objective: float
    routing_duals: Mapping[str, float]
    compute_duals: Mapping[str, float]
    capacity_slacks: Mapping[str, float]
    fractional_flow_concentration: float
    solver_status: str
    schema: str = PARENT_CONTEXT_SCHEMA

    def validate(
        self,
        problem: NativeMorpherProblem,
        state: NativeMappingState,
    ) -> None:
        expected_state = native_state_hash(state)
        failures = []
        if self.schema != PARENT_CONTEXT_SCHEMA:
            failures.append(f"schema={self.schema!r}")
        if self.state_hash != expected_state:
            failures.append("state hash")
        if self.architecture_hash != str(problem.mrrg.get("architecture_hash")):
            failures.append("architecture hash")
        if self.dfg_hash != str(problem.dfg.get("dfg_hash")):
            failures.append("DFG hash")
        if self.ii != problem.ii:
            failures.append("II")
        if not math.isfinite(float(self.objective)):
            failures.append("objective")
        if not math.isfinite(float(self.fractional_flow_concentration)):
            failures.append("fractional flow concentration")
        unknown_routing = set(self.routing_duals) - set(problem.resources)
        unknown_compute = set(self.compute_duals) - set(problem.dp_by_id)
        unknown_slack = set(self.capacity_slacks) - set(problem.resources)
        if unknown_routing:
            failures.append(f"unknown routing dual IDs ({len(unknown_routing)})")
        if unknown_compute:
            failures.append(f"unknown compute dual IDs ({len(unknown_compute)})")
        if unknown_slack:
            failures.append(f"unknown slack IDs ({len(unknown_slack)})")
        arrays = (
            list(self.routing_duals.values())
            + list(self.compute_duals.values())
            + list(self.capacity_slacks.values())
        )
        if any(not math.isfinite(float(value)) for value in arrays):
            failures.append("non-finite dual/slack")
        if any(float(value) < -1e-8 for value in self.routing_duals.values()):
            failures.append("negative routing dual")
        if any(float(value) < -1e-8 for value in self.compute_duals.values()):
            failures.append("negative compute dual")
        if failures:
            raise NativeProposalCompatibilityError(
                "invalid parent relaxation context: " + ", ".join(failures)
            )

    @property
    def semantic_hash(self) -> str:
        payload = {
            "schema": self.schema,
            "state_hash": self.state_hash,
            "architecture_hash": self.architecture_hash,
            "dfg_hash": self.dfg_hash,
            "ii": self.ii,
            "objective": float(self.objective),
            "routing_duals": sorted(
                (str(key), float(value))
                for key, value in self.routing_duals.items()
            ),
            "compute_duals": sorted(
                (str(key), float(value))
                for key, value in self.compute_duals.items()
            ),
            "capacity_slacks": sorted(
                (str(key), float(value))
                for key, value in self.capacity_slacks.items()
            ),
            "fractional_flow_concentration": float(
                self.fractional_flow_concentration
            ),
            "solver_status": str(self.solver_status),
        }
        return _sha_json(payload)


@runtime_checkable
class NativeParentContextProvider(Protocol):
    def parent_context(
        self,
        problem: NativeMorpherProblem,
        state: NativeMappingState,
    ) -> NativeParentRelaxationContext:
        """Return the exact cached/solved relaxation for ``state``."""


class NativeRelaxationParentContextProvider:
    """Adapt ``NativeRelaxationSolver`` to the frozen proposal contract."""

    def __init__(self, solver: Any) -> None:
        if not callable(getattr(solver, "solve", None)):
            raise TypeError("native relaxation provider requires solve(problem, state)")
        self.solver = solver
        self.last_result: Any | None = None
        self.calls = 0
        self.cache_hits = 0
        self.request_wall_seconds = 0.0
        self.cache_read_seconds = 0.0
        self.logical_solve_seconds = 0.0
        self.canonicalization_seconds = 0.0

    def parent_context(
        self,
        problem: NativeMorpherProblem,
        state: NativeMappingState,
    ) -> NativeParentRelaxationContext:
        self.calls += 1
        request_start = time.perf_counter()
        result = self.solver.solve(problem, state)
        request_seconds = time.perf_counter() - request_start
        self.last_result = result
        self.request_wall_seconds += request_seconds
        if bool(getattr(result, "cache_hit", False)):
            self.cache_hits += 1
            self.cache_read_seconds += request_seconds
        else:
            self.logical_solve_seconds += float(
                getattr(result, "solve_seconds", 0.0)
            )
            self.canonicalization_seconds += float(
                getattr(result, "canonicalization_seconds", 0.0)
            )
        if not bool(getattr(result, "feasible", False)):
            raise NativeProposalCompatibilityError(
                "parent native relaxation is not feasible: "
                f"status={getattr(result, 'status', 'unknown')!r}, "
                f"error={getattr(result, 'error', '')!r}"
            )
        context = NativeParentRelaxationContext(
            state_hash=native_state_hash(state),
            architecture_hash=str(problem.mrrg.get("architecture_hash")),
            dfg_hash=str(problem.dfg.get("dfg_hash")),
            ii=problem.ii,
            objective=float(result.objective),
            routing_duals=dict(result.routing_resource_duals),
            compute_duals=dict(result.compute_duals),
            capacity_slacks=dict(result.capacity_slacks),
            fractional_flow_concentration=float(
                result.fractional_flow_concentration
            ),
            solver_status=str(result.status),
        )
        context.validate(problem, state)
        return context


class NativeDualLinearActionScorer(NativeActionScorer):
    """First-order native dual baseline from one exact parent relaxation.

    The scorer uses native resource IDs and native DataPath IDs directly.  A
    repeated value on multiple fanout routes consumes one source/resource
    capacity; distinct source values consume separately.  The score is the
    prescribed immediate route cost plus parent compute/routing shadow prices.
    """

    name = "dual_linear_native"

    def __init__(self, parent_provider: NativeParentContextProvider) -> None:
        if not isinstance(parent_provider, NativeParentContextProvider):
            raise TypeError("dual-linear scorer requires a parent provider")
        self.parent_provider = parent_provider

    def score_actions(
        self,
        problem: NativeMorpherProblem,
        state: NativeMappingState,
        actions: Sequence[NativeAction],
    ) -> Sequence[float]:
        parent = self.parent_provider.parent_context(problem, state)
        parent.validate(problem, state)
        scores: list[float] = []
        for action in actions:
            immediate = float(
                sum(max(0, len(route.resource_ids) - 1) for route in action.routes)
            )
            # Count each source value once per native resource, preserving
            # Morpher broadcast semantics while charging distinct fanouts.
            consumed: set[tuple[str, str]] = set()
            routing = 0.0
            for route in action.routes:
                for resource_id in route.resource_ids:
                    key = (route.source_key, resource_id)
                    if key in consumed:
                        continue
                    consumed.add(key)
                    routing += float(parent.routing_duals.get(resource_id, 0.0))
            compute = float(parent.compute_duals.get(action.placement.dp_id, 0.0))
            scores.append(immediate + routing + compute)
        return tuple(scores)


@dataclass(frozen=True)
class NativeProposalTiming:
    compatibility_seconds: float
    context_seconds: float
    static_graph_seconds: float
    state_feature_seconds: float
    action_feature_seconds: float
    feature_seconds: float
    dfg_encoding_seconds: float
    mrrg_encoding_seconds: float
    action_head_seconds: float
    total_seconds: float
    actions: int
    state_cache_hit: bool


@dataclass(frozen=True)
class NativeCompatibilityReport:
    total_nodes: int
    arithmetic_compute_nodes: int
    fixed_terminal_nodes: int
    unsupported_compute_nodes: int
    unsupported_fraction: float
    unsupported: tuple[tuple[str, str], ...]

    @property
    def compatible(self) -> bool:
        return self.unsupported_fraction <= 0.10 + 1e-12


@dataclass
class _StaticGraphContext:
    node_keys: tuple[str, ...]
    node_index: dict[str, int]
    dfg_edges: torch.Tensor
    dfg_base: np.ndarray
    resource_ids: tuple[str, ...]
    resource_index: dict[str, int]
    mrrg_edges: torch.Tensor
    edge_pairs: tuple[tuple[str, str], ...]
    edge_index: dict[tuple[str, str], int]
    mrrg_base: np.ndarray
    edge_base: np.ndarray
    x_min: float
    x_span: float
    y_min: float
    y_span: float
    diameter: float
    edge_betweenness: np.ndarray
    bridges: frozenset[tuple[str, str]]


@dataclass
class _EncodedState:
    dfg_hidden: torch.Tensor
    mrrg_hidden: torch.Tensor
    context: "_NativeStateFeatureContext"


@dataclass
class _NativeStateFeatureContext:
    residual_adjacency: dict[str, tuple[str, ...]]
    occupied_resources: frozenset[str]
    occupied_edges: frozenset[tuple[str, str]]
    remaining_dependencies: tuple[Mapping[str, Any], ...]
    shortest_exposure: np.ndarray
    remaining_no_path: int
    remaining_cut_dependencies: int
    mean_residual_distance: float
    largest_component_fraction: float
    remaining_capacity_fraction: float
    slack_quantiles: np.ndarray


def _sha_json(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=str
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def native_state_hash(state: NativeMappingState) -> str:
    return hashlib.sha256(
        repr(state.stable_key()).encode("utf-8")
    ).hexdigest()


def _phase_projection(time_slot: int, ii: int) -> tuple[float, float, float]:
    """Project normalized modulo phase onto the checkpoint's three knots."""

    if ii <= 1:
        return (1.0, 0.0, 0.0)
    position = 2.0 * (int(time_slot) % ii) / (ii - 1)
    left = min(2, int(math.floor(position)))
    right = min(2, left + 1)
    fraction = position - left
    values = [0.0, 0.0, 0.0]
    values[left] += 1.0 - fraction
    values[right] += fraction
    return tuple(values)


def _operation_projection(opcode: str) -> tuple[float, float, str]:
    """Capability projection used by the frozen ADD/MUL encoder.

    General ALU/select operations share the non-multiply compute channel;
    multiply/MAC operations share the multiply channel.  Memory, constants,
    and predicate/I/O nodes are explicit zero-channel terminals.  An unknown
    capability is never mapped to either learned channel.
    """

    translated = translate_operation(opcode)
    capabilities = set(translated["capabilities"])
    if capabilities & {"load", "store", "constant", "predicate"}:
        return (0.0, 0.0, "fixed_terminal")
    if capabilities & {"multiply", "multiply_accumulate"}:
        return (0.0, 1.0, "mul_compatible")
    if capabilities & {
        "integer_alu",
        "floating_alu",
        "shift",
        "logic",
        "compare",
        "select_or_mux",
    }:
        return (1.0, 0.0, "add_compatible")
    return (0.0, 0.0, "unsupported_compute")


def compatibility_report(problem: NativeMorpherProblem) -> NativeCompatibilityReport:
    arithmetic = terminals = unsupported = 0
    details = []
    for key, node in problem.nodes.items():
        _, _, classification = _operation_projection(str(node.get("opcode", "")))
        if classification == "fixed_terminal":
            terminals += 1
        elif classification == "unsupported_compute":
            unsupported += 1
            details.append((key, str(node.get("opcode", ""))))
        else:
            arithmetic += 1
    compute_total = arithmetic + unsupported
    fraction = unsupported / max(1, compute_total)
    return NativeCompatibilityReport(
        total_nodes=len(problem.nodes),
        arithmetic_compute_nodes=arithmetic,
        fixed_terminal_nodes=terminals,
        unsupported_compute_nodes=unsupported,
        unsupported_fraction=float(fraction),
        unsupported=tuple(sorted(details)),
    )


class FrozenFlowAdvantageNativeProposalScorer(NativeActionScorer):
    """Batched frozen-checkpoint proposal scorer for native actions."""

    name = "flowadvantage_revision_v3_seed23_hybrid_native"

    def __init__(
        self,
        parent_provider: NativeParentContextProvider,
        *,
        checkpoint: str | Path = DEFAULT_CHECKPOINT,
        checkpoint_sha256: str = SELECTED_CHECKPOINT_SHA256,
        device: str | torch.device | None = None,
        max_state_cache_entries: int = 16,
    ) -> None:
        if not isinstance(parent_provider, NativeParentContextProvider):
            raise TypeError(
                "selected hybrid proposal requires a NativeParentContextProvider"
            )
        self.parent_provider = parent_provider
        self.checkpoint_path = Path(checkpoint).resolve()
        actual_hash = _sha_file(self.checkpoint_path)
        if actual_hash != checkpoint_sha256:
            raise NativeProposalCompatibilityError(
                "checkpoint SHA-256 mismatch: "
                f"expected {checkpoint_sha256}, observed {actual_hash}"
            )
        self.checkpoint_sha256 = actual_hash
        payload = torch.load(
            self.checkpoint_path, map_location="cpu", weights_only=False
        )
        self._validate_checkpoint(payload)
        metadata = payload["metadata"]
        self.feature_names = tuple(metadata["feature_names"])
        self.residual_center = float(metadata["residual_center"])
        self.residual_scale = float(metadata["residual_scale"])
        self.scaler = StandardScaler(
            mean=np.asarray(payload["scaler"]["mean"], dtype=np.float32),
            scale=np.asarray(payload["scaler"]["scale"], dtype=np.float32),
        )
        target_device = (
            torch.device(device)
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.device = target_device
        self.model = ResidualGNN(
            feature_dim=len(self.feature_names),
            hidden=64,
            include_duals=True,
        ).to(self.device)
        self.model.load_state_dict(payload["state_dict"], strict=True)
        self.model.eval()
        self.feature_schema_hash = _sha_json(
            {
                "schema": NATIVE_PROPOSAL_SCHEMA,
                "feature_names": self.feature_names,
                "phase_projection": "piecewise_linear_three_knots_v1",
                "operation_projection": "capability_add_mul_terminal_v1",
                "native_graph": "complete_ii_expanded_resource_graph_v1",
            }
        )
        self._static_cache: dict[tuple[str, str, int], _StaticGraphContext] = {}
        self._state_cache: OrderedDict[
            tuple[int, str, str], _EncodedState
        ] = OrderedDict()
        self.max_state_cache_entries = max(1, int(max_state_cache_entries))
        self.last_timing: NativeProposalTiming | None = None

    def _validate_checkpoint(self, payload: Mapping[str, Any]) -> None:
        required = {"model_type", "seed", "state_dict", "scaler", "metadata"}
        missing = required - set(payload)
        if missing:
            raise NativeProposalCompatibilityError(
                f"checkpoint fields missing: {sorted(missing)}"
            )
        if payload["model_type"] != SELECTED_MODEL_TYPE:
            raise NativeProposalCompatibilityError(
                f"expected {SELECTED_MODEL_TYPE}, got {payload['model_type']}"
            )
        if int(payload["seed"]) != SELECTED_MODEL_SEED:
            raise NativeProposalCompatibilityError(
                f"expected seed {SELECTED_MODEL_SEED}, got {payload['seed']}"
            )
        metadata = payload["metadata"]
        if metadata.get("model_type") != SELECTED_MODEL_TYPE:
            raise NativeProposalCompatibilityError("metadata model type mismatch")
        if int(metadata.get("seed", -1)) != SELECTED_MODEL_SEED:
            raise NativeProposalCompatibilityError("metadata seed mismatch")
        names = tuple(metadata.get("feature_names", ()))
        mean = np.asarray(payload["scaler"].get("mean"))
        scale = np.asarray(payload["scaler"].get("scale"))
        if not names or mean.shape != (len(names),) or scale.shape != mean.shape:
            raise NativeProposalCompatibilityError(
                "checkpoint feature/scaler dimensionality mismatch"
            )
        if not np.isfinite(mean).all() or not np.isfinite(scale).all():
            raise NativeProposalCompatibilityError("non-finite checkpoint scaler")
        if np.any(scale <= 0):
            raise NativeProposalCompatibilityError("non-positive checkpoint scale")
        if tuple(names) != tuple(metadata["feature_names"]):
            raise NativeProposalCompatibilityError("unstable checkpoint feature order")

    def score_actions(
        self,
        problem: NativeMorpherProblem,
        state: NativeMappingState,
        actions: Sequence[NativeAction],
    ) -> Sequence[float]:
        started = time.perf_counter()
        if not actions:
            self.last_timing = NativeProposalTiming(
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0, False
            )
            return ()
        if state.problem is not problem:
            raise ValueError("state belongs to a different native problem")
        compatibility_started = time.perf_counter()
        report = compatibility_report(problem)
        if not report.compatible:
            raise NativeProposalCompatibilityError(
                "unsupported compute fraction exceeds 10%: "
                f"{report.unsupported_compute_nodes}/"
                f"{report.arithmetic_compute_nodes + report.unsupported_compute_nodes}"
                f"={report.unsupported_fraction:.3%}; "
                f"examples={report.unsupported[:5]}"
            )
        operation_keys = {action.placement.node_key for action in actions}
        if len(operation_keys) != 1:
            raise ValueError(
                "one batched scorer call must contain candidates for one operation"
            )
        if any(state.apply_action(action) is None for action in actions):
            raise ValueError("proposal received an illegal native action")
        compatibility_seconds = time.perf_counter() - compatibility_started

        context_started = time.perf_counter()
        parent = self.parent_provider.parent_context(problem, state)
        parent.validate(problem, state)
        context_seconds = time.perf_counter() - context_started
        static_started = time.perf_counter()
        static = self._static_context(problem)
        static_seconds = time.perf_counter() - static_started
        cache_key = (id(problem), native_state_hash(state), parent.semantic_hash)
        encoded = self._state_cache.get(cache_key)
        state_cache_hit = encoded is not None
        dfg_seconds = mrrg_seconds = state_feature_seconds = 0.0
        if encoded is None:
            state_feature_started = time.perf_counter()
            feature_context = self._state_feature_context(
                problem, state, parent, static
            )
            state_feature_seconds = (
                time.perf_counter() - state_feature_started
            )
            dfg_started = time.perf_counter()
            dfg_hidden = self._encode_dfg(problem, state, static)
            dfg_seconds = time.perf_counter() - dfg_started
            mrrg_started = time.perf_counter()
            mrrg_hidden = self._encode_mrrg(
                problem, state, parent, static
            )
            mrrg_seconds = time.perf_counter() - mrrg_started
            encoded = _EncodedState(
                dfg_hidden=dfg_hidden,
                mrrg_hidden=mrrg_hidden,
                context=feature_context,
            )
            self._state_cache[cache_key] = encoded
            self._state_cache.move_to_end(cache_key)
            while len(self._state_cache) > self.max_state_cache_entries:
                self._state_cache.popitem(last=False)
        else:
            self._state_cache.move_to_end(cache_key)

        feature_started = time.perf_counter()
        rows, baselines = self._action_features(
            problem, state, actions, parent, static, encoded.context
        )
        values = np.asarray(
            [[row[name] for name in self.feature_names] for row in rows],
            dtype=np.float32,
        )
        if not np.isfinite(values).all():
            location = np.argwhere(~np.isfinite(values))[0]
            raise NativeProposalCompatibilityError(
                "non-finite native action feature "
                f"{self.feature_names[int(location[1])]}"
            )
        scaled = torch.as_tensor(
            self.scaler.transform(values),
            dtype=torch.float32,
            device=self.device,
        )
        action_feature_seconds = time.perf_counter() - feature_started
        feature_seconds = state_feature_seconds + action_feature_seconds

        action_started = time.perf_counter()
        with torch.inference_mode():
            correction_scaled = self._action_head(
                problem, actions, static, encoded, scaled
            )
            correction = (
                correction_scaled * self.residual_scale + self.residual_center
            )
            prediction = torch.as_tensor(
                baselines, dtype=torch.float32, device=self.device
            ) + correction
            scores = prediction.detach().cpu().double().numpy()
        action_seconds = time.perf_counter() - action_started
        if not np.isfinite(scores).all():
            raise NativeProposalCompatibilityError(
                "frozen proposal produced non-finite scores"
            )
        total = time.perf_counter() - started
        self.last_timing = NativeProposalTiming(
            compatibility_seconds=compatibility_seconds,
            context_seconds=context_seconds,
            static_graph_seconds=static_seconds,
            state_feature_seconds=state_feature_seconds,
            action_feature_seconds=action_feature_seconds,
            feature_seconds=feature_seconds,
            dfg_encoding_seconds=dfg_seconds,
            mrrg_encoding_seconds=mrrg_seconds,
            action_head_seconds=action_seconds,
            total_seconds=total,
            actions=len(actions),
            state_cache_hit=state_cache_hit,
        )
        return tuple(float(value) for value in scores)

    def _static_context(
        self, problem: NativeMorpherProblem
    ) -> _StaticGraphContext:
        key = (
            str(problem.dfg.get("dfg_hash")),
            str(problem.mrrg.get("architecture_hash")),
            problem.ii,
        )
        if key in self._static_cache:
            return self._static_cache[key]
        node_keys = tuple(sorted(problem.nodes))
        node_index = {value: index for index, value in enumerate(node_keys)}
        dfg_pairs = [
            (node_index[source], node_index[destination])
            for source, destination in sorted(problem.dependencies)
        ]
        dfg_edges = _edge_tensor(dfg_pairs, self.device)
        indegree = defaultdict(int)
        outdegree = defaultdict(int)
        for source, destination in problem.dependencies:
            outdegree[source] += 1
            indegree[destination] += 1
        max_level = max(
            (int(problem.nodes[key].get("asap", 0)) for key in node_keys),
            default=1,
        )
        dfg_base = np.zeros((len(node_keys), 11), dtype=np.float32)
        for index, node_key in enumerate(node_keys):
            node = problem.nodes[node_key]
            add, mul, _ = _operation_projection(str(node.get("opcode", "")))
            phase = int(node.get("asap", 0)) % problem.ii
            dfg_base[index, :8] = (
                add,
                mul,
                indegree[node_key] / max(1, len(node_keys) - 1),
                outdegree[node_key] / max(1, len(node_keys) - 1),
                int(node.get("asap", 0)) / max(1, max_level),
                *_phase_projection(phase, problem.ii),
            )

        resource_ids = tuple(sorted(problem.resources))
        resource_index = {
            value: index for index, value in enumerate(resource_ids)
        }
        edge_pairs = tuple(
            (source, destination)
            for source in resource_ids
            for destination in problem.adjacency.get(source, ())
        )
        numeric_edges = [
            (resource_index[source], resource_index[destination])
            for source, destination in edge_pairs
        ]
        mrrg_edges = _edge_tensor(numeric_edges, self.device)
        edge_index = {edge: index for index, edge in enumerate(edge_pairs)}
        reverse_adjacency: dict[str, list[str]] = defaultdict(list)
        for source, destination in edge_pairs:
            reverse_adjacency[destination].append(source)
        coordinates = [
            (
                float(resource.get("x")),
                float(resource.get("y")),
            )
            for resource in problem.resources.values()
            if isinstance(resource.get("x"), (int, float))
            and isinstance(resource.get("y"), (int, float))
        ]
        xs = [value[0] for value in coordinates] or [0.0]
        ys = [value[1] for value in coordinates] or [0.0]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        x_span, y_span = max(1.0, x_max - x_min), max(1.0, y_max - y_min)
        mrrg_base = np.zeros((len(resource_ids), 9), dtype=np.float32)
        for index, resource_id in enumerate(resource_ids):
            resource = problem.resources[resource_id]
            x = resource.get("x")
            y = resource.get("y")
            mrrg_base[index, 0] = (
                (float(x) - x_min) / x_span
                if isinstance(x, (int, float)) else 0.0
            )
            mrrg_base[index, 1] = (
                (float(y) - y_min) / y_span
                if isinstance(y, (int, float)) else 0.0
            )
            phase = int(resource.get("time_slot", 0) or 0)
            mrrg_base[index, 2] = phase / max(1, problem.ii - 1)
            mrrg_base[index, 3] = len(
                reverse_adjacency.get(resource_id, ())
            ) / max(1, len(resource_ids) - 1)
            mrrg_base[index, 4] = len(
                problem.adjacency.get(resource_id, ())
            ) / max(1, len(resource_ids) - 1)
            mrrg_base[index, 6] = float(
                isinstance(x, (int, float)) and float(x) in (x_min, x_max)
            )
            mrrg_base[index, 7] = float(
                isinstance(y, (int, float)) and float(y) in (y_min, y_max)
            )
        edge_base = np.zeros((len(edge_pairs), 9), dtype=np.float32)
        for index, (source, destination) in enumerate(edge_pairs):
            left, right = problem.resources[source], problem.resources[destination]
            lx, ly = left.get("x"), left.get("y")
            rx, ry = right.get("x"), right.get("y")
            dx = (
                float(rx) - float(lx)
                if isinstance(lx, (int, float))
                and isinstance(rx, (int, float))
                else 0.0
            )
            dy = (
                float(ry) - float(ly)
                if isinstance(ly, (int, float))
                and isinstance(ry, (int, float))
                else 0.0
            )
            phase_left = int(left.get("time_slot", 0) or 0)
            phase_right = int(right.get("time_slot", 0) or 0)
            wrap = abs(dx) > 1 or abs(dy) > 1
            diagonal = dx != 0 and dy != 0 and not wrap
            cardinal = (dx != 0) ^ (dy != 0) and not wrap
            wait_or_internal = not cardinal and not diagonal and not wrap
            edge_base[index, :4] = (
                float(wait_or_internal),
                float(cardinal),
                float(diagonal),
                float(wrap),
            )
            edge_base[index, 6] = dx / x_span
            edge_base[index, 7] = dy / y_span

        graph = nx.DiGraph()
        graph.add_nodes_from(resource_ids)
        graph.add_edges_from(edge_pairs)
        diameter = _bounded_directed_diameter(graph, resource_ids)
        if edge_pairs:
            sample_count = min(64, len(resource_ids))
            betweenness_map = nx.edge_betweenness_centrality(
                graph,
                k=sample_count,
                normalized=True,
                weight=None,
                seed=24072026,
            )
        else:
            betweenness_map = {}
        edge_betweenness = np.asarray(
            [betweenness_map.get(pair, 0.0) for pair in edge_pairs],
            dtype=np.float32,
        )
        undirected = graph.to_undirected()
        bridges = (
            frozenset(
                edge
                for left, right in nx.bridges(undirected)
                for edge in ((left, right), (right, left))
            )
            if undirected.number_of_edges()
            else frozenset()
        )
        result = _StaticGraphContext(
            node_keys=node_keys,
            node_index=node_index,
            dfg_edges=dfg_edges,
            dfg_base=dfg_base,
            resource_ids=resource_ids,
            resource_index=resource_index,
            mrrg_edges=mrrg_edges,
            edge_pairs=edge_pairs,
            edge_index=edge_index,
            mrrg_base=mrrg_base,
            edge_base=edge_base,
            x_min=x_min,
            x_span=x_span,
            y_min=y_min,
            y_span=y_span,
            diameter=max(1.0, float(diameter)),
            edge_betweenness=edge_betweenness,
            bridges=bridges,
        )
        self._static_cache[key] = result
        return result

    def _encode_dfg(
        self,
        problem: NativeMorpherProblem,
        state: NativeMappingState,
        static: _StaticGraphContext,
    ) -> torch.Tensor:
        values = static.dfg_base.copy()
        for key, placement in state.placements.items():
            index = static.node_index[key]
            values[index, 8] = 1.0
            values[index, 9] = (
                (float(placement.pe_x) - static.x_min) / static.x_span
                if placement.pe_x is not None else 0.0
            )
            values[index, 10] = (
                (float(placement.pe_y) - static.y_min) / static.y_span
                if placement.pe_y is not None else 0.0
            )
        hidden = self.model.dfg_input(
            torch.as_tensor(values, dtype=torch.float32, device=self.device)
        )
        for _ in range(self.model.steps):
            hidden = self.model.dfg_layer(hidden, static.dfg_edges)
        return hidden

    def _encode_mrrg(
        self,
        problem: NativeMorpherProblem,
        state: NativeMappingState,
        parent: NativeParentRelaxationContext,
        static: _StaticGraphContext,
    ) -> torch.Tensor:
        node_values = static.mrrg_base.copy()
        for resource_id in static.resource_ids:
            index = static.resource_index[resource_id]
            node_values[index, 5] = float(
                resource_id in state.resource_signals
                and bool(state.resource_signals[resource_id])
                or resource_id in state.dp_occupancy
            )
        positive_compute = np.asarray(
            [
                max(0.0, float(value))
                for value in parent.compute_duals.values()
                if float(value) > 1e-10
            ],
            dtype=float,
        )
        compute_scale = (
            max(float(np.median(positive_compute)), 1e-8)
            if len(positive_compute) else 1.0
        )
        for dp_id, dual in parent.compute_duals.items():
            if dp_id in static.resource_index:
                node_values[static.resource_index[dp_id], 8] = math.log1p(
                    max(0.0, float(dual)) / compute_scale
                )
        edge_values = static.edge_base.copy()
        occupied_edges = _occupied_edges(state)
        positive_route = np.asarray(
            [
                max(0.0, float(value))
                for value in parent.routing_duals.values()
                if float(value) > 1e-10
            ],
            dtype=float,
        )
        routing_scale = (
            max(float(np.median(positive_route)), 1e-8)
            if len(positive_route) else 1.0
        )
        for index, (source, destination) in enumerate(static.edge_pairs):
            occupied = (source, destination) in occupied_edges
            edge_values[index, 4] = float(occupied)
            edge_values[index, 5] = float(not occupied)
            edge_values[index, 8] = math.log1p(
                max(0.0, float(parent.routing_duals.get(destination, 0.0)))
                / routing_scale
            )
        hidden = self.model.arch_input(
            torch.as_tensor(node_values, dtype=torch.float32, device=self.device)
        )
        edge_tensor = torch.as_tensor(
            edge_values, dtype=torch.float32, device=self.device
        )
        for _ in range(self.model.steps):
            hidden = self.model.arch_layer(
                hidden, static.mrrg_edges, edge_tensor
            )
        return hidden

    def _state_feature_context(
        self,
        problem: NativeMorpherProblem,
        state: NativeMappingState,
        parent: NativeParentRelaxationContext,
        static: _StaticGraphContext,
    ) -> _NativeStateFeatureContext:
        occupied_resources = frozenset(
            resource_id
            for resource_id, signals in state.resource_signals.items()
            if signals
        )
        remaining = tuple(
            dependency
            for key, dependency in sorted(problem.dependencies.items())
            if key not in state.routes
            and bool(dependency.get(
                "requires_route",
                str(dependency.get("edge_type", "")).upper() != "PS",
            ))
        )
        remaining_sources = tuple(sorted({
            str(
                dependency.get(
                    "source_node_key", dependency.get("source_node")
                )
            )
            for dependency in remaining
        }))
        # A resource is residual when at least one remaining value can legally
        # occupy it under Morpher's broadcast/mutex/mux rules.  Treating every
        # occupied port as unavailable would incorrectly erase legal broadcast
        # paths.
        residual_resources = {
            resource_id
            for resource_id in static.resource_ids
            if not remaining_sources
            or any(
                state.can_occupy(
                    resource_id,
                    _feature_signal(problem, source_key),
                )
                for source_key in remaining_sources
            )
        }
        residual_adjacency = {
            source: tuple(
                destination
                for destination in problem.adjacency.get(source, ())
                if source in residual_resources
                and destination in residual_resources
            )
            for source in static.resource_ids
            if source in residual_resources
        }
        residual_graph = nx.DiGraph()
        residual_graph.add_nodes_from(residual_adjacency)
        residual_graph.add_edges_from(
            (source, destination)
            for source, values in residual_adjacency.items()
            for destination in values
        )
        components = list(nx.weakly_connected_components(residual_graph))
        largest_fraction = max(
            (len(component) for component in components), default=0
        ) / max(1, len(static.resource_ids))
        exposure = np.zeros(len(static.edge_pairs), dtype=np.float64)
        no_path = cut_dependencies = 0
        distances = []
        for dependency in remaining:
            source_key = str(
                dependency.get(
                    "source_node_key", dependency.get("source_node")
                )
            )
            signal = _feature_signal(problem, source_key)
            demand_adjacency = {
                source: tuple(
                    destination
                    for destination in problem.adjacency.get(source, ())
                    if state.can_occupy(source, signal)
                    and state.can_occupy(destination, signal)
                )
                for source in static.resource_ids
                if state.can_occupy(source, signal)
            }
            sources = _endpoint_resources(
                problem, state, dependency, source=True
            )
            sinks = _endpoint_resources(
                problem, state, dependency, source=False
            )
            path = _multi_endpoint_shortest_path(
                demand_adjacency, sources, frozenset(sinks)
            )
            if path is None:
                no_path += 1
                continue
            distances.append(max(0, len(path) - 1))
            path_edges = list(zip(path, path[1:]))
            if any(edge in static.bridges for edge in path_edges):
                cut_dependencies += 1
            for edge in path_edges:
                edge_id = static.edge_index.get(edge)
                if edge_id is not None:
                    exposure[edge_id] += 1.0
        exposure /= max(1, len(remaining))
        slacks = np.asarray(
            [
                max(0.0, float(parent.capacity_slacks.get(resource, 0.0)))
                for resource in static.resource_ids
            ],
            dtype=float,
        )
        return _NativeStateFeatureContext(
            residual_adjacency=residual_adjacency,
            occupied_resources=occupied_resources,
            occupied_edges=_occupied_edges(state),
            remaining_dependencies=remaining,
            shortest_exposure=exposure,
            remaining_no_path=no_path,
            remaining_cut_dependencies=cut_dependencies,
            mean_residual_distance=float(np.mean(distances)) if distances else 0.0,
            largest_component_fraction=float(largest_fraction),
            remaining_capacity_fraction=(
                len(residual_resources)
            ) / max(1, len(static.resource_ids)),
            slack_quantiles=(
                np.quantile(slacks, [0, .25, .5, .75, 1])
                if len(slacks) else np.zeros(5)
            ),
        )

    def _action_features(
        self,
        problem: NativeMorpherProblem,
        state: NativeMappingState,
        actions: Sequence[NativeAction],
        parent: NativeParentRelaxationContext,
        static: _StaticGraphContext,
        context: _NativeStateFeatureContext,
    ) -> tuple[list[dict[str, float]], np.ndarray]:
        operation_key = actions[0].placement.node_key
        node = problem.nodes[operation_key]
        add, mul, _ = _operation_projection(str(node.get("opcode", "")))
        new_resource_sets = [
            frozenset(
                resource
                for route in action.routes
                for resource in route.resource_ids
                if resource not in context.occupied_resources
            )
            for action in actions
        ]
        new_edge_sets = [
            frozenset(
                edge
                for route in action.routes
                for edge in zip(route.resource_ids, route.resource_ids[1:])
                if edge not in context.occupied_edges
            )
            for action in actions
        ]
        action_alternatives: dict[
            tuple[str, str, str, str], set[tuple[str, ...]]
        ] = defaultdict(set)
        for action in actions:
            for route in action.routes:
                key = (
                    route.source_key,
                    route.destination_key,
                    action.placement.dp_id,
                    route.edge_type,
                )
                action_alternatives[key].add(route.resource_ids)
        remaining_operations = len(problem.nodes) - len(state.placements) - 1
        remaining_dependencies = max(1, len(context.remaining_dependencies))
        total_edges = max(1, len(static.edge_pairs))
        total_resources = max(1, len(static.resource_ids))
        positive_duals = np.asarray(
            [
                max(0.0, float(value))
                for value in parent.routing_duals.values()
                if float(value) > 1e-10
            ],
            dtype=float,
        )
        top_threshold = (
            float(np.quantile(positive_duals, .9))
            if len(positive_duals) else math.inf
        )
        rows: list[dict[str, float]] = []
        baselines = []
        scarcity_values = []
        for action, resources, edges in zip(
            actions, new_resource_sets, new_edge_sets
        ):
            route_length = float(sum(
                max(0, len(route.resource_ids) - 1)
                for route in action.routes
            ))
            route_duals = np.asarray(
                [
                    max(0.0, float(parent.routing_duals.get(resource, 0.0)))
                    for resource in sorted(resources)
                ],
                dtype=float,
            )
            compute_dual = max(
                0.0,
                float(parent.compute_duals.get(action.placement.dp_id, 0.0)),
            )
            routing_component = float(route_duals.sum())
            baseline = route_length + routing_component + compute_dual
            baselines.append(baseline)
            edge_ids = [
                static.edge_index[edge]
                for edge in sorted(edges)
                if edge in static.edge_index
            ]
            betweenness = (
                static.edge_betweenness[edge_ids]
                if edge_ids else np.zeros(0)
            )
            exposure = (
                context.shortest_exposure[edge_ids]
                if edge_ids else np.zeros(0)
            )
            edge_duals = np.asarray(
                [
                    max(
                        0.0,
                        float(parent.routing_duals.get(edge[1], 0.0)),
                    )
                    for edge in sorted(edges)
                    if edge in static.edge_index
                ],
                dtype=float,
            )
            bridges = [
                float(edge in static.bridges) for edge in edges
            ]
            alt_counts = []
            inverse_disjoint = []
            for route in action.routes:
                alternatives = sorted(action_alternatives[(
                    route.source_key,
                    route.destination_key,
                    action.placement.dp_id,
                    route.edge_type,
                )])
                alt_counts.append(min(4, len(alternatives)) / 4.0)
                accepted: list[set[tuple[str, str]]] = []
                for alternative in alternatives:
                    candidate = set(zip(alternative, alternative[1:]))
                    if all(not candidate & previous for previous in accepted):
                        accepted.append(candidate)
                inverse_disjoint.append(1.0 / max(1, len(accepted)))
            alt_counts = alt_counts or [0.0]
            inverse_disjoint = inverse_disjoint or [0.0]
            active = route_duals > 1e-8
            replacement_count: dict[str, int] = {}
            for route in action.routes:
                alternatives = action_alternatives[(
                    route.source_key,
                    route.destination_key,
                    action.placement.dp_id,
                    route.edge_type,
                )]
                for resource in route.resource_ids:
                    replacement_count[resource] = sum(
                        resource not in alternative
                        for alternative in alternatives
                    )
            scarcity_component = sum(
                max(0.0, float(parent.routing_duals.get(resource, 0.0)))
                / max(1, replacement_count.get(resource, 0))
                for resource in resources
            )
            scarcity = route_length + compute_dual + scarcity_component
            scarcity_values.append(scarcity)
            row = {
                "immediate_cost": route_length,
                "route_length_normalized": route_length / static.diameter,
                "new_link_fraction": len(edges) / total_edges,
                "candidate_x": (
                    (float(action.placement.pe_x) - static.x_min) / static.x_span
                    if action.placement.pe_x is not None else 0.0
                ),
                "candidate_y": (
                    (float(action.placement.pe_y) - static.y_min) / static.y_span
                    if action.placement.pe_y is not None else 0.0
                ),
                "candidate_time": action.placement.modulo_time
                / max(1, problem.ii - 1),
                "operation_is_add": add,
                "operation_is_mul": mul,
                "operation_indegree_fraction": len(
                    problem.incoming.get(operation_key, ())
                ) / max(1, len(problem.nodes) - 1),
                "operation_outdegree_fraction": len(
                    problem.outgoing.get(operation_key, ())
                ) / max(1, len(problem.nodes) - 1),
                "partial_depth": len(state.placements) / max(1, len(problem.nodes)),
                "remaining_operation_fraction": remaining_operations
                / max(1, len(problem.nodes)),
                "occupied_link_fraction": len(context.occupied_edges) / total_edges,
                "remaining_routing_capacity_fraction":
                    context.remaining_capacity_fraction,
                "largest_component_fraction": context.largest_component_fraction,
                "saturated_link_fraction": (
                    len(context.occupied_resources | resources) / total_resources
                ),
                "remaining_no_path_fraction": context.remaining_no_path
                / remaining_dependencies,
                "cut_dependency_fraction": context.remaining_cut_dependencies
                / remaining_dependencies,
                "alternative_count_fraction_min": float(min(alt_counts)),
                "alternative_count_fraction_mean": float(np.mean(alt_counts)),
                "alternative_count_fraction_max": float(max(alt_counts)),
                "mean_residual_distance_normalized":
                    context.mean_residual_distance / static.diameter,
                "dual_sum_per_new_link": routing_component / max(1, len(resources)),
                "dual_max": float(route_duals.max()) if len(route_duals) else 0.0,
                "dual_top2_mean": (
                    float(np.sort(route_duals)[-2:].mean())
                    if len(route_duals) else 0.0
                ),
                "scarcity_adjusted_dual_mean": scarcity_component
                / max(1, len(resources)),
                "positive_dual_link_fraction": (
                    float(np.mean(route_duals > 1e-10))
                    if len(route_duals) else 0.0
                ),
                "top_dual_decile_fraction": (
                    float(np.mean(route_duals >= top_threshold))
                    if len(route_duals) else 0.0
                ),
                "selected_edge_betweenness_mean": (
                    float(np.mean(betweenness)) if len(betweenness) else 0.0
                ),
                "selected_edge_betweenness_max": (
                    float(np.max(betweenness)) if len(betweenness) else 0.0
                ),
                "selected_bridge_fraction": (
                    float(np.mean(bridges)) if bridges else 0.0
                ),
                "remaining_pair_shortest_path_exposure_mean": (
                    float(np.mean(exposure)) if len(exposure) else 0.0
                ),
                "remaining_pair_shortest_path_exposure_max": (
                    float(np.max(exposure)) if len(exposure) else 0.0
                ),
                "residual_alternative_path_fraction": float(np.mean(alt_counts)),
                "demand_to_min_cut_ratio_mean": float(np.mean(inverse_disjoint)),
                "demand_to_min_cut_ratio_max": float(max(inverse_disjoint)),
                "dual_weighted_path_congestion_mean": (
                    float(np.mean(edge_duals * (1.0 + exposure)))
                    if len(edge_duals) else 0.0
                ),
                "dual_weighted_path_congestion_max": (
                    float(np.max(edge_duals * (1.0 + exposure)))
                    if len(edge_duals) else 0.0
                ),
                "active_capacity_link_fraction": (
                    float(np.mean(active)) if len(active) else 0.0
                ),
                "parent_objective_per_remaining_dependency":
                    float(parent.objective) / remaining_dependencies,
                "parent_fractional_flow_concentration": float(
                    parent.fractional_flow_concentration
                ),
                "parent_slack_q0": float(context.slack_quantiles[0]),
                "parent_slack_q25": float(context.slack_quantiles[1]),
                "parent_slack_q50": float(context.slack_quantiles[2]),
                "parent_slack_q75": float(context.slack_quantiles[3]),
                "parent_slack_q100": float(context.slack_quantiles[4]),
                "compute_dual_target": compute_dual,
                "routing_dual_baseline_component": routing_component,
                "compute_dual_baseline_component": compute_dual,
                "dual_baseline": baseline,
            }
            rows.append(row)
        relative_sources = (
            ("immediate_cost", "immediate_cost"),
            ("dual_baseline", "dual_baseline"),
            ("route_length_normalized", "route_length"),
        )
        for source, prefix in relative_sources:
            _add_relative_features(rows, source, prefix)
        for index, row in enumerate(rows):
            row["_scarcity_cost"] = scarcity_values[index]
        _add_relative_features(rows, "_scarcity_cost", "scarcity_cost")
        for row in rows:
            row.pop("_scarcity_cost")
        expected = set(self.feature_names)
        for row in rows:
            if set(row) != expected:
                raise NativeProposalCompatibilityError(
                    "native/checkpoint feature schema mismatch: "
                    f"missing={sorted(expected - set(row))}, "
                    f"extra={sorted(set(row) - expected)}"
                )
        return rows, np.asarray(baselines, dtype=np.float64)

    def _action_head(
        self,
        problem: NativeMorpherProblem,
        actions: Sequence[NativeAction],
        static: _StaticGraphContext,
        encoded: _EncodedState,
        features: torch.Tensor,
    ) -> torch.Tensor:
        count = len(actions)
        operation_indices = torch.as_tensor(
            [
                static.node_index[action.placement.node_key]
                for action in actions
            ],
            dtype=torch.long,
            device=self.device,
        )
        target_indices = torch.as_tensor(
            [
                static.resource_index[action.placement.dp_id]
                for action in actions
            ],
            dtype=torch.long,
            device=self.device,
        )
        operation_h = encoded.dfg_hidden[operation_indices]
        target_h = encoded.mrrg_hidden[target_indices]
        dfg_mean = encoded.dfg_hidden.mean(0, keepdim=True).expand(count, -1)
        dfg_max = encoded.dfg_hidden.max(0).values.unsqueeze(0).expand(count, -1)
        arch_mean = encoded.mrrg_hidden.mean(0, keepdim=True).expand(count, -1)
        arch_max = (
            encoded.mrrg_hidden.max(0).values.unsqueeze(0).expand(count, -1)
        )
        src, dst = static.mrrg_edges
        edge_hidden = (
            encoded.mrrg_hidden[src] + encoded.mrrg_hidden[dst]
        ) / 2.0
        routed_mean = []
        routed_max = []
        for action in actions:
            indices = [
                static.edge_index[edge]
                for route in action.routes
                for edge in zip(route.resource_ids, route.resource_ids[1:])
                if edge in static.edge_index
            ]
            if indices:
                selected = edge_hidden[torch.as_tensor(
                    indices, dtype=torch.long, device=self.device
                )]
                routed_mean.append(selected.mean(0))
                routed_max.append(selected.max(0).values)
            else:
                routed_mean.append(torch.zeros_like(encoded.mrrg_hidden[0]))
                routed_max.append(torch.zeros_like(encoded.mrrg_hidden[0]))
        combined = torch.cat(
            (
                operation_h,
                target_h,
                dfg_mean,
                dfg_max,
                arch_mean,
                arch_max,
                torch.stack(routed_mean),
                torch.stack(routed_max),
                features,
            ),
            dim=1,
        )
        return self.model.head(combined).squeeze(-1)


def _edge_tensor(
    pairs: Sequence[tuple[int, int]], device: torch.device
) -> torch.Tensor:
    if not pairs:
        return torch.empty((2, 0), dtype=torch.long, device=device)
    return torch.as_tensor(pairs, dtype=torch.long, device=device).T.contiguous()


def _occupied_edges(state: NativeMappingState) -> frozenset[tuple[str, str]]:
    return frozenset(
        edge
        for route in state.routes.values()
        for edge in zip(route.resource_ids, route.resource_ids[1:])
    )


def _feature_signal(
    problem: NativeMorpherProblem, source_key: str
) -> NativeSignal:
    node = problem.nodes[source_key]
    return NativeSignal(
        source_key=source_key,
        destination_node=int(node["dfg_node_id"]),
        latency=0,
        source_node=int(node["dfg_node_id"]),
    )


def _bounded_directed_diameter(
    graph: nx.DiGraph, nodes: Sequence[str]
) -> float:
    if not nodes:
        return 1.0
    sample_count = min(64, len(nodes))
    positions = np.linspace(0, len(nodes) - 1, sample_count, dtype=int)
    diameter = 1
    for index in positions:
        lengths = nx.single_source_shortest_path_length(graph, nodes[int(index)])
        diameter = max(diameter, max(lengths.values(), default=0))
    return float(diameter)


def _endpoint_resources(
    problem: NativeMorpherProblem,
    state: NativeMappingState,
    dependency: Mapping[str, Any],
    *,
    source: bool,
) -> tuple[str, ...]:
    source_key = str(
        dependency.get("source_node_key", dependency.get("source_node"))
    )
    destination_key = str(
        dependency.get(
            "destination_node_key", dependency.get("destination_node")
        )
    )
    key = source_key if source else destination_key
    placement = state.placements.get(key)
    if placement is not None:
        if source:
            return (problem.output_port(placement),)
        edge_type = str(dependency.get("edge_type", "")).upper()
        return (problem.operand_port(placement, edge_type),)
    opcode = str(problem.nodes[key].get("opcode", "")).upper()
    resources = []
    for dp in problem.dp_records:
        if opcode not in dp["supported_operations"]:
            continue
        if source:
            resources.append(str(dp["output_port"]))
        else:
            edge_type = str(dependency.get("edge_type", "")).upper()
            candidate = f"{dp['dp_container']}.{edge_type}"
            if candidate in problem.resources:
                resources.append(candidate)
    return tuple(sorted(set(resources)))


def _multi_endpoint_shortest_path(
    adjacency: Mapping[str, Sequence[str]],
    sources: Sequence[str],
    targets: frozenset[str],
) -> tuple[str, ...] | None:
    valid_sources = tuple(sorted(source for source in sources if source in adjacency))
    if not valid_sources or not targets:
        return None
    queue = deque(valid_sources)
    previous: dict[str, str | None] = {source: None for source in valid_sources}
    goal = None
    while queue:
        current = queue.popleft()
        if current in targets:
            goal = current
            break
        for destination in adjacency.get(current, ()):
            if destination not in previous:
                previous[destination] = current
                queue.append(destination)
    if goal is None:
        return None
    path = []
    cursor: str | None = goal
    while cursor is not None:
        path.append(cursor)
        cursor = previous[cursor]
    return tuple(reversed(path))


def _add_relative_features(
    rows: Sequence[dict[str, float]], source: str, prefix: str
) -> None:
    values = np.asarray([row[source] for row in rows], dtype=float)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    ranks /= max(1, len(values) - 1)
    minimum = float(values.min())
    mean = float(values.mean())
    std = float(values.std())
    for index, row in enumerate(rows):
        row[f"{prefix}_percentile"] = float(ranks[index])
        row[f"{prefix}_from_min"] = float(values[index] - minimum)
        row[f"{prefix}_candidate_z"] = (
            float((values[index] - mean) / std) if std > 1e-8 else 0.0
        )


__all__ = [
    "DEFAULT_CHECKPOINT",
    "NativeDualLinearActionScorer",
    "FrozenFlowAdvantageNativeProposalScorer",
    "NATIVE_PROPOSAL_SCHEMA",
    "NativeCompatibilityReport",
    "NativeParentContextProvider",
    "NativeRelaxationParentContextProvider",
    "NativeParentRelaxationContext",
    "NativeProposalCompatibilityError",
    "NativeNoParentUnavailable",
    "NativeProposalTiming",
    "PARENT_CONTEXT_SCHEMA",
    "SELECTED_CHECKPOINT_SHA256",
    "compatibility_report",
    "native_state_hash",
]
