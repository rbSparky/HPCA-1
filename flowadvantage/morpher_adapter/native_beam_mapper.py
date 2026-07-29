"""Deterministic beam search over the exact Morpher-native mapping contract."""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Protocol, Sequence, runtime_checkable

from .legality_bridge import validate_mapping
from .native_mapper import (
    NativeAction,
    NativeMappingState,
    NativeMorpherProblem,
)


@dataclass(frozen=True)
class ScheduleHorizon:
    """Finite, explicit Morpher scheduling horizon.

    Morpher's PathFinder can delay operations beyond their reported ALAP while
    preserving modulo legality.  The default includes one full II beyond the
    larger of ASAP and ALAP, matching every placement in the authoritative
    array_add native reference.  This is an explicit bounded search policy,
    not a claim that the infinite modulo schedule space was exhausted.
    """

    extra_ii_periods: int = 1

    def bounds(
        self, problem: NativeMorpherProblem, node_key: str
    ) -> tuple[int, int]:
        if self.extra_ii_periods < 0:
            raise ValueError("extra_ii_periods must be nonnegative")
        node = problem.nodes[node_key]
        earliest = int(node.get("asap", 0))
        latest = max(earliest, int(node.get("alap", earliest)))
        latest += self.extra_ii_periods * problem.ii
        return earliest, latest


@runtime_checkable
class NativeActionScorer(Protocol):
    """Batch scorer interface used for length, proposal, and corrected search."""

    name: str

    def score_actions(
        self,
        problem: NativeMorpherProblem,
        state: NativeMappingState,
        actions: Sequence[NativeAction],
    ) -> Sequence[float]:
        """Return one finite lower-is-better score per action."""


@runtime_checkable
class ExactChildActionEvaluator(Protocol):
    """Strict interface for exact child-relaxation evaluation."""

    def evaluate_children(
        self,
        problem: NativeMorpherProblem,
        state: NativeMappingState,
        actions: Sequence[NativeAction],
    ) -> Sequence[float]:
        """Return exact relaxed child costs for every supplied action."""


@dataclass(frozen=True)
class LengthActionScorer:
    name: str = "native_length"

    def score_actions(
        self,
        problem: NativeMorpherProblem,
        state: NativeMappingState,
        actions: Sequence[NativeAction],
    ) -> Sequence[float]:
        del problem, state
        return tuple(
            float(
                sum(
                    max(0, len(route.resource_ids) - 1)
                    for route in action.routes
                )
            )
            for action in actions
        )


@dataclass(frozen=True)
class TopKExactRerankScorer:
    """Proposal pruning followed by a real exact evaluator.

    No fallback or proxy child value exists.  Construction or invocation fails
    when either component is unavailable or returns malformed scores.
    """

    proposal: NativeActionScorer
    child_evaluator: ExactChildActionEvaluator
    k: int = 4
    name: str = "native_topk_exact_rerank"

    def __post_init__(self) -> None:
        if self.k <= 0:
            raise ValueError("top-k exact reranking requires k > 0")
        if not isinstance(self.proposal, NativeActionScorer):
            raise TypeError("proposal does not implement NativeActionScorer")
        if not isinstance(self.child_evaluator, ExactChildActionEvaluator):
            raise TypeError(
                "child_evaluator does not implement ExactChildActionEvaluator"
            )

    def score_actions(
        self,
        problem: NativeMorpherProblem,
        state: NativeMappingState,
        actions: Sequence[NativeAction],
    ) -> Sequence[float]:
        proposal_scores = _checked_scores(
            self.proposal.score_actions(problem, state, actions),
            len(actions),
            component=f"{self.name}.proposal",
        )
        selected_indices = sorted(
            range(len(actions)),
            key=lambda index: (
                proposal_scores[index],
                actions[index].stable_key(),
            ),
        )[: min(self.k, len(actions))]
        selected = [actions[index] for index in selected_indices]
        exact = _checked_scores(
            self.child_evaluator.evaluate_children(problem, state, selected),
            len(selected),
            component=f"{self.name}.child_evaluator",
            permit_positive_infinity=True,
        )
        scores = [math.inf] * len(actions)
        for index, exact_score in zip(selected_indices, exact):
            scores[index] = exact_score
        return tuple(scores)


def _checked_scores(
    values: Sequence[float],
    expected: int,
    *,
    component: str,
    permit_positive_infinity: bool = False,
) -> tuple[float, ...]:
    if len(values) != expected:
        raise ValueError(
            f"{component} returned {len(values)} scores for {expected} actions"
        )
    result = tuple(float(value) for value in values)
    for value in result:
        if math.isnan(value) or value == -math.inf:
            raise ValueError(f"{component} returned invalid score {value}")
        if math.isinf(value) and not permit_positive_infinity:
            raise ValueError(f"{component} returned unavailable score {value}")
    return result


@dataclass(frozen=True)
class NativeBeamConfig:
    beam_width: int = 4
    k_paths: int = 4
    max_route_combinations_per_target: int | None = 16
    max_route_expansions_per_dependency: int = 1_000_000
    per_state_action_limit: int | None = None
    max_expansions: int | None = None
    schedule_horizon: ScheduleHorizon = ScheduleHorizon()

    def __post_init__(self) -> None:
        if self.beam_width <= 0:
            raise ValueError("beam_width must be positive")
        if self.k_paths <= 0:
            raise ValueError("k_paths must be positive")
        if (
            self.max_route_combinations_per_target is not None
            and self.max_route_combinations_per_target <= 0
        ):
            raise ValueError("route-combination limit must be positive")
        if self.max_route_expansions_per_dependency <= 0:
            raise ValueError("route expansion limit must be positive")
        if self.per_state_action_limit is not None and self.per_state_action_limit <= 0:
            raise ValueError("per-state action limit must be positive")
        if self.max_expansions is not None and self.max_expansions <= 0:
            raise ValueError("max_expansions must be positive")


@dataclass(frozen=True)
class NativeProgressEvent:
    sequence: int
    stage: str
    operation_key: str | None
    depth: int
    beam_states: int
    expansions: int
    generated_targets: int
    generated_actions: int
    routed_actions: int
    failed_targets: int
    duplicate_states: int
    elapsed_seconds: float
    stage_seconds: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class NativeBeamMetrics:
    success: bool
    legal: bool
    mapped_operations: int
    total_operations: int
    expansions: int
    generated_targets: int
    generated_actions: int
    routed_actions: int
    failed_targets: int
    duplicate_states: int
    scorer_calls: int
    beam_score: float | None
    route_cost: float | None
    elapsed_seconds: float
    termination: str
    legality_violations: tuple[dict[str, Any], ...]
    stage_seconds: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class NativeBeamResult:
    state: NativeMappingState | None
    mapping: dict[str, Any] | None
    metrics: NativeBeamMetrics


ProgressCallback = Callable[[NativeProgressEvent], None]
CancelCheck = Callable[[], bool]


class DeterministicNativeBeamMapper:
    """Schedule-aware deterministic beam mapper over native resources."""

    def __init__(
        self,
        problem: NativeMorpherProblem,
        *,
        scorer: NativeActionScorer,
        config: NativeBeamConfig | None = None,
        progress_callback: ProgressCallback | None = None,
        cancel_check: CancelCheck | None = None,
    ) -> None:
        if not isinstance(scorer, NativeActionScorer):
            raise TypeError(
                "a concrete NativeActionScorer is required; proposal or child "
                "relaxation scores are never synthesized"
            )
        self.problem = problem
        self.scorer = scorer
        self.config = config or NativeBeamConfig()
        self.progress_callback = progress_callback
        self.cancel_check = cancel_check

    def _emit(
        self,
        *,
        sequence: int,
        stage: str,
        operation_key: str | None,
        depth: int,
        beam_states: int,
        counters: dict[str, int],
        start: float,
        stage_seconds: dict[str, float] | None = None,
    ) -> None:
        if self.progress_callback is None:
            return
        self.progress_callback(
            NativeProgressEvent(
                sequence=sequence,
                stage=stage,
                operation_key=operation_key,
                depth=depth,
                beam_states=beam_states,
                expansions=counters["expansions"],
                generated_targets=counters["generated_targets"],
                generated_actions=counters["generated_actions"],
                routed_actions=counters["routed_actions"],
                failed_targets=counters["failed_targets"],
                duplicate_states=counters["duplicate_states"],
                elapsed_seconds=time.monotonic() - start,
                stage_seconds=dict(stage_seconds or {}),
            )
        )

    def map(
        self, initial_state: NativeMappingState | None = None
    ) -> NativeBeamResult:
        start = time.monotonic()
        state = initial_state.copy() if initial_state is not None else NativeMappingState(
            self.problem
        )
        if state.problem is not self.problem:
            raise ValueError("initial state belongs to a different native problem")
        counters = {
            "expansions": 0,
            "generated_targets": 0,
            "generated_actions": 0,
            "routed_actions": 0,
            "failed_targets": 0,
            "duplicate_states": 0,
            "scorer_calls": 0,
        }
        stage_seconds: dict[str, float] = {
            "candidate_generation": 0.0,
            "routing": 0.0,
            "scoring": 0.0,
            "legality": 0.0,
        }
        sequence = 0
        beam: list[tuple[float, tuple[Any, ...], NativeMappingState]] = [
            (0.0, state.stable_key(), state)
        ]
        operation_order = [
            key
            for key in self.problem.operation_order()
            if key not in state.placements
        ]
        self._emit(
            sequence=sequence,
            stage="START",
            operation_key=None,
            depth=len(state.placements),
            beam_states=1,
            counters=counters,
            start=start,
            stage_seconds=stage_seconds,
        )
        for operation_key in operation_order:
            if self.cancel_check is not None and self.cancel_check():
                return self._failure(
                    beam,
                    counters,
                    start,
                    "CANCELLED",
                    stage_seconds=stage_seconds,
                )
            successor_by_key: dict[
                tuple[Any, ...],
                tuple[float, tuple[Any, ...], NativeMappingState],
            ] = {}
            for cumulative_score, _, parent in beam:
                if (
                    self.config.max_expansions is not None
                    and counters["expansions"] >= self.config.max_expansions
                ):
                    return self._failure(
                        beam, counters, start, "MAX_EXPANSIONS",
                        stage_seconds=stage_seconds,
                    )
                counters["expansions"] += 1
                earliest, latest = self.config.schedule_horizon.bounds(
                    self.problem, operation_key
                )
                candidate_started = time.perf_counter()
                targets = self.problem.placement_candidates(
                    operation_key,
                    earliest_latency=earliest,
                    latest_latency=latest,
                    state=parent,
                )
                counters["generated_targets"] += len(targets)
                stage_seconds["candidate_generation"] += time.perf_counter() - candidate_started
                actions: list[NativeAction] = []
                for target in targets:
                    route_started = time.perf_counter()
                    routed = self.problem.actions_for_placement(
                        target,
                        parent,
                        k_paths=self.config.k_paths,
                        max_route_combinations=(
                            self.config.max_route_combinations_per_target
                        ),
                        max_route_expansions=(
                            self.config.max_route_expansions_per_dependency
                        ),
                    )
                    stage_seconds["routing"] += time.perf_counter() - route_started
                    if not routed:
                        counters["failed_targets"] += 1
                        continue
                    actions.extend(routed)
                actions.sort(key=NativeAction.stable_key)
                counters["generated_actions"] += len(actions)
                if not actions:
                    continue
                scoring_started = time.perf_counter()
                scores = _checked_scores(
                    self.scorer.score_actions(
                        self.problem, parent, actions
                    ),
                    len(actions),
                    component=self.scorer.name,
                    # Top-k exact correction deliberately marks unretained
                    # proposals +inf; they are filtered immediately below.
                    permit_positive_infinity=isinstance(
                        self.scorer, TopKExactRerankScorer
                    ),
                )
                stage_seconds["scoring"] += time.perf_counter() - scoring_started
                counters["scorer_calls"] += 1
                ranked = sorted(
                    zip(scores, actions),
                    key=lambda item: (item[0], item[1].stable_key()),
                )
                if self.config.per_state_action_limit is not None:
                    ranked = ranked[: self.config.per_state_action_limit]
                for local_score, action in ranked:
                    if not math.isfinite(local_score):
                        continue
                    child = parent.apply_action(action)
                    if child is None:
                        # An action was generated on exactly this parent, so a
                        # failed atomic replay is an implementation error.
                        raise RuntimeError(
                            "generated native action failed atomic replay"
                        )
                    counters["routed_actions"] += 1
                    key = child.stable_key()
                    candidate = (
                        cumulative_score + local_score,
                        action.stable_key(),
                        child,
                    )
                    previous = successor_by_key.get(key)
                    if previous is None or candidate[:2] < previous[:2]:
                        if previous is not None:
                            counters["duplicate_states"] += 1
                        successor_by_key[key] = candidate
                    else:
                        counters["duplicate_states"] += 1
            if not successor_by_key:
                return self._failure(
                    beam, counters, start, "NO_LEGAL_ACTION",
                    stage_seconds=stage_seconds,
                )
            beam = sorted(
                successor_by_key.values(),
                key=lambda item: (item[0], item[1], item[2].stable_key()),
            )[: self.config.beam_width]
            sequence += 1
            self._emit(
                sequence=sequence,
                stage="OPERATION_COMMITTED",
                operation_key=operation_key,
                depth=len(beam[0][2].placements),
                beam_states=len(beam),
                counters=counters,
                start=start,
                stage_seconds=stage_seconds,
            )

        legality_failures: list[dict[str, Any]] = []
        for score, _, complete_state in beam:
            mapping = complete_state.to_mapping()
            legality_started = time.perf_counter()
            legality = validate_mapping(
                mapping, self.problem.dfg, self.problem.mrrg
            )
            stage_seconds["legality"] += time.perf_counter() - legality_started
            if legality["legal"]:
                sequence += 1
                self._emit(
                    sequence=sequence,
                    stage="DONE",
                    operation_key=None,
                    depth=len(complete_state.placements),
                    beam_states=len(beam),
                    counters=counters,
                    start=start,
                )
                return NativeBeamResult(
                    state=complete_state,
                    mapping=mapping,
                    metrics=NativeBeamMetrics(
                        success=True,
                        legal=True,
                        mapped_operations=len(complete_state.placements),
                        total_operations=len(self.problem.nodes),
                        expansions=counters["expansions"],
                        generated_targets=counters["generated_targets"],
                        generated_actions=counters["generated_actions"],
                        routed_actions=counters["routed_actions"],
                        failed_targets=counters["failed_targets"],
                        duplicate_states=counters["duplicate_states"],
                        scorer_calls=counters["scorer_calls"],
                        beam_score=score,
                        route_cost=float(
                            sum(
                                max(0, len(route.resource_ids) - 1)
                                for route in complete_state.routes.values()
                            )
                        ),
                        elapsed_seconds=time.monotonic() - start,
                        termination="DONE",
                        legality_violations=(),
                        stage_seconds=dict(stage_seconds),
                    ),
                )
            legality_failures.extend(legality["violations"])
        return self._failure(
            beam,
            counters,
            start,
            "INDEPENDENT_LEGALITY_FAILURE",
            legality_failures,
            stage_seconds=stage_seconds,
        )

    def _failure(
        self,
        beam: Sequence[tuple[float, tuple[Any, ...], NativeMappingState]],
        counters: dict[str, int],
        start: float,
        termination: str,
        violations: Sequence[dict[str, Any]] = (),
        stage_seconds: dict[str, float] | None = None,
    ) -> NativeBeamResult:
        best_state = beam[0][2] if beam else None
        return NativeBeamResult(
            state=best_state,
            mapping=None,
            metrics=NativeBeamMetrics(
                success=False,
                legal=False,
                mapped_operations=(
                    len(best_state.placements) if best_state is not None else 0
                ),
                total_operations=len(self.problem.nodes),
                expansions=counters["expansions"],
                generated_targets=counters["generated_targets"],
                generated_actions=counters["generated_actions"],
                routed_actions=counters["routed_actions"],
                failed_targets=counters["failed_targets"],
                duplicate_states=counters["duplicate_states"],
                scorer_calls=counters["scorer_calls"],
                beam_score=None,
                route_cost=None,
                elapsed_seconds=time.monotonic() - start,
                termination=termination,
                legality_violations=tuple(dict(value) for value in violations),
                stage_seconds=dict(stage_seconds or {}),
            ),
        )
