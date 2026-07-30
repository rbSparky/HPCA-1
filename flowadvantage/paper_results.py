"""Fail-closed aggregation for paper-grade atomic mapping results.

Workers are the only writers of experimental observations.  This module reads
their immutable JSON results, verifies them against a frozen CSV manifest, and
constructs paired tables without imputing operational failures as mapping
failures.  It intentionally has no dependency on the mapper implementation.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


BOOTSTRAP_SEED = 24072026
BOOTSTRAP_RESAMPLES = 10_000
DEFAULT_PAIR_FIELDS = (
    "kernel",
    "architecture",
    "seed",
    "budget_seconds",
    "initialization_depth_fraction",
    "replicate",
)

NORMAL_STATUSES = frozenset({"LEGAL_SUCCESS", "VALID_MAPPING_FAILURE"})
OPERATIONAL_STATUSES = frozenset(
    {
        "TIMEOUT",
        "ERROR",
        "UNSUPPORTED",
        "CANCELLED",
        "SOLVER_REJECTED",
        "VALIDATOR_FAILURE",
        "BUDGET_EXHAUSTED",
    }
)
NONTERMINAL_STATUSES = frozenset({"PENDING", "RUNNING", "RETRYABLE"})
ALL_STATUSES = NORMAL_STATUSES | OPERATIONAL_STATUSES | NONTERMINAL_STATUSES

_STATUS_ALIASES = {
    "DONE": "LEGAL_SUCCESS",
    "MAPPED": "LEGAL_SUCCESS",
}
_ATTEMPT_STATUSES = frozenset(
    {
        "ATTEMPTED",
        "MAPPED",
        "LEGAL_SUCCESS",
        "FAILED",
        "VALID_MAPPING_FAILURE",
        "TIMEOUT",
        "ERROR",
        "UNSUPPORTED",
        "CANCELLED",
        "SOLVER_REJECTED",
        "VALIDATOR_FAILURE",
        "BUDGET_EXHAUSTED",
    }
)
_IDENTITY_FIELDS = (
    "work_id",
    "kernel",
    "architecture",
    "method",
    "seed",
    "config_hash",
)
_HASH_FIELDS = (
    "config_hash",
    "source_commit",
    "source_tree_hash",
    "toolchain_hash",
    "dfg_hash",
    "architecture_hash",
    "checkpoint_hash",
)
_NONNEGATIVE_FIELDS = (
    "compile_wall_seconds",
    "cpu_seconds",
    "feature_seconds",
    "proposal_seconds",
    "relaxation_seconds",
    "native_mapper_seconds",
    "route_cost",
    "routing_attempts",
    "failed_routes",
    "backtracks",
    "expansions",
    "generated_actions",
    "parent_solves",
    "child_solves",
)


class AggregationError(ValueError):
    """An atomic result is unsafe to include in a scientific aggregate."""


def normalize_status(value: Any) -> str:
    status = str(value or "").strip().upper()
    status = _STATUS_ALIASES.get(status, status)
    if status not in ALL_STATUSES:
        raise AggregationError(f"unknown status {value!r}")
    return status


def _strict_bool(value: Any, *, field: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1"}:
            return True
        if normalized in {"false", "0"}:
            return False
    raise AggregationError(f"{field} must be an explicit Boolean, got {value!r}")


def _finite_number(value: Any, *, field: str, nonnegative: bool = False) -> float:
    if value is None or value == "":
        raise AggregationError(f"{field} is required")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise AggregationError(f"{field} must be numeric, got {value!r}") from error
    if not math.isfinite(number):
        raise AggregationError(f"{field} must be finite, got {value!r}")
    if nonnegative and number < 0:
        raise AggregationError(f"{field} must be nonnegative, got {number}")
    return number


def _first_present(payload: dict[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        if name in payload and payload[name] not in (None, ""):
            return payload[name]
    return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_scalar(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value).strip()


@dataclass(frozen=True)
class AtomicRecord:
    work_id: str
    kernel: str
    architecture: str
    method: str
    seed: str
    status: str
    success: bool | None
    minimum_ii: int | None
    result_path: Path | None
    result_sha256: str
    manifest: dict[str, str]
    payload: dict[str, Any]

    @property
    def normally_completed(self) -> bool:
        return self.status in NORMAL_STATUSES

    def value(self, field: str) -> Any:
        if field == "minimum_ii":
            return self.minimum_ii
        if field == "success":
            return self.success
        value = self.payload.get(field)
        if value in (None, ""):
            value = self.manifest.get(field)
        return value


def _resolve_result_path(manifest_path: Path, row: dict[str, str]) -> Path:
    local = manifest_path.parent / "work_items" / f"{row['work_id']}.json"
    configured = Path(row.get("result_path", "")) if row.get("result_path") else None
    if local.is_file():
        return local
    if configured is not None and configured.is_file():
        return configured
    return local


def _validate_legality(payload: dict[str, Any], policy: str) -> None:
    legal = _strict_bool(payload.get("legal"), field="legal")
    if not legal:
        raise AggregationError("LEGAL_SUCCESS requires legal=true")
    mapping_hash = _first_present(payload, ("mapping_hash",))
    document_hashes = payload.get("document_hashes")
    if mapping_hash is None and isinstance(document_hashes, dict):
        mapping_hash = document_hashes.get("mapping.json")
    if policy == "dual" and not str(mapping_hash or "").strip():
        raise AggregationError("paper-grade LEGAL_SUCCESS omitted mapping artifact hash")
    if policy == "legacy":
        return
    independent = _first_present(
        payload,
        (
            "flowadvantage_legality",
            "flow_legality",
            "FlowAdvantage_legality",
            "independent_legality",
        ),
    )
    native = _first_present(
        payload,
        ("morpher_legality", "native_legality", "Morpher_legality"),
    )
    if independent is None or native is None:
        raise AggregationError(
            "paper-grade LEGAL_SUCCESS requires independent and native legality evidence"
        )
    if not _strict_bool(independent, field="independent legality"):
        raise AggregationError("independent legality rejected LEGAL_SUCCESS")
    if not _strict_bool(native, field="native legality"):
        raise AggregationError("native legality rejected LEGAL_SUCCESS")


def _validate_attempts(
    payload: dict[str, Any],
    status: str,
    *,
    recorded_lower_bound: Any = None,
    require_attempts: bool = True,
) -> tuple[list[dict[str, Any]], int | None]:
    attempts = payload.get("ii_attempts", [])
    if attempts is None:
        attempts = []
    if not isinstance(attempts, list):
        raise AggregationError("ii_attempts must be a list")
    if require_attempts and status in NORMAL_STATUSES and not attempts:
        raise AggregationError(
            f"paper-grade {status} must preserve every attempted II"
        )
    normalized: list[dict[str, Any]] = []
    seen: set[int] = set()
    for index, attempt in enumerate(attempts):
        if not isinstance(attempt, dict):
            raise AggregationError(f"ii_attempts[{index}] must be an object")
        raw_status = str(attempt.get("status", "")).strip().upper()
        raw_status = "LEGAL_SUCCESS" if raw_status == "DONE" else raw_status
        if raw_status not in _ATTEMPT_STATUSES:
            raise AggregationError(
                f"ii_attempts[{index}] has unknown status {attempt.get('status')!r}"
            )
        try:
            ii = int(attempt["ii"])
        except (KeyError, TypeError, ValueError) as error:
            raise AggregationError(f"ii_attempts[{index}] has invalid II") from error
        if ii < 1 or ii in seen:
            raise AggregationError(f"II attempts must be unique positive integers: {ii}")
        if normalized and ii <= int(normalized[-1]["ii"]):
            raise AggregationError("II attempts must be strictly increasing")
        seen.add(ii)
        normalized.append({**attempt, "ii": ii, "status": raw_status})
    mapped = [
        int(attempt["ii"])
        for attempt in normalized
        if attempt["status"] in {"MAPPED", "LEGAL_SUCCESS"}
    ]
    if mapped:
        first_mapped_index = next(
            index
            for index, attempt in enumerate(normalized)
            if attempt["status"] in {"MAPPED", "LEGAL_SUCCESS"}
        )
        if first_mapped_index != len(normalized) - 1:
            raise AggregationError("II search continued after its first legal mapping")
    minimum_ii = min(mapped) if mapped else None
    top_level_ii = payload.get("ii")
    if status == "LEGAL_SUCCESS":
        if minimum_ii is None:
            if top_level_ii in (None, ""):
                raise AggregationError("LEGAL_SUCCESS has no successful II evidence")
            minimum_ii = int(top_level_ii)
        if top_level_ii not in (None, "") and int(top_level_ii) != minimum_ii:
            raise AggregationError(
                f"top-level II {top_level_ii} does not equal minimum mapped II {minimum_ii}"
            )
    elif minimum_ii is not None:
        raise AggregationError(f"{status} contains a successful II attempt")
    lower_bound = _first_present(payload, ("lower_bound_ii", "resource_recurrence_lower_bound"))
    if lower_bound is None:
        lower_bound = recorded_lower_bound
    if lower_bound is not None and normalized and int(lower_bound) != normalized[0]["ii"]:
        raise AggregationError("II search did not begin at the recorded lower bound")
    return normalized, minimum_ii


def load_atomic_records(
    manifest_path: Path,
    *,
    legality_policy: str = "dual",
) -> tuple[list[AtomicRecord], list[dict[str, Any]]]:
    """Read and verify a frozen manifest and its immutable atomic results.

    Nonterminal manifest rows may lack a result.  Every terminal row must have
    one, and manifest/payload statuses and hashes must agree.  This function
    raises before returning any records when an inconsistency is found.
    """

    if legality_policy not in {"dual", "legacy"}:
        raise ValueError("legality_policy must be 'dual' or 'legacy'")
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise AggregationError(f"empty manifest: {manifest_path}")
    required = {"work_id", "kernel", "architecture", "method", "seed", "status"}
    missing = required - set(rows[0])
    if missing:
        raise AggregationError(f"manifest missing columns {sorted(missing)}")
    ids = [row["work_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise AggregationError("manifest contains duplicate work_id values")

    records: list[AtomicRecord] = []
    attempt_rows: list[dict[str, Any]] = []
    result_hash_owners: dict[str, str] = {}
    semantic_owners: dict[tuple[str, ...], str] = {}
    for row in rows:
        manifest_status = normalize_status(row["status"])
        semantic_key = tuple(
            _canonical_scalar(row.get(field, ""))
            for field in (
                "kernel",
                "architecture",
                "method",
                "seed",
                "budget_seconds",
                "initialization_depth_fraction",
                "replicate",
                "initial_ii",
                "evaluation_mode",
            )
        )
        previous = semantic_owners.setdefault(semantic_key, row["work_id"])
        if previous != row["work_id"]:
            raise AggregationError(
                f"duplicate semantic work items {previous!r} and {row['work_id']!r}"
            )
        path = _resolve_result_path(manifest_path, row)
        if not path.is_file():
            if manifest_status in ALL_STATUSES - NONTERMINAL_STATUSES:
                raise AggregationError(
                    f"terminal work item {row['work_id']} has no atomic result"
                )
            records.append(
                AtomicRecord(
                    row["work_id"], row["kernel"], row["architecture"],
                    row["method"], row["seed"], manifest_status, None, None,
                    None, "", row, {},
                )
            )
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise AggregationError(f"invalid atomic result {path}: {error}") from error
        if not isinstance(payload, dict):
            raise AggregationError(f"atomic result {path} must contain an object")
        if legality_policy == "dual" and not str(payload.get("schema", "")).strip():
            raise AggregationError(f"paper-grade atomic result {path} omitted schema")
        result_status = normalize_status(payload.get("status"))
        if result_status != manifest_status:
            raise AggregationError(
                f"status mismatch for {row['work_id']}: manifest={manifest_status}, result={result_status}"
            )
        for field in _IDENTITY_FIELDS:
            expected = _canonical_scalar(row.get(field))
            observed = _canonical_scalar(payload.get(field))
            if expected and observed and expected != observed:
                raise AggregationError(
                    f"{field} mismatch for {row['work_id']}: {expected!r} != {observed!r}"
                )
            if field in {"work_id", "config_hash"} and expected and not observed:
                raise AggregationError(f"atomic result omitted required {field}")
        for field in _HASH_FIELDS:
            expected = _canonical_scalar(row.get(field))
            observed = _canonical_scalar(payload.get(field))
            if expected and observed and expected != observed:
                raise AggregationError(f"{field} mismatch for {row['work_id']}")
            if legality_policy == "dual" and expected and not observed:
                raise AggregationError(
                    f"paper-grade atomic result {row['work_id']} omitted frozen {field}"
                )
        result_hash = _sha256(path)
        owner = result_hash_owners.setdefault(result_hash, row["work_id"])
        if owner != row["work_id"]:
            raise AggregationError(
                f"identical atomic payload reused by {owner!r} and {row['work_id']!r}"
            )
        attempts, minimum_ii = _validate_attempts(
            payload,
            result_status,
            recorded_lower_bound=_first_present(
                row, ("lower_bound_ii", "resource_recurrence_lower_bound")
            ),
            require_attempts=legality_policy == "dual",
        )
        for index, attempt in enumerate(attempts):
            attempt_rows.append(
                {
                    "work_id": row["work_id"],
                    "kernel": row["kernel"],
                    "architecture": row["architecture"],
                    "method": row["method"],
                    "seed": row["seed"],
                    "attempt_index": index,
                    **attempt,
                }
            )
        success: bool | None = None
        if result_status == "LEGAL_SUCCESS":
            success = _strict_bool(payload.get("success"), field="success")
            if not success:
                raise AggregationError("LEGAL_SUCCESS requires success=true")
            _validate_legality(payload, legality_policy)
        elif result_status == "VALID_MAPPING_FAILURE":
            success = _strict_bool(payload.get("success"), field="success")
            if success:
                raise AggregationError("VALID_MAPPING_FAILURE requires success=false")
        elif payload.get("success") not in (None, "", False, 0, "false", "False"):
            raise AggregationError(f"operational status {result_status} cannot claim success")
        for field in _NONNEGATIVE_FIELDS:
            if payload.get(field) not in (None, ""):
                _finite_number(payload[field], field=field, nonnegative=True)
        records.append(
            AtomicRecord(
                row["work_id"], row["kernel"], row["architecture"],
                row["method"], row["seed"], result_status, success,
                minimum_ii, path, result_hash, row, payload,
            )
        )
    return records, attempt_rows


def _pair_key(record: AtomicRecord, fields: Sequence[str]) -> tuple[str, ...]:
    values = []
    for field in fields:
        value = record.value(field)
        values.append(_canonical_scalar(value))
    return tuple(values)


def paired_records(
    records: Iterable[AtomicRecord],
    baseline: str,
    candidate: str,
    *,
    pair_fields: Sequence[str] = DEFAULT_PAIR_FIELDS,
) -> list[tuple[AtomicRecord, AtomicRecord]]:
    """Return an exact intersection of normally completed paired work items."""

    by_method: dict[str, dict[tuple[str, ...], AtomicRecord]] = {
        baseline: {}, candidate: {}
    }
    for record in records:
        if record.method not in by_method or not record.normally_completed:
            continue
        key = _pair_key(record, pair_fields)
        if key in by_method[record.method]:
            raise AggregationError(
                f"multiple {record.method} rows share paired key {key}; add a replicate field"
            )
        by_method[record.method][key] = record
    common = sorted(set(by_method[baseline]) & set(by_method[candidate]))
    return [(by_method[baseline][key], by_method[candidate][key]) for key in common]


def paired_bootstrap(
    baseline_values: Sequence[float],
    candidate_values: Sequence[float],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float, float]:
    """Mean(candidate-baseline) and deterministic percentile interval."""

    if len(baseline_values) != len(candidate_values) or not baseline_values:
        raise AggregationError("paired bootstrap requires equal nonempty vectors")
    differences = [float(b) - float(a) for a, b in zip(baseline_values, candidate_values)]
    if any(not math.isfinite(value) for value in differences):
        raise AggregationError("paired bootstrap received a non-finite value")
    point = statistics.fmean(differences)
    rng = random.Random(seed)
    count = len(differences)
    samples = [
        statistics.fmean(differences[rng.randrange(count)] for _ in range(count))
        for _ in range(resamples)
    ]
    samples.sort()
    lower = samples[max(0, math.floor(0.025 * resamples))]
    upper = samples[min(resamples - 1, math.ceil(0.975 * resamples) - 1)]
    return point, lower, upper


def comparison_rows(
    records: Iterable[AtomicRecord],
    comparisons: Iterable[tuple[str, str]],
    *,
    pair_fields: Sequence[str] = DEFAULT_PAIR_FIELDS,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> list[dict[str, Any]]:
    records = list(records)
    output: list[dict[str, Any]] = []
    for baseline, candidate in comparisons:
        method_records = {
            baseline: [record for record in records if record.method == baseline],
            candidate: [record for record in records if record.method == candidate],
        }
        normal_keys: dict[str, set[tuple[str, ...]]] = {baseline: set(), candidate: set()}
        for record in records:
            if record.method in normal_keys and record.normally_completed:
                normal_keys[record.method].add(_pair_key(record, pair_fields))
        pairs = paired_records(
            records, baseline, candidate, pair_fields=pair_fields
        )
        metric_specs = (
            ("success", "all_normal_pairs", lambda r: float(bool(r.success))),
            ("failed_routes", "available_normal_pairs", lambda r: r.value("failed_routes")),
            ("compile_wall_seconds", "available_normal_pairs", lambda r: r.value("compile_wall_seconds")),
            ("minimum_ii", "common_successes", lambda r: r.minimum_ii),
            ("route_cost", "common_successes", lambda r: r.value("route_cost")),
        )
        for metric, scope, getter in metric_specs:
            selected: list[tuple[float, float]] = []
            for left, right in pairs:
                if scope == "common_successes" and not (left.success and right.success):
                    continue
                left_value, right_value = getter(left), getter(right)
                if left_value in (None, "") or right_value in (None, ""):
                    continue
                selected.append(
                    (
                        _finite_number(left_value, field=metric),
                        _finite_number(right_value, field=metric),
                    )
                )
            if not selected:
                continue
            baseline_values = [item[0] for item in selected]
            candidate_values = [item[1] for item in selected]
            point, lower, upper = paired_bootstrap(
                baseline_values,
                candidate_values,
                resamples=resamples,
                seed=seed,
            )
            output.append(
                {
                    "baseline": baseline,
                    "candidate": candidate,
                    "metric": metric,
                    "direction": "candidate_minus_baseline",
                    "scope": scope,
                    "paired_instances": len(selected),
                    "baseline_mean": statistics.fmean(baseline_values),
                    "candidate_mean": statistics.fmean(candidate_values),
                    "point_estimate": point,
                    "ci95_lower": lower,
                    "ci95_upper": upper,
                    "bootstrap_resamples": resamples,
                    "bootstrap_seed": seed,
                    "pair_fields": ",".join(pair_fields),
                    "normal_pair_intersection": len(pairs),
                    "baseline_normal_rows": len(normal_keys[baseline]),
                    "candidate_normal_rows": len(normal_keys[candidate]),
                    "baseline_only_normal_rows": len(
                        normal_keys[baseline] - normal_keys[candidate]
                    ),
                    "candidate_only_normal_rows": len(
                        normal_keys[candidate] - normal_keys[baseline]
                    ),
                    "baseline_manifest_rows": len(method_records[baseline]),
                    "candidate_manifest_rows": len(method_records[candidate]),
                    "baseline_operational_rows": sum(
                        record.status in OPERATIONAL_STATUSES
                        for record in method_records[baseline]
                    ),
                    "candidate_operational_rows": sum(
                        record.status in OPERATIONAL_STATUSES
                        for record in method_records[candidate]
                    ),
                }
            )
    return output


def status_census(records: Iterable[AtomicRecord]) -> list[dict[str, Any]]:
    counts = Counter((record.method, record.status) for record in records)
    return [
        {"method": method, "status": status, "count": count}
        for (method, status), count in sorted(counts.items())
    ]


def failure_census(records: Iterable[AtomicRecord]) -> list[dict[str, Any]]:
    counts = Counter()
    for record in records:
        if record.status == "LEGAL_SUCCESS" or record.status in NONTERMINAL_STATUSES:
            continue
        reason = _canonical_scalar(
            _first_present(record.payload, ("termination", "error_type", "error_message"))
        ) or "UNSPECIFIED"
        counts[(record.method, record.status, reason)] += 1
    return [
        {"method": method, "status": status, "reason": reason, "count": count}
        for (method, status, reason), count in sorted(counts.items())
    ]


def record_rows(records: Iterable[AtomicRecord]) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        rows.append(
            {
                **record.manifest,
                **record.payload,
                "work_id": record.work_id,
                "kernel": record.kernel,
                "architecture": record.architecture,
                "method": record.method,
                "seed": record.seed,
                "status": record.status,
                "success": record.success,
                "minimum_ii": record.minimum_ii,
                "result_path": str(record.result_path or ""),
                "result_sha256": record.result_sha256,
            }
        )
    return rows
