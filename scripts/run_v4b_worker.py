#!/usr/bin/env python3
"""One atomic FlowAdvantage mapping work item.

The process deliberately owns exactly one instance/method/budget.  It writes a
heartbeat independently of mapper internals, then atomically publishes a small
JSON result only after schema validation.
"""
from __future__ import annotations

import argparse
import faulthandler
import json
import os
import pickle
import threading
import time
from pathlib import Path

# Must precede numpy/torch/cvxpy imports.
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
              "BLIS_NUM_THREADS", "TORCH_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import psutil
import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

ROOT = Path(__file__).resolve().parents[1]


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temp.open("w") as handle:
        json.dump(value, handle, sort_keys=True, default=str)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    args = parser.parse_args()
    spec = json.loads(Path(args.spec).read_text())
    heartbeat_path = Path(spec["heartbeat_path"])
    result_path = Path(spec["result_path"])
    started = time.time()
    process = psutil.Process()
    progress = {"stage": "initializing", "mapped_operations": 0,
                "expansions": 0, "candidate_actions": 0,
                "parent_solves": 0, "child_solves": 0,
                "cache_hits": 0, "cache_misses": 0}
    stop = threading.Event()

    def beat() -> None:
        while not stop.is_set():
            cpu = process.cpu_times()
            rss = process.memory_info().rss / 1024**2
            atomic_json(heartbeat_path, {**progress, "pid": os.getpid(),
                "wall_seconds": time.time() - started,
                "cpu_seconds": cpu.user + cpu.system, "rss_mb": rss,
                "timestamp": time.time()})
            stop.wait(5.0)

    faulthandler.enable()
    heartbeat_thread = threading.Thread(target=beat, daemon=True)
    heartbeat_thread.start()
    try:
        # Import only after thread controls have been set.
        from scripts.run_v3_mappers import (  # noqa: PLC0415
            BUDGETS, FlowScorer, FrozenV2Scorer, base_row, cached_parent,
            load_selected_flow_models,
        )
        from quotientflow.mapper import map_dfg  # noqa: PLC0415
        from quotientflow.flow_advantage import dual_baseline, target_complete_path_bounded  # noqa: PLC0415
        from quotientflow.action_model import load_action_model  # noqa: PLC0415
        from quotientflow.model import QuotientFlowGNN, predict_numpy  # noqa: PLC0415
        from scripts.run_revision2_scarcity_suite import adjust as scarcity_adjust  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415

        progress["stage"] = "instance_loading"
        with (ROOT / spec["instance_path"]).open("rb") as handle:
            instance = pickle.load(handle)
        method, budget_name = spec["method"], spec["budget"]
        budget = BUDGETS[budget_name]
        progress["stage"] = "model_loading"
        started_mapper = time.perf_counter()
        if method == "length":
            progress["stage"] = "mapping"
            result = map_dfg(instance["dfg"], instance["arch"], instance["state"],
                k_paths=4, timeout_seconds=float(spec["mapper_timeout_seconds"]),
                **budget)
            stats = {}
        elif method == "dual_linear":
            class FastDualScorer:
                def __init__(self): self.parent_solves=0; self.parent_cache_hits=0; self.child_solves=0; self.child_cache_hits=0; self.feature_ms=0.; self.gnn_ms=0.; self.actual_feature_ms=0.; self.actual_relaxation_ms=0.; self.relaxation_ms=0.
                def preselect(self,d,a,s,o,actions): return target_complete_path_bounded(actions,4)
                def __call__(self,d,a,s,o,actions):
                    parent,hit,_,actual=cached_parent(d,a,s); self.parent_solves+=1; self.parent_cache_hits+=int(hit); self.actual_relaxation_ms+=actual; self.relaxation_ms+=1000*parent.solve_seconds
                    if not parent.solved: return np.full(len(actions),1e6)
                    return np.asarray([parent.objective + dual_baseline(x,parent)[0] for x in actions],float)
                def stats(self): return {'parent_solves':self.parent_solves,'parent_cache_hits':self.parent_cache_hits,'child_solves':0,'child_cache_hits':0,'relaxation_time_ms':self.relaxation_ms,'actual_relaxation_time_ms':self.actual_relaxation_ms,'feature_computation_ms':0.,'actual_feature_computation_ms':0.,'gnn_inference_ms':0.}
            scorer=FastDualScorer()
            progress["stage"]="mapping"
            result=map_dfg(instance["dfg"],instance["arch"],instance["state"],k_paths=4,timeout_seconds=float(spec["mapper_timeout_seconds"]),action_scorer=scorer,action_preselector=scorer.preselect,action_value_is_cost_to_go=True,**budget)
            stats=scorer.stats()
        else:
            flow_models, _ = load_selected_flow_models("cpu")
            if method == "revision_v2_action_gnn":
                model, scaler, model_type, _ = load_action_model(
                    ROOT / "results/revision_v2/checkpoints/action_gnn_seed_37.pt", "cpu")
                parent, _, _, _ = cached_parent(instance["dfg"], instance["arch"], instance["state"])
                sample = {**instance, "normalized_prices": parent.normalized_prices if parent.solved else np.zeros(len(instance["arch"].edge_list))}
                scorer = FrozenV2Scorer(sample, model, scaler, model_type)
            elif method == "revision_v1_pred_link":
                import torch as _torch
                checkpoint = _torch.load(ROOT / "results/quick_v1/checkpoints/model_seed_11.pt", map_location="cpu", weights_only=False)
                model = QuotientFlowGNN(); model.load_state_dict(checkpoint["state_dict"])
                prices = predict_numpy(model, instance["dfg"], instance["arch"], instance["state"])[0]
                sample = {**instance, "normalized_prices": np.zeros(len(instance["arch"].edge_list))}
                parent, _, _, _ = cached_parent(instance["dfg"], instance["arch"], instance["state"])
                sample["normalized_prices"] = parent.normalized_prices if parent.solved else sample["normalized_prices"]
                progress["stage"] = "mapping"
                result = map_dfg(instance["dfg"], instance["arch"], instance["state"],
                    prices=scarcity_adjust(sample, prices), price_weight=.5, price_mode="sum",
                    k_paths=4, timeout_seconds=float(spec["mapper_timeout_seconds"]), **budget)
                stats = {}
                row = base_row(instance, method, budget_name, result, stats, started_mapper)
                atomic_json(result_path, {"schema_version": 1, "work_id": spec["work_id"], "row": row})
                return 0
            else:
                if method in ("selected_v3_top4", "hybrid_top4"):
                    flow_method = "residual_gnn_top4"
                elif method == "full_relaxed_lookahead":
                    # FlowScorer's exhaustive relaxed oracle is named
                    # full_action_oracle internally; keep the v4b external
                    # terminology explicit in manifests and reports.
                    flow_method = "full_action_oracle"
                else:
                    flow_method = method
                scorer = FlowScorer(flow_method, flow_models, {})
            progress["stage"] = "mapping"
            result = map_dfg(instance["dfg"], instance["arch"], instance["state"],
                k_paths=4, timeout_seconds=float(spec["mapper_timeout_seconds"]),
                action_scorer=scorer, action_preselector=scorer.preselect,
                action_value_is_cost_to_go=True, **budget)
            stats = scorer.stats() if hasattr(scorer, "stats") else {
                "feature_computation_ms": scorer.feature_ms,
                "actual_feature_computation_ms": scorer.feature_ms,
                "gnn_inference_ms": scorer.gnn_ms,
            }
        progress.update({"mapped_operations": int(result.mapped_operations),
                         "expansions": int(result.expansions),
                         "candidate_actions": int(result.raw_candidate_actions),
                         "parent_solves": int(stats.get("parent_solves", 0)),
                         "child_solves": int(stats.get("child_solves", 0)),
                         "cache_hits": int(stats.get("parent_cache_hits", 0)) + int(stats.get("child_cache_hits", 0))})
        progress["stage"] = "serializing"
        row = base_row(instance, method, budget_name, result, stats, started_mapper)
        required = {"instance_id", "method", "budget", "success", "total_compile_time_ms"}
        if not required <= set(row):
            raise RuntimeError("result schema validation failed")
        atomic_json(result_path, {"schema_version": 1, "work_id": spec["work_id"], "row": row})
        return 0
    except Exception as exc:
        atomic_json(result_path, {"schema_version": 1, "work_id": spec["work_id"],
            "error_type": type(exc).__name__, "error_message": str(exc)})
        return 2
    finally:
        progress["stage"] = "finished"
        stop.set(); heartbeat_thread.join(timeout=1)


if __name__ == "__main__":
    raise SystemExit(main())
