"""Compute per-trajectory, per-task, and overall statistics from a run directory.

A "run directory" is what `eval/run.py` writes: a folder containing
`scorecard.json` and a `trajectories/` subdir of one JSONL per episode.

This module reads the trajectories, computes the rich statistics defined in
analysis/STATS_PLAN.md, and emits a single `stats.json` file per run.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from rideshare_gym.core.recorder import Trajectory

# --------------------------------------------------------------------------- #
# Difficulty mapping (used for difficulty-tier rollups)
# --------------------------------------------------------------------------- #

EASY_TASKS = {
    "rideshare/match_single_ride",
    "rideshare/refund_cancelled_trip",
    "rideshare/verify_driver_documents",
}
MEDIUM_TASKS = {
    "rideshare/surge_demand_spike",
    "rideshare/fraud_ring_detection",
    "rideshare/lost_item_recovery",
    "rideshare/driver_pay_dispute",
    "rideshare/accident_incident_response",
    "rideshare/account_takeover_response",
}
HARD_TASKS = {
    "rideshare/realtime_dispatch_window",
    "rideshare/event_surge_planning",
    "rideshare/coordinated_fraud_response",
}


def _difficulty_of(task_id: str) -> str:
    base = task_id.split("__")[0]
    if base in EASY_TASKS:
        return "easy"
    if base in MEDIUM_TASKS:
        return "medium"
    if base in HARD_TASKS:
        return "hard"
    return "unknown"


# --------------------------------------------------------------------------- #
# Per-trajectory feature extraction
# --------------------------------------------------------------------------- #

def trajectory_features(t: Trajectory) -> dict[str, Any]:
    """Extract all per-trajectory features defined in STATS_PLAN §2."""
    n_calls = len(t.steps)
    tool_ok = 0
    wrong_args = 0
    unknown_tool = 0
    unique_tools: set[str] = set()
    for s in t.steps:
        info = s.info or {}
        if info.get("tool_ok"):
            tool_ok += 1
        err = info.get("tool_error") or ""
        if "wrong_args" in err:
            wrong_args += 1
        if "unknown_tool" in err:
            unknown_tool += 1
        unique_tools.add(s.action.name)

    return {
        "task_id": t.task_id,
        "seed": t.seed,
        "success": bool(t.success),
        "final_reward": float(t.final_reward),
        "n_steps": n_calls,
        "wall_time_seconds": float(t.meta.get("wall_time_seconds") or 0.0),
        "total_tokens_in": int(t.meta.get("total_tokens_in") or 0),
        "total_tokens_out": int(t.meta.get("total_tokens_out") or 0),
        "error_category": t.error_category,
        "tool_calls_made": n_calls,
        "tool_calls_ok": tool_ok,
        "tool_calls_wrong_args": wrong_args,
        "tool_calls_unknown_tool": unknown_tool,
        "unique_tools_used": len(unique_tools),
    }


# --------------------------------------------------------------------------- #
# Bootstrap confidence interval for proportions / means
# --------------------------------------------------------------------------- #

def bootstrap_ci(values: list[float], *, n_iter: int = 1000,
                 alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """Two-sided bootstrap CI for the mean of `values`.
    Returns (lower, upper) at confidence 1-alpha. Uses simple percentile bootstrap."""
    if not values:
        return (0.0, 0.0)
    import random
    rng = random.Random(seed)
    n = len(values)
    samples = []
    for _ in range(n_iter):
        idx = [rng.randrange(n) for _ in range(n)]
        samples.append(sum(values[i] for i in idx) / n)
    samples.sort()
    lo = samples[int(n_iter * (alpha / 2))]
    hi = samples[int(n_iter * (1 - alpha / 2)) - 1]
    return (lo, hi)


# --------------------------------------------------------------------------- #
# Per-task aggregation
# --------------------------------------------------------------------------- #

def per_task_stats(features: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Aggregate features by task_id, computing the stats defined in §3."""
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for f in features:
        by_task[f["task_id"]].append(f)

    out: dict[str, dict[str, Any]] = {}
    for tid, fs in by_task.items():
        n = len(fs)
        successes = [int(f["success"]) for f in fs]
        rewards = [f["final_reward"] for f in fs]
        steps = [f["n_steps"] for f in fs]
        walls = [f["wall_time_seconds"] for f in fs]
        tokens_in = [f["total_tokens_in"] for f in fs]
        tokens_out = [f["total_tokens_out"] for f in fs]
        tool_calls = sum(f["tool_calls_made"] for f in fs)
        tool_ok = sum(f["tool_calls_ok"] for f in fs)
        wrong_args = sum(f["tool_calls_wrong_args"] for f in fs)
        unknown_tool = sum(f["tool_calls_unknown_tool"] for f in fs)
        unique_tools = [f["unique_tools_used"] for f in fs]

        sr_lo, sr_hi = bootstrap_ci([float(s) for s in successes], n_iter=1000)
        rew_lo, rew_hi = bootstrap_ci(rewards, n_iter=1000)

        err_counts: Counter[str] = Counter()
        for f in fs:
            cat = f["error_category"]
            if cat is not None:
                err_counts[cat] += 1
        for k in ("goal_incomplete", "wrong_tool", "wrong_args", "crashed"):
            err_counts.setdefault(k, 0)

        out[tid] = {
            "n": n,
            "difficulty": _difficulty_of(tid),
            "success_rate": sum(successes) / n if n else 0.0,
            "success_rate_ci_95": [sr_lo, sr_hi],
            "mean_reward": (sum(rewards) / n) if rewards else 0.0,
            "mean_reward_ci_95": [rew_lo, rew_hi],
            "std_reward": (statistics.stdev(rewards) if len(rewards) > 1 else 0.0),
            "median_reward": (statistics.median(rewards) if rewards else 0.0),
            "mean_steps": (sum(steps) / n) if steps else 0.0,
            "median_steps": (statistics.median(steps) if steps else 0.0),
            "mean_wall_time_s": (sum(walls) / n) if walls else 0.0,
            "mean_tokens_in": (sum(tokens_in) / n) if tokens_in else 0.0,
            "mean_tokens_out": (sum(tokens_out) / n) if tokens_out else 0.0,
            "error_breakdown": dict(err_counts),
            "tool_call_validity_rate": (tool_ok / tool_calls) if tool_calls else 0.0,
            "wrong_args_rate": (wrong_args / tool_calls) if tool_calls else 0.0,
            "unknown_tool_rate": (unknown_tool / tool_calls) if tool_calls else 0.0,
            "mean_unique_tools": (sum(unique_tools) / n) if unique_tools else 0.0,
        }
    return out


# --------------------------------------------------------------------------- #
# Overall aggregation (across all tasks)
# --------------------------------------------------------------------------- #

def overall_stats(features: list[dict[str, Any]]) -> dict[str, Any]:
    """All-task aggregation defined in §4."""
    if not features:
        return {"n": 0}
    n = len(features)
    successes = [int(f["success"]) for f in features]
    rewards = [f["final_reward"] for f in features]
    walls = [f["wall_time_seconds"] for f in features]

    sr_lo, sr_hi = bootstrap_ci([float(s) for s in successes], n_iter=1000)
    rew_lo, rew_hi = bootstrap_ci(rewards, n_iter=1000)

    by_diff: dict[str, list[float]] = defaultdict(list)
    for f in features:
        by_diff[_difficulty_of(f["task_id"])].append(float(f["success"]))
    success_by_diff = {
        d: (sum(vs) / len(vs)) if vs else 0.0
        for d, vs in by_diff.items()
    }

    err_counts: Counter[str] = Counter()
    for f in features:
        if f["error_category"] is not None:
            err_counts[f["error_category"]] += 1

    total_calls = sum(f["tool_calls_made"] for f in features)
    total_ok = sum(f["tool_calls_ok"] for f in features)
    total_wrong_args = sum(f["tool_calls_wrong_args"] for f in features)
    total_unknown = sum(f["tool_calls_unknown_tool"] for f in features)

    return {
        "n": n,
        "overall_success_rate": sum(successes) / n,
        "overall_success_rate_ci_95": [sr_lo, sr_hi],
        "mean_reward": sum(rewards) / n,
        "mean_reward_ci_95": [rew_lo, rew_hi],
        "success_rate_by_difficulty": success_by_diff,
        "error_breakdown": dict(err_counts),
        "dominant_failure_mode": (err_counts.most_common(1)[0][0]
                                    if err_counts else None),
        "total_tokens_in": sum(f["total_tokens_in"] for f in features),
        "total_tokens_out": sum(f["total_tokens_out"] for f in features),
        "mean_episode_wall_time_s": sum(walls) / n,
        "tool_call_validity_rate": (total_ok / total_calls) if total_calls else 0.0,
        "wrong_args_rate": (total_wrong_args / total_calls) if total_calls else 0.0,
        "unknown_tool_rate": (total_unknown / total_calls) if total_calls else 0.0,
    }


# --------------------------------------------------------------------------- #
# Run-directory loading
# --------------------------------------------------------------------------- #

def load_run_dir(run_dir: Path) -> list[Trajectory]:
    traj_dir = run_dir / "trajectories"
    if not traj_dir.is_dir():
        # Maybe the user pointed directly at a trajectories dir.
        traj_dir = run_dir
    out: list[Trajectory] = []
    for p in sorted(traj_dir.glob("*.jsonl")):
        if p.name == "index.jsonl":
            continue
        try:
            text = p.read_text(encoding="utf-8").strip()
            if not text:
                continue
            out.append(Trajectory.model_validate_json(text))
        except Exception as e:
            print(f"  skip {p.name}: {type(e).__name__}: {e}")
    return out


def compute_run_stats(run_dir: Path) -> dict[str, Any]:
    """Top-level: read a run directory, compute per-task + overall stats."""
    trajs = load_run_dir(run_dir)
    features = [trajectory_features(t) for t in trajs]
    return {
        "run_dir": str(run_dir),
        "n_trajectories": len(trajs),
        "per_task": per_task_stats(features),
        "overall": overall_stats(features),
        "_trajectory_features": features,    # kept for paired comparison
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path,
                     help="Path to a run dir containing trajectories/*.jsonl")
    ap.add_argument("--output", required=True, type=Path,
                     help="Where to write stats.json")
    ap.add_argument("--summary", action="store_true",
                     help="Print a compact text summary to stdout too.")
    args = ap.parse_args()

    stats = compute_run_stats(args.run_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(stats, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {args.output}")

    if args.summary:
        ov = stats["overall"]
        print()
        print(f"  episodes:                {stats['n_trajectories']}")
        if "overall_success_rate" in ov:
            sr = ov["overall_success_rate"]
            ci = ov.get("overall_success_rate_ci_95", [sr, sr])
            print(f"  overall success rate:    {sr:.1%}  "
                  f"95% CI [{ci[0]:.1%}, {ci[1]:.1%}]")
            print(f"  mean reward:             {ov['mean_reward']:.3f}")
            print(f"  tool-call validity rate: {ov.get('tool_call_validity_rate', 0):.1%}")
            print(f"  wrong-args rate:         {ov.get('wrong_args_rate', 0):.1%}")
            print(f"  by difficulty:           {ov.get('success_rate_by_difficulty')}")
            print(f"  dominant failure:        {ov.get('dominant_failure_mode')}")


if __name__ == "__main__":
    main()
