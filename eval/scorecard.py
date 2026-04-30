"""Aggregate per-task and overall metrics from a list of Trajectories."""

from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Any

from rideshare_gym.core.recorder import Trajectory

from eval.error_taxonomy import aggregate as aggregate_errors


def per_task(trajectories: list[Trajectory]) -> dict[str, dict[str, Any]]:
    by_task: dict[str, list[Trajectory]] = defaultdict(list)
    for t in trajectories:
        by_task[t.task_id].append(t)

    out: dict[str, dict[str, Any]] = {}
    for task_id, ts in by_task.items():
        n = len(ts)
        success = sum(1 for t in ts if t.success)
        rewards = [t.final_reward for t in ts]
        n_steps = [len(t.steps) for t in ts]
        wall_times = [t.meta.get("wall_time_seconds", 0) for t in ts]
        tokens_in = [t.meta.get("total_tokens_in") or 0 for t in ts]
        tokens_out = [t.meta.get("total_tokens_out") or 0 for t in ts]
        out[task_id] = {
            "n": n,
            "success_rate": success / n if n else 0.0,
            "mean_reward": statistics.mean(rewards) if rewards else 0.0,
            "mean_steps": statistics.mean(n_steps) if n_steps else 0.0,
            "mean_wall_time_seconds": statistics.mean(wall_times) if wall_times else 0.0,
            "total_tokens_in": sum(tokens_in),
            "total_tokens_out": sum(tokens_out),
            "errors": aggregate_errors(ts),
        }
    return out


def overall(trajectories: list[Trajectory]) -> dict[str, Any]:
    n = len(trajectories)
    if n == 0:
        return {"n": 0}
    success = sum(1 for t in trajectories if t.success)
    return {
        "n": n,
        "success_rate": success / n,
        "mean_reward": statistics.mean(t.final_reward for t in trajectories),
        "total_tokens_in": sum(t.meta.get("total_tokens_in") or 0 for t in trajectories),
        "total_tokens_out": sum(t.meta.get("total_tokens_out") or 0 for t in trajectories),
        "wall_clock_seconds": sum(t.meta.get("wall_time_seconds") or 0 for t in trajectories),
        "errors": aggregate_errors(trajectories),
    }


def make_scorecard(
    trajectories: list[Trajectory],
    *,
    agent_id: str,
    model: str | None = None,
) -> dict[str, Any]:
    return {
        "agent": agent_id,
        "model": model,
        "tasks": per_task(trajectories),
        "overall": overall(trajectories),
    }
