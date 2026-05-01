"""Failure miner — cluster failed trajectories by failure pattern.

Output: a structured report identifying *what specifically* the model is
bad at. This is what the mentor wants to see — the gym surfaces concrete
failure patterns, which become the targets for the next training round.

Clusters by:
  - task_id (which task)
  - error_category (goal_incomplete | wrong_tool | wrong_args | crashed)
  - last_tool_name (which tool the agent gave up on)
  - first_failed_assertion (which verifier check first broke)

Usage:
    python -m training.data.failure_miner \
        --traj-dir runs/baseline-rollouts/trajectories \
        --output analysis/failure_report.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rideshare_gym.core.recorder import Trajectory
from training.data.trajectory_to_sft import load_trajectories


@dataclass
class FailureCluster:
    cluster_key: tuple[str, str, str, str]
    """task_id, error_category, last_tool, first_failed_assertion"""
    count: int = 0
    example_episode_ids: list[str] = field(default_factory=list)
    example_trajectories: list[Trajectory] = field(default_factory=list)

    @property
    def task_id(self) -> str: return self.cluster_key[0]
    @property
    def error_category(self) -> str: return self.cluster_key[1]
    @property
    def last_tool(self) -> str: return self.cluster_key[2]
    @property
    def first_failed_assertion(self) -> str: return self.cluster_key[3]


def _last_tool_name(traj: Trajectory) -> str:
    if not traj.steps:
        return "<no_steps>"
    return traj.steps[-1].action.name


def _first_failed_assertion(traj: Trajectory) -> str:
    """Walk steps; find the first verifier `failed` list with content."""
    for step in traj.steps:
        v = (step.info or {}).get("verifier") or {}
        failed = v.get("failed") or []
        if failed:
            return failed[0]
    return "<none>"


def cluster_failures(trajs: list[Trajectory]) -> list[FailureCluster]:
    failures = [t for t in trajs if not t.success]
    clusters: dict[tuple[str, str, str, str], FailureCluster] = {}
    for t in failures:
        key = (
            t.task_id,
            t.error_category or "<none>",
            _last_tool_name(t),
            _first_failed_assertion(t),
        )
        if key not in clusters:
            clusters[key] = FailureCluster(cluster_key=key)
        c = clusters[key]
        c.count += 1
        if len(c.example_episode_ids) < 5:
            c.example_episode_ids.append(t.episode_id)
            c.example_trajectories.append(t)
    return sorted(clusters.values(), key=lambda c: -c.count)


def cluster_to_dict(c: FailureCluster, *, with_traj_summaries: bool = True) -> dict[str, Any]:
    out = {
        "task_id": c.task_id,
        "error_category": c.error_category,
        "last_tool": c.last_tool,
        "first_failed_assertion": c.first_failed_assertion,
        "count": c.count,
        "example_episode_ids": c.example_episode_ids,
    }
    if with_traj_summaries:
        out["example_trajectories"] = [
            {
                "episode_id": t.episode_id,
                "n_steps": len(t.steps),
                "final_reward": t.final_reward,
                "last_action": (t.steps[-1].action.model_dump()
                                  if t.steps else None),
            } for t in c.example_trajectories
        ]
    return out


def per_task_summary(trajs: list[Trajectory]) -> dict[str, dict[str, Any]]:
    by_task: dict[str, list[Trajectory]] = defaultdict(list)
    for t in trajs:
        by_task[t.task_id].append(t)
    out: dict[str, dict[str, Any]] = {}
    for tid, ts in by_task.items():
        n = len(ts)
        n_success = sum(1 for t in ts if t.success)
        err_counts = Counter(t.error_category for t in ts if not t.success)
        out[tid] = {
            "n_episodes": n,
            "success_rate": n_success / n if n else 0.0,
            "error_breakdown": dict(err_counts),
        }
    return out


def write_failure_report(traj_dir: Path, output: Path,
                          *, top_k: int = 20) -> None:
    trajs = load_trajectories(traj_dir)
    clusters = cluster_failures(trajs)
    report = {
        "n_trajectories": len(trajs),
        "n_failures": sum(1 for t in trajs if not t.success),
        "per_task": per_task_summary(trajs),
        "top_failure_clusters": [
            cluster_to_dict(c) for c in clusters[:top_k]
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    # Stdout summary for quick reading.
    print(f"Loaded {report['n_trajectories']} trajectories "
          f"({report['n_failures']} failures)")
    print()
    print("Top failure clusters:")
    print(f"{'count':>5} {'task':<35} {'last_tool':<25} {'first_failed':<25} {'err':<20}")
    for c in clusters[:top_k]:
        print(f"{c.count:>5} {c.task_id:<35} {c.last_tool:<25} "
              f"{c.first_failed_assertion:<25} {c.error_category:<20}")
    print()
    print(f"Full report written to {output}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj-dir", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--top-k", type=int, default=20)
    args = ap.parse_args()
    write_failure_report(args.traj_dir, args.output, top_k=args.top_k)


if __name__ == "__main__":
    main()
