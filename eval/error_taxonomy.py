"""τ-bench-style error categorization — buckets failure trajectories.

Categories:
  * goal_incomplete  — agent exhausted step budget without terminating
  * wrong_tool       — called a tool not in the registry / off-task
  * wrong_args       — JSON Schema validation failure (correct tool, bad args)
  * crashed          — uncaught exception in the agent or env
  * None             — episode succeeded
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable

from rideshare_gym.core.recorder import Trajectory


def categorize(traj: Trajectory) -> str | None:
    """Return the dominant error for a single trajectory, or None on success."""
    if traj.success:
        return None
    if traj.error_category is not None:
        return traj.error_category
    # Fallback: scan steps.
    counts: Counter[str] = Counter()
    for step in traj.steps:
        err = step.info.get("tool_error", "") if step.info else ""
        if "wrong_args" in err:
            counts["wrong_args"] += 1
        elif "unknown_tool" in err:
            counts["wrong_tool"] += 1
        elif "handler_exception" in err:
            counts["crashed"] += 1
    if counts:
        return counts.most_common(1)[0][0]
    return "goal_incomplete"


def aggregate(trajectories: Iterable[Trajectory]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for t in trajectories:
        cat = categorize(t)
        if cat is not None:
            counts[cat] += 1
    # Always include all four buckets so the scorecard format is stable.
    for k in ("goal_incomplete", "wrong_tool", "wrong_args", "crashed"):
        counts.setdefault(k, 0)
    return dict(counts)
