"""Convert trajectories → DPO preference pairs (chosen vs rejected).

Pairing strategy: for each (task_id, seed), take ALL trajectory pairs where
one succeeded and the other failed, and pair them at the FIRST step where
they diverge (different assistant_message). The chosen completion is from
the success trajectory; rejected from the failure.

This isolates the *decisive action* — the moment where the successful agent
made the right call and the failing one didn't.

Usage:
    python -m training.data.trajectory_to_dpo \
        --traj-dir runs/baseline-rollouts/trajectories \
        --output training_data/dpo_v1.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from rideshare_gym.core.recorder import Trajectory
from training.data.trajectory_to_sft import load_trajectories


def _action_signature(assistant_message: dict[str, Any] | None) -> str:
    """Stable hash of the assistant turn — used to detect divergence."""
    if assistant_message is None:
        return ""
    payload = json.dumps(assistant_message, sort_keys=True, default=str)
    return hashlib.md5(payload.encode()).hexdigest()


def build_pairs(success: Trajectory, failure: Trajectory) -> list[dict[str, Any]]:
    """For one (success, failure) traj pair, build DPO examples at every
    step where their assistant messages diverge.

    Most pairings yield 1 example (the first divergence). For longer
    trajectories we may yield more if both agents made multiple distinct
    decisions on the same context.
    """
    if success.system_prompt is None or success.initial_user_message is None:
        return []

    examples: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = [
        {"role": "system", "content": success.system_prompt},
        {"role": "user", "content": success.initial_user_message},
    ]

    n = min(len(success.steps), len(failure.steps))
    for i in range(n):
        s_step, f_step = success.steps[i], failure.steps[i]
        if s_step.assistant_message is None or f_step.assistant_message is None:
            break
        if _action_signature(s_step.assistant_message) != _action_signature(f_step.assistant_message):
            examples.append({
                "task_id": success.task_id,
                "seed": success.seed,
                "step_index": i,
                "messages": list(history),
                "chosen": s_step.assistant_message,
                "rejected": f_step.assistant_message,
                "tools": success.tools_serialized or [],
                "chosen_terminated": s_step.terminated,
                "chosen_reward": s_step.reward,
                "rejected_reward": f_step.reward,
            })
            # Pair only at the FIRST divergence — actions after this point
            # diverge from different states, so they don't represent the
            # same decision.
            break
        # No divergence yet — the trajectories are still on the same path.
        history.append(s_step.assistant_message)
        if s_step.tool_result_message is not None:
            history.append(s_step.tool_result_message)
    return examples


def build_dpo_dataset(traj_dir: Path, output: Path) -> int:
    trajs = load_trajectories(traj_dir)

    # Group by (task_id, seed).
    groups: dict[tuple[str, int], list[Trajectory]] = defaultdict(list)
    for t in trajs:
        groups[(t.task_id, t.seed)].append(t)

    output.parent.mkdir(parents=True, exist_ok=True)
    n_examples = 0
    n_eligible_groups = 0
    with output.open("w", encoding="utf-8") as f:
        for (task_id, seed), members in groups.items():
            successes = [t for t in members if t.success]
            failures = [t for t in members if not t.success]
            if not successes or not failures:
                continue
            n_eligible_groups += 1
            for s in successes:
                for fl in failures:
                    for ex in build_pairs(s, fl):
                        f.write(json.dumps(ex) + "\n")
                        n_examples += 1

    print(f"Loaded {len(trajs)} trajectories")
    print(f"{n_eligible_groups} (task,seed) groups had both success+fail")
    print(f"Wrote {n_examples} DPO preference pairs to {output}")
    return n_examples


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj-dir", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    build_dpo_dataset(args.traj_dir, args.output)


if __name__ == "__main__":
    main()
