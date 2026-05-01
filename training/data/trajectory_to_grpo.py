"""Build a GRPO prompt dataset from gym trajectories (or just a task list).

For GRPO we don't need (input, target) pairs — the trainer generates rollouts
itself and the gym scores them. We just need diverse PROMPTS that point at
real gym states.

Strategy: enumerate (task, seed) pairs across the registry. For each, build
the system+user prompt that the agent would normally see at episode start.
Embed (task_id, seed) as an HTML-comment marker so `gym_reward_fn` can
recover them when scoring.

Usage:
    python -m training.data.trajectory_to_grpo \
        --output training_data/grpo_prompts.jsonl \
        --seeds-per-task 50 --start-seed 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rideshare_gym.core.env import GymEnvironment
from rideshare_gym.rideshare_sandbox import in_process_sandbox_factory
from rideshare_gym.tasks import REGISTRY as TASK_REGISTRY

PROMPT_DIR = Path(__file__).resolve().parent.parent.parent / "agents" / "prompts"


def load_system_prompt() -> str:
    return (PROMPT_DIR / "rideshare_system.md").read_text(encoding="utf-8")


def build_prompt_dataset(
    output: Path,
    *,
    seeds_per_task: int = 50,
    start_seed: int = 0,
    include_tasks: list[str] | None = None,
) -> int:
    factory = in_process_sandbox_factory(tenant_prefix="grpo_data")
    system = load_system_prompt()

    task_ids = include_tasks or sorted(TASK_REGISTRY.keys())
    n_written = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for tid in task_ids:
            for seed in range(start_seed, start_seed + seeds_per_task):
                task = TASK_REGISTRY[tid](seed=seed)
                env = GymEnvironment(task=task, sandbox_factory=factory)
                try:
                    obs, info = env.reset()
                    user = obs.to_agent_message()
                    # Embed metadata so gym_reward_fn can recover it.
                    user_with_meta = (
                        f"<!-- gym_meta {json.dumps({'task_id': tid, 'seed': seed})} -->\n"
                        + user
                    )
                    f.write(json.dumps({
                        "task_id": tid,
                        "seed": seed,
                        "prompt": [
                            {"role": "system", "content": system},
                            {"role": "user", "content": user_with_meta},
                        ],
                        "tools": info.get("tools", []),
                    }) + "\n")
                    n_written += 1
                finally:
                    env.close()

    print(f"Wrote {n_written} GRPO prompts to {output} "
          f"({len(task_ids)} tasks x {seeds_per_task} seeds)")
    return n_written


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--seeds-per-task", type=int, default=50)
    ap.add_argument("--start-seed", type=int, default=0)
    ap.add_argument("--tasks", nargs="*", default=None)
    args = ap.parse_args()
    build_prompt_dataset(
        args.output,
        seeds_per_task=args.seeds_per_task,
        start_seed=args.start_seed,
        include_tasks=args.tasks,
    )


if __name__ == "__main__":
    main()
