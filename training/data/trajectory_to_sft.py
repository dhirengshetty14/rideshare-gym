"""Convert rideshare-gym Trajectory JSONL → HuggingFace Dataset of
(messages, completion) pairs for supervised fine-tuning.

For each successful trajectory, we extract one (input, target) example per
agent step:
  - input:  the chat history the LLM saw before the assistant turn
            (system + tools + initial_user + prior assistant/tool turns)
  - target: the assistant turn the model emitted (Hermes-style tool call)

Rejection-sampling SFT (RFT) means we filter to `success == True` first.
For more aggressive filtering you can also require `final_reward >= threshold`.

Usage:
    python -m training.data.trajectory_to_sft \
        --traj-dir runs/baseline-rollouts/trajectories \
        --output training_data/sft_v1.jsonl \
        --filter "success==True"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rideshare_gym.core.recorder import Trajectory


def trajectory_to_sft_examples(traj: Trajectory) -> list[dict[str, Any]]:
    """Walk one Trajectory and emit per-step (messages, completion) pairs.

    Each step produces one example. The input is the chat-history snapshot
    BEFORE the assistant turn; the target is the assistant_message dict.
    Skips steps where assistant_message is None (e.g. gold_oracle).
    """
    if traj.system_prompt is None or traj.initial_user_message is None:
        return []

    examples: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = [
        {"role": "system", "content": traj.system_prompt},
        {"role": "user", "content": traj.initial_user_message},
    ]
    for step in traj.steps:
        if step.assistant_message is None:
            continue
        # Snapshot history BEFORE this assistant turn = the SFT input.
        examples.append({
            "task_id": traj.task_id,
            "seed": traj.seed,
            "step_index": step.index,
            "messages": list(history),
            "completion": step.assistant_message,
            "tools": traj.tools_serialized or [],
            "reward": step.reward,
            "step_terminated": step.terminated,
            "trajectory_success": traj.success,
        })
        # Advance history with the actual assistant + tool turns the agent
        # produced (so subsequent steps see what really happened).
        history.append(step.assistant_message)
        if step.tool_result_message is not None:
            history.append(step.tool_result_message)
    return examples


def filter_trajectory(traj: Trajectory, expr: str) -> bool:
    """Tiny expression evaluator.
    Supported: 'success==True', 'success==False', 'final_reward>=0.9', etc.
    """
    expr = expr.strip()
    safe_locals = {
        "success": traj.success,
        "final_reward": traj.final_reward,
        "n_steps": len(traj.steps),
        "task_id": traj.task_id,
        "error_category": traj.error_category,
        "True": True, "False": False, "None": None,
    }
    try:
        return bool(eval(expr, {"__builtins__": {}}, safe_locals))
    except Exception:
        return False


def load_trajectories(traj_dir: Path) -> list[Trajectory]:
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


def build_sft_dataset(
    traj_dir: Path,
    output: Path,
    filter_expr: str = "success==True",
) -> int:
    trajs = load_trajectories(traj_dir)
    kept = [t for t in trajs if filter_trajectory(t, filter_expr)]
    print(f"Loaded {len(trajs)} trajectories; kept {len(kept)} after "
          f"filter: {filter_expr!r}")

    output.parent.mkdir(parents=True, exist_ok=True)
    n_examples = 0
    with output.open("w", encoding="utf-8") as f:
        for t in kept:
            for ex in trajectory_to_sft_examples(t):
                f.write(json.dumps(ex) + "\n")
                n_examples += 1
    print(f"Wrote {n_examples} SFT examples to {output}")
    return n_examples


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj-dir", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--filter", default="success==True",
                     help="Python expression over (success, final_reward, "
                          "n_steps, task_id, error_category)")
    args = ap.parse_args()
    build_sft_dataset(args.traj_dir, args.output, args.filter)


if __name__ == "__main__":
    main()
