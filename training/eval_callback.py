"""TRL callback that runs the gym eval on the current checkpoint and logs
to wandb. Hook into training to track success-rate progress every N steps."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def make_periodic_gym_eval_callback(
    *,
    eval_interval_steps: int = 200,
    n_episodes: int = 2,
    tasks: list[str] | None = None,
    base_model: str = "Qwen/Qwen2.5-7B-Instruct",
):
    """Returns a TrainerCallback that snapshots the current training
    weights, runs eval/run.py over the gym, parses the scorecard, and
    logs metrics to wandb.

    NOTE: this is intentionally lightweight — we do a small `n_episodes`
    eval to keep training throughput high. Full evals happen offline."""
    try:
        from transformers import TrainerCallback, TrainerControl, TrainerState
    except ImportError:
        return None

    class GymEvalCallback(TrainerCallback):
        def on_save(self, args, state: "TrainerState", control: "TrainerControl",
                     **kwargs):
            if state.global_step % eval_interval_steps != 0:
                return control
            # Find the most-recently-saved checkpoint dir.
            ckpt = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            if not ckpt.exists():
                return control
            # Run quick eval; use subprocess so OOM in eval doesn't kill training.
            import subprocess
            tasks_arg = ",".join(tasks) if tasks else "rideshare/*"
            out_dir = ckpt / "eval"
            cmd = [
                "python", "eval/run.py",
                "--tasks", tasks_arg,
                "--agent", "trained_local",
                "--checkpoint", str(ckpt),
                "--base-model", base_model,
                "--n-episodes", str(n_episodes),
                "--out", str(out_dir),
            ]
            try:
                subprocess.run(cmd, check=True, timeout=1800)
            except Exception as e:  # noqa: BLE001
                print(f"[eval_callback] eval failed: {e}")
                return control
            # Parse scorecard, log to wandb.
            sc = out_dir / "scorecard.json"
            if not sc.exists() or os.environ.get("WANDB_API_KEY") is None:
                return control
            try:
                import json as _json
                import wandb
                metrics = _json.loads(sc.read_text(encoding="utf-8"))
                ov = metrics.get("overall", {})
                wandb.log({
                    "eval/success_rate": ov.get("success_rate"),
                    "eval/mean_reward": ov.get("mean_reward"),
                    "eval/n_episodes": ov.get("n"),
                    "step": state.global_step,
                })
            except Exception as e:  # noqa: BLE001
                print(f"[eval_callback] wandb log failed: {e}")
            return control

    return GymEvalCallback()
