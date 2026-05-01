"""Verifier-as-reward function for GRPO online training.

The model emits a tool-call response. This function:
  1. Spins up a fresh gym episode for the (task_id, seed) the prompt was for
  2. Replays the prior agent steps from the prompt history
  3. Parses the model's completion as a tool call
  4. Steps the env once with that tool call
  5. Returns the verifier's reward as the scalar

For multi-step tasks we take a simpler shortcut by default: parse the model's
completion as a single tool call, replay the trajectory up to the current
step from the prompt, step once, return the reward at that single step.

This is the pattern Tülu 3 RLVR + DeepSeek-R1 use. The verifier is a deterministic
Python function; the reward is fully verifiable; no learned reward model needed.
"""

from __future__ import annotations

import json
from typing import Any

from rideshare_gym.core.env import GymEnvironment
from rideshare_gym.core.types import ToolCall
from rideshare_gym.rideshare_sandbox import in_process_sandbox_factory
from rideshare_gym.tasks import REGISTRY as TASK_REGISTRY

# Late import — only needed at training time.
try:
    from agents.trained_local import parse_tool_calls
except Exception:  # noqa: BLE001
    parse_tool_calls = None  # type: ignore[assignment]


# --------------------------------------------------------------------------- #
# Sandbox factory cache — re-use across rollouts to avoid app rebuild churn.
# --------------------------------------------------------------------------- #

_FACTORY = None


def _get_factory():
    global _FACTORY
    if _FACTORY is None:
        _FACTORY = in_process_sandbox_factory(tenant_prefix="train")
    return _FACTORY


# --------------------------------------------------------------------------- #
# Reward function for GRPO
# --------------------------------------------------------------------------- #

def gym_step_reward(
    *,
    task_id: str,
    seed: int,
    completion: str,
) -> float:
    """Single-step reward: spin a fresh gym, parse the model's completion as
    one tool call, step the env once, return the verifier reward.

    For multi-step training, you'd extend this to replay the full chat
    history into the gym before stepping. The single-step version is
    sufficient for SFT-style + GRPO-with-trajectory-completions training.
    """
    if parse_tool_calls is None:
        return 0.0

    # Strip any task-id suffix from FixtureMutator wrapping.
    base_task = task_id.split("__")[0]
    if base_task not in TASK_REGISTRY:
        return 0.0

    factory = _get_factory()
    task = TASK_REGISTRY[base_task](seed=seed)
    env = GymEnvironment(task=task, sandbox_factory=factory)
    try:
        env.reset()
        tcs = parse_tool_calls(completion)
        if not tcs:
            return 0.0
        # Take the first tool call.
        tc = tcs[0]
        call = ToolCall(name=tc["name"], arguments=tc["arguments"])
        _, reward, terminated, _, _ = env.step(call)
        # Bonus for terminating successfully.
        return float(reward) + (0.5 if terminated else 0.0)
    except Exception:
        return 0.0
    finally:
        try:
            env.close()
        except Exception:  # noqa: BLE001
            pass


def parse_metadata_from_prompt(prompt: str) -> tuple[str, int]:
    """Pulls (task_id, seed) out of a training prompt that was emitted by
    `trajectory_to_grpo` (which includes them as an inline JSON header)."""
    # By convention, recipe_03_grpo.py prepends:
    #   <!-- gym_meta {"task_id":"...","seed":N} -->
    marker = "<!-- gym_meta "
    if marker not in prompt:
        return ("rideshare/match_single_ride", 0)
    try:
        start = prompt.index(marker) + len(marker)
        end = prompt.index(" -->", start)
        meta = json.loads(prompt[start:end])
        return (meta["task_id"], int(meta["seed"]))
    except Exception:
        return ("rideshare/match_single_ride", 0)


def grpo_reward_fn(prompts, completions, **kwargs) -> list[float]:
    """TRL-compatible reward function signature.

    Takes a list of prompts + completions, returns list of scalar rewards.
    Each (prompt, completion) pair is independently scored by the gym.
    """
    rewards: list[float] = []
    for prompt, completion in zip(prompts, completions):
        # TRL passes completions in different shapes depending on the
        # trainer config. Normalise to a string.
        if isinstance(completion, list) and completion:
            text = (completion[0].get("content")
                     if isinstance(completion[0], dict) else str(completion[0]))
        elif isinstance(completion, dict):
            text = completion.get("content", "")
        else:
            text = str(completion)
        prompt_text = (prompt if isinstance(prompt, str)
                        else (prompt[-1].get("content", "")
                              if isinstance(prompt, list) and prompt else ""))
        task_id, seed = parse_metadata_from_prompt(prompt_text)
        rewards.append(gym_step_reward(
            task_id=task_id, seed=seed, completion=text or "",
        ))
    return rewards
