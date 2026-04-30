"""Claude baseline agent — Anthropic SDK tool-calling loop with prompt caching.

Runs an agent against a `GymEnvironment`, returns a `Trajectory`. Re-rolls
amortise the system prompt + tool block via `cache_control: ephemeral`.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from rideshare_gym.core.env import GymEnvironment
from rideshare_gym.core.recorder import Trajectory, TrajectoryRecorder
from rideshare_gym.core.types import Observation, ToolCall

PROMPTS_DIR = Path(__file__).parent / "prompts"


def load_prompt(name: str = "rideshare_system") -> str:
    return (PROMPTS_DIR / f"{name}.md").read_text(encoding="utf-8")


def _truncate_for_log(payload: Any, max_chars: int = 2000) -> str:
    s = json.dumps(payload, default=str)
    return s if len(s) <= max_chars else s[:max_chars] + "..."


class ClaudeBaselineAgent:
    """One agent instance per process. Use `run(env)` to play one episode."""

    def __init__(
        self,
        *,
        model: str = "claude-opus-4-7",
        max_tokens: int = 1024,
        api_key: str | None = None,
        prompt_name: str = "rideshare_system",
        verbose: bool = False,
    ) -> None:
        try:
            from anthropic import Anthropic
        except ImportError as e:
            raise RuntimeError(
                "anthropic SDK not installed. Run `pip install anthropic`."
            ) from e
        self.client = Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self.model = model
        self.max_tokens = max_tokens
        self.prompt_name = prompt_name
        self.verbose = verbose

    def run(
        self,
        env: GymEnvironment,
        *,
        on_step: Callable[[int, ToolCall, Observation, float, bool, dict], None] | None = None,
    ) -> Trajectory:
        obs, info = env.reset()

        system = [{
            "type": "text",
            "text": load_prompt(self.prompt_name),
            "cache_control": {"type": "ephemeral"},
        }]
        tools = env.tool_registry.to_anthropic()

        rec = TrajectoryRecorder(
            task_id=info["task_id"],
            seed=env.task.seed,
            ground_truth=env.ground_truth,
            perturbations=info["perturbations"],
            meta={
                "agent_id": "claude_baseline",
                "model": self.model,
                "max_steps": env.task.max_steps,
            },
        )
        rec.set_initial(info["initial_state_hash"])

        messages: list[dict[str, Any]] = [
            {"role": "user", "content": obs.to_agent_message()}
        ]

        success = False
        error_category: str | None = None

        for step in range(env.task.max_steps):
            try:
                resp = self.client.messages.create(
                    model=self.model,
                    system=system,
                    tools=tools,
                    messages=messages,
                    max_tokens=self.max_tokens,
                )
            except Exception as e:
                error_category = "crashed"
                if self.verbose:
                    print(f"[step {step}] anthropic error: {e}")
                break

            usage_in = getattr(resp.usage, "input_tokens", 0) or 0
            usage_out = getattr(resp.usage, "output_tokens", 0) or 0

            tool_use = next((b for b in resp.content if getattr(b, "type", None) == "tool_use"), None)
            assistant_blocks = [b.model_dump() if hasattr(b, "model_dump") else b for b in resp.content]
            messages.append({"role": "assistant", "content": assistant_blocks})

            if tool_use is None:
                # Agent emitted only text — no progress possible.
                if self.verbose:
                    print(f"[step {step}] no tool_use, stopping")
                if env.step_count == 0:
                    error_category = "wrong_tool"
                else:
                    error_category = error_category or "goal_incomplete"
                break

            call = ToolCall(
                name=tool_use.name,
                arguments=tool_use.input or {},
                tool_use_id=tool_use.id,
            )
            obs, reward, terminated, truncated, env_info = env.step(call)
            rec.record(
                action=call, observation=obs, reward=reward,
                terminated=terminated, truncated=truncated,
                latency_ms=env_info.get("tool_latency_ms", 0.0),
                info=env_info,
                tokens_in=usage_in, tokens_out=usage_out,
            )
            if on_step:
                on_step(step, call, obs, reward, terminated, env_info)
            if self.verbose:
                print(f"[step {step}] {call.name}({_truncate_for_log(call.arguments, 200)}) "
                      f"→ reward={reward:.3f} term={terminated} ok={env_info['tool_ok']}")

            tool_result_block = {
                "type": "tool_result",
                "tool_use_id": tool_use.id,
                "is_error": not env_info["tool_ok"],
                "content": json.dumps({
                    "text": obs.text[:500],
                    "data": obs.data,
                    "tool_error": env_info.get("tool_error"),
                }, default=str),
            }
            messages.append({"role": "user", "content": [tool_result_block]})

            if terminated:
                success = True
                error_category = None
                break
            if truncated:
                error_category = "goal_incomplete"
                break

            # Track per-step error categories for the dominant failure mode
            if not env_info["tool_ok"]:
                err = env_info.get("tool_error", "")
                if "wrong_args" in err:
                    error_category = error_category or "wrong_args"
                elif "unknown_tool" in err:
                    error_category = error_category or "wrong_tool"

        return rec.finalize(
            final_state_hash=env.final_state_hash,
            success=success,
            error_category=None if success else (error_category or "goal_incomplete"),
        )
