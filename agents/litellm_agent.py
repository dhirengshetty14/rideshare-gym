"""LiteLLM / OpenAI-compatible tool-calling agent.

Works with any OpenAI-compatible Chat Completions endpoint — vanilla OpenAI,
LAS LiteLLM proxy (`https://llm-west.ncsu-las.net/v1`), Together, Groq, Fireworks,
local vLLM, etc. The model is selected by `--model`; for LAS the format is
`<provider>/<model>` (e.g. `openai/gpt-4o`, `openai/gpt-5`,
`us.anthropic.claude-3-7-sonnet-20250219-v1:0`).

Differences from the Claude baseline:
  * uses `client.chat.completions.create` instead of `messages.create`
  * tool block is the OpenAI shape (`{"type":"function","function":{...}}`)
  * tool args come back as a JSON STRING that we parse
  * tool result messages have role="tool" + tool_call_id (no list content)
  * we set `parallel_tool_calls=False` to keep one action per turn (matches
    Claude semantics and our verifier-per-step contract)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from rideshare_gym.core.env import GymEnvironment
from rideshare_gym.core.recorder import Trajectory, TrajectoryRecorder
from rideshare_gym.core.types import ToolCall

PROMPTS_DIR = Path(__file__).parent / "prompts"
DEFAULT_BASE_URL = "https://llm-west.ncsu-las.net/v1"
DEFAULT_MODEL = "openai/gpt-4o"


def load_prompt(name: str = "rideshare_system") -> str:
    return (PROMPTS_DIR / f"{name}.md").read_text(encoding="utf-8")


def _truncate(s: Any, max_chars: int = 200) -> str:
    s = str(s)
    return s if len(s) <= max_chars else s[:max_chars] + "..."


class LiteLLMAgent:
    """OpenAI-compatible tool-calling loop. One agent per process; reuse across episodes."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        api_key: str | None = None,
        max_tokens: int = 1024,
        prompt_name: str = "rideshare_system",
        verbose: bool = False,
        litellm_tags: list[str] | None = None,
        parallel_tool_calls: bool = False,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise RuntimeError(
                "openai SDK not installed. Run `pip install openai`."
            ) from e
        key = (
            api_key
            or os.environ.get("OPENAI_API_KEY")
            or os.environ.get("LAS_API_TOKEN")
            or os.environ.get("LAS_API_KEY")
        )
        if not key:
            raise SystemExit(
                "No API key found. Set one of: OPENAI_API_KEY, LAS_API_TOKEN, LAS_API_KEY."
            )
        headers: dict[str, str] = {}
        if litellm_tags:
            headers["x-litellm-tags"] = ",".join(litellm_tags)

        self.client = OpenAI(api_key=key, base_url=base_url, default_headers=headers)
        self.model = model
        self.base_url = base_url
        self.max_tokens = max_tokens
        self.prompt_name = prompt_name
        self.verbose = verbose
        self.parallel_tool_calls = parallel_tool_calls

    def run(
        self,
        env: GymEnvironment,
        *,
        on_step=None,
        on_event=None,
    ) -> Trajectory:
        """Run one episode.

        Hooks (optional, used by the UI for live updates):
          * on_step(step_idx, tool_call, observation, reward, terminated, info)
          * on_event(event_dict) — for non-tool events ("model_start", "no_tool_calls",
                                    "error", "finished")
        """
        obs, info = env.reset()
        rec = TrajectoryRecorder(
            task_id=info["task_id"],
            seed=env.task.seed,
            ground_truth=env.ground_truth,
            perturbations=info["perturbations"],
            meta={
                "agent_id": "litellm",
                "model": self.model,
                "base_url": self.base_url,
                "max_steps": env.task.max_steps,
            },
        )
        rec.set_initial(info["initial_state_hash"])

        system = load_prompt(self.prompt_name)
        tools = env.tool_registry.to_openai()
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": obs.to_agent_message()},
        ]

        success = False
        error_category: str | None = None
        if on_event:
            on_event({"event": "episode_start", "task_id": info["task_id"],
                      "tools": info["tools"], "perturbations": info["perturbations"]})

        for step in range(env.task.max_steps):
            if on_event:
                on_event({"event": "model_start", "step": step})
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    parallel_tool_calls=self.parallel_tool_calls,
                    max_tokens=self.max_tokens,
                )
            except TypeError:
                # Some proxies / model families reject `parallel_tool_calls`.
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    max_tokens=self.max_tokens,
                )
            except Exception as e:
                error_category = "crashed"
                if self.verbose:
                    print(f"[step {step}] LLM error: {type(e).__name__}: {e}")
                if on_event:
                    on_event({"event": "error", "step": step,
                              "error": f"{type(e).__name__}: {e}"})
                break

            msg = resp.choices[0].message
            usage_in = (resp.usage.prompt_tokens if resp.usage else 0) or 0
            usage_out = (resp.usage.completion_tokens if resp.usage else 0) or 0

            assistant_record: dict[str, Any] = {
                "role": "assistant",
                "content": msg.content or "",
            }
            if msg.tool_calls:
                assistant_record["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ]
            messages.append(assistant_record)

            if not msg.tool_calls:
                if self.verbose:
                    print(f"[step {step}] no tool_calls -- model said: {_truncate(msg.content)}")
                if on_event:
                    on_event({"event": "no_tool_calls", "step": step,
                              "content": (msg.content or "")[:500]})
                if env.step_count == 0:
                    error_category = "wrong_tool"
                else:
                    error_category = error_category or "goal_incomplete"
                break

            terminated_any = False
            truncated_any = False
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                call = ToolCall(
                    name=tc.function.name, arguments=args, tool_use_id=tc.id,
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

                tool_message_content = json.dumps({
                    "text": obs.text[:500],
                    "data": obs.data,
                    "tool_error": env_info.get("tool_error"),
                }, default=str)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": tool_message_content,
                })

                if self.verbose:
                    print(
                        f"[step {step}] {call.name}({_truncate(call.arguments, 160)}) "
                        f"-> reward={reward:.3f} term={terminated} ok={env_info['tool_ok']}"
                    )

                if not env_info["tool_ok"]:
                    err = env_info.get("tool_error", "")
                    if "wrong_args" in err:
                        error_category = error_category or "wrong_args"
                    elif "unknown_tool" in err:
                        error_category = error_category or "wrong_tool"

                if terminated:
                    terminated_any = True
                    success = True
                    error_category = None
                if truncated:
                    truncated_any = True

            if terminated_any:
                break
            if truncated_any:
                error_category = "goal_incomplete"
                break

        traj = rec.finalize(
            final_state_hash=env.final_state_hash,
            success=success,
            error_category=None if success else (error_category or "goal_incomplete"),
        )
        if on_event:
            on_event({"event": "finished", "success": success,
                      "final_reward": traj.final_reward,
                      "error_category": traj.error_category})
        return traj
