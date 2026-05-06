"""Local-checkpoint agent.

Loads a HuggingFace model (base or fine-tuned) and runs the rideshare gym
loop with the same interface as litellm_agent / claude_baseline. Supports
Qwen2.5-Instruct-style native tool-calling via chat template.

Usage:
    from agents.trained_local import TrainedLocalAgent
    agent = TrainedLocalAgent(
        checkpoint_path="checkpoints/sft_v1",   # or HF hub id
        base_model="Qwen/Qwen2.5-7B-Instruct",  # for tokenizer + template
        device="cuda",
        load_in_4bit=True,                       # QLoRA-friendly inference
    )
    traj = agent.run(env)

This is the agent we eval against during/after training. The same module
also powers `--agent trained_local --checkpoint <path>` in eval/run.py.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from rideshare_gym.core.env import GymEnvironment
from rideshare_gym.core.recorder import Trajectory, TrajectoryRecorder
from rideshare_gym.core.types import ToolCall

PROMPTS_DIR = Path(__file__).parent / "prompts"
DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def load_prompt(name: str = "rideshare_system") -> str:
    return (PROMPTS_DIR / f"{name}.md").read_text(encoding="utf-8")


def _truncate(s: Any, max_chars: int = 200) -> str:
    s = str(s)
    return s if len(s) <= max_chars else s[:max_chars] + "..."


# --------------------------------------------------------------------------- #
# Tool-call parsing — Qwen2.5 emits Hermes-style:
#     <tool_call>{"name": "match_ride", "arguments": {...}}</tool_call>
# --------------------------------------------------------------------------- #

TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
    re.DOTALL,
)


def parse_tool_calls(text: str) -> list[dict[str, Any]]:
    """Extract Hermes-style tool calls from raw model output.
    Returns a list of {"name": ..., "arguments": dict}. Empty list if none."""
    out: list[dict[str, Any]] = []
    for m in TOOL_CALL_RE.finditer(text or ""):
        try:
            obj = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        name = obj.get("name") or obj.get("tool")
        args = obj.get("arguments") or obj.get("args") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        if name:
            out.append({"name": name, "arguments": args})
    return out


# --------------------------------------------------------------------------- #
# Agent
# --------------------------------------------------------------------------- #

class TrainedLocalAgent:
    """Same interface as the LLM agents (`run(env, on_step=, on_event=)
    -> Trajectory`), but uses a local HuggingFace model instead of a remote
    API. Supports any model whose chat template emits Hermes-style tool
    calls (Qwen2.5-Instruct family is the canonical choice)."""

    def __init__(
        self,
        *,
        checkpoint_path: str | Path,
        base_model: str = DEFAULT_BASE_MODEL,
        device: str = "auto",
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        torch_dtype: str = "bfloat16",
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        do_sample: bool = True,
        prompt_name: str = "rideshare_system",
        verbose: bool = False,
    ) -> None:
        try:
            import torch
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                BitsAndBytesConfig,
            )
        except ImportError as e:
            raise RuntimeError(
                "transformers + torch + bitsandbytes required. "
                "Install with: pip install -e '.[training]'"
            ) from e

        self.checkpoint_path = str(checkpoint_path)
        self.base_model = base_model
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.do_sample = do_sample
        self.prompt_name = prompt_name
        self.verbose = verbose

        # Tokenizer (always from base model — fine-tunes don't change vocab).
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model, trust_remote_code=False,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Model.
        load_kwargs: dict[str, Any] = {
            "device_map": device,
            "torch_dtype": getattr(torch, torch_dtype, torch.bfloat16),
        }
        if load_in_4bit:
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        elif load_in_8bit:
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

        # Detect whether checkpoint_path is a LoRA adapter directory or a full
        # model. LoRA dirs have adapter_config.json but no full model weights —
        # they need the base model loaded first, then the adapter applied on top
        # via peft.PeftModel. Full-FT checkpoints can be loaded directly.
        adapter_config_path = Path(self.checkpoint_path) / "adapter_config.json"
        is_lora_adapter = adapter_config_path.exists()

        if is_lora_adapter:
            with open(adapter_config_path, encoding="utf-8") as f:
                adapter_config = json.load(f)
            base_model_name = adapter_config.get(
                "base_model_name_or_path", self.base_model,
            )
            if self.verbose:
                print(f"[trained_local] Loading LoRA adapter from "
                      f"{self.checkpoint_path} on base {base_model_name}")
            base = AutoModelForCausalLM.from_pretrained(
                base_model_name, **load_kwargs,
            )
            try:
                from peft import PeftModel
            except ImportError as e:
                raise RuntimeError(
                    "peft is required to load LoRA-adapter checkpoints. "
                    "Install with: pip install -e '.[training]'"
                ) from e
            self.model = PeftModel.from_pretrained(base, self.checkpoint_path)
        else:
            if self.verbose:
                print(f"[trained_local] Loading full model from "
                      f"{self.checkpoint_path}")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.checkpoint_path, **load_kwargs,
            )
        self.model.eval()
        self._torch = torch

    # ------------------------------------------------------------------ #
    # Episode loop
    # ------------------------------------------------------------------ #

    def run(
        self,
        env: GymEnvironment,
        *,
        on_step=None,
        on_event=None,
    ) -> Trajectory:
        obs, info = env.reset()

        rec = TrajectoryRecorder(
            task_id=info["task_id"],
            seed=env.task.seed,
            ground_truth=env.ground_truth,
            perturbations=info["perturbations"],
            meta={
                "agent_id": "trained_local",
                "model": self.checkpoint_path,
                "base_model": self.base_model,
                "max_steps": env.task.max_steps,
                "temperature": self.temperature,
            },
        )
        rec.set_initial(info["initial_state_hash"])

        system = load_prompt(self.prompt_name)
        tools_anthropic = env.tool_registry.to_anthropic()
        # Qwen tool-calling chat template uses OpenAI-style tool schema.
        tools_openai = env.tool_registry.to_openai()
        initial_user = obs.to_agent_message()

        rec.set_episode_context(
            system_prompt=system,
            tools_serialized=tools_anthropic,
            initial_user_message=initial_user,
        )

        if on_event:
            on_event({"event": "episode_start", "task_id": info["task_id"]})

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": initial_user},
        ]

        success = False
        error_category: str | None = None

        for step in range(env.task.max_steps):
            if on_event:
                on_event({"event": "model_start", "step": step})

            # Render via the model's chat template with tool definitions.
            try:
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tools=tools_openai,
                    add_generation_prompt=True,
                    tokenize=False,
                )
            except Exception:
                # Some chat templates don't accept tools=; fall back to
                # injecting tool definitions into the system message.
                tool_block = json.dumps(tools_openai, indent=2)
                fallback_messages = [
                    {"role": "system",
                     "content": system + "\n\nAvailable tools:\n" + tool_block},
                    *messages[1:],
                ]
                prompt = self.tokenizer.apply_chat_template(
                    fallback_messages, add_generation_prompt=True, tokenize=False,
                )

            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            # transformers raises if do_sample=True with temperature=0. Fall
            # back to greedy decoding when the user asked for temperature=0
            # (deterministic eval).
            _do_sample = bool(self.do_sample and (self.temperature or 0) > 0)
            _temperature = self.temperature if (self.temperature or 0) > 0 else 1.0
            try:
                with self._torch.inference_mode():
                    out_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=_do_sample,
                        temperature=_temperature,
                        top_p=0.9,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )
            except Exception as e:
                # Always print the traceback to stderr so silent crashes
                # don't waste a 12h eval run. The trajectory still records
                # error_category="crashed" for downstream analysis.
                import sys, traceback
                print(f"[trained_local] generate() raised on step {step}: "
                       f"{type(e).__name__}: {e}", file=sys.stderr, flush=True)
                traceback.print_exc(file=sys.stderr)
                error_category = "crashed"
                if on_event:
                    on_event({"event": "error", "step": step,
                              "error": f"{type(e).__name__}: {e}"})
                break

            new_tokens = out_ids[0, inputs["input_ids"].shape[1]:]
            generated = self.tokenizer.decode(new_tokens, skip_special_tokens=False)

            usage_in = int(inputs["input_ids"].shape[1])
            usage_out = int(new_tokens.shape[0])

            tool_calls = parse_tool_calls(generated)

            assistant_record: dict[str, Any] = {
                "role": "assistant",
                "content": generated.strip(),
            }
            messages.append(assistant_record)

            if not tool_calls:
                if self.verbose:
                    print(f"[step {step}] no tool_call -- model said: "
                          f"{_truncate(generated)}")
                if on_event:
                    on_event({"event": "no_tool_calls", "step": step,
                              "content": generated[:500]})
                if env.step_count == 0:
                    error_category = "wrong_tool"
                else:
                    error_category = error_category or "goal_incomplete"
                break

            terminated_any = False
            truncated_any = False
            for i, tc in enumerate(tool_calls):
                tcid = f"call_{step}_{i}"
                call = ToolCall(
                    name=tc["name"], arguments=tc["arguments"], tool_use_id=tcid,
                )
                obs, reward, terminated, truncated, env_info = env.step(call)

                tool_message_content = json.dumps({
                    "text": obs.text[:500],
                    "data": obs.data,
                    "tool_error": env_info.get("tool_error"),
                }, default=str)
                tool_result_msg = {
                    "role": "tool",
                    "tool_call_id": tcid,
                    "content": tool_message_content,
                }
                rec.record(
                    action=call, observation=obs, reward=reward,
                    terminated=terminated, truncated=truncated,
                    latency_ms=env_info.get("tool_latency_ms", 0.0),
                    info=env_info,
                    tokens_in=usage_in, tokens_out=usage_out,
                    assistant_message=assistant_record,
                    tool_result_message=tool_result_msg,
                )
                messages.append(tool_result_msg)

                if on_step:
                    on_step(step, call, obs, reward, terminated, env_info)

                if self.verbose:
                    print(f"[step {step}] {call.name}({_truncate(call.arguments, 160)}) "
                          f"-> reward={reward:.3f} term={terminated} "
                          f"ok={env_info['tool_ok']}")

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
