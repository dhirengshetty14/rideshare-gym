"""Trajectory recorder — captures (action, observation, reward, ...) tuples
for every step. Output JSONL files are directly consumable for SFT, DPO, and
reward-model training.
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from rideshare_gym.core.types import Observation, ToolCall


class Step(BaseModel):
    """One env step: agent action → observed result."""

    index: int
    action: ToolCall
    observation: Observation
    reward: float
    terminated: bool
    truncated: bool = False
    latency_ms: float = 0.0
    """Wall-clock latency of the env.step call (tool dispatch + verifier)."""
    tokens_in: int | None = None
    tokens_out: int | None = None
    """Optional model usage (filled by the agent runner, not the env)."""
    info: dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    # === SFT / DPO data capture (populated by LLM-driven agents only) === #
    assistant_message: dict[str, Any] | None = None
    """The full assistant turn the model emitted at this step:
        {"role": "assistant", "content": "...", "tool_calls": [...]}
    Captured by the agent runner. Populated for litellm / claude_baseline /
    trained_local; None for gold_oracle. Used to reconstruct (input, target)
    pairs for supervised fine-tuning."""
    tool_result_message: dict[str, Any] | None = None
    """The tool-result message we sent BACK to the model after env.step.
    Shape: {"role": "tool", "tool_call_id": "...", "content": "..."}
    Combined with `assistant_message` lets SFT walk the conversation.
    None for gold_oracle."""


class Trajectory(BaseModel):
    """Full episode record."""

    episode_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    task_id: str
    seed: int = 0
    initial_state_hash: str = ""
    final_state_hash: str = ""
    steps: list[Step] = Field(default_factory=list)
    final_reward: float = 0.0
    success: bool = False
    """`True` iff the episode terminated (verifier `done=True`) before truncation."""
    error_category: str | None = None
    """One of: goal_incomplete | wrong_tool | wrong_args | crashed | None (success)."""
    ground_truth: dict[str, Any] = Field(default_factory=dict)
    perturbations: list[dict[str, Any]] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)
    """agent_id, model, wall_time_seconds, total_tokens_in/out, etc."""
    started_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    finished_at: str | None = None
    # === SFT / DPO data capture (constant per episode) === #
    system_prompt: str | None = None
    """The system prompt the LLM saw. None for gold_oracle."""
    tools_serialized: list[dict[str, Any]] | None = None
    """JSON-Schema tool definitions exposed to the LLM (Anthropic-shape).
    Combined with system_prompt + step assistant/tool messages gives the
    full conversation log."""
    initial_user_message: str | None = None
    """The first user-role message (goal prompt + initial state observation)."""

    def to_jsonl(self) -> str:
        return self.model_dump_json() + "\n"

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_jsonl(), encoding="utf-8")


class TrajectoryRecorder:
    """Builds a Trajectory step-by-step. Owned by the agent runner; the
    GymEnvironment doesn't know about it (keeps Gymnasium API pure)."""

    def __init__(
        self,
        task_id: str,
        seed: int = 0,
        ground_truth: dict[str, Any] | None = None,
        perturbations: list[dict[str, Any]] | None = None,
        meta: dict[str, Any] | None = None,
    ) -> None:
        self._traj = Trajectory(
            task_id=task_id,
            seed=seed,
            ground_truth=ground_truth or {},
            perturbations=perturbations or [],
            meta=meta or {},
        )
        self._t0 = time.perf_counter()
        self._step_idx = 0

    def set_initial(self, state_hash: str) -> None:
        self._traj.initial_state_hash = state_hash

    def record(
        self,
        action: ToolCall,
        observation: Observation,
        reward: float,
        terminated: bool,
        truncated: bool = False,
        latency_ms: float = 0.0,
        info: dict[str, Any] | None = None,
        tokens_in: int | None = None,
        tokens_out: int | None = None,
        assistant_message: dict[str, Any] | None = None,
        tool_result_message: dict[str, Any] | None = None,
    ) -> Step:
        step = Step(
            index=self._step_idx,
            action=action,
            observation=observation,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            latency_ms=latency_ms,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            info=info or {},
            assistant_message=assistant_message,
            tool_result_message=tool_result_message,
        )
        self._traj.steps.append(step)
        self._step_idx += 1
        return step

    def set_episode_context(
        self,
        *,
        system_prompt: str | None = None,
        tools_serialized: list[dict[str, Any]] | None = None,
        initial_user_message: str | None = None,
    ) -> None:
        """Captured once per episode by the LLM-driven agent runner.
        Required for SFT / DPO data extraction; ignored for gold_oracle."""
        if system_prompt is not None:
            self._traj.system_prompt = system_prompt
        if tools_serialized is not None:
            self._traj.tools_serialized = tools_serialized
        if initial_user_message is not None:
            self._traj.initial_user_message = initial_user_message

    def finalize(
        self,
        *,
        final_state_hash: str = "",
        success: bool = False,
        error_category: str | None = None,
    ) -> Trajectory:
        self._traj.finished_at = datetime.now(timezone.utc).isoformat()
        self._traj.final_state_hash = final_state_hash
        self._traj.final_reward = self._traj.steps[-1].reward if self._traj.steps else 0.0
        self._traj.success = success
        self._traj.error_category = error_category
        wall = time.perf_counter() - self._t0
        self._traj.meta.setdefault("wall_time_seconds", round(wall, 3))
        self._traj.meta.setdefault("n_steps", len(self._traj.steps))
        if any(s.tokens_in for s in self._traj.steps):
            self._traj.meta["total_tokens_in"] = sum(s.tokens_in or 0 for s in self._traj.steps)
            self._traj.meta["total_tokens_out"] = sum(s.tokens_out or 0 for s in self._traj.steps)
        return self._traj

    @property
    def trajectory(self) -> Trajectory:
        return self._traj


def write_run_index(run_dir: Path, trajectories: list[Trajectory]) -> Path:
    """Emit a single `index.jsonl` aggregating all episodes in a run.
    Useful for downstream analysis without crawling the trajectory dir."""
    run_dir.mkdir(parents=True, exist_ok=True)
    out = run_dir / "index.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for t in trajectories:
            f.write(json.dumps({
                "episode_id": t.episode_id,
                "task_id": t.task_id,
                "seed": t.seed,
                "success": t.success,
                "final_reward": t.final_reward,
                "n_steps": len(t.steps),
                "error_category": t.error_category,
                "wall_time_seconds": t.meta.get("wall_time_seconds"),
                "total_tokens_in": t.meta.get("total_tokens_in"),
                "total_tokens_out": t.meta.get("total_tokens_out"),
            }) + "\n")
    return out
