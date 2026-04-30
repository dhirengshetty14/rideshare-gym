"""GymEnvironment — Gymnasium-compatible wrapper around (Task, Sandbox).

The agent's action space is a `ToolCall`; the observation space is `Observation`.
Both are `gym.spaces.Dict`-shaped so a strictly-typed Gymnasium consumer is happy,
but in practice LLM agents bypass the gym `space` machinery and just exchange
JSON/text directly.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rideshare_gym.core.sandbox import Sandbox
from rideshare_gym.core.task import AbstractTask, InitialState
from rideshare_gym.core.tools import ToolRegistry
from rideshare_gym.core.types import Observation, ToolCall, ToolResult
from rideshare_gym.core.verifier import Verifier, state_hash


class GymEnvironment(gym.Env):  # type: ignore[misc]
    """Standard Gymnasium env: `reset()` and `step(action)`. Owned by the
    agent runner. One env wraps one Task instance + one Sandbox tenant."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        task: AbstractTask,
        sandbox_factory: Callable[[], Sandbox],
        *,
        record_state_hash: bool = True,
    ) -> None:
        self.task = task
        self._sandbox_factory = sandbox_factory
        self._sandbox: Sandbox | None = None
        self._verifier: Verifier | None = None
        self._initial: InitialState | None = None
        self._step_count = 0
        self._record_state_hash = record_state_hash
        self._initial_state_hash: str = ""
        self._final_state_hash: str = ""

        # Spaces are defined for Gymnasium API compliance but LLM agents
        # exchange typed pydantic objects directly via observation/action.
        self.action_space = spaces.Dict({
            "name": spaces.Text(max_length=64),
            "arguments": spaces.Text(max_length=8192),
        })
        self.observation_space = spaces.Dict({
            "text": spaces.Text(max_length=32768),
            "data": spaces.Text(max_length=65536),
        })

    # ------------------------------------------------------------------ #
    # Gymnasium API
    # ------------------------------------------------------------------ #

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Observation, dict[str, Any]]:
        if seed is not None:
            self.task.seed = seed
            np.random.seed(seed)
        if self._sandbox is not None:
            self.task.teardown(self._sandbox)
        self._sandbox = self._sandbox_factory()
        self._sandbox.reset()
        self._initial = self.task.setup(self._sandbox)
        self._verifier = self.task.verifier()
        self._step_count = 0
        if self._record_state_hash:
            self._initial_state_hash = state_hash(self._sandbox.snapshot())
        else:
            self._initial_state_hash = ""

        obs = Observation(
            text=self.task.goal_prompt(self._initial),
            data=self._initial.snapshot,
        )
        info = {
            "task_id": self.task.task_id,
            "difficulty": self.task.difficulty,
            "max_steps": self.task.max_steps,
            "tools": self.tool_registry.names(),
            "initial_state_hash": self._initial_state_hash,
            "perturbations": [
                p.model_dump() if hasattr(p, "model_dump") else dict(p.__dict__)
                for p in self.task.perturbations
            ],
        }
        return obs, info

    def step(
        self,
        action: ToolCall | dict[str, Any],
    ) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        if self._sandbox is None or self._verifier is None or self._initial is None:
            raise RuntimeError("env.step() called before env.reset()")

        if isinstance(action, dict):
            action = ToolCall(**action)

        self._step_count += 1

        result: ToolResult = self.tool_registry.dispatch(self._sandbox, action)
        v = self._verifier.validate(self._sandbox)

        terminated = bool(v.done)
        truncated = self._step_count >= self.task.max_steps
        reward = float(v.reward) - self.task.step_penalty

        if terminated or truncated:
            self._final_state_hash = (
                state_hash(self._sandbox.snapshot()) if self._record_state_hash else ""
            )

        obs = Observation(
            text=result.summary or ("ok" if result.ok else (result.error or "error")),
            data=result.payload,
        )
        info = {
            "tool_ok": result.ok,
            "tool_error": result.error,
            "tool_latency_ms": result.latency_ms,
            "step": self._step_count,
            "verifier": v.info,
            "final_state_hash": self._final_state_hash,
        }
        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        if self._sandbox is not None:
            try:
                self.task.teardown(self._sandbox)
            except Exception:
                pass  # never let teardown failure mask the episode result
            try:
                self._sandbox.teardown()
            except Exception:
                pass
            self._sandbox = None

    # ------------------------------------------------------------------ #
    # Convenience
    # ------------------------------------------------------------------ #

    @property
    def tool_registry(self) -> ToolRegistry:
        return self.task.tool_registry

    @property
    def sandbox(self) -> Sandbox:
        if self._sandbox is None:
            raise RuntimeError("sandbox not yet initialised; call reset() first")
        return self._sandbox

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def initial_state_hash(self) -> str:
        return self._initial_state_hash

    @property
    def final_state_hash(self) -> str:
        return self._final_state_hash

    @property
    def ground_truth(self) -> dict[str, Any]:
        return self._initial.ground_truth if self._initial else {}
