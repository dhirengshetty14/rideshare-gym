"""Shared dataclasses used across the gym core (avoids circular imports)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    """A structured action emitted by an LLM agent."""

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    tool_use_id: str | None = None


class ToolResult(BaseModel):
    """Result of dispatching a ToolCall to a Sandbox."""

    ok: bool
    payload: dict[str, Any] = Field(default_factory=dict)
    summary: str = ""
    error: str | None = None
    latency_ms: float = 0.0


class Observation(BaseModel):
    """What the env returns to the agent at each step.

    `data` carries the structured payload (JSON-serialisable). It was originally
    named `json` but that shadowed Pydantic's deprecated `.json()` method.
    """

    text: str
    data: dict[str, Any] = Field(default_factory=dict)

    def to_agent_message(self) -> str:
        """Render this observation into the user-message form an LLM sees."""
        if not self.data:
            return self.text
        import json as _json
        return f"{self.text}\n\nState:\n{_json.dumps(self.data, indent=2, default=str)}"
