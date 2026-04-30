"""ToolSpec and ToolRegistry — convert Python callables into LLM-callable tools.

Each tool has a JSON-Schema input contract (Anthropic-compatible). The registry
emits the exact `tools=[...]` block consumed by `Anthropic.messages.create`,
plus a parallel OpenAI shape.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import jsonschema

from rideshare_gym.core.types import ToolCall, ToolResult


@dataclass
class ToolSpec:
    """One tool the agent can call. Wraps a handler with a JSON-Schema contract."""

    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[[Any, dict[str, Any]], ToolResult]
    """Signature: `handler(sandbox, args) -> ToolResult`. Sandbox is opaque."""

    def validate(self, args: dict[str, Any]) -> tuple[bool, str | None]:
        """Validate arguments against the JSON Schema. Returns (ok, error)."""
        try:
            jsonschema.validate(instance=args, schema=self.input_schema)
            return True, None
        except jsonschema.ValidationError as e:
            return False, e.message

    def to_anthropic(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }

    def to_openai(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }


@dataclass
class ToolRegistry:
    """Holds the set of tools available for a task. Dispatches ToolCalls."""

    specs: dict[str, ToolSpec] = field(default_factory=dict)

    @classmethod
    def from_specs(cls, specs: list[ToolSpec]) -> ToolRegistry:
        return cls(specs={s.name: s for s in specs})

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self.specs:
            raise ValueError(f"Tool {spec.name!r} already registered")
        self.specs[spec.name] = spec

    def get(self, name: str) -> ToolSpec | None:
        return self.specs.get(name)

    def to_anthropic(self) -> list[dict[str, Any]]:
        """Anthropic `tools=[...]` shape. Mark the LAST tool with cache_control
        so the entire tool block becomes part of the prompt cache prefix."""
        out: list[dict[str, Any]] = [s.to_anthropic() for s in self.specs.values()]
        if out:
            out[-1] = {**out[-1], "cache_control": {"type": "ephemeral"}}
        return out

    def to_openai(self) -> list[dict[str, Any]]:
        return [s.to_openai() for s in self.specs.values()]

    def dispatch(self, sandbox: Any, call: ToolCall) -> ToolResult:
        """Validate the ToolCall against schema and invoke the handler."""
        spec = self.specs.get(call.name)
        if spec is None:
            return ToolResult(
                ok=False,
                error=f"unknown_tool: {call.name!r} not in registry",
                summary=f"Tool {call.name!r} does not exist",
            )
        ok, err = spec.validate(call.arguments)
        if not ok:
            return ToolResult(
                ok=False,
                error=f"wrong_args: {err}",
                summary=f"Invalid arguments for {call.name}: {err}",
            )
        t0 = time.perf_counter()
        try:
            result = spec.handler(sandbox, call.arguments)
        except Exception as e:
            return ToolResult(
                ok=False,
                error=f"handler_exception: {type(e).__name__}: {e}",
                summary=f"{call.name} crashed: {e}",
                latency_ms=(time.perf_counter() - t0) * 1000,
            )
        if result.latency_ms == 0.0:
            result.latency_ms = (time.perf_counter() - t0) * 1000
        return result

    def __len__(self) -> int:
        return len(self.specs)

    def names(self) -> list[str]:
        return list(self.specs.keys())
