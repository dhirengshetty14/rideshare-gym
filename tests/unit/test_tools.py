"""Unit tests for ToolSpec / ToolRegistry."""

from __future__ import annotations

import pytest

from rideshare_gym.core.sandbox import StubSandbox
from rideshare_gym.core.tools import ToolRegistry, ToolSpec
from rideshare_gym.core.types import ToolCall, ToolResult


def _ping_handler(sandbox: StubSandbox, args: dict) -> ToolResult:
    sandbox.record_call("ping", args)
    return ToolResult(ok=True, payload={"pong": args.get("msg", "hi")}, summary="pong")


PING = ToolSpec(
    name="ping",
    description="Echo a message back.",
    input_schema={
        "type": "object",
        "required": ["msg"],
        "properties": {"msg": {"type": "string"}},
        "additionalProperties": False,
    },
    handler=_ping_handler,
)


def test_registry_register_and_dispatch():
    reg = ToolRegistry.from_specs([PING])
    sb = StubSandbox()
    result = reg.dispatch(sb, ToolCall(name="ping", arguments={"msg": "hello"}))
    assert result.ok
    assert result.payload == {"pong": "hello"}
    assert result.latency_ms >= 0
    assert sb.call_log == [{"name": "ping", "args": {"msg": "hello"}}]


def test_unknown_tool():
    reg = ToolRegistry.from_specs([PING])
    sb = StubSandbox()
    result = reg.dispatch(sb, ToolCall(name="nope", arguments={}))
    assert not result.ok
    assert result.error is not None
    assert "unknown_tool" in result.error


def test_wrong_args_schema_violation():
    reg = ToolRegistry.from_specs([PING])
    sb = StubSandbox()
    # Missing required `msg`
    result = reg.dispatch(sb, ToolCall(name="ping", arguments={}))
    assert not result.ok
    assert result.error is not None
    assert "wrong_args" in result.error


def test_handler_exception_caught():
    def boom(sandbox, args):
        raise RuntimeError("kaboom")

    reg = ToolRegistry.from_specs([
        ToolSpec(
            name="boom",
            description="Always crashes.",
            input_schema={"type": "object", "properties": {}},
            handler=boom,
        )
    ])
    result = reg.dispatch(StubSandbox(), ToolCall(name="boom", arguments={}))
    assert not result.ok
    assert result.error is not None
    assert "handler_exception" in result.error


def test_anthropic_serialization_marks_cache():
    reg = ToolRegistry.from_specs([PING])
    out = reg.to_anthropic()
    assert len(out) == 1
    assert out[0]["name"] == "ping"
    assert out[0]["input_schema"]["type"] == "object"
    # Last tool should carry cache_control marker for prompt caching
    assert out[-1].get("cache_control") == {"type": "ephemeral"}


def test_openai_serialization_shape():
    reg = ToolRegistry.from_specs([PING])
    out = reg.to_openai()
    assert out[0]["type"] == "function"
    assert out[0]["function"]["name"] == "ping"


def test_duplicate_registration_rejected():
    reg = ToolRegistry()
    reg.register(PING)
    with pytest.raises(ValueError):
        reg.register(PING)
