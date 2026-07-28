"""Orchestration of the guarding proxy.

The property that matters most is negative: a call that would be charged for a
body the provider will not honour must never reach the upstream charging tool.
The stub upstream records every call so that can be asserted directly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mcp.types as types
import pytest

from mainager.mcp_proxy import GuardingProxy
from mainager.preflight.registry import ModelRegistry
from mainager.providers.vibemarketolog.capabilities import read_snapshot

SNAPSHOT_DIR = Path(__file__).resolve().parent.parent / "data"

TRAP_ESTIMATE = {
    "valid": True,
    "model": "grok-ttv-10",
    "estimated_cost_rub": 196,
    "rejected": ["image_input"],
    "warnings": ["image_input is for type=image only."],
}
CLEAN_ESTIMATE = {
    "valid": True,
    "model": "grok-itv",
    "estimated_cost_rub": 36,
    "rejected": [],
    "warnings": [],
}


def _result(payload: dict[str, Any]) -> types.CallToolResult:
    return types.CallToolResult(content=[types.TextContent(type="text", text=json.dumps(payload))])


class StubUpstream:
    """Minimal stand-in for the provider's MCP session."""

    def __init__(self, estimate: dict[str, Any]) -> None:
        self._estimate = estimate
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def list_tools(self) -> types.ListToolsResult:
        schema = {
            "type": "object",
            "properties": {"type": {}, "model": {}, "prompt": {}, "params": {}},
            "required": ["type", "model"],
        }
        return types.ListToolsResult(
            tools=[
                types.Tool(name="get_balance", description="balance", inputSchema={}),
                types.Tool(name="estimate_generation", description="dry run", inputSchema=schema),
                types.Tool(name="generate_content", description="charges", inputSchema=schema),
            ]
        )

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> types.CallToolResult:
        self.calls.append((name, arguments))
        if name == "estimate_generation":
            return _result(self._estimate)
        if name == "generate_content":
            return _result({"status": "processing", "generation_id": "gen_x"})
        return _result({"status": "ok", "balance": 600})

    @property
    def charged(self) -> bool:
        return any(name == "generate_content" for name, _ in self.calls)


@pytest.fixture(scope="module")
def registry() -> ModelRegistry:
    return ModelRegistry.from_capabilities(read_snapshot(SNAPSHOT_DIR))


def _text(result: types.CallToolResult) -> str:
    return "\n".join(block.text for block in result.content if isinstance(block, types.TextContent))


async def test_strict_appears_only_on_the_tools_that_can_charge(
    registry: ModelRegistry,
) -> None:
    proxy = GuardingProxy(StubUpstream(CLEAN_ESTIMATE), registry)  # type: ignore[arg-type]

    tools = {t.name: t for t in await proxy.list_tools()}

    assert "strict" in tools["generate_content"].inputSchema["properties"]
    assert "strict" in tools["estimate_generation"].inputSchema["properties"]
    assert "strict" not in (tools["get_balance"].inputSchema or {}).get("properties", {})
    assert "MAINAGER" in (tools["generate_content"].description or "")


async def test_a_trap_body_never_reaches_the_charging_tool(registry: ModelRegistry) -> None:
    upstream = StubUpstream(TRAP_ESTIMATE)
    proxy = GuardingProxy(upstream, registry)  # type: ignore[arg-type]

    result = await proxy.call_tool(
        "generate_content",
        {
            "type": "video",
            "model": "grok-ttv-10",
            "prompt": "a cat",
            "params": {"image_input": ["https://x/a.png"]},
        },
    )

    assert result.isError is True
    assert upstream.charged is False
    assert [name for name, _ in upstream.calls] == ["estimate_generation"]
    assert "image_input" in _text(result)


async def test_a_clean_call_is_forwarded_and_the_cost_reported(
    registry: ModelRegistry,
) -> None:
    upstream = StubUpstream(CLEAN_ESTIMATE)
    proxy = GuardingProxy(upstream, registry, ceiling_rub=100.0)  # type: ignore[arg-type]

    result = await proxy.call_tool(
        "generate_content",
        {
            "type": "video",
            "model": "grok-itv",
            "prompt": "a cat",
            "params": {"duration": 5, "image_urls": ["https://x/a.png"]},
        },
    )

    assert result.isError is not True
    assert upstream.charged is True
    assert "charged 36 RUB" in _text(result)


async def test_the_ceiling_blocks_before_the_charge(registry: ModelRegistry) -> None:
    upstream = StubUpstream(CLEAN_ESTIMATE)
    proxy = GuardingProxy(upstream, registry, ceiling_rub=10.0)  # type: ignore[arg-type]

    result = await proxy.call_tool(
        "generate_content",
        {
            "type": "video",
            "model": "grok-itv",
            "prompt": "a cat",
            "params": {"duration": 5, "image_urls": ["https://x/a.png"]},
        },
    )

    assert result.isError is True
    assert upstream.charged is False
    assert "ceiling" in _text(result)


async def test_strict_is_injected_into_the_forwarded_arguments(
    registry: ModelRegistry,
) -> None:
    upstream = StubUpstream(CLEAN_ESTIMATE)
    proxy = GuardingProxy(upstream, registry)  # type: ignore[arg-type]

    await proxy.call_tool(
        "estimate_generation", {"type": "video", "model": "grok-ttv", "prompt": "a cat"}
    )

    _, forwarded = upstream.calls[0]
    assert forwarded["strict"] is True


async def test_an_unreadable_estimate_refuses_rather_than_guessing(
    registry: ModelRegistry,
) -> None:
    class Unreadable(StubUpstream):
        async def call_tool(self, name: str, arguments: dict[str, Any]) -> types.CallToolResult:
            self.calls.append((name, arguments))
            return types.CallToolResult(
                content=[types.TextContent(type="text", text="upstream is down")]
            )

    upstream = Unreadable(CLEAN_ESTIMATE)
    proxy = GuardingProxy(upstream, registry)  # type: ignore[arg-type]

    result = await proxy.call_tool(
        "generate_content",
        {
            "type": "video",
            "model": "grok-itv",
            "prompt": "a cat",
            "params": {"duration": 5, "image_urls": ["https://x/a.png"]},
        },
    )

    assert result.isError is True
    assert upstream.charged is False
    assert "cost of this call is unknown" in _text(result)


async def test_estimate_flags_a_valid_true_response_that_would_drop_a_parameter(
    registry: ModelRegistry,
) -> None:
    proxy = GuardingProxy(StubUpstream(TRAP_ESTIMATE), registry)  # type: ignore[arg-type]

    result = await proxy.call_tool(
        "estimate_generation",
        {
            "type": "video",
            "model": "grok-ttv-10",
            "prompt": "a cat",
            "params": {"image_input": ["https://x/a.png"]},
        },
    )

    text = _text(result)
    assert "NOT safe to generate" in text
    assert "image_input" in text


async def test_read_only_tools_pass_through_unchanged(registry: ModelRegistry) -> None:
    upstream = StubUpstream(CLEAN_ESTIMATE)
    proxy = GuardingProxy(upstream, registry)  # type: ignore[arg-type]

    result = await proxy.call_tool("get_balance", {})

    assert upstream.calls == [("get_balance", {})]
    assert "600" in _text(result)
