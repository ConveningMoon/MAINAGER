"""A guarding proxy in front of the provider's own MCP server.

The provider ships an MCP connector for external chat clients. It forwards nine
tools, two of which spend money, and neither of those declares `strict` — so an
agent reading the schema has no way to know the protection exists. This proxy
sits in front of it and forwards everything unchanged except for the guarding
described in `mainager.guard.mcp_guard`.

It is deliberately a proxy and not a replacement. Every tool the provider adds
appears here automatically; the only tools whose behaviour changes are the two
that can produce a charge.

Run it with `python mcp_server.py`. It speaks stdio downstream and streamable
HTTP upstream.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import mcp.types as types
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcp.server import Server
from mcp.server.stdio import stdio_server

from mainager.config import Settings
from mainager.guard.mcp_guard import (
    CHARGING_TOOL,
    GENERATION_TOOLS,
    augment_tool_schema,
    ceiling_refusal,
    guard_call,
    read_verdict,
)
from mainager.preflight.registry import ModelRegistry
from mainager.providers.vibemarketolog.capabilities import read_snapshot

UPSTREAM_URL = "https://lk.vibemarketolog.ru/api/mcp"
ESTIMATE_TOOL = "estimate_generation"

_GUARD_NOTE = (
    "\n\nProxied through MAINAGER: strict defaults to true, the verdict is "
    "derived from `rejected` rather than `valid`, and a spend ceiling is "
    "enforced before any charge."
)


def _text_blocks(result: types.CallToolResult) -> str:
    return "".join(block.text for block in result.content if isinstance(block, types.TextContent))


def _json_payload(result: types.CallToolResult) -> dict[str, Any] | None:
    try:
        parsed = json.loads(_text_blocks(result))
    except (ValueError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _refuse(reason: str, notes: tuple[str, ...] = ()) -> types.CallToolResult:
    lines = [f"refused by MAINAGER: {reason}", *notes]
    return types.CallToolResult(
        content=[types.TextContent(type="text", text="\n".join(lines))],
        isError=True,
    )


def _annotate(result: types.CallToolResult, notes: tuple[str, ...]) -> types.CallToolResult:
    """Prepend the guard's notes so no rewrite is invisible to the caller."""
    if not notes:
        return result
    header = types.TextContent(type="text", text="\n".join(f"[mainager] {n}" for n in notes))
    return types.CallToolResult(
        content=[header, *result.content],
        structuredContent=result.structuredContent,
        isError=result.isError,
    )


def _estimate_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    """Trim a generate_content call down to what estimate_generation accepts."""
    return {
        key: value
        for key, value in arguments.items()
        if key in {"type", "model", "prompt", "params", "strict"}
    }


class GuardingProxy:
    """Forwards the upstream tool surface, guarding the two that can charge."""

    def __init__(
        self,
        upstream: ClientSession,
        registry: ModelRegistry,
        *,
        ceiling_rub: float | None = None,
    ) -> None:
        self._upstream = upstream
        self._registry = registry
        self._ceiling_rub = ceiling_rub

    async def list_tools(self) -> list[types.Tool]:
        listing = await self._upstream.list_tools()
        tools: list[types.Tool] = []
        for tool in listing.tools:
            schema = augment_tool_schema(tool.name, tool.inputSchema)
            description = tool.description or ""
            if tool.name in GENERATION_TOOLS:
                description += _GUARD_NOTE
            tools.append(
                tool.model_copy(update={"inputSchema": schema, "description": description})
            )
        return tools

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> types.CallToolResult:
        guarded = guard_call(name, arguments, self._registry)

        if name != CHARGING_TOOL:
            result = await self._upstream.call_tool(name, guarded.arguments)
            if name == ESTIMATE_TOOL:
                return _annotate(result, guarded.notes + self._verdict_notes(result))
            return _annotate(result, guarded.notes)

        return await self._call_charging_tool(guarded.arguments, guarded.notes)

    async def _call_charging_tool(
        self, arguments: dict[str, Any], notes: tuple[str, ...]
    ) -> types.CallToolResult:
        """Price the call before forwarding it, and refuse rather than charge."""
        probe = await self._upstream.call_tool(ESTIMATE_TOOL, _estimate_arguments(arguments))
        payload = _json_payload(probe)
        if payload is None:
            return _refuse(
                "the upstream dry-run did not return a readable estimate, so the "
                "cost of this call is unknown",
                notes,
            )

        verdict = read_verdict(payload)

        reason = verdict.refusal_reason()
        if reason is not None and arguments.get("strict") is not False:
            return _refuse(reason, notes)

        over_ceiling = ceiling_refusal(verdict, self._ceiling_rub)
        if over_ceiling is not None:
            return _refuse(over_ceiling, notes)

        result = await self._upstream.call_tool(CHARGING_TOOL, arguments)
        charged = (f"charged {verdict.cost_rub:g} RUB (estimated before the call)",)
        return _annotate(result, notes + charged)

    def _verdict_notes(self, result: types.CallToolResult) -> tuple[str, ...]:
        payload = _json_payload(result)
        if payload is None:
            return ()
        verdict = read_verdict(payload)
        if verdict.safe_to_charge:
            return ()
        reason = verdict.refusal_reason()
        if reason is None:
            return ()
        if verdict.provider_valid is True and not verdict.body_is_clean:
            return (
                f"upstream reported valid=true but would drop "
                f"{', '.join(verdict.rejected)}; treat this as NOT safe to generate",
            )
        return (reason,)


def build_server(proxy: GuardingProxy) -> Server[object, object]:
    server: Server[object, object] = Server("mainager-guard")

    # The runtime's registration decorators carry no annotations, so strict mode
    # cannot see through them. The handlers themselves are fully typed.
    @server.list_tools()  # type: ignore[no-untyped-call, untyped-decorator]
    async def _list_tools() -> list[types.Tool]:
        return await proxy.list_tools()

    @server.call_tool()  # type: ignore[untyped-decorator]
    async def _call_tool(name: str, arguments: dict[str, Any]) -> types.CallToolResult:
        return await proxy.call_tool(name, arguments)

    return server


@asynccontextmanager
async def upstream_session(settings: Settings) -> AsyncIterator[ClientSession]:
    """Connect to the provider's MCP server with the configured bearer token."""
    async with (
        streamablehttp_client(UPSTREAM_URL, headers=settings.auth_header) as (
            read_stream,
            write_stream,
            _,
        ),
        ClientSession(read_stream, write_stream) as session,
    ):
        await session.initialize()
        yield session


async def serve(settings: Settings) -> None:
    registry = ModelRegistry.from_capabilities(read_snapshot(settings.data_dir))
    async with upstream_session(settings) as upstream:
        proxy = GuardingProxy(upstream, registry, ceiling_rub=settings.plan_ceiling_rub)
        server = build_server(proxy)
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, server.create_initialization_options())
