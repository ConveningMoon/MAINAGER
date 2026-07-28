"""Entry point for the guarding MCP proxy.

    python mcp_server.py

Speaks stdio to the client and streamable HTTP to the provider's own MCP server,
adding the guards described in `mainager.guard.mcp_guard`.
"""

from __future__ import annotations

import asyncio
import sys

from pydantic import ValidationError

from mainager.config import load_settings


def main() -> int:
    try:
        settings = load_settings()
    except ValidationError as exc:
        print("configuration is incomplete; copy .env.example to .env", file=sys.stderr)
        print(exc, file=sys.stderr)
        return 2

    try:
        from mainager.mcp_proxy import serve
    except ImportError:
        print(
            'the MCP runtime is not installed; run: pip install -e ".[mcp]"',
            file=sys.stderr,
        )
        return 2

    asyncio.run(serve(settings))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
