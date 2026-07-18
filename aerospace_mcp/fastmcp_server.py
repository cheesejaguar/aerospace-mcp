"""FastMCP server implementation for Aerospace flight planning tools.

This module provides the MCP (Model Context Protocol) server using the
FastMCP framework, which generates JSON schemas automatically from Python
function signatures and docstrings.

Concurrency note:
    All registered tool functions are synchronous.  FastMCP executes sync
    tools in a worker thread (via anyio), so long-running numeric work
    (orbit propagation, Monte Carlo, GA/PSO) does not block the server's
    event loop for other clients.

Architecture:
    1. Tool functions are defined in ``aerospace_mcp/tools/`` submodules, each
       decorated with type hints and Google-style docstrings.
    2. This module imports all tool functions and registers them with the FastMCP
       server instance via ``mcp.tool(func)``.
    3. FastMCP automatically generates MCP-compatible JSON schemas, handles
       argument parsing, and routes incoming tool calls to the correct handler.

Transport Modes:
    - **stdio** (default): Standard input/output for production MCP clients.
      Used when launched as ``aerospace-mcp`` from the command line.
    - **SSE** (Server-Sent Events): HTTP-based transport for debugging and
      browser-based MCP clients. Activated via ``aerospace-mcp sse [host] [port]``.

Deferred Tool Loading:
    The server supports Anthropic's deferred tool loading pattern. Discovery
    tools (``search_aerospace_tools``, ``list_tool_categories``) are loaded
    eagerly, while domain-specific tools can be deferred to save context window
    space. See the TOOL REGISTRATION section below for configuration details.

Loads environment from .env before importing tools so feature flags and
API keys are available at import time.

WARNING:
    This module is for educational and research purposes only.
    Do NOT use for real flight planning, navigation, or aircraft operations.
"""

import logging
import sys

# Load environment from .env before importing tool modules
try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

from fastmcp import FastMCP

# Single source of truth for tool registration: aerospace_mcp/tools/registry.py
from .tools.registry import ALL_TOOLS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP("aerospace-mcp")

# =============================================================================
# TOOL REGISTRATION
# =============================================================================
# aerospace-mcp supports deferred tool loading for efficient context usage.
# When using with Anthropic's API, configure the mcp_toolset like this:
#
#   {
#     "type": "mcp_toolset",
#     "mcp_server_name": "aerospace-mcp",
#     "default_config": {"defer_loading": true},
#     "configs": {
#       "search_aerospace_tools": {"defer_loading": false},
#       "list_tool_categories": {"defer_loading": false}
#     }
#   }
#
# This loads only the discovery tools initially. Claude uses search_aerospace_tools
# to find relevant tools, which returns tool_reference blocks that the API
# automatically expands into full tool definitions.
# =============================================================================

# All tools (discovery tools first) come from the shared registry, so the
# server, CLI, and search metadata can never drift apart.
for _tool_fn in ALL_TOOLS.values():
    mcp.tool(_tool_fn)


def run():
    """Start the FastMCP aerospace tools server.

    Selects the transport mode based on command-line arguments:
        - No args (default): stdio mode for production MCP clients.
        - ``sse [host] [port]``: SSE (Server-Sent Events) mode for HTTP-based
          MCP clients and browser debugging. Defaults to localhost:8001.

    This function is the console_scripts entry point registered as
    ``aerospace-mcp`` in pyproject.toml.
    """
    # Check for SSE mode — useful for debugging with browser-based MCP clients
    if len(sys.argv) > 1 and sys.argv[1] == "sse":
        host = sys.argv[2] if len(sys.argv) > 2 else "localhost"
        port = int(sys.argv[3]) if len(sys.argv) > 3 else 8001
        logger.info(f"Starting FastMCP server in SSE mode on {host}:{port}")
        mcp.run(transport="sse", host=host, port=port)
    else:
        # stdio mode: MCP client communicates over stdin/stdout
        logger.info("Starting FastMCP server in stdio mode")
        mcp.run()


if __name__ == "__main__":
    run()
