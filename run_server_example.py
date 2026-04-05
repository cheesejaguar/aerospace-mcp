#!/usr/bin/env python3
"""Example script demonstrating how to run the Aerospace MCP Server.

This script launches the low-level MCP server (``aerospace_mcp.server``) which
communicates via the Model Context Protocol over stdio or TCP transports. It
adds the project root to ``sys.path`` so the package can be imported without
installation.

For production use, prefer the installed console_scripts entry point:
    ``aerospace-mcp``          (FastMCP server — recommended)
    ``aerospace-mcp-http``     (FastAPI HTTP server)

Requirements:
    pip install mcp fastapi openap airportsdata geographiclib pydantic uvicorn

Usage:
    # Run with stdio transport (default for MCP clients)
    python run_server_example.py

    # Run with TCP transport for debugging
    python run_server_example.py --tcp localhost:8000

WARNING:
    This module is for educational and research purposes only.
    Do NOT use for real flight planning, navigation, or aircraft operations.
"""

import os
import sys

# Add the project root to sys.path so ``aerospace_mcp`` can be imported
# directly without requiring a pip install.
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

if __name__ == "__main__":
    # Import and start the low-level MCP server (stdio/TCP transport)
    from aerospace_mcp.server import run

    run()
