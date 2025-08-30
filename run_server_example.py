#!/usr/bin/env python3
"""
Example of how to run the Aerospace MCP Server

This script demonstrates the different ways to start the MCP server.

Requirements:
  pip install mcp fastapi openap airportsdata geographiclib pydantic uvicorn

Usage:
  # Run with stdio transport (default for MCP clients)
  python run_server_example.py

  # Run with TCP transport for debugging
  python run_server_example.py --tcp localhost:8000
"""

import sys
import os

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

if __name__ == "__main__":
    # Import and run the server
    from aerospace_mcp.server import run
    run()