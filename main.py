"""Compatibility shim for ``uvicorn main:app`` from a source checkout.

The FastAPI application lives in :mod:`aerospace_mcp.http_app` so that it
ships inside the installed package (``aerospace-mcp-http`` imports it from
there); this module only re-exports it.
"""

from aerospace_mcp.http_app import app

__all__ = ["app"]
