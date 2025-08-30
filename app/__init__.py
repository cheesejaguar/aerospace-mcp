"""Aerospace MCP HTTP Server Application.

This package provides the FastAPI HTTP server implementation for the
Aerospace MCP project, offering flight planning and aerospace operations
through RESTful APIs.
"""

__version__ = "0.1.0"
__author__ = "Aaron"
__email__ = "aaron@example.com"

from .main import app

__all__ = ["app", "__version__"]
