"""FastAPI HTTP Server Application for Aerospace MCP.

This module provides the HTTP interface entry point for the aerospace flight
planning system. It wraps the FastAPI application defined in ``main`` and
exposes a ``run()`` function that starts a uvicorn ASGI server with
environment-driven configuration.

Configuration (environment variables):
    AEROSPACE_MCP_HOST: Bind address (default: "0.0.0.0")
    AEROSPACE_MCP_PORT: Listen port (default: 8080)
    AEROSPACE_MCP_LOG_LEVEL: Logging verbosity (default: "info")
    AEROSPACE_MCP_ENV: "development" enables auto-reload (default: "production")

The module also ensures environment variables from a local ``.env`` file are
loaded before any other imports.

WARNING:
    This module is for educational and research purposes only.
    Do NOT use for real flight planning, navigation, or aircraft operations.
"""

# Load environment from .env as early as possible
try:  # Prefer optional dependency without hard requirement
    from dotenv import load_dotenv

    load_dotenv()  # Loads from .env in CWD or project root if present
except Exception:
    pass

from main import app

# Re-export the app for use by the package
__all__ = ["app", "run"]


def run() -> None:
    """Start the FastAPI server using uvicorn with environment-driven configuration.

    Reads host, port, log level, and environment mode from environment
    variables. In development mode, enables auto-reload for rapid iteration.
    In production mode, uses uvloop for improved async performance and
    allows multiple workers.

    This function is the console_scripts entry point registered as
    ``aerospace-mcp-http`` in pyproject.toml.
    """
    import os

    import uvicorn

    # Get configuration from environment variables
    host = os.getenv("AEROSPACE_MCP_HOST", "0.0.0.0")
    port = int(os.getenv("AEROSPACE_MCP_PORT", "8080"))
    log_level = os.getenv("AEROSPACE_MCP_LOG_LEVEL", "info").lower()

    # Disable reload in production (when not in development)
    reload = os.getenv("AEROSPACE_MCP_ENV", "production").lower() == "development"

    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=reload,
        log_level=log_level,
        access_log=True,
        # Additional production settings
        workers=1 if reload else None,  # Single worker when reloading
        loop="uvloop" if not reload else "auto",  # Use uvloop for better performance
    )


if __name__ == "__main__":
    run()
