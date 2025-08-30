"""FastAPI HTTP Server Application for Aerospace MCP.

This module provides a wrapper around the main FastAPI application
defined in main.py, making it available as a proper package.
"""

from main import app

# Re-export the app for use by the package
__all__ = ["app", "run"]


def run() -> None:
    """Run the FastAPI server using uvicorn."""
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
        loop="uvloop" if not reload else "auto"  # Use uvloop for better performance
    )


if __name__ == "__main__":
    run()
