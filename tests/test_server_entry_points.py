"""Tests for server entry points and run functions."""

import sys
from unittest.mock import MagicMock, patch

import pytest


class TestFastMCPServerRun:
    """Tests for FastMCP server run() function."""

    @pytest.mark.unit
    def test_run_stdio_mode(self):
        """Test run() in default stdio mode."""
        mock_mcp = MagicMock()

        with patch("aerospace_mcp.fastmcp_server.mcp", mock_mcp):
            with patch.object(sys, "argv", ["aerospace-mcp"]):
                from aerospace_mcp.fastmcp_server import run

                run()
                mock_mcp.run.assert_called_once_with()

    @pytest.mark.unit
    def test_run_sse_mode_default_params(self):
        """Test run() in SSE mode with default parameters."""
        mock_mcp = MagicMock()

        with patch("aerospace_mcp.fastmcp_server.mcp", mock_mcp):
            with patch.object(sys, "argv", ["aerospace-mcp", "sse"]):
                from aerospace_mcp.fastmcp_server import run

                run()
                mock_mcp.run.assert_called_once_with(
                    transport="sse", host="localhost", port=8001
                )

    @pytest.mark.unit
    def test_run_sse_mode_custom_host(self):
        """Test run() in SSE mode with custom host."""
        mock_mcp = MagicMock()

        with patch("aerospace_mcp.fastmcp_server.mcp", mock_mcp):
            with patch.object(sys, "argv", ["aerospace-mcp", "sse", "0.0.0.0"]):
                from aerospace_mcp.fastmcp_server import run

                run()
                mock_mcp.run.assert_called_once_with(
                    transport="sse", host="0.0.0.0", port=8001
                )

    @pytest.mark.unit
    def test_run_sse_mode_custom_host_port(self):
        """Test run() in SSE mode with custom host and port."""
        mock_mcp = MagicMock()

        with patch("aerospace_mcp.fastmcp_server.mcp", mock_mcp):
            with patch.object(
                sys, "argv", ["aerospace-mcp", "sse", "127.0.0.1", "9000"]
            ):
                from aerospace_mcp.fastmcp_server import run

                run()
                mock_mcp.run.assert_called_once_with(
                    transport="sse", host="127.0.0.1", port=9000
                )


class TestFastMCPDotenvLoading:
    """Tests for dotenv loading in FastMCP server."""

    @pytest.mark.unit
    def test_dotenv_import_failure(self):
        """Test graceful handling of dotenv import failure."""
        # This tests lines 15-16 in fastmcp_server.py
        # The module should load even if dotenv fails

        # Force reimport with mocked dotenv failure
        with patch.dict("sys.modules", {"dotenv": None}):
            # The module should still be importable
            import aerospace_mcp.fastmcp_server

            # Verify the module loaded successfully
            assert hasattr(aerospace_mcp.fastmcp_server, "mcp")
            assert hasattr(aerospace_mcp.fastmcp_server, "run")


class TestAppMainRun:
    """Tests for app.main run() function."""

    @pytest.mark.unit
    def test_run_default_config(self):
        """Test run() with default configuration."""
        mock_uvicorn_run = MagicMock()

        with patch("uvicorn.run", mock_uvicorn_run):
            with patch.dict(
                "os.environ",
                {},
                clear=True,
            ):
                from app.main import run

                run()

                mock_uvicorn_run.assert_called_once()
                call_kwargs = mock_uvicorn_run.call_args[1]
                assert call_kwargs["host"] == "0.0.0.0"
                assert call_kwargs["port"] == 8080
                assert call_kwargs["log_level"] == "info"
                assert call_kwargs["reload"] is False

    @pytest.mark.unit
    def test_run_custom_config(self):
        """Test run() with custom configuration."""
        mock_uvicorn_run = MagicMock()

        with patch("uvicorn.run", mock_uvicorn_run):
            with patch.dict(
                "os.environ",
                {
                    "AEROSPACE_MCP_HOST": "127.0.0.1",
                    "AEROSPACE_MCP_PORT": "9000",
                    "AEROSPACE_MCP_LOG_LEVEL": "DEBUG",
                    "AEROSPACE_MCP_ENV": "development",
                },
            ):
                from app.main import run

                run()

                mock_uvicorn_run.assert_called_once()
                call_kwargs = mock_uvicorn_run.call_args[1]
                assert call_kwargs["host"] == "127.0.0.1"
                assert call_kwargs["port"] == 9000
                assert call_kwargs["log_level"] == "debug"
                assert call_kwargs["reload"] is True

    @pytest.mark.unit
    def test_run_production_mode(self):
        """Test run() in production mode (no reload)."""
        mock_uvicorn_run = MagicMock()

        with patch("uvicorn.run", mock_uvicorn_run):
            with patch.dict(
                "os.environ",
                {
                    "AEROSPACE_MCP_ENV": "production",
                },
            ):
                from app.main import run

                run()

                mock_uvicorn_run.assert_called_once()
                call_kwargs = mock_uvicorn_run.call_args[1]
                assert call_kwargs["reload"] is False


class TestAppMainDotenvLoading:
    """Tests for dotenv loading in app.main."""

    @pytest.mark.unit
    def test_dotenv_import_failure(self):
        """Test graceful handling of dotenv import failure."""
        # This tests lines 13-14 in app/main.py

        # The module should load even if dotenv fails
        with patch.dict("sys.modules", {"dotenv": None}):
            import app.main

            # Verify the module loaded successfully
            assert hasattr(app.main, "app")
            assert hasattr(app.main, "run")


class TestAppMainImport:
    """Tests for app.main module import."""

    @pytest.mark.unit
    def test_module_exports(self):
        """Test that module exports expected items."""
        from app.main import __all__, app, run

        assert "app" in __all__
        assert "run" in __all__
        assert app is not None
        assert callable(run)
