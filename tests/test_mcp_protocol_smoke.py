"""Protocol smoke tests for MCP server.

These tests spawn the MCP server as a subprocess and test it like a real client would,
verifying server startup, lifecycle, and basic process management.

This catches framing/stdio quirks, env var wiring, and lifecycle issues that
in-process tests might miss.
"""

import asyncio
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tests.mcp_stdio import MCPStdioClient

# Mark all tests in this module as protocol tests
pytestmark = [pytest.mark.protocol, pytest.mark.slow]

CORE_TOOLS = {
    "search_airports",
    "plan_flight",
    "calculate_distance",
    "get_aircraft_performance",
    "get_system_status",
}


class TestMCPProtocolSmoke:
    """Black-box protocol smoke tests using subprocess."""

    def test_list_tools_returns_expected_tools(self):
        """tools/list over stdio returns the core aerospace tools."""
        client = MCPStdioClient()
        try:
            client.initialize()
            response = client.request("tools/list", {})
        finally:
            assert client.close() == 0

        names = {tool["name"] for tool in response["result"]["tools"]}
        assert CORE_TOOLS <= names
        assert {"search_aerospace_tools", "list_tool_categories"} <= names


class TestMCPServerStartup:
    """Test server startup and initialization."""

    def test_server_module_is_importable(self):
        """Test that the server module can be imported."""
        from aerospace_mcp import fastmcp_server

        assert callable(fastmcp_server.run)
        assert fastmcp_server.mcp.name == "aerospace-mcp"

    def test_console_script_entry_point_serves_stdio(self):
        """The installed ``aerospace-mcp`` script answers an MCP handshake
        and exits cleanly when the client disconnects."""
        script = shutil.which("aerospace-mcp", path=str(Path(sys.executable).parent))
        assert script, "aerospace-mcp console script is not installed"

        proc = subprocess.run(
            [script],
            input=(
                '{"jsonrpc":"2.0","id":1,"method":"initialize","params":'
                '{"protocolVersion":"2024-11-05","capabilities":{},'
                '"clientInfo":{"name":"smoke","version":"1"}}}\n'
            ),
            capture_output=True,
            text=True,
            timeout=60,
        )

        assert proc.returncode == 0, proc.stderr[-2000:]
        assert '"serverInfo":{"name":"aerospace-mcp"' in proc.stdout


class TestMCPClientIntegration:
    """Integration tests using the FastMCP client library."""

    def test_client_can_connect_and_list_tools(self):
        """A FastMCP client spawns the server on stdio, lists and calls tools."""
        from fastmcp import Client
        from fastmcp.client.transports import StdioTransport

        transport = StdioTransport(
            command=sys.executable, args=["-m", "aerospace_mcp.fastmcp_server"]
        )

        async def _run():
            async with Client(transport) as client:
                tools = await client.list_tools()
                result = await client.call_tool(
                    "search_airports", {"query": "JFK", "query_type": "iata"}
                )
                return tools, result

        tools, result = asyncio.run(asyncio.wait_for(_run(), timeout=120))

        assert CORE_TOOLS <= {tool.name for tool in tools}
        assert "JFK (KJFK)" in result.content[0].text


class TestHTTPAPISmoke:
    """Smoke tests for the HTTP API server."""

    @pytest.fixture
    def http_server(self):
        """Start HTTP server on an ephemeral port and wait until it is up."""
        import httpx

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]

        proc = subprocess.Popen(
            [
                sys.executable,
                "-c",
                f"""
import uvicorn
from aerospace_mcp.http_app import app
uvicorn.run(app, host="127.0.0.1", port={port}, log_level="error")
""",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        url = f"http://127.0.0.1:{port}"

        deadline = time.monotonic() + 60
        while True:
            assert proc.poll() is None, "HTTP server exited during startup"
            try:
                httpx.get(f"{url}/health", timeout=1)
                break
            except httpx.TransportError:
                assert time.monotonic() < deadline, "HTTP server did not start"
                time.sleep(0.2)

        yield url

        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

    def test_health_endpoint(self, http_server):
        """Test that health endpoint responds."""
        import httpx

        response = httpx.get(f"{http_server}/health", timeout=5)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["airports_count"] > 1000

    def test_airports_endpoint(self, http_server):
        """Test that airports endpoint responds."""
        import httpx

        response = httpx.get(f"{http_server}/airports/by_city?city=Tokyo", timeout=5)
        assert response.status_code == 200
        assert [a["iata"] for a in response.json()] == ["HND", "NRT"]


class TestServerLifecycle:
    """Test server lifecycle and graceful shutdown."""

    def test_server_handles_sigterm(self):
        """Test that server handles SIGTERM gracefully."""
        client = MCPStdioClient()
        client.initialize()  # fully started
        proc = client.process

        proc.send_signal(signal.SIGTERM)

        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            pytest.fail("Server did not shut down gracefully")

    def test_server_handles_stdin_close(self):
        """Closing stdin (client disconnect) ends the server with exit code 0."""
        client = MCPStdioClient()
        client.initialize()

        assert client.close() == 0
