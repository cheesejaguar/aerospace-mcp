"""Transport-realistic tests for MCP server.

These tests verify that the MCP server works correctly over actual transports
(stdio subprocess, SSE/HTTP) rather than just in-process testing.
"""

import asyncio
import json
import os
import socket
import subprocess
import sys
import time

import pytest

from tests.mcp_stdio import PROTOCOL_VERSION, MCPStdioClient

# Mark all tests in this module as transport tests
pytestmark = [pytest.mark.transport, pytest.mark.slow]


@pytest.fixture
def stdio_server():
    """Start the MCP server on stdio and complete the initialize handshake."""
    client = MCPStdioClient()
    client.initialize()
    yield client
    client.close()


class TestStdioTransport:
    """Test MCP server over stdio transport."""

    def test_initialize_handshake(self):
        """The server answers initialize with its name and tool capability."""
        client = MCPStdioClient()
        try:
            response = client.initialize()
        finally:
            client.close()

        assert "error" not in response
        result = response["result"]
        assert result["protocolVersion"] == PROTOCOL_VERSION
        assert result["serverInfo"]["name"] == "aerospace-mcp"
        assert "tools" in result["capabilities"]

    def test_list_tools_over_stdio(self, stdio_server):
        """Every registered tool is listed, with an input schema."""
        from aerospace_mcp.tools.registry import ALL_TOOLS

        response = stdio_server.request("tools/list", {})

        tools = response["result"]["tools"]
        assert {t["name"] for t in tools} == set(ALL_TOOLS)
        by_name = {t["name"]: t for t in tools}
        schema = by_name["calculate_distance"]["inputSchema"]
        assert schema["required"] == ["lat1", "lon1", "lat2", "lon2"]

    def test_tool_call_over_stdio(self, stdio_server):
        """A tool call returns the tool's JSON as text content."""
        response = stdio_server.call_tool(
            "calculate_distance", {"lat1": 0, "lon1": 0, "lat2": 0, "lon2": 90}
        )

        result = response["result"]
        assert result["isError"] is False
        data = json.loads(result["content"][0]["text"])
        # Quarter of the WGS-84 equator
        assert data["distance_km"] == pytest.approx(10018.754, abs=0.01)

    def test_unknown_tool_is_an_error_result(self, stdio_server):
        response = stdio_server.call_tool("no_such_tool", {})
        assert response["result"]["isError"] is True
        assert "no_such_tool" in response["result"]["content"][0]["text"]


class TestSSETransport:
    """Test MCP server over SSE transport."""

    @pytest.fixture
    def sse_server(self):
        """Start MCP server with SSE transport on an ephemeral port."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]

        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "aerospace_mcp.fastmcp_server",
                "sse",
                "127.0.0.1",
                str(port),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # Wait until the port accepts connections
        deadline = time.monotonic() + 60
        while True:
            assert proc.poll() is None, "SSE server exited during startup"
            try:
                socket.create_connection(("127.0.0.1", port), timeout=1).close()
                break
            except OSError:
                assert time.monotonic() < deadline, "SSE server did not start"
                time.sleep(0.2)

        yield f"http://127.0.0.1:{port}"

        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

    def test_sse_list_and_call_tools(self, sse_server):
        """A real MCP SSE client can list tools and call one."""
        from mcp import ClientSession
        from mcp.client.sse import sse_client

        async def _run():
            async with sse_client(f"{sse_server}/sse") as (read, write):
                async with ClientSession(read, write) as session:
                    init = await session.initialize()
                    tools = await session.list_tools()
                    call = await session.call_tool(
                        "search_airports", {"query": "SJC", "query_type": "iata"}
                    )
                    return init, tools, call

        init, tools, call = asyncio.run(asyncio.wait_for(_run(), timeout=60))

        assert init.serverInfo.name == "aerospace-mcp"
        assert "plan_flight" in {t.name for t in tools.tools}
        assert call.isError is False
        assert "SJC (KSJC)" in call.content[0].text


class TestTransportEnvironmentVariables:
    """Test that environment variables are properly passed to server."""

    def test_openai_api_key_not_required(self):
        """The server initializes and serves tools without OPENAI_API_KEY."""
        env = os.environ.copy()
        env.pop("OPENAI_API_KEY", None)
        env.pop("LLM_TOOLS_ENABLED", None)

        client = MCPStdioClient(env=env)
        try:
            assert "result" in client.initialize()
            response = client.call_tool(
                "search_airports", {"query": "NRT", "query_type": "iata"}
            )
            assert "NRT (RJAA)" in response["result"]["content"][0]["text"]
            # The LLM agent tools report that they are disabled
            agent = client.call_tool(
                "select_aerospace_tool", {"user_task": "plan a flight"}
            )
            assert "disabled" in agent["result"]["content"][0]["text"]
        finally:
            client.close()


class TestTransportConcurrency:
    """Test server behavior under sequential and pipelined access."""

    def test_multiple_sequential_requests(self, stdio_server):
        """Repeated requests on one session all succeed."""
        for _ in range(5):
            response = stdio_server.request("tools/list", {})
            assert len(response["result"]["tools"]) > 40

    def test_rapid_tool_calls(self, stdio_server):
        """Pipelined tool calls are all answered, each with its own result."""
        latitudes = [0, 10, 20, 30, 40]
        for i, lat in enumerate(latitudes, start=100):
            stdio_server.send_raw(
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": i,
                        "method": "tools/call",
                        "params": {
                            "name": "calculate_distance",
                            "arguments": {
                                "lat1": 0,
                                "lon1": 0,
                                "lat2": lat,
                                "lon2": 0,
                            },
                        },
                    }
                )
                + "\n"
            )

        results = {}
        while len(results) < len(latitudes):
            line = stdio_server._lines.get(timeout=stdio_server.timeout)
            assert line is not None, "server closed stdout"
            message = json.loads(line)
            if message.get("id") in range(100, 100 + len(latitudes)):
                results[message["id"]] = json.loads(
                    message["result"]["content"][0]["text"]
                )

        distances = [results[i]["distance_km"] for i in sorted(results)]
        assert distances[0] == 0.0
        # Meridian distance grows with latitude (~1,106-1,110 km per 10 deg)
        assert all(b - a > 1100 for a, b in zip(distances, distances[1:], strict=False))


class TestTransportErrorRecovery:
    """Test server recovery from transport errors."""

    def test_server_handles_malformed_json(self, stdio_server):
        """A malformed line does not kill the session."""
        stdio_server.send_raw("not valid json\n")

        response = stdio_server.request("tools/list", {})
        assert len(response["result"]["tools"]) > 40
        assert stdio_server.process.poll() is None

    def test_server_handles_empty_input(self, stdio_server):
        """Blank lines are ignored."""
        stdio_server.send_raw("\n\n\n")

        response = stdio_server.call_tool(
            "search_airports", {"query": "LHR", "query_type": "iata"}
        )
        assert "LHR (EGLL)" in response["result"]["content"][0]["text"]
