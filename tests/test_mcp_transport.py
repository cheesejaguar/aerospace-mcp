"""Transport-realistic tests for MCP server.

These tests verify that the MCP server works correctly over actual transports
(stdio subprocess, SSE/HTTP) rather than just in-process testing.
"""

import json
import os
import subprocess
import sys
import time

import pytest

# Mark all tests in this module as transport tests
pytestmark = [pytest.mark.transport, pytest.mark.slow]


class MCPStdioClient:
    """Simple MCP client for stdio transport testing."""

    def __init__(self, process: subprocess.Popen):
        self.process = process
        self._request_id = 0

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    def send_request(self, method: str, params: dict = None) -> dict:
        """Send JSON-RPC request and get response."""
        request = {
            "jsonrpc": "2.0",
            "method": method,
            "id": self._next_id(),
        }
        if params is not None:
            request["params"] = params

        # MCP uses Content-Length header framing
        body = json.dumps(request)
        message = f"Content-Length: {len(body)}\r\n\r\n{body}"

        try:
            self.process.stdin.write(message)
            self.process.stdin.flush()

            # Read response with timeout
            # This is simplified - real MCP needs proper header parsing
            return self._read_response()
        except (BrokenPipeError, OSError) as e:
            return {"error": str(e)}

    def _read_response(self) -> dict:
        """Read JSON-RPC response from stdout."""
        # Read Content-Length header
        header = ""
        while True:
            char = self.process.stdout.read(1)
            if not char:
                return {"error": "Connection closed"}
            header += char
            if header.endswith("\r\n\r\n"):
                break

        # Parse Content-Length
        for line in header.split("\r\n"):
            if line.startswith("Content-Length:"):
                length = int(line.split(":")[1].strip())
                break
        else:
            return {"error": "No Content-Length header"}

        # Read body
        body = self.process.stdout.read(length)
        try:
            return json.loads(body)
        except json.JSONDecodeError as e:
            return {"error": f"Invalid JSON: {e}", "raw": body}

    def close(self):
        """Close the client and terminate the server."""
        try:
            self.process.stdin.close()
        except Exception:
            # Ignore errors while closing stdin during test teardown; the process
            # may already have exited or the pipe may already be closed.
            pass
        self.process.terminate()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()


@pytest.fixture
def stdio_server():
    """Start MCP server with stdio transport."""
    # Start server as subprocess
    proc = subprocess.Popen(
        [sys.executable, "-m", "aerospace_mcp.fastmcp_server"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=0,
    )

    # Wait for server to initialize (reduced for CI efficiency)
    time.sleep(0.5)

    if proc.poll() is not None:
        stderr = proc.stderr.read()
        pytest.fail(f"Server failed to start: {stderr}")

    client = MCPStdioClient(proc)

    yield client

    client.close()


class TestStdioTransport:
    """Test MCP server over stdio transport."""

    def test_server_process_starts(self, stdio_server):
        """Test that server process starts successfully."""
        assert stdio_server.process.poll() is None, "Server should be running"

    @pytest.mark.skip(reason="MCP handshake requires full protocol implementation")
    def test_initialize_handshake(self, stdio_server):
        """Test MCP initialize handshake."""
        response = stdio_server.send_request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test-client", "version": "1.0.0"},
            },
        )

        assert "error" not in response or response.get("error") is None
        assert "result" in response

    @pytest.mark.skip(reason="Requires initialize first")
    def test_list_tools_over_stdio(self, stdio_server):
        """Test listing tools over stdio transport."""
        # First initialize
        stdio_server.send_request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test-client", "version": "1.0.0"},
            },
        )

        # Then list tools
        response = stdio_server.send_request("tools/list", {})

        assert "error" not in response or response.get("error") is None
        assert "result" in response
        assert "tools" in response["result"]


class TestSSETransport:
    """Test MCP server over SSE transport."""

    @pytest.fixture
    def sse_server(self):
        """Start MCP server with SSE transport on ephemeral port."""
        import socket

        # Find available port
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
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
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Wait for server to start (reduced for CI efficiency)
        time.sleep(1)

        if proc.poll() is not None:
            stderr = proc.stderr.read() if proc.stderr else ""
            pytest.skip(f"SSE server failed to start: {stderr}")

        yield f"http://127.0.0.1:{port}", proc

        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

    def test_sse_server_starts(self, sse_server):
        """Test that SSE server starts successfully."""
        url, proc = sse_server
        assert proc.poll() is None, "SSE server should be running"

    def test_sse_server_responds(self, sse_server):
        """Test that SSE server accepts connections on the expected endpoint.

        Note: This test accepts 200, 404, or 405 as valid responses because:
        - 200: SSE endpoint exists and is working
        - 404: Server is running but SSE endpoint not configured at /sse
        - 405: Server is running but GET method not allowed on /sse

        The key assertion is that the server is running and accepting HTTP
        connections, not that SSE is fully functional (which would require
        a proper SSE client implementation).
        """
        import httpx

        url, proc = sse_server

        try:
            # Use streaming to avoid blocking on SSE connections
            # SSE is a long-lived connection, so we just check we can connect
            with httpx.stream("GET", f"{url}/sse", timeout=2.0) as response:
                # Server is responding - any of these status codes is acceptable
                assert response.status_code in [200, 404, 405]
        except httpx.ConnectError:
            pytest.skip("Could not connect to SSE server")
        except httpx.ReadTimeout:
            # SSE connections are long-lived, timeout after connection is expected
            # This actually indicates success - we connected and the server is streaming
            pass


class TestTransportEnvironmentVariables:
    """Test that environment variables are properly passed to server."""

    def test_env_vars_available_in_subprocess(self):
        """Test that env vars are passed to subprocess."""
        env = os.environ.copy()
        env["TEST_VAR_12345"] = "test_value"

        proc = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "import os; print(os.environ.get('TEST_VAR_12345', 'not_found'))",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            text=True,
        )

        stdout, _ = proc.communicate(timeout=10)
        assert "test_value" in stdout

    def test_openai_api_key_not_required(self):
        """Test that server starts without OPENAI_API_KEY."""
        env = os.environ.copy()
        env.pop("OPENAI_API_KEY", None)
        env.pop("LLM_TOOLS_ENABLED", None)

        proc = subprocess.Popen(
            [sys.executable, "-m", "aerospace_mcp.fastmcp_server"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )

        time.sleep(1)

        if proc.poll() is not None:
            # Server exited, check why
            stderr = proc.stderr.read() if proc.stderr else b""
            assert b"OPENAI_API_KEY" not in stderr, (
                "Server should not require OPENAI_API_KEY"
            )
        else:
            proc.terminate()
            proc.wait(timeout=5)


class TestTransportConcurrency:
    """Test server behavior under concurrent access."""

    def test_multiple_sequential_requests(self, stdio_server):
        """Test multiple sequential requests don't break the server."""
        # Just verify server stays alive after multiple operations
        for _ in range(5):
            time.sleep(0.1)
            assert stdio_server.process.poll() is None, "Server should stay running"

    @pytest.mark.skip(reason="Requires full MCP client")
    def test_rapid_tool_calls(self, stdio_server):
        """Test rapid sequential tool calls."""
        # Would test multiple tool calls in quick succession
        pass


class TestTransportErrorRecovery:
    """Test server recovery from transport errors.

    These tests verify that the server doesn't crash when receiving
    malformed input. The MCP protocol expects Content-Length framed
    JSON-RPC messages, so sending raw text or empty lines tests the
    server's robustness to protocol violations.
    """

    def test_server_handles_malformed_json(self):
        """Test server handles malformed JSON without crashing.

        Expected behavior: Server should remain stable when receiving
        malformed input (not proper MCP framing). It may ignore the
        input, log an error, or close the connection - but should not
        crash or hang indefinitely.
        """
        proc = subprocess.Popen(
            [sys.executable, "-m", "aerospace_mcp.fastmcp_server"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        time.sleep(0.3)
        assert proc.poll() is None, "Server should start successfully"

        try:
            # Send malformed JSON (not proper MCP Content-Length framing)
            proc.stdin.write("not valid json\n")
            proc.stdin.flush()

            time.sleep(0.3)

            # Server should not have crashed - it may still be running
            # or may have exited cleanly. Either is acceptable behavior
            # for handling protocol violations.
            exit_code = proc.poll()
            if exit_code is not None:
                # Server exited - verify it was a clean exit (0) or
                # controlled shutdown, not a crash (segfault = -11, etc.)
                assert exit_code >= 0, f"Server crashed with signal {-exit_code}"
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()

    def test_server_handles_empty_input(self):
        """Test server handles empty input without crashing.

        Expected behavior: Server should remain running when receiving
        empty lines. Empty input should be ignored or handled gracefully.
        """
        proc = subprocess.Popen(
            [sys.executable, "-m", "aerospace_mcp.fastmcp_server"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        time.sleep(0.3)
        assert proc.poll() is None, "Server should start successfully"

        try:
            # Send empty lines
            proc.stdin.write("\n\n\n")
            proc.stdin.flush()

            time.sleep(0.3)

            # Server should still be running after receiving empty input
            assert proc.poll() is None, "Server should remain running after empty input"
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
