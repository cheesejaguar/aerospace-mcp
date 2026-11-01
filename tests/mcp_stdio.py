"""Minimal MCP stdio client for subprocess-level protocol tests.

The MCP stdio transport exchanges newline-delimited JSON-RPC 2.0 messages
(no Content-Length framing).  Responses are read on a background thread so
a hung server fails the test with a timeout instead of blocking the run.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
from typing import Any

SERVER_CMD = [sys.executable, "-m", "aerospace_mcp.fastmcp_server"]
PROTOCOL_VERSION = "2024-11-05"


class MCPStdioClient:
    """Drive ``aerospace_mcp.fastmcp_server`` over its stdio transport."""

    def __init__(self, env: dict[str, str] | None = None, timeout: float = 60.0):
        self.timeout = timeout
        self.process = subprocess.Popen(
            SERVER_CMD,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
            env=env if env is not None else os.environ.copy(),
        )
        self._lines: queue.Queue[str | None] = queue.Queue()
        self._next_id = 0
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self) -> None:
        assert self.process.stdout is not None
        for line in self.process.stdout:
            self._lines.put(line)
        self._lines.put(None)  # EOF

    def send_raw(self, text: str) -> None:
        assert self.process.stdin is not None
        self.process.stdin.write(text)
        self.process.stdin.flush()

    def notify(self, method: str, params: dict | None = None) -> None:
        message: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            message["params"] = params
        self.send_raw(json.dumps(message) + "\n")

    def request(self, method: str, params: dict | None = None) -> dict:
        """Send a request and return the response with the matching id."""
        self._next_id += 1
        request_id = self._next_id
        message: dict[str, Any] = {"jsonrpc": "2.0", "id": request_id, "method": method}
        if params is not None:
            message["params"] = params
        self.send_raw(json.dumps(message) + "\n")
        while True:
            line = self._lines.get(timeout=self.timeout)
            if line is None:
                raise ConnectionError("server closed stdout")
            response = json.loads(line)
            if response.get("id") == request_id:
                return response

    def initialize(self) -> dict:
        response = self.request(
            "initialize",
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "aerospace-mcp-tests", "version": "1.0"},
            },
        )
        self.notify("notifications/initialized")
        return response

    def call_tool(self, name: str, arguments: dict) -> dict:
        return self.request("tools/call", {"name": name, "arguments": arguments})

    def close(self) -> int:
        """Close stdin (client disconnect) and return the exit code."""
        try:
            assert self.process.stdin is not None
            self.process.stdin.close()
        except OSError:
            pass
        try:
            return self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
            raise
