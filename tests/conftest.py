"""Test configuration and fixtures for Aerospace MCP tests."""

import os

import pytest
from fastapi.testclient import TestClient

# Disable HTTP rate limiting for the test suite (all TestClient requests
# share one client IP and would otherwise trip the per-IP limiter).
# Hardening tests opt back in explicitly via monkeypatch.
os.environ.setdefault("RATE_LIMIT_RPM", "0")

from aerospace_mcp.http_app import app  # noqa: E402


@pytest.fixture
def client():
    """FastAPI test client."""
    return TestClient(app)
