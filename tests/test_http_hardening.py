"""Tests for HTTP hardening: CORS, rate limiting, and body-size limits."""

import importlib

import pytest
from fastapi.testclient import TestClient

import main as main_module


@pytest.fixture(autouse=True)
def clean_rate_log():
    """Isolate the rate limiter's per-IP state between tests."""
    main_module._request_log.clear()
    yield
    main_module._request_log.clear()


class TestRateLimiting:
    """Per-client-IP sliding-window rate limiting."""

    @pytest.mark.unit
    def test_rate_limit_returns_429(self, client, monkeypatch):
        """Requests beyond RATE_LIMIT_RPM within a minute get 429."""
        monkeypatch.setenv("RATE_LIMIT_RPM", "5")

        for _ in range(5):
            assert client.get("/health").status_code == 200

        response = client.get("/health")
        assert response.status_code == 429
        assert "Rate limit" in response.json()["detail"]

    @pytest.mark.unit
    def test_rate_limit_disabled_with_zero(self, client, monkeypatch):
        """RATE_LIMIT_RPM=0 disables limiting entirely."""
        monkeypatch.setenv("RATE_LIMIT_RPM", "0")

        for _ in range(20):
            assert client.get("/health").status_code == 200

    @pytest.mark.unit
    def test_rate_limit_invalid_env_falls_back(self, client, monkeypatch):
        """A malformed env value falls back to the default (no crash)."""
        monkeypatch.setenv("RATE_LIMIT_RPM", "not-a-number")

        assert client.get("/health").status_code == 200


class TestBodySizeLimit:
    """Oversized request bodies are rejected before processing."""

    @pytest.mark.unit
    def test_oversized_body_returns_413(self, client, monkeypatch):
        monkeypatch.setenv("MAX_BODY_BYTES", "100")

        payload = {
            "depart_city": "San Jose",
            "arrive_city": "Tokyo",
            "ac_type": "A320",
            "padding": "x" * 500,
        }
        response = client.post("/plan", json=payload)
        assert response.status_code == 413
        assert "too large" in response.json()["detail"]

    @pytest.mark.unit
    def test_normal_body_accepted(self, client, mock_airports_iata, monkeypatch):
        """Bodies under the limit pass through to normal validation."""
        monkeypatch.setenv("MAX_BODY_BYTES", str(1024 * 1024))

        response = client.post("/plan", json={"depart_city": "San Jose"})
        # 422 (validation error for missing fields), not 413
        assert response.status_code == 422


class TestCORS:
    """CORS is opt-in via the CORS_ORIGINS environment variable."""

    @pytest.mark.unit
    def test_no_cors_headers_by_default(self, client):
        """Without CORS_ORIGINS, no CORS headers are emitted."""
        response = client.get("/health", headers={"Origin": "https://evil.example"})
        assert "access-control-allow-origin" not in response.headers

    @pytest.mark.unit
    def test_cors_enabled_via_env(self, monkeypatch):
        """Setting CORS_ORIGINS enables the middleware for listed origins."""
        monkeypatch.setenv("CORS_ORIGINS", "https://allowed.example")
        monkeypatch.setenv("RATE_LIMIT_RPM", "0")

        reloaded = importlib.reload(main_module)
        try:
            local_client = TestClient(reloaded.app)
            response = local_client.get(
                "/health", headers={"Origin": "https://allowed.example"}
            )
            assert (
                response.headers.get("access-control-allow-origin")
                == "https://allowed.example"
            )

            response = local_client.get(
                "/health", headers={"Origin": "https://other.example"}
            )
            assert "access-control-allow-origin" not in response.headers
        finally:
            # Restore the module to its env-default (no CORS) state so other
            # tests see the original behavior.
            monkeypatch.delenv("CORS_ORIGINS")
            importlib.reload(main_module)


class TestQueryBounds:
    """Query parameter validation limits."""

    @pytest.mark.unit
    def test_city_too_long_rejected(self, client):
        response = client.get(f"/airports/by_city?city={'x' * 101}")
        assert response.status_code == 422

    @pytest.mark.unit
    def test_country_too_long_rejected(self, client):
        response = client.get("/airports/by_city?city=Tokyo&country=JPN")
        assert response.status_code == 422

    @pytest.mark.unit
    def test_route_step_km_upper_bound(self, client, mock_airports_iata):
        payload = {
            "depart_city": "San Jose",
            "arrive_city": "Tokyo",
            "ac_type": "A320",
            "route_step_km": 5000.0,
        }
        response = client.post("/plan", json=payload)
        assert response.status_code == 422
