"""FastAPI HTTP interface for the aerospace flight planning system.

This module is a thin interface layer: all business logic (airport
resolution, great-circle routing, OpenAP performance estimation) lives in
``aerospace_mcp.core`` and is shared with the MCP server.  Endpoints here
only translate HTTP requests/responses and map domain exceptions onto
status codes.

Environment configuration:
    CORS_ORIGINS: Comma-separated list of allowed origins.  CORS is
        disabled (browser same-origin policy applies) when unset.
    RATE_LIMIT_RPM: Per-client-IP request budget per minute (default 120,
        ``0`` disables rate limiting).
    MAX_BODY_BYTES: Maximum accepted request body size (default 1 MiB).

WARNING: For educational and research purposes only.  Do NOT use for real
flight planning, navigation, or aircraft operations.
"""

from __future__ import annotations

# Load environment variables from .env for local/dev runs
try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

import os
import time
from collections import deque
from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from aerospace_mcp import core
from aerospace_mcp.core import (
    AirportOut,
    AirportResolutionError,
    FlightPlanError,
    OpenAPError,
    PlanRequest,
    PlanResponse,
)

app = FastAPI(title="Flight Planner API", version="0.1.0")

# ----------------------------
# CORS (explicit opt-in via CORS_ORIGINS env)
# ----------------------------
_cors_origins = [
    o.strip() for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()
]
if _cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

# ----------------------------
# Rate limiting + body-size limits (in-process, no external deps)
# ----------------------------
DEFAULT_RATE_LIMIT_RPM = 120
DEFAULT_MAX_BODY_BYTES = 1024 * 1024

# Per-client-IP sliding window of request timestamps (seconds).
_request_log: dict[str, deque[float]] = {}


def _int_env(name: str, default: int) -> int:
    """Read an integer env var, falling back to ``default`` on bad values.

    Read per-request (cheap) so tests and operators can adjust limits
    without re-importing the module.
    """
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


@app.middleware("http")
async def limits_middleware(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Reject oversized bodies (413) and rate-limit per client IP (429).

    Responses are returned directly (not raised as HTTPException) because
    exceptions raised in middleware bypass FastAPI's exception handlers.
    """
    max_body = _int_env("MAX_BODY_BYTES", DEFAULT_MAX_BODY_BYTES)
    rate_limit = _int_env("RATE_LIMIT_RPM", DEFAULT_RATE_LIMIT_RPM)

    content_length = request.headers.get("content-length")
    if content_length and content_length.isdigit():
        if int(content_length) > max_body:
            return JSONResponse(
                status_code=413, content={"detail": "Request body too large."}
            )

    if rate_limit > 0:
        client_ip = request.client.host if request.client else "unknown"
        now = time.monotonic()
        window = _request_log.setdefault(client_ip, deque())
        while window and now - window[0] > 60.0:
            window.popleft()
        if len(window) >= rate_limit:
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded; retry later."},
            )
        window.append(now)

    return await call_next(request)


# ----------------------------
# Endpoints
# ----------------------------
@app.get("/health")
def health() -> dict[str, Any]:
    return core.health()


@app.get("/airports/by_city", response_model=list[AirportOut])
def airports_by_city(
    city: str = Query(..., min_length=1, max_length=100),
    country: str | None = Query(None, max_length=2),
) -> list[AirportOut]:
    return core.airports_by_city(city, country)


@app.post("/plan", response_model=PlanResponse)
def plan(req: PlanRequest) -> PlanResponse:
    try:
        return core.create_flight_plan(req)
    except AirportResolutionError as e:
        status = 400 if e.kind == "iata_not_found" else 404
        raise HTTPException(status_code=status, detail=str(e)) from e
    except OpenAPError as e:
        raise HTTPException(status_code=501, detail=str(e)) from e
    except FlightPlanError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
