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
from starlette.types import ASGIApp, Message, Receive, Scope, Send

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


_BODY_TOO_LARGE = {"detail": "Request body too large."}


class BodySizeLimitMiddleware:
    """Enforce MAX_BODY_BYTES on the bytes actually received (ASGI level).

    The ``Content-Length`` check in ``limits_middleware`` is only a fast
    path: chunked uploads carry no length header.  This middleware reads
    the request body as it streams in, counting bytes, and answers 413 as
    soon as the running total exceeds the limit -- before the application
    sees any of it.  The buffered messages (at most MAX_BODY_BYTES) are
    then replayed to the application unchanged.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        max_body = _int_env("MAX_BODY_BYTES", DEFAULT_MAX_BODY_BYTES)
        buffered: list[Message] = []
        received = 0
        while True:
            message = await receive()
            buffered.append(message)
            if message["type"] != "http.request":
                break  # e.g. http.disconnect: hand it to the app as-is
            received += len(message.get("body", b""))
            if received > max_body:
                response = JSONResponse(status_code=413, content=_BODY_TOO_LARGE)
                await response(scope, receive, send)
                return
            if not message.get("more_body", False):
                break

        async def replay() -> Message:
            if buffered:
                return buffered.pop(0)
            return await receive()

        await self.app(scope, replay, send)


# Registered before the decorator-based middleware below, so it runs inside
# it: rate limiting and the Content-Length fast path still reject a request
# before its body is read.
app.add_middleware(BodySizeLimitMiddleware)


@app.middleware("http")
async def limits_middleware(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Reject oversized bodies (413) and rate-limit per client IP (429).

    Only the declared ``Content-Length`` is checked here; bodies without
    one (chunked uploads) are metered by ``BodySizeLimitMiddleware``.
    Responses are returned directly (not raised as HTTPException) because
    exceptions raised in middleware bypass FastAPI's exception handlers.
    """
    max_body = _int_env("MAX_BODY_BYTES", DEFAULT_MAX_BODY_BYTES)
    rate_limit = _int_env("RATE_LIMIT_RPM", DEFAULT_RATE_LIMIT_RPM)

    content_length = request.headers.get("content-length")
    if content_length and content_length.isdigit():
        if int(content_length) > max_body:
            return JSONResponse(status_code=413, content=_BODY_TOO_LARGE)

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
        # 501 only when the backend itself is missing; an aircraft type
        # OpenAP cannot model is a client input error, like an unknown IATA.
        status = {"unavailable": 501, "unsupported_aircraft": 400}.get(e.kind, 500)
        raise HTTPException(status_code=status, detail=str(e)) from e
    except FlightPlanError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
