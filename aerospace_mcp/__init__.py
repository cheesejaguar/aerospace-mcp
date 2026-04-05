"""Aerospace MCP — A dual-mode aerospace flight planning system.

This package provides identical flight planning functionality through both
HTTP API (FastAPI) and Model Context Protocol (MCP) interfaces. It includes
44 tools spanning 11 aerospace domains: core flight planning, atmosphere
modeling, coordinate frames, aerodynamics, propellers, rockets, orbital
mechanics, guidance/navigation/control, performance analysis, trajectory
optimization, and AI agent helpers.

Public API:
    - Models: AirportOut, PlanRequest, PlanResponse, SegmentEst
    - Functions: plan_flight, airports_by_city, health, great_circle_points,
      estimates_openap, search_airports_by_city, get_health_status,
      create_flight_plan
    - Exceptions: AirportResolutionError, OpenAPError, FlightPlanError
    - Constants: NM_PER_KM, KM_PER_NM, OPENAP_AVAILABLE

WARNING:
    This package is for educational and research purposes only.
    Do NOT use for real flight planning, navigation, or aircraft operations.
"""

# --- Constants ---
# --- Data Models ---
# --- Exceptions ---
# --- Core Functions ---
from .core import (
    KM_PER_NM,
    NM_PER_KM,
    OPENAP_AVAILABLE,
    AirportOut,
    AirportResolutionError,
    OpenAPError,
    PlanRequest,
    PlanResponse,
    SegmentEst,
    _airport_from_iata,
    _find_city_airports,
    _resolve_endpoint,
    airports_by_city,
    estimates_openap,
    great_circle_points,
    health,
    plan_flight,
)

__version__ = "0.1.0"

# Symbols exported when using ``from aerospace_mcp import *``.
__all__ = [
    # Constants
    "NM_PER_KM",
    "KM_PER_NM",
    "OPENAP_AVAILABLE",
    # Data models
    "AirportOut",
    "PlanRequest",
    "SegmentEst",
    "PlanResponse",
    # Internal helpers (prefixed with _ but still public for advanced use)
    "_airport_from_iata",
    "_find_city_airports",
    "_resolve_endpoint",
    # Core functions
    "great_circle_points",
    "estimates_openap",
    "search_airports_by_city",
    "get_health_status",
    "create_flight_plan",
    # Exceptions
    "AirportResolutionError",
    "OpenAPError",
    "FlightPlanError",
]
