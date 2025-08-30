# Aerospace MCP - Core flight planning functionality
from .core import (
    # Constants
    NM_PER_KM,
    KM_PER_NM,
    OPENAP_AVAILABLE,
    
    # Models
    AirportOut,
    PlanRequest,
    SegmentEst,
    PlanResponse,
    
    # Functions
    _airport_from_iata,
    _find_city_airports,
    _resolve_endpoint,
    great_circle_points,
    estimates_openap,
    health,
    airports_by_city,
    plan_flight,
    
    # Exceptions
    AirportResolutionError,
    OpenAPError,
)

__version__ = "0.1.0"
__all__ = [
    "NM_PER_KM",
    "KM_PER_NM", 
    "OPENAP_AVAILABLE",
    "AirportOut",
    "PlanRequest",
    "SegmentEst", 
    "PlanResponse",
    "_airport_from_iata",
    "_find_city_airports",
    "_resolve_endpoint",
    "great_circle_points",
    "estimates_openap",
    "search_airports_by_city",
    "get_health_status",
    "create_flight_plan",
    "AirportResolutionError",
    "OpenAPError",
    "FlightPlanError",
]