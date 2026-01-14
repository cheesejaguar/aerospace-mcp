"""Core flight planning tools for the Aerospace MCP server."""

import json
import logging
from typing import Literal

from ..core import (
    OPENAP_AVAILABLE,
    AirportResolutionError,
    OpenAPError,
    PlanRequest,
    _airport_from_iata,
    _find_city_airports,
    _resolve_endpoint,
    estimates_openap,
    great_circle_points,
)

logger = logging.getLogger(__name__)


def search_airports(
    query: str,
    country: str | None = None,
    query_type: Literal["iata", "city", "auto"] = "auto",
) -> str:
    """Search for airports by IATA code or city name.

    Args:
        query: IATA code (e.g., 'SJC') or city name (e.g., 'San Jose')
        country: Optional ISO country code to filter by (e.g., 'US', 'JP')
        query_type: Type of query - 'iata' for IATA codes, 'city' for city names, 'auto' to detect

    Returns:
        Formatted string with airport information
    """
    query = query.strip()
    if not query:
        return "Error: Query parameter is required"

    results = []

    # Auto-detect query type if needed
    if query_type == "auto":
        query_type = "iata" if len(query) == 3 and query.isalpha() else "city"

    try:
        if query_type == "iata":
            # Search by IATA code
            airport = _airport_from_iata(query)
            if airport:
                results = [airport]
        else:
            # Search by city name
            results = _find_city_airports(query, country)

        if not results:
            message = f"No airports found for {query_type} '{query}'"
            if country:
                message += f" in country '{country}'"
            return message

        # Format results
        response_lines = [f"Found {len(results)} airport(s):"]
        for airport in results:
            line = f"• {airport.iata} ({airport.icao}) - {airport.name}"
            line += f"\n  City: {airport.city}, {airport.country}"
            line += f"\n  Coordinates: {airport.lat:.4f}, {airport.lon:.4f}"
            if airport.tz:
                line += f"\n  Timezone: {airport.tz}"
            response_lines.append(line)

        return "\n\n".join(response_lines)

    except Exception as e:
        return f"Search error: {str(e)}"


def plan_flight(
    departure: dict,
    arrival: dict,
    aircraft: dict | None = None,
    route_options: dict | None = None,
) -> str:
    """Plan a flight route between two airports with performance estimates.

    Args:
        departure: Dict with departure info (city, country, iata)
        arrival: Dict with arrival info (city, country, iata)
        aircraft: Optional aircraft config (ac_type, cruise_alt_ft, route_step_km)
        route_options: Optional route options

    Returns:
        JSON string with flight plan details
    """
    try:
        # Build request object
        request_data = {
            "depart_city": departure["city"],
            "arrive_city": arrival["city"],
        }

        # Add optional fields
        if departure.get("country"):
            request_data["depart_country"] = departure["country"]
        if arrival.get("country"):
            request_data["arrive_country"] = arrival["country"]
        if departure.get("iata"):
            request_data["prefer_depart_iata"] = departure["iata"]
        if arrival.get("iata"):
            request_data["prefer_arrive_iata"] = arrival["iata"]

        # Add aircraft options (ac_type is required by PlanRequest)
        if aircraft and aircraft.get("ac_type"):
            request_data["ac_type"] = aircraft["ac_type"]
            if aircraft.get("cruise_alt_ft"):
                request_data["cruise_alt_ft"] = aircraft["cruise_alt_ft"]
            if aircraft.get("route_step_km"):
                request_data["route_step_km"] = aircraft["route_step_km"]
        else:
            # Use a default aircraft type if none provided
            request_data["ac_type"] = "A320"  # Default aircraft type

        # Create and validate request
        try:
            request = PlanRequest(**request_data)
        except Exception as e:
            return f"Invalid request: {str(e)}"

        # Resolve airports
        try:
            depart_airport = _resolve_endpoint(
                request.depart_city,
                request.depart_country,
                request.prefer_depart_iata,
                role="departure",
            )
            arrive_airport = _resolve_endpoint(
                request.arrive_city,
                request.arrive_country,
                request.prefer_arrive_iata,
                role="arrival",
            )
        except AirportResolutionError as e:
            return f"Airport resolution error: {str(e)}"

        # Calculate route - great_circle_points returns (list[tuple], distance_km)
        from geographiclib.geodesic import Geodesic

        points_list, distance_km = great_circle_points(
            depart_airport.lat,
            depart_airport.lon,
            arrive_airport.lat,
            arrive_airport.lon,
            step_km=request.route_step_km,
        )

        # Calculate bearings using geodesic
        g = Geodesic.WGS84.Inverse(
            depart_airport.lat,
            depart_airport.lon,
            arrive_airport.lat,
            arrive_airport.lon,
        )
        initial_bearing = g["azi1"]
        final_bearing = g["azi2"]

        # Convert to nautical miles
        NM_PER_KM = 0.539956803
        distance_nm = distance_km * NM_PER_KM

        # Build response
        response = {
            "departure": {
                "airport": {
                    "iata": depart_airport.iata,
                    "icao": depart_airport.icao,
                    "name": depart_airport.name,
                    "city": depart_airport.city,
                    "country": depart_airport.country,
                    "coordinates": {
                        "lat": depart_airport.lat,
                        "lon": depart_airport.lon,
                    },
                }
            },
            "arrival": {
                "airport": {
                    "iata": arrive_airport.iata,
                    "icao": arrive_airport.icao,
                    "name": arrive_airport.name,
                    "city": arrive_airport.city,
                    "country": arrive_airport.country,
                    "coordinates": {
                        "lat": arrive_airport.lat,
                        "lon": arrive_airport.lon,
                    },
                }
            },
            "route": {
                "distance_km": distance_km,
                "distance_nm": distance_nm,
                "initial_bearing_deg": initial_bearing,
                "final_bearing_deg": final_bearing,
                "points": [{"lat": p[0], "lon": p[1]} for p in points_list],
            },
        }

        # Add performance estimates if available
        if request.ac_type and OPENAP_AVAILABLE:
            try:
                performance, engine_name = estimates_openap(
                    request.ac_type,
                    request.cruise_alt_ft,
                    request.mass_kg,
                    distance_km,
                )
                response["performance"] = performance
                response["engine"] = engine_name
            except OpenAPError as e:
                response["performance_note"] = (
                    f"Performance estimation failed: {str(e)}"
                )
        elif request.ac_type:
            response["performance_note"] = (
                f"OpenAP not available - no performance estimates for {request.ac_type}"
            )

        return json.dumps(response, indent=2)

    except Exception as e:
        logger.error(f"Flight planning error: {str(e)}", exc_info=True)
        return f"Flight planning error: {str(e)}"


def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> str:
    """Calculate great circle distance between two points.

    Args:
        lat1: Latitude of first point in degrees
        lon1: Longitude of first point in degrees
        lat2: Latitude of second point in degrees
        lon2: Longitude of second point in degrees

    Returns:
        JSON string with distance information
    """
    try:
        from geographiclib.geodesic import Geodesic

        # Calculate great circle route - returns (list[tuple], distance_km)
        _, distance_km = great_circle_points(
            lat1, lon1, lat2, lon2, step_km=1000000
        )  # Single segment

        # Calculate bearings
        g = Geodesic.WGS84.Inverse(lat1, lon1, lat2, lon2)
        initial_bearing = g["azi1"]
        final_bearing = g["azi2"]

        # Convert to nautical miles
        NM_PER_KM = 0.539956803
        distance_nm = distance_km * NM_PER_KM

        return json.dumps(
            {
                "distance_km": distance_km,
                "distance_nm": distance_nm,
                "initial_bearing_deg": initial_bearing,
                "final_bearing_deg": final_bearing,
                "coordinates": {
                    "start": {"lat": lat1, "lon": lon1},
                    "end": {"lat": lat2, "lon": lon2},
                },
            },
            indent=2,
        )

    except Exception as e:
        return f"Distance calculation error: {str(e)}"


def get_aircraft_performance(
    aircraft_type: str, distance_km: float, cruise_altitude_ft: float = 35000
) -> str:
    """Get performance estimates for an aircraft type (requires OpenAP).

    Args:
        aircraft_type: ICAO aircraft type code (e.g., 'A320', 'B737')
        distance_km: Flight distance in kilometers
        cruise_altitude_ft: Cruise altitude in feet

    Returns:
        JSON string with performance estimates or error message
    """
    if not OPENAP_AVAILABLE:
        return "OpenAP library is not available. Install with: pip install openap"

    try:
        # estimates_openap signature: (ac_type, cruise_alt_ft, mass_kg, route_dist_km)
        # returns: (dict, engine_name_str)
        performance, engine_name = estimates_openap(
            aircraft_type,
            int(cruise_altitude_ft),
            None,  # mass_kg - will use default 85% MTOW
            distance_km,
        )
        result = {"performance": performance, "engine": engine_name}
        return json.dumps(result, indent=2)
    except OpenAPError as e:
        return f"Performance estimation error: {str(e)}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"


def get_system_status() -> str:
    """Get system status and capabilities.

    Returns:
        JSON string with system status information
    """
    status = {
        "system": "Aerospace MCP Server",
        "version": "0.1.0",
        "status": "operational",
        "capabilities": {
            "airport_search": True,
            "flight_planning": True,
            "great_circle_distance": True,
            "openap_performance": OPENAP_AVAILABLE,
        },
        "optional_features": {
            "openap_available": OPENAP_AVAILABLE,
        },
    }

    if OPENAP_AVAILABLE:
        status["openap_info"] = {
            "description": "OpenAP aircraft performance modeling available",
            "supported_aircraft": "A319, A320, A321, A332, A333, A343, A346, A359, A388, B737, B738, B739, B744, B747, B752, B753, B762, B763, B772, B773, B777, B787, and more",
        }
    else:
        status["openap_info"] = {
            "description": "OpenAP not available - install with: pip install openap",
            "note": "Flight planning will work without performance estimates",
        }

    return json.dumps(status, indent=2)
