"""MCP Server implementation for Aerospace flight planning tools."""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Sequence
from mcp.server import Server
from mcp.types import (
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
    CallToolRequest,
    ListToolsRequest,
)
import mcp.server.stdio
import mcp.server.sse
from pydantic import BaseModel, ValidationError

from .core import (
    _airport_from_iata,
    _find_city_airports,
    _resolve_endpoint,
    great_circle_points,
    estimates_openap,
    AirportOut,
    PlanRequest,
    AirportResolutionError,
    OpenAPError,
    OPENAP_AVAILABLE,
    NM_PER_KM,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize MCP server
server = Server("aerospace-mcp")


# Tool definitions
TOOLS = [
    Tool(
        name="search_airports",
        description="Search for airports by IATA code or city name",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "IATA code (e.g., 'SJC') or city name (e.g., 'San Jose')"
                },
                "country": {
                    "type": "string", 
                    "description": "Optional ISO country code to filter by (e.g., 'US', 'JP')"
                },
                "query_type": {
                    "type": "string",
                    "enum": ["iata", "city", "auto"],
                    "description": "Type of query - 'iata' for IATA codes, 'city' for city names, 'auto' to detect",
                    "default": "auto"
                }
            },
            "required": ["query"]
        }
    ),
    
    Tool(
        name="plan_flight",
        description="Plan a flight route between two airports with performance estimates",
        inputSchema={
            "type": "object",
            "properties": {
                "departure": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "Departure city name"},
                        "country": {"type": "string", "description": "Departure country code (optional)"},
                        "iata": {"type": "string", "description": "Preferred departure IATA code (optional)"}
                    },
                    "required": ["city"]
                },
                "arrival": {
                    "type": "object", 
                    "properties": {
                        "city": {"type": "string", "description": "Arrival city name"},
                        "country": {"type": "string", "description": "Arrival country code (optional)"},
                        "iata": {"type": "string", "description": "Preferred arrival IATA code (optional)"}
                    },
                    "required": ["city"]
                },
                "aircraft": {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string", "description": "ICAO aircraft type (e.g., 'A320', 'B738', 'A359')"},
                        "cruise_altitude": {"type": "integer", "description": "Cruise altitude in feet", "minimum": 8000, "maximum": 45000, "default": 35000},
                        "mass_kg": {"type": "number", "description": "Aircraft mass in kg (optional, uses 85% MTOW if not specified)"}
                    },
                    "required": ["type"]
                },
                "route_options": {
                    "type": "object", 
                    "properties": {
                        "step_km": {"type": "number", "description": "Distance between polyline points in km", "minimum": 1.0, "default": 25.0}
                    }
                }
            },
            "required": ["departure", "arrival", "aircraft"]
        }
    ),
    
    Tool(
        name="calculate_distance",
        description="Calculate great circle distance between two points",
        inputSchema={
            "type": "object",
            "properties": {
                "origin": {
                    "type": "object",
                    "properties": {
                        "latitude": {"type": "number", "minimum": -90, "maximum": 90},
                        "longitude": {"type": "number", "minimum": -180, "maximum": 180}
                    },
                    "required": ["latitude", "longitude"]
                },
                "destination": {
                    "type": "object",
                    "properties": {
                        "latitude": {"type": "number", "minimum": -90, "maximum": 90},
                        "longitude": {"type": "number", "minimum": -180, "maximum": 180}
                    },
                    "required": ["latitude", "longitude"]
                },
                "step_km": {
                    "type": "number",
                    "description": "Step size for polyline generation in km",
                    "minimum": 1.0,
                    "default": 50.0
                }
            },
            "required": ["origin", "destination"]
        }
    ),
    
    Tool(
        name="get_aircraft_performance",
        description="Get performance estimates for an aircraft type (requires OpenAP)",
        inputSchema={
            "type": "object", 
            "properties": {
                "aircraft_type": {
                    "type": "string",
                    "description": "ICAO aircraft type code (e.g., 'A320', 'B738')"
                },
                "distance_km": {
                    "type": "number",
                    "description": "Route distance in kilometers",
                    "minimum": 1.0
                },
                "cruise_altitude": {
                    "type": "integer", 
                    "description": "Cruise altitude in feet",
                    "minimum": 8000,
                    "maximum": 45000,
                    "default": 35000
                },
                "mass_kg": {
                    "type": "number",
                    "description": "Aircraft mass in kg (optional)"
                }
            },
            "required": ["aircraft_type", "distance_km"]
        }
    ),
    
    Tool(
        name="get_system_status",
        description="Get system status and capabilities",
        inputSchema={
            "type": "object",
            "properties": {},
            "additionalProperties": False
        }
    )
]


@server.list_tools()
async def handle_list_tools() -> list[Tool]:
    """List all available tools."""
    return TOOLS


@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[TextContent | ImageContent | EmbeddedResource]:
    """Handle tool calls."""
    try:
        if name == "search_airports":
            return await _handle_search_airports(arguments)
        elif name == "plan_flight":
            return await _handle_plan_flight(arguments)
        elif name == "calculate_distance":
            return await _handle_calculate_distance(arguments)
        elif name == "get_aircraft_performance":
            return await _handle_get_aircraft_performance(arguments)
        elif name == "get_system_status":
            return await _handle_get_system_status(arguments)
        else:
            raise ValueError(f"Unknown tool: {name}")
    
    except Exception as e:
        logger.error(f"Error in tool {name}: {str(e)}", exc_info=True)
        return [TextContent(type="text", text=f"Error: {str(e)}")]


async def _handle_search_airports(arguments: dict) -> list[TextContent]:
    """Handle airport search requests."""
    query = arguments.get("query", "").strip()
    country = arguments.get("country")
    query_type = arguments.get("query_type", "auto")
    
    if not query:
        return [TextContent(type="text", text="Error: Query parameter is required")]
    
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
            return [TextContent(type="text", text=message)]
        
        # Format results
        response_lines = [f"Found {len(results)} airport(s):"]
        for airport in results:
            line = f"• {airport.iata} ({airport.icao}) - {airport.name}"
            line += f"\n  City: {airport.city}, {airport.country}"
            line += f"\n  Coordinates: {airport.lat:.4f}, {airport.lon:.4f}"
            if airport.tz:
                line += f"\n  Timezone: {airport.tz}"
            response_lines.append(line)
        
        return [TextContent(type="text", text="\n\n".join(response_lines))]
    
    except Exception as e:
        return [TextContent(type="text", text=f"Search error: {str(e)}")]


async def _handle_plan_flight(arguments: dict) -> list[TextContent]:
    """Handle flight planning requests."""
    try:
        # Extract parameters
        departure = arguments.get("departure", {})
        arrival = arguments.get("arrival", {})
        aircraft = arguments.get("aircraft", {})
        route_options = arguments.get("route_options", {})
        
        # Build PlanRequest
        plan_request = PlanRequest(
            depart_city=departure.get("city", ""),
            arrive_city=arrival.get("city", ""),
            depart_country=departure.get("country"),
            arrive_country=arrival.get("country"),
            prefer_depart_iata=departure.get("iata"),
            prefer_arrive_iata=arrival.get("iata"),
            ac_type=aircraft.get("type", ""),
            cruise_alt_ft=aircraft.get("cruise_altitude", 35000),
            mass_kg=aircraft.get("mass_kg"),
            route_step_km=route_options.get("step_km", 25.0)
        )
        
        # Validate same city check
        if (plan_request.depart_city.strip().lower() == plan_request.arrive_city.strip().lower() 
            and not plan_request.prefer_arrive_iata and not plan_request.prefer_depart_iata):
            return [TextContent(type="text", text="Error: Departure and arrival cities are identical. Please specify airports explicitly.")]
        
        # Resolve airports
        try:
            dep = _resolve_endpoint(plan_request.depart_city, plan_request.depart_country, 
                                    plan_request.prefer_depart_iata, "departure")
            arr = _resolve_endpoint(plan_request.arrive_city, plan_request.arrive_country,
                                    plan_request.prefer_arrive_iata, "arrival")
        except AirportResolutionError as e:
            return [TextContent(type="text", text=f"Airport resolution error: {str(e)}")]
        
        # Calculate route
        polyline, distance_km = great_circle_points(
            dep.lat, dep.lon, arr.lat, arr.lon, plan_request.route_step_km
        )
        distance_nm = distance_km * NM_PER_KM
        
        # Get performance estimates
        try:
            estimates, engine_name = estimates_openap(
                plan_request.ac_type, 
                plan_request.cruise_alt_ft,
                plan_request.mass_kg, 
                distance_km
            )
        except OpenAPError as e:
            return [TextContent(type="text", text=f"Performance estimation error: {str(e)}")]
        
        # Format response
        response_lines = [
            f"Flight Plan: {dep.iata} → {arr.iata}",
            f"Route: {dep.name} → {arr.name}",
            f"Distance: {distance_km:.0f} km ({distance_nm:.0f} NM)",
            f"Aircraft: {plan_request.ac_type}",
            f"Cruise Altitude: {plan_request.cruise_alt_ft:,} ft",
            "",
            "Performance Estimates (OpenAP):",
            f"• Block Time: {estimates['block']['time_min']:.0f} minutes ({estimates['block']['time_min']/60:.1f} hours)",
            f"• Block Fuel: {estimates['block']['fuel_kg']:.0f} kg",
            "",
            "Flight Segments:",
            f"• Climb: {estimates['climb']['time_min']:.0f} min, {estimates['climb']['distance_km']:.0f} km, {estimates['climb']['fuel_kg']:.0f} kg fuel",
            f"• Cruise: {estimates['cruise']['time_min']:.0f} min, {estimates['cruise']['distance_km']:.0f} km, {estimates['cruise']['fuel_kg']:.0f} kg fuel",
            f"• Descent: {estimates['descent']['time_min']:.0f} min, {estimates['descent']['distance_km']:.0f} km, {estimates['descent']['fuel_kg']:.0f} kg fuel",
            "",
            f"Route Polyline: {len(polyline)} points (every {plan_request.route_step_km} km)",
            f"Mass Assumption: {estimates['assumptions']['mass_kg']:.0f} kg"
        ]
        
        return [TextContent(type="text", text="\n".join(response_lines))]
        
    except ValidationError as e:
        return [TextContent(type="text", text=f"Validation error: {str(e)}")]
    except Exception as e:
        return [TextContent(type="text", text=f"Flight planning error: {str(e)}")]


async def _handle_calculate_distance(arguments: dict) -> list[TextContent]:
    """Handle distance calculation requests."""
    try:
        origin = arguments.get("origin", {})
        destination = arguments.get("destination", {})
        step_km = arguments.get("step_km", 50.0)
        
        lat1 = origin.get("latitude")
        lon1 = origin.get("longitude")
        lat2 = destination.get("latitude") 
        lon2 = destination.get("longitude")
        
        if None in [lat1, lon1, lat2, lon2]:
            return [TextContent(type="text", text="Error: Origin and destination coordinates are required")]
        
        polyline, distance_km = great_circle_points(lat1, lon1, lat2, lon2, step_km)
        distance_nm = distance_km * NM_PER_KM
        
        response_lines = [
            f"Great Circle Distance Calculation",
            f"Origin: {lat1:.4f}, {lon1:.4f}",
            f"Destination: {lat2:.4f}, {lon2:.4f}",
            f"Distance: {distance_km:.2f} km ({distance_nm:.2f} NM)", 
            f"Polyline Points: {len(polyline)} (every {step_km} km)"
        ]
        
        return [TextContent(type="text", text="\n".join(response_lines))]
        
    except Exception as e:
        return [TextContent(type="text", text=f"Distance calculation error: {str(e)}")]


async def _handle_get_aircraft_performance(arguments: dict) -> list[TextContent]:
    """Handle aircraft performance requests."""
    try:
        aircraft_type = arguments.get("aircraft_type", "")
        distance_km = arguments.get("distance_km", 0)
        cruise_altitude = arguments.get("cruise_altitude", 35000)
        mass_kg = arguments.get("mass_kg")
        
        if not aircraft_type:
            return [TextContent(type="text", text="Error: Aircraft type is required")]
        
        if distance_km <= 0:
            return [TextContent(type="text", text="Error: Distance must be positive")]
        
        estimates, engine_name = estimates_openap(aircraft_type, cruise_altitude, mass_kg, distance_km)
        
        response_lines = [
            f"Aircraft Performance Estimates ({engine_name})",
            f"Aircraft: {aircraft_type}",
            f"Distance: {distance_km:.0f} km",
            f"Cruise Altitude: {cruise_altitude:,} ft",
            f"Mass: {estimates['assumptions']['mass_kg']:.0f} kg",
            "",
            "Block Estimates:",
            f"• Time: {estimates['block']['time_min']:.0f} minutes ({estimates['block']['time_min']/60:.1f} hours)",
            f"• Fuel: {estimates['block']['fuel_kg']:.0f} kg",
            "",
            "Segment Breakdown:",
            f"• Climb: {estimates['climb']['time_min']:.0f} min, {estimates['climb']['distance_km']:.0f} km, {estimates['climb']['fuel_kg']:.0f} kg",
            f"• Cruise: {estimates['cruise']['time_min']:.0f} min, {estimates['cruise']['distance_km']:.0f} km, {estimates['cruise']['fuel_kg']:.0f} kg", 
            f"• Descent: {estimates['descent']['time_min']:.0f} min, {estimates['descent']['distance_km']:.0f} km, {estimates['descent']['fuel_kg']:.0f} kg"
        ]
        
        return [TextContent(type="text", text="\n".join(response_lines))]
        
    except OpenAPError as e:
        return [TextContent(type="text", text=f"Performance estimation error: {str(e)}")]
    except Exception as e:
        return [TextContent(type="text", text=f"Aircraft performance error: {str(e)}")]


async def _handle_get_system_status(arguments: dict) -> list[TextContent]:
    """Handle system status requests."""
    try:
        from .core import _AIRPORTS_IATA
        
        status_lines = [
            "Aerospace MCP Server Status",
            f"OpenAP Available: {'Yes' if OPENAP_AVAILABLE else 'No'}",
            f"Airports Loaded: {len(_AIRPORTS_IATA):,}",
            "",
            "Available Tools:",
            "• search_airports - Search for airports by IATA or city",
            "• plan_flight - Plan flight routes with performance estimates", 
            "• calculate_distance - Calculate great circle distances",
            "• get_aircraft_performance - Get aircraft performance estimates",
            "• get_system_status - Get this status information"
        ]
        
        if not OPENAP_AVAILABLE:
            status_lines.extend([
                "",
                "Note: OpenAP is not available. Flight performance estimates will not work.",
                "Install with: pip install openap"
            ])
        
        return [TextContent(type="text", text="\n".join(status_lines))]
        
    except Exception as e:
        return [TextContent(type="text", text=f"Status error: {str(e)}")]


def run_stdio():
    """Run the MCP server with stdio transport."""
    async def _main():
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options()
            )
    
    asyncio.run(_main())


def run_sse(host: str = "localhost", port: int = 8001):
    """Run the MCP server with SSE transport."""
    import mcp.server.sse
    
    sse_app = mcp.server.sse.SseServerTransport("/sse")
    
    async def _main():
        async with sse_app.run_server() as server_context:
            await server.run(
                server_context.read_stream,
                server_context.write_stream, 
                server.create_initialization_options()
            )
    
    asyncio.run(_main())


def run():
    """Main entry point - defaults to stdio."""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "sse":
        host = sys.argv[2] if len(sys.argv) > 2 else "localhost"
        port = int(sys.argv[3]) if len(sys.argv) > 3 else 8001
        run_sse(host, port)
    else:
        run_stdio()


if __name__ == "__main__":
    run()