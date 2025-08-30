# API Reference

Complete documentation for the Aerospace MCP Flight Planner API endpoints, data models, and integration examples.

## Table of Contents

- [Overview](#overview)
- [Base URL](#base-url)
- [Data Models](#data-models)
- [Endpoints](#endpoints)
- [Error Handling](#error-handling)
- [Examples](#examples)
- [MCP Tools](#mcp-tools)
- [Rate Limits](#rate-limits)

## Overview

The Aerospace MCP API provides flight planning services including:
- Airport lookup and resolution
- Great-circle route calculation
- Aircraft performance estimation using OpenAP
- Flight time and fuel consumption calculations

The API is built with FastAPI and provides both REST endpoints and MCP (Model Context Protocol) integration.

## Base URL

- **Development**: `http://localhost:8000`
- **Production**: Configure based on your deployment

All endpoints are prefixed with the base URL.

## Data Models

### Airport

Represents an airport with geographic and identification data.

```typescript
interface Airport {
  iata: string;           // IATA airport code (e.g., "SFO")
  icao: string;           // ICAO airport code (e.g., "KSFO")
  name: string;           // Full airport name
  city: string;           // City name
  country: string;        // ISO country code (e.g., "US")
  lat: number;           // Latitude in decimal degrees
  lon: number;           // Longitude in decimal degrees
  tz?: string;           // Timezone identifier (optional)
}
```

**Example**:
```json
{
  "iata": "SFO",
  "icao": "KSFO",
  "name": "San Francisco International Airport",
  "city": "San Francisco",
  "country": "US",
  "lat": 37.621311,
  "lon": -122.378958,
  "tz": "America/Los_Angeles"
}
```

### PlanRequest

Request model for flight planning.

```typescript
interface PlanRequest {
  depart_city: string;              // Departure city name
  arrive_city: string;              // Arrival city name
  depart_country?: string;          // ISO country code for departure
  arrive_country?: string;          // ISO country code for arrival
  prefer_depart_iata?: string;      // Force specific departure airport
  prefer_arrive_iata?: string;      // Force specific arrival airport
  ac_type: string;                  // ICAO aircraft type (e.g., "A320")
  cruise_alt_ft?: number;           // Cruise altitude in feet (default: 35000)
  mass_kg?: number;                 // Aircraft mass in kg (defaults to 85% MTOW)
  route_step_km?: number;           // Sampling step for route points (default: 25.0)
  backend: "openap";               // Performance estimation backend
}
```

**Field Constraints**:
- `cruise_alt_ft`: 8,000 - 45,000 feet
- `route_step_km`: > 1.0 km
- `ac_type`: Must be valid ICAO aircraft type
- `backend`: Currently only "openap" is supported

### SegmentEstimate

Performance estimates for a flight segment.

```typescript
interface SegmentEstimate {
  time_min: number;         // Segment time in minutes
  distance_km: number;      // Segment distance in kilometers
  avg_gs_kts: number;       // Average ground speed in knots
  fuel_kg: number;          // Fuel consumption in kilograms
}
```

### PlanResponse

Complete flight plan response.

```typescript
interface PlanResponse {
  engine: string;                           // Estimation engine used
  depart: Airport;                          // Departure airport
  arrive: Airport;                          // Arrival airport
  distance_km: number;                      // Total distance in kilometers
  distance_nm: number;                      // Total distance in nautical miles
  polyline: Array<[number, number]>;       // Route coordinates [[lat, lon], ...]
  estimates: {
    block: {                                // Total flight estimates
      time_min: number;
      fuel_kg: number;
    };
    climb: SegmentEstimate;                 // Climb phase
    cruise: SegmentEstimate;                // Cruise phase
    descent: SegmentEstimate;               // Descent phase
    assumptions: {
      zero_wind: boolean;                   // Wind assumptions
      mass_kg: number;                      // Aircraft mass used
      cruise_alt_ft: number;               // Cruise altitude used
    };
  };
}
```

## Endpoints

### GET /health

Health check endpoint providing service status and capabilities.

**Response**:
```typescript
interface HealthResponse {
  status: "ok" | "error";
  openap: boolean;           // Whether OpenAP is available
  airports_count: number;    // Number of airports in database
}
```

**Example Request**:
```bash
curl http://localhost:8000/health
```

**Example Response**:
```json
{
  "status": "ok",
  "openap": true,
  "airports_count": 28756
}
```

**Status Codes**:
- `200`: Service is healthy
- `503`: Service unavailable

---

### GET /airports/by_city

Find airports by city name with optional country filtering.

**Parameters**:
- `city` (required): City name to search for
- `country` (optional): ISO country code filter

**Response**: Array of `Airport` objects

**Example Requests**:
```bash
# Find airports in San Francisco
curl "http://localhost:8000/airports/by_city?city=San%20Francisco"

# Find airports in London, UK specifically
curl "http://localhost:8000/airports/by_city?city=London&country=GB"

# Find airports in Paris, France
curl "http://localhost:8000/airports/by_city?city=Paris&country=FR"
```

**Example Response**:
```json
[
  {
    "iata": "SFO",
    "icao": "KSFO",
    "name": "San Francisco International Airport",
    "city": "San Francisco",
    "country": "US",
    "lat": 37.621311,
    "lon": -122.378958,
    "tz": "America/Los_Angeles"
  },
  {
    "iata": "OAK",
    "icao": "KOAK",
    "name": "Oakland International Airport",
    "city": "Oakland",
    "country": "US",
    "lat": 37.721278,
    "lon": -122.22075,
    "tz": "America/Los_Angeles"
  }
]
```

**Status Codes**:
- `200`: Airports found
- `404`: No airports found for the specified city
- `400`: Invalid parameters

---

### POST /plan

Generate a complete flight plan with route calculation and performance estimates.

**Request Body**: `PlanRequest` object

**Response**: `PlanResponse` object

**Example Request**:
```bash
curl -X POST "http://localhost:8000/plan" \
  -H "Content-Type: application/json" \
  -d '{
    "depart_city": "New York",
    "arrive_city": "Los Angeles",
    "ac_type": "A320",
    "cruise_alt_ft": 37000,
    "route_step_km": 50.0,
    "backend": "openap"
  }'
```

**Example Request with Specific Airports**:
```bash
curl -X POST "http://localhost:8000/plan" \
  -H "Content-Type: application/json" \
  -d '{
    "depart_city": "New York",
    "arrive_city": "Los Angeles",
    "prefer_depart_iata": "JFK",
    "prefer_arrive_iata": "LAX",
    "ac_type": "B737",
    "cruise_alt_ft": 35000,
    "mass_kg": 65000,
    "route_step_km": 25.0,
    "backend": "openap"
  }'
```

**Example Response**:
```json
{
  "engine": "openap",
  "depart": {
    "iata": "JFK",
    "icao": "KJFK",
    "name": "John F Kennedy International Airport",
    "city": "New York",
    "country": "US",
    "lat": 40.641766,
    "lon": -73.780968,
    "tz": "America/New_York"
  },
  "arrive": {
    "iata": "LAX",
    "icao": "KLAX",
    "name": "Los Angeles International Airport",
    "city": "Los Angeles",
    "country": "US",
    "lat": 33.943001,
    "lon": -118.410042,
    "tz": "America/Los_Angeles"
  },
  "distance_km": 3983.5,
  "distance_nm": 2150.4,
  "polyline": [
    [40.641766, -73.780968],
    [40.688, -74.234],
    [40.734, -74.687],
    "... more coordinates ...",
    [33.943001, -118.410042]
  ],
  "estimates": {
    "block": {
      "time_min": 318.2,
      "fuel_kg": 8542.1
    },
    "climb": {
      "time_min": 12.5,
      "distance_km": 87.3,
      "avg_gs_kts": 285.4,
      "fuel_kg": 1876.2
    },
    "cruise": {
      "time_min": 285.1,
      "distance_km": 3821.7,
      "avg_gs_kts": 459.8,
      "fuel_kg": 5894.3
    },
    "descent": {
      "time_min": 20.6,
      "distance_km": 74.5,
      "avg_gs_kts": 198.7,
      "fuel_kg": 771.6
    },
    "assumptions": {
      "zero_wind": true,
      "mass_kg": 68850.0,
      "cruise_alt_ft": 35000
    }
  }
}
```

**Status Codes**:
- `200`: Flight plan generated successfully
- `400`: Invalid request parameters
- `404`: Airport not found for specified city
- `501`: Backend unavailable (e.g., OpenAP not installed)

## Error Handling

The API uses standard HTTP status codes and returns error details in JSON format.

### Error Response Format

```typescript
interface ErrorResponse {
  detail: string;           // Human-readable error message
}
```

### Common Error Scenarios

#### 400 Bad Request

**Cause**: Invalid input parameters

**Example**:
```json
{
  "detail": "departure: IATA 'XYZ' not found."
}
```

**Common triggers**:
- Invalid IATA code
- Identical departure and arrival cities
- Aircraft type not recognized
- Altitude outside valid range
- Invalid route step size

#### 404 Not Found

**Cause**: Resource not found

**Example**:
```json
{
  "detail": "departure: no airport for city='InvalidCity' (country=ANY)."
}
```

**Common triggers**:
- City name not in airport database
- Misspelled city names
- Very small cities without commercial airports

#### 501 Not Implemented

**Cause**: Required backend service unavailable

**Example**:
```json
{
  "detail": "OpenAP backend unavailable. Please `pip install openap`."
}
```

**Common triggers**:
- OpenAP package not installed
- Unsupported backend specified

## Examples

### Complete Flight Planning Workflow

```python
import requests
import json

BASE_URL = "http://localhost:8000"

# 1. Check service health
health = requests.get(f"{BASE_URL}/health")
print(f"Service status: {health.json()}")

# 2. Search for departure airports
departure_airports = requests.get(
    f"{BASE_URL}/airports/by_city",
    params={"city": "Tokyo", "country": "JP"}
)

print("Tokyo airports:")
for airport in departure_airports.json():
    print(f"  {airport['iata']}: {airport['name']}")

# 3. Search for arrival airports
arrival_airports = requests.get(
    f"{BASE_URL}/airports/by_city",
    params={"city": "Sydney", "country": "AU"}
)

print("Sydney airports:")
for airport in arrival_airports.json():
    print(f"  {airport['iata']}: {airport['name']}")

# 4. Plan the flight
flight_plan = {
    "depart_city": "Tokyo",
    "arrive_city": "Sydney",
    "prefer_depart_iata": "NRT",  # Narita
    "prefer_arrive_iata": "SYD",  # Sydney Kingsford Smith
    "ac_type": "B777",
    "cruise_alt_ft": 39000,
    "mass_kg": 220000,  # Heavy long-haul config
    "route_step_km": 100.0,  # Lower resolution for long route
    "backend": "openap"
}

response = requests.post(
    f"{BASE_URL}/plan",
    headers={"Content-Type": "application/json"},
    json=flight_plan
)

if response.status_code == 200:
    plan = response.json()

    print(f"\n=== Flight Plan: {plan['depart']['iata']} → {plan['arrive']['iata']} ===")
    print(f"Distance: {plan['distance_nm']:.0f} NM ({plan['distance_km']:.0f} km)")
    print(f"Aircraft: {flight_plan['ac_type']} at FL{flight_plan['cruise_alt_ft']//100}")
    print(f"Mass: {plan['estimates']['assumptions']['mass_kg']:,.0f} kg")

    print(f"\n=== Performance Estimates ===")
    print(f"Block Time: {plan['estimates']['block']['time_min']:.0f} min ({plan['estimates']['block']['time_min']/60:.1f} hours)")
    print(f"Block Fuel: {plan['estimates']['block']['fuel_kg']:,.0f} kg")

    print(f"\n=== Segment Breakdown ===")
    for segment in ['climb', 'cruise', 'descent']:
        seg = plan['estimates'][segment]
        print(f"{segment.capitalize()}: {seg['time_min']:.0f}min, {seg['distance_km']:.0f}km, {seg['fuel_kg']:.0f}kg")

    print(f"\n=== Route Points ===")
    print(f"Total waypoints: {len(plan['polyline'])}")
    print(f"First point: {plan['polyline'][0]}")  # Departure
    print(f"Last point: {plan['polyline'][-1]}")  # Arrival

else:
    print(f"Error {response.status_code}: {response.json()}")
```

### Batch Processing Multiple Routes

```python
import asyncio
import aiohttp
import json

async def plan_flight_async(session, flight_request):
    """Plan a single flight asynchronously."""
    async with session.post(
        "http://localhost:8000/plan",
        json=flight_request
    ) as response:
        if response.status == 200:
            return await response.json()
        else:
            error = await response.json()
            return {"error": error, "request": flight_request}

async def plan_multiple_flights():
    """Plan multiple flights concurrently."""

    # Define multiple flight requests
    flight_requests = [
        {
            "depart_city": "New York", "arrive_city": "London",
            "ac_type": "A330", "cruise_alt_ft": 37000, "backend": "openap"
        },
        {
            "depart_city": "London", "arrive_city": "Dubai",
            "ac_type": "B777", "cruise_alt_ft": 39000, "backend": "openap"
        },
        {
            "depart_city": "Dubai", "arrive_city": "Singapore",
            "ac_type": "A380", "cruise_alt_ft": 41000, "backend": "openap"
        },
        {
            "depart_city": "Singapore", "arrive_city": "Sydney",
            "ac_type": "B787", "cruise_alt_ft": 40000, "backend": "openap"
        }
    ]

    async with aiohttp.ClientSession() as session:
        # Execute all flight plans concurrently
        tasks = [plan_flight_async(session, req) for req in flight_requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        total_distance = 0
        total_time = 0
        total_fuel = 0

        print("=== Multi-Leg Journey Results ===")
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"Leg {i+1}: ERROR - {result}")
                continue

            if "error" in result:
                print(f"Leg {i+1}: ERROR - {result['error']['detail']}")
                continue

            # Extract key metrics
            depart = result['depart']['iata']
            arrive = result['arrive']['iata']
            distance_nm = result['distance_nm']
            time_min = result['estimates']['block']['time_min']
            fuel_kg = result['estimates']['block']['fuel_kg']

            total_distance += distance_nm
            total_time += time_min
            total_fuel += fuel_kg

            print(f"Leg {i+1}: {depart} → {arrive}")
            print(f"  Distance: {distance_nm:.0f} NM")
            print(f"  Time: {time_min:.0f} min ({time_min/60:.1f} hours)")
            print(f"  Fuel: {fuel_kg:,.0f} kg")

        print(f"\n=== Journey Totals ===")
        print(f"Total Distance: {total_distance:.0f} NM")
        print(f"Total Time: {total_time:.0f} min ({total_time/60:.1f} hours)")
        print(f"Total Fuel: {total_fuel:,.0f} kg")

# Run the example
if __name__ == "__main__":
    asyncio.run(plan_multiple_flights())
```

### Aircraft Comparison

```python
def compare_aircraft_performance():
    """Compare different aircraft on the same route."""

    aircraft_types = ["A320", "A321", "B737", "B738"]
    base_request = {
        "depart_city": "Chicago",
        "arrive_city": "Denver",
        "cruise_alt_ft": 35000,
        "backend": "openap"
    }

    print("=== Aircraft Performance Comparison ===")
    print("Route: Chicago → Denver")
    print(f"{'Aircraft':<8} {'Time (min)':<10} {'Fuel (kg)':<10} {'Range (NM)':<12}")
    print("-" * 50)

    for aircraft in aircraft_types:
        request = {**base_request, "ac_type": aircraft}

        response = requests.post(
            "http://localhost:8000/plan",
            json=request
        )

        if response.status_code == 200:
            plan = response.json()
            time_min = plan['estimates']['block']['time_min']
            fuel_kg = plan['estimates']['block']['fuel_kg']
            distance_nm = plan['distance_nm']

            print(f"{aircraft:<8} {time_min:<10.0f} {fuel_kg:<10.0f} {distance_nm:<12.0f}")
        else:
            print(f"{aircraft:<8} {'ERROR':<10} {'ERROR':<10} {'ERROR':<12}")

compare_aircraft_performance()
```

## MCP Tools

When using the Aerospace MCP through Model Context Protocol, the following tools are available:

### flight_plan

Plan a flight route with performance estimates.

**Parameters**:
```typescript
{
  departure_city: string;      // Departure city name
  arrival_city: string;        // Arrival city name
  aircraft_type: string;       // ICAO aircraft type
  cruise_altitude?: number;    // Cruise altitude in feet
  departure_country?: string;  // ISO country code
  arrival_country?: string;    // ISO country code
}
```

**Returns**: Complete flight plan with route and performance data

### find_airports

Search for airports by city name.

**Parameters**:
```typescript
{
  city: string;           // City name to search
  country?: string;       // Optional country filter
}
```

**Returns**: Array of matching airports

### service_status

Check the health and capabilities of the flight planning service.

**Parameters**: None

**Returns**: Service status including OpenAP availability

## Rate Limits

Currently, the API does not enforce rate limits, but consider implementing them for production use:

**Recommended Limits**:
- Health checks: 60/minute
- Airport searches: 100/minute
- Flight planning: 20/minute (due to computational complexity)

**Implementation Example**:
```python
# Using slowapi (not currently implemented)
@app.post("/plan")
@limiter.limit("20/minute")
def plan(request: Request, req: PlanRequest):
    # Implementation
    pass
```

**Rate Limit Headers** (future):
- `X-RateLimit-Limit`: Request limit per window
- `X-RateLimit-Remaining`: Requests remaining in current window
- `X-RateLimit-Reset`: Window reset time

For high-volume usage, consider:
- Implementing caching for repeated requests
- Using batch endpoints for multiple plans
- Optimizing route resolution with larger step sizes
- Implementing request queuing for burst traffic

## OpenAPI Specification

The complete OpenAPI specification is available at:
- **Interactive Docs**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`
- **JSON Schema**: `http://localhost:8000/openapi.json`

This provides machine-readable API definitions for code generation and testing tools.
