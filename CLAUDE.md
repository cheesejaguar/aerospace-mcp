# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Running the application
```bash
# Run the FastAPI application
uvicorn main:app --reload

# Run on a specific port
uvicorn main:app --reload --port 8000
```

### Installing dependencies
```bash
# Required dependencies
pip install fastapi uvicorn pydantic airportsdata geographiclib

# Optional dependency for enhanced performance estimates
pip install openap
```

### Testing the API
```bash
# Check health endpoint
curl http://localhost:8000/health

# Interactive API documentation
# Navigate to: http://localhost:8000/docs
```

## Architecture

This is a Flight Planner API built with FastAPI that provides flight planning and estimation services. The application consists of a single main.py file with the following key components:

### Core Functionality
- **Airport Resolution**: Uses the `airportsdata` library to resolve cities to airports via IATA codes. Supports finding airports by city name with optional country filtering.
- **Route Calculation**: Computes great-circle routes between airports using the `geographiclib` library, generating polyline points at configurable intervals.
- **Performance Estimation**: Integrates with OpenAP (optional) to provide flight performance estimates including climb, cruise, and descent segments with fuel consumption calculations.

### API Endpoints
- `GET /health`: Service health check and capability reporting
- `GET /airports/by_city`: Find airports for a given city
- `POST /plan`: Generate a complete flight plan with route and performance estimates

### Key Design Patterns
- Uses Pydantic models for request/response validation (PlanRequest, PlanResponse, AirportOut, SegmentEst)
- Graceful degradation when OpenAP is not available (checked via OPENAP_AVAILABLE flag)
- In-memory airport database loaded at startup for fast lookups
- Conservative defaults (85% MTOW for mass when not specified)

### External Dependencies
- **airportsdata**: Airport database with IATA/ICAO codes and geographic data
- **geographiclib**: Geodesic calculations for great-circle routing
- **openap** (optional): Aircraft performance modeling for fuel and time estimates