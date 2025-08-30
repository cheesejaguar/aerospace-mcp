# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Development Setup
```bash
# Install with UV (recommended)
uv venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
uv sync

# Alternative: Install with pip
pip install -e .[dev]
```

### Running the Application
```bash
# Run FastAPI HTTP server
aerospace-mcp-http
# Or with custom config
UVICORN_HOST=localhost UVICORN_PORT=8080 aerospace-mcp-http

# Run MCP server (stdio mode for production)
aerospace-mcp

# Run MCP server in TCP mode for debugging
aerospace-mcp --tcp localhost:8000
```

### Development & Testing
```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=app --cov=aerospace_mcp --cov-report=html

# Run specific test file
pytest tests/test_plan.py

# Run single test
pytest tests/test_plan.py::test_flight_planning_sjc_nrt -v

# Code formatting and linting
black .
ruff check .
ruff check --fix .

# Type checking
mypy app/ aerospace_mcp/

# Pre-commit hooks (runs all checks)
pre-commit run --all-files
```

### Docker Operations
```bash
# Build and run with Docker
docker build -t aerospace-mcp .
docker run -p 8080:8080 aerospace-mcp

# Run with Docker Compose
docker-compose up
docker-compose up --build  # Rebuild image
```

## Architecture

This is a dual-mode aerospace flight planning system that provides identical functionality through both HTTP API (FastAPI) and Model Context Protocol (MCP) interfaces.

### Core Architecture

The system follows a layered architecture with shared business logic:

**Shared Core Layer** (`aerospace_mcp/core.py`):
- Airport data management using `airportsdata` (loaded once at module level)
- Great-circle route calculations with `geographiclib`
- OpenAP aircraft performance modeling (with graceful fallback)
- Pydantic models for all data structures (AirportOut, PlanRequest, PlanResponse, SegmentEst)
- Business logic functions: `health()`, `airports_by_city()`, `plan_flight()`

**HTTP Interface** (`app/main.py`):
- FastAPI application with three endpoints: `/health`, `/airports/by_city`, `/plan`
- Environment-driven configuration (UVICORN_HOST, UVICORN_PORT, CORS_ORIGINS)
- OpenAPI documentation at `/docs` and AI plugin manifest at `/.well-known/ai-plugin.json`

**MCP Interface** (`aerospace_mcp/server.py`):
- Five MCP tools: search_airports, plan_flight, calculate_distance, get_aircraft_performance, get_system_status
- Dual transport modes: stdio (production) and SSE (development)
- JSON schema validation for all tool inputs

### Key Components

**Airport Resolution**:
- In-memory IATA database loaded at startup (`_AIRPORTS_IATA`)
- City-to-airport matching with country filtering and "International" airport preference
- Flexible input: city names, IATA codes, or explicit preferences

**Flight Performance Modeling**:
- OpenAP integration for realistic climb/cruise/descent profiles
- Aircraft mass resolution (85% MTOW default, with fallbacks)
- Fuel consumption calculations per flight phase
- Zero-wind assumptions for baseline estimates

**Route Generation**:
- Great-circle path calculation between airports
- Configurable sampling intervals for polyline generation
- Distance calculations in both km and nautical miles

### Entry Points

The package provides two console scripts:
- `aerospace-mcp-http`: Starts FastAPI server (calls `app.main:run`)
- `aerospace-mcp`: Starts MCP server (calls `aerospace_mcp.server:run`)

### Development Patterns

**Error Handling**: Custom exceptions (`FlightPlanError`, `AirportResolutionError`, `OpenAPError`) with proper HTTP status codes and MCP error messages.

**Testing Strategy**: Comprehensive mocking of external dependencies (OpenAP, airport data) to ensure deterministic tests. Separate test files for each interface layer.

**Configuration**: Environment variable-driven configuration with sensible defaults. Production configs use container environment variables.

**Package Management**: Uses UV for fast dependency resolution with fallback to standard pip. Development dependencies separated into `[dev]` optional group.

### External Dependencies

- **airportsdata**: Offline airport database (IATA/ICAO codes, coordinates)
- **geographiclib**: Precise geodesic calculations for great-circle routes
- **openap**: Open Aircraft Performance library for realistic flight modeling
- **fastapi/uvicorn**: HTTP API framework and ASGI server
- **mcp**: Anthropic's Model Context Protocol SDK
- **pydantic**: Data validation and serialization

### Safety Considerations

This codebase includes prominent safety disclaimers throughout documentation. The system is explicitly designed for educational/research purposes only and should never be used for real navigation or flight planning.
