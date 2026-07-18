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

# Run MCP server in SSE mode for debugging (defaults localhost:8001)
aerospace-mcp sse [host] [port]

# Invoke any tool directly from the terminal
aerospace-mcp-cli run convert_units --value 100 --from_unit kts --to_unit mps
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

**HTTP Interface** (root `main.py`, launched via `app/main.py`):
- Thin FastAPI layer over `aerospace_mcp.core` with three endpoints: `/health`, `/airports/by_city`, `/plan`
- Environment-driven configuration: UVICORN_HOST, UVICORN_PORT, CORS_ORIGINS (comma-separated origins; CORS disabled when unset), RATE_LIMIT_RPM (per-IP requests/minute, default 120, 0 disables), MAX_BODY_BYTES (default 1 MiB)
- OpenAPI documentation at `/docs`

**MCP Interface** (`aerospace_mcp/fastmcp_server.py`):
- 47 MCP tools plus 2 discovery tools, organized across 11 domain modules (core, atmosphere, frames, aerodynamics, propellers, rockets, orbits, gnc, performance, optimization, agents)
- Single registration source: `aerospace_mcp/tools/registry.py` (`ALL_TOOLS`) — shared by the FastMCP server and the CLI; a contract test keeps it in sync with `TOOL_REGISTRY` search metadata. New tools register in exactly two places: `ALL_TOOLS` and `TOOL_REGISTRY`.
- FastMCP framework for simplified tool development with decorators and automatic schema generation
- **Tool search tool** for dynamic tool discovery following Anthropic's guide

**Tool Discovery** (`aerospace_mcp/tools/tool_search.py`):
- `search_aerospace_tools`: Search tools by name, description, or functionality
- Supports regex patterns (e.g., `".*orbit.*"`) and natural language queries
- Category filtering for targeted discovery
- Returns `tool_reference` blocks compatible with Anthropic's tool search protocol
- `list_tool_categories`: Lists all 11 tool categories with tool counts

### Key Components

**Airport Resolution**:
- In-memory IATA database loaded at startup (`_AIRPORTS_IATA`)
- City-to-airport matching with country filtering and "International" airport preference
- Flexible input: city names, IATA codes, or explicit preferences

**Flight Performance Modeling**:
- OpenAP integration for realistic climb/cruise/descent profiles
- Aircraft mass resolution (85% MTOW default, with fallbacks; `assumptions.mass_source` records which)
- Fuel consumption calculations per flight phase
- Zero-wind default; optional cruise headwind via `estimates_openap(headwind_kts=...)` and the `wind` parameter on the `plan_flight` tool
- Multi-leg journeys via `plan_multi_leg()` / `plan_multi_leg_flight` tool (2-10 waypoints)

**Route Generation**:
- Great-circle path calculation between airports
- Configurable sampling intervals for polyline generation
- Distance calculations in both km and nautical miles

### Entry Points

The package provides three console scripts:
- `aerospace-mcp-http`: Starts FastAPI server (calls `app.main:run`)
- `aerospace-mcp`: Starts MCP server (calls `aerospace_mcp.fastmcp_server:run`)
- `aerospace-mcp-cli`: Direct CLI for any registered tool (list/search/info/run)

The legacy low-level MCP server (`aerospace_mcp/server.py`) was removed; `fastmcp_server.py` is the only MCP server.

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
- **fastmcp**: High-level Model Context Protocol framework for simplified tool development
- **pydantic**: Data validation and serialization
- **Optional packages**: ambiance (atmosphere), poliastro/astropy (space), aerosandbox (aerodynamics), rocketpy (rockets), filterpy/control (GNC)

### Safety Considerations

This codebase includes prominent safety disclaimers throughout documentation. The system is explicitly designed for educational/research purposes only and should never be used for real navigation or flight planning.
