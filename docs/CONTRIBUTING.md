# Contributing to Aerospace MCP

Thank you for your interest in contributing to the Aerospace MCP project! This document provides comprehensive guidelines for developers looking to contribute to the project.

## Table of Contents

- [Development Environment Setup](#development-environment-setup)
- [Project Architecture Overview](#project-architecture-overview)
- [Code Style Guidelines](#code-style-guidelines)
- [Testing Requirements](#testing-requirements)
- [Pull Request Process](#pull-request-process)
- [Adding New Features](#adding-new-features)
- [Common Development Tasks](#common-development-tasks)

## Development Environment Setup

### Prerequisites

- Python 3.8+ (Python 3.10+ recommended)
- UV (recommended) or pip for dependency management
- Git

### Setting Up with UV (Recommended)

UV is a fast Python package manager that provides excellent dependency management and virtual environment handling.

```bash
# Install UV if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/your-org/aerospace-mcp.git
cd aerospace-mcp

# Create and activate virtual environment with UV
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install core dependencies
uv add fastapi uvicorn pydantic airportsdata geographiclib

# Install optional dependencies for enhanced functionality
uv add openap

# Install development dependencies
uv add --dev pytest pytest-asyncio httpx black isort mypy pre-commit
```

### Setting Up with Pip

```bash
# Clone the repository
git clone https://github.com/your-org/aerospace-mcp.git
cd aerospace-mcp

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install fastapi uvicorn pydantic airportsdata geographiclib
pip install openap  # Optional for performance estimates
pip install pytest pytest-asyncio httpx black isort mypy pre-commit  # Development
```

### Environment Variables

Create a `.env` file in the project root for local development:

```bash
# Optional: Set custom port
PORT=8000

# Optional: Enable debug mode
DEBUG=true

# Optional: Set log level
LOG_LEVEL=info
```

### Running the Development Server

```bash
# Run with hot reload
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Or use the convenience command
python -m uvicorn main:app --reload
```

The API will be available at:
- Main application: http://localhost:8000
- Interactive documentation: http://localhost:8000/docs
- OpenAPI schema: http://localhost:8000/openapi.json

## Project Architecture Overview

### Core Components

```
aerospace-mcp/
├── main.py                 # Main FastAPI application
├── docs/                   # Documentation
├── tests/                  # Test suite
├── aerospace_mcp/          # Future: MCP-specific components
└── app/                    # Future: Modular application structure
```

### Key Modules in main.py

1. **Models** (`AirportOut`, `PlanRequest`, `PlanResponse`, `SegmentEst`)
   - Pydantic models for API request/response validation
   - Type-safe data structures

2. **Airport Resolution**
   - IATA/ICAO airport lookup using `airportsdata`
   - City-to-airport matching with country filtering
   - Intelligent airport selection (prefers international airports)

3. **Route Calculation**
   - Great-circle distance calculation using `geographiclib`
   - Polyline generation for route visualization
   - Configurable sampling intervals

4. **Performance Estimation**
   - OpenAP integration for aircraft performance modeling
   - Climb/cruise/descent segment analysis
   - Fuel consumption calculations

### Design Principles

- **Graceful Degradation**: Optional dependencies (like OpenAP) are handled gracefully
- **Type Safety**: Extensive use of Pydantic and type hints
- **Performance**: In-memory airport database for fast lookups
- **Extensibility**: Backend system allows for multiple performance estimation engines

## Code Style Guidelines

### Python Style

We follow PEP 8 with some specific preferences:

```bash
# Format code with Black
black main.py tests/

# Sort imports with isort
isort main.py tests/

# Type checking with mypy
mypy main.py
```

### Specific Guidelines

1. **Imports**
   - Use absolute imports
   - Group imports: standard library, third-party, local
   - Use type imports from `__future__` annotations

2. **Naming Conventions**
   - Functions and variables: `snake_case`
   - Classes: `PascalCase`
   - Constants: `UPPER_SNAKE_CASE`
   - Private functions: `_leading_underscore`

3. **Documentation**
   - Use docstrings for all public functions and classes
   - Include type hints for all function parameters and returns
   - Document complex algorithms and business logic

4. **Error Handling**
   - Use HTTPException for API errors with appropriate status codes
   - Provide meaningful error messages
   - Handle optional dependencies gracefully

### Example Code Style

```python
from typing import List, Optional
from fastapi import HTTPException


def _find_city_airports(city: str, country: Optional[str] = None) -> List[AirportOut]:
    """Find airports matching a city name.

    Args:
        city: City name to search for
        country: Optional ISO country code filter

    Returns:
        List of matching airports, sorted by preference

    Raises:
        HTTPException: When no airports are found
    """
    city_normalized = city.strip().lower()
    # Implementation...
```

## Testing Requirements

### Test Structure

```bash
tests/
├── __init__.py
├── test_main.py            # Main API endpoint tests
├── test_airport_resolution.py  # Airport lookup tests
├── test_route_calculation.py   # Route planning tests
└── test_performance.py     # Performance estimation tests
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/test_main.py

# Run tests with verbose output
pytest -v
```

### Writing Tests

Use pytest with async support for testing FastAPI endpoints:

```python
import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()

def test_airport_search():
    response = client.get("/airports/by_city?city=San Francisco")
    assert response.status_code == 200
    airports = response.json()
    assert len(airports) > 0
    assert any(ap["iata"] == "SFO" for ap in airports)
```

### Test Coverage Requirements

- Minimum 80% code coverage for new features
- All public API endpoints must have tests
- Error conditions should be tested
- Edge cases (empty responses, invalid inputs) should be covered

## Pull Request Process

### Before Submitting

1. **Code Quality**
   ```bash
   # Run formatting
   black main.py tests/
   isort main.py tests/

   # Run type checking
   mypy main.py

   # Run tests
   pytest --cov=. --cov-report=term-missing
   ```

2. **Documentation**
   - Update relevant documentation files
   - Add docstrings for new functions
   - Update API documentation if endpoints changed

3. **Testing**
   - Add tests for new features
   - Ensure all tests pass
   - Verify test coverage meets requirements

### PR Guidelines

1. **Title**: Use descriptive titles (e.g., "Add support for helicopter airports in city search")
2. **Description**:
   - Explain the problem being solved
   - Describe the solution approach
   - List any breaking changes
   - Include testing notes

3. **Size**: Keep PRs focused and reasonably sized
4. **Commits**: Use clear, descriptive commit messages

### Review Process

1. All PRs require at least one approval
2. CI/CD checks must pass
3. Code coverage should not decrease
4. Documentation must be updated for user-facing changes

## Adding New Features

### Airport Data Enhancements

To add support for additional airport data sources:

```python
# Add new backend in airport resolution section
def _find_airports_custom_backend(query: str) -> List[AirportOut]:
    # Implementation for new data source
    pass
```

### New Performance Backends

To add alternative performance estimation engines:

```python
# Add new backend option to PlanRequest model
class PlanRequest(BaseModel):
    backend: Literal["openap", "eurocontrol", "custom"] = "openap"

# Implement new estimation function
def estimates_custom(ac_type: str, ...) -> Tuple[dict, str]:
    # Implementation
    pass

# Update plan endpoint to handle new backend
```

### New API Endpoints

Follow this pattern for new endpoints:

```python
# 1. Define request/response models
class NewFeatureRequest(BaseModel):
    parameter: str = Field(..., description="Description")

class NewFeatureResponse(BaseModel):
    result: str

# 2. Implement business logic
def _handle_new_feature(request: NewFeatureRequest) -> NewFeatureResponse:
    # Implementation
    pass

# 3. Add endpoint
@app.post("/new-feature", response_model=NewFeatureResponse)
def new_feature_endpoint(req: NewFeatureRequest):
    try:
        return _handle_new_feature(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### MCP Integration

Future MCP (Model Context Protocol) integration should be added in the `aerospace_mcp/` directory:

```python
# aerospace_mcp/server.py
from mcp.server import Server
from mcp.types import Tool

server = Server("aerospace-mcp")

@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="plan_flight",
            description="Plan a flight route between cities",
            inputSchema={
                "type": "object",
                "properties": {
                    "departure": {"type": "string"},
                    "arrival": {"type": "string"}
                }
            }
        )
    ]
```

## Common Development Tasks

### Adding a New Aircraft Type

1. Verify aircraft is supported by OpenAP
2. Add to documentation
3. Test with sample flight plans

### Debugging Performance Issues

```bash
# Profile the application
python -m cProfile -o profile.stats main.py
python -c "import pstats; pstats.Stats('profile.stats').sort_stats('cumulative').print_stats()"

# Memory profiling
pip install memory-profiler
python -m memory_profiler main.py
```

### Database Schema Changes

Currently using in-memory airport data, but for future database integration:

1. Create migration scripts
2. Update model definitions
3. Add backward compatibility
4. Update tests

### Deployment Preparation

```bash
# Create requirements.txt for production
uv export --no-dev > requirements.txt

# Build Docker image
docker build -t aerospace-mcp .

# Test production build locally
docker run -p 8000:8000 aerospace-mcp
```

## Getting Help

- **Issues**: Create GitHub issues for bugs and feature requests
- **Discussions**: Use GitHub Discussions for questions and ideas
- **Documentation**: Check the `/docs` directory for detailed guides
- **Code Examples**: See tests for usage examples

## License

By contributing to this project, you agree that your contributions will be licensed under the project's LICENSE file.
