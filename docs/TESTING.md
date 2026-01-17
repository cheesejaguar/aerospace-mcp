# Aerospace MCP Test Suite

This document describes the comprehensive test suite for the Aerospace MCP project.

## Overview

The test suite provides >80% code coverage and includes both unit tests and integration tests for all major components:

- **Airport resolution functionality** (`tests/test_airports.py`)
- **Flight planning and performance estimation** (`tests/test_plan.py`)
- **FastAPI health endpoints** (`tests/test_health.py`)
- **MCP server functionality** (`tests/test_mcp.py`)
- **Shared test fixtures and configuration** (`tests/conftest.py`)

## Test Organization

### Test Types

- **Unit Tests** (`@pytest.mark.unit`): Fast, isolated tests of individual functions
- **Integration Tests** (`@pytest.mark.integration`): Tests that verify component interactions
- **Slow Tests** (`@pytest.mark.slow`): Tests that may take longer to execute

### Test Files

```
tests/
├── conftest.py          # Shared fixtures and test configuration
├── test_airports.py     # Airport resolution and search tests
├── test_plan.py         # Flight planning and distance calculation tests
├── test_health.py       # FastAPI endpoint tests
└── test_mcp.py          # MCP server and tool tests
```

## Running Tests

### Prerequisites

Install test dependencies:

```bash
pip install -e ".[dev]"
# or with uv:
uv sync
```

### Quick Start

```bash
# Run all tests
python3 run_tests.py

# Run with coverage
python3 run_tests.py coverage

# Run only unit tests
python3 run_tests.py unit

# Run only integration tests
python3 run_tests.py integration

# Run fast tests (exclude slow ones)
python3 run_tests.py fast
```

### Direct pytest Usage

```bash
# All tests
pytest tests/

# Unit tests only
pytest -m unit tests/

# Integration tests only
pytest -m integration tests/

# Specific test file
pytest tests/test_airports.py

# Specific test class or function
pytest tests/test_airports.py::TestAirportFromIata::test_valid_iata_code

# With coverage
pytest --cov=aerospace_mcp --cov=app --cov-report=html tests/

# Parallel execution
pytest -n auto tests/
```

## Test Structure

### Fixtures (conftest.py)

The test suite includes comprehensive fixtures for:

- **Mock airport data**: Sample IATA airport database for consistent testing
- **Sample objects**: Pre-configured `AirportOut`, `PlanRequest` objects
- **OpenAP mocks**: Mock flight generators and fuel flow calculators
- **FastAPI client**: Test client for HTTP endpoint testing
- **Parameterized test data**: Common test cases for various scenarios

### Mock Data

Tests use realistic mock data:
- SJC (San Jose) ↔ NRT (Tokyo) for long-haul flights (~9000 km)
- SJC ↔ SFO for short flights (~65 km)
- Various aircraft types (A320, A359, B738)
- OpenAP performance estimates with realistic values

## Test Coverage Areas

### Airport Resolution (`test_airports.py`)

- ✅ IATA code lookup (valid/invalid/case-insensitive)
- ✅ City-based airport search with country filtering
- ✅ International airport preference in results
- ✅ Error handling for non-existent cities/airports
- ✅ Airport resolution with preferred IATA codes

### Flight Planning (`test_plan.py`)

- ✅ Great circle distance calculations
- ✅ Polyline generation with different step sizes
- ✅ OpenAP performance estimates (climb/cruise/descent)
- ✅ Aircraft mass handling (explicit vs. MTOW-based)
- ✅ Error cases (OpenAP unavailable, invalid aircraft)
- ✅ Short route handling (climb+descent > total distance)
- ✅ Unit conversion constants (NM ↔ KM)

### FastAPI Endpoints (`test_health.py`)

- ✅ `/health` endpoint with/without OpenAP
- ✅ `/airports/by_city` search functionality
- ✅ `/plan` flight planning endpoint
- ✅ Request validation and error responses
- ✅ Response format verification
- ✅ Custom parameter handling

### MCP Server (`test_mcp.py`)

- ✅ Server initialization and tool registration
- ✅ All MCP tools (`search_airports`, `plan_flight`, etc.)
- ✅ Tool input validation and error handling
- ✅ Text response formatting
- ✅ Transport mechanism support (stdio/SSE)
- ✅ Full workflow integration tests

## Mocking Strategy

The test suite uses comprehensive mocking to:

1. **Isolate dependencies**: Mock OpenAP, airport data, external libraries
2. **Control test data**: Use predictable, realistic sample data
3. **Test error conditions**: Simulate failures and edge cases
4. **Speed up execution**: Avoid network calls and heavy computations

### Key Mock Patterns

```python
# Mock airport data
@pytest.fixture
def mock_airports_iata(sample_airport_data):
    with patch('aerospace_mcp.core._AIRPORTS_IATA', sample_airport_data):
        yield sample_airport_data

# Mock OpenAP availability
@pytest.fixture(params=["with_openap", "without_openap"])
def openap_availability(request):
    available = request.param == "with_openap"
    with patch('aerospace_mcp.core.OPENAP_AVAILABLE', available):
        yield available

# Mock performance estimates
with patch('aerospace_mcp.server.estimates_openap') as mock_estimates:
    mock_estimates.return_value = (sample_estimates, "openap")
```

## Performance Testing

The test suite includes performance-related tests:

- Distance calculation accuracy for known routes
- Reasonable flight time estimates
- Fuel consumption within expected ranges
- Polyline generation efficiency

## Error Testing

Comprehensive error condition testing:

- Invalid IATA codes and non-existent cities
- OpenAP unavailability scenarios
- Aircraft type validation
- Request parameter validation
- Network/dependency failures

## Integration Testing

End-to-end workflow tests:

- Full flight planning pipeline (search → resolve → plan → estimate)
- MCP tool chaining and data flow
- FastAPI request/response cycles
- Error propagation through system layers

## Coverage Requirements

The test suite maintains >80% code coverage across:

- `aerospace_mcp/` core functionality
- `app/` FastAPI application
- `main.py` primary application logic

Coverage excludes:
- Test files themselves
- `__init__.py` files (configuration only)
- Abstract methods and protocols

## Continuous Integration

Tests are designed to run reliably in CI environments:

- No external dependencies (mocked)
- Deterministic execution
- Clear failure reporting
- Parallel execution support

## Adding New Tests

When adding new functionality:

1. **Add unit tests** for individual functions
2. **Add integration tests** for component interactions
3. **Update fixtures** if new mock data is needed
4. **Mark tests appropriately** (`@pytest.mark.unit`, etc.)
5. **Maintain coverage** above 80%

### Test Naming Convention

- Test files: `test_<module>.py`
- Test classes: `Test<Functionality>`
- Test methods: `test_<specific_behavior>`

Example:
```python
class TestAirportResolution:
    def test_valid_iata_code_returns_airport(self):
        ...

    def test_invalid_iata_code_returns_none(self):
        ...
```

## Debugging Tests

Useful debugging techniques:

```bash
# Run with verbose output
pytest -v tests/test_airports.py

# Run with print statements (disable capture)
pytest -s tests/test_airports.py

# Run single test with debugging
pytest --pdb tests/test_airports.py::test_specific_function

# Show test duration
pytest --durations=10 tests/
```
