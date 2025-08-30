"""Test configuration and fixtures for Aerospace MCP tests."""

import pytest
from typing import Dict, Any, List, Optional
from unittest.mock import MagicMock, patch
import pandas as pd
from fastapi.testclient import TestClient

from aerospace_mcp.core import AirportOut, PlanRequest, SegmentEst, OPENAP_AVAILABLE
from main import app


# Test data fixtures
@pytest.fixture
def sample_airport_data() -> Dict[str, Dict[str, Any]]:
    """Mock airport data for testing."""
    return {
        "SJC": {
            "iata": "SJC",
            "icao": "KSJC",
            "name": "San Jose International Airport",
            "city": "San Jose",
            "country": "US",
            "lat": 37.3626,
            "lon": -121.929,
            "tz": "America/Los_Angeles"
        },
        "NRT": {
            "iata": "NRT",
            "icao": "RJAA",
            "name": "Narita International Airport",
            "city": "Tokyo",
            "country": "JP",
            "lat": 35.7647,
            "lon": 140.386,
            "tz": "Asia/Tokyo"
        },
        "SFO": {
            "iata": "SFO",
            "icao": "KSFO",
            "name": "San Francisco International Airport",
            "city": "San Francisco",
            "country": "US",
            "lat": 37.6213,
            "lon": -122.379,
            "tz": "America/Los_Angeles"
        },
        "JFK": {
            "iata": "JFK",
            "icao": "KJFK",
            "name": "John F Kennedy International Airport",
            "city": "New York",
            "country": "US",
            "lat": 40.6398,
            "lon": -73.7789,
            "tz": "America/New_York"
        },
        "XXX": {
            # Invalid airport for error testing
            "iata": "XXX",
            "icao": "",
            "name": "Non-existent Airport",
            "city": "Nowhere",
            "country": "XX",
            "lat": 0.0,
            "lon": 0.0
        }
    }


@pytest.fixture
def sjc_airport() -> AirportOut:
    """Sample SJC airport for testing."""
    return AirportOut(
        iata="SJC",
        icao="KSJC",
        name="San Jose International Airport",
        city="San Jose",
        country="US",
        lat=37.3626,
        lon=-121.929,
        tz="America/Los_Angeles"
    )


@pytest.fixture
def nrt_airport() -> AirportOut:
    """Sample NRT airport for testing."""
    return AirportOut(
        iata="NRT",
        icao="RJAA",
        name="Narita International Airport",
        city="Tokyo",
        country="JP",
        lat=35.7647,
        lon=140.386,
        tz="Asia/Tokyo"
    )


@pytest.fixture
def sample_plan_request() -> PlanRequest:
    """Sample flight plan request for testing."""
    return PlanRequest(
        depart_city="San Jose",
        arrive_city="Tokyo",
        depart_country="US",
        arrive_country="JP",
        ac_type="A359",
        cruise_alt_ft=35000,
        route_step_km=25.0
    )


@pytest.fixture
def mock_openap_flight_generator():
    """Mock OpenAP FlightGenerator for testing."""
    mock_gen = MagicMock()

    # Mock climb segment
    climb_data = pd.DataFrame({
        't': [0, 10, 20, 30, 40, 50],
        's': [0, 1000, 2500, 4500, 7000, 10000],  # meters
        'altitude': [1000, 5000, 10000, 15000, 20000, 25000],  # feet
        'groundspeed': [200, 220, 240, 260, 280, 300],  # knots
        'vertical_rate': [2000, 1800, 1600, 1400, 1200, 1000]  # fpm
    })

    # Mock cruise segment
    cruise_data = pd.DataFrame({
        't': [0, 10, 20, 30],
        's': [0, 5000, 10000, 15000],  # meters
        'altitude': [35000, 35000, 35000, 35000],  # feet
        'groundspeed': [450, 450, 450, 450],  # knots
        'vertical_rate': [0, 0, 0, 0]  # fpm
    })

    # Mock descent segment
    descent_data = pd.DataFrame({
        't': [0, 10, 20, 30, 40, 50],
        's': [0, 2000, 4000, 6000, 8000, 12000],  # meters
        'altitude': [35000, 30000, 25000, 15000, 8000, 2000],  # feet
        'groundspeed': [420, 400, 380, 350, 320, 280],  # knots
        'vertical_rate': [-1000, -1200, -1400, -1600, -1800, -2000]  # fpm
    })

    mock_gen.climb.return_value = climb_data
    mock_gen.cruise.return_value = cruise_data
    mock_gen.descent.return_value = descent_data

    return mock_gen


@pytest.fixture
def mock_openap_fuel_flow():
    """Mock OpenAP FuelFlow for testing."""
    mock_ff = MagicMock()
    mock_ff.enroute.return_value = 0.5  # kg/s
    return mock_ff


@pytest.fixture
def mock_openap_props():
    """Mock OpenAP aircraft properties for testing."""
    return {
        "limits": {"MTOW": 85000.0},  # kg
        "mtow": 85000.0
    }


@pytest.fixture
def mock_airports_iata(sample_airport_data):
    """Mock the airports IATA data loading."""
    with patch('aerospace_mcp.core._AIRPORTS_IATA', sample_airport_data):
        with patch('main._AIRPORTS_IATA', sample_airport_data):
            yield sample_airport_data


@pytest.fixture
def client():
    """FastAPI test client."""
    return TestClient(app)


@pytest.fixture(params=["with_openap", "without_openap"])
def openap_availability(request):
    """Parameterized fixture for testing with and without OpenAP."""
    if request.param == "with_openap":
        with patch('aerospace_mcp.core.OPENAP_AVAILABLE', True):
            with patch('main.OPENAP_AVAILABLE', True):
                yield True
    else:
        with patch('aerospace_mcp.core.OPENAP_AVAILABLE', False):
            with patch('main.OPENAP_AVAILABLE', False):
                yield False


# Test data for parametrized tests
@pytest.fixture
def airport_test_cases():
    """Test cases for airport resolution."""
    return [
        ("SJC", "SJC", True),  # Valid IATA
        ("sjc", "SJC", True),  # Case insensitive
        ("INVALID", None, False),  # Invalid IATA
        ("", None, False),  # Empty string
    ]


@pytest.fixture
def city_search_test_cases():
    """Test cases for city airport search."""
    return [
        ("San Jose", "US", ["SJC"]),  # City with country
        ("San Jose", None, ["SJC"]),  # City without country
        ("Tokyo", "JP", ["NRT"]),  # International city
        ("Nonexistent", "US", []),  # Non-existent city
        ("", "US", []),  # Empty city name
    ]


@pytest.fixture
def flight_plan_test_cases():
    """Test cases for flight planning."""
    return [
        {
            "name": "SJC to NRT",
            "request": {
                "depart_city": "San Jose",
                "arrive_city": "Tokyo",
                "depart_country": "US",
                "arrive_country": "JP",
                "ac_type": "A359"
            },
            "expected_distance_range": (8000, 12000),  # km
            "should_succeed": True
        },
        {
            "name": "Same city error",
            "request": {
                "depart_city": "San Jose",
                "arrive_city": "San Jose",
                "ac_type": "A320"
            },
            "should_succeed": False
        },
        {
            "name": "Invalid aircraft type",
            "request": {
                "depart_city": "San Jose",
                "arrive_city": "Tokyo",
                "ac_type": "INVALID_AC"
            },
            "should_succeed": False
        }
    ]


# Skip decorators for optional dependencies
skip_without_openap = pytest.mark.skipif(
    not OPENAP_AVAILABLE,
    reason="OpenAP not available"
)

skip_with_openap = pytest.mark.skipif(
    OPENAP_AVAILABLE,
    reason="OpenAP is available"
)


# Markers for test organization
pytest.mark.unit = pytest.mark.unit
pytest.mark.integration = pytest.mark.integration
pytest.mark.slow = pytest.mark.slow
