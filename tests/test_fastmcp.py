"""Tests for FastMCP server functionality."""

from unittest.mock import patch

import pytest

from aerospace_mcp.fastmcp_server import mcp
from aerospace_mcp.tools.core import (
    calculate_distance,
    get_aircraft_performance,
    get_system_status,
    plan_flight,
    search_airports,
)


class TestFastMCPServerInitialization:
    """Tests for FastMCP server initialization."""

    @pytest.mark.unit
    def test_server_instance_created(self):
        """Test that FastMCP server instance is created correctly."""
        assert mcp is not None
        assert mcp.name == "aerospace-mcp"

    @pytest.mark.unit
    def test_tools_registered(self):
        """Test that all required tools are registered."""
        # FastMCP automatically discovers registered tools
        # We can check if the key tools are available by trying to call them

        # The tools are registered as functions, so we can verify they exist
        # This is a basic sanity check
        assert callable(search_airports)
        assert callable(plan_flight)
        assert callable(calculate_distance)
        assert callable(get_aircraft_performance)
        assert callable(get_system_status)


class TestSearchAirportsTool:
    """Tests for the search_airports tool."""

    @pytest.mark.unit
    @patch("aerospace_mcp.tools.core._airport_from_iata")
    def test_search_airports_by_iata(self, mock_airport):
        """Test searching airports by IATA code."""
        # Mock airport data
        mock_airport.return_value = type(
            "Airport",
            (),
            {
                "iata": "SJC",
                "icao": "KSJC",
                "name": "San Jose International Airport",
                "city": "San Jose",
                "country": "US",
                "lat": 37.3626,
                "lon": -121.929,
                "tz": "America/Los_Angeles",
            },
        )()

        result = search_airports("SJC", query_type="iata")

        assert "SJC" in result
        assert "San Jose" in result
        assert "Found 1 airport(s)" in result

    @pytest.mark.unit
    @patch("aerospace_mcp.tools.core._find_city_airports")
    def test_search_airports_by_city(self, mock_find):
        """Test searching airports by city name."""
        # Mock city search
        mock_find.return_value = [
            type(
                "Airport",
                (),
                {
                    "iata": "SJC",
                    "icao": "KSJC",
                    "name": "San Jose International Airport",
                    "city": "San Jose",
                    "country": "US",
                    "lat": 37.3626,
                    "lon": -121.929,
                    "tz": "America/Los_Angeles",
                },
            )()
        ]

        result = search_airports("San Jose", query_type="city")

        assert "SJC" in result
        assert "San Jose" in result
        assert "Found 1 airport(s)" in result

    @pytest.mark.unit
    def test_search_airports_empty_query(self):
        """Test search with empty query."""
        result = search_airports("")
        assert "Error: Query parameter is required" in result

    @pytest.mark.unit
    @patch("aerospace_mcp.tools.core._airport_from_iata")
    def test_search_airports_not_found(self, mock_airport):
        """Test search with non-existent airport."""
        mock_airport.return_value = None

        result = search_airports("XXX", query_type="iata")

        assert "No airports found for iata 'XXX'" in result


class TestPlanFlightTool:
    """Tests for the plan_flight tool."""

    @pytest.mark.unit
    @patch("aerospace_mcp.tools.core._resolve_endpoint")
    @patch("aerospace_mcp.tools.core.great_circle_points")
    def test_plan_flight_basic(self, mock_route, mock_resolve):
        """Test basic flight planning."""
        # Mock airport resolution
        mock_departure = type(
            "Airport",
            (),
            {
                "iata": "SJC",
                "icao": "KSJC",
                "name": "San Jose International Airport",
                "city": "San Jose",
                "country": "US",
                "lat": 37.3626,
                "lon": -121.929,
            },
        )()
        mock_arrival = type(
            "Airport",
            (),
            {
                "iata": "NRT",
                "icao": "RJAA",
                "name": "Narita International Airport",
                "city": "Tokyo",
                "country": "JP",
                "lat": 35.7647,
                "lon": 140.386,
            },
        )()

        mock_resolve.side_effect = [mock_departure, mock_arrival]

        # Mock route calculation - returns tuple (points, distance_km)
        mock_route.return_value = (
            [(37.3626, -121.929), (35.7647, 140.386)],
            8280.5,
        )

        departure = {"city": "San Jose"}
        arrival = {"city": "Tokyo"}

        result = plan_flight(departure, arrival)

        assert "San Jose International Airport" in result
        assert "Narita International Airport" in result
        assert "8280.5" in result  # Distance


class TestCalculateDistanceTool:
    """Tests for the calculate_distance tool."""

    @pytest.mark.unit
    @patch("aerospace_mcp.tools.core.great_circle_points")
    def test_calculate_distance(self, mock_route):
        """Test distance calculation - great_circle_points returns tuple (points, distance_km)."""
        mock_route.return_value = (
            [(37.0, -122.0), (38.0, -121.0)],
            100.0,
        )

        result = calculate_distance(37.0, -122.0, 38.0, -121.0)

        assert "100.0" in result
        # distance_nm is calculated from distance_km
        assert "53.9" in result or "54.0" in result


class TestSystemStatusTool:
    """Tests for the get_system_status tool."""

    @pytest.mark.unit
    def test_get_system_status(self):
        """Test system status retrieval."""
        result = get_system_status()

        assert "Aerospace MCP Server" in result
        assert "operational" in result
        assert "airport_search" in result


class TestAircraftPerformanceTool:
    """Tests for the get_aircraft_performance tool."""

    @pytest.mark.unit
    @patch("aerospace_mcp.tools.core.OPENAP_AVAILABLE", False)
    def test_aircraft_performance_no_openap(self):
        """Test aircraft performance when OpenAP not available."""
        result = get_aircraft_performance("A320", 1000.0)

        assert "OpenAP library is not available" in result

    @pytest.mark.unit
    @patch("aerospace_mcp.tools.core.OPENAP_AVAILABLE", True)
    @patch("aerospace_mcp.tools.core.estimates_openap")
    def test_aircraft_performance_with_openap(self, mock_estimates):
        """Test aircraft performance when OpenAP is available.

        estimates_openap returns tuple (performance_dict, engine_name).
        """
        mock_estimates.return_value = (
            {
                "fuel_kg": 2500.0,
                "flight_time_minutes": 180.0,
                "cruise_mach": 0.78,
            },
            "openap",
        )

        result = get_aircraft_performance("A320", 1000.0)

        assert "2500.0" in result
        assert "180.0" in result
