"""Tests for FastMCP server functionality."""

import json

import pytest

import aerospace_mcp.tools.core as core_tools
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


class TestSearchAirportsTool:
    """Tests for the search_airports tool (real airport database)."""

    @pytest.mark.unit
    def test_search_airports_by_iata(self):
        """Test searching airports by IATA code."""
        result = search_airports("SJC", query_type="iata")

        assert result.startswith("Found 1 airport(s):")
        assert "SJC (KSJC) - Norman Y Mineta San Jose International Airport" in result
        assert "City: San Jose, US" in result
        assert "Coordinates: 37.3630, -121.9286" in result
        assert "Timezone: America/Los_Angeles" in result

    @pytest.mark.unit
    def test_search_airports_auto_detects_iata(self):
        """A three-letter query is treated as an IATA code in auto mode."""
        assert search_airports("nrt") == search_airports("NRT", query_type="iata")
        assert "NRT (RJAA) - Narita International Airport" in search_airports("nrt")

    @pytest.mark.unit
    def test_search_airports_by_city(self):
        """Test searching airports by city name with a country filter."""
        result = search_airports("San Jose", country="US", query_type="city")

        assert result.startswith("Found 2 airport(s):")
        # Best match first
        assert result.index("SJC (KSJC)") < result.index("RHV")
        assert "SJO" not in result  # Costa Rica filtered out

    @pytest.mark.unit
    def test_search_airports_empty_query(self):
        """Test search with empty query."""
        assert search_airports("") == "Error: Query parameter is required"
        assert search_airports("   ") == "Error: Query parameter is required"

    @pytest.mark.unit
    def test_search_airports_not_found(self):
        """Test search with non-existent airport."""
        assert search_airports("XXX", query_type="iata") == (
            "No airports found for iata 'XXX'"
        )
        assert search_airports("Atlantis", country="GR") == (
            "No airports found for city 'Atlantis' in country 'GR'"
        )


class TestPlanFlightTool:
    """Tests for the plan_flight tool."""

    @pytest.mark.unit
    def test_plan_flight_basic(self):
        """Test basic flight planning."""
        result = json.loads(
            plan_flight({"city": "San Jose", "country": "US"}, {"iata": "NRT"})
        )

        assert result["departure"]["airport"]["iata"] == "SJC"
        assert result["arrival"]["airport"]["name"] == "Narita International Airport"
        assert 8200 < result["route"]["distance_km"] < 8400
        assert result["route"]["distance_nm"] == pytest.approx(
            result["route"]["distance_km"] / 1.852
        )
        # Default aircraft is the A320, estimated by OpenAP
        assert result["engine"] == "openap"
        assert result["performance"]["block"]["time_min"] > 0


class TestCalculateDistanceTool:
    """Tests for the calculate_distance tool."""

    @pytest.mark.unit
    def test_calculate_distance(self):
        """WGS-84 geodesic distance and bearings between two points."""
        result = json.loads(calculate_distance(37.0, -122.0, 38.0, -121.0))

        # 1 deg of latitude and longitude at 37.5 N: ~141.9 km, heading NE
        assert result["distance_km"] == pytest.approx(141.9, abs=0.1)
        assert result["distance_nm"] == pytest.approx(
            result["distance_km"] / 1.852, rel=1e-6
        )
        assert 35 < result["initial_bearing_deg"] < 42
        assert result["coordinates"]["end"] == {"lat": 38.0, "lon": -121.0}

    @pytest.mark.unit
    def test_calculate_distance_same_point(self):
        result = json.loads(calculate_distance(10.0, 20.0, 10.0, 20.0))
        assert result["distance_km"] == 0.0


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
    def test_aircraft_performance_no_openap(self, monkeypatch):
        """Test aircraft performance when OpenAP is not installed."""
        monkeypatch.setattr(core_tools, "OPENAP_AVAILABLE", False)
        result = get_aircraft_performance("A320", 1000.0)

        assert "OpenAP library is not available" in result

    @pytest.mark.unit
    def test_aircraft_performance_with_openap(self):
        """Real OpenAP estimate for a 1,000 km A320 sector."""
        result = json.loads(get_aircraft_performance("A320", 1000.0))

        assert result["engine"] == "openap"
        performance = result["performance"]
        seg_km = sum(
            performance[k]["distance_km"] for k in ("climb", "cruise", "descent")
        )
        assert seg_km == pytest.approx(1000.0)
        # ~1.5 h and a few tonnes of fuel for a 1,000 km A320 sector
        assert 70 < performance["block"]["time_min"] < 110
        assert 2000 < performance["block"]["fuel_kg"] < 6000
        assert performance["assumptions"]["mass_source"] == "mtow_85pct"

    @pytest.mark.unit
    def test_aircraft_performance_unsupported_type(self):
        result = get_aircraft_performance("C172", 300.0)
        assert result.startswith("Performance estimation error:")
        assert "not supported by OpenAP" in result
