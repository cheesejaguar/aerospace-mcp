"""Tests for MCP server functionality."""

from unittest.mock import patch

import pytest
from mcp.types import TextContent

from aerospace_mcp.server import (
    TOOLS,
    _handle_calculate_distance,
    _handle_get_aircraft_performance,
    _handle_get_system_status,
    _handle_plan_flight,
    _handle_search_airports,
    handle_call_tool,
    handle_list_tools,
    server,
)


class TestMCPServerInitialization:
    """Tests for MCP server initialization."""

    @pytest.mark.unit
    def test_server_instance_created(self):
        """Test that server instance is created correctly."""
        assert server is not None
        assert server.name == "aerospace-mcp"

    @pytest.mark.unit
    def test_tools_definition(self):
        """Test that all required tools are defined."""
        tool_names = [tool.name for tool in TOOLS]
        expected_tools = [
            "search_airports",
            "plan_flight",
            "calculate_distance",
            "get_aircraft_performance",
            "get_system_status",
        ]

        for expected_tool in expected_tools:
            assert expected_tool in tool_names

# Updated for Phase 4 - now we have many more tools, just check required ones exist
        assert len(TOOLS) >= len(expected_tools)

    @pytest.mark.unit
    def test_tool_schemas_valid(self):
        """Test that all tool schemas are valid."""
        for tool in TOOLS:
            assert isinstance(tool.name, str)
            assert len(tool.name) > 0
            assert isinstance(tool.description, str)
            assert len(tool.description) > 0
            assert isinstance(tool.inputSchema, dict)
            assert "type" in tool.inputSchema
            assert tool.inputSchema["type"] == "object"


class TestListTools:
    """Tests for the list tools handler."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_handle_list_tools(self):
        """Test listing all available tools."""
        tools = await handle_list_tools()

        assert isinstance(tools, list)
        assert len(tools) == len(TOOLS)

        tool_names = [tool.name for tool in tools]
        assert "search_airports" in tool_names
        assert "plan_flight" in tool_names
        assert "calculate_distance" in tool_names
        assert "get_aircraft_performance" in tool_names
        assert "get_system_status" in tool_names


class TestSearchAirportsTool:
    """Tests for the search_airports tool."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_search_airports_by_iata(self, mock_airports_iata):
        """Test searching airports by IATA code."""
        with patch("aerospace_mcp.server._airport_from_iata") as mock_airport:
            mock_airport.return_value = mock_airports_iata["SJC"]

            result = await _handle_search_airports(
                {"query": "SJC", "query_type": "iata"}
            )

            assert len(result) == 1
            assert isinstance(result[0], TextContent)
            text_content = result[0].text
            assert "SJC" in text_content
            assert "San Jose" in text_content

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_search_airports_by_city(self, mock_airports_iata, sjc_airport):
        """Test searching airports by city name."""
        with patch("aerospace_mcp.server._find_city_airports") as mock_find:
            mock_find.return_value = [sjc_airport]

            result = await _handle_search_airports(
                {"query": "San Jose", "query_type": "city", "country": "US"}
            )

            assert len(result) == 1
            assert isinstance(result[0], TextContent)
            text_content = result[0].text
            assert "Found 1 airport(s)" in text_content
            assert "SJC" in text_content

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_search_airports_auto_detect_iata(self, mock_airports_iata):
        """Test auto-detection of IATA codes."""
        with patch("aerospace_mcp.server._airport_from_iata") as mock_airport:
            mock_airport.return_value = mock_airports_iata["SJC"]

            await _handle_search_airports({"query": "SJC", "query_type": "auto"})

            mock_airport.assert_called_once_with("SJC")

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_search_airports_auto_detect_city(self, sjc_airport):
        """Test auto-detection of city names."""
        with patch("aerospace_mcp.server._find_city_airports") as mock_find:
            mock_find.return_value = [sjc_airport]

            await _handle_search_airports({"query": "San Jose", "query_type": "auto"})

            mock_find.assert_called_once_with("San Jose", None)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_search_airports_not_found(self):
        """Test searching for non-existent airports."""
        with patch("aerospace_mcp.server._airport_from_iata", return_value=None):
            result = await _handle_search_airports(
                {"query": "XYZ", "query_type": "iata"}
            )

            assert len(result) == 1
            assert "No airports found" in result[0].text

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_search_airports_empty_query(self):
        """Test searching with empty query."""
        result = await _handle_search_airports({"query": ""})

        assert len(result) == 1
        assert "Error: Query parameter is required" in result[0].text


class TestPlanFlightTool:
    """Tests for the plan_flight tool."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_plan_flight_success(
        self,
        mock_airports_iata,
        sjc_airport,
        nrt_airport,
        mock_openap_flight_generator,
        mock_openap_fuel_flow,
        mock_openap_props,
    ):
        """Test successful flight planning."""
        with patch("aerospace_mcp.server._resolve_endpoint") as mock_resolve:
            mock_resolve.side_effect = [sjc_airport, nrt_airport]

            with patch("aerospace_mcp.server.great_circle_points") as mock_gc:
                mock_gc.return_value = ([(37.36, -121.93), (35.76, 140.39)], 9000.0)

                with patch("aerospace_mcp.server.estimates_openap") as mock_estimates:
                    mock_estimates.return_value = (
                        {
                            "block": {"time_min": 600.0, "fuel_kg": 15000.0},
                            "climb": {
                                "time_min": 20.0,
                                "distance_km": 100.0,
                                "fuel_kg": 1000.0,
                            },
                            "cruise": {
                                "time_min": 560.0,
                                "distance_km": 8800.0,
                                "fuel_kg": 13500.0,
                            },
                            "descent": {
                                "time_min": 20.0,
                                "distance_km": 100.0,
                                "fuel_kg": 500.0,
                            },
                            "assumptions": {"mass_kg": 85000.0, "cruise_alt_ft": 35000},
                        },
                        "openap",
                    )

                    result = await _handle_plan_flight(
                        {
                            "departure": {"city": "San Jose", "country": "US"},
                            "arrival": {"city": "Tokyo", "country": "JP"},
                            "aircraft": {"type": "A359"},
                        }
                    )

                    assert len(result) == 1
                    text = result[0].text
                    assert "Flight Plan: SJC → NRT" in text
                    assert "Distance: 9000 km" in text
                    assert "Block Time: 600 minutes" in text
                    assert "Block Fuel: 15000 kg" in text

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_plan_flight_same_city_error(self):
        """Test error when departure and arrival are the same city."""
        result = await _handle_plan_flight(
            {
                "departure": {"city": "San Jose"},
                "arrival": {"city": "San Jose"},
                "aircraft": {"type": "A320"},
            }
        )

        assert len(result) == 1
        assert "identical" in result[0].text.lower()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_plan_flight_airport_resolution_error(self):
        """Test error during airport resolution."""
        with patch("aerospace_mcp.server._resolve_endpoint") as mock_resolve:
            from aerospace_mcp.core import AirportResolutionError

            mock_resolve.side_effect = AirportResolutionError("Airport not found")

            result = await _handle_plan_flight(
                {
                    "departure": {"city": "Nonexistent"},
                    "arrival": {"city": "Tokyo"},
                    "aircraft": {"type": "A320"},
                }
            )

            assert len(result) == 1
            assert "Airport resolution error" in result[0].text

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_plan_flight_openap_error(self, sjc_airport, nrt_airport):
        """Test error during OpenAP estimation."""
        with patch("aerospace_mcp.server._resolve_endpoint") as mock_resolve:
            mock_resolve.side_effect = [sjc_airport, nrt_airport]

            with patch("aerospace_mcp.server.great_circle_points") as mock_gc:
                mock_gc.return_value = ([(37.36, -121.93)], 9000.0)

                with patch("aerospace_mcp.server.estimates_openap") as mock_estimates:
                    from aerospace_mcp.core import OpenAPError

                    mock_estimates.side_effect = OpenAPError("OpenAP unavailable")

                    result = await _handle_plan_flight(
                        {
                            "departure": {"city": "San Jose"},
                            "arrival": {"city": "Tokyo"},
                            "aircraft": {"type": "A320"},
                        }
                    )

                    assert len(result) == 1
                    assert "Performance estimation error" in result[0].text

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_plan_flight_with_preferred_iata(self, sjc_airport, nrt_airport):
        """Test flight planning with preferred IATA codes."""
        with patch("aerospace_mcp.server._resolve_endpoint") as mock_resolve:
            mock_resolve.side_effect = [sjc_airport, nrt_airport]

            with patch("aerospace_mcp.server.great_circle_points") as mock_gc:
                mock_gc.return_value = ([(37.36, -121.93)], 1000.0)

                with patch("aerospace_mcp.server.estimates_openap") as mock_estimates:
                    mock_estimates.return_value = (
                        {
                            "block": {"time_min": 120.0, "fuel_kg": 3000.0},
                            "climb": {
                                "time_min": 10.0,
                                "distance_km": 50.0,
                                "fuel_kg": 200.0,
                            },
                            "cruise": {
                                "time_min": 100.0,
                                "distance_km": 900.0,
                                "fuel_kg": 2600.0,
                            },
                            "descent": {
                                "time_min": 10.0,
                                "distance_km": 50.0,
                                "fuel_kg": 200.0,
                            },
                            "assumptions": {"mass_kg": 70000.0, "cruise_alt_ft": 35000},
                        },
                        "openap",
                    )

                    await _handle_plan_flight(
                        {
                            "departure": {"city": "Any City", "iata": "SJC"},
                            "arrival": {"city": "Any City", "iata": "NRT"},
                            "aircraft": {"type": "A320"},
                        }
                    )

                    # Should call resolve_endpoint with the preferred IATA codes
                    mock_resolve.assert_any_call("Any City", None, "SJC", "departure")
                    mock_resolve.assert_any_call("Any City", None, "NRT", "arrival")


class TestCalculateDistanceTool:
    """Tests for the calculate_distance tool."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_calculate_distance_success(self):
        """Test successful distance calculation."""
        with patch("aerospace_mcp.server.great_circle_points") as mock_gc:
            mock_gc.return_value = ([(37.36, -121.93), (35.76, 140.39)], 9000.0)

            result = await _handle_calculate_distance(
                {
                    "origin": {"latitude": 37.3626, "longitude": -121.929},
                    "destination": {"latitude": 35.7647, "longitude": 140.386},
                    "step_km": 100.0,
                }
            )

            assert len(result) == 1
            text = result[0].text
            assert "Great Circle Distance Calculation" in text
            assert "9000.00 km" in text
            assert "Origin: 37.3626, -121.9290" in text
            assert "Destination: 35.7647, 140.3860" in text

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_calculate_distance_missing_coordinates(self):
        """Test error with missing coordinates."""
        result = await _handle_calculate_distance(
            {
                "origin": {"latitude": 37.3626},  # Missing longitude
                "destination": {"latitude": 35.7647, "longitude": 140.386},
            }
        )

        assert len(result) == 1
        assert "coordinates are required" in result[0].text

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_calculate_distance_with_default_step(self):
        """Test distance calculation with default step size."""
        with patch("aerospace_mcp.server.great_circle_points") as mock_gc:
            mock_gc.return_value = ([(37.36, -121.93)], 500.0)

            await _handle_calculate_distance(
                {
                    "origin": {"latitude": 37.3626, "longitude": -121.929},
                    "destination": {"latitude": 35.7647, "longitude": 140.386},
                }
            )

            # Should use default step_km of 50.0
            mock_gc.assert_called_once_with(37.3626, -121.929, 35.7647, 140.386, 50.0)


class TestGetAircraftPerformanceTool:
    """Tests for the get_aircraft_performance tool."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_aircraft_performance_success(self):
        """Test successful aircraft performance estimation."""
        with patch("aerospace_mcp.server.estimates_openap") as mock_estimates:
            mock_estimates.return_value = (
                {
                    "block": {"time_min": 180.0, "fuel_kg": 4500.0},
                    "climb": {"time_min": 15.0, "distance_km": 75.0, "fuel_kg": 500.0},
                    "cruise": {
                        "time_min": 150.0,
                        "distance_km": 1350.0,
                        "fuel_kg": 3500.0,
                    },
                    "descent": {
                        "time_min": 15.0,
                        "distance_km": 75.0,
                        "fuel_kg": 500.0,
                    },
                    "assumptions": {"mass_kg": 70000.0, "cruise_alt_ft": 35000},
                },
                "openap",
            )

            result = await _handle_get_aircraft_performance(
                {
                    "aircraft_type": "A320",
                    "distance_km": 1500.0,
                    "cruise_altitude": 35000,
                }
            )

            assert len(result) == 1
            text = result[0].text
            assert "Aircraft Performance Estimates (openap)" in text
            assert "Aircraft: A320" in text
            assert "Distance: 1500 km" in text
            assert "Time: 180 minutes" in text
            assert "Fuel: 4500 kg" in text

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_aircraft_performance_missing_aircraft(self):
        """Test error with missing aircraft type."""
        result = await _handle_get_aircraft_performance({"distance_km": 1500.0})

        assert len(result) == 1
        assert "Aircraft type is required" in result[0].text

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_aircraft_performance_invalid_distance(self):
        """Test error with invalid distance."""
        result = await _handle_get_aircraft_performance(
            {"aircraft_type": "A320", "distance_km": -100.0}
        )

        assert len(result) == 1
        assert "Distance must be positive" in result[0].text

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_aircraft_performance_openap_error(self):
        """Test error during OpenAP estimation."""
        with patch("aerospace_mcp.server.estimates_openap") as mock_estimates:
            from aerospace_mcp.core import OpenAPError

            mock_estimates.side_effect = OpenAPError("OpenAP not available")

            result = await _handle_get_aircraft_performance(
                {"aircraft_type": "A320", "distance_km": 1500.0}
            )

            assert len(result) == 1
            assert "Performance estimation error" in result[0].text


class TestGetSystemStatusTool:
    """Tests for the get_system_status tool."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_system_status_with_openap(self):
        """Test system status when OpenAP is available."""
        with patch("aerospace_mcp.server.OPENAP_AVAILABLE", True):
            with patch("aerospace_mcp.core._AIRPORTS_IATA", {"SJC": {}, "NRT": {}}):
                result = await _handle_get_system_status({})

                assert len(result) == 1
                text = result[0].text
                assert "Aerospace MCP Server Status" in text
                assert "OpenAP Available: Yes" in text
                assert "Airports Loaded: 2" in text
                assert "search_airports" in text
                assert "plan_flight" in text

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_system_status_without_openap(self):
        """Test system status when OpenAP is not available."""
        with patch("aerospace_mcp.server.OPENAP_AVAILABLE", False):
            with patch("aerospace_mcp.core._AIRPORTS_IATA", {"SJC": {}}):
                result = await _handle_get_system_status({})

                assert len(result) == 1
                text = result[0].text
                assert "OpenAP Available: No" in text
                assert "pip install openap" in text


class TestHandleCallTool:
    """Tests for the main tool call handler."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_handle_call_tool_search_airports(self, sjc_airport):
        """Test calling search_airports tool through main handler."""
        with patch("aerospace_mcp.server._handle_search_airports") as mock_handler:
            mock_handler.return_value = [TextContent(type="text", text="Test result")]

            result = await handle_call_tool("search_airports", {"query": "SJC"})

            mock_handler.assert_called_once_with({"query": "SJC"})
            assert len(result) == 1
            assert result[0].text == "Test result"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_handle_call_tool_unknown_tool(self):
        """Test calling an unknown tool."""
        result = await handle_call_tool("unknown_tool", {})

        assert len(result) == 1
        assert "Unknown tool: unknown_tool" in result[0].text

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_handle_call_tool_exception_handling(self):
        """Test exception handling in tool calls."""
        with patch("aerospace_mcp.server._handle_search_airports") as mock_handler:
            mock_handler.side_effect = Exception("Test exception")

            result = await handle_call_tool("search_airports", {"query": "SJC"})

            assert len(result) == 1
            assert "Error: Test exception" in result[0].text

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_all_tools_callable_through_handler(self):
        """Test that all defined tools are callable through the main handler."""
        tool_names = [tool.name for tool in TOOLS]

        for tool_name in tool_names:
            # Mock the specific handler for each tool
            handler_name = f"_handle_{tool_name}"
            with patch(f"aerospace_mcp.server.{handler_name}") as mock_handler:
                mock_handler.return_value = [
                    TextContent(type="text", text="Mock result")
                ]

                result = await handle_call_tool(tool_name, {})

                assert len(result) == 1
                assert result[0].text == "Mock result"
                mock_handler.assert_called_once()


class TestMCPServerIntegration:
    """Integration tests for MCP server functionality."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_full_flight_planning_flow(
        self, mock_airports_iata, sjc_airport, nrt_airport
    ):
        """Test full flight planning workflow through MCP interface."""
        # First, search for airports
        with patch("aerospace_mcp.server._find_city_airports") as mock_find:
            mock_find.return_value = [sjc_airport]

            search_result = await handle_call_tool(
                "search_airports", {"query": "San Jose", "country": "US"}
            )

            assert len(search_result) == 1
            assert "SJC" in search_result[0].text

        # Then, plan a flight
        with patch("aerospace_mcp.server._resolve_endpoint") as mock_resolve:
            mock_resolve.side_effect = [sjc_airport, nrt_airport]

            with patch("aerospace_mcp.server.great_circle_points") as mock_gc:
                mock_gc.return_value = ([(37.36, -121.93), (35.76, 140.39)], 9000.0)

                with patch("aerospace_mcp.server.estimates_openap") as mock_estimates:
                    mock_estimates.return_value = (
                        {
                            "block": {"time_min": 600.0, "fuel_kg": 15000.0},
                            "climb": {
                                "time_min": 20.0,
                                "distance_km": 100.0,
                                "fuel_kg": 1000.0,
                            },
                            "cruise": {
                                "time_min": 560.0,
                                "distance_km": 8800.0,
                                "fuel_kg": 13500.0,
                            },
                            "descent": {
                                "time_min": 20.0,
                                "distance_km": 100.0,
                                "fuel_kg": 500.0,
                            },
                            "assumptions": {"mass_kg": 85000.0, "cruise_alt_ft": 35000},
                        },
                        "openap",
                    )

                    plan_result = await handle_call_tool(
                        "plan_flight",
                        {
                            "departure": {"city": "San Jose", "country": "US"},
                            "arrival": {"city": "Tokyo", "country": "JP"},
                            "aircraft": {"type": "A359"},
                        },
                    )

                    assert len(plan_result) == 1
                    text = plan_result[0].text
                    assert "Flight Plan: SJC → NRT" in text
                    assert "Block Time: 600 minutes" in text

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_error_propagation_through_mcp(self):
        """Test that errors propagate correctly through MCP interface."""
        # Test airport resolution error
        with patch("aerospace_mcp.server._resolve_endpoint") as mock_resolve:
            from aerospace_mcp.core import AirportResolutionError

            mock_resolve.side_effect = AirportResolutionError("Airport not found")

            result = await handle_call_tool(
                "plan_flight",
                {
                    "departure": {"city": "InvalidCity"},
                    "arrival": {"city": "Tokyo"},
                    "aircraft": {"type": "A320"},
                },
            )

            assert len(result) == 1
            assert "Airport resolution error" in result[0].text

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_system_status_integration(self):
        """Test system status provides accurate information."""
        result = await handle_call_tool("get_system_status", {})

        assert len(result) == 1
        text = result[0].text
        assert "Aerospace MCP Server Status" in text
        assert "Available Tools:" in text

        # Check that all tools are listed
        for tool in TOOLS:
            assert tool.name in text


class TestMCPServerTransports:
    """Tests for MCP server transport mechanisms."""

    @pytest.mark.unit
    def test_stdio_transport_function_exists(self):
        """Test that stdio transport function exists."""
        from aerospace_mcp.server import run_stdio

        assert callable(run_stdio)

    @pytest.mark.unit
    def test_sse_transport_function_exists(self):
        """Test that SSE transport function exists."""
        from aerospace_mcp.server import run_sse

        assert callable(run_sse)

    @pytest.mark.unit
    def test_main_run_function_exists(self):
        """Test that main run function exists."""
        from aerospace_mcp.server import run

        assert callable(run)

    @pytest.mark.unit
    def test_run_function_argument_parsing(self):
        """Test run function argument parsing logic."""
        import sys
        from unittest.mock import patch

        from aerospace_mcp.server import run

        # Test stdio mode (default)
        with patch("aerospace_mcp.server.run_stdio") as mock_stdio:
            with patch.object(sys, "argv", ["server.py"]):
                try:
                    run()
                except Exception:
                    pass  # run_stdio will fail in test, but we just want to check it's called
                mock_stdio.assert_called_once()

        # Test SSE mode
        with patch("aerospace_mcp.server.run_sse") as mock_sse:
            with patch.object(sys, "argv", ["server.py", "sse"]):
                try:
                    run()
                except Exception:
                    pass  # run_sse will fail in test, but we just want to check it's called
                mock_sse.assert_called_once_with("localhost", 8001)
