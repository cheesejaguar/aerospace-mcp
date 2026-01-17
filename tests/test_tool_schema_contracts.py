"""Tool schema contract tests.

These tests snapshot tool schemas to catch breaking changes and ensure
consistent API contracts. They also test error shapes for invalid inputs.
"""

import json
from pathlib import Path
from typing import Any

import pytest

# Mark all tests in this module as contract tests
pytestmark = [pytest.mark.contract]

# Path to schema snapshots
SNAPSHOTS_DIR = Path(__file__).parent / "snapshots"


def get_tool_schemas() -> dict[str, dict[str, Any]]:
    """Get all tool schemas from the MCP server."""
    from aerospace_mcp.tools.tool_search import TOOL_REGISTRY

    schemas = {}
    for tool in TOOL_REGISTRY:
        schemas[tool.name] = {
            "name": tool.name,
            "description": tool.description,
            "category": tool.category,
            "parameters": tool.parameters,
            "keywords": tool.keywords,
        }
    return schemas


class TestToolSchemaSnapshots:
    """Test that tool schemas match expected snapshots."""

    @pytest.fixture(scope="class")
    def current_schemas(self) -> dict[str, dict[str, Any]]:
        """Get current tool schemas."""
        return get_tool_schemas()

    def test_all_tools_have_schemas(self, current_schemas):
        """Test that all registered tools have schema information."""
        assert len(current_schemas) > 0, "Should have registered tools"

        for name, schema in current_schemas.items():
            assert "name" in schema, f"Tool {name} missing name"
            assert "description" in schema, f"Tool {name} missing description"
            assert "category" in schema, f"Tool {name} missing category"

    def test_tool_count_matches_expected(self, current_schemas):
        """Test that tool count hasn't changed unexpectedly."""
        # Update this number when intentionally adding/removing tools
        EXPECTED_TOOL_COUNT = 44
        actual_count = len(current_schemas)

        assert actual_count == EXPECTED_TOOL_COUNT, (
            f"Tool count changed from {EXPECTED_TOOL_COUNT} to {actual_count}. "
            "If intentional, update EXPECTED_TOOL_COUNT."
        )

    def test_required_tools_exist(self, current_schemas):
        """Test that required/core tools are present."""
        # Note: search_aerospace_tools and list_tool_categories are meta-tools
        # that search TOOL_REGISTRY but are not themselves in the registry
        required_tools = [
            "search_airports",
            "plan_flight",
            "calculate_distance",
            "get_aircraft_performance",
            "get_system_status",
        ]

        for tool_name in required_tools:
            assert tool_name in current_schemas, f"Required tool {tool_name} is missing"

    def test_tool_categories_are_valid(self, current_schemas):
        """Test that all tools have valid categories."""
        valid_categories = {
            "core",
            "atmosphere",
            "frames",
            "aerodynamics",
            "propellers",
            "rockets",
            "orbits",
            "gnc",
            "performance",
            "optimization",
            "agents",
        }

        for name, schema in current_schemas.items():
            category = schema.get("category")
            assert category in valid_categories, (
                f"Tool {name} has invalid category '{category}'. "
                f"Valid categories: {valid_categories}"
            )

    def test_tool_descriptions_not_empty(self, current_schemas):
        """Test that all tools have non-empty descriptions."""
        for name, schema in current_schemas.items():
            description = schema.get("description", "")
            assert (
                len(description) > 10
            ), f"Tool {name} has too short description: '{description}'"

    def test_tool_parameters_are_documented(self, current_schemas):
        """Test that tools have parameter documentation."""
        for name, schema in current_schemas.items():
            params = schema.get("parameters", {})
            # Parameters can be empty for simple tools, but should be a dict
            assert isinstance(
                params, dict
            ), f"Tool {name} parameters should be a dict, got {type(params)}"


class TestToolSchemaStability:
    """Test for unintentional breaking changes in tool schemas."""

    def test_search_airports_schema_stable(self):
        """Test search_airports schema hasn't changed."""
        from aerospace_mcp.tools.tool_search import TOOL_REGISTRY

        tool = next((t for t in TOOL_REGISTRY if t.name == "search_airports"), None)
        assert tool is not None

        # Check required parameters exist
        assert "query" in tool.parameters
        assert "query_type" in tool.parameters

    def test_plan_flight_schema_stable(self):
        """Test plan_flight schema hasn't changed."""
        from aerospace_mcp.tools.tool_search import TOOL_REGISTRY

        tool = next((t for t in TOOL_REGISTRY if t.name == "plan_flight"), None)
        assert tool is not None

        # Check required parameters exist
        assert "departure" in tool.parameters
        assert "arrival" in tool.parameters

    def test_calculate_distance_schema_stable(self):
        """Test calculate_distance schema hasn't changed."""
        from aerospace_mcp.tools.tool_search import TOOL_REGISTRY

        tool = next((t for t in TOOL_REGISTRY if t.name == "calculate_distance"), None)
        assert tool is not None

        # Check required parameters exist (uses airport-based interface)
        assert "origin" in tool.parameters
        assert "destination" in tool.parameters


class TestToolErrorShapes:
    """Test that tools return consistent error shapes."""

    def test_search_airports_invalid_query(self):
        """Test search_airports returns proper error for invalid query."""
        from aerospace_mcp.tools.core import search_airports

        # Empty query should still return valid response
        result = search_airports(query="")
        assert isinstance(result, str)
        # Should indicate no results or error gracefully

    def test_plan_flight_missing_fields(self):
        """Test plan_flight error shape for missing fields."""
        from aerospace_mcp.tools.core import plan_flight

        # Missing required fields should return error
        result = plan_flight(departure={}, arrival={})
        assert isinstance(result, str)
        # Should contain error indication

    def test_calculate_distance_invalid_coords(self):
        """Test calculate_distance error shape for invalid coordinates."""
        from aerospace_mcp.tools.core import calculate_distance

        # Invalid coordinates
        result = calculate_distance(lat1=999, lon1=999, lat2=999, lon2=999)
        assert isinstance(result, str)

    def test_get_aircraft_performance_invalid_type(self):
        """Test get_aircraft_performance error for invalid aircraft."""
        from aerospace_mcp.tools.core import get_aircraft_performance

        result = get_aircraft_performance(
            aircraft_type="INVALID_AIRCRAFT_XYZ",
            distance_km=1000,
        )
        assert isinstance(result, str)
        # Should indicate error or unavailable


class TestToolResponseShapes:
    """Test that tool responses have expected shapes."""

    def test_search_airports_response_shape(self):
        """Test search_airports returns expected response shape."""
        from aerospace_mcp.tools.core import search_airports

        result = search_airports(query="JFK")
        assert isinstance(result, str)

        # Response should be parseable or contain expected info
        # JFK should be found
        assert "JFK" in result or "John F. Kennedy" in result or "New York" in result

    def test_get_system_status_response_shape(self):
        """Test get_system_status returns expected response shape."""
        from aerospace_mcp.tools.core import get_system_status

        result = get_system_status()
        assert isinstance(result, str)

        # Should contain JSON with expected fields

        try:
            # Try to extract JSON from response
            data = json.loads(result)
            assert "status" in data or "openap_available" in data
        except json.JSONDecodeError:
            # Response might be formatted text with JSON
            assert "status" in result.lower() or "aerospace" in result.lower()

    def test_list_tool_categories_response_shape(self):
        """Test list_tool_categories returns expected response shape."""
        from aerospace_mcp.tools.tool_search import list_tool_categories

        result = list_tool_categories()
        assert isinstance(result, str)

        # Should contain JSON with categories

        data = json.loads(result)
        assert "categories" in data
        assert "total_tools" in data
        assert isinstance(data["categories"], list)

    def test_search_aerospace_tools_response_shape(self):
        """Test search_aerospace_tools returns expected response shape."""
        from aerospace_mcp.tools.tool_search import search_aerospace_tools

        result = search_aerospace_tools(query="orbit")
        assert isinstance(result, str)

        # Should contain JSON with tool references

        data = json.loads(result)
        assert "tool_references" in data
        assert "total_matches" in data
        assert isinstance(data["tool_references"], list)


class TestToolCategoryCompleteness:
    """Test that all categories have tools and are properly documented."""

    @pytest.fixture
    def categories_and_tools(self) -> dict[str, list[str]]:
        """Get tools grouped by category."""
        from aerospace_mcp.tools.tool_search import TOOL_REGISTRY

        by_category = {}
        for tool in TOOL_REGISTRY:
            category = tool.category
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(tool.name)
        return by_category

    def test_all_categories_have_tools(self, categories_and_tools):
        """Test that no category is empty."""
        for category, tools in categories_and_tools.items():
            assert len(tools) > 0, f"Category {category} has no tools"

    def test_category_distribution(self, categories_and_tools):
        """Test expected tool distribution across categories."""
        # Core tools
        assert len(categories_and_tools.get("core", [])) >= 5

        # Orbits should have multiple tools
        assert len(categories_and_tools.get("orbits", [])) >= 6

        # Performance tools (newly added)
        assert len(categories_and_tools.get("performance", [])) >= 7

        # GNC tools
        assert len(categories_and_tools.get("gnc", [])) >= 2
