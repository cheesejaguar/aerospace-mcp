"""Tests for the tool search functionality."""

import json

import pytest

from aerospace_mcp.tools.tool_search import (
    CATEGORIES,
    MAX_PATTERN_LENGTH,
    TOOL_REGISTRY,
    ToolMetadata,
    list_tool_categories,
    search_aerospace_tools,
    search_tools_regex,
    search_tools_text,
)


class TestToolRegistry:
    """Test suite for the tool registry."""

    def test_registry_not_empty(self):
        """Test that the tool registry has tools."""
        assert len(TOOL_REGISTRY) > 0

    def test_registry_has_expected_count(self):
        """Test that the registry has expected number of tools."""
        # Should have 34+ tools
        assert len(TOOL_REGISTRY) >= 34

    def test_tool_metadata_structure(self):
        """Test that each tool has required metadata fields."""
        for tool in TOOL_REGISTRY:
            assert isinstance(tool, ToolMetadata)
            assert tool.name
            assert tool.description
            assert tool.category
            assert isinstance(tool.parameters, dict)
            assert isinstance(tool.keywords, list)

    def test_categories_defined(self):
        """Test that categories are properly defined."""
        expected_categories = [
            "core",
            "atmosphere",
            "frames",
            "aerodynamics",
            "propellers",
            "rockets",
            "orbits",
            "optimization",
            "agents",
        ]
        for cat in expected_categories:
            assert cat in CATEGORIES

    def test_all_tools_have_valid_category(self):
        """Test that all tools have a valid category."""
        for tool in TOOL_REGISTRY:
            assert tool.category in CATEGORIES

    def test_searchable_text(self):
        """Test that searchable_text returns combined text."""
        tool = TOOL_REGISTRY[0]
        searchable = tool.searchable_text()

        assert isinstance(searchable, str)
        assert tool.name.lower() in searchable
        assert tool.description.lower() in searchable


class TestRegexSearch:
    """Test suite for regex-based tool search."""

    def test_simple_regex_match(self):
        """Test simple regex pattern matching."""
        results = search_tools_regex("airport")
        assert len(results) > 0
        assert any("airport" in r.name.lower() for r in results)

    def test_regex_case_insensitive(self):
        """Test case-insensitive regex with (?i) flag."""
        results = search_tools_regex("(?i)ORBIT")
        assert len(results) > 0
        assert any(
            "orbit" in r.name.lower() or "orbit" in r.description.lower()
            for r in results
        )

    def test_regex_wildcard(self):
        """Test regex wildcard pattern."""
        results = search_tools_regex(".*flight.*")
        assert len(results) > 0

    def test_regex_or_pattern(self):
        """Test regex OR pattern."""
        results = search_tools_regex("rocket|orbit")
        assert len(results) > 0
        # Should match tools from both categories
        categories = {r.category for r in results}
        assert len(categories) >= 1

    def test_regex_max_results(self):
        """Test max_results parameter."""
        results = search_tools_regex(".*", max_results=3)
        assert len(results) == 3

    def test_regex_category_filter(self):
        """Test category filter."""
        results = search_tools_regex(".*", max_results=10, category="orbits")
        assert len(results) > 0
        for result in results:
            assert result.category == "orbits"

    def test_regex_invalid_pattern(self):
        """Test invalid regex pattern raises error."""
        with pytest.raises(ValueError, match="Invalid regex pattern"):
            search_tools_regex("[invalid")

    def test_regex_pattern_too_long(self):
        """Test pattern exceeding max length raises error."""
        long_pattern = "a" * (MAX_PATTERN_LENGTH + 1)
        with pytest.raises(ValueError, match="exceeds maximum length"):
            search_tools_regex(long_pattern)

    def test_regex_no_matches(self):
        """Test regex with no matches returns empty list."""
        results = search_tools_regex("zzzznonexistent")
        assert results == []


class TestTextSearch:
    """Test suite for text-based tool search."""

    def test_simple_text_search(self):
        """Test simple text query."""
        results = search_tools_text("flight planning")
        assert len(results) > 0

    def test_text_search_ranks_by_relevance(self):
        """Test that text search ranks by relevance."""
        results = search_tools_text("orbital elements")
        assert len(results) > 0
        # First result should be most relevant
        first = results[0]
        assert "orbit" in first.name.lower() or "orbit" in first.description.lower()

    def test_text_search_finds_by_description(self):
        """Test text search finds tools by description."""
        results = search_tools_text("great circle distance")
        assert len(results) > 0
        assert any("distance" in r.name.lower() for r in results)

    def test_text_search_finds_by_keywords(self):
        """Test text search finds tools by keywords."""
        results = search_tools_text("delta-v maneuver")
        assert len(results) > 0

    def test_text_search_max_results(self):
        """Test max_results parameter."""
        results = search_tools_text("calculate", max_results=2)
        assert len(results) <= 2

    def test_text_search_category_filter(self):
        """Test category filter."""
        results = search_tools_text("analysis", category="aerodynamics")
        assert len(results) > 0
        for result in results:
            assert result.category == "aerodynamics"

    def test_text_search_empty_query(self):
        """Test empty query returns empty list."""
        results = search_tools_text("")
        assert results == []

    def test_text_search_no_matches(self):
        """Test query with no matches returns empty list."""
        results = search_tools_text("zzzznonexistentterm")
        assert results == []


class TestSearchAerospaceTools:
    """Test suite for the main MCP tool function."""

    def test_auto_detects_text_search(self):
        """Test auto mode uses text search for plain queries."""
        result = search_aerospace_tools("atmospheric pressure")
        data = json.loads(result)

        assert "tool_references" in data
        assert data["search_type"] == "text"

    def test_auto_detects_regex_search(self):
        """Test auto mode uses regex for pattern-like queries."""
        result = search_aerospace_tools(".*orbit.*")
        data = json.loads(result)

        assert "tool_references" in data
        assert data["search_type"] == "regex"

    def test_explicit_regex_mode(self):
        """Test explicit regex search mode."""
        result = search_aerospace_tools("orbit", search_type="regex")
        data = json.loads(result)

        assert data["search_type"] == "regex"
        assert len(data["tool_references"]) > 0

    def test_explicit_text_mode(self):
        """Test explicit text search mode."""
        result = search_aerospace_tools("calculate distance", search_type="text")
        data = json.loads(result)

        assert data["search_type"] == "text"
        assert len(data["tool_references"]) > 0

    def test_tool_reference_format(self):
        """Test tool references match Anthropic's format for deferred loading."""
        result = search_aerospace_tools("airport")
        data = json.loads(result)

        # tool_references should match Anthropic's format exactly
        assert "tool_references" in data
        for ref in data["tool_references"]:
            assert ref["type"] == "tool_reference"
            assert "tool_name" in ref
            # Anthropic's format only includes type and tool_name
            assert set(ref.keys()) == {"type", "tool_name"}

        # tool_details provides human-readable metadata
        assert "tool_details" in data
        for detail in data["tool_details"]:
            assert "name" in detail
            assert "description" in detail
            assert "category" in detail

    def test_response_includes_metadata(self):
        """Test response includes helpful metadata."""
        result = search_aerospace_tools("flight")
        data = json.loads(result)

        assert "total_matches" in data
        assert "query" in data
        assert "search_type" in data
        assert "available_categories" in data

    def test_max_results_capped(self):
        """Test max_results is capped at 10."""
        result = search_aerospace_tools(".*", search_type="regex", max_results=100)
        data = json.loads(result)

        assert len(data["tool_references"]) <= 10

    def test_invalid_category_returns_error(self):
        """Test invalid category returns error message."""
        result = search_aerospace_tools("flight", category="invalid_category")
        data = json.loads(result)

        assert "error" in data
        assert "Invalid category" in data["error"]
        assert data["tool_references"] == []

    def test_category_filter(self):
        """Test category filter works."""
        result = search_aerospace_tools(".*", search_type="regex", category="rockets")
        data = json.loads(result)

        # Check tool_details for category (tool_references only has type/tool_name)
        for detail in data["tool_details"]:
            assert detail["category"] == "rockets"

    def test_regex_error_handling(self):
        """Test invalid regex returns error."""
        result = search_aerospace_tools("[invalid", search_type="regex")
        data = json.loads(result)

        assert "error" in data
        assert data["tool_references"] == []


class TestListToolCategories:
    """Test suite for list_tool_categories function."""

    def test_returns_valid_json(self):
        """Test function returns valid JSON."""
        result = list_tool_categories()
        data = json.loads(result)

        assert "categories" in data
        assert "total_tools" in data

    def test_categories_have_counts(self):
        """Test each category has a tool count."""
        result = list_tool_categories()
        data = json.loads(result)

        for cat in data["categories"]:
            assert "name" in cat
            assert "tool_count" in cat
            assert cat["tool_count"] >= 0

    def test_total_tools_matches_registry(self):
        """Test total_tools matches registry length."""
        result = list_tool_categories()
        data = json.loads(result)

        assert data["total_tools"] == len(TOOL_REGISTRY)


class TestToolDiscoveryScenarios:
    """Test real-world tool discovery scenarios."""

    def test_find_orbital_mechanics_tools(self):
        """Test finding orbital mechanics related tools."""
        result = search_aerospace_tools("orbital mechanics propagation")
        data = json.loads(result)

        tool_names = [ref["tool_name"] for ref in data["tool_references"]]
        assert any("orbit" in name or "propagate" in name for name in tool_names)

    def test_find_rocket_trajectory_tools(self):
        """Test finding rocket trajectory tools."""
        result = search_aerospace_tools("(?i)rocket.*trajectory", search_type="regex")
        data = json.loads(result)

        assert len(data["tool_references"]) > 0

    def test_find_atmospheric_tools(self):
        """Test finding atmospheric tools."""
        result = search_aerospace_tools("atmosphere pressure altitude")
        data = json.loads(result)

        tool_names = [ref["tool_name"] for ref in data["tool_references"]]
        assert "get_atmosphere_profile" in tool_names

    def test_find_coordinate_transformation_tools(self):
        """Test finding coordinate transformation tools."""
        result = search_aerospace_tools("ecef geodetic transform")
        data = json.loads(result)

        tool_names = [ref["tool_name"] for ref in data["tool_references"]]
        assert any(
            "ecef" in name or "geodetic" in name or "transform" in name
            for name in tool_names
        )

    def test_find_optimization_tools(self):
        """Test finding optimization tools."""
        result = search_aerospace_tools(
            "genetic algorithm optimization", category="optimization"
        )
        data = json.loads(result)

        assert len(data["tool_references"]) > 0
        assert all(
            detail["category"] == "optimization" for detail in data["tool_details"]
        )
