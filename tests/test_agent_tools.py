"""Tests for the agent tools functionality."""

import os

import pytest

from aerospace_mcp.tools.agents import (
    AEROSPACE_TOOLS,
    format_data_for_tool,
    select_aerospace_tool,
)


class TestAgentTools:
    """Test suite for agent tools."""

    def test_aerospace_tools_catalog(self):
        """Test that the aerospace tools catalog is properly structured."""
        assert len(AEROSPACE_TOOLS) > 0

        # Check that each tool has required fields
        for tool in AEROSPACE_TOOLS:
            assert tool.name
            assert tool.description
            assert isinstance(tool.parameters, dict)
            assert isinstance(tool.examples, list)

    def test_format_data_for_tool_invalid_tool(self):
        """Test format_data_for_tool with invalid tool name."""
        result = format_data_for_tool(
            tool_name="nonexistent_tool", user_requirements="Some requirements"
        )

        assert "Error: Tool 'nonexistent_tool' not found" in result
        assert "Available tools:" in result

    def test_format_data_for_tool_valid_tool_no_api_key(self):
        """Test format_data_for_tool with valid tool but no API key."""
        # Temporarily remove API key if it exists
        original_key = os.environ.get("OPENAI_API_KEY")
        if original_key:
            del os.environ["OPENAI_API_KEY"]

        try:
            result = format_data_for_tool(
                tool_name="search_airports", user_requirements="Find airports in Tokyo"
            )

            assert "Error: OPENAI_API_KEY environment variable not set" in result
        finally:
            # Restore API key if it existed
            if original_key:
                os.environ["OPENAI_API_KEY"] = original_key

    def test_select_aerospace_tool_no_api_key(self):
        """Test select_aerospace_tool with no API key."""
        # Temporarily remove API key if it exists
        original_key = os.environ.get("OPENAI_API_KEY")
        if original_key:
            del os.environ["OPENAI_API_KEY"]

        try:
            result = select_aerospace_tool(
                user_task="Plan a flight", user_context="Using A320 aircraft"
            )

            assert "Error: OPENAI_API_KEY environment variable not set" in result
        finally:
            # Restore API key if it existed
            if original_key:
                os.environ["OPENAI_API_KEY"] = original_key

    @pytest.mark.skipif(
        "OPENAI_API_KEY" not in os.environ,
        reason="OPENAI_API_KEY not available for integration testing",
    )
    def test_format_data_for_tool_with_api_key(self):
        """Integration test for format_data_for_tool with API key."""
        result = format_data_for_tool(
            tool_name="search_airports",
            user_requirements="Find airports in Tokyo, Japan",
            raw_data="",
        )

        # Should not contain error messages
        assert "Error:" not in result
        # Should be parseable JSON (basic check)
        assert result.startswith("{") or result.startswith("[")

    @pytest.mark.skipif(
        "OPENAI_API_KEY" not in os.environ,
        reason="OPENAI_API_KEY not available for integration testing",
    )
    def test_select_aerospace_tool_with_api_key(self):
        """Integration test for select_aerospace_tool with API key."""
        result = select_aerospace_tool(
            user_task="I want to plan a flight from San Francisco to New York",
            user_context="Using an Airbus A320",
        )

        # Should not contain error messages
        assert "Error:" not in result
        # Should contain recommendations
        assert len(result) > 50  # Should be a substantial response

    def test_tool_catalog_completeness(self):
        """Test that key aerospace tools are included in the catalog."""
        tool_names = [tool.name for tool in AEROSPACE_TOOLS]

        # Check for essential tools
        essential_tools = [
            "search_airports",
            "plan_flight",
            "calculate_distance",
            "get_aircraft_performance",
        ]

        for tool in essential_tools:
            assert tool in tool_names, f"Essential tool '{tool}' missing from catalog"
