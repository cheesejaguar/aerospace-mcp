"""Tests for the agent tools functionality."""

import os

import pytest

from aerospace_mcp.tools.agents import (
    AEROSPACE_TOOLS,
    LITELLM_AVAILABLE,
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

    @pytest.mark.skipif(not LITELLM_AVAILABLE, reason="litellm not installed")
    def test_format_data_for_tool_invalid_tool(self):
        """Test format_data_for_tool with invalid tool name when LLM enabled."""
        # Temporarily enable LLM tools and set API key
        original_enabled = os.environ.get("LLM_TOOLS_ENABLED")
        original_key = os.environ.get("OPENAI_API_KEY")

        os.environ["LLM_TOOLS_ENABLED"] = "true"
        os.environ["OPENAI_API_KEY"] = "dummy_key"  # Dummy key for validation test

        try:
            # Need to reload the module to pick up the new environment variable
            import importlib

            from aerospace_mcp.tools import agents

            importlib.reload(agents)

            result = agents.format_data_for_tool(
                tool_name="nonexistent_tool", user_requirements="Some requirements"
            )

            assert "Error: Tool 'nonexistent_tool' not found" in result
            assert "Available tools:" in result
        finally:
            # Restore original settings
            if original_enabled is not None:
                os.environ["LLM_TOOLS_ENABLED"] = original_enabled
            else:
                os.environ.pop("LLM_TOOLS_ENABLED", None)
            if original_key is not None:
                os.environ["OPENAI_API_KEY"] = original_key
            else:
                os.environ.pop("OPENAI_API_KEY", None)

    def test_format_data_for_tool_llm_disabled(self):
        """Test format_data_for_tool with LLM tools disabled."""
        # Temporarily set LLM_TOOLS_ENABLED to false
        original_enabled = os.environ.get("LLM_TOOLS_ENABLED")
        os.environ["LLM_TOOLS_ENABLED"] = "false"

        try:
            # Need to reload the module to pick up the new environment variable
            import importlib

            from aerospace_mcp.tools import agents

            importlib.reload(agents)

            result = agents.format_data_for_tool(
                tool_name="search_airports", user_requirements="Find airports in Tokyo"
            )

            # If litellm not installed, we get that error first
            if not LITELLM_AVAILABLE:
                assert "Error: litellm not installed" in result
            else:
                assert "Error: LLM agent tools are disabled" in result
        finally:
            # Restore original setting
            if original_enabled is not None:
                os.environ["LLM_TOOLS_ENABLED"] = original_enabled
            else:
                os.environ.pop("LLM_TOOLS_ENABLED", None)

    @pytest.mark.skipif(not LITELLM_AVAILABLE, reason="litellm not installed")
    def test_format_data_for_tool_valid_tool_no_api_key(self):
        """Test format_data_for_tool with valid tool but no API key."""
        # Temporarily set LLM_TOOLS_ENABLED to true and remove API key
        original_enabled = os.environ.get("LLM_TOOLS_ENABLED")
        original_key = os.environ.get("OPENAI_API_KEY")

        os.environ["LLM_TOOLS_ENABLED"] = "true"
        if original_key:
            del os.environ["OPENAI_API_KEY"]

        try:
            # Need to reload the module to pick up the new environment variable
            import importlib

            from aerospace_mcp.tools import agents

            importlib.reload(agents)

            result = agents.format_data_for_tool(
                tool_name="search_airports", user_requirements="Find airports in Tokyo"
            )

            assert "Error: OPENAI_API_KEY environment variable not set" in result
        finally:
            # Restore original settings
            if original_enabled is not None:
                os.environ["LLM_TOOLS_ENABLED"] = original_enabled
            else:
                os.environ.pop("LLM_TOOLS_ENABLED", None)
            if original_key:
                os.environ["OPENAI_API_KEY"] = original_key

    def test_select_aerospace_tool_llm_disabled(self):
        """Test select_aerospace_tool with LLM tools disabled."""
        # Temporarily set LLM_TOOLS_ENABLED to false
        original_enabled = os.environ.get("LLM_TOOLS_ENABLED")
        os.environ["LLM_TOOLS_ENABLED"] = "false"

        try:
            # Need to reload the module to pick up the new environment variable
            import importlib

            from aerospace_mcp.tools import agents

            importlib.reload(agents)

            result = agents.select_aerospace_tool(
                user_task="Plan a flight", user_context="Using A320 aircraft"
            )

            # If litellm not installed, we get that error first
            if not LITELLM_AVAILABLE:
                assert "Error: litellm not installed" in result
            else:
                assert "Error: LLM agent tools are disabled" in result
        finally:
            # Restore original setting
            if original_enabled is not None:
                os.environ["LLM_TOOLS_ENABLED"] = original_enabled
            else:
                os.environ.pop("LLM_TOOLS_ENABLED", None)

    @pytest.mark.skipif(not LITELLM_AVAILABLE, reason="litellm not installed")
    def test_select_aerospace_tool_no_api_key(self):
        """Test select_aerospace_tool with no API key."""
        # Temporarily set LLM_TOOLS_ENABLED to true and remove API key
        original_enabled = os.environ.get("LLM_TOOLS_ENABLED")
        original_key = os.environ.get("OPENAI_API_KEY")

        os.environ["LLM_TOOLS_ENABLED"] = "true"
        if original_key:
            del os.environ["OPENAI_API_KEY"]

        try:
            # Need to reload the module to pick up the new environment variable
            import importlib

            from aerospace_mcp.tools import agents

            importlib.reload(agents)

            result = agents.select_aerospace_tool(
                user_task="Plan a flight", user_context="Using A320 aircraft"
            )

            assert "Error: OPENAI_API_KEY environment variable not set" in result
        finally:
            # Restore original settings
            if original_enabled is not None:
                os.environ["LLM_TOOLS_ENABLED"] = original_enabled
            else:
                os.environ.pop("LLM_TOOLS_ENABLED", None)
            if original_key:
                os.environ["OPENAI_API_KEY"] = original_key

    @pytest.mark.skipif(
        not LITELLM_AVAILABLE
        or "OPENAI_API_KEY" not in os.environ
        or os.environ.get("LLM_TOOLS_ENABLED", "false").lower() != "true",
        reason="litellm not installed, OPENAI_API_KEY not available, or LLM tools disabled",
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
        not LITELLM_AVAILABLE
        or "OPENAI_API_KEY" not in os.environ
        or os.environ.get("LLM_TOOLS_ENABLED", "false").lower() != "true",
        reason="litellm not installed, OPENAI_API_KEY not available, or LLM tools disabled",
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

    @pytest.mark.skipif(LITELLM_AVAILABLE, reason="litellm is installed")
    def test_litellm_not_installed_error(self):
        """Test that appropriate error is returned when litellm is not installed."""
        result = format_data_for_tool(
            tool_name="search_airports",
            user_requirements="Find airports in Tokyo",
        )
        assert "Error: litellm not installed" in result
        assert "pip install aerospace-mcp[agents]" in result

        result = select_aerospace_tool(
            user_task="Plan a flight",
            user_context="Using A320 aircraft",
        )
        assert "Error: litellm not installed" in result
