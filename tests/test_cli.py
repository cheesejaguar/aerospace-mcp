"""Tests for the aerospace-mcp CLI tool."""

import inspect
import json
import sys
from unittest.mock import patch

import pytest

from aerospace_mcp.cli import (
    TOOL_MAP,
    build_parser,
    cmd_info,
    cmd_list,
    cmd_run,
    cmd_search,
    coerce_value,
    get_param_info,
    main,
    parse_tool_args,
    pretty_print,
)
from aerospace_mcp.tools.tool_search import TOOL_REGISTRY

# ---------------------------------------------------------------------------
# TestToolMap
# ---------------------------------------------------------------------------


class TestToolMap:
    """Tests for the TOOL_MAP registry."""

    def test_tool_map_not_empty(self):
        assert len(TOOL_MAP) > 0

    def test_tool_map_has_all_registry_tools(self):
        registry_names = {t.name for t in TOOL_REGISTRY}
        map_names = set(TOOL_MAP.keys())
        missing = registry_names - map_names
        assert not missing, f"Tools in registry but not in TOOL_MAP: {missing}"

    def test_tool_map_values_are_callable(self):
        for name, func in TOOL_MAP.items():
            assert callable(func), f"TOOL_MAP['{name}'] is not callable"

    def test_tool_map_count(self):
        # Should have at least as many tools as the registry
        assert len(TOOL_MAP) >= len(TOOL_REGISTRY)


# ---------------------------------------------------------------------------
# TestParseToolArgs
# ---------------------------------------------------------------------------


class TestParseToolArgs:
    """Tests for parse_tool_args."""

    def test_simple_key_value(self):
        result = parse_tool_args(["--query", "SFO"])
        assert result == {"query": "SFO"}

    def test_multiple_params(self):
        result = parse_tool_args(
            ["--lat1", "37.0", "--lon1", "-122.0", "--lat2", "35.0", "--lon2", "140.0"]
        )
        assert result == {
            "lat1": "37.0",
            "lon1": "-122.0",
            "lat2": "35.0",
            "lon2": "140.0",
        }

    def test_json_dict_param(self):
        raw = ["--departure", '{"city":"San Jose","iata":"SJC"}']
        result = parse_tool_args(raw)
        assert result["departure"] == '{"city":"San Jose","iata":"SJC"}'
        # Should be parseable as JSON
        parsed = json.loads(result["departure"])
        assert parsed["city"] == "San Jose"

    def test_json_list_param(self):
        raw = ["--altitudes_m", "[0,1000,5000]"]
        result = parse_tool_args(raw)
        parsed = json.loads(result["altitudes_m"])
        assert parsed == [0, 1000, 5000]

    def test_skip_leading_separator(self):
        result = parse_tool_args(["--", "--query", "SFO"])
        assert result == {"query": "SFO"}

    def test_empty_args(self):
        result = parse_tool_args([])
        assert result == {}

    def test_boolean_flag(self):
        result = parse_tool_args(["--verbose", "--query", "test"])
        assert result["verbose"] == "true"
        assert result["query"] == "test"


# ---------------------------------------------------------------------------
# TestCoerceValue
# ---------------------------------------------------------------------------


class TestCoerceValue:
    """Tests for coerce_value type coercion."""

    def test_coerce_float(self):
        assert coerce_value("3.14", float) == pytest.approx(3.14)

    def test_coerce_int(self):
        assert coerce_value("42", int) == 42

    def test_coerce_str(self):
        assert coerce_value("hello", str) == "hello"

    def test_coerce_bool_true(self):
        assert coerce_value("true", bool) is True

    def test_coerce_bool_false(self):
        assert coerce_value("no", bool) is False

    def test_coerce_dict(self):
        result = coerce_value('{"key": "value"}', dict)
        assert result == {"key": "value"}

    def test_coerce_list_float(self):
        result = coerce_value("[1.0, 2.0, 3.0]", list[float])
        assert result == [1.0, 2.0, 3.0]

    def test_coerce_list_plain(self):
        result = coerce_value("[1, 2, 3]", list)
        assert result == [1, 2, 3]

    def test_coerce_tuple(self):
        result = coerce_value("[45.0, 90.0]", tuple)
        assert result == (45.0, 90.0)

    def test_coerce_optional_float(self):
        result = coerce_value("3.14", float | None)
        assert result == pytest.approx(3.14)

    def test_coerce_optional_str(self):
        result = coerce_value("test", str | None)
        assert result == "test"

    def test_coerce_literal_valid(self):
        from typing import Literal

        result = coerce_value("iata", Literal["iata", "city", "auto"])
        assert result == "iata"

    def test_coerce_literal_invalid(self):
        from typing import Literal

        with pytest.raises(ValueError, match="not in allowed"):
            coerce_value("invalid", Literal["iata", "city", "auto"])

    def test_coerce_json_error(self):
        with pytest.raises(json.JSONDecodeError):
            coerce_value("not-json", dict)


# ---------------------------------------------------------------------------
# TestGetParamInfo
# ---------------------------------------------------------------------------


class TestGetParamInfo:
    """Tests for get_param_info introspection."""

    def test_simple_function(self):
        def sample(x: float, y: str = "default") -> str:
            return ""

        info = get_param_info(sample)
        assert "x" in info
        assert "y" in info
        assert info["x"]["annotation"] is float
        assert info["x"]["default"] is inspect.Parameter.empty
        assert info["y"]["annotation"] is str
        assert info["y"]["default"] == "default"

    def test_tool_function(self):
        from aerospace_mcp.tools.core import search_airports

        info = get_param_info(search_airports)
        assert "query" in info
        assert info["query"]["annotation"] is str


# ---------------------------------------------------------------------------
# TestPrettyPrint
# ---------------------------------------------------------------------------


class TestPrettyPrint:
    """Tests for pretty_print output."""

    def test_json_output(self, capsys):
        pretty_print('{"key": "value"}')
        captured = capsys.readouterr()
        assert '"key": "value"' in captured.out

    def test_text_output(self, capsys):
        pretty_print("Just plain text")
        captured = capsys.readouterr()
        assert "Just plain text" in captured.out


# ---------------------------------------------------------------------------
# TestCmdList
# ---------------------------------------------------------------------------


class TestCmdList:
    """Tests for the list subcommand."""

    def test_list_all(self, capsys):
        cmd_list()
        captured = capsys.readouterr()
        # Should contain at least some known tool names
        assert "search_airports" in captured.out
        assert "hohmann_transfer" in captured.out

    def test_list_category_filter(self, capsys):
        cmd_list(category="core")
        captured = capsys.readouterr()
        assert "search_airports" in captured.out
        # Should NOT contain orbital tools
        assert "hohmann_transfer" not in captured.out

    def test_list_invalid_category(self, capsys):
        cmd_list(category="nonexistent")
        captured = capsys.readouterr()
        assert "No tools in category" in captured.out


# ---------------------------------------------------------------------------
# TestCmdSearch
# ---------------------------------------------------------------------------


class TestCmdSearch:
    """Tests for the search subcommand."""

    def test_search_returns_results(self, capsys):
        cmd_search("orbit")
        captured = capsys.readouterr()
        assert "Found" in captured.out

    def test_search_no_results(self, capsys):
        cmd_search("xyzzynonexistent12345")
        captured = capsys.readouterr()
        assert "No tools found" in captured.out


# ---------------------------------------------------------------------------
# TestCmdInfo
# ---------------------------------------------------------------------------


class TestCmdInfo:
    """Tests for the info subcommand."""

    def test_info_known_tool(self, capsys):
        cmd_info("hohmann_transfer")
        captured = capsys.readouterr()
        assert "Tool: hohmann_transfer" in captured.out
        assert "r1_m" in captured.out
        assert "r2_m" in captured.out

    def test_info_unknown_tool(self):
        with pytest.raises(SystemExit):
            cmd_info("nonexistent_tool")


# ---------------------------------------------------------------------------
# TestCmdRun
# ---------------------------------------------------------------------------


class TestCmdRun:
    """Tests for the run subcommand."""

    def test_run_get_system_status(self, capsys):
        cmd_run("get_system_status", [])
        captured = capsys.readouterr()
        output = json.loads(captured.out)
        assert output["status"] == "operational"

    def test_run_list_tool_categories(self, capsys):
        cmd_run("list_tool_categories", [])
        captured = capsys.readouterr()
        output = json.loads(captured.out)
        assert "categories" in output
        assert output["total_tools"] > 0

    def test_run_search_aerospace_tools(self, capsys):
        cmd_run("search_aerospace_tools", ["--query", "orbit"])
        captured = capsys.readouterr()
        output = json.loads(captured.out)
        assert output["total_matches"] > 0

    def test_run_unknown_tool(self):
        with pytest.raises(SystemExit):
            cmd_run("nonexistent_tool", [])

    def test_run_missing_required_param(self):
        with pytest.raises(SystemExit):
            cmd_run("search_airports", [])  # missing --query

    def test_run_hohmann_transfer(self, capsys):
        cmd_run("hohmann_transfer", ["--r1_m", "6778000", "--r2_m", "42164000"])
        captured = capsys.readouterr()
        output = json.loads(captured.out)
        assert "total_delta_v_ms" in output

    def test_run_density_altitude(self, capsys):
        cmd_run(
            "density_altitude_calculator",
            ["--pressure_altitude_ft", "5000", "--temperature_c", "30"],
        )
        captured = capsys.readouterr()
        # Should produce some output about density altitude
        assert "density" in captured.out.lower() or "altitude" in captured.out.lower()

    def test_run_with_json_args(self, capsys):
        """Test running a tool that takes dict args via JSON."""
        cmd_run(
            "search_aerospace_tools",
            ["--query", "fuel", "--category", "performance"],
        )
        captured = capsys.readouterr()
        output = json.loads(captured.out)
        assert "tool_references" in output


# ---------------------------------------------------------------------------
# TestBuildParser
# ---------------------------------------------------------------------------


class TestBuildParser:
    """Tests for the argument parser."""

    def test_parser_list(self):
        parser = build_parser()
        args = parser.parse_args(["list"])
        assert args.command == "list"
        assert args.category is None

    def test_parser_list_with_category(self):
        parser = build_parser()
        args = parser.parse_args(["list", "--category", "core"])
        assert args.command == "list"
        assert args.category == "core"

    def test_parser_search(self):
        parser = build_parser()
        args = parser.parse_args(["search", "orbit"])
        assert args.command == "search"
        assert args.query == "orbit"

    def test_parser_info(self):
        parser = build_parser()
        args = parser.parse_args(["info", "hohmann_transfer"])
        assert args.command == "info"
        assert args.tool_name == "hohmann_transfer"

    def test_parser_run(self):
        parser = build_parser()
        args = parser.parse_args(["run", "get_system_status"])
        assert args.command == "run"
        assert args.tool_name == "get_system_status"

    def test_parser_run_with_args(self):
        parser = build_parser()
        args = parser.parse_args(["run", "search_airports", "--query", "SFO"])
        assert args.command == "run"
        assert args.tool_name == "search_airports"
        assert "--query" in args.tool_args
        assert "SFO" in args.tool_args


# ---------------------------------------------------------------------------
# TestMainEntryPoint
# ---------------------------------------------------------------------------


class TestMainEntryPoint:
    """Tests for the main() entry point."""

    def test_main_no_args(self, capsys):
        with patch.object(sys, "argv", ["aerospace-mcp-cli"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0

    def test_main_list(self, capsys):
        with patch.object(sys, "argv", ["aerospace-mcp-cli", "list"]):
            main()
        captured = capsys.readouterr()
        assert "search_airports" in captured.out

    def test_main_search(self, capsys):
        with patch.object(sys, "argv", ["aerospace-mcp-cli", "search", "orbit"]):
            main()
        captured = capsys.readouterr()
        assert "Found" in captured.out

    def test_main_info(self, capsys):
        with patch.object(
            sys, "argv", ["aerospace-mcp-cli", "info", "hohmann_transfer"]
        ):
            main()
        captured = capsys.readouterr()
        assert "Tool: hohmann_transfer" in captured.out

    def test_main_run(self, capsys):
        with patch.object(
            sys, "argv", ["aerospace-mcp-cli", "run", "get_system_status"]
        ):
            main()
        captured = capsys.readouterr()
        assert "operational" in captured.out
