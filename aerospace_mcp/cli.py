"""CLI tool for invoking aerospace-mcp tools directly from the command line.

Usage:
    aerospace-mcp-cli list [--category CATEGORY]
    aerospace-mcp-cli search QUERY [--category CATEGORY] [--max-results N]
    aerospace-mcp-cli info TOOL_NAME
    aerospace-mcp-cli run TOOL_NAME [--param value ...]
"""

import argparse
import inspect
import json
import sys
import types
from collections.abc import Callable
from typing import Any, Literal, Union, get_type_hints

# Tool imports — mirrors aerospace_mcp/fastmcp_server.py
from .tools.aerodynamics import (
    airfoil_polar_analysis,
    calculate_stability_derivatives,
    get_airfoil_database,
    wing_vlm_analysis,
)
from .tools.agents import (
    format_data_for_tool,
    select_aerospace_tool,
)
from .tools.atmosphere import (
    get_atmosphere_profile,
    wind_model_simple,
)
from .tools.core import (
    calculate_distance,
    get_aircraft_performance,
    get_system_status,
    plan_flight,
    search_airports,
)
from .tools.frames import (
    ecef_to_geodetic,
    geodetic_to_ecef,
    transform_frames,
)
from .tools.gnc import (
    kalman_filter_state_estimation,
    lqr_controller_design,
)
from .tools.optimization import (
    genetic_algorithm_optimization,
    monte_carlo_uncertainty_analysis,
    optimize_thrust_profile,
    particle_swarm_optimization,
    porkchop_plot_analysis,
    trajectory_sensitivity_analysis,
)
from .tools.orbits import (
    calculate_ground_track,
    elements_to_state_vector,
    hohmann_transfer,
    lambert_problem_solver,
    orbital_rendezvous_planning,
    propagate_orbit_j2,
    state_vector_to_elements,
)
from .tools.performance import (
    density_altitude_calculator,
    fuel_reserve_calculator,
    landing_performance,
    stall_speed_calculator,
    takeoff_performance,
    true_airspeed_converter,
    weight_and_balance,
)
from .tools.propellers import (
    get_propeller_database,
    propeller_bemt_analysis,
    uav_energy_estimate,
)
from .tools.rockets import (
    estimate_rocket_sizing,
    optimize_launch_angle,
    rocket_3dof_trajectory,
)
from .tools.tool_search import (
    CATEGORIES,
    TOOL_REGISTRY,
    list_tool_categories,
    search_aerospace_tools,
)

# Complete mapping of tool name -> callable for all 44 tools
TOOL_MAP: dict[str, Callable[..., str]] = {
    # Discovery
    "search_aerospace_tools": search_aerospace_tools,
    "list_tool_categories": list_tool_categories,
    # Core
    "search_airports": search_airports,
    "plan_flight": plan_flight,
    "calculate_distance": calculate_distance,
    "get_aircraft_performance": get_aircraft_performance,
    "get_system_status": get_system_status,
    # Atmosphere
    "get_atmosphere_profile": get_atmosphere_profile,
    "wind_model_simple": wind_model_simple,
    # Frames
    "transform_frames": transform_frames,
    "geodetic_to_ecef": geodetic_to_ecef,
    "ecef_to_geodetic": ecef_to_geodetic,
    # Aerodynamics
    "wing_vlm_analysis": wing_vlm_analysis,
    "airfoil_polar_analysis": airfoil_polar_analysis,
    "calculate_stability_derivatives": calculate_stability_derivatives,
    "get_airfoil_database": get_airfoil_database,
    # Propellers
    "propeller_bemt_analysis": propeller_bemt_analysis,
    "uav_energy_estimate": uav_energy_estimate,
    "get_propeller_database": get_propeller_database,
    # Rockets
    "rocket_3dof_trajectory": rocket_3dof_trajectory,
    "estimate_rocket_sizing": estimate_rocket_sizing,
    "optimize_launch_angle": optimize_launch_angle,
    # Orbits
    "elements_to_state_vector": elements_to_state_vector,
    "state_vector_to_elements": state_vector_to_elements,
    "propagate_orbit_j2": propagate_orbit_j2,
    "calculate_ground_track": calculate_ground_track,
    "hohmann_transfer": hohmann_transfer,
    "orbital_rendezvous_planning": orbital_rendezvous_planning,
    "lambert_problem_solver": lambert_problem_solver,
    # GNC
    "kalman_filter_state_estimation": kalman_filter_state_estimation,
    "lqr_controller_design": lqr_controller_design,
    # Performance
    "density_altitude_calculator": density_altitude_calculator,
    "true_airspeed_converter": true_airspeed_converter,
    "stall_speed_calculator": stall_speed_calculator,
    "weight_and_balance": weight_and_balance,
    "takeoff_performance": takeoff_performance,
    "landing_performance": landing_performance,
    "fuel_reserve_calculator": fuel_reserve_calculator,
    # Optimization
    "optimize_thrust_profile": optimize_thrust_profile,
    "trajectory_sensitivity_analysis": trajectory_sensitivity_analysis,
    "genetic_algorithm_optimization": genetic_algorithm_optimization,
    "particle_swarm_optimization": particle_swarm_optimization,
    "porkchop_plot_analysis": porkchop_plot_analysis,
    "monte_carlo_uncertainty_analysis": monte_carlo_uncertainty_analysis,
    # Agents
    "format_data_for_tool": format_data_for_tool,
    "select_aerospace_tool": select_aerospace_tool,
}


def parse_tool_args(raw_args: list[str]) -> dict[str, str]:
    """Parse --key value pairs from a raw argument list.

    Args:
        raw_args: List of strings from argparse REMAINDER, e.g.
                  ["--query", "SFO", "--country", "US"]

    Returns:
        Dict mapping parameter names to their string values.
    """
    result: dict[str, str] = {}
    i = 0
    # Skip leading '--' separator if present
    if raw_args and raw_args[0] == "--":
        i = 1
    while i < len(raw_args):
        arg = raw_args[i]
        if arg.startswith("--"):
            key = arg[2:]
            if i + 1 < len(raw_args) and not raw_args[i + 1].startswith("--"):
                result[key] = raw_args[i + 1]
                i += 2
            else:
                # Boolean flag with no value
                result[key] = "true"
                i += 1
        else:
            i += 1  # skip unknown positional
    return result


def _get_annotation_origin(annotation: Any) -> Any:
    """Get the origin of a type annotation, handling both typing and builtins."""
    return getattr(annotation, "__origin__", None)


def _is_union(annotation: Any) -> bool:
    """Check if annotation is a Union type (typing.Union or X | Y)."""
    if _get_annotation_origin(annotation) is Union:
        return True
    if isinstance(annotation, types.UnionType):
        return True
    return False


def _get_union_args(annotation: Any) -> tuple[Any, ...]:
    """Get args from a Union type."""
    return getattr(annotation, "__args__", ())


def coerce_value(raw_value: str, annotation: Any) -> Any:
    """Coerce a string value to the expected Python type based on annotation.

    Args:
        raw_value: The raw string value from the CLI.
        annotation: The type annotation from the function signature.

    Returns:
        The coerced value in the correct Python type.
    """
    # Handle Union types (e.g., float | None, str | None)
    if _is_union(annotation):
        args = _get_union_args(annotation)
        non_none = [a for a in args if a is not type(None)]
        if non_none:
            return coerce_value(raw_value, non_none[0])
        return raw_value

    origin = _get_annotation_origin(annotation)

    # Literal types — validate against allowed values
    if origin is Literal:
        allowed = annotation.__args__
        if raw_value in [str(a) for a in allowed]:
            # Return the actual typed value (could be int in some Literals)
            for a in allowed:
                if str(a) == raw_value:
                    return a
        raise ValueError(
            f"Value '{raw_value}' not in allowed values: "
            f"{[str(a) for a in allowed]}"
        )

    # dict types — parse as JSON
    if annotation is dict or origin is dict:
        return json.loads(raw_value)

    # list types — parse as JSON array
    if annotation is list or origin is list:
        return json.loads(raw_value)

    # tuple types — parse as JSON array, convert to tuple
    if annotation is tuple or origin is tuple:
        return tuple(json.loads(raw_value))

    # float
    if annotation is float:
        return float(raw_value)

    # int
    if annotation is int:
        return int(raw_value)

    # bool
    if annotation is bool:
        return raw_value.lower() in ("true", "1", "yes")

    # str (default)
    return raw_value


def get_param_info(func: Callable[..., Any]) -> dict[str, dict[str, Any]]:
    """Get parameter info (type annotation, default) for a function.

    Returns:
        Dict mapping param name to {"annotation": ..., "default": ... or EMPTY}.
    """
    hints = get_type_hints(func)
    sig = inspect.signature(func)
    info: dict[str, dict[str, Any]] = {}
    for name, param in sig.parameters.items():
        info[name] = {
            "annotation": hints.get(name, str),
            "default": param.default,
        }
    return info


def pretty_print(result: str) -> None:
    """Pretty-print tool output. JSON is re-indented; text is printed as-is."""
    try:
        parsed = json.loads(result)
        print(json.dumps(parsed, indent=2))
    except (json.JSONDecodeError, TypeError):
        print(result)


def _format_annotation(annotation: Any) -> str:
    """Format a type annotation for display."""
    if annotation is inspect.Parameter.empty:
        return "any"
    # Handle common types nicely
    if annotation is str:
        return "str"
    if annotation is float:
        return "float"
    if annotation is int:
        return "int"
    if annotation is bool:
        return "bool"
    if annotation is dict:
        return "dict (JSON)"
    if annotation is list:
        return "list (JSON)"
    origin = _get_annotation_origin(annotation)
    if origin is Literal:
        vals = ", ".join(str(a) for a in annotation.__args__)
        return f"one of [{vals}]"
    if _is_union(annotation):
        args = _get_union_args(annotation)
        parts = [_format_annotation(a) for a in args if a is not type(None)]
        if len(parts) == 1:
            return f"{parts[0]} (optional)"
        return " | ".join(parts)
    if origin is dict:
        return "dict (JSON)"
    if origin is list:
        inner_args = getattr(annotation, "__args__", None)
        if inner_args:
            return f"list[{_format_annotation(inner_args[0])}] (JSON)"
        return "list (JSON)"
    return str(annotation)


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


def cmd_list(category: str | None = None) -> None:
    """List all available tools, optionally filtered by category."""
    tools = TOOL_REGISTRY
    if category:
        cat_lower = category.lower()
        tools = [t for t in tools if t.category.lower() == cat_lower]
        if not tools:
            print(
                f"No tools in category '{category}'. "
                f"Available: {', '.join(CATEGORIES)}"
            )
            return

    current_cat = ""
    for tool in tools:
        if tool.category != current_cat:
            current_cat = tool.category
            print(f"\n[{current_cat}]")
        print(f"  {tool.name:<40} {tool.description}")

    print(f"\n{len(tools)} tool(s) total")


def cmd_search(query: str, category: str | None = None, max_results: int = 5) -> None:
    """Search for tools matching a query."""
    result = search_aerospace_tools(query, category=category, max_results=max_results)
    data = json.loads(result)

    if data.get("error"):
        print(f"Error: {data['error']}", file=sys.stderr)
        return

    details = data.get("tool_details", [])
    if not details:
        print(f"No tools found for query: {query}")
        return

    print(f"Found {len(details)} tool(s) matching '{query}':\n")
    for detail in details:
        print(f"  {detail['name']:<40} [{detail['category']}]")
        print(f"    {detail['description']}")


def cmd_info(tool_name: str) -> None:
    """Show detailed information about a specific tool."""
    if tool_name not in TOOL_MAP:
        print(f"Error: Unknown tool '{tool_name}'", file=sys.stderr)
        sys.exit(1)

    # Get metadata from registry
    meta = next((t for t in TOOL_REGISTRY if t.name == tool_name), None)
    func = TOOL_MAP[tool_name]
    param_info = get_param_info(func)

    print(f"Tool: {tool_name}")
    if meta:
        print(f"Category: {meta.category}")
        print(f"Description: {meta.description}")
    else:
        doc = inspect.getdoc(func)
        if doc:
            print(f"Description: {doc.split(chr(10))[0]}")

    print("\nParameters:")
    if not param_info:
        print("  (none)")
    else:
        for name, info in param_info.items():
            type_str = _format_annotation(info["annotation"])
            default = info["default"]
            if default is inspect.Parameter.empty:
                default_str = " (required)"
            elif default is None:
                default_str = ""  # type annotation already shows "optional"
            else:
                default_str = f" (default: {default})"
            print(f"  --{name}: {type_str}{default_str}")
            if meta:
                desc = meta.parameters.get(name, "")
                if desc:
                    print(f"      {desc}")

    if meta and meta.keywords:
        print(f"\nKeywords: {', '.join(meta.keywords)}")

    print("\nExample:")
    example_args = []
    for name, info in param_info.items():
        if info["default"] is inspect.Parameter.empty:
            example_args.append(f"--{name} <value>")
    print(f"  aerospace-mcp-cli run {tool_name} {' '.join(example_args)}")


def cmd_run(tool_name: str, raw_args: list[str]) -> None:
    """Run a tool with the given arguments."""
    if tool_name not in TOOL_MAP:
        print(f"Error: Unknown tool '{tool_name}'", file=sys.stderr)
        print("\nUse 'aerospace-mcp-cli list' to see available tools.")
        sys.exit(1)

    func = TOOL_MAP[tool_name]
    raw_kv = parse_tool_args(raw_args)
    param_info = get_param_info(func)

    kwargs: dict[str, Any] = {}
    for key, raw_val in raw_kv.items():
        if key not in param_info:
            print(
                f"Warning: Unknown parameter '{key}' for {tool_name}",
                file=sys.stderr,
            )
            continue
        try:
            kwargs[key] = coerce_value(raw_val, param_info[key]["annotation"])
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Error parsing --{key}: {e}", file=sys.stderr)
            sys.exit(1)

    # Validate required parameters are present
    missing = []
    for name, info in param_info.items():
        if info["default"] is inspect.Parameter.empty and name not in kwargs:
            missing.append(name)
    if missing:
        print(
            f"Error: Missing required parameter(s): {', '.join('--' + m for m in missing)}",
            file=sys.stderr,
        )
        print(f"\nUse 'aerospace-mcp-cli info {tool_name}' for parameter details.")
        sys.exit(1)

    try:
        result = func(**kwargs)
        pretty_print(result)
    except Exception as e:
        print(f"Error running {tool_name}: {e}", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="aerospace-mcp-cli",
        description=(
            "CLI for aerospace-mcp — invoke any of 44 aerospace engineering "
            "tools directly from the command line."
        ),
    )
    subparsers = parser.add_subparsers(dest="command")

    # list
    list_parser = subparsers.add_parser("list", help="List available tools")
    list_parser.add_argument(
        "--category",
        "-c",
        default=None,
        help=f"Filter by category ({', '.join(CATEGORIES)})",
    )

    # search
    search_parser = subparsers.add_parser("search", help="Search for tools")
    search_parser.add_argument("query", help="Search query (text or regex)")
    search_parser.add_argument(
        "--category", "-c", default=None, help="Filter by category"
    )
    search_parser.add_argument(
        "--max-results", "-n", type=int, default=5, help="Max results (default: 5)"
    )

    # info
    info_parser = subparsers.add_parser("info", help="Show tool details")
    info_parser.add_argument("tool_name", help="Name of the tool")

    # run
    run_parser = subparsers.add_parser(
        "run",
        help="Run a tool with arguments",
        description=(
            "Run a specific tool. Pass arguments as --param value pairs. "
            "For dict/list parameters, pass JSON strings."
        ),
    )
    run_parser.add_argument("tool_name", help="Name of the tool to run")
    run_parser.add_argument(
        "tool_args",
        nargs=argparse.REMAINDER,
        help="Tool arguments as --param value pairs",
    )

    return parser


def main() -> None:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    if args.command == "list":
        cmd_list(args.category)
    elif args.command == "search":
        cmd_search(args.query, args.category, args.max_results)
    elif args.command == "info":
        cmd_info(args.tool_name)
    elif args.command == "run":
        cmd_run(args.tool_name, args.tool_args)


if __name__ == "__main__":
    main()
