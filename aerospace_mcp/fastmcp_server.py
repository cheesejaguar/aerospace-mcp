"""FastMCP server implementation for Aerospace flight planning tools.

This module provides the high-level MCP (Model Context Protocol) server using
the FastMCP framework. Unlike the low-level ``server.py`` which manually defines
JSON schemas and dispatches tool calls, this implementation leverages FastMCP's
decorator-based tool registration to automatically generate schemas from Python
function signatures and docstrings.

Architecture:
    1. Tool functions are defined in ``aerospace_mcp/tools/`` submodules, each
       decorated with type hints and Google-style docstrings.
    2. This module imports all tool functions and registers them with the FastMCP
       server instance via ``mcp.tool(func)``.
    3. FastMCP automatically generates MCP-compatible JSON schemas, handles
       argument parsing, and routes incoming tool calls to the correct handler.

Transport Modes:
    - **stdio** (default): Standard input/output for production MCP clients.
      Used when launched as ``aerospace-mcp`` from the command line.
    - **SSE** (Server-Sent Events): HTTP-based transport for debugging and
      browser-based MCP clients. Activated via ``aerospace-mcp sse [host] [port]``.

Deferred Tool Loading:
    The server supports Anthropic's deferred tool loading pattern. Discovery
    tools (``search_aerospace_tools``, ``list_tool_categories``) are loaded
    eagerly, while domain-specific tools can be deferred to save context window
    space. See the TOOL REGISTRATION section below for configuration details.

Loads environment from .env before importing tools so feature flags and
API keys are available at import time.

WARNING:
    This module is for educational and research purposes only.
    Do NOT use for real flight planning, navigation, or aircraft operations.
"""

import logging
import sys

# Load environment from .env before importing tool modules
try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

from fastmcp import FastMCP

# =============================================================================
# TOOL IMPORTS — organized by aerospace domain
# Each submodule in aerospace_mcp/tools/ defines one or more MCP tools as
# plain Python functions. FastMCP inspects their signatures and docstrings
# to auto-generate the MCP tool schemas.
# =============================================================================
# --- Aerodynamics: wing analysis, airfoil polars, stability derivatives ---
from .tools.aerodynamics import (
    airfoil_polar_analysis,
    calculate_stability_derivatives,
    get_airfoil_database,
    wing_vlm_analysis,
)

# --- AI Agent Helpers: tool selection and data formatting for LLM agents ---
from .tools.agents import (
    format_data_for_tool,
    select_aerospace_tool,
)

# --- Atmosphere: ISA model profiles and wind models ---
from .tools.atmosphere import (
    get_atmosphere_profile,
    wind_model_simple,
)

# --- Core Flight Planning: airport search, route planning, distance, performance ---
from .tools.core import (
    calculate_distance,
    get_aircraft_performance,
    get_system_status,
    plan_flight,
    search_airports,
)

# --- Coordinate Frames: ECEF/ECI/geodetic transforms ---
from .tools.frames import (
    ecef_to_geodetic,
    geodetic_to_ecef,
    transform_frames,
)

# --- GNC (Guidance, Navigation & Control): Kalman filtering, LQR control ---
from .tools.gnc import (
    kalman_filter_state_estimation,
    lqr_controller_design,
)

# --- Optimization: thrust profiles, sensitivity, GA, PSO, Monte Carlo ---
from .tools.optimization import (
    genetic_algorithm_optimization,
    monte_carlo_uncertainty_analysis,
    optimize_thrust_profile,
    particle_swarm_optimization,
    porkchop_plot_analysis,
    trajectory_sensitivity_analysis,
)

# --- Orbital Mechanics: Kepler elements, propagation, transfers, Lambert ---
from .tools.orbits import (
    calculate_ground_track,
    elements_to_state_vector,
    hohmann_transfer,
    lambert_problem_solver,
    orbital_rendezvous_planning,
    propagate_orbit_j2,
    state_vector_to_elements,
)

# --- Performance: density altitude, W&B, takeoff/landing, fuel, airspeed ---
from .tools.performance import (
    density_altitude_calculator,
    fuel_reserve_calculator,
    landing_performance,
    stall_speed_calculator,
    takeoff_performance,
    true_airspeed_converter,
    weight_and_balance,
)

# --- Propellers & UAVs: BEMT analysis, energy estimation ---
from .tools.propellers import (
    get_propeller_database,
    propeller_bemt_analysis,
    uav_energy_estimate,
)

# --- Rockets: 3-DOF trajectories, sizing, launch optimization ---
from .tools.rockets import (
    estimate_rocket_sizing,
    optimize_launch_angle,
    rocket_3dof_trajectory,
)

# --- Tool Discovery: search and categorize available aerospace tools ---
from .tools.tool_search import (
    list_tool_categories,
    search_aerospace_tools,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP("aerospace-mcp")

# =============================================================================
# TOOL REGISTRATION
# =============================================================================
# aerospace-mcp supports deferred tool loading for efficient context usage.
# When using with Anthropic's API, configure the mcp_toolset like this:
#
#   {
#     "type": "mcp_toolset",
#     "mcp_server_name": "aerospace-mcp",
#     "default_config": {"defer_loading": true},
#     "configs": {
#       "search_aerospace_tools": {"defer_loading": false},
#       "list_tool_categories": {"defer_loading": false}
#     }
#   }
#
# This loads only the discovery tools initially. Claude uses search_aerospace_tools
# to find relevant tools, which returns tool_reference blocks that the API
# automatically expands into full tool definitions.
# =============================================================================

# --- DISCOVERY TOOLS (should NOT be deferred) ---
# These tools enable Claude to discover other tools dynamically
mcp.tool(search_aerospace_tools)
mcp.tool(list_tool_categories)

# --- DISCOVERABLE TOOLS (can be deferred for context efficiency) ---
# Core flight planning tools
mcp.tool(search_airports)
mcp.tool(plan_flight)
mcp.tool(calculate_distance)
mcp.tool(get_aircraft_performance)
mcp.tool(get_system_status)

# Atmospheric tools
mcp.tool(get_atmosphere_profile)
mcp.tool(wind_model_simple)

# Coordinate frame tools
mcp.tool(transform_frames)
mcp.tool(geodetic_to_ecef)
mcp.tool(ecef_to_geodetic)

# Aerodynamics tools
mcp.tool(wing_vlm_analysis)
mcp.tool(airfoil_polar_analysis)
mcp.tool(calculate_stability_derivatives)
mcp.tool(get_airfoil_database)

# Propeller/UAV tools
mcp.tool(propeller_bemt_analysis)
mcp.tool(uav_energy_estimate)
mcp.tool(get_propeller_database)

# Rocket tools
mcp.tool(rocket_3dof_trajectory)
mcp.tool(estimate_rocket_sizing)
mcp.tool(optimize_launch_angle)

# Orbital mechanics tools
mcp.tool(elements_to_state_vector)
mcp.tool(state_vector_to_elements)
mcp.tool(propagate_orbit_j2)
mcp.tool(calculate_ground_track)
mcp.tool(hohmann_transfer)
mcp.tool(orbital_rendezvous_planning)
mcp.tool(lambert_problem_solver)

# GNC tools
mcp.tool(kalman_filter_state_estimation)
mcp.tool(lqr_controller_design)

# Performance tools
mcp.tool(density_altitude_calculator)
mcp.tool(true_airspeed_converter)
mcp.tool(stall_speed_calculator)
mcp.tool(weight_and_balance)
mcp.tool(takeoff_performance)
mcp.tool(landing_performance)
mcp.tool(fuel_reserve_calculator)

# Optimization tools
mcp.tool(optimize_thrust_profile)
mcp.tool(trajectory_sensitivity_analysis)
mcp.tool(genetic_algorithm_optimization)
mcp.tool(particle_swarm_optimization)
mcp.tool(porkchop_plot_analysis)
mcp.tool(monte_carlo_uncertainty_analysis)

# Agent tools for helping users
mcp.tool(format_data_for_tool)
mcp.tool(select_aerospace_tool)


def run():
    """Start the FastMCP aerospace tools server.

    Selects the transport mode based on command-line arguments:
        - No args (default): stdio mode for production MCP clients.
        - ``sse [host] [port]``: SSE (Server-Sent Events) mode for HTTP-based
          MCP clients and browser debugging. Defaults to localhost:8001.

    This function is the console_scripts entry point registered as
    ``aerospace-mcp`` in pyproject.toml.
    """
    # Check for SSE mode — useful for debugging with browser-based MCP clients
    if len(sys.argv) > 1 and sys.argv[1] == "sse":
        host = sys.argv[2] if len(sys.argv) > 2 else "localhost"
        port = int(sys.argv[3]) if len(sys.argv) > 3 else 8001
        logger.info(f"Starting FastMCP server in SSE mode on {host}:{port}")
        mcp.run(transport="sse", host=host, port=port)
    else:
        # stdio mode: MCP client communicates over stdin/stdout
        logger.info("Starting FastMCP server in stdio mode")
        mcp.run()


if __name__ == "__main__":
    run()
