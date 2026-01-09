"""FastMCP server implementation for Aerospace flight planning tools.

Loads environment from .env before importing tools so feature flags and
API keys are available at import time.
"""

import logging
import sys

# Load environment from .env before importing tool modules
try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover
    pass

from fastmcp import FastMCP

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

# Import all tool modules
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
    orbital_rendezvous_planning,
    propagate_orbit_j2,
    state_vector_to_elements,
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP("aerospace-mcp")

# Register all tools
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
    """Main entry point for the FastMCP server."""
    # Check for SSE mode
    if len(sys.argv) > 1 and sys.argv[1] == "sse":
        host = sys.argv[2] if len(sys.argv) > 2 else "localhost"
        port = int(sys.argv[3]) if len(sys.argv) > 3 else 8001
        logger.info(f"Starting FastMCP server in SSE mode on {host}:{port}")
        mcp.run(transport="sse", host=host, port=port)
    else:
        logger.info("Starting FastMCP server in stdio mode")
        mcp.run()


if __name__ == "__main__":
    run()
