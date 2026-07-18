"""Single source of truth for aerospace-mcp tool registration.

``ALL_TOOLS`` maps tool name -> callable for every tool exposed by the
project.  Both the FastMCP server (``aerospace_mcp/fastmcp_server.py``)
and the CLI (``aerospace_mcp/cli.py``) consume this dict, so a tool added
here is automatically available in both interfaces.

Adding a new tool requires exactly two registrations:
    1. An entry in ``ALL_TOOLS`` below (implementation callable).
    2. A ``ToolMetadata`` entry in ``tool_search.TOOL_REGISTRY``
       (search/discovery metadata).
A contract test (``tests/test_tool_schema_contracts.py``) enforces that
the two stay in sync.

Discovery tools (``search_aerospace_tools``, ``list_tool_categories``)
are listed first: MCP clients using deferred loading keep these eagerly
loaded, and they intentionally have no ``TOOL_REGISTRY`` metadata entry
(they discover the other tools; they are not themselves discoverable).
"""

from collections.abc import Callable

from .aerodynamics import (
    airfoil_polar_analysis,
    calculate_stability_derivatives,
    get_airfoil_database,
    wing_vlm_analysis,
)
from .agents import (
    format_data_for_tool,
    select_aerospace_tool,
)
from .atmosphere import (
    get_atmosphere_profile,
    wind_model_simple,
)
from .core import (
    calculate_distance,
    get_aircraft_database,
    get_aircraft_performance,
    get_system_status,
    plan_flight,
    plan_multi_leg_flight,
    search_airports,
)
from .frames import (
    ecef_to_geodetic,
    geodetic_to_ecef,
    transform_frames,
)
from .gnc import (
    kalman_filter_state_estimation,
    lqr_controller_design,
)
from .optimization import (
    genetic_algorithm_optimization,
    monte_carlo_uncertainty_analysis,
    optimize_thrust_profile,
    particle_swarm_optimization,
    porkchop_plot_analysis,
    trajectory_sensitivity_analysis,
)
from .orbits import (
    calculate_ground_track,
    elements_to_state_vector,
    hohmann_transfer,
    lambert_problem_solver,
    orbital_rendezvous_planning,
    propagate_orbit_j2,
    state_vector_to_elements,
)
from .performance import (
    density_altitude_calculator,
    fuel_reserve_calculator,
    landing_performance,
    stall_speed_calculator,
    takeoff_performance,
    true_airspeed_converter,
    weight_and_balance,
)
from .propellers import (
    get_propeller_database,
    propeller_bemt_analysis,
    uav_energy_estimate,
)
from .rockets import (
    estimate_rocket_sizing,
    optimize_launch_angle,
    rocket_3dof_trajectory,
)
from .tool_search import (
    list_tool_categories,
    search_aerospace_tools,
)
from .units import (
    convert_units,
)

# Discovery tools first (kept eagerly loaded by deferred-loading clients),
# then domain tools grouped by category.
ALL_TOOLS: dict[str, Callable[..., str]] = {
    # Discovery
    "search_aerospace_tools": search_aerospace_tools,
    "list_tool_categories": list_tool_categories,
    # Core
    "search_airports": search_airports,
    "plan_flight": plan_flight,
    "plan_multi_leg_flight": plan_multi_leg_flight,
    "calculate_distance": calculate_distance,
    "get_aircraft_performance": get_aircraft_performance,
    "get_aircraft_database": get_aircraft_database,
    "get_system_status": get_system_status,
    # Utility
    "convert_units": convert_units,
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

# Tools that must stay eagerly loaded under deferred-loading configs.
DISCOVERY_TOOLS = ("search_aerospace_tools", "list_tool_categories")
