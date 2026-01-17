"""Tool search functionality for aerospace-mcp.

Implements a tool search tool following Anthropic's guide for dynamic tool discovery.
Supports both regex and text-based search patterns.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class ToolMetadata:
    """Metadata for a searchable aerospace tool."""

    name: str
    description: str
    category: str
    parameters: dict[str, str] = field(default_factory=dict)
    keywords: list[str] = field(default_factory=list)

    def searchable_text(self) -> str:
        """Return all searchable text concatenated."""
        parts = [
            self.name,
            self.description,
            self.category,
            " ".join(self.parameters.keys()),
            " ".join(self.parameters.values()),
            " ".join(self.keywords),
        ]
        return " ".join(parts).lower()


# Comprehensive registry of all aerospace-mcp tools
TOOL_REGISTRY: list[ToolMetadata] = [
    # Core Flight Planning Tools
    ToolMetadata(
        name="search_airports",
        description="Search for airports by IATA code or city name",
        category="core",
        parameters={
            "query": "IATA code (e.g., 'SJC') or city name (e.g., 'San Jose')",
            "country": "Optional ISO country code filter (e.g., 'US', 'JP')",
            "query_type": "Type of query - 'iata', 'city', or 'auto'",
        },
        keywords=["airport", "flight", "iata", "icao", "city", "lookup", "find"],
    ),
    ToolMetadata(
        name="plan_flight",
        description="Plan a flight route between two airports with performance estimates",
        category="core",
        parameters={
            "departure": "Departure airport info (city, country, iata)",
            "arrival": "Arrival airport info (city, country, iata)",
            "aircraft": "Aircraft config (ac_type, cruise_alt_ft, mass_kg)",
            "route_options": "Route options (step_km)",
        },
        keywords=[
            "flight",
            "route",
            "planning",
            "aircraft",
            "fuel",
            "distance",
            "navigation",
        ],
    ),
    ToolMetadata(
        name="calculate_distance",
        description="Calculate great-circle distance between airports",
        category="core",
        parameters={
            "origin": "Origin airport with city and optional iata/country",
            "destination": "Destination airport with city and optional iata/country",
            "step_km": "Step size for route polyline generation",
        },
        keywords=[
            "distance",
            "great-circle",
            "geodesic",
            "route",
            "kilometers",
            "nautical miles",
        ],
    ),
    ToolMetadata(
        name="get_aircraft_performance",
        description="Get performance estimates for aircraft using OpenAP",
        category="core",
        parameters={
            "aircraft_type": "Aircraft type code (e.g., 'A320', 'B737', 'B777')",
            "distance_km": "Flight distance in kilometers",
            "cruise_altitude_ft": "Cruise altitude in feet",
            "mass_kg": "Aircraft mass in kg",
        },
        keywords=[
            "performance",
            "aircraft",
            "fuel",
            "consumption",
            "climb",
            "cruise",
            "descent",
        ],
    ),
    ToolMetadata(
        name="get_system_status",
        description="Get current system capabilities and status",
        category="core",
        parameters={},
        keywords=["status", "health", "capabilities", "system", "info"],
    ),
    # Atmospheric Tools
    ToolMetadata(
        name="get_atmosphere_profile",
        description="Calculate atmospheric conditions at various altitudes using ISA model",
        category="atmosphere",
        parameters={
            "altitudes_m": "List of altitudes in meters",
            "model_type": "Atmospheric model type - 'isa' or 'enhanced'",
        },
        keywords=[
            "atmosphere",
            "pressure",
            "temperature",
            "density",
            "altitude",
            "isa",
            "standard",
        ],
    ),
    ToolMetadata(
        name="wind_model_simple",
        description="Calculate wind profiles at various altitudes using logarithmic or power-law models",
        category="atmosphere",
        parameters={
            "altitudes_m": "List of altitudes in meters",
            "surface_wind_mps": "Surface wind speed in m/s",
            "model": "Wind profile model - 'logarithmic' or 'power_law'",
            "surface_roughness_m": "Surface roughness in meters",
        },
        keywords=["wind", "profile", "altitude", "speed", "meteorology"],
    ),
    # Coordinate Frame Tools
    ToolMetadata(
        name="transform_frames",
        description="Transform coordinates between reference frames (ECEF, ECI, ITRF, GCRS, GEODETIC)",
        category="frames",
        parameters={
            "position": "Position vector [x, y, z] in source frame",
            "velocity": "Optional velocity vector [vx, vy, vz]",
            "source_frame": "Source reference frame",
            "target_frame": "Target reference frame",
            "epoch_utc": "UTC epoch for time-dependent transformations",
        },
        keywords=[
            "coordinate",
            "transform",
            "frame",
            "ecef",
            "eci",
            "geodetic",
            "reference",
        ],
    ),
    ToolMetadata(
        name="geodetic_to_ecef",
        description="Convert geodetic coordinates (lat/lon/alt) to ECEF",
        category="frames",
        parameters={
            "latitude_deg": "Latitude in degrees",
            "longitude_deg": "Longitude in degrees",
            "altitude_m": "Altitude above ellipsoid in meters",
        },
        keywords=["geodetic", "ecef", "latitude", "longitude", "altitude", "convert"],
    ),
    ToolMetadata(
        name="ecef_to_geodetic",
        description="Convert ECEF coordinates to geodetic (lat/lon/alt)",
        category="frames",
        parameters={
            "x_m": "ECEF X coordinate in meters",
            "y_m": "ECEF Y coordinate in meters",
            "z_m": "ECEF Z coordinate in meters",
        },
        keywords=["ecef", "geodetic", "latitude", "longitude", "altitude", "convert"],
    ),
    # Aerodynamics Tools
    ToolMetadata(
        name="wing_vlm_analysis",
        description="Perform Vortex Lattice Method analysis on a wing",
        category="aerodynamics",
        parameters={
            "span_m": "Wing span in meters",
            "chord_m": "Wing chord in meters",
            "alpha_deg": "Angle of attack in degrees",
            "velocity_mps": "Freestream velocity in m/s",
            "n_panels_span": "Number of panels along span",
            "n_panels_chord": "Number of panels along chord",
        },
        keywords=[
            "vlm",
            "vortex",
            "lattice",
            "wing",
            "lift",
            "aerodynamic",
            "analysis",
        ],
    ),
    ToolMetadata(
        name="airfoil_polar_analysis",
        description="Generate airfoil polar data (CL, CD, CM vs alpha)",
        category="aerodynamics",
        parameters={
            "airfoil": "Airfoil name (e.g., 'naca0012', 'naca2412')",
            "reynolds": "Reynolds number",
            "alpha_range": "Range of angles of attack [start, end, step]",
        },
        keywords=[
            "airfoil",
            "polar",
            "lift",
            "drag",
            "coefficient",
            "cl",
            "cd",
            "cm",
            "alpha",
        ],
    ),
    ToolMetadata(
        name="calculate_stability_derivatives",
        description="Calculate longitudinal stability derivatives for an aircraft",
        category="aerodynamics",
        parameters={
            "wing_area_m2": "Wing reference area in m^2",
            "wing_span_m": "Wing span in meters",
            "mac_m": "Mean aerodynamic chord in meters",
            "cl_alpha": "Lift curve slope per radian",
            "cm_alpha": "Pitching moment slope per radian",
        },
        keywords=[
            "stability",
            "derivatives",
            "longitudinal",
            "control",
            "dynamics",
            "aircraft",
        ],
    ),
    ToolMetadata(
        name="get_airfoil_database",
        description="Look up airfoil data from database",
        category="aerodynamics",
        parameters={
            "airfoil_name": "Name of airfoil to look up",
            "data_type": "Type of data to retrieve",
        },
        keywords=["airfoil", "database", "lookup", "naca", "profile"],
    ),
    # Propeller/UAV Tools
    ToolMetadata(
        name="propeller_bemt_analysis",
        description="Perform Blade Element Momentum Theory analysis on a propeller",
        category="propellers",
        parameters={
            "diameter_m": "Propeller diameter in meters",
            "pitch_m": "Propeller pitch in meters",
            "rpm": "Rotational speed in RPM",
            "velocity_mps": "Freestream velocity in m/s",
            "n_blades": "Number of blades",
            "n_elements": "Number of blade elements for analysis",
        },
        keywords=[
            "propeller",
            "bemt",
            "blade",
            "element",
            "momentum",
            "thrust",
            "torque",
        ],
    ),
    ToolMetadata(
        name="uav_energy_estimate",
        description="Estimate flight time and energy consumption for a UAV",
        category="propellers",
        parameters={
            "mass_kg": "UAV mass in kg",
            "battery_wh": "Battery capacity in Wh",
            "motor_efficiency": "Motor efficiency (0-1)",
            "propeller_efficiency": "Propeller efficiency (0-1)",
            "cruise_velocity_mps": "Cruise velocity in m/s",
        },
        keywords=[
            "uav",
            "drone",
            "energy",
            "battery",
            "flight time",
            "endurance",
            "consumption",
        ],
    ),
    ToolMetadata(
        name="get_propeller_database",
        description="Look up propeller data from database",
        category="propellers",
        parameters={
            "propeller_name": "Name or model of propeller",
            "data_type": "Type of data to retrieve",
        },
        keywords=["propeller", "database", "lookup", "apc", "specs"],
    ),
    # Rocket Tools
    ToolMetadata(
        name="rocket_3dof_trajectory",
        description="Simulate 3DOF rocket trajectory with atmospheric effects",
        category="rockets",
        parameters={
            "geometry": "Rocket geometry (mass_kg, thrust_n, burn_time_s, drag_coeff, reference_area_m2)",
            "dt_s": "Time step in seconds",
            "max_time_s": "Maximum simulation time in seconds",
            "launch_angle_deg": "Launch angle in degrees from vertical",
        },
        keywords=[
            "rocket",
            "trajectory",
            "3dof",
            "simulation",
            "ballistic",
            "launch",
            "thrust",
        ],
    ),
    ToolMetadata(
        name="estimate_rocket_sizing",
        description="Estimate rocket sizing for target altitude and payload",
        category="rockets",
        parameters={
            "target_altitude_m": "Target altitude in meters",
            "payload_mass_kg": "Payload mass in kg",
            "propellant_type": "Type of propellant",
        },
        keywords=[
            "rocket",
            "sizing",
            "design",
            "payload",
            "altitude",
            "mass",
            "propellant",
        ],
    ),
    ToolMetadata(
        name="optimize_launch_angle",
        description="Optimize rocket launch angle for maximum range or altitude",
        category="rockets",
        parameters={
            "geometry": "Rocket geometry parameters",
            "objective": "Optimization objective - 'range' or 'altitude'",
            "constraints": "Optional constraints on the optimization",
        },
        keywords=[
            "rocket",
            "launch",
            "angle",
            "optimize",
            "range",
            "altitude",
            "trajectory",
        ],
    ),
    # Orbital Mechanics Tools
    ToolMetadata(
        name="elements_to_state_vector",
        description="Convert orbital elements to state vector (position and velocity)",
        category="orbits",
        parameters={
            "elements": "Orbital elements (semi_major_axis_m, eccentricity, inclination_deg, raan_deg, arg_periapsis_deg, true_anomaly_deg, epoch_utc)"
        },
        keywords=[
            "orbital",
            "elements",
            "state",
            "vector",
            "kepler",
            "position",
            "velocity",
        ],
    ),
    ToolMetadata(
        name="state_vector_to_elements",
        description="Convert state vector to orbital elements",
        category="orbits",
        parameters={
            "position": "Position vector [x, y, z] in meters",
            "velocity": "Velocity vector [vx, vy, vz] in m/s",
            "epoch_utc": "UTC epoch timestamp",
        },
        keywords=[
            "orbital",
            "elements",
            "state",
            "vector",
            "kepler",
            "convert",
            "classical",
        ],
    ),
    ToolMetadata(
        name="propagate_orbit_j2",
        description="Propagate satellite orbit with J2 perturbations",
        category="orbits",
        parameters={
            "initial_state": "Initial orbital elements or state vector",
            "time_span_s": "Propagation time span in seconds",
            "time_step_s": "Time step for propagation in seconds",
        },
        keywords=[
            "orbit",
            "propagate",
            "j2",
            "perturbation",
            "satellite",
            "trajectory",
            "prediction",
        ],
    ),
    ToolMetadata(
        name="calculate_ground_track",
        description="Calculate satellite ground track from orbital state",
        category="orbits",
        parameters={
            "orbital_state": "Orbital state (elements or state vector)",
            "duration_s": "Duration for ground track calculation",
            "step_s": "Time step for ground track points",
        },
        keywords=[
            "ground",
            "track",
            "satellite",
            "orbit",
            "latitude",
            "longitude",
            "path",
        ],
    ),
    ToolMetadata(
        name="hohmann_transfer",
        description="Calculate Hohmann transfer orbit between two circular orbits",
        category="orbits",
        parameters={
            "r1_m": "Initial orbit radius in meters",
            "r2_m": "Final orbit radius in meters",
        },
        keywords=[
            "hohmann",
            "transfer",
            "orbit",
            "delta-v",
            "maneuver",
            "circular",
            "leo",
            "geo",
        ],
    ),
    ToolMetadata(
        name="orbital_rendezvous_planning",
        description="Plan spacecraft rendezvous maneuvers",
        category="orbits",
        parameters={
            "chaser_state": "Chaser spacecraft orbital state",
            "target_state": "Target spacecraft orbital state",
            "approach_strategy": "Rendezvous approach strategy",
        },
        keywords=[
            "rendezvous",
            "docking",
            "spacecraft",
            "approach",
            "maneuver",
            "relative",
            "motion",
        ],
    ),
    # Optimization Tools
    ToolMetadata(
        name="optimize_thrust_profile",
        description="Optimize rocket thrust profile for efficiency",
        category="optimization",
        parameters={
            "geometry": "Rocket geometry parameters",
            "constraints": "Optimization constraints",
            "objective": "Optimization objective function",
        },
        keywords=[
            "thrust",
            "profile",
            "optimize",
            "efficiency",
            "rocket",
            "propulsion",
        ],
    ),
    ToolMetadata(
        name="trajectory_sensitivity_analysis",
        description="Perform sensitivity analysis on trajectory parameters",
        category="optimization",
        parameters={
            "baseline_trajectory": "Baseline trajectory to analyze",
            "parameters": "Parameters to vary",
            "variation_range": "Range of variation for each parameter",
        },
        keywords=[
            "sensitivity",
            "analysis",
            "trajectory",
            "variation",
            "uncertainty",
            "parameter",
        ],
    ),
    ToolMetadata(
        name="genetic_algorithm_optimization",
        description="Perform genetic algorithm-based trajectory optimization",
        category="optimization",
        parameters={
            "objective_function": "Objective function to optimize",
            "constraints": "Optimization constraints",
            "population_size": "GA population size",
            "generations": "Number of generations",
        },
        keywords=[
            "genetic",
            "algorithm",
            "ga",
            "optimization",
            "evolutionary",
            "trajectory",
        ],
    ),
    ToolMetadata(
        name="particle_swarm_optimization",
        description="Perform particle swarm optimization for trajectory design",
        category="optimization",
        parameters={
            "objective_function": "Objective function to optimize",
            "constraints": "Optimization constraints",
            "swarm_size": "Number of particles in swarm",
            "iterations": "Number of iterations",
        },
        keywords=["particle", "swarm", "pso", "optimization", "trajectory", "global"],
    ),
    ToolMetadata(
        name="porkchop_plot_analysis",
        description="Generate porkchop plot for interplanetary transfer opportunities",
        category="optimization",
        parameters={
            "departure_body": "Departure celestial body",
            "arrival_body": "Arrival celestial body",
            "departure_date_range": "Range of departure dates",
            "arrival_date_range": "Range of arrival dates",
        },
        keywords=[
            "porkchop",
            "interplanetary",
            "transfer",
            "launch",
            "window",
            "delta-v",
            "mars",
        ],
    ),
    ToolMetadata(
        name="monte_carlo_uncertainty_analysis",
        description="Perform Monte Carlo uncertainty quantification on trajectories",
        category="optimization",
        parameters={
            "baseline_trajectory": "Baseline trajectory",
            "uncertainty_parameters": "Parameters with uncertainty distributions",
            "num_samples": "Number of Monte Carlo samples",
        },
        keywords=[
            "monte",
            "carlo",
            "uncertainty",
            "quantification",
            "statistical",
            "dispersion",
        ],
    ),
    # Agent Tools
    ToolMetadata(
        name="format_data_for_tool",
        description="Help format data in the correct format for a specific aerospace-mcp tool using LLM",
        category="agents",
        parameters={
            "tool_name": "Name of the aerospace-mcp tool to format data for",
            "user_requirements": "Description of what the user wants to accomplish",
            "raw_data": "Any raw data that needs to be formatted",
        },
        keywords=["format", "data", "helper", "llm", "assistant", "convert"],
    ),
    ToolMetadata(
        name="select_aerospace_tool",
        description="Help select the most appropriate aerospace-mcp tool for a given task using LLM",
        category="agents",
        parameters={
            "user_task": "Description of what the user wants to accomplish",
            "user_context": "Additional context about the user's situation",
        },
        keywords=["select", "recommend", "tool", "llm", "assistant", "help"],
    ),
]

# Build category index for filtering
CATEGORIES = sorted({tool.category for tool in TOOL_REGISTRY})

# Maximum regex pattern length (matching Anthropic's limit)
MAX_PATTERN_LENGTH = 200


def _score_text_match(query_terms: list[str], tool: ToolMetadata) -> float:
    """Score a tool based on text query match.

    Scoring weights:
    - Name match: 10 points per term
    - Description match: 5 points per term
    - Category match: 3 points per term
    - Parameter match: 2 points per term
    - Keyword match: 4 points per term
    """
    score = 0.0
    name_lower = tool.name.lower()
    desc_lower = tool.description.lower()
    category_lower = tool.category.lower()
    params_text = " ".join(tool.parameters.keys()).lower()
    params_desc = " ".join(tool.parameters.values()).lower()
    keywords_text = " ".join(tool.keywords).lower()

    for term in query_terms:
        term = term.lower()
        if term in name_lower:
            score += 10
        if term in desc_lower:
            score += 5
        if term in category_lower:
            score += 3
        if term in params_text or term in params_desc:
            score += 2
        if term in keywords_text:
            score += 4

    return score


def search_tools_regex(
    pattern: str,
    max_results: int = 5,
    category: str | None = None,
) -> list[ToolMetadata]:
    """Search tools using regex pattern.

    Args:
        pattern: Regex pattern to search (max 200 chars)
        max_results: Maximum number of results to return
        category: Optional category filter

    Returns:
        List of matching ToolMetadata objects
    """
    if len(pattern) > MAX_PATTERN_LENGTH:
        raise ValueError(f"Pattern exceeds maximum length of {MAX_PATTERN_LENGTH}")

    try:
        regex = re.compile(pattern)
    except re.error as e:
        raise ValueError(f"Invalid regex pattern: {e}")

    results = []
    for tool in TOOL_REGISTRY:
        # Apply category filter
        if category and tool.category.lower() != category.lower():
            continue

        # Search in all searchable text
        searchable = tool.searchable_text()
        if regex.search(searchable):
            results.append(tool)

    return results[:max_results]


def search_tools_text(
    query: str,
    max_results: int = 5,
    category: str | None = None,
) -> list[ToolMetadata]:
    """Search tools using natural language query.

    Args:
        query: Natural language search query
        max_results: Maximum number of results to return
        category: Optional category filter

    Returns:
        List of matching ToolMetadata objects, sorted by relevance
    """
    # Tokenize query into terms
    query_terms = [term.strip() for term in query.lower().split() if term.strip()]

    if not query_terms:
        return []

    # Score all tools
    scored_tools = []
    for tool in TOOL_REGISTRY:
        # Apply category filter
        if category and tool.category.lower() != category.lower():
            continue

        score = _score_text_match(query_terms, tool)
        if score > 0:
            scored_tools.append((score, tool))

    # Sort by score (descending) and return top results
    scored_tools.sort(key=lambda x: x[0], reverse=True)
    return [tool for _, tool in scored_tools[:max_results]]


def search_aerospace_tools(
    query: str,
    search_type: Literal["regex", "text", "auto"] = "auto",
    max_results: int = 5,
    category: str | None = None,
) -> str:
    """Search for aerospace-mcp tools by name, description, or functionality.

    This tool enables dynamic tool discovery, allowing Claude to find relevant
    tools from the 34+ available aerospace tools without loading all definitions
    upfront. Returns tool references matching the search query.

    Args:
        query: Search query - regex pattern (for regex mode) or natural language (for text mode)
        search_type: Search mode - 'regex' for pattern matching, 'text' for natural language,
                     'auto' to detect based on query characteristics
        max_results: Maximum number of tools to return (default 5, max 10)
        category: Optional category filter (core, atmosphere, frames, aerodynamics,
                  propellers, rockets, orbits, optimization, agents)

    Returns:
        JSON string with tool references matching the query
    """
    # Validate inputs
    max_results = min(max(1, max_results), 10)

    if category and category.lower() not in [c.lower() for c in CATEGORIES]:
        return json.dumps(
            {
                "error": f"Invalid category '{category}'. Valid categories: {', '.join(CATEGORIES)}",
                "tool_references": [],
            }
        )

    # Auto-detect search type
    if search_type == "auto":
        # Use regex if query contains regex metacharacters
        regex_chars = r".*+?^${}[]|()\\"
        if any(c in query for c in regex_chars):
            search_type = "regex"
        else:
            search_type = "text"

    # Perform search
    try:
        if search_type == "regex":
            results = search_tools_regex(query, max_results, category)
        else:
            results = search_tools_text(query, max_results, category)
    except ValueError as e:
        return json.dumps(
            {
                "error": str(e),
                "tool_references": [],
            }
        )

    # Format response matching Anthropic's tool_reference pattern
    tool_references = [
        {
            "type": "tool_reference",
            "tool_name": tool.name,
            "description": tool.description,
            "category": tool.category,
        }
        for tool in results
    ]

    return json.dumps(
        {
            "tool_references": tool_references,
            "total_matches": len(results),
            "query": query,
            "search_type": search_type,
            "available_categories": CATEGORIES,
        },
        indent=2,
    )


def list_tool_categories() -> str:
    """List all available tool categories with tool counts.

    Returns:
        JSON string with category information
    """
    category_counts = {}
    for tool in TOOL_REGISTRY:
        category_counts[tool.category] = category_counts.get(tool.category, 0) + 1

    return json.dumps(
        {
            "categories": [
                {"name": cat, "tool_count": category_counts.get(cat, 0)}
                for cat in CATEGORIES
            ],
            "total_tools": len(TOOL_REGISTRY),
        },
        indent=2,
    )
