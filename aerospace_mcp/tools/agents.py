"""Agent tools for helping users interact with aerospace-mcp tools effectively."""

import json
import logging
import os
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Check if litellm is available (optional dependency)
LITELLM_AVAILABLE = False
litellm = None
try:
    import litellm as _litellm

    litellm = _litellm
    LITELLM_AVAILABLE = True
    litellm.set_verbose = False
except ImportError:
    logger.info(
        "litellm not installed. Install with: pip install aerospace-mcp[agents]"
    )

# Check if LLM tools are enabled via environment variable
LLM_TOOLS_ENABLED = os.environ.get("LLM_TOOLS_ENABLED", "false").lower() == "true"

# Log status of LLM tools
if not LITELLM_AVAILABLE:
    pass  # Already logged above
elif not LLM_TOOLS_ENABLED:
    logger.info("LLM tools disabled via LLM_TOOLS_ENABLED environment variable.")
elif "OPENAI_API_KEY" not in os.environ:
    logger.warning(
        "LLM_TOOLS_ENABLED=true but OPENAI_API_KEY not set. Agent tools will not function without it."
    )


class ToolReference(BaseModel):
    """Reference to an aerospace-mcp tool with its schema."""

    name: str
    description: str
    parameters: dict[str, Any]
    examples: list[str] = []


# Define all available aerospace-mcp tools with their schemas
AEROSPACE_TOOLS = [
    ToolReference(
        name="search_airports",
        description="Search for airports by IATA code or city name",
        parameters={
            "query": "str - IATA code (e.g., 'SJC') or city name (e.g., 'San Jose')",
            "country": "str | None - Optional ISO country code filter (e.g., 'US', 'JP')",
            "query_type": "Literal['iata', 'city', 'auto'] - Type of query, defaults to 'auto'",
        },
        examples=[
            'search_airports("SFO")',
            'search_airports("London", "GB")',
            'search_airports("Tokyo")',
        ],
    ),
    ToolReference(
        name="plan_flight",
        description="Generate complete flight plan between airports",
        parameters={
            "departure": "dict - Airport info with city, iata (optional), country (optional)",
            "arrival": "dict - Airport info with city, iata (optional), country (optional)",
            "aircraft": "dict - Aircraft config with type, cruise_alt_ft, mass_kg (optional)",
            "route_options": "dict - Route config with step_km (optional, default 25.0)",
        },
        examples=[
            'plan_flight({"city": "San Francisco"}, {"city": "New York"}, {"type": "A320", "cruise_alt_ft": 37000}, {})',
            'plan_flight({"city": "London", "iata": "LHR"}, {"city": "Dubai", "iata": "DXB"}, {"type": "B777", "cruise_alt_ft": 39000, "mass_kg": 220000}, {"step_km": 50.0})',
        ],
    ),
    ToolReference(
        name="calculate_distance",
        description="Calculate great-circle distance between airports",
        parameters={
            "origin": "dict - Origin airport with city and optional iata/country",
            "destination": "dict - Destination airport with city and optional iata/country",
            "step_km": "float - Optional step size for route polyline generation (default 25.0)",
        },
        examples=[
            'calculate_distance({"city": "New York"}, {"city": "Los Angeles"})',
            'calculate_distance({"city": "Paris", "iata": "CDG"}, {"city": "Tokyo", "iata": "NRT"}, 100.0)',
        ],
    ),
    ToolReference(
        name="get_aircraft_performance",
        description="Get performance estimates for aircraft",
        parameters={
            "aircraft_type": "str - Aircraft type code (e.g., 'A320', 'B737', 'B777')",
            "distance_km": "float - Flight distance in kilometers",
            "cruise_altitude_ft": "float - Cruise altitude in feet (optional, default 35000)",
            "mass_kg": "float - Aircraft mass in kg (optional, uses 85% MTOW if not provided)",
        },
        examples=[
            'get_aircraft_performance("A320", 2500.0, 37000)',
            'get_aircraft_performance("B777", 5500.0, 39000, 250000)',
        ],
    ),
    ToolReference(
        name="get_atmosphere_profile",
        description="Calculate atmospheric conditions at various altitudes",
        parameters={
            "altitudes_m": "List[float] - List of altitudes in meters",
            "model_type": "Literal['isa', 'enhanced'] - Atmospheric model type (default 'isa')",
        },
        examples=[
            "get_atmosphere_profile([0, 1000, 5000, 10000])",
            'get_atmosphere_profile([0, 2000, 4000, 6000, 8000, 10000], "enhanced")',
        ],
    ),
    ToolReference(
        name="wind_model_simple",
        description="Calculate wind profiles at various altitudes",
        parameters={
            "altitudes_m": "List[float] - List of altitudes in meters",
            "surface_wind_mps": "float - Surface wind speed in m/s",
            "model": "Literal['logarithmic', 'power_law'] - Wind profile model (default 'logarithmic')",
            "surface_roughness_m": "float - Surface roughness in meters (default 0.1)",
        },
        examples=[
            "wind_model_simple([0, 100, 500, 1000], 10.0)",
            'wind_model_simple([0, 200, 1000, 3000], 15.0, "power_law", 0.05)',
        ],
    ),
    ToolReference(
        name="elements_to_state_vector",
        description="Convert orbital elements to state vector",
        parameters={
            "elements": "dict - Orbital elements with semi_major_axis_m, eccentricity, inclination_deg, raan_deg, arg_periapsis_deg, true_anomaly_deg, epoch_utc"
        },
        examples=[
            'elements_to_state_vector({"semi_major_axis_m": 6793000, "eccentricity": 0.001, "inclination_deg": 51.6, "raan_deg": 0.0, "arg_periapsis_deg": 0.0, "true_anomaly_deg": 0.0, "epoch_utc": "2024-01-01T12:00:00"})'
        ],
    ),
    ToolReference(
        name="propagate_orbit_j2",
        description="Propagate satellite orbit with J2 perturbations",
        parameters={
            "initial_state": "dict - Initial orbital elements or state vector",
            "time_span_s": "float - Propagation time span in seconds",
            "time_step_s": "float - Time step for propagation in seconds (default 300)",
        },
        examples=[
            'propagate_orbit_j2({"semi_major_axis_m": 6793000, "eccentricity": 0.001, "inclination_deg": 51.6, "raan_deg": 0.0, "arg_periapsis_deg": 0.0, "true_anomaly_deg": 0.0, "epoch_utc": "2024-01-01T12:00:00"}, 86400, 600)'
        ],
    ),
    ToolReference(
        name="hohmann_transfer",
        description="Calculate Hohmann transfer orbit between two circular orbits",
        parameters={
            "r1_m": "float - Initial orbit radius in meters",
            "r2_m": "float - Final orbit radius in meters",
        },
        examples=[
            "hohmann_transfer(6778000, 42164000)",  # LEO to GEO
            "hohmann_transfer(6578000, 6793000)",  # Lower LEO to ISS altitude
        ],
    ),
    ToolReference(
        name="rocket_3dof_trajectory",
        description="Simulate 3DOF rocket trajectory with atmospheric effects",
        parameters={
            "geometry": "dict - Rocket geometry with mass_kg, thrust_n, burn_time_s, drag_coeff, reference_area_m2",
            "dt_s": "float - Time step in seconds (default 0.1)",
            "max_time_s": "float - Maximum simulation time in seconds (default 300)",
            "launch_angle_deg": "float - Launch angle in degrees from vertical (default 0)",
        },
        examples=[
            'rocket_3dof_trajectory({"mass_kg": 500, "thrust_n": 8000, "burn_time_s": 60, "drag_coeff": 0.3, "reference_area_m2": 0.5}, 0.1, 300, 15)'
        ],
    ),
]


def format_data_for_tool(
    tool_name: str, user_requirements: str, raw_data: str = ""
) -> str:
    """
    Help format data in the correct format for a specific aerospace-mcp tool.

    Uses GPT-5-Medium to analyze the user's requirements and raw data, then provides
    the correctly formatted parameters for the specified tool.

    Args:
        tool_name: Name of the aerospace-mcp tool to format data for
        user_requirements: Description of what the user wants to accomplish
        raw_data: Any raw data that needs to be formatted (optional)

    Returns:
        Formatted JSON string with the correct parameters for the tool
    """
    # Check if litellm is available
    if not LITELLM_AVAILABLE:
        return "Error: litellm not installed. Install with: pip install aerospace-mcp[agents]"

    # Check if LLM tools are enabled
    if not LLM_TOOLS_ENABLED:
        return "Error: LLM agent tools are disabled. Set LLM_TOOLS_ENABLED=true to enable them."

    # Find the tool reference first
    tool_ref = None
    for tool in AEROSPACE_TOOLS:
        if tool.name == tool_name:
            tool_ref = tool
            break

    if not tool_ref:
        available_tools = [t.name for t in AEROSPACE_TOOLS]
        return f"Error: Tool '{tool_name}' not found. Available tools: {', '.join(available_tools)}"

    if "OPENAI_API_KEY" not in os.environ:
        return "Error: OPENAI_API_KEY environment variable not set. Cannot use agent tools."

    # Build the prompt for GPT-5-Medium
    system_prompt = f"""You are a data formatting assistant for aerospace-mcp tools. Your job is to help format data correctly for the '{tool_name}' tool.

Tool Information:
- Name: {tool_ref.name}
- Description: {tool_ref.description}
- Parameters: {json.dumps(tool_ref.parameters, indent=2)}
- Examples: {json.dumps(tool_ref.examples, indent=2)}

User Requirements: {user_requirements}

Raw Data (if provided): {raw_data}

Please provide ONLY a valid JSON object with the correctly formatted parameters for this tool. Do not include any explanation or additional text - just the JSON object that can be directly used as input to the tool.

If the user's requirements are unclear or insufficient data is provided, return a JSON object with an "error" field explaining what additional information is needed."""

    try:
        # Call GPT-5-Medium via LiteLLM
        response = litellm.completion(
            model="gpt-5-medium",
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"Format data for {tool_name}: {user_requirements}",
                },
            ],
            temperature=0.1,
            max_tokens=1000,
        )

        formatted_result = response.choices[0].message.content.strip()

        # Validate it's valid JSON
        try:
            json.loads(formatted_result)
            return formatted_result
        except json.JSONDecodeError:
            return f'{{"error": "Failed to generate valid JSON format. Raw response: {formatted_result}"}}'

    except Exception as e:
        return f'{{"error": "Failed to format data: {str(e)}"}}'


def select_aerospace_tool(user_task: str, user_context: str = "") -> str:
    """
    Help select the most appropriate aerospace-mcp tool for a given task.

    Uses GPT-5-Medium to analyze the user's task and recommend the best tool(s)
    along with guidance on how to use them.

    Args:
        user_task: Description of what the user wants to accomplish
        user_context: Additional context about the user's situation (optional)

    Returns:
        Recommendation with tool name(s) and usage guidance
    """
    # Check if litellm is available
    if not LITELLM_AVAILABLE:
        return "Error: litellm not installed. Install with: pip install aerospace-mcp[agents]"

    # Check if LLM tools are enabled
    if not LLM_TOOLS_ENABLED:
        return "Error: LLM agent tools are disabled. Set LLM_TOOLS_ENABLED=true to enable them."

    if "OPENAI_API_KEY" not in os.environ:
        return "Error: OPENAI_API_KEY environment variable not set. Cannot use agent tools."

    # Build comprehensive tool catalog for the AI
    tools_catalog = []
    for tool in AEROSPACE_TOOLS:
        tools_catalog.append(
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
                "examples": tool.examples,
            }
        )

    system_prompt = f"""You are an aerospace engineering assistant specialized in helping users select the right tools for their aerospace calculations and analysis tasks.

Available Aerospace Tools:
{json.dumps(tools_catalog, indent=2)}

User Task: {user_task}
User Context: {user_context}

Your job is to:
1. Analyze the user's task and context
2. Recommend the most appropriate tool(s) from the available aerospace tools
3. Provide clear guidance on how to use the recommended tool(s)
4. If multiple tools are needed, explain the workflow and order of operations
5. Highlight any prerequisites, limitations, or important considerations

Respond in a clear, structured format with:
- PRIMARY_TOOL: The main tool to use
- SECONDARY_TOOLS: Any additional tools needed (if applicable)
- WORKFLOW: Step-by-step guidance
- CONSIDERATIONS: Important notes, limitations, or prerequisites
- EXAMPLE: A concrete example of how to use the tool(s) for this task

If the user's task cannot be accomplished with the available tools, clearly explain what's missing and suggest alternatives."""

    try:
        # Call GPT-5-Medium via LiteLLM
        response = litellm.completion(
            model="gpt-5-medium",
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"Help me with this task: {user_task}\n\nContext: {user_context}",
                },
            ],
            temperature=0.2,
            max_tokens=1500,
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"Error: Failed to select appropriate tool: {str(e)}"
