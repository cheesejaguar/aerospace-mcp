"""MCP Server implementation for Aerospace flight planning tools."""

import asyncio
import json
import logging
import math
from dataclasses import asdict

import mcp.server.sse
import mcp.server.stdio
from mcp.server import Server
from mcp.types import (
    EmbeddedResource,
    ImageContent,
    TextContent,
    Tool,
)
from pydantic import ValidationError

from .core import (
    NM_PER_KM,
    OPENAP_AVAILABLE,
    AirportResolutionError,
    OpenAPError,
    PlanRequest,
    _airport_from_iata,
    _find_city_airports,
    _resolve_endpoint,
    estimates_openap,
    great_circle_points,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize MCP server
server = Server("aerospace-mcp")


# Tool definitions
TOOLS = [
    Tool(
        name="search_airports",
        description="Search for airports by IATA code or city name",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "IATA code (e.g., 'SJC') or city name (e.g., 'San Jose')"
                },
                "country": {
                    "type": "string",
                    "description": "Optional ISO country code to filter by (e.g., 'US', 'JP')"
                },
                "query_type": {
                    "type": "string",
                    "enum": ["iata", "city", "auto"],
                    "description": "Type of query - 'iata' for IATA codes, 'city' for city names, 'auto' to detect",
                    "default": "auto"
                }
            },
            "required": ["query"]
        }
    ),

    Tool(
        name="plan_flight",
        description="Plan a flight route between two airports with performance estimates",
        inputSchema={
            "type": "object",
            "properties": {
                "departure": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "Departure city name"},
                        "country": {"type": "string", "description": "Departure country code (optional)"},
                        "iata": {"type": "string", "description": "Preferred departure IATA code (optional)"}
                    },
                    "required": ["city"]
                },
                "arrival": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "Arrival city name"},
                        "country": {"type": "string", "description": "Arrival country code (optional)"},
                        "iata": {"type": "string", "description": "Preferred arrival IATA code (optional)"}
                    },
                    "required": ["city"]
                },
                "aircraft": {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string", "description": "ICAO aircraft type (e.g., 'A320', 'B738', 'A359')"},
                        "cruise_altitude": {"type": "integer", "description": "Cruise altitude in feet", "minimum": 8000, "maximum": 45000, "default": 35000},
                        "mass_kg": {"type": "number", "description": "Aircraft mass in kg (optional, uses 85% MTOW if not specified)"}
                    },
                    "required": ["type"]
                },
                "route_options": {
                    "type": "object",
                    "properties": {
                        "step_km": {"type": "number", "description": "Distance between polyline points in km", "minimum": 1.0, "default": 25.0}
                    }
                }
            },
            "required": ["departure", "arrival", "aircraft"]
        }
    ),

    Tool(
        name="calculate_distance",
        description="Calculate great circle distance between two points",
        inputSchema={
            "type": "object",
            "properties": {
                "origin": {
                    "type": "object",
                    "properties": {
                        "latitude": {"type": "number", "minimum": -90, "maximum": 90},
                        "longitude": {"type": "number", "minimum": -180, "maximum": 180}
                    },
                    "required": ["latitude", "longitude"]
                },
                "destination": {
                    "type": "object",
                    "properties": {
                        "latitude": {"type": "number", "minimum": -90, "maximum": 90},
                        "longitude": {"type": "number", "minimum": -180, "maximum": 180}
                    },
                    "required": ["latitude", "longitude"]
                },
                "step_km": {
                    "type": "number",
                    "description": "Step size for polyline generation in km",
                    "minimum": 1.0,
                    "default": 50.0
                }
            },
            "required": ["origin", "destination"]
        }
    ),

    Tool(
        name="get_aircraft_performance",
        description="Get performance estimates for an aircraft type (requires OpenAP)",
        inputSchema={
            "type": "object",
            "properties": {
                "aircraft_type": {
                    "type": "string",
                    "description": "ICAO aircraft type code (e.g., 'A320', 'B738')"
                },
                "distance_km": {
                    "type": "number",
                    "description": "Route distance in kilometers",
                    "minimum": 1.0
                },
                "cruise_altitude": {
                    "type": "integer",
                    "description": "Cruise altitude in feet",
                    "minimum": 8000,
                    "maximum": 45000,
                    "default": 35000
                },
                "mass_kg": {
                    "type": "number",
                    "description": "Aircraft mass in kg (optional)"
                }
            },
            "required": ["aircraft_type", "distance_km"]
        }
    ),

    Tool(
        name="get_system_status",
        description="Get system status and capabilities",
        inputSchema={
            "type": "object",
            "properties": {},
            "additionalProperties": False
        }
    ),

    Tool(
        name="get_atmosphere_profile",
        description="Get atmospheric properties (pressure, temperature, density) at specified altitudes using ISA model",
        inputSchema={
            "type": "object",
            "properties": {
                "altitudes_m": {
                    "type": "array",
                    "items": {"type": "number", "minimum": 0, "maximum": 86000},
                    "description": "List of altitudes in meters (0-86000m)",
                    "minItems": 1
                },
                "model_type": {
                    "type": "string",
                    "enum": ["ISA", "COESA"],
                    "default": "ISA",
                    "description": "Atmosphere model type"
                }
            },
            "required": ["altitudes_m"],
            "additionalProperties": False
        }
    ),

    Tool(
        name="wind_model_simple",
        description="Calculate wind speeds at different altitudes using logarithmic or power law models",
        inputSchema={
            "type": "object",
            "properties": {
                "altitudes_m": {
                    "type": "array",
                    "items": {"type": "number", "minimum": 0},
                    "description": "List of altitudes in meters",
                    "minItems": 1
                },
                "surface_wind_mps": {
                    "type": "number",
                    "minimum": 0,
                    "description": "Wind speed at reference height (m/s)"
                },
                "surface_altitude_m": {
                    "type": "number",
                    "default": 0.0,
                    "description": "Surface elevation in meters"
                },
                "model": {
                    "type": "string",
                    "enum": ["logarithmic", "power"],
                    "default": "logarithmic",
                    "description": "Wind profile model"
                },
                "roughness_length_m": {
                    "type": "number",
                    "minimum": 0.001,
                    "maximum": 10.0,
                    "default": 0.1,
                    "description": "Surface roughness length for logarithmic model"
                }
            },
            "required": ["altitudes_m", "surface_wind_mps"],
            "additionalProperties": False
        }
    ),

    Tool(
        name="transform_frames",
        description="Transform coordinates between reference frames (ECEF, ECI, ITRF, GCRS, GEODETIC)",
        inputSchema={
            "type": "object",
            "properties": {
                "xyz": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 3,
                    "maxItems": 3,
                    "description": "Coordinates [x, y, z] in meters (or lat, lon, alt for GEODETIC)"
                },
                "from_frame": {
                    "type": "string",
                    "enum": ["ECEF", "ECI", "ITRF", "GCRS", "GEODETIC"],
                    "description": "Source coordinate frame"
                },
                "to_frame": {
                    "type": "string",
                    "enum": ["ECEF", "ECI", "ITRF", "GCRS", "GEODETIC"],
                    "description": "Target coordinate frame"
                },
                "epoch_iso": {
                    "type": "string",
                    "default": "2000-01-01T12:00:00",
                    "description": "Reference epoch in ISO format"
                }
            },
            "required": ["xyz", "from_frame", "to_frame"],
            "additionalProperties": False
        }
    ),

    Tool(
        name="geodetic_to_ecef",
        description="Convert geodetic coordinates (lat/lon/alt) to Earth-centered Earth-fixed (ECEF) coordinates",
        inputSchema={
            "type": "object",
            "properties": {
                "latitude_deg": {
                    "type": "number",
                    "minimum": -90,
                    "maximum": 90,
                    "description": "Latitude in degrees"
                },
                "longitude_deg": {
                    "type": "number",
                    "minimum": -180,
                    "maximum": 180,
                    "description": "Longitude in degrees"
                },
                "altitude_m": {
                    "type": "number",
                    "description": "Height above WGS84 ellipsoid in meters"
                }
            },
            "required": ["latitude_deg", "longitude_deg", "altitude_m"],
            "additionalProperties": False
        }
    ),

    Tool(
        name="ecef_to_geodetic",
        description="Convert ECEF coordinates to geodetic (lat/lon/alt) coordinates",
        inputSchema={
            "type": "object",
            "properties": {
                "x": {"type": "number", "description": "ECEF X coordinate in meters"},
                "y": {"type": "number", "description": "ECEF Y coordinate in meters"},
                "z": {"type": "number", "description": "ECEF Z coordinate in meters"}
            },
            "required": ["x", "y", "z"],
            "additionalProperties": False
        }
    ),

    Tool(
        name="wing_vlm_analysis",
        description="Analyze wing aerodynamics using Vortex Lattice Method or simplified lifting line theory",
        inputSchema={
            "type": "object",
            "properties": {
                "geometry": {
                    "type": "object",
                    "properties": {
                        "span_m": {"type": "number", "minimum": 0.1, "maximum": 100},
                        "chord_root_m": {"type": "number", "minimum": 0.05, "maximum": 10},
                        "chord_tip_m": {"type": "number", "minimum": 0.05, "maximum": 10},
                        "sweep_deg": {"type": "number", "minimum": -45, "maximum": 45, "default": 0},
                        "dihedral_deg": {"type": "number", "minimum": -15, "maximum": 15, "default": 0},
                        "twist_deg": {"type": "number", "minimum": -10, "maximum": 10, "default": 0},
                        "airfoil_root": {"type": "string", "default": "NACA2412"},
                        "airfoil_tip": {"type": "string", "default": "NACA2412"}
                    },
                    "required": ["span_m", "chord_root_m", "chord_tip_m"],
                    "description": "Wing planform geometry"
                },
                "alpha_deg_list": {
                    "type": "array",
                    "items": {"type": "number", "minimum": -30, "maximum": 30},
                    "minItems": 1,
                    "description": "List of angles of attack to analyze (degrees)"
                },
                "mach": {
                    "type": "number",
                    "minimum": 0.05,
                    "maximum": 0.8,
                    "default": 0.2,
                    "description": "Mach number"
                }
            },
            "required": ["geometry", "alpha_deg_list"],
            "additionalProperties": False
        }
    ),

    Tool(
        name="airfoil_polar_analysis",
        description="Generate airfoil polar data (CL, CD, CM vs alpha) using database or advanced methods",
        inputSchema={
            "type": "object",
            "properties": {
                "airfoil_name": {
                    "type": "string",
                    "description": "Airfoil name (e.g., NACA2412, NACA0012, CLARKY)"
                },
                "alpha_deg_list": {
                    "type": "array",
                    "items": {"type": "number", "minimum": -25, "maximum": 25},
                    "minItems": 1,
                    "description": "Angles of attack to analyze (degrees)"
                },
                "reynolds": {
                    "type": "number",
                    "minimum": 50000,
                    "maximum": 10000000,
                    "default": 1000000,
                    "description": "Reynolds number"
                },
                "mach": {
                    "type": "number",
                    "minimum": 0.05,
                    "maximum": 0.7,
                    "default": 0.1,
                    "description": "Mach number"
                }
            },
            "required": ["airfoil_name", "alpha_deg_list"],
            "additionalProperties": False
        }
    ),

    Tool(
        name="calculate_stability_derivatives",
        description="Calculate basic longitudinal stability derivatives for a wing",
        inputSchema={
            "type": "object",
            "properties": {
                "geometry": {
                    "type": "object",
                    "properties": {
                        "span_m": {"type": "number", "minimum": 0.1, "maximum": 100},
                        "chord_root_m": {"type": "number", "minimum": 0.05, "maximum": 10},
                        "chord_tip_m": {"type": "number", "minimum": 0.05, "maximum": 10},
                        "sweep_deg": {"type": "number", "minimum": -45, "maximum": 45, "default": 0},
                        "airfoil_root": {"type": "string", "default": "NACA2412"}
                    },
                    "required": ["span_m", "chord_root_m", "chord_tip_m"],
                    "description": "Wing planform geometry"
                },
                "alpha_deg": {
                    "type": "number",
                    "minimum": -10,
                    "maximum": 10,
                    "default": 2.0,
                    "description": "Reference angle of attack"
                },
                "mach": {
                    "type": "number",
                    "minimum": 0.05,
                    "maximum": 0.8,
                    "default": 0.2,
                    "description": "Mach number"
                }
            },
            "required": ["geometry"],
            "additionalProperties": False
        }
    ),

    Tool(
        name="propeller_bemt_analysis",
        description="Analyze propeller performance using Blade Element Momentum Theory",
        inputSchema={
            "type": "object",
            "properties": {
                "geometry": {
                    "type": "object",
                    "properties": {
                        "diameter_m": {"type": "number", "minimum": 0.05, "maximum": 5.0},
                        "pitch_m": {"type": "number", "minimum": 0.02, "maximum": 3.0},
                        "num_blades": {"type": "integer", "minimum": 2, "maximum": 6},
                        "activity_factor": {"type": "number", "minimum": 50, "maximum": 200, "default": 100},
                        "cl_design": {"type": "number", "minimum": 0.1, "maximum": 1.5, "default": 0.5},
                        "cd_design": {"type": "number", "minimum": 0.005, "maximum": 0.1, "default": 0.02}
                    },
                    "required": ["diameter_m", "pitch_m", "num_blades"],
                    "description": "Propeller geometry parameters"
                },
                "rpm_list": {
                    "type": "array",
                    "items": {"type": "number", "minimum": 100, "maximum": 15000},
                    "minItems": 1,
                    "description": "List of RPM values to analyze"
                },
                "velocity_ms": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 100,
                    "default": 0,
                    "description": "Forward velocity in m/s (0 for static thrust)"
                },
                "altitude_m": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 15000,
                    "default": 0,
                    "description": "Altitude in meters"
                }
            },
            "required": ["geometry", "rpm_list"],
            "additionalProperties": False
        }
    ),

    Tool(
        name="uav_energy_estimate",
        description="Estimate UAV flight time and energy consumption for mission planning",
        inputSchema={
            "type": "object",
            "properties": {
                "uav_config": {
                    "type": "object",
                    "properties": {
                        "mass_kg": {"type": "number", "minimum": 0.1, "maximum": 100},
                        "wing_area_m2": {"type": "number", "minimum": 0, "maximum": 50},
                        "disk_area_m2": {"type": "number", "minimum": 0, "maximum": 10},
                        "cd0": {"type": "number", "minimum": 0.01, "maximum": 0.2, "default": 0.03},
                        "cl_cruise": {"type": "number", "minimum": 0.1, "maximum": 2.0},
                        "num_motors": {"type": "integer", "minimum": 1, "maximum": 8, "default": 1},
                        "motor_efficiency": {"type": "number", "minimum": 0.5, "maximum": 1.0, "default": 0.85},
                        "esc_efficiency": {"type": "number", "minimum": 0.8, "maximum": 1.0, "default": 0.95}
                    },
                    "required": ["mass_kg"],
                    "description": "UAV configuration parameters"
                },
                "battery_config": {
                    "type": "object",
                    "properties": {
                        "capacity_ah": {"type": "number", "minimum": 0.1, "maximum": 50},
                        "voltage_nominal_v": {"type": "number", "minimum": 3, "maximum": 50},
                        "mass_kg": {"type": "number", "minimum": 0.01, "maximum": 20},
                        "energy_density_wh_kg": {"type": "number", "minimum": 50, "maximum": 300, "default": 150},
                        "discharge_efficiency": {"type": "number", "minimum": 0.8, "maximum": 1.0, "default": 0.95}
                    },
                    "required": ["capacity_ah", "voltage_nominal_v", "mass_kg"],
                    "description": "Battery configuration"
                },
                "mission_profile": {
                    "type": "object",
                    "properties": {
                        "velocity_ms": {"type": "number", "minimum": 1, "maximum": 50, "default": 15},
                        "altitude_m": {"type": "number", "minimum": 0, "maximum": 5000, "default": 100}
                    },
                    "description": "Mission parameters"
                }
            },
            "required": ["uav_config", "battery_config"],
            "additionalProperties": False
        }
    ),

    Tool(
        name="get_airfoil_database",
        description="Get available airfoil database with aerodynamic coefficients",
        inputSchema={
            "type": "object",
            "properties": {},
            "additionalProperties": False
        }
    ),

    Tool(
        name="get_propeller_database",
        description="Get available propeller database with geometric and performance data",
        inputSchema={
            "type": "object",
            "properties": {},
            "additionalProperties": False
        }
    ),

    Tool(
        name="rocket_3dof_trajectory",
        description="Calculate 3DOF rocket trajectory using numerical integration",
        inputSchema={
            "type": "object",
            "properties": {
                "geometry": {
                    "type": "object",
                    "properties": {
                        "dry_mass_kg": {"type": "number", "minimum": 0.1, "maximum": 100000},
                        "propellant_mass_kg": {"type": "number", "minimum": 0.1, "maximum": 500000},
                        "diameter_m": {"type": "number", "minimum": 0.01, "maximum": 10},
                        "length_m": {"type": "number", "minimum": 0.1, "maximum": 100},
                        "cd": {"type": "number", "minimum": 0.1, "maximum": 2.0, "default": 0.3},
                        "thrust_curve": {
                            "type": "array",
                            "items": {
                                "type": "array",
                                "items": {"type": "number"},
                                "minItems": 2,
                                "maxItems": 2
                            },
                            "description": "Array of [time_s, thrust_N] points"
                        }
                    },
                    "required": ["dry_mass_kg", "propellant_mass_kg", "diameter_m", "length_m"],
                    "description": "Rocket geometry and mass properties"
                },
                "dt_s": {
                    "type": "number",
                    "minimum": 0.01,
                    "maximum": 1.0,
                    "default": 0.1,
                    "description": "Time step for integration (seconds)"
                },
                "max_time_s": {
                    "type": "number",
                    "minimum": 10,
                    "maximum": 1000,
                    "default": 300,
                    "description": "Maximum simulation time (seconds)"
                },
                "launch_angle_deg": {
                    "type": "number",
                    "minimum": 45,
                    "maximum": 90,
                    "default": 90,
                    "description": "Launch angle from horizontal (degrees)"
                }
            },
            "required": ["geometry"],
            "additionalProperties": False
        }
    ),

    Tool(
        name="estimate_rocket_sizing",
        description="Estimate rocket sizing requirements for target altitude and payload",
        inputSchema={
            "type": "object",
            "properties": {
                "target_altitude_m": {
                    "type": "number",
                    "minimum": 100,
                    "maximum": 100000,
                    "description": "Target apogee altitude in meters"
                },
                "payload_mass_kg": {
                    "type": "number",
                    "minimum": 0.01,
                    "maximum": 10000,
                    "description": "Payload mass in kg"
                },
                "propellant_type": {
                    "type": "string",
                    "enum": ["solid", "liquid"],
                    "default": "solid",
                    "description": "Propellant type"
                }
            },
            "required": ["target_altitude_m", "payload_mass_kg"],
            "additionalProperties": False
        }
    ),

    Tool(
        name="optimize_launch_angle",
        description="Optimize rocket launch angle for maximum altitude or range",
        inputSchema={
            "type": "object",
            "properties": {
                "geometry": {
                    "type": "object",
                    "properties": {
                        "dry_mass_kg": {"type": "number", "minimum": 0.1, "maximum": 100000},
                        "propellant_mass_kg": {"type": "number", "minimum": 0.1, "maximum": 500000},
                        "diameter_m": {"type": "number", "minimum": 0.01, "maximum": 10},
                        "length_m": {"type": "number", "minimum": 0.1, "maximum": 100},
                        "cd": {"type": "number", "minimum": 0.1, "maximum": 2.0, "default": 0.3},
                        "thrust_curve": {
                            "type": "array",
                            "items": {
                                "type": "array",
                                "items": {"type": "number"},
                                "minItems": 2,
                                "maxItems": 2
                            },
                            "description": "Array of [time_s, thrust_N] points"
                        }
                    },
                    "required": ["dry_mass_kg", "propellant_mass_kg", "diameter_m", "length_m"],
                    "description": "Rocket geometry and mass properties"
                },
                "objective": {
                    "type": "string",
                    "enum": ["max_altitude", "max_range"],
                    "default": "max_altitude",
                    "description": "Optimization objective"
                },
                "angle_bounds": {
                    "type": "array",
                    "items": {"type": "number", "minimum": 45, "maximum": 90},
                    "minItems": 2,
                    "maxItems": 2,
                    "default": [80, 90],
                    "description": "Launch angle bounds [min_deg, max_deg]"
                }
            },
            "required": ["geometry"],
            "additionalProperties": False
        }
    ),

    Tool(
        name="optimize_thrust_profile",
        description="Optimize rocket thrust profile for better performance using trajectory optimization",
        inputSchema={
            "type": "object",
            "properties": {
                "geometry": {
                    "type": "object",
                    "properties": {
                        "dry_mass_kg": {"type": "number", "minimum": 0.1, "maximum": 100000},
                        "propellant_mass_kg": {"type": "number", "minimum": 0.1, "maximum": 500000},
                        "diameter_m": {"type": "number", "minimum": 0.01, "maximum": 10},
                        "length_m": {"type": "number", "minimum": 0.1, "maximum": 100},
                        "cd": {"type": "number", "minimum": 0.1, "maximum": 2.0, "default": 0.3}
                    },
                    "required": ["dry_mass_kg", "propellant_mass_kg", "diameter_m", "length_m"],
                    "description": "Base rocket geometry (thrust curve will be optimized)"
                },
                "burn_time_s": {
                    "type": "number",
                    "minimum": 1,
                    "maximum": 300,
                    "description": "Total burn time in seconds"
                },
                "total_impulse_target": {
                    "type": "number",
                    "minimum": 100,
                    "maximum": 10000000,
                    "description": "Target total impulse (N·s)"
                },
                "n_segments": {
                    "type": "integer",
                    "minimum": 3,
                    "maximum": 10,
                    "default": 5,
                    "description": "Number of thrust profile segments"
                },
                "objective": {
                    "type": "string",
                    "enum": ["max_altitude", "min_max_q", "min_gravity_loss"],
                    "default": "max_altitude",
                    "description": "Optimization objective"
                }
            },
            "required": ["geometry", "burn_time_s", "total_impulse_target"],
            "additionalProperties": False
        }
    ),

    Tool(
        name="trajectory_sensitivity_analysis",
        description="Perform sensitivity analysis on rocket trajectory parameters",
        inputSchema={
            "type": "object",
            "properties": {
                "base_geometry": {
                    "type": "object",
                    "properties": {
                        "dry_mass_kg": {"type": "number", "minimum": 0.1, "maximum": 100000},
                        "propellant_mass_kg": {"type": "number", "minimum": 0.1, "maximum": 500000},
                        "diameter_m": {"type": "number", "minimum": 0.01, "maximum": 10},
                        "length_m": {"type": "number", "minimum": 0.1, "maximum": 100},
                        "cd": {"type": "number", "minimum": 0.1, "maximum": 2.0, "default": 0.3},
                        "thrust_curve": {
                            "type": "array",
                            "items": {
                                "type": "array",
                                "items": {"type": "number"},
                                "minItems": 2,
                                "maxItems": 2
                            },
                            "description": "Array of [time_s, thrust_N] points"
                        }
                    },
                    "required": ["dry_mass_kg", "propellant_mass_kg", "diameter_m", "length_m"],
                    "description": "Baseline rocket geometry"
                },
                "parameter_variations": {
                    "type": "object",
                    "description": "Parameters to vary with their variation ranges",
                    "additionalProperties": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 3
                    }
                },
                "objective": {
                    "type": "string",
                    "enum": ["max_altitude", "max_velocity", "specific_impulse"],
                    "default": "max_altitude",
                    "description": "Objective metric for sensitivity analysis"
                }
            },
            "required": ["base_geometry", "parameter_variations"],
            "additionalProperties": False
        }
    ),

    Tool(
        name="elements_to_state_vector",
        description="Convert orbital elements to state vector in J2000 frame",
        inputSchema={
            "type": "object",
            "properties": {
                "elements": {
                    "type": "object",
                    "properties": {
                        "semi_major_axis_m": {"type": "number", "minimum": 6.6e6, "maximum": 1e12},
                        "eccentricity": {"type": "number", "minimum": 0.0, "maximum": 0.99},
                        "inclination_deg": {"type": "number", "minimum": 0.0, "maximum": 180.0},
                        "raan_deg": {"type": "number", "minimum": 0.0, "maximum": 360.0},
                        "arg_periapsis_deg": {"type": "number", "minimum": 0.0, "maximum": 360.0},
                        "true_anomaly_deg": {"type": "number", "minimum": 0.0, "maximum": 360.0},
                        "epoch_utc": {"type": "string", "description": "Epoch in UTC ISO format"}
                    },
                    "required": ["semi_major_axis_m", "eccentricity", "inclination_deg", "raan_deg", "arg_periapsis_deg", "true_anomaly_deg", "epoch_utc"],
                    "description": "Classical orbital elements"
                }
            },
            "required": ["elements"],
            "additionalProperties": False
        }
    ),

    Tool(
        name="state_vector_to_elements",
        description="Convert state vector to classical orbital elements",
        inputSchema={
            "type": "object",
            "properties": {
                "state": {
                    "type": "object",
                    "properties": {
                        "position_m": {
                            "type": "array",
                            "items": {"type": "number"},
                            "minItems": 3,
                            "maxItems": 3,
                            "description": "Position vector [x, y, z] in meters"
                        },
                        "velocity_ms": {
                            "type": "array",
                            "items": {"type": "number"},
                            "minItems": 3,
                            "maxItems": 3,
                            "description": "Velocity vector [vx, vy, vz] in m/s"
                        },
                        "epoch_utc": {"type": "string", "description": "Epoch in UTC ISO format"},
                        "frame": {"type": "string", "default": "J2000", "description": "Reference frame"}
                    },
                    "required": ["position_m", "velocity_ms", "epoch_utc"],
                    "description": "Spacecraft state vector"
                }
            },
            "required": ["state"],
            "additionalProperties": False
        }
    ),

    Tool(
        name="propagate_orbit_j2",
        description="Propagate orbit with J2 perturbations using numerical integration",
        inputSchema={
            "type": "object",
            "properties": {
                "initial_state": {
                    "type": "object",
                    "properties": {
                        "position_m": {
                            "type": "array",
                            "items": {"type": "number"},
                            "minItems": 3,
                            "maxItems": 3,
                            "description": "Position vector [x, y, z] in meters"
                        },
                        "velocity_ms": {
                            "type": "array",
                            "items": {"type": "number"},
                            "minItems": 3,
                            "maxItems": 3,
                            "description": "Velocity vector [vx, vy, vz] in m/s"
                        },
                        "epoch_utc": {"type": "string", "description": "Epoch in UTC ISO format"},
                        "frame": {"type": "string", "default": "J2000", "description": "Reference frame"}
                    },
                    "required": ["position_m", "velocity_ms", "epoch_utc"],
                    "description": "Initial spacecraft state"
                },
                "time_span_s": {
                    "type": "number",
                    "minimum": 60,
                    "maximum": 86400 * 30,
                    "description": "Propagation time span (seconds)"
                },
                "time_step_s": {
                    "type": "number",
                    "minimum": 1,
                    "maximum": 3600,
                    "default": 60,
                    "description": "Integration time step (seconds)"
                }
            },
            "required": ["initial_state", "time_span_s"],
            "additionalProperties": False
        }
    ),

    Tool(
        name="calculate_ground_track",
        description="Calculate ground track from orbital state vectors",
        inputSchema={
            "type": "object",
            "properties": {
                "orbit_states": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "position_m": {
                                "type": "array",
                                "items": {"type": "number"},
                                "minItems": 3,
                                "maxItems": 3
                            },
                            "velocity_ms": {
                                "type": "array",
                                "items": {"type": "number"},
                                "minItems": 3,
                                "maxItems": 3
                            },
                            "epoch_utc": {"type": "string"},
                            "frame": {"type": "string", "default": "J2000"}
                        },
                        "required": ["position_m", "velocity_ms", "epoch_utc"]
                    },
                    "minItems": 1,
                    "description": "Array of orbital state vectors"
                },
                "time_step_s": {
                    "type": "number",
                    "minimum": 1,
                    "maximum": 3600,
                    "default": 60,
                    "description": "Time step between states (seconds)"
                }
            },
            "required": ["orbit_states"],
            "additionalProperties": False
        }
    ),

    Tool(
        name="hohmann_transfer",
        description="Calculate Hohmann transfer orbit parameters between two circular orbits",
        inputSchema={
            "type": "object",
            "properties": {
                "r1_m": {
                    "type": "number",
                    "minimum": 6.6e6,
                    "maximum": 1e12,
                    "description": "Initial circular orbit radius from Earth center (m)"
                },
                "r2_m": {
                    "type": "number",
                    "minimum": 6.6e6,
                    "maximum": 1e12,
                    "description": "Final circular orbit radius from Earth center (m)"
                }
            },
            "required": ["r1_m", "r2_m"],
            "additionalProperties": False
        }
    ),

    Tool(
        name="orbital_rendezvous_planning",
        description="Plan orbital rendezvous maneuvers between two spacecraft",
        inputSchema={
            "type": "object",
            "properties": {
                "chaser_elements": {
                    "type": "object",
                    "properties": {
                        "semi_major_axis_m": {"type": "number", "minimum": 6.6e6, "maximum": 1e12},
                        "eccentricity": {"type": "number", "minimum": 0.0, "maximum": 0.99},
                        "inclination_deg": {"type": "number", "minimum": 0.0, "maximum": 180.0},
                        "raan_deg": {"type": "number", "minimum": 0.0, "maximum": 360.0},
                        "arg_periapsis_deg": {"type": "number", "minimum": 0.0, "maximum": 360.0},
                        "true_anomaly_deg": {"type": "number", "minimum": 0.0, "maximum": 360.0},
                        "epoch_utc": {"type": "string"}
                    },
                    "required": ["semi_major_axis_m", "eccentricity", "inclination_deg", "raan_deg", "arg_periapsis_deg", "true_anomaly_deg", "epoch_utc"],
                    "description": "Chaser spacecraft orbital elements"
                },
                "target_elements": {
                    "type": "object",
                    "properties": {
                        "semi_major_axis_m": {"type": "number", "minimum": 6.6e6, "maximum": 1e12},
                        "eccentricity": {"type": "number", "minimum": 0.0, "maximum": 0.99},
                        "inclination_deg": {"type": "number", "minimum": 0.0, "maximum": 180.0},
                        "raan_deg": {"type": "number", "minimum": 0.0, "maximum": 360.0},
                        "arg_periapsis_deg": {"type": "number", "minimum": 0.0, "maximum": 360.0},
                        "true_anomaly_deg": {"type": "number", "minimum": 0.0, "maximum": 360.0},
                        "epoch_utc": {"type": "string"}
                    },
                    "required": ["semi_major_axis_m", "eccentricity", "inclination_deg", "raan_deg", "arg_periapsis_deg", "true_anomaly_deg", "epoch_utc"],
                    "description": "Target spacecraft orbital elements"
                }
            },
            "required": ["chaser_elements", "target_elements"],
            "additionalProperties": False
        }
    ),

    Tool(
        name="genetic_algorithm_optimization",
        description="Optimize spacecraft trajectory using genetic algorithm",
        inputSchema={
            "type": "object",
            "properties": {
                "initial_trajectory": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "time_s": {"type": "number"},
                            "position_m": {
                                "type": "array",
                                "items": {"type": "number"},
                                "minItems": 3,
                                "maxItems": 3
                            },
                            "velocity_ms": {
                                "type": "array",
                                "items": {"type": "number"},
                                "minItems": 3,
                                "maxItems": 3
                            },
                            "thrust_n": {
                                "type": "array",
                                "items": {"type": "number"},
                                "minItems": 3,
                                "maxItems": 3
                            },
                            "mass_kg": {"type": "number", "minimum": 100, "maximum": 10000}
                        },
                        "required": ["time_s", "position_m", "velocity_ms", "mass_kg"]
                    },
                    "minItems": 2,
                    "maxItems": 20,
                    "description": "Initial trajectory waypoints"
                },
                "objective": {
                    "type": "object",
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": ["minimize_fuel", "minimize_time", "minimize_delta_v", "maximize_payload"]
                        },
                        "target_state": {
                            "type": "array",
                            "items": {"type": "number"},
                            "minItems": 6,
                            "maxItems": 6,
                            "description": "Target state [x, y, z, vx, vy, vz] if applicable"
                        }
                    },
                    "required": ["type"],
                    "description": "Optimization objective"
                },
                "constraints": {
                    "type": "object",
                    "properties": {
                        "max_thrust_n": {"type": "number", "minimum": 1, "maximum": 100000, "default": 10000},
                        "max_acceleration_ms2": {"type": "number", "minimum": 0.1, "maximum": 100, "default": 50},
                        "min_altitude_m": {"type": "number", "minimum": 150000, "maximum": 1000000, "default": 200000},
                        "max_delta_v_ms": {"type": "number", "minimum": 100, "maximum": 15000, "default": 5000}
                    },
                    "description": "Optimization constraints"
                },
                "ga_params": {
                    "type": "object",
                    "properties": {
                        "population_size": {"type": "integer", "minimum": 10, "maximum": 200, "default": 50},
                        "generations": {"type": "integer", "minimum": 10, "maximum": 500, "default": 100},
                        "mutation_rate": {"type": "number", "minimum": 0.01, "maximum": 0.5, "default": 0.1},
                        "crossover_rate": {"type": "number", "minimum": 0.1, "maximum": 1.0, "default": 0.8}
                    },
                    "description": "Genetic algorithm parameters"
                }
            },
            "required": ["initial_trajectory", "objective"],
            "additionalProperties": False
        }
    ),

    Tool(
        name="particle_swarm_optimization",
        description="Optimize spacecraft trajectory using particle swarm optimization",
        inputSchema={
            "type": "object",
            "properties": {
                "initial_trajectory": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "time_s": {"type": "number"},
                            "position_m": {
                                "type": "array",
                                "items": {"type": "number"},
                                "minItems": 3,
                                "maxItems": 3
                            },
                            "velocity_ms": {
                                "type": "array",
                                "items": {"type": "number"},
                                "minItems": 3,
                                "maxItems": 3
                            },
                            "thrust_n": {
                                "type": "array",
                                "items": {"type": "number"},
                                "minItems": 3,
                                "maxItems": 3
                            },
                            "mass_kg": {"type": "number", "minimum": 100, "maximum": 10000}
                        },
                        "required": ["time_s", "position_m", "velocity_ms", "mass_kg"]
                    },
                    "minItems": 2,
                    "maxItems": 20,
                    "description": "Initial trajectory waypoints"
                },
                "objective": {
                    "type": "object",
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": ["minimize_fuel", "minimize_time", "minimize_delta_v", "maximize_payload"]
                        },
                        "target_state": {
                            "type": "array",
                            "items": {"type": "number"},
                            "minItems": 6,
                            "maxItems": 6,
                            "description": "Target state [x, y, z, vx, vy, vz] if applicable"
                        }
                    },
                    "required": ["type"],
                    "description": "Optimization objective"
                },
                "constraints": {
                    "type": "object",
                    "properties": {
                        "max_thrust_n": {"type": "number", "minimum": 1, "maximum": 100000, "default": 10000},
                        "max_acceleration_ms2": {"type": "number", "minimum": 0.1, "maximum": 100, "default": 50},
                        "min_altitude_m": {"type": "number", "minimum": 150000, "maximum": 1000000, "default": 200000},
                        "max_delta_v_ms": {"type": "number", "minimum": 100, "maximum": 15000, "default": 5000}
                    },
                    "description": "Optimization constraints"
                },
                "pso_params": {
                    "type": "object",
                    "properties": {
                        "num_particles": {"type": "integer", "minimum": 10, "maximum": 100, "default": 30},
                        "max_iterations": {"type": "integer", "minimum": 10, "maximum": 300, "default": 100},
                        "w": {"type": "number", "minimum": 0.1, "maximum": 1.0, "default": 0.7},
                        "c1": {"type": "number", "minimum": 0.5, "maximum": 3.0, "default": 1.5},
                        "c2": {"type": "number", "minimum": 0.5, "maximum": 3.0, "default": 1.5}
                    },
                    "description": "Particle swarm optimization parameters"
                }
            },
            "required": ["initial_trajectory", "objective"],
            "additionalProperties": False
        }
    ),

    Tool(
        name="porkchop_plot_analysis",
        description="Generate porkchop plot for interplanetary transfer opportunities",
        inputSchema={
            "type": "object",
            "properties": {
                "departure_body": {
                    "type": "string",
                    "enum": ["Earth", "Mars", "Venus", "Jupiter"],
                    "default": "Earth",
                    "description": "Departure celestial body"
                },
                "arrival_body": {
                    "type": "string",
                    "enum": ["Earth", "Mars", "Venus", "Jupiter"],
                    "default": "Mars",
                    "description": "Arrival celestial body"
                },
                "departure_dates": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of departure dates (ISO format)"
                },
                "arrival_dates": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of arrival dates (ISO format)"
                },
                "min_tof_days": {
                    "type": "integer",
                    "minimum": 30,
                    "maximum": 1000,
                    "default": 100,
                    "description": "Minimum time of flight in days"
                },
                "max_tof_days": {
                    "type": "integer",
                    "minimum": 50,
                    "maximum": 2000,
                    "default": 400,
                    "description": "Maximum time of flight in days"
                }
            },
            "required": []
        }
    ),
    Tool(
        name="monte_carlo_uncertainty_analysis",
        description="Perform Monte Carlo uncertainty analysis on spacecraft trajectory",
        inputSchema={
            "type": "object",
            "properties": {
                "nominal_trajectory": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "time_s": {"type": "number"},
                            "position_m": {
                                "type": "array",
                                "items": {"type": "number"},
                                "minItems": 3,
                                "maxItems": 3
                            },
                            "velocity_ms": {
                                "type": "array",
                                "items": {"type": "number"},
                                "minItems": 3,
                                "maxItems": 3
                            },
                            "thrust_n": {
                                "type": "array",
                                "items": {"type": "number"},
                                "minItems": 3,
                                "maxItems": 3
                            },
                            "mass_kg": {"type": "number"}
                        },
                        "required": ["time_s", "position_m", "velocity_ms", "mass_kg"]
                    },
                    "minItems": 2,
                    "description": "Nominal trajectory waypoints"
                },
                "uncertainty_params": {
                    "type": "object",
                    "properties": {
                        "position_m": {
                            "type": "object",
                            "properties": {
                                "std": {"type": "number", "minimum": 1, "maximum": 10000, "default": 100}
                            },
                            "description": "Position uncertainty parameters"
                        },
                        "velocity_ms": {
                            "type": "object",
                            "properties": {
                                "std": {"type": "number", "minimum": 0.1, "maximum": 100, "default": 10}
                            },
                            "description": "Velocity uncertainty parameters"
                        },
                        "thrust_n": {
                            "type": "object",
                            "properties": {
                                "std": {"type": "number", "minimum": 1, "maximum": 1000, "default": 50}
                            },
                            "description": "Thrust uncertainty parameters"
                        }
                    },
                    "description": "Uncertainty parameter definitions"
                },
                "n_samples": {
                    "type": "integer",
                    "minimum": 100,
                    "maximum": 10000,
                    "default": 1000,
                    "description": "Number of Monte Carlo samples"
                }
            },
            "required": ["nominal_trajectory"],
            "additionalProperties": False
        }
    )
]


@server.list_tools()
async def handle_list_tools() -> list[Tool]:
    """List all available tools."""
    return TOOLS


@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[TextContent | ImageContent | EmbeddedResource]:
    """Handle tool calls."""
    try:
        if name == "search_airports":
            return await _handle_search_airports(arguments)
        elif name == "plan_flight":
            return await _handle_plan_flight(arguments)
        elif name == "calculate_distance":
            return await _handle_calculate_distance(arguments)
        elif name == "get_aircraft_performance":
            return await _handle_get_aircraft_performance(arguments)
        elif name == "get_system_status":
            return await _handle_get_system_status(arguments)
        elif name == "get_atmosphere_profile":
            return await _handle_get_atmosphere_profile(arguments)
        elif name == "wind_model_simple":
            return await _handle_wind_model_simple(arguments)
        elif name == "transform_frames":
            return await _handle_transform_frames(arguments)
        elif name == "geodetic_to_ecef":
            return await _handle_geodetic_to_ecef(arguments)
        elif name == "ecef_to_geodetic":
            return await _handle_ecef_to_geodetic(arguments)
        elif name == "wing_vlm_analysis":
            return await _handle_wing_vlm_analysis(arguments)
        elif name == "airfoil_polar_analysis":
            return await _handle_airfoil_polar_analysis(arguments)
        elif name == "calculate_stability_derivatives":
            return await _handle_calculate_stability_derivatives(arguments)
        elif name == "propeller_bemt_analysis":
            return await _handle_propeller_bemt_analysis(arguments)
        elif name == "uav_energy_estimate":
            return await _handle_uav_energy_estimate(arguments)
        elif name == "get_airfoil_database":
            return await _handle_get_airfoil_database(arguments)
        elif name == "get_propeller_database":
            return await _handle_get_propeller_database(arguments)
        elif name == "rocket_3dof_trajectory":
            return await _handle_rocket_3dof_trajectory(arguments)
        elif name == "estimate_rocket_sizing":
            return await _handle_estimate_rocket_sizing(arguments)
        elif name == "optimize_launch_angle":
            return await _handle_optimize_launch_angle(arguments)
        elif name == "optimize_thrust_profile":
            return await _handle_optimize_thrust_profile(arguments)
        elif name == "trajectory_sensitivity_analysis":
            return await _handle_trajectory_sensitivity_analysis(arguments)
        elif name == "elements_to_state_vector":
            return await _handle_elements_to_state_vector(arguments)
        elif name == "state_vector_to_elements":
            return await _handle_state_vector_to_elements(arguments)
        elif name == "propagate_orbit_j2":
            return await _handle_propagate_orbit_j2(arguments)
        elif name == "calculate_ground_track":
            return await _handle_calculate_ground_track(arguments)
        elif name == "hohmann_transfer":
            return await _handle_hohmann_transfer(arguments)
        elif name == "orbital_rendezvous_planning":
            return await _handle_orbital_rendezvous_planning(arguments)
        elif name == "genetic_algorithm_optimization":
            return await _handle_genetic_algorithm_optimization(arguments)
        elif name == "particle_swarm_optimization":
            return await _handle_particle_swarm_optimization(arguments)
        elif name == "porkchop_plot_analysis":
            return await _handle_porkchop_plot_analysis(arguments)
        elif name == "monte_carlo_uncertainty_analysis":
            return await _handle_monte_carlo_uncertainty_analysis(arguments)
        else:
            raise ValueError(f"Unknown tool: {name}")

    except Exception as e:
        logger.error(f"Error in tool {name}: {str(e)}", exc_info=True)
        return [TextContent(type="text", text=f"Error: {str(e)}")]


async def _handle_search_airports(arguments: dict) -> list[TextContent]:
    """Handle airport search requests."""
    query = arguments.get("query", "").strip()
    country = arguments.get("country")
    query_type = arguments.get("query_type", "auto")

    if not query:
        return [TextContent(type="text", text="Error: Query parameter is required")]

    results = []

    # Auto-detect query type if needed
    if query_type == "auto":
        query_type = "iata" if len(query) == 3 and query.isalpha() else "city"

    try:
        if query_type == "iata":
            # Search by IATA code
            airport = _airport_from_iata(query)
            if airport:
                results = [airport]
        else:
            # Search by city name
            results = _find_city_airports(query, country)

        if not results:
            message = f"No airports found for {query_type} '{query}'"
            if country:
                message += f" in country '{country}'"
            return [TextContent(type="text", text=message)]

        # Format results
        response_lines = [f"Found {len(results)} airport(s):"]
        for airport in results:
            line = f"• {airport.iata} ({airport.icao}) - {airport.name}"
            line += f"\n  City: {airport.city}, {airport.country}"
            line += f"\n  Coordinates: {airport.lat:.4f}, {airport.lon:.4f}"
            if airport.tz:
                line += f"\n  Timezone: {airport.tz}"
            response_lines.append(line)

        return [TextContent(type="text", text="\n\n".join(response_lines))]

    except Exception as e:
        return [TextContent(type="text", text=f"Search error: {str(e)}")]


async def _handle_plan_flight(arguments: dict) -> list[TextContent]:
    """Handle flight planning requests."""
    try:
        # Extract parameters
        departure = arguments.get("departure", {})
        arrival = arguments.get("arrival", {})
        aircraft = arguments.get("aircraft", {})
        route_options = arguments.get("route_options", {})

        # Build PlanRequest
        plan_request = PlanRequest(
            depart_city=departure.get("city", ""),
            arrive_city=arrival.get("city", ""),
            depart_country=departure.get("country"),
            arrive_country=arrival.get("country"),
            prefer_depart_iata=departure.get("iata"),
            prefer_arrive_iata=arrival.get("iata"),
            ac_type=aircraft.get("type", ""),
            cruise_alt_ft=aircraft.get("cruise_altitude", 35000),
            mass_kg=aircraft.get("mass_kg"),
            route_step_km=route_options.get("step_km", 25.0)
        )

        # Validate same city check
        if (plan_request.depart_city.strip().lower() == plan_request.arrive_city.strip().lower()
            and not plan_request.prefer_arrive_iata and not plan_request.prefer_depart_iata):
            return [TextContent(type="text", text="Error: Departure and arrival cities are identical. Please specify airports explicitly.")]

        # Resolve airports
        try:
            dep = _resolve_endpoint(plan_request.depart_city, plan_request.depart_country,
                                    plan_request.prefer_depart_iata, "departure")
            arr = _resolve_endpoint(plan_request.arrive_city, plan_request.arrive_country,
                                    plan_request.prefer_arrive_iata, "arrival")
        except AirportResolutionError as e:
            return [TextContent(type="text", text=f"Airport resolution error: {str(e)}")]

        # Calculate route
        polyline, distance_km = great_circle_points(
            dep.lat, dep.lon, arr.lat, arr.lon, plan_request.route_step_km
        )
        distance_nm = distance_km * NM_PER_KM

        # Get performance estimates
        try:
            estimates, engine_name = estimates_openap(
                plan_request.ac_type,
                plan_request.cruise_alt_ft,
                plan_request.mass_kg,
                distance_km
            )
        except OpenAPError as e:
            return [TextContent(type="text", text=f"Performance estimation error: {str(e)}")]

        # Format response
        response_lines = [
            f"Flight Plan: {dep.iata} → {arr.iata}",
            f"Route: {dep.name} → {arr.name}",
            f"Distance: {distance_km:.0f} km ({distance_nm:.0f} NM)",
            f"Aircraft: {plan_request.ac_type}",
            f"Cruise Altitude: {plan_request.cruise_alt_ft:,} ft",
            "",
            "Performance Estimates (OpenAP):",
            f"• Block Time: {estimates['block']['time_min']:.0f} minutes ({estimates['block']['time_min']/60:.1f} hours)",
            f"• Block Fuel: {estimates['block']['fuel_kg']:.0f} kg",
            "",
            "Flight Segments:",
            f"• Climb: {estimates['climb']['time_min']:.0f} min, {estimates['climb']['distance_km']:.0f} km, {estimates['climb']['fuel_kg']:.0f} kg fuel",
            f"• Cruise: {estimates['cruise']['time_min']:.0f} min, {estimates['cruise']['distance_km']:.0f} km, {estimates['cruise']['fuel_kg']:.0f} kg fuel",
            f"• Descent: {estimates['descent']['time_min']:.0f} min, {estimates['descent']['distance_km']:.0f} km, {estimates['descent']['fuel_kg']:.0f} kg fuel",
            "",
            f"Route Polyline: {len(polyline)} points (every {plan_request.route_step_km} km)",
            f"Mass Assumption: {estimates['assumptions']['mass_kg']:.0f} kg"
        ]

        return [TextContent(type="text", text="\n".join(response_lines))]

    except ValidationError as e:
        return [TextContent(type="text", text=f"Validation error: {str(e)}")]
    except Exception as e:
        return [TextContent(type="text", text=f"Flight planning error: {str(e)}")]


async def _handle_calculate_distance(arguments: dict) -> list[TextContent]:
    """Handle distance calculation requests."""
    try:
        origin = arguments.get("origin", {})
        destination = arguments.get("destination", {})
        step_km = arguments.get("step_km", 50.0)

        lat1 = origin.get("latitude")
        lon1 = origin.get("longitude")
        lat2 = destination.get("latitude")
        lon2 = destination.get("longitude")

        if None in [lat1, lon1, lat2, lon2]:
            return [TextContent(type="text", text="Error: Origin and destination coordinates are required")]

        polyline, distance_km = great_circle_points(lat1, lon1, lat2, lon2, step_km)
        distance_nm = distance_km * NM_PER_KM

        response_lines = [
            "Great Circle Distance Calculation",
            f"Origin: {lat1:.4f}, {lon1:.4f}",
            f"Destination: {lat2:.4f}, {lon2:.4f}",
            f"Distance: {distance_km:.2f} km ({distance_nm:.2f} NM)",
            f"Polyline Points: {len(polyline)} (every {step_km} km)"
        ]

        return [TextContent(type="text", text="\n".join(response_lines))]

    except Exception as e:
        return [TextContent(type="text", text=f"Distance calculation error: {str(e)}")]


async def _handle_get_aircraft_performance(arguments: dict) -> list[TextContent]:
    """Handle aircraft performance requests."""
    try:
        aircraft_type = arguments.get("aircraft_type", "")
        distance_km = arguments.get("distance_km", 0)
        cruise_altitude = arguments.get("cruise_altitude", 35000)
        mass_kg = arguments.get("mass_kg")

        if not aircraft_type:
            return [TextContent(type="text", text="Error: Aircraft type is required")]

        if distance_km <= 0:
            return [TextContent(type="text", text="Error: Distance must be positive")]

        estimates, engine_name = estimates_openap(aircraft_type, cruise_altitude, mass_kg, distance_km)

        response_lines = [
            f"Aircraft Performance Estimates ({engine_name})",
            f"Aircraft: {aircraft_type}",
            f"Distance: {distance_km:.0f} km",
            f"Cruise Altitude: {cruise_altitude:,} ft",
            f"Mass: {estimates['assumptions']['mass_kg']:.0f} kg",
            "",
            "Block Estimates:",
            f"• Time: {estimates['block']['time_min']:.0f} minutes ({estimates['block']['time_min']/60:.1f} hours)",
            f"• Fuel: {estimates['block']['fuel_kg']:.0f} kg",
            "",
            "Segment Breakdown:",
            f"• Climb: {estimates['climb']['time_min']:.0f} min, {estimates['climb']['distance_km']:.0f} km, {estimates['climb']['fuel_kg']:.0f} kg",
            f"• Cruise: {estimates['cruise']['time_min']:.0f} min, {estimates['cruise']['distance_km']:.0f} km, {estimates['cruise']['fuel_kg']:.0f} kg",
            f"• Descent: {estimates['descent']['time_min']:.0f} min, {estimates['descent']['distance_km']:.0f} km, {estimates['descent']['fuel_kg']:.0f} kg"
        ]

        return [TextContent(type="text", text="\n".join(response_lines))]

    except OpenAPError as e:
        return [TextContent(type="text", text=f"Performance estimation error: {str(e)}")]
    except Exception as e:
        return [TextContent(type="text", text=f"Aircraft performance error: {str(e)}")]


async def _handle_get_system_status(arguments: dict) -> list[TextContent]:
    """Handle system status requests."""
    try:
        from .core import _AIRPORTS_IATA

        status_lines = [
            "Aerospace MCP Server Status",
            f"OpenAP Available: {'Yes' if OPENAP_AVAILABLE else 'No'}",
            f"Airports Loaded: {len(_AIRPORTS_IATA):,}",
            "",
            "Available Tools:",
            "• search_airports - Search for airports by IATA or city",
            "• plan_flight - Plan flight routes with performance estimates",
            "• calculate_distance - Calculate great circle distances",
            "• get_aircraft_performance - Get aircraft performance estimates",
            "• get_system_status - Get this status information"
        ]

        if not OPENAP_AVAILABLE:
            status_lines.extend([
                "",
                "Note: OpenAP is not available. Flight performance estimates will not work.",
                "Install with: pip install openap"
            ])

        return [TextContent(type="text", text="\n".join(status_lines))]

    except Exception as e:
        return [TextContent(type="text", text=f"Status error: {str(e)}")]


async def _handle_get_atmosphere_profile(arguments: dict) -> list[TextContent]:
    """Handle atmosphere profile requests."""
    try:
        from .integrations.atmosphere import get_atmosphere_profile

        altitudes_m = arguments.get("altitudes_m", [])
        model_type = arguments.get("model_type", "ISA")

        if not altitudes_m:
            return [TextContent(type="text", text="Error: altitudes_m is required")]

        profile = get_atmosphere_profile(altitudes_m, model_type)

        # Format response
        result_lines = [f"Atmospheric Profile ({model_type})", "=" * 50]
        result_lines.append(f"{'Alt (m)':>8} {'Press (Pa)':>12} {'Temp (K)':>9} {'Density':>10} {'Sound (m/s)':>12}")
        result_lines.append("-" * 60)

        for point in profile:
            result_lines.append(
                f"{point.altitude_m:8.0f} {point.pressure_pa:12.1f} {point.temperature_k:9.2f} "
                f"{point.density_kg_m3:10.6f} {point.speed_of_sound_mps:12.1f}"
            )

        # Add JSON data for programmatic use
        json_data = json.dumps([p.model_dump() for p in profile], indent=2)
        result_lines.extend(["", "JSON Data:", json_data])

        return [TextContent(type="text", text="\n".join(result_lines))]

    except Exception as e:
        return [TextContent(type="text", text=f"Atmosphere profile error: {str(e)}")]


async def _handle_wind_model_simple(arguments: dict) -> list[TextContent]:
    """Handle simple wind model requests."""
    try:
        from .integrations.atmosphere import wind_model_simple

        altitudes_m = arguments.get("altitudes_m", [])
        surface_wind_mps = arguments.get("surface_wind_mps")
        surface_altitude_m = arguments.get("surface_altitude_m", 0.0)
        model = arguments.get("model", "logarithmic")
        roughness_length_m = arguments.get("roughness_length_m", 0.1)

        if not altitudes_m or surface_wind_mps is None:
            return [TextContent(type="text", text="Error: altitudes_m and surface_wind_mps are required")]

        wind_profile = wind_model_simple(
            altitudes_m, surface_wind_mps, surface_altitude_m, model, roughness_length_m
        )

        # Format response
        result_lines = [f"Wind Profile ({model} model)", "=" * 40]
        result_lines.append(f"Surface wind: {surface_wind_mps} m/s")
        if model == "logarithmic":
            result_lines.append(f"Roughness length: {roughness_length_m} m")
        result_lines.extend(["", f"{'Alt (m)':>8} {'Wind (m/s)':>12}", "-" * 25])

        for point in wind_profile:
            result_lines.append(f"{point.altitude_m:8.0f} {point.wind_speed_mps:12.2f}")

        # Add JSON data
        json_data = json.dumps([p.model_dump() for p in wind_profile], indent=2)
        result_lines.extend(["", "JSON Data:", json_data])

        return [TextContent(type="text", text="\n".join(result_lines))]

    except Exception as e:
        return [TextContent(type="text", text=f"Wind model error: {str(e)}")]


async def _handle_transform_frames(arguments: dict) -> list[TextContent]:
    """Handle coordinate frame transformation requests."""
    try:
        from .integrations.frames import transform_frames

        xyz = arguments.get("xyz", [])
        from_frame = arguments.get("from_frame")
        to_frame = arguments.get("to_frame")
        epoch_iso = arguments.get("epoch_iso", "2000-01-01T12:00:00")

        if not xyz or not from_frame or not to_frame:
            return [TextContent(type="text", text="Error: xyz, from_frame, and to_frame are required")]

        result = transform_frames(xyz, from_frame, to_frame, epoch_iso)

        # Format response
        result_lines = ["Coordinate Frame Transformation", "=" * 40]
        result_lines.append(f"From: {from_frame}")
        result_lines.append(f"To: {to_frame}")
        result_lines.append(f"Epoch: {epoch_iso}")
        result_lines.extend(["", "Input coordinates:"])

        if from_frame == "GEODETIC":
            result_lines.append(f"  Latitude: {xyz[0]:11.6f}°")
            result_lines.append(f"  Longitude: {xyz[1]:11.6f}°")
            result_lines.append(f"  Altitude: {xyz[2]:11.2f} m")
        else:
            result_lines.append(f"  X: {xyz[0]:15.2f} m")
            result_lines.append(f"  Y: {xyz[1]:15.2f} m")
            result_lines.append(f"  Z: {xyz[2]:15.2f} m")

        result_lines.extend(["", "Transformed coordinates:"])

        if to_frame == "GEODETIC":
            result_lines.append(f"  Latitude: {result.x:11.6f}°")
            result_lines.append(f"  Longitude: {result.y:11.6f}°")
            result_lines.append(f"  Altitude: {result.z:11.2f} m")
        else:
            result_lines.append(f"  X: {result.x:15.2f} m")
            result_lines.append(f"  Y: {result.y:15.2f} m")
            result_lines.append(f"  Z: {result.z:15.2f} m")

        # Add JSON data
        json_data = json.dumps(result.model_dump(), indent=2)
        result_lines.extend(["", "JSON Data:", json_data])

        return [TextContent(type="text", text="\n".join(result_lines))]

    except Exception as e:
        return [TextContent(type="text", text=f"Frame transformation error: {str(e)}")]


async def _handle_geodetic_to_ecef(arguments: dict) -> list[TextContent]:
    """Handle geodetic to ECEF conversion."""
    try:
        from .integrations.frames import geodetic_to_ecef

        latitude_deg = arguments.get("latitude_deg")
        longitude_deg = arguments.get("longitude_deg")
        altitude_m = arguments.get("altitude_m")

        if latitude_deg is None or longitude_deg is None or altitude_m is None:
            return [TextContent(type="text", text="Error: latitude_deg, longitude_deg, and altitude_m are required")]

        result = geodetic_to_ecef(latitude_deg, longitude_deg, altitude_m)

        # Format response
        result_lines = [
            "Geodetic to ECEF Conversion",
            "=" * 35,
            f"Latitude: {latitude_deg:11.6f}°",
            f"Longitude: {longitude_deg:11.6f}°",
            f"Altitude: {altitude_m:11.2f} m",
            "",
            "ECEF Coordinates:",
            f"  X: {result.x:15.2f} m",
            f"  Y: {result.y:15.2f} m",
            f"  Z: {result.z:15.2f} m"
        ]

        # Add JSON data
        json_data = json.dumps(result.model_dump(), indent=2)
        result_lines.extend(["", "JSON Data:", json_data])

        return [TextContent(type="text", text="\n".join(result_lines))]

    except Exception as e:
        return [TextContent(type="text", text=f"Geodetic to ECEF error: {str(e)}")]


async def _handle_ecef_to_geodetic(arguments: dict) -> list[TextContent]:
    """Handle ECEF to geodetic conversion."""
    try:
        from .integrations.frames import ecef_to_geodetic

        x = arguments.get("x")
        y = arguments.get("y")
        z = arguments.get("z")

        if x is None or y is None or z is None:
            return [TextContent(type="text", text="Error: x, y, and z coordinates are required")]

        result = ecef_to_geodetic(x, y, z)

        # Format response
        result_lines = [
            "ECEF to Geodetic Conversion",
            "=" * 35,
            f"ECEF X: {x:15.2f} m",
            f"ECEF Y: {y:15.2f} m",
            f"ECEF Z: {z:15.2f} m",
            "",
            "Geodetic Coordinates:",
            f"  Latitude: {result.latitude_deg:11.6f}°",
            f"  Longitude: {result.longitude_deg:11.6f}°",
            f"  Altitude: {result.altitude_m:11.2f} m"
        ]

        # Add JSON data
        json_data = json.dumps(result.model_dump(), indent=2)
        result_lines.extend(["", "JSON Data:", json_data])

        return [TextContent(type="text", text="\n".join(result_lines))]

    except Exception as e:
        return [TextContent(type="text", text=f"ECEF to geodetic error: {str(e)}")]


async def _handle_wing_vlm_analysis(arguments: dict) -> list[TextContent]:
    """Handle wing VLM analysis requests."""
    try:
        from .integrations.aero import WingGeometry, wing_vlm_analysis

        geometry_data = arguments.get("geometry", {})
        alpha_deg_list = arguments.get("alpha_deg_list", [])
        mach = arguments.get("mach", 0.2)

        if not geometry_data or not alpha_deg_list:
            return [TextContent(type="text", text="Error: geometry and alpha_deg_list are required")]

        # Create geometry object
        geometry = WingGeometry(**geometry_data)

        # Run analysis
        results = wing_vlm_analysis(geometry, alpha_deg_list, mach)

        # Format response
        result_lines = [
            f"Wing VLM Analysis (Mach {mach})",
            "=" * 50,
            f"Geometry: {geometry.span_m:.2f}m span, AR={geometry.span_m**2/((geometry.chord_root_m+geometry.chord_tip_m)*geometry.span_m/2):.1f}",
            f"Airfoils: {geometry.airfoil_root} (root) -> {geometry.airfoil_tip} (tip)",
            "",
            f"{'Alpha (°)':>8} {'CL':>8} {'CD':>8} {'CM':>8} {'L/D':>8} {'Eff':>8}",
            "-" * 55
        ]

        for point in results:
            eff_str = f"{point.span_efficiency:.3f}" if point.span_efficiency else "N/A"
            result_lines.append(
                f"{point.alpha_deg:8.1f} {point.CL:8.4f} {point.CD:8.5f} {point.CM:8.4f} "
                f"{point.L_D_ratio:8.1f} {eff_str:>8}"
            )

        # Add JSON data
        json_data = json.dumps([p.model_dump() for p in results], indent=2)
        result_lines.extend(["", "JSON Data:", json_data])

        return [TextContent(type="text", text="\n".join(result_lines))]

    except Exception as e:
        return [TextContent(type="text", text=f"Wing analysis error: {str(e)}")]


async def _handle_airfoil_polar_analysis(arguments: dict) -> list[TextContent]:
    """Handle airfoil polar analysis requests."""
    try:
        from .integrations.aero import airfoil_polar_analysis

        airfoil_name = arguments.get("airfoil_name", "NACA2412")
        alpha_deg_list = arguments.get("alpha_deg_list", [])
        reynolds = arguments.get("reynolds", 1000000)
        mach = arguments.get("mach", 0.1)

        if not alpha_deg_list:
            return [TextContent(type="text", text="Error: alpha_deg_list is required")]

        # Run analysis
        results = airfoil_polar_analysis(airfoil_name, alpha_deg_list, reynolds, mach)

        # Format response
        result_lines = [
            f"Airfoil Polar Analysis: {airfoil_name}",
            "=" * 50,
            f"Reynolds: {reynolds:.0e}, Mach: {mach:.3f}",
            "",
            f"{'Alpha (°)':>8} {'CL':>8} {'CD':>8} {'CM':>8} {'L/D':>8}",
            "-" * 45
        ]

        for point in results:
            result_lines.append(
                f"{point.alpha_deg:8.1f} {point.cl:8.4f} {point.cd:8.5f} "
                f"{point.cm:8.4f} {point.cl_cd_ratio:8.1f}"
            )

        # Add JSON data
        json_data = json.dumps([p.model_dump() for p in results], indent=2)
        result_lines.extend(["", "JSON Data:", json_data])

        return [TextContent(type="text", text="\n".join(result_lines))]

    except Exception as e:
        return [TextContent(type="text", text=f"Airfoil polar error: {str(e)}")]


async def _handle_calculate_stability_derivatives(arguments: dict) -> list[TextContent]:
    """Handle stability derivatives calculation requests."""
    try:
        from .integrations.aero import (
            WingGeometry,
            calculate_stability_derivatives,
            estimate_wing_area,
        )

        geometry_data = arguments.get("geometry", {})
        alpha_deg = arguments.get("alpha_deg", 2.0)
        mach = arguments.get("mach", 0.2)

        if not geometry_data:
            return [TextContent(type="text", text="Error: geometry is required")]

        # Create geometry object
        geometry = WingGeometry(**geometry_data)

        # Calculate stability derivatives
        stability = calculate_stability_derivatives(geometry, alpha_deg, mach)

        # Calculate wing properties
        wing_props = estimate_wing_area(geometry)

        # Format response
        result_lines = [
            "Stability Derivatives Analysis",
            "=" * 40,
            f"Reference: α = {alpha_deg}°, M = {mach}",
            "",
            "Wing Geometry:",
            f"  Area: {wing_props['wing_area_m2']:.2f} m²",
            f"  Aspect Ratio: {wing_props['aspect_ratio']:.1f}",
            f"  Taper Ratio: {wing_props['taper_ratio']:.2f}",
            f"  MAC: {wing_props['mean_aerodynamic_chord_m']:.3f} m",
            "",
            "Stability Derivatives:",
            f"  CL_α = {stability.CL_alpha:.3f} /rad ({math.degrees(stability.CL_alpha):.3f} /deg)",
            f"  CM_α = {stability.CM_alpha:.4f} /rad ({math.degrees(stability.CM_alpha):.4f} /deg)",
        ]

        if stability.CL_alpha_dot:
            result_lines.append(f"  CL_α̇ = {stability.CL_alpha_dot:.4f}")
        if stability.CM_alpha_dot:
            result_lines.append(f"  CM_α̇ = {stability.CM_alpha_dot:.4f}")

        # Stability assessment
        result_lines.extend(["", "Stability Assessment:"])
        if stability.CM_alpha < 0:
            result_lines.append("  ✓ Statically stable (CM_α < 0)")
        else:
            result_lines.append("  ⚠ Statically unstable (CM_α > 0)")

        # Add JSON data
        combined_data = {
            "stability_derivatives": stability.model_dump(),
            "wing_properties": wing_props
        }
        json_data = json.dumps(combined_data, indent=2)
        result_lines.extend(["", "JSON Data:", json_data])

        return [TextContent(type="text", text="\n".join(result_lines))]

    except Exception as e:
        return [TextContent(type="text", text=f"Stability derivatives error: {str(e)}")]


async def _handle_propeller_bemt_analysis(arguments: dict) -> list[TextContent]:
    """Handle propeller BEMT analysis requests."""
    try:
        from .integrations.propellers import PropellerGeometry, propeller_bemt_analysis

        geometry_data = arguments.get("geometry", {})
        rpm_list = arguments.get("rpm_list", [])
        velocity_ms = arguments.get("velocity_ms", 0.0)
        altitude_m = arguments.get("altitude_m", 0.0)

        if not geometry_data or not rpm_list:
            return [TextContent(type="text", text="Error: geometry and rpm_list are required")]

        # Create geometry object
        geometry = PropellerGeometry(**geometry_data)

        # Run analysis
        results = propeller_bemt_analysis(geometry, rpm_list, velocity_ms, altitude_m)

        # Format response
        result_lines = [
            "Propeller BEMT Analysis",
            "=" * 60,
            f"Propeller: {geometry.diameter_m:.3f}m dia × {geometry.pitch_m:.3f}m pitch, {geometry.num_blades} blades",
            f"Conditions: V = {velocity_ms:.1f} m/s, Alt = {altitude_m:.0f} m",
            "",
            f"{'RPM':>6} {'Thrust(N)':>10} {'Power(W)':>10} {'Torque(Nm)':>11} {'Efficiency':>10} {'J':>6} {'CT':>8} {'CP':>8}",
            "-" * 85
        ]

        for point in results:
            result_lines.append(
                f"{point.rpm:6.0f} {point.thrust_n:10.1f} {point.power_w:10.1f} "
                f"{point.torque_nm:11.3f} {point.efficiency:10.3f} {point.advance_ratio:6.3f} "
                f"{point.thrust_coefficient:8.4f} {point.power_coefficient:8.4f}"
            )

        # Find peak efficiency point
        if results:
            peak_eff_point = max(results, key=lambda x: x.efficiency)
            result_lines.extend([
                "",
                f"Peak Efficiency: {peak_eff_point.efficiency:.1%} at {peak_eff_point.rpm:.0f} RPM",
                f"  Thrust: {peak_eff_point.thrust_n:.1f} N, Power: {peak_eff_point.power_w:.1f} W"
            ])

        # Add JSON data
        json_data = json.dumps([p.model_dump() for p in results], indent=2)
        result_lines.extend(["", "JSON Data:", json_data])

        return [TextContent(type="text", text="\n".join(result_lines))]

    except Exception as e:
        return [TextContent(type="text", text=f"Propeller analysis error: {str(e)}")]


async def _handle_uav_energy_estimate(arguments: dict) -> list[TextContent]:
    """Handle UAV energy estimation requests."""
    try:
        from .integrations.propellers import (
            BatteryConfiguration,
            UAVConfiguration,
            uav_energy_estimate,
        )

        uav_data = arguments.get("uav_config", {})
        battery_data = arguments.get("battery_config", {})
        mission_data = arguments.get("mission_profile", {})

        if not uav_data or not battery_data:
            return [TextContent(type="text", text="Error: uav_config and battery_config are required")]

        # Create configuration objects
        uav_config = UAVConfiguration(**uav_data)
        battery_config = BatteryConfiguration(**battery_data)

        # Run analysis
        result = uav_energy_estimate(uav_config, battery_config, mission_data)

        # Determine aircraft type
        aircraft_type = "Fixed-Wing" if uav_config.wing_area_m2 else "Multirotor"

        # Format response
        result_lines = [
            f"UAV Energy Analysis ({aircraft_type})",
            "=" * 45,
            f"Aircraft Mass: {uav_config.mass_kg:.1f} kg",
            f"Battery: {battery_config.capacity_ah:.1f} Ah @ {battery_config.voltage_nominal_v:.1f}V ({battery_config.mass_kg:.2f} kg)",
        ]

        if uav_config.wing_area_m2:
            result_lines.extend([
                f"Wing Area: {uav_config.wing_area_m2:.2f} m²",
                f"Wing Loading: {uav_config.mass_kg * 9.81 / uav_config.wing_area_m2:.1f} N/m²"
            ])
        elif uav_config.disk_area_m2:
            result_lines.extend([
                f"Rotor Disk Area: {uav_config.disk_area_m2:.2f} m²",
                f"Disk Loading: {uav_config.mass_kg * 9.81 / uav_config.disk_area_m2:.1f} N/m²"
            ])

        result_lines.extend([
            "",
            "Energy Analysis:",
            f"  Battery Energy: {result.battery_energy_wh:.0f} Wh",
            f"  Usable Energy: {result.energy_consumed_wh:.0f} Wh",
            f"  Power Required: {result.power_required_w:.0f} W",
            f"  System Efficiency: {result.efficiency_overall:.1%}",
            "",
            "Mission Performance:",
            f"  Flight Time: {result.flight_time_min:.1f} minutes ({result.flight_time_min/60:.1f} hours)"
        ])

        if result.range_km:
            result_lines.append(f"  Range: {result.range_km:.1f} km")

        if result.hover_time_min:
            result_lines.append(f"  Hover Time: {result.hover_time_min:.1f} minutes")

        # Add recommendations
        result_lines.extend(["", "Recommendations:"])

        if result.flight_time_min < 10:
            result_lines.append("  ⚠ Very short flight time - consider larger battery or lighter aircraft")
        elif result.flight_time_min < 20:
            result_lines.append("  ⚠ Short flight time - optimize for efficiency or add battery capacity")
        elif result.flight_time_min > 120:
            result_lines.append("  ✓ Excellent endurance - well optimized configuration")
        else:
            result_lines.append("  ✓ Good flight time for mission requirements")

        if result.efficiency_overall < 0.7:
            result_lines.append("  ⚠ Low system efficiency - check motor/propeller matching")
        else:
            result_lines.append("  ✓ Good system efficiency")

        # Add JSON data
        json_data = json.dumps(result.model_dump(), indent=2)
        result_lines.extend(["", "JSON Data:", json_data])

        return [TextContent(type="text", text="\n".join(result_lines))]

    except Exception as e:
        return [TextContent(type="text", text=f"UAV energy analysis error: {str(e)}")]


async def _handle_get_airfoil_database(arguments: dict) -> list[TextContent]:
    """Handle airfoil database requests."""
    try:
        from .integrations.aero import get_airfoil_database

        database = get_airfoil_database()

        # Format response
        result_lines = [
            "Available Airfoil Database",
            "=" * 35,
            f"{'Airfoil':>12} {'CL_α':>8} {'CD0':>8} {'CL_max':>8} {'α_stall':>8}",
            "-" * 50
        ]

        for name, data in database.items():
            result_lines.append(
                f"{name:>12} {data['cl_alpha']:8.2f} {data['cd0']:8.4f} "
                f"{data['cl_max']:8.2f} {data['alpha_stall_deg']:8.1f}°"
            )

        result_lines.extend([
            "",
            "Notes:",
            "  CL_α: Lift curve slope (per radian)",
            "  CD0: Zero-lift drag coefficient",
            "  CL_max: Maximum lift coefficient",
            "  α_stall: Stall angle of attack"
        ])

        # Add JSON data
        json_data = json.dumps(database, indent=2)
        result_lines.extend(["", "JSON Data:", json_data])

        return [TextContent(type="text", text="\n".join(result_lines))]

    except Exception as e:
        return [TextContent(type="text", text=f"Airfoil database error: {str(e)}")]


async def _handle_get_propeller_database(arguments: dict) -> list[TextContent]:
    """Handle propeller database requests."""
    try:
        from .integrations.propellers import get_propeller_database

        database = get_propeller_database()

        # Format response
        result_lines = [
            "Available Propeller Database",
            "=" * 40,
            f"{'Propeller':>15} {'Diameter':>9} {'Pitch':>7} {'Blades':>7} {'η_max':>7}",
            "-" * 50
        ]

        for name, data in database.items():
            diameter_in = data["diameter_m"] * 39.37  # Convert to inches
            pitch_in = data["pitch_m"] * 39.37
            result_lines.append(
                f"{name:>15} {diameter_in:9.1f}\" {pitch_in:7.1f}\" "
                f"{data['num_blades']:7d} {data['efficiency_max']:7.1%}"
            )

        result_lines.extend([
            "",
            "Notes:",
            "  Diameter and pitch shown in inches",
            "  η_max: Maximum efficiency (estimated)",
            "  Data includes activity factor and design coefficients"
        ])

        # Add JSON data
        json_data = json.dumps(database, indent=2)
        result_lines.extend(["", "JSON Data:", json_data])

        return [TextContent(type="text", text="\n".join(result_lines))]

    except Exception as e:
        return [TextContent(type="text", text=f"Propeller database error: {str(e)}")]


async def _handle_rocket_3dof_trajectory(arguments: dict) -> list[TextContent]:
    """Handle rocket trajectory calculation requests."""
    try:
        from .integrations.rockets import (
            RocketGeometry,
            analyze_rocket_performance,
            rocket_3dof_trajectory,
        )

        geometry_data = arguments.get("geometry", {})
        dt_s = arguments.get("dt_s", 0.1)
        max_time_s = arguments.get("max_time_s", 300.0)
        launch_angle_deg = arguments.get("launch_angle_deg", 90.0)

        if not geometry_data:
            return [TextContent(type="text", text="Error: geometry is required")]

        # Create geometry object
        geometry = RocketGeometry(**geometry_data)

        # Run trajectory calculation
        trajectory = rocket_3dof_trajectory(geometry, dt_s, max_time_s, launch_angle_deg)
        performance = analyze_rocket_performance(trajectory)

        # Format response
        result_lines = [
            "3DOF Rocket Trajectory Analysis",
            "=" * 50,
            f"Launch Angle: {launch_angle_deg}°",
            f"Rocket: {geometry.dry_mass_kg + geometry.propellant_mass_kg:.0f} kg total ({geometry.dry_mass_kg:.0f} kg dry)",
            f"Geometry: {geometry.diameter_m:.2f}m × {geometry.length_m:.1f}m (CD = {geometry.cd:.2f})",
            "",
            "Performance Summary:",
            f"  Max Altitude: {performance.max_altitude_m/1000:.2f} km ({performance.max_altitude_m:.0f} m)",
            f"  Apogee Time: {performance.apogee_time_s:.1f} seconds",
            f"  Max Velocity: {performance.max_velocity_ms:.0f} m/s (Mach {performance.max_mach:.2f})",
            f"  Max Dynamic Pressure: {performance.max_q_pa/1000:.1f} kPa",
            "",
            "Engine Performance:",
            f"  Burnout Altitude: {performance.burnout_altitude_m/1000:.2f} km",
            f"  Burnout Velocity: {performance.burnout_velocity_ms:.0f} m/s",
            f"  Burnout Time: {performance.burnout_time_s:.1f} s",
            f"  Total Impulse: {performance.total_impulse_ns/1000:.1f} kN·s",
            f"  Specific Impulse: {performance.specific_impulse_s:.0f} s",
            "",
            f"Trajectory Points: {len(trajectory)} (Δt = {dt_s:.2f} s)"
        ]

        # Add JSON data for trajectory points (sample every 10th point to avoid huge output)
        sample_trajectory = trajectory[::max(1, len(trajectory)//50)]  # Max 50 points
        json_data = {
            "performance": performance.model_dump() if hasattr(performance, 'model_dump') else asdict(performance),
            "trajectory_sample": [asdict(point) for point in sample_trajectory]
        }
        result_lines.extend(["", "JSON Data (sampled trajectory):", json.dumps(json_data, indent=2)])

        return [TextContent(type="text", text="\n".join(result_lines))]

    except Exception as e:
        return [TextContent(type="text", text=f"Rocket trajectory error: {str(e)}")]


async def _handle_estimate_rocket_sizing(arguments: dict) -> list[TextContent]:
    """Handle rocket sizing estimation requests."""
    try:
        from .integrations.rockets import estimate_rocket_sizing

        target_altitude_m = arguments.get("target_altitude_m")
        payload_mass_kg = arguments.get("payload_mass_kg")
        propellant_type = arguments.get("propellant_type", "solid")

        if target_altitude_m is None or payload_mass_kg is None:
            return [TextContent(type="text", text="Error: target_altitude_m and payload_mass_kg are required")]

        # Run sizing analysis
        sizing = estimate_rocket_sizing(target_altitude_m, payload_mass_kg, propellant_type)

        # Format response
        result_lines = [
            f"Rocket Sizing Estimate ({propellant_type.title()} Propellant)",
            "=" * 50,
            "Mission Requirements:",
            f"  Target Altitude: {target_altitude_m/1000:.1f} km",
            f"  Payload Mass: {payload_mass_kg:.2f} kg",
            "",
            f"Mission Feasible: {'✓ Yes' if sizing['feasible'] else '✗ No (requires staging)'}",
        ]

        if sizing['feasible']:
            result_lines.extend([
                "",
                "Rocket Sizing:",
                f"  Total Mass: {sizing['total_mass_kg']:.0f} kg",
                f"  Propellant Mass: {sizing['propellant_mass_kg']:.0f} kg ({sizing['propellant_mass_kg']/sizing['total_mass_kg']*100:.1f}%)",
                f"  Structure Mass: {sizing['structure_mass_kg']:.0f} kg ({sizing['structure_mass_kg']/sizing['total_mass_kg']*100:.1f}%)",
                f"  Payload Mass: {payload_mass_kg:.0f} kg ({payload_mass_kg/sizing['total_mass_kg']*100:.1f}%)",
                "",
                "Performance:",
                f"  Mass Ratio: {sizing['mass_ratio']:.2f}",
                f"  Delta-V Required: {sizing['delta_v_ms']:.0f} m/s",
                f"  Specific Impulse: {sizing['specific_impulse_s']:.0f} s",
                f"  Thrust Required: {sizing['thrust_n']/1000:.1f} kN",
                f"  Thrust-to-Weight: {sizing['thrust_to_weight']:.1f}",
                "",
                "Geometry Estimates:",
                f"  Diameter: {sizing['diameter_m']:.2f} m",
                f"  Length: {sizing['length_m']:.1f} m",
                f"  L/D Ratio: {sizing['length_m']/sizing['diameter_m']:.1f}"
            ])
        else:
            result_lines.extend([
                "",
                "⚠ Mission requires staging or different propellant",
                "Consider:",
                "  - Multi-stage rocket design",
                "  - Higher performance propellant (liquid vs solid)",
                "  - Reduced payload mass",
                "  - Lower target altitude"
            ])

        # Add JSON data
        json_data = json.dumps(sizing, indent=2)
        result_lines.extend(["", "JSON Data:", json_data])

        return [TextContent(type="text", text="\n".join(result_lines))]

    except Exception as e:
        return [TextContent(type="text", text=f"Rocket sizing error: {str(e)}")]


async def _handle_optimize_launch_angle(arguments: dict) -> list[TextContent]:
    """Handle launch angle optimization requests."""
    try:
        from .integrations.rockets import RocketGeometry
        from .integrations.trajopt import (
            optimize_launch_angle,
        )

        geometry_data = arguments.get("geometry", {})
        objective = arguments.get("objective", "max_altitude")
        angle_bounds = arguments.get("angle_bounds", [80.0, 90.0])

        if not geometry_data:
            return [TextContent(type="text", text="Error: geometry is required")]

        # Create geometry object
        geometry = RocketGeometry(**geometry_data)

        # Run optimization
        result = optimize_launch_angle(geometry, objective, tuple(angle_bounds))

        # Format response
        result_lines = [
            f"Launch Angle Optimization ({objective})",
            "=" * 50,
            f"Rocket: {geometry.dry_mass_kg + geometry.propellant_mass_kg:.0f} kg total",
            f"Search Range: {angle_bounds[0]:.1f}° to {angle_bounds[1]:.1f}°",
            "",
            "Optimization Results:",
            f"  Optimal Launch Angle: {result.optimal_parameters['launch_angle_deg']:.2f}°",
            f"  Optimal {objective.replace('_', ' ').title()}: {result.optimal_objective:.1f} {'m' if 'altitude' in objective else 'units'}",
            f"  Converged: {'✓ Yes' if result.converged else '✗ No'}",
            f"  Iterations: {result.iterations}",
            "",
            "Performance at Optimal Angle:",
            f"  Max Altitude: {result.performance.max_altitude_m/1000:.2f} km",
            f"  Max Velocity: {result.performance.max_velocity_ms:.0f} m/s",
            f"  Burnout Time: {result.performance.burnout_time_s:.1f} s",
            f"  Total Impulse: {result.performance.total_impulse_ns/1000:.1f} kN·s"
        ]

        # Add JSON data
        json_data = {
            "optimization_result": {
                "optimal_parameters": result.optimal_parameters,
                "optimal_objective": result.optimal_objective,
                "converged": result.converged,
                "iterations": result.iterations
            },
            "performance": asdict(result.performance)
        }
        result_lines.extend(["", "JSON Data:", json.dumps(json_data, indent=2)])

        return [TextContent(type="text", text="\n".join(result_lines))]

    except Exception as e:
        return [TextContent(type="text", text=f"Launch angle optimization error: {str(e)}")]


async def _handle_optimize_thrust_profile(arguments: dict) -> list[TextContent]:
    """Handle thrust profile optimization requests."""
    try:
        from .integrations.rockets import RocketGeometry
        from .integrations.trajopt import optimize_thrust_profile

        geometry_data = arguments.get("geometry", {})
        burn_time_s = arguments.get("burn_time_s")
        total_impulse_target = arguments.get("total_impulse_target")
        n_segments = arguments.get("n_segments", 5)
        objective = arguments.get("objective", "max_altitude")

        if not geometry_data or burn_time_s is None or total_impulse_target is None:
            return [TextContent(type="text", text="Error: geometry, burn_time_s, and total_impulse_target are required")]

        # Create geometry object
        geometry = RocketGeometry(**geometry_data)

        # Run optimization
        result = optimize_thrust_profile(geometry, burn_time_s, total_impulse_target, n_segments, objective)

        # Format response
        result_lines = [
            f"Thrust Profile Optimization ({objective})",
            "=" * 60,
            f"Target: {objective.replace('_', ' ').title()}",
            f"Burn Time: {burn_time_s:.1f} s, Target Impulse: {total_impulse_target/1000:.1f} kN·s",
            f"Segments: {n_segments}",
            "",
            "Optimization Results:",
            f"  Converged: {'✓ Yes' if result.converged else '✗ No'}",
            f"  Iterations: {result.iterations}",
            f"  Optimal {objective.replace('_', ' ').title()}: {result.optimal_objective:.1f} {'m' if 'altitude' in objective else 'units'}",
            "",
            "Optimized Performance:",
            f"  Max Altitude: {result.performance.max_altitude_m/1000:.2f} km",
            f"  Max Velocity: {result.performance.max_velocity_ms:.0f} m/s",
            f"  Max Dynamic Pressure: {result.performance.max_q_pa/1000:.1f} kPa",
            f"  Actual Total Impulse: {result.performance.total_impulse_ns/1000:.1f} kN·s",
            "",
            "Thrust Profile (segment multipliers):"
        ]

        # Show thrust multipliers
        for i in range(n_segments):
            param_key = f"thrust_mult_seg_{i+1}"
            if param_key in result.optimal_parameters:
                multiplier = result.optimal_parameters[param_key]
                time_start = i * burn_time_s / n_segments
                time_end = (i + 1) * burn_time_s / n_segments
                result_lines.append(f"  Segment {i+1} ({time_start:.1f}-{time_end:.1f}s): {multiplier:.2f}x")

        # Add JSON data (excluding large thrust_curve for brevity)
        json_data = {
            "optimization_result": {
                "converged": result.converged,
                "iterations": result.iterations,
                "optimal_objective": result.optimal_objective,
                "thrust_multipliers": {k: v for k, v in result.optimal_parameters.items() if k.startswith("thrust_mult")}
            },
            "performance": asdict(result.performance)
        }
        result_lines.extend(["", "JSON Data:", json.dumps(json_data, indent=2)])

        return [TextContent(type="text", text="\n".join(result_lines))]

    except Exception as e:
        return [TextContent(type="text", text=f"Thrust profile optimization error: {str(e)}")]


async def _handle_trajectory_sensitivity_analysis(arguments: dict) -> list[TextContent]:
    """Handle trajectory sensitivity analysis requests."""
    try:
        from .integrations.rockets import RocketGeometry
        from .integrations.trajopt import trajectory_sensitivity_analysis

        geometry_data = arguments.get("base_geometry", {})
        parameter_variations = arguments.get("parameter_variations", {})
        objective = arguments.get("objective", "max_altitude")

        if not geometry_data or not parameter_variations:
            return [TextContent(type="text", text="Error: base_geometry and parameter_variations are required")]

        # Create geometry object
        base_geometry = RocketGeometry(**geometry_data)

        # Run sensitivity analysis
        result = trajectory_sensitivity_analysis(base_geometry, parameter_variations, objective)

        # Format response
        result_lines = [
            f"Trajectory Sensitivity Analysis ({objective})",
            "=" * 60,
            f"Baseline {objective.replace('_', ' ').title()}: {result['baseline_value']:.1f}",
            f"Parameters Analyzed: {len(parameter_variations)}",
            ""
        ]

        # Show sensitivity for each parameter
        for param_name, param_results in result["parameter_sensitivities"].items():
            result_lines.extend([
                f"Parameter: {param_name}",
                "-" * 40
            ])

            # Calculate average sensitivity (excluding failed simulations)
            valid_results = [r for r in param_results if "sensitivity" in r and r["sensitivity"] is not None]
            if valid_results:
                avg_sensitivity = sum(abs(r["sensitivity"]) for r in valid_results) / len(valid_results)
                result_lines.append(f"  Average |Sensitivity|: {avg_sensitivity:.3f} %/% change")

                # Show most sensitive point
                max_sens = max(valid_results, key=lambda x: abs(x.get("sensitivity", 0)))
                result_lines.append(f"  Max Sensitivity: {max_sens['sensitivity']:.3f} at {param_name} = {max_sens['parameter_value']:.3f}")

                # Classification
                if avg_sensitivity > 2.0:
                    sensitivity_class = "HIGH"
                elif avg_sensitivity > 0.5:
                    sensitivity_class = "MEDIUM"
                else:
                    sensitivity_class = "LOW"
                result_lines.append(f"  Sensitivity Class: {sensitivity_class}")
            else:
                result_lines.append("  No valid sensitivity data")

            result_lines.append("")

        # Ranking by average sensitivity
        param_sensitivities = {}
        for param_name, param_results in result["parameter_sensitivities"].items():
            valid_results = [r for r in param_results if "sensitivity" in r and r["sensitivity"] is not None]
            if valid_results:
                param_sensitivities[param_name] = sum(abs(r["sensitivity"]) for r in valid_results) / len(valid_results)

        if param_sensitivities:
            sorted_params = sorted(param_sensitivities.items(), key=lambda x: x[1], reverse=True)
            result_lines.extend([
                "Parameter Sensitivity Ranking:",
                "-" * 35
            ])
            for i, (param, sens) in enumerate(sorted_params, 1):
                result_lines.append(f"  {i}. {param}: {sens:.3f}")

        # Add JSON data
        json_data = json.dumps(result, indent=2)
        result_lines.extend(["", "JSON Data:", json_data])

        return [TextContent(type="text", text="\n".join(result_lines))]

    except Exception as e:
        return [TextContent(type="text", text=f"Trajectory sensitivity analysis error: {str(e)}")]


async def _handle_elements_to_state_vector(arguments: dict) -> list[TextContent]:
    """Handle orbital elements to state vector conversion."""
    try:
        from .integrations.orbits import OrbitElements, elements_to_state_vector

        elements_data = arguments.get("elements", {})

        if not elements_data:
            return [TextContent(type="text", text="Error: elements are required")]

        # Create elements object
        elements = OrbitElements(**elements_data)

        # Convert to state vector
        state = elements_to_state_vector(elements)

        # Format response
        result_lines = [
            "Orbital Elements → State Vector Conversion",
            "=" * 55,
            "Input Elements:",
            f"  Semi-major axis: {elements.semi_major_axis_m/1000:.1f} km",
            f"  Eccentricity: {elements.eccentricity:.4f}",
            f"  Inclination: {elements.inclination_deg:.2f}°",
            f"  RAAN: {elements.raan_deg:.2f}°",
            f"  Arg. Periapsis: {elements.arg_periapsis_deg:.2f}°",
            f"  True Anomaly: {elements.true_anomaly_deg:.2f}°",
            "",
            "Output State Vector (J2000):",
            f"  Position: [{state.position_m[0]:.0f}, {state.position_m[1]:.0f}, {state.position_m[2]:.0f}] m",
            f"  Velocity: [{state.velocity_ms[0]:.3f}, {state.velocity_ms[1]:.3f}, {state.velocity_ms[2]:.3f}] m/s",
            f"  Epoch: {state.epoch_utc}",
            f"  Frame: {state.frame}"
        ]

        # Add JSON data
        json_data = json.dumps(asdict(state), indent=2)
        result_lines.extend(["", "JSON Data:", json_data])

        return [TextContent(type="text", text="\n".join(result_lines))]

    except Exception as e:
        return [TextContent(type="text", text=f"Elements to state vector error: {str(e)}")]


async def _handle_state_vector_to_elements(arguments: dict) -> list[TextContent]:
    """Handle state vector to orbital elements conversion."""
    try:
        from .integrations.orbits import (
            StateVector,
            calculate_orbit_properties,
            state_vector_to_elements,
        )

        state_data = arguments.get("state", {})

        if not state_data:
            return [TextContent(type="text", text="Error: state is required")]

        # Create state vector object
        state = StateVector(**state_data)

        # Convert to elements
        elements = state_vector_to_elements(state)

        # Calculate orbital properties
        properties = calculate_orbit_properties(elements)

        # Format response
        result_lines = [
            "State Vector → Orbital Elements Conversion",
            "=" * 55,
            "Input State Vector:",
            f"  Position: [{state.position_m[0]:.0f}, {state.position_m[1]:.0f}, {state.position_m[2]:.0f}] m",
            f"  Velocity: [{state.velocity_ms[0]:.3f}, {state.velocity_ms[1]:.3f}, {state.velocity_ms[2]:.3f}] m/s",
            "",
            "Output Elements:",
            f"  Semi-major axis: {elements.semi_major_axis_m/1000:.1f} km",
            f"  Eccentricity: {elements.eccentricity:.4f}",
            f"  Inclination: {elements.inclination_deg:.2f}°",
            f"  RAAN: {elements.raan_deg:.2f}°",
            f"  Arg. Periapsis: {elements.arg_periapsis_deg:.2f}°",
            f"  True Anomaly: {elements.true_anomaly_deg:.2f}°",
            "",
            "Orbital Properties:",
            f"  Period: {properties.period_s/3600:.2f} hours",
            f"  Apoapsis: {properties.apoapsis_m/1000:.1f} km altitude",
            f"  Periapsis: {properties.periapsis_m/1000:.1f} km altitude",
            f"  Energy: {properties.energy_j_kg/1000:.1f} kJ/kg"
        ]

        # Add JSON data
        combined_data = {
            "elements": asdict(elements),
            "properties": asdict(properties)
        }
        json_data = json.dumps(combined_data, indent=2)
        result_lines.extend(["", "JSON Data:", json_data])

        return [TextContent(type="text", text="\n".join(result_lines))]

    except Exception as e:
        return [TextContent(type="text", text=f"State vector to elements error: {str(e)}")]


async def _handle_propagate_orbit_j2(arguments: dict) -> list[TextContent]:
    """Handle J2 orbit propagation."""
    try:
        from .integrations.orbits import StateVector, propagate_orbit_j2

        initial_state_data = arguments.get("initial_state", {})
        time_span_s = arguments.get("time_span_s", 3600.0)
        time_step_s = arguments.get("time_step_s", 60.0)

        if not initial_state_data:
            return [TextContent(type="text", text="Error: initial_state is required")]

        # Create initial state
        initial_state = StateVector(**initial_state_data)

        # Propagate orbit
        states = propagate_orbit_j2(initial_state, time_span_s, time_step_s)

        # Format response
        result_lines = [
            "J2 Orbit Propagation",
            "=" * 40,
            f"Time Span: {time_span_s/3600:.2f} hours ({len(states)} states)",
            f"Time Step: {time_step_s:.0f} seconds",
            "",
            "Initial State:",
            f"  Position: {vector_magnitude(initial_state.position_m)/1000:.1f} km altitude",
            f"  Velocity: {vector_magnitude(initial_state.velocity_ms):.3f} m/s",
            "",
            "Final State:",
            f"  Position: {vector_magnitude(states[-1].position_m)/1000:.1f} km altitude",
            f"  Velocity: {vector_magnitude(states[-1].velocity_ms):.3f} m/s",
            "",
            "Propagation Summary:",
            f"  Total States: {len(states)}",
            f"  Position Change: {vector_magnitude([states[-1].position_m[i] - states[0].position_m[i] for i in range(3)])/1000:.1f} km",
            f"  Velocity Change: {vector_magnitude([states[-1].velocity_ms[i] - states[0].velocity_ms[i] for i in range(3)]):.3f} m/s"
        ]

        # Sample states for JSON (every 10th state to avoid huge output)
        sample_states = states[::max(1, len(states)//20)]  # Max 20 states
        json_data = {
            "propagation_summary": {
                "total_states": len(states),
                "time_span_s": time_span_s,
                "time_step_s": time_step_s
            },
            "sample_states": [asdict(state) for state in sample_states]
        }
        result_lines.extend(["", "JSON Data (sampled states):", json.dumps(json_data, indent=2)])

        return [TextContent(type="text", text="\n".join(result_lines))]

    except Exception as e:
        return [TextContent(type="text", text=f"Orbit propagation error: {str(e)}")]


async def _handle_calculate_ground_track(arguments: dict) -> list[TextContent]:
    """Handle ground track calculation."""
    try:
        from .integrations.orbits import StateVector, calculate_ground_track

        orbit_states_data = arguments.get("orbit_states", [])
        time_step_s = arguments.get("time_step_s", 60.0)

        if not orbit_states_data:
            return [TextContent(type="text", text="Error: orbit_states are required")]

        # Create state vector objects
        orbit_states = [StateVector(**state_data) for state_data in orbit_states_data]

        # Calculate ground track
        ground_track = calculate_ground_track(orbit_states, time_step_s)

        # Format response
        result_lines = [
            "Ground Track Calculation",
            "=" * 35,
            f"Orbital States: {len(orbit_states)}",
            f"Ground Track Points: {len(ground_track)}",
            "",
            "Ground Track Summary:",
            f"  Latitude Range: {min(p.latitude_deg for p in ground_track):.2f}° to {max(p.latitude_deg for p in ground_track):.2f}°",
            f"  Longitude Range: {min(p.longitude_deg for p in ground_track):.2f}° to {max(p.longitude_deg for p in ground_track):.2f}°",
            f"  Altitude Range: {min(p.altitude_m for p in ground_track)/1000:.1f} to {max(p.altitude_m for p in ground_track)/1000:.1f} km",
            "",
            "Sample Ground Track Points:"
        ]

        # Show sample points
        sample_points = ground_track[::max(1, len(ground_track)//10)]  # Max 10 points
        for i, point in enumerate(sample_points):
            result_lines.append(f"  Point {i+1}: {point.latitude_deg:.2f}°N, {point.longitude_deg:.2f}°E, {point.altitude_m/1000:.1f}km")

        # Add JSON data
        json_data = {
            "ground_track_summary": {
                "total_points": len(ground_track),
                "latitude_range": [min(p.latitude_deg for p in ground_track), max(p.latitude_deg for p in ground_track)],
                "longitude_range": [min(p.longitude_deg for p in ground_track), max(p.longitude_deg for p in ground_track)]
            },
            "ground_track_points": [asdict(point) for point in ground_track]
        }
        result_lines.extend(["", "JSON Data:", json.dumps(json_data, indent=2)])

        return [TextContent(type="text", text="\n".join(result_lines))]

    except Exception as e:
        return [TextContent(type="text", text=f"Ground track calculation error: {str(e)}")]


async def _handle_hohmann_transfer(arguments: dict) -> list[TextContent]:
    """Handle Hohmann transfer calculation."""
    try:
        from .integrations.orbits import hohmann_transfer

        r1_m = arguments.get("r1_m")
        r2_m = arguments.get("r2_m")

        if r1_m is None or r2_m is None:
            return [TextContent(type="text", text="Error: r1_m and r2_m are required")]

        # Calculate Hohmann transfer
        transfer = hohmann_transfer(r1_m, r2_m)

        # Format response
        result_lines = [
            "Hohmann Transfer Analysis",
            "=" * 35,
            f"Initial Orbit: {r1_m/1000:.0f} km radius ({(r1_m-6.378137e6)/1000:.0f} km altitude)",
            f"Final Orbit: {r2_m/1000:.0f} km radius ({(r2_m-6.378137e6)/1000:.0f} km altitude)",
            "",
            "Transfer Requirements:",
            f"  First Burn (ΔV₁): {transfer['delta_v_1_ms']:.1f} m/s",
            f"  Second Burn (ΔV₂): {transfer['delta_v_2_ms']:.1f} m/s",
            f"  Total ΔV: {transfer['delta_v_total_ms']:.1f} m/s",
            "",
            "Transfer Orbit:",
            f"  Semi-major Axis: {transfer['semi_major_axis_m']/1000:.0f} km",
            f"  Transfer Time: {transfer['transfer_time_h']:.2f} hours",
            "",
            "Mission Summary:",
            f"  Orbit Ratio: {r2_m/r1_m:.2f}",
            f"  Altitude Change: {(r2_m-r1_m)/1000:.0f} km",
            f"  Transfer Efficiency: {transfer['delta_v_total_ms']/(abs(math.sqrt(3.986004418e14/r1_m) - math.sqrt(3.986004418e14/r2_m))):.2f}"
        ]

        # Add JSON data
        json_data = json.dumps(transfer, indent=2)
        result_lines.extend(["", "JSON Data:", json_data])

        return [TextContent(type="text", text="\n".join(result_lines))]

    except Exception as e:
        return [TextContent(type="text", text=f"Hohmann transfer error: {str(e)}")]


async def _handle_orbital_rendezvous_planning(arguments: dict) -> list[TextContent]:
    """Handle orbital rendezvous planning."""
    try:
        from .integrations.orbits import OrbitElements, orbital_rendezvous_planning

        chaser_data = arguments.get("chaser_elements", {})
        target_data = arguments.get("target_elements", {})

        if not chaser_data or not target_data:
            return [TextContent(type="text", text="Error: both chaser_elements and target_elements are required")]

        # Create elements objects
        chaser_elements = OrbitElements(**chaser_data)
        target_elements = OrbitElements(**target_data)

        # Plan rendezvous
        plan = orbital_rendezvous_planning(chaser_elements, target_elements)

        # Format response
        result_lines = [
            "Orbital Rendezvous Planning",
            "=" * 40,
            "Spacecraft Orbits:",
            f"  Chaser:  {chaser_elements.semi_major_axis_m/1000:.1f} km × {chaser_elements.eccentricity:.3f} e × {chaser_elements.inclination_deg:.1f}° i",
            f"  Target:  {target_elements.semi_major_axis_m/1000:.1f} km × {target_elements.eccentricity:.3f} e × {target_elements.inclination_deg:.1f}° i",
            "",
            "Rendezvous Analysis:",
            f"  Relative Distance: {plan['relative_distance_km']:.1f} km",
            f"  Phase Angle: {plan['phase_angle_deg']:.1f}°",
            f"  Altitude Difference: {plan['altitude_difference_m']/1000:.1f} km",
            "",
            "Timing Analysis:",
            f"  Chaser Period: {plan['chaser_period_s']/3600:.2f} hours",
            f"  Target Period: {plan['target_period_s']/3600:.2f} hours",
            f"  Period Difference: {plan['period_difference_s']:.0f} seconds",
        ]

        if plan['phasing_time_h'] != float('inf'):
            result_lines.append(f"  Phasing Time: {plan['phasing_time_h']:.1f} hours")
        else:
            result_lines.append("  Phasing Time: ∞ (similar periods)")

        result_lines.extend([
            "",
            "Mission Assessment:",
            f"  Feasibility: {plan['feasibility']}",
            f"  Est. Circularization ΔV: {plan['estimated_circularization_dv_ms']:.1f} m/s"
        ])

        # Add recommendations
        if plan['feasibility'] == "Good":
            result_lines.append("  ✓ Favorable rendezvous conditions")
        else:
            result_lines.append("  ⚠ Challenging rendezvous - consider phasing orbits")

        # Add JSON data
        json_data = json.dumps(plan, indent=2)
        result_lines.extend(["", "JSON Data:", json_data])

        return [TextContent(type="text", text="\n".join(result_lines))]

    except Exception as e:
        return [TextContent(type="text", text=f"Rendezvous planning error: {str(e)}")]


async def _handle_genetic_algorithm_optimization(arguments: dict) -> list[TextContent]:
    """Handle genetic algorithm trajectory optimization."""
    try:
        from .integrations.gnc import (
            GeneticAlgorithm,
            GeneticAlgorithmParams,
            OptimizationConstraints,
            OptimizationObjective,
            TrajectoryWaypoint,
        )

        initial_trajectory_data = arguments.get("initial_trajectory", [])
        objective_data = arguments.get("objective", {})
        constraints_data = arguments.get("constraints", {})
        ga_params_data = arguments.get("ga_params", {})

        if not initial_trajectory_data or not objective_data:
            return [TextContent(type="text", text="Error: initial_trajectory and objective are required")]

        # Create objects
        initial_trajectory = [TrajectoryWaypoint(**wp) for wp in initial_trajectory_data]
        objective = OptimizationObjective(**objective_data)
        constraints = OptimizationConstraints(**constraints_data)
        ga_params = GeneticAlgorithmParams(**ga_params_data)

        # Run optimization
        ga = GeneticAlgorithm(ga_params)
        result = ga.optimize(initial_trajectory, objective, constraints)

        # Format response
        result_lines = [
            "Genetic Algorithm Trajectory Optimization",
            "=" * 55,
            f"Objective: {objective.type.replace('_', ' ').title()}",
            f"Waypoints: {len(initial_trajectory)}",
            "",
            "Algorithm Parameters:",
            f"  Population Size: {ga_params.population_size}",
            f"  Generations: {ga_params.generations}",
            f"  Mutation Rate: {ga_params.mutation_rate:.2f}",
            f"  Crossover Rate: {ga_params.crossover_rate:.2f}",
            "",
            "Optimization Results:",
            f"  Converged: {'✓ Yes' if result.converged else '✗ No'}",
            f"  Iterations: {result.iterations}",
            f"  Computation Time: {result.computation_time_s:.2f} seconds",
            f"  Optimal Cost: {result.optimal_cost:.3f}",
            "",
            "Trajectory Performance:",
            f"  Total ΔV: {result.delta_v_total_ms:.1f} m/s",
            f"  Fuel Mass: {result.fuel_mass_kg:.2f} kg",
            f"  Flight Time: {result.optimal_trajectory[-1].time_s - result.optimal_trajectory[0].time_s:.0f} seconds"
        ]

        if result.converged:
            result_lines.append("  ✓ Optimization successful")
        else:
            result_lines.append("  ⚠ Optimization may not have converged fully")

        # Add JSON data (summary only to avoid huge output)
        json_data = {
            "optimization_summary": {
                "algorithm": result.algorithm,
                "converged": result.converged,
                "iterations": result.iterations,
                "computation_time_s": result.computation_time_s,
                "optimal_cost": result.optimal_cost,
                "delta_v_total_ms": result.delta_v_total_ms,
                "fuel_mass_kg": result.fuel_mass_kg
            },
            "trajectory_length": len(result.optimal_trajectory)
        }
        result_lines.extend(["", "JSON Data (summary):", json.dumps(json_data, indent=2)])

        return [TextContent(type="text", text="\n".join(result_lines))]

    except Exception as e:
        return [TextContent(type="text", text=f"Genetic algorithm optimization error: {str(e)}")]


async def _handle_particle_swarm_optimization(arguments: dict) -> list[TextContent]:
    """Handle particle swarm optimization."""
    try:
        from .integrations.gnc import (
            OptimizationConstraints,
            OptimizationObjective,
            ParticleSwarmOptimizer,
            ParticleSwarmParams,
            TrajectoryWaypoint,
        )

        initial_trajectory_data = arguments.get("initial_trajectory", [])
        objective_data = arguments.get("objective", {})
        constraints_data = arguments.get("constraints", {})
        pso_params_data = arguments.get("pso_params", {})

        if not initial_trajectory_data or not objective_data:
            return [TextContent(type="text", text="Error: initial_trajectory and objective are required")]

        # Create objects
        initial_trajectory = [TrajectoryWaypoint(**wp) for wp in initial_trajectory_data]
        objective = OptimizationObjective(**objective_data)
        constraints = OptimizationConstraints(**constraints_data)
        pso_params = ParticleSwarmParams(**pso_params_data)

        # Run optimization
        pso = ParticleSwarmOptimizer(pso_params)
        result = pso.optimize(initial_trajectory, objective, constraints)

        # Format response
        result_lines = [
            "Particle Swarm Trajectory Optimization",
            "=" * 50,
            f"Objective: {objective.type.replace('_', ' ').title()}",
            f"Waypoints: {len(initial_trajectory)}",
            "",
            "Algorithm Parameters:",
            f"  Particles: {pso_params.num_particles}",
            f"  Max Iterations: {pso_params.max_iterations}",
            f"  Inertia Weight (w): {pso_params.w:.2f}",
            f"  Cognitive (c1): {pso_params.c1:.2f}",
            f"  Social (c2): {pso_params.c2:.2f}",
            "",
            "Optimization Results:",
            f"  Converged: {'✓ Yes' if result.converged else '✗ No'}",
            f"  Iterations: {result.iterations}",
            f"  Computation Time: {result.computation_time_s:.2f} seconds",
            f"  Optimal Cost: {result.optimal_cost:.3f}",
            "",
            "Trajectory Performance:",
            f"  Total ΔV: {result.delta_v_total_ms:.1f} m/s",
            f"  Fuel Mass: {result.fuel_mass_kg:.2f} kg",
            f"  Flight Time: {result.optimal_trajectory[-1].time_s - result.optimal_trajectory[0].time_s:.0f} seconds"
        ]

        # Add JSON data (summary only)
        json_data = {
            "optimization_summary": {
                "algorithm": result.algorithm,
                "converged": result.converged,
                "iterations": result.iterations,
                "computation_time_s": result.computation_time_s,
                "optimal_cost": result.optimal_cost,
                "delta_v_total_ms": result.delta_v_total_ms,
                "fuel_mass_kg": result.fuel_mass_kg
            },
            "trajectory_length": len(result.optimal_trajectory)
        }
        result_lines.extend(["", "JSON Data (summary):", json.dumps(json_data, indent=2)])

        return [TextContent(type="text", text="\n".join(result_lines))]

    except Exception as e:
        return [TextContent(type="text", text=f"Particle swarm optimization error: {str(e)}")]


async def _handle_porkchop_plot_analysis(arguments: dict) -> list[TextContent]:
    """Handle porkchop plot analysis requests."""
    try:
        from .integrations.orbits import porkchop_plot_analysis

        departure_body = arguments.get("departure_body", "Earth")
        arrival_body = arguments.get("arrival_body", "Mars")
        departure_dates = arguments.get("departure_dates")
        arrival_dates = arguments.get("arrival_dates")
        min_tof_days = arguments.get("min_tof_days", 100)
        max_tof_days = arguments.get("max_tof_days", 400)

        # Run porkchop analysis
        analysis = porkchop_plot_analysis(
            departure_body, arrival_body, departure_dates, arrival_dates, min_tof_days, max_tof_days
        )

        # Format response
        result_lines = [
            f"Porkchop Plot Analysis: {departure_body} to {arrival_body}",
            "=" * 60,
            "Transfer Opportunities Analysis",
            f"Time of Flight Range: {min_tof_days} - {max_tof_days} days",
            ""
        ]

        stats = analysis["summary_statistics"]
        if stats["feasible_transfers"] > 0:
            result_lines.extend([
                f"Feasible Transfers: {stats['feasible_transfers']} of {stats['total_transfers_computed']} computed",
                f"C3 Range: {stats['min_c3_km2_s2']:.2f} - {stats['max_c3_km2_s2']:.2f} km²/s²",
                f"Mean C3: {stats['mean_c3_km2_s2']:.2f} km²/s²",
                f"TOF Range: {stats['min_tof_days']:.0f} - {stats['max_tof_days']:.0f} days",
                ""
            ])

            if analysis["optimal_transfer"]:
                opt = analysis["optimal_transfer"]
                result_lines.extend([
                    "Optimal Transfer (Minimum C3):",
                    f"  Departure: {opt['departure_date']}",
                    f"  Arrival: {opt['arrival_date']}",
                    f"  Time of Flight: {opt['time_of_flight_days']:.1f} days",
                    f"  C3: {opt['c3_km2_s2']:.2f} km²/s²",
                    f"  Delta-V Estimate: {opt['delta_v_ms']/1000:.2f} km/s",
                    ""
                ])
        else:
            result_lines.extend([
                "⚠ No feasible transfers found in the specified date range.",
                "Consider adjusting time-of-flight constraints or date ranges.",
                ""
            ])

        # Add sample of transfer opportunities
        if analysis["transfer_grid"]:
            result_lines.extend([
                "Transfer Grid Sample (first 10 opportunities):",
                f"{'Departure':>12} {'Arrival':>12} {'TOF(d)':>8} {'C3':>8} {'Feasible':>10}",
                "-" * 65
            ])

            for i, transfer in enumerate(analysis["transfer_grid"][:10]):
                dep_short = transfer["departure_date"][:10]  # Just the date part
                arr_short = transfer["arrival_date"][:10]
                tof = transfer["time_of_flight_days"]
                c3 = transfer["c3_km2_s2"] if transfer["c3_km2_s2"] != float('inf') else 999.9
                feasible = "Yes" if transfer["transfer_feasible"] else "No"

                result_lines.append(f"{dep_short:>12} {arr_short:>12} {tof:8.1f} {c3:8.2f} {feasible:>10}")

        result_lines.extend([
            "",
            analysis["note"]
        ])

        # Add JSON data for the full transfer grid
        import json
        json_data = json.dumps(analysis, indent=2, default=str)
        result_lines.extend(["", "Full Transfer Grid JSON:", json_data])

        return [TextContent(type="text", text="\n".join(result_lines))]

    except Exception as e:
        return [TextContent(type="text", text=f"Porkchop analysis error: {str(e)}")]


async def _handle_monte_carlo_uncertainty_analysis(arguments: dict) -> list[TextContent]:
    """Handle Monte Carlo uncertainty analysis."""
    try:
        from .integrations.gnc import (
            TrajectoryWaypoint,
            monte_carlo_uncertainty_analysis,
        )

        nominal_trajectory_data = arguments.get("nominal_trajectory", [])
        uncertainty_params = arguments.get("uncertainty_params", {})
        n_samples = arguments.get("n_samples", 1000)

        if not nominal_trajectory_data:
            return [TextContent(type="text", text="Error: nominal_trajectory is required")]

        # Create trajectory waypoints
        nominal_trajectory = [TrajectoryWaypoint(**wp) for wp in nominal_trajectory_data]

        # Run Monte Carlo analysis
        analysis = monte_carlo_uncertainty_analysis(nominal_trajectory, uncertainty_params, n_samples)

        if "error" in analysis:
            return [TextContent(type="text", text=f"Monte Carlo analysis error: {analysis['error']}")]

        # Format response
        result_lines = [
            "Monte Carlo Uncertainty Analysis",
            "=" * 45,
            f"Nominal Trajectory: {len(nominal_trajectory)} waypoints",
            f"Monte Carlo Samples: {analysis['n_samples']}",
            "",
            "Delta-V Uncertainty:",
            f"  Mean: {analysis['delta_v_statistics']['mean_ms']:.1f} ± {analysis['delta_v_statistics']['std_ms']:.1f} m/s",
            f"  Range: {analysis['delta_v_statistics']['min_ms']:.1f} to {analysis['delta_v_statistics']['max_ms']:.1f} m/s",
            f"  95% Confidence: [{analysis['confidence_intervals']['delta_v_95_ms'][0]:.1f}, {analysis['confidence_intervals']['delta_v_95_ms'][1]:.1f}] m/s",
            "",
            "Flight Time Uncertainty:",
            f"  Mean: {analysis['flight_time_statistics']['mean_s']:.0f} ± {analysis['flight_time_statistics']['std_s']:.0f} seconds",
            f"  Range: {analysis['flight_time_statistics']['min_s']:.0f} to {analysis['flight_time_statistics']['max_s']:.0f} seconds",
            "",
            "Position Error Statistics:",
            f"  Mean Error: {analysis['position_error_statistics']['mean_m']:.0f} ± {analysis['position_error_statistics']['std_m']:.0f} m",
            f"  Maximum Error: {analysis['position_error_statistics']['max_m']:.0f} m",
            f"  95% Confidence: [{analysis['confidence_intervals']['position_95_m'][0]:.0f}, {analysis['confidence_intervals']['position_95_m'][1]:.0f}] m"
        ]

        # Assessment
        delta_v_uncertainty = analysis['delta_v_statistics']['std_ms'] / analysis['delta_v_statistics']['mean_ms'] * 100
        pos_uncertainty = analysis['position_error_statistics']['std_m']

        result_lines.extend([
            "",
            "Uncertainty Assessment:",
            f"  Delta-V Uncertainty: {delta_v_uncertainty:.1f}%",
            f"  Position Uncertainty: {pos_uncertainty/1000:.1f} km (1σ)"
        ])

        if delta_v_uncertainty < 5.0:
            result_lines.append("  ✓ Low delta-V uncertainty")
        elif delta_v_uncertainty < 15.0:
            result_lines.append("  ⚠ Moderate delta-V uncertainty")
        else:
            result_lines.append("  ⚠ High delta-V uncertainty - review mission design")

        if pos_uncertainty < 1000:
            result_lines.append("  ✓ Good position accuracy")
        elif pos_uncertainty < 10000:
            result_lines.append("  ⚠ Moderate position accuracy")
        else:
            result_lines.append("  ⚠ Large position uncertainties")

        # Add JSON data
        json_data = json.dumps(analysis, indent=2)
        result_lines.extend(["", "JSON Data:", json_data])

        return [TextContent(type="text", text="\n".join(result_lines))]

    except Exception as e:
        return [TextContent(type="text", text=f"Monte Carlo uncertainty analysis error: {str(e)}")]


def run_stdio():
    """Run the MCP server with stdio transport."""
    async def _main():
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options()
            )

    asyncio.run(_main())


def run_sse(host: str = "localhost", port: int = 8001):
    """Run the MCP server with SSE transport."""
    import mcp.server.sse

    sse_app = mcp.server.sse.SseServerTransport("/sse")

    async def _main():
        async with sse_app.run_server() as server_context:
            await server.run(
                server_context.read_stream,
                server_context.write_stream,
                server.create_initialization_options()
            )

    asyncio.run(_main())


def run():
    """Main entry point - defaults to stdio."""
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "sse":
        host = sys.argv[2] if len(sys.argv) > 2 else "localhost"
        port = int(sys.argv[3]) if len(sys.argv) > 3 else 8001
        run_sse(host, port)
    else:
        run_stdio()


if __name__ == "__main__":
    run()
