"""Aerodynamics analysis tools for the Aerospace MCP server."""

import json
import logging

logger = logging.getLogger(__name__)


def wing_vlm_analysis(
    wing_config: dict, flight_conditions: dict, analysis_options: dict | None = None
) -> str:
    """Analyze wing aerodynamics using Vortex Lattice Method or simplified lifting line theory.

    Args:
        wing_config: Wing configuration (span_m, chord_m, sweep_deg, etc.)
        flight_conditions: Flight conditions (airspeed_ms, altitude_m, alpha_deg)
        analysis_options: Optional analysis settings

    Returns:
        Formatted string with aerodynamic analysis results
    """
    try:
        from ..integrations.aero import wing_vlm_analysis as _wing_analysis

        result = _wing_analysis(wing_config, flight_conditions, analysis_options)

        # Format response
        result_lines = ["Wing Aerodynamic Analysis (VLM)", "=" * 50]

        # Wing configuration
        result_lines.extend(
            [
                "Configuration:",
                f"  Span: {wing_config.get('span_m', 0):.1f} m",
                f"  Chord: {wing_config.get('chord_m', 0):.1f} m",
                f"  Aspect Ratio: {result.get('aspect_ratio', 0):.1f}",
                f"  Sweep: {wing_config.get('sweep_deg', 0):.1f}°",
                "",
            ]
        )

        # Flight conditions
        result_lines.extend(
            [
                "Flight Conditions:",
                f"  Airspeed: {flight_conditions.get('airspeed_ms', 0):.1f} m/s",
                f"  Altitude: {flight_conditions.get('altitude_m', 0):.0f} m",
                f"  Angle of Attack: {flight_conditions.get('alpha_deg', 0):.1f}°",
                "",
            ]
        )

        # Results
        result_lines.extend(
            [
                "Aerodynamic Results:",
                f"  Lift Coefficient: {result.get('cl', 0):.3f}",
                f"  Drag Coefficient: {result.get('cd', 0):.4f}",
                f"  L/D Ratio: {result.get('cl_cd_ratio', 0):.1f}",
                f"  Lift (N): {result.get('lift_n', 0):.0f}",
                f"  Drag (N): {result.get('drag_n', 0):.0f}",
                "",
            ]
        )

        # Add JSON data
        json_data = json.dumps(result, indent=2)
        result_lines.extend(["JSON Data:", json_data])

        return "\n".join(result_lines)

    except ImportError:
        return "Wing analysis not available - install aerodynamics packages"
    except Exception as e:
        logger.error(f"Wing analysis error: {str(e)}", exc_info=True)
        return f"Wing analysis error: {str(e)}"


def airfoil_polar_analysis(
    airfoil_name: str,
    reynolds_number: float = 1000000,
    mach_number: float = 0.1,
    alpha_range_deg: list[float] | None = None,
) -> str:
    """Generate airfoil polar data (CL, CD, CM vs alpha) using database or advanced methods.

    Args:
        airfoil_name: Airfoil name (e.g., 'NACA2412', 'NACA0012')
        reynolds_number: Reynolds number
        mach_number: Mach number
        alpha_range_deg: Optional angle of attack range, defaults to [-10, 20] deg

    Returns:
        Formatted string with airfoil polar data
    """
    try:
        from ..integrations.aero import airfoil_polar_analysis as _airfoil_analysis

        alpha_range_deg = alpha_range_deg or list(range(-10, 21, 2))

        result = _airfoil_analysis(
            airfoil_name, reynolds_number, mach_number, alpha_range_deg
        )

        # Format response
        result_lines = [f"Airfoil Polar Analysis: {airfoil_name}", "=" * 60]
        result_lines.extend(
            [
                f"Reynolds Number: {reynolds_number:.0e}",
                f"Mach Number: {mach_number:.3f}",
                "",
                f"{'Alpha (°)':>8} {'CL':>8} {'CD':>8} {'CM':>8} {'L/D':>8}",
            ]
        )
        result_lines.append("-" * 50)

        for point in result.get("polar_data", []):
            result_lines.append(
                f"{point.get('alpha_deg', 0):8.1f} {point.get('cl', 0):8.4f} "
                f"{point.get('cd', 0):8.5f} {point.get('cm', 0):8.4f} "
                f"{point.get('cl_cd_ratio', 0):8.1f}"
            )

        # Add JSON data
        json_data = json.dumps(result, indent=2)
        result_lines.extend(["", "JSON Data:", json_data])

        return "\n".join(result_lines)

    except ImportError:
        return "Airfoil analysis not available - install aerodynamics packages"
    except Exception as e:
        logger.error(f"Airfoil analysis error: {str(e)}", exc_info=True)
        return f"Airfoil analysis error: {str(e)}"


def calculate_stability_derivatives(wing_config: dict, flight_conditions: dict) -> str:
    """Calculate basic longitudinal stability derivatives for a wing.

    Args:
        wing_config: Wing configuration parameters
        flight_conditions: Flight conditions

    Returns:
        JSON string with stability derivatives
    """
    try:
        from ..integrations.aero import calculate_stability_derivatives as _stability

        result = _stability(wing_config, flight_conditions)
        return json.dumps(result, indent=2)

    except ImportError:
        return "Stability analysis not available - install aerodynamics packages"
    except Exception as e:
        logger.error(f"Stability analysis error: {str(e)}", exc_info=True)
        return f"Stability analysis error: {str(e)}"


def get_airfoil_database() -> str:
    """Get available airfoil database with aerodynamic coefficients.

    Returns:
        JSON string with airfoil database
    """
    try:
        from ..integrations.aero import AIRFOIL_DATABASE

        return json.dumps(AIRFOIL_DATABASE, indent=2)

    except ImportError:
        return "Airfoil database not available"
    except Exception as e:
        logger.error(f"Airfoil database error: {str(e)}", exc_info=True)
        return f"Airfoil database error: {str(e)}"
