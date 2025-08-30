"""Atmospheric modeling tools for the Aerospace MCP server."""

import json
import logging
from typing import Literal

logger = logging.getLogger(__name__)


def get_atmosphere_profile(
    altitudes_m: list[float], model_type: Literal["ISA", "enhanced"] = "ISA"
) -> str:
    """Get atmospheric properties (pressure, temperature, density) at specified altitudes using ISA model.

    Args:
        altitudes_m: List of altitudes in meters
        model_type: Atmospheric model type ('ISA' for standard, 'enhanced' for extended)

    Returns:
        Formatted string with atmospheric profile data
    """
    try:
        from ..integrations.atmosphere import get_atmosphere_profile as _get_profile

        profile = _get_profile(altitudes_m, model_type)

        # Format response
        result_lines = [f"Atmospheric Profile ({model_type})", "=" * 50]
        result_lines.append(
            f"{'Alt (m)':>8} {'Press (Pa)':>12} {'Temp (K)':>9} {'Density':>10} {'Sound (m/s)':>12}"
        )
        result_lines.append("-" * 60)

        for point in profile:
            result_lines.append(
                f"{point.altitude_m:8.0f} {point.pressure_pa:12.1f} {point.temperature_k:9.2f} "
                f"{point.density_kg_m3:10.6f} {point.speed_of_sound_mps:12.1f}"
            )

        # Add JSON data for programmatic use
        json_data = json.dumps([p.model_dump() for p in profile], indent=2)
        result_lines.extend(["", "JSON Data:", json_data])

        return "\n".join(result_lines)

    except ImportError:
        return "Atmospheric modeling not available - install with: pip install ambiance"
    except Exception as e:
        logger.error(f"Atmosphere profile error: {str(e)}", exc_info=True)
        return f"Atmosphere profile error: {str(e)}"


def wind_model_simple(
    altitudes_m: list[float],
    surface_wind_speed_ms: float = 5.0,
    surface_wind_direction_deg: float = 270.0,
    model_type: Literal["logarithmic", "power_law"] = "logarithmic",
    roughness_length_m: float = 0.03,
) -> str:
    """Calculate wind speeds at different altitudes using logarithmic or power law models.

    Args:
        altitudes_m: List of altitudes in meters
        surface_wind_speed_ms: Wind speed at 10m reference height in m/s
        surface_wind_direction_deg: Wind direction at surface in degrees (0=North, 90=East)
        model_type: Wind model type ('logarithmic' or 'power_law')
        roughness_length_m: Surface roughness length in meters

    Returns:
        Formatted string with wind profile data
    """
    try:
        from ..integrations.atmosphere import wind_model_simple as _wind_model

        wind_profile = _wind_model(
            altitudes_m,
            surface_wind_speed_ms,
            surface_wind_direction_deg,
            model_type,
            roughness_length_m,
        )

        # Format response
        result_lines = [f"Wind Profile ({model_type} model)", "=" * 50]
        result_lines.extend(
            [
                f"Surface Reference: {surface_wind_speed_ms:.1f} m/s @ {surface_wind_direction_deg:.0f}° (10m height)",
                f"Roughness Length: {roughness_length_m:.3f} m",
                "",
                f"{'Alt (m)':>8} {'Speed (m/s)':>12} {'Dir (deg)':>10} {'Gust Factor':>12}",
            ]
        )
        result_lines.append("-" * 50)

        for point in wind_profile:
            result_lines.append(
                f"{point.altitude_m:8.0f} {point.wind_speed_ms:12.1f} {point.wind_direction_deg:10.0f} "
                f"{point.gust_factor:12.2f}"
            )

        # Add JSON data
        json_data = json.dumps([p.model_dump() for p in wind_profile], indent=2)
        result_lines.extend(["", "JSON Data:", json_data])

        return "\n".join(result_lines)

    except ImportError:
        return "Wind modeling not available - atmospheric integration required"
    except Exception as e:
        logger.error(f"Wind model error: {str(e)}", exc_info=True)
        return f"Wind model error: {str(e)}"
