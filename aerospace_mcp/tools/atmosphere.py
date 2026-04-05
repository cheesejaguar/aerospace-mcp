"""Atmospheric modeling tools for the Aerospace MCP server.

Provides tools for computing atmospheric properties (pressure, temperature,
density, speed of sound) using the International Standard Atmosphere (ISA)
model, as well as simplified wind profile models (logarithmic and power-law)
for estimating wind speeds at various altitudes above ground level.

WARNING: This module is for educational and research purposes only.
Do NOT use for real flight planning, navigation, or aircraft operations.
"""

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
        Formatted string with atmospheric profile data including pressure,
        temperature, density, and speed of sound at each altitude.

    Raises:
        No exceptions are raised directly; errors are returned as formatted strings.
        ImportError is caught when the ``ambiance`` package is not installed.
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
        Formatted string with wind profile data at each requested altitude.

    Raises:
        No exceptions are raised directly; errors are returned as formatted strings.

    Note:
        The **logarithmic wind profile** (Ref: Stull, "Meteorology for Scientists
        and Engineers", 2000) models wind speed as:
            U(z) = (u* / kappa) * ln(z / z0)
        where u* is friction velocity, kappa ~0.4 is the von Karman constant,
        and z0 is the aerodynamic roughness length.

        The **power-law wind profile** (empirical approximation) models wind as:
            U(z) = U_ref * (z / z_ref) ^ alpha
        where alpha (Hellmann exponent) depends on terrain roughness, typically
        ~0.14 for open terrain and ~0.40 for urban areas.
    """
    try:
        from ..integrations.atmosphere import wind_model_simple as _wind_model

        # Call integration function with correct argument mapping.
        # The integration layer implements the logarithmic or power-law profile
        # equations described above, using 10 m as the standard reference height.
        wind_profile = _wind_model(
            altitudes_m,
            surface_wind_speed_ms,
            0.0,  # surface_altitude_m
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
                f"{'Alt (m)':>8} {'Speed (m/s)':>12} {'Dir (deg)':>10}",
            ]
        )
        result_lines.append("-" * 40)

        for point in wind_profile:
            # Use correct attribute name (wind_speed_mps not wind_speed_ms)
            # Direction is assumed constant (from surface_wind_direction_deg)
            direction = (
                point.wind_direction_deg
                if point.wind_direction_deg is not None
                else surface_wind_direction_deg
            )
            result_lines.append(
                f"{point.altitude_m:8.0f} {point.wind_speed_mps:12.1f} {direction:10.0f}"
            )

        # Add JSON data with direction filled in
        json_output = [
            {
                "altitude_m": p.altitude_m,
                "wind_speed_mps": p.wind_speed_mps,
                "wind_direction_deg": (
                    p.wind_direction_deg
                    if p.wind_direction_deg is not None
                    else surface_wind_direction_deg
                ),
            }
            for p in wind_profile
        ]
        json_data = json.dumps(json_output, indent=2)
        result_lines.extend(["", "JSON Data:", json_data])

        return "\n".join(result_lines)

    except ImportError:
        return "Wind modeling not available - atmospheric integration required"
    except Exception as e:
        logger.error(f"Wind model error: {str(e)}", exc_info=True)
        return f"Wind model error: {str(e)}"
