"""
Atmosphere Models and Wind Profiles

Provides ISA/COESA atmosphere models and wind profile calculations.
Falls back to manual ISA calculations when optional dependencies unavailable.
"""

import math

from pydantic import BaseModel, Field

from . import update_availability

# Constants
R_SPECIFIC = 287.0528  # J/(kg·K) - specific gas constant for dry air
GAMMA = 1.4  # ratio of specific heats
G0 = 9.80665  # m/s² - standard gravity

# ISA Standard atmosphere layers (altitude_m, temp_K, lapse_rate_K_per_m)
ISA_LAYERS = [
    (0, 288.15, -0.0065),  # Troposphere
    (11000, 216.65, 0.0),  # Tropopause
    (20000, 216.65, 0.001),  # Stratosphere 1
    (32000, 228.65, 0.0028),  # Stratosphere 2
    (47000, 270.65, 0.0),  # Stratopause
    (51000, 270.65, -0.0028),  # Mesosphere 1
    (71000, 214.65, -0.002),  # Mesosphere 2
    (84852, 186.946, 0.0),  # Mesopause
]

# Optional library imports
AMBIANCE_AVAILABLE = False
try:
    import ambiance

    AMBIANCE_AVAILABLE = True
    # Try to get version, but don't fail if not available
    try:
        version = ambiance.__version__
    except AttributeError:
        version = "unknown"
    update_availability("atmosphere", True, {"ambiance": version})
except ImportError:
    update_availability("atmosphere", True, {})  # Still available with manual ISA


# Data models
class AtmospherePoint(BaseModel):
    """Single atmosphere condition point."""

    altitude_m: float = Field(..., description="Geometric altitude in meters")
    pressure_pa: float = Field(..., description="Static pressure in Pascals")
    temperature_k: float = Field(..., description="Temperature in Kelvin")
    density_kg_m3: float = Field(..., description="Air density in kg/m³")
    speed_of_sound_mps: float = Field(..., description="Speed of sound in m/s")
    viscosity_pa_s: float | None = Field(None, description="Dynamic viscosity in Pa·s")


class WindPoint(BaseModel):
    """Single wind profile point."""

    altitude_m: float = Field(..., description="Altitude in meters")
    wind_speed_mps: float = Field(..., description="Wind speed in m/s")
    wind_direction_deg: float | None = Field(
        None, description="Wind direction in degrees"
    )


def _isa_manual(altitude_m: float) -> tuple[float, float, float]:
    """
    Manual ISA calculation for fallback when ambiance unavailable.
    Returns: (pressure_pa, temperature_k, density_kg_m3)
    """
    h = altitude_m

    # Find appropriate layer
    for i, (h_base, T_base, lapse_rate) in enumerate(ISA_LAYERS):
        if i == len(ISA_LAYERS) - 1 or h < ISA_LAYERS[i + 1][0]:
            break

    dh = h - h_base

    if abs(lapse_rate) < 1e-10:  # Isothermal layer
        T = T_base
        p_ratio = math.exp(-G0 * dh / (R_SPECIFIC * T_base))
    else:  # Temperature gradient layer
        T = T_base + lapse_rate * dh
        p_ratio = (T / T_base) ** (-G0 / (R_SPECIFIC * lapse_rate))

    # Get base pressure from previous layer
    p_base = 101325.0  # Sea level pressure
    for j in range(i):
        h_layer_base, T_layer_base, lr = ISA_LAYERS[j]
        h_layer_top = ISA_LAYERS[j + 1][0]
        dh_layer = h_layer_top - h_layer_base

        if abs(lr) < 1e-10:
            p_base *= math.exp(-G0 * dh_layer / (R_SPECIFIC * T_layer_base))
        else:
            T_layer_top = T_layer_base + lr * dh_layer
            p_base *= (T_layer_top / T_layer_base) ** (-G0 / (R_SPECIFIC * lr))

    pressure = p_base * p_ratio
    density = pressure / (R_SPECIFIC * T)

    return pressure, T, density


def get_atmosphere_profile(
    altitudes_m: list[float], model_type: str = "ISA"
) -> list[AtmospherePoint]:
    """
    Get atmospheric properties at specified altitudes.

    Args:
        altitudes_m: List of geometric altitudes in meters (0-86000m)
        model_type: Atmosphere model ("ISA", "COESA") - currently only ISA supported

    Returns:
        List of AtmospherePoint objects with pressure, temperature, density, etc.
    """
    if model_type not in ["ISA", "COESA"]:
        raise ValueError(f"Unknown model type: {model_type}. Use 'ISA' or 'COESA'")

    results = []

    for altitude in altitudes_m:
        if altitude < 0 or altitude > 86000:
            raise ValueError(f"Altitude {altitude}m out of ISA range (0-86000m)")

        if AMBIANCE_AVAILABLE and model_type == "ISA":
            # Use ambiance library if available
            atm = ambiance.Atmosphere(altitude)
            point = AtmospherePoint(
                altitude_m=altitude,
                pressure_pa=float(atm.pressure),
                temperature_k=float(atm.temperature),
                density_kg_m3=float(atm.density),
                speed_of_sound_mps=float(atm.speed_of_sound),
                viscosity_pa_s=float(atm.dynamic_viscosity)
                if hasattr(atm, "dynamic_viscosity")
                else None,
            )
        else:
            # Fall back to manual calculation
            pressure, temperature, density = _isa_manual(altitude)
            speed_of_sound = math.sqrt(GAMMA * R_SPECIFIC * temperature)

            point = AtmospherePoint(
                altitude_m=altitude,
                pressure_pa=pressure,
                temperature_k=temperature,
                density_kg_m3=density,
                speed_of_sound_mps=speed_of_sound,
                viscosity_pa_s=None,  # Not calculated in manual mode
            )

        results.append(point)

    return results


def wind_model_simple(
    altitudes_m: list[float],
    surface_wind_mps: float,
    surface_altitude_m: float = 0.0,
    model: str = "logarithmic",
    roughness_length_m: float = 0.1,
    reference_height_m: float = 10.0,
) -> list[WindPoint]:
    """
    Simple wind profile models for low-altitude studies.

    Args:
        altitudes_m: Altitude points for wind calculation
        surface_wind_mps: Wind speed at reference height
        surface_altitude_m: Surface elevation
        model: "logarithmic" or "power" law
        roughness_length_m: Surface roughness length (for logarithmic)
        reference_height_m: Height of surface wind measurement

    Returns:
        List of WindPoint objects with wind speeds
    """
    if model not in ["logarithmic", "power"]:
        raise ValueError(f"Unknown wind model: {model}. Use 'logarithmic' or 'power'")

    results = []

    for altitude in altitudes_m:
        height_agl = altitude - surface_altitude_m

        if height_agl < 0:
            wind_speed = 0.0  # Below ground
        elif height_agl < reference_height_m:
            # Linear interpolation below reference height
            wind_speed = surface_wind_mps * (height_agl / reference_height_m)
        else:
            if model == "logarithmic":
                # Logarithmic wind profile
                if roughness_length_m <= 0:
                    raise ValueError("Roughness length must be positive")

                wind_speed = surface_wind_mps * (
                    math.log(height_agl / roughness_length_m)
                    / math.log(reference_height_m / roughness_length_m)
                )
            else:  # power law
                # Power law with typical exponent
                alpha = 0.143  # Typical for open terrain
                wind_speed = (
                    surface_wind_mps * (height_agl / reference_height_m) ** alpha
                )

        results.append(
            WindPoint(
                altitude_m=altitude,
                wind_speed_mps=max(0.0, wind_speed),  # Ensure non-negative
            )
        )

    return results


def get_atmospheric_properties(altitude_m: float) -> dict[str, float]:
    """
    Get atmospheric properties at a single altitude (convenience function).

    Args:
        altitude_m: Altitude in meters

    Returns:
        Dictionary with atmospheric properties
    """
    profile = get_atmosphere_profile([altitude_m])
    point = profile[0]

    return {
        "altitude_m": point.altitude_m,
        "pressure_pa": point.pressure_pa,
        "temperature_k": point.temperature_k,
        "temperature_c": point.temperature_k - 273.15,
        "density_kg_m3": point.density_kg_m3,
        "speed_of_sound_mps": point.speed_of_sound_mps,
        "viscosity_pa_s": point.viscosity_pa_s,
    }
