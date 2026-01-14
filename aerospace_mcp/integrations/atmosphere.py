"""
Atmosphere Models and Wind Profiles

Provides ISA/COESA atmosphere models and wind profile calculations.
Falls back to manual ISA calculations when optional dependencies unavailable.

Uses NumPy for vectorized calculations with CuPy compatibility for GPU acceleration.
"""

from pydantic import BaseModel, Field

from . import update_availability
from ._array_backend import np, to_numpy

# Constants
R_SPECIFIC = 287.0528  # J/(kg·K) - specific gas constant for dry air
GAMMA = 1.4  # ratio of specific heats
G0 = 9.80665  # m/s² - standard gravity

# ISA Standard atmosphere layers (altitude_m, temp_K, lapse_rate_K_per_m)
# Stored as NumPy arrays for vectorized lookups
ISA_LAYER_ALTITUDES = np.array([0, 11000, 20000, 32000, 47000, 51000, 71000, 84852])
ISA_LAYER_TEMPS = np.array(
    [288.15, 216.65, 216.65, 228.65, 270.65, 270.65, 214.65, 186.946]
)
ISA_LAYER_LAPSE_RATES = np.array(
    [-0.0065, 0.0, 0.001, 0.0028, 0.0, -0.0028, -0.002, 0.0]
)

# Legacy list format for compatibility
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


def _isa_manual_vectorized(
    altitudes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorized ISA calculation for multiple altitudes.
    Returns: (pressure_pa, temperature_k, density_kg_m3) as arrays
    """
    altitudes = np.asarray(altitudes)
    n = len(altitudes)

    # Output arrays
    pressures = np.zeros(n)
    temperatures = np.zeros(n)
    densities = np.zeros(n)

    # Pre-compute base pressures at each layer boundary
    layer_base_pressures = np.zeros(len(ISA_LAYERS))
    layer_base_pressures[0] = 101325.0  # Sea level

    for j in range(len(ISA_LAYERS) - 1):
        h_base, T_base, lapse_rate = ISA_LAYERS[j]
        h_top = ISA_LAYERS[j + 1][0]
        dh = h_top - h_base

        if abs(lapse_rate) < 1e-10:  # Isothermal
            layer_base_pressures[j + 1] = layer_base_pressures[j] * np.exp(
                -G0 * dh / (R_SPECIFIC * T_base)
            )
        else:  # Temperature gradient
            T_top = T_base + lapse_rate * dh
            layer_base_pressures[j + 1] = layer_base_pressures[j] * (
                T_top / T_base
            ) ** (-G0 / (R_SPECIFIC * lapse_rate))

    # Process each altitude
    for idx, h in enumerate(to_numpy(altitudes)):
        # Find layer index
        layer_idx = 0
        for i in range(len(ISA_LAYERS) - 1):
            if h >= ISA_LAYERS[i + 1][0]:
                layer_idx = i + 1
            else:
                break

        h_base, T_base, lapse_rate = ISA_LAYERS[layer_idx]
        p_base = layer_base_pressures[layer_idx]
        dh = h - h_base

        if abs(lapse_rate) < 1e-10:  # Isothermal
            T = T_base
            p_ratio = np.exp(-G0 * dh / (R_SPECIFIC * T_base))
        else:  # Temperature gradient
            T = T_base + lapse_rate * dh
            p_ratio = (T / T_base) ** (-G0 / (R_SPECIFIC * lapse_rate))

        pressure = p_base * p_ratio
        density = pressure / (R_SPECIFIC * T)

        pressures[idx] = pressure
        temperatures[idx] = T
        densities[idx] = density

    return pressures, temperatures, densities


def _isa_manual(altitude_m: float) -> tuple[float, float, float]:
    """
    Manual ISA calculation for fallback when ambiance unavailable.
    Returns: (pressure_pa, temperature_k, density_kg_m3)
    """
    # Use vectorized version for consistency
    pressures, temperatures, densities = _isa_manual_vectorized(np.array([altitude_m]))
    return float(pressures[0]), float(temperatures[0]), float(densities[0])


def get_atmosphere_profile(
    altitudes_m: list[float], model_type: str = "ISA"
) -> list[AtmospherePoint]:
    """
    Get atmospheric properties at specified altitudes.

    Uses vectorized NumPy calculations for efficient batch processing.

    Args:
        altitudes_m: List of geometric altitudes in meters (0-81020m when using ambiance, 0-86000m for manual ISA)
        model_type: Atmosphere model ("ISA", "COESA") - currently only ISA supported

    Returns:
        List of AtmospherePoint objects with pressure, temperature, density, etc.
    """
    if model_type not in ["ISA", "COESA"]:
        raise ValueError(f"Unknown model type: {model_type}. Use 'ISA' or 'COESA'")

    # Convert to NumPy array for vectorized operations
    altitudes = np.asarray(altitudes_m, dtype=np.float64)

    # Validate altitude range
    max_altitude = 81020 if AMBIANCE_AVAILABLE else 86000
    alt_numpy = to_numpy(altitudes)
    if np.any(alt_numpy < 0) or np.any(alt_numpy > max_altitude):
        range_str = f"0-{max_altitude}m"
        raise ValueError(f"Altitude out of ISA range ({range_str})")

    results = []

    if AMBIANCE_AVAILABLE and model_type == "ISA":
        # Use ambiance library if available (already vectorized internally)
        for altitude in alt_numpy:
            atm = ambiance.Atmosphere(altitude)
            point = AtmospherePoint(
                altitude_m=float(altitude),
                pressure_pa=float(atm.pressure),
                temperature_k=float(atm.temperature),
                density_kg_m3=float(atm.density),
                speed_of_sound_mps=float(atm.speed_of_sound),
                viscosity_pa_s=(
                    float(atm.dynamic_viscosity)
                    if hasattr(atm, "dynamic_viscosity")
                    else None
                ),
            )
            results.append(point)
    else:
        # Use vectorized manual calculation
        pressures, temperatures, densities = _isa_manual_vectorized(altitudes)
        speeds_of_sound = np.sqrt(GAMMA * R_SPECIFIC * temperatures)

        # Convert to output format
        for i, altitude in enumerate(alt_numpy):
            point = AtmospherePoint(
                altitude_m=float(altitude),
                pressure_pa=float(pressures[i]),
                temperature_k=float(temperatures[i]),
                density_kg_m3=float(densities[i]),
                speed_of_sound_mps=float(speeds_of_sound[i]),
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

    Uses vectorized NumPy calculations for efficient batch processing.

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

    if model == "logarithmic" and roughness_length_m <= 0:
        raise ValueError("Roughness length must be positive")

    # Convert to NumPy array for vectorized operations
    altitudes = np.asarray(altitudes_m, dtype=np.float64)
    heights_agl = altitudes - surface_altitude_m

    # Initialize wind speeds
    wind_speeds = np.zeros_like(altitudes)

    # Below ground mask
    below_ground = heights_agl < 0
    wind_speeds[below_ground] = 0.0

    # Below reference height - linear interpolation
    below_ref = (heights_agl >= 0) & (heights_agl < reference_height_m)
    wind_speeds[below_ref] = (
        surface_wind_mps * heights_agl[below_ref] / reference_height_m
    )

    # Above reference height
    above_ref = heights_agl >= reference_height_m

    if model == "logarithmic":
        # Logarithmic wind profile
        log_ratio_ref = np.log(reference_height_m / roughness_length_m)
        wind_speeds[above_ref] = surface_wind_mps * (
            np.log(heights_agl[above_ref] / roughness_length_m) / log_ratio_ref
        )
    else:  # power law
        # Power law with typical exponent
        alpha = 0.143  # Typical for open terrain
        wind_speeds[above_ref] = (
            surface_wind_mps * (heights_agl[above_ref] / reference_height_m) ** alpha
        )

    # Ensure non-negative
    wind_speeds = np.maximum(wind_speeds, 0.0)

    # Convert to output format
    results = []
    alt_numpy = to_numpy(altitudes)
    ws_numpy = to_numpy(wind_speeds)

    for i, altitude in enumerate(alt_numpy):
        results.append(
            WindPoint(
                altitude_m=float(altitude),
                wind_speed_mps=float(ws_numpy[i]),
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
