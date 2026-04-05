"""Atmosphere Models and Wind Profiles.

Provides International Standard Atmosphere (ISA) calculations and simple
wind profile models for aerospace applications.

The ISA model defines temperature, pressure, and density as functions of
geopotential altitude through a series of piecewise-linear temperature
layers.  Two barometric formulas are used:

    **Gradient layers** (non-zero lapse rate L)::

        P = P_base * (T / T_base) ^ (-g0 / (R * L))

    **Isothermal layers** (L = 0)::

        P = P_base * exp(-g0 * dh / (R * T_base))

Density is then computed from the ideal gas law::

    rho = P / (R * T)

Falls back to manual ISA calculations when the ``ambiance`` library is
unavailable.

Uses NumPy for vectorized calculations with CuPy compatibility for GPU
acceleration via the ``_array_backend`` module.

References:
    - ICAO Document 7488/3, "Manual of the ICAO Standard Atmosphere" (1993)
    - ISO 2533:1975, "Standard Atmosphere"
    - U.S. Standard Atmosphere, 1976 (COESA)

WARNING: This module is for educational and research purposes only.
Do NOT use for real flight planning, navigation, or aircraft operations.
"""

from pydantic import BaseModel, Field

from . import update_availability
from ._array_backend import np, to_numpy

# ===========================================================================
# Physical Constants
# ===========================================================================

# Specific gas constant for dry air: R = R_universal / M_air.
# Units: J / (kg * K).
R_SPECIFIC = 287.0528

# Ratio of specific heats for dry air: gamma = Cp / Cv.
# Dimensionless.  Used for speed of sound: a = sqrt(gamma * R * T).
GAMMA = 1.4

# Standard gravitational acceleration (m/s^2).
G0 = 9.80665

# ===========================================================================
# ISA (International Standard Atmosphere) Layer Definitions
# ===========================================================================

# Each layer is defined by its base altitude (m), base temperature (K),
# and temperature lapse rate (K/m).  Negative lapse rate means temperature
# decreases with altitude.
#
# The 7 standard layers from sea level to 84,852 m (mesopause) are:
#   Layer 0: Troposphere        (0-11 km)     L = -6.5  K/km
#   Layer 1: Tropopause         (11-20 km)    L =  0    K/km (isothermal)
#   Layer 2: Stratosphere I     (20-32 km)    L = +1.0  K/km
#   Layer 3: Stratosphere II    (32-47 km)    L = +2.8  K/km
#   Layer 4: Stratopause        (47-51 km)    L =  0    K/km (isothermal)
#   Layer 5: Mesosphere I       (51-71 km)    L = -2.8  K/km
#   Layer 6: Mesosphere II      (71-84.852 km) L = -2.0 K/km

# NumPy array format for vectorized lookups
ISA_LAYER_ALTITUDES = np.array([0, 11000, 20000, 32000, 47000, 51000, 71000, 84852])
ISA_LAYER_TEMPS = np.array(
    [288.15, 216.65, 216.65, 228.65, 270.65, 270.65, 214.65, 186.946]
)
ISA_LAYER_LAPSE_RATES = np.array(
    [-0.0065, 0.0, 0.001, 0.0028, 0.0, -0.0028, -0.002, 0.0]
)

# Legacy tuple format: (base_altitude_m, base_temperature_K, lapse_rate_K_per_m)
ISA_LAYERS = [
    (0, 288.15, -0.0065),  # Troposphere
    (11000, 216.65, 0.0),  # Tropopause (isothermal)
    (20000, 216.65, 0.001),  # Stratosphere I
    (32000, 228.65, 0.0028),  # Stratosphere II
    (47000, 270.65, 0.0),  # Stratopause (isothermal)
    (51000, 270.65, -0.0028),  # Mesosphere I
    (71000, 214.65, -0.002),  # Mesosphere II
    (84852, 186.946, 0.0),  # Mesopause
]

# ===========================================================================
# Optional Library Imports
# ===========================================================================

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


# ===========================================================================
# Data Models
# ===========================================================================


class AtmospherePoint(BaseModel):
    """Atmospheric conditions at a single altitude."""

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


# ===========================================================================
# ISA Calculation Functions
# ===========================================================================


def _isa_manual_vectorized(
    altitudes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute ISA pressure, temperature, and density for multiple altitudes.

    Algorithm:
        1. Pre-compute the base pressure at each layer boundary by
           integrating the barometric formula upward through each layer.
        2. For each query altitude, determine which layer it falls in.
        3. Apply the appropriate barometric formula from the layer base
           to the query altitude.
        4. Compute density from the ideal gas law.

    Barometric formulas:
        - **Gradient layer** (lapse rate L != 0):
          ``P = P_base * (T/T_base)^(-g0 / (R*L))``
        - **Isothermal layer** (L = 0):
          ``P = P_base * exp(-g0 * dh / (R * T_base))``

    Args:
        altitudes: Array of geopotential altitudes in meters.

    Returns:
        Tuple of ``(pressures, temperatures, densities)`` as NumPy arrays
        in SI units (Pa, K, kg/m^3).
    """
    altitudes = np.asarray(altitudes)
    n = len(altitudes)

    # Output arrays
    pressures = np.zeros(n)
    temperatures = np.zeros(n)
    densities = np.zeros(n)

    # Pre-compute base pressures at each ISA layer boundary.
    # Start from sea-level standard pressure: 101325 Pa.
    layer_base_pressures = np.zeros(len(ISA_LAYERS))
    layer_base_pressures[0] = 101325.0  # ISA sea-level pressure (Pa)

    for j in range(len(ISA_LAYERS) - 1):
        h_base, T_base, lapse_rate = ISA_LAYERS[j]
        h_top = ISA_LAYERS[j + 1][0]
        dh = h_top - h_base

        if abs(lapse_rate) < 1e-10:
            # Isothermal layer: P = P_base * exp(-g0 * dh / (R * T))
            layer_base_pressures[j + 1] = layer_base_pressures[j] * np.exp(
                -G0 * dh / (R_SPECIFIC * T_base)
            )
        else:
            # Gradient layer: P = P_base * (T_top / T_base)^(-g0 / (R * L))
            T_top = T_base + lapse_rate * dh
            layer_base_pressures[j + 1] = layer_base_pressures[j] * (
                T_top / T_base
            ) ** (-G0 / (R_SPECIFIC * lapse_rate))

    # Compute P, T, rho for each query altitude
    for idx, h in enumerate(to_numpy(altitudes)):
        # Determine which ISA layer this altitude falls in
        layer_idx = 0
        for i in range(len(ISA_LAYERS) - 1):
            if h >= ISA_LAYERS[i + 1][0]:
                layer_idx = i + 1
            else:
                break

        h_base, T_base, lapse_rate = ISA_LAYERS[layer_idx]
        p_base = layer_base_pressures[layer_idx]
        dh = h - h_base  # Height above layer base

        if abs(lapse_rate) < 1e-10:
            # Isothermal: T = const, P decays exponentially
            T = T_base
            p_ratio = np.exp(-G0 * dh / (R_SPECIFIC * T_base))
        else:
            # Gradient: T varies linearly, P follows power law
            T = T_base + lapse_rate * dh
            p_ratio = (T / T_base) ** (-G0 / (R_SPECIFIC * lapse_rate))

        pressure = p_base * p_ratio
        # Ideal gas law: rho = P / (R * T)
        density = pressure / (R_SPECIFIC * T)

        pressures[idx] = pressure
        temperatures[idx] = T
        densities[idx] = density

    return pressures, temperatures, densities


def _isa_manual(altitude_m: float) -> tuple[float, float, float]:
    """Compute ISA properties at a single altitude (scalar wrapper).

    Delegates to ``_isa_manual_vectorized`` for consistency.

    Args:
        altitude_m: Geopotential altitude in meters.

    Returns:
        Tuple of ``(pressure_pa, temperature_k, density_kg_m3)``.
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
            # Use .item() to extract scalar from 0-d arrays (NumPy 1.25+ deprecation fix)
            point = AtmospherePoint(
                altitude_m=float(altitude),
                pressure_pa=float(np.asarray(atm.pressure).item()),
                temperature_k=float(np.asarray(atm.temperature).item()),
                density_kg_m3=float(np.asarray(atm.density).item()),
                speed_of_sound_mps=float(np.asarray(atm.speed_of_sound).item()),
                viscosity_pa_s=(
                    float(np.asarray(atm.dynamic_viscosity).item())
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


# ===========================================================================
# Wind Profile Models
# ===========================================================================


def wind_model_simple(
    altitudes_m: list[float],
    surface_wind_mps: float,
    surface_altitude_m: float = 0.0,
    model: str = "logarithmic",
    roughness_length_m: float = 0.1,
    reference_height_m: float = 10.0,
) -> list[WindPoint]:
    """Simple wind profile models for low-altitude boundary-layer studies.

    Two models are supported:

    **Logarithmic** (neutral atmospheric stability)::

        U(z) = U_ref * ln(z / z0) / ln(z_ref / z0)

    where z0 is the aerodynamic roughness length and z_ref is the
    measurement reference height.

    **Power law**::

        U(z) = U_ref * (z / z_ref) ^ alpha

    where alpha is the wind shear exponent (default 0.143 for open
    terrain, per Davenport).

    Args:
        altitudes_m: Altitude points for wind calculation (m ASL).
        surface_wind_mps: Wind speed at reference height (m/s).
        surface_altitude_m: Ground elevation (m ASL).
        model: ``"logarithmic"`` or ``"power"`` law.
        roughness_length_m: Aerodynamic roughness length z0 (m).
            Typical values: 0.0002 (water), 0.03 (grass), 0.1 (crops),
            1.0 (suburban).
        reference_height_m: Height of surface wind measurement (m AGL).

    Returns:
        List of wind profile points with wind speed at each altitude.
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
        # Logarithmic wind profile (neutral stability):
        # U(z) = U_ref * ln(z/z0) / ln(z_ref/z0)
        log_ratio_ref = np.log(reference_height_m / roughness_length_m)
        wind_speeds[above_ref] = surface_wind_mps * (
            np.log(heights_agl[above_ref] / roughness_length_m) / log_ratio_ref
        )
    else:  # power law
        # Power law: U(z) = U_ref * (z/z_ref)^alpha
        # alpha = 0.143 (1/7 power law, Davenport, open terrain)
        alpha = 0.143
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
