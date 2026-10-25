"""Atmosphere Models and Wind Profiles.

Provides International Standard Atmosphere (ISA) calculations and simple
wind profile models for aerospace applications.

The ISA model defines temperature, pressure, and density as functions of
geopotential altitude H through a series of piecewise-linear temperature
layers.  All public functions take *geometric* altitude z (height above
mean sea level), which is converted first (US Standard Atmosphere 1976,
eq. 18)::

    H = r0 * z / (r0 + z),   r0 = 6,356,766 m

Two barometric formulas are then used:

    **Gradient layers** (non-zero lapse rate L)::

        P = P_base * (T / T_base) ^ (-g0 / (R * L))

    **Isothermal layers** (L = 0)::

        P = P_base * exp(-g0 * dh / (R * T_base))

Density is then computed from the ideal gas law::

    rho = P / (R * T)

and dynamic viscosity from Sutherland's law (US76 eq. 51)::

    mu = beta * T^1.5 / (T + S),   beta = 1.458e-6, S = 110.4 K

Below 86 km geometric altitude the ICAO/ISO ISA and the US Standard
Atmosphere 1976 (COESA) coincide, so ``"ISA"`` and ``"COESA"`` return the
same values.

Uses the ``ambiance`` library when it is installed (and the altitudes are
within its 0-81020 m range), otherwise the built-in implementation below.

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

# Effective Earth radius for the geometric -> geopotential altitude
# conversion (US Standard Atmosphere 1976).  Units: m.
EARTH_RADIUS_US76_M = 6356766.0

# Upper limit of the model: 86 km geometric = 84,852 m geopotential.
MAX_GEOMETRIC_ALTITUDE_M = 86000.0

# Upper limit of the ``ambiance`` package (geometric altitude, m).  Profiles
# that reach above it use the built-in implementation (same model).
AMBIANCE_MAX_ALTITUDE_M = 81020.0

# Sutherland's law constants for air (US Standard Atmosphere 1976).
SUTHERLAND_BETA = 1.458e-6  # kg / (m * s * K^0.5)
SUTHERLAND_S = 110.4  # K

# ===========================================================================
# ISA (International Standard Atmosphere) Layer Definitions
# ===========================================================================

# Each layer is defined by its base altitude (geopotential m), base
# temperature (K), and temperature lapse rate (K/m).  Negative lapse rate
# means temperature decreases with altitude.
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


def geometric_to_geopotential(altitude_m):
    """Convert geometric altitude to geopotential altitude (US76 eq. 18).

    ``H = r0 * z / (r0 + z)`` with ``r0 = 6,356,766 m``.

    Args:
        altitude_m: Geometric altitude(s) in meters (scalar or array).

    Returns:
        Geopotential altitude(s) in meters, same shape as the input.
    """
    z = np.asarray(altitude_m, dtype=np.float64)
    return EARTH_RADIUS_US76_M * z / (EARTH_RADIUS_US76_M + z)


def sutherland_viscosity(temperature_k):
    """Dynamic viscosity of air from Sutherland's law (US76 eq. 51).

    ``mu = beta * T^1.5 / (T + S)``; gives 1.7894e-5 Pa*s at 288.15 K.

    Args:
        temperature_k: Temperature(s) in Kelvin (scalar or array).

    Returns:
        Dynamic viscosity in Pa*s, same shape as the input.
    """
    T = np.asarray(temperature_k, dtype=np.float64)
    return SUTHERLAND_BETA * T**1.5 / (T + SUTHERLAND_S)


def _layer_base_pressures() -> np.ndarray:
    """Pressure at the base of each ISA layer, integrated up from 101325 Pa.

    Returns:
        Array of base pressures (Pa), one per entry of ``ISA_LAYERS``.
    """
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

    return layer_base_pressures


def _isa_manual_vectorized(
    altitudes: np.ndarray, geopotential: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute ISA pressure, temperature, and density for multiple altitudes.

    Algorithm:
        1. Convert geometric altitude z to geopotential altitude
           ``H = r0 * z / (r0 + z)`` (skipped when ``geopotential=True``).
        2. Pre-compute the base pressure at each layer boundary by
           integrating the barometric formula upward through each layer.
        3. For each query altitude, determine which layer it falls in.
        4. Apply the appropriate barometric formula from the layer base
           to the query altitude.
        5. Compute density from the ideal gas law.

    Barometric formulas:
        - **Gradient layer** (lapse rate L != 0):
          ``P = P_base * (T/T_base)^(-g0 / (R*L))``
        - **Isothermal layer** (L = 0):
          ``P = P_base * exp(-g0 * dh / (R * T_base))``

    Args:
        altitudes: Array of geometric altitudes in meters (geopotential
            altitudes when ``geopotential=True``).
        geopotential: Treat ``altitudes`` as geopotential altitudes.

    Returns:
        Tuple of ``(pressures, temperatures, densities)`` as NumPy arrays
        in SI units (Pa, K, kg/m^3).
    """
    altitudes = np.atleast_1d(np.asarray(altitudes, dtype=np.float64))
    h = altitudes if geopotential else geometric_to_geopotential(altitudes)

    layer_base_pressures = _layer_base_pressures()

    # Layer index of each altitude: the last layer whose base is <= h
    layer_idx = np.searchsorted(ISA_LAYER_ALTITUDES, h, side="right") - 1
    layer_idx = np.clip(layer_idx, 0, len(ISA_LAYERS) - 1)

    h_base = ISA_LAYER_ALTITUDES[layer_idx]
    T_base = ISA_LAYER_TEMPS[layer_idx]
    lapse = ISA_LAYER_LAPSE_RATES[layer_idx]
    p_base = layer_base_pressures[layer_idx]
    dh = h - h_base  # Height above layer base

    # Gradient layers: T varies linearly, P follows a power law.
    # Isothermal layers: T = const, P decays exponentially.
    isothermal = np.abs(lapse) < 1e-10
    temperatures = T_base + lapse * dh
    safe_lapse = np.where(isothermal, 1.0, lapse)  # avoid 1/0 in unused branch
    p_ratio = np.where(
        isothermal,
        np.exp(-G0 * dh / (R_SPECIFIC * T_base)),
        (temperatures / T_base) ** (-G0 / (R_SPECIFIC * safe_lapse)),
    )

    pressures = p_base * p_ratio
    # Ideal gas law: rho = P / (R * T)
    densities = pressures / (R_SPECIFIC * temperatures)

    return pressures, temperatures, densities


def _isa_manual(altitude_m: float) -> tuple[float, float, float]:
    """Compute ISA properties at a single altitude (scalar wrapper).

    Delegates to ``_isa_manual_vectorized`` for consistency.

    Args:
        altitude_m: Geometric altitude in meters.

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
    ``"ISA"`` uses the ``ambiance`` package when it is installed and every
    altitude is within its range (0-81020 m); otherwise, and for
    ``"COESA"``, the built-in US Standard Atmosphere 1976 implementation is
    used.  The two models are identical below 86 km.

    Args:
        altitudes_m: List of geometric altitudes in meters (0-86000 m)
        model_type: Atmosphere model, "ISA" or "COESA" (case-insensitive)

    Returns:
        List of AtmospherePoint objects with pressure, temperature, density,
        speed of sound and dynamic viscosity.
    """
    model = str(model_type).upper()
    if model not in ["ISA", "COESA"]:
        raise ValueError(f"Unknown model type: {model_type}. Use 'ISA' or 'COESA'")

    # Convert to NumPy array for vectorized operations
    altitudes = np.asarray(altitudes_m, dtype=np.float64).reshape(-1)

    # Validate altitude range
    alt_numpy = to_numpy(altitudes)
    if np.any(~np.isfinite(alt_numpy)) or np.any(
        (alt_numpy < 0) | (alt_numpy > MAX_GEOMETRIC_ALTITUDE_M)
    ):
        raise ValueError(
            f"Altitude out of ISA range (0-{MAX_GEOMETRIC_ALTITUDE_M:.0f}m)"
        )

    if alt_numpy.size == 0:
        return []

    use_ambiance = (
        AMBIANCE_AVAILABLE
        and model == "ISA"
        and float(alt_numpy.max()) <= AMBIANCE_MAX_ALTITUDE_M
    )

    if use_ambiance:
        # One vectorised ambiance call for the whole profile
        atm = ambiance.Atmosphere(alt_numpy)
        columns = [
            np.asarray(atm.pressure, dtype=np.float64).reshape(-1),
            np.asarray(atm.temperature, dtype=np.float64).reshape(-1),
            np.asarray(atm.density, dtype=np.float64).reshape(-1),
            np.asarray(atm.speed_of_sound, dtype=np.float64).reshape(-1),
            np.asarray(atm.dynamic_viscosity, dtype=np.float64).reshape(-1),
        ]
    else:
        # Use vectorized manual calculation
        pressures, temperatures, densities = _isa_manual_vectorized(altitudes)
        speeds_of_sound = np.sqrt(GAMMA * R_SPECIFIC * temperatures)
        viscosities = sutherland_viscosity(temperatures)
        columns = [
            to_numpy(arr)
            for arr in (
                pressures,
                temperatures,
                densities,
                speeds_of_sound,
                viscosities,
            )
        ]

    pressures, temperatures, densities, speeds_of_sound, viscosities = columns

    # Convert to output format
    return [
        AtmospherePoint(
            altitude_m=float(alt_numpy[i]),
            pressure_pa=float(pressures[i]),
            temperature_k=float(temperatures[i]),
            density_kg_m3=float(densities[i]),
            speed_of_sound_mps=float(speeds_of_sound[i]),
            viscosity_pa_s=float(viscosities[i]),
        )
        for i in range(alt_numpy.size)
    ]


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

        U(z) = U_ref * ln(z / z0) / ln(z_ref / z0)   for z > z0
        U(z) = 0                                     for z <= z0

    where z0 is the aerodynamic roughness length and z_ref is the
    measurement reference height (0 < z0 < z_ref is required).

    **Power law**::

        U(z) = U_ref * (z / z_ref) ^ alpha           for z > 0

    where alpha is the wind shear exponent (0.143 for open terrain, the
    1/7 law).  The roughness length is not used by the power law.

    Both laws apply at every height above the surface, including heights
    below the reference height.

    Args:
        altitudes_m: Altitude points for wind calculation (m ASL).
        surface_wind_mps: Wind speed at reference height (m/s).
        surface_altitude_m: Ground elevation (m ASL).
        model: ``"logarithmic"`` or ``"power"`` law (``"power_law"`` is
            accepted as an alias).
        roughness_length_m: Aerodynamic roughness length z0 (m).
            Typical values: 0.0002 (water), 0.03 (grass), 0.1 (crops),
            1.0 (suburban).
        reference_height_m: Height of surface wind measurement (m AGL).

    Returns:
        List of wind profile points with wind speed at each altitude.
    """
    if model == "power_law":
        model = "power"
    if model not in ["logarithmic", "power"]:
        raise ValueError(
            f"Unknown wind model: {model}. Use 'logarithmic' or 'power' (or 'power_law')"
        )

    if reference_height_m <= 0:
        raise ValueError("Reference height must be positive")

    if model == "logarithmic":
        if roughness_length_m <= 0:
            raise ValueError("Roughness length must be positive")
        if roughness_length_m >= reference_height_m:
            raise ValueError(
                f"Roughness length ({roughness_length_m} m) must be smaller than "
                f"the reference height ({reference_height_m} m)"
            )

    # Convert to NumPy array for vectorized operations
    altitudes = np.asarray(altitudes_m, dtype=np.float64).reshape(-1)
    heights_agl = altitudes - surface_altitude_m

    # Zero wind at and below the surface (and below z0 for the log law)
    wind_speeds = np.zeros_like(altitudes)

    if model == "logarithmic":
        # Logarithmic wind profile (neutral stability):
        # U(z) = U_ref * ln(z/z0) / ln(z_ref/z0)
        above = heights_agl > roughness_length_m
        log_ratio_ref = np.log(reference_height_m / roughness_length_m)
        wind_speeds[above] = surface_wind_mps * (
            np.log(heights_agl[above] / roughness_length_m) / log_ratio_ref
        )
    else:  # power law
        # Power law: U(z) = U_ref * (z/z_ref)^alpha
        # alpha = 0.143 (1/7 power law, Davenport, open terrain)
        alpha = 0.143
        above = heights_agl > 0
        wind_speeds[above] = (
            surface_wind_mps * (heights_agl[above] / reference_height_m) ** alpha
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
        altitude_m: Geometric altitude in meters

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
