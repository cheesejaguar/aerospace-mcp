"""Aircraft performance tools for the Aerospace MCP server.

Provides tools for aircraft performance calculations including:
- Density altitude computation from pressure altitude and temperature.
- Airspeed conversions between IAS, CAS, EAS, TAS, and Mach number.
- Stall speed calculation for clean, takeoff, and landing configurations.
- Weight and balance (CG position and limit checks).
- Takeoff and landing field length estimation.
- Fuel reserve requirements per FAR/JAR/ICAO regulations.

All atmospheric calculations use the International Standard Atmosphere (ISA)
model with the barometric formula for pressure variation with altitude.

WARNING: This module is for educational and research purposes only.
Do NOT use for real flight planning, navigation, or aircraft operations.
"""

import json
import logging
import math
from typing import Literal

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ISA (International Standard Atmosphere) Constants
# ---------------------------------------------------------------------------
# Reference: ICAO Doc 7488/3, "Manual of the ICAO Standard Atmosphere"
# These define sea-level conditions and troposphere properties for the ISA model.

G0 = 9.80665  # m/s^2 -- Standard acceleration due to gravity (exact by definition)
R_AIR = 287.05  # J/(kg*K) -- Specific gas constant for dry air (R_universal / M_air)
GAMMA = 1.4  # dimensionless -- Ratio of specific heats (Cp/Cv) for diatomic air
P0 = 101325.0  # Pa -- ISA sea-level standard pressure (1 atm = 101325 Pa)
T0 = 288.15  # K -- ISA sea-level standard temperature (15 deg C)
RHO0 = 1.225  # kg/m^3 -- ISA sea-level standard air density (P0 / (R_AIR * T0))
A0 = 340.29  # m/s -- ISA sea-level speed of sound (sqrt(GAMMA * R_AIR * T0))
LAPSE_RATE = 0.0065  # K/m -- Temperature lapse rate in the troposphere (0-11 km)


def _get_isa_conditions(altitude_m: float) -> dict:
    """Get ISA atmospheric conditions at a given geopotential altitude.

    Uses the barometric formula to compute pressure, temperature, density,
    and speed of sound. The troposphere (0-11 km) has a linear temperature
    lapse; the tropopause/lower stratosphere (11-20 km) is isothermal.

    Args:
        altitude_m: Geopotential altitude in meters. Negative values are
            clamped to 0.

    Returns:
        Dict with pressure_pa, temperature_k, temperature_c, density_kg_m3,
        speed_of_sound_ms, and dimensionless ratios (pressure, density,
        temperature) relative to sea-level ISA values.
    """
    if altitude_m < 0:
        altitude_m = 0

    if altitude_m <= 11000:
        # Troposphere (0 to 11 km): linear temperature decrease.
        # T = T0 - L * h  where L = LAPSE_RATE = 0.0065 K/m
        T = T0 - LAPSE_RATE * altitude_m
        # Barometric formula for troposphere (with lapse rate):
        # P = P0 * (T / T0) ^ (g0 / (R * L))
        # Exponent g0/(R*L) = 9.80665 / (287.05 * 0.0065) = 5.2559
        P = P0 * (T / T0) ** (G0 / (R_AIR * LAPSE_RATE))
    else:
        # Tropopause / lower stratosphere (11 km+): isothermal at 216.65 K.
        # First compute conditions at the tropopause boundary (11 km).
        T_trop = T0 - LAPSE_RATE * 11000  # 216.65 K
        P_trop = P0 * (T_trop / T0) ** (G0 / (R_AIR * LAPSE_RATE))
        T = T_trop
        # Barometric formula for isothermal layer:
        # P = P_trop * exp(-g0 * (h - h_trop) / (R * T_trop))
        P = P_trop * math.exp(-G0 * (altitude_m - 11000) / (R_AIR * T_trop))

    # Ideal gas law: rho = P / (R * T)
    rho = P / (R_AIR * T)
    # Speed of sound in an ideal gas: a = sqrt(gamma * R * T)
    a = math.sqrt(GAMMA * R_AIR * T)

    return {
        "pressure_pa": P,
        "temperature_k": T,
        "temperature_c": T - 273.15,
        "density_kg_m3": rho,
        "speed_of_sound_ms": a,
        "pressure_ratio": P / P0,
        "density_ratio": rho / RHO0,
        "temperature_ratio": T / T0,
    }


# ---------------------------------------------------------------------------
# Unit conversions (exact definitions)
# ---------------------------------------------------------------------------
KTS_TO_MS = 1852.0 / 3600.0  # 1 kt = 1 NM/h = 1852 m / 3600 s
MS_TO_KTS = 3600.0 / 1852.0
M_TO_FT = 1.0 / 0.3048  # 1 ft = 0.3048 m exactly

# ---------------------------------------------------------------------------
# Compressible-flow airspeed relations (ISA, gamma = 1.4)
# ---------------------------------------------------------------------------
# Impact pressure qc = pt - p sensed by a pitot tube, as a ratio qc/p:
#   subsonic (St. Venant):  qc/p = (1 + 0.2 M^2)^3.5 - 1
#   supersonic (Rayleigh pitot formula, normal shock ahead of the probe):
#                           qc/p = 166.9216 M^7 / (7 M^2 - 1)^2.5 - 1
# Calibrated airspeed is the speed that produces the same qc at sea-level
# ISA conditions:  qc = P0 * f(CAS / a0).  Mach follows from qc and the
# static pressure p at the pressure altitude.  Both relations are
# temperature-independent; temperature only enters TAS = M * a(T).
# Reference: e.g. Anderson, "Introduction to Flight", ch. 4 (pitot-static
# measurement of airspeed); USAF TO 1F-1A-1 / NASA RP-1046 airspeed tables.
_QC_RATIO_AT_M1 = 1.2**3.5 - 1.0  # 0.89293, where both branches meet


def _qc_over_p(mach: float) -> float:
    """Impact pressure ratio qc/p for a pitot probe at the given Mach."""
    if mach <= 1.0:
        return (1.0 + 0.2 * mach * mach) ** 3.5 - 1.0
    return 166.9216 * mach**7 / (7.0 * mach * mach - 1.0) ** 2.5 - 1.0


def _mach_from_qc_over_p(ratio: float) -> float:
    """Invert ``_qc_over_p``: closed form subsonic, bisection supersonic."""
    if ratio <= 0.0:
        return 0.0
    if ratio <= _QC_RATIO_AT_M1:
        return math.sqrt(5.0 * ((ratio + 1.0) ** (2.0 / 7.0) - 1.0))
    lo, hi = 1.0, 2.0
    while _qc_over_p(hi) < ratio:  # bracket the root
        hi *= 2.0
    for _ in range(80):  # ~1e-24 relative bracket width
        mid = 0.5 * (lo + hi)
        if _qc_over_p(mid) < ratio:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _cas_to_mach(cas_ms: float, pressure_pa: float) -> float:
    """Mach number for a calibrated airspeed at static pressure ``p``."""
    qc = P0 * _qc_over_p(cas_ms / A0)
    return _mach_from_qc_over_p(qc / pressure_pa)


def _mach_to_cas(mach: float, pressure_pa: float) -> float:
    """Calibrated airspeed (m/s) for a Mach number at static pressure ``p``."""
    qc = pressure_pa * _qc_over_p(mach)
    return A0 * _mach_from_qc_over_p(qc / P0)


def _density_altitude_m(rho: float) -> float:
    """Exact ISA density altitude: the height where rho_ISA(h) equals ``rho``.

    Inverts the module's ISA model (``_get_isa_conditions``): in the
    troposphere rho/rho0 = (T/T0)^(g/(R L) - 1) with T = T0 - L h, and in
    the isothermal layer above 11 km rho = rho11 * exp(-g (h - 11000) / (R T11)).
    Densities above sea-level ISA give a negative density altitude.
    """
    rho0_isa = P0 / (R_AIR * T0)
    tropopause = _get_isa_conditions(11000.0)
    rho11 = tropopause["density_kg_m3"]
    if rho >= rho11:
        n = G0 / (R_AIR * LAPSE_RATE) - 1.0  # 4.2559
        return (T0 / LAPSE_RATE) * (1.0 - (rho / rho0_isa) ** (1.0 / n))
    t11 = tropopause["temperature_k"]
    return 11000.0 + (R_AIR * t11 / G0) * math.log(rho11 / rho)


def _vapor_pressure_pa(dewpoint_c: float) -> float:
    """Saturation vapour pressure over water at the dewpoint (Magnus-Tetens)."""
    return 610.78 * 10.0 ** (7.5 * dewpoint_c / (dewpoint_c + 237.3))


def density_altitude_calculator(
    pressure_altitude_ft: float,
    temperature_c: float,
    dewpoint_c: float | None = None,
) -> str:
    """Calculate density altitude from pressure altitude and temperature.

    Density altitude is the altitude in the standard atmosphere at which
    the air density equals the actual air density at the given conditions.
    Essential for aircraft performance calculations.

    Args:
        pressure_altitude_ft: Pressure altitude in feet
        temperature_c: Outside air temperature in Celsius
        dewpoint_c: Optional dewpoint for the humidity correction (moist-air
            density via the virtual temperature)

    The density altitude is found by inverting the ISA density profile
    exactly (troposphere and the isothermal layer above 11 km) rather than
    with the 118.8 ft/degC rule of thumb.

    Returns:
        Formatted string with density altitude calculation results including
        air density, density ratio (sigma), pressure ratio (delta), and ISA
        deviation.

    Raises:
        No exceptions are raised directly; errors are returned as formatted strings.
    """
    try:
        # Convert to SI
        pressure_alt_m = pressure_altitude_ft * 0.3048
        temp_k = temperature_c + 273.15

        # ISA temperature at pressure altitude
        if pressure_alt_m <= 11000:
            isa_temp_k = T0 - LAPSE_RATE * pressure_alt_m
        else:
            isa_temp_k = 216.65  # Tropopause temperature

        # Temperature deviation from ISA
        delta_isa = temp_k - isa_temp_k

        # Calculate actual pressure from pressure altitude using the ISA
        # barometric formula. Pressure altitude is defined as the altitude in
        # the ISA that corresponds to the observed station pressure.
        if pressure_alt_m <= 11000:
            # Troposphere barometric formula: P = P0 * (1 - L*h/T0)^(g/(R*L))
            p = P0 * (1 - LAPSE_RATE * pressure_alt_m / T0) ** (
                G0 / (R_AIR * LAPSE_RATE)
            )
        else:
            # Isothermal layer above tropopause
            p_trop = P0 * (1 - LAPSE_RATE * 11000 / T0) ** (G0 / (R_AIR * LAPSE_RATE))
            p = p_trop * math.exp(-G0 * (pressure_alt_m - 11000) / (R_AIR * 216.65))

        if temp_k <= 0:
            return "Error: temperature_c must be above absolute zero"

        # Dry-air density from the ideal gas law
        rho_dry = p / (R_AIR * temp_k)
        rho = rho_dry

        # Humidity correction (optional): water vapour is lighter than dry
        # air, so moist air at the same p and T is less dense.  Use the
        # virtual temperature Tv = T / (1 - (e/p)(1 - eps)), eps = Rd/Rv =
        # 0.622, with the vapour pressure e at the dewpoint, and evaluate
        # the density as p / (Rd * Tv).
        vapor_pressure_pa = None
        virtual_temp_k = None
        if dewpoint_c is not None:
            vapor_pressure_pa = _vapor_pressure_pa(dewpoint_c)
            virtual_temp_k = temp_k / (1.0 - (vapor_pressure_pa / p) * (1.0 - 0.622))
            rho = p / (R_AIR * virtual_temp_k)

        # Density altitude: the ISA altitude whose standard density equals
        # the actual density, found by inverting the ISA density profile
        # exactly (no 118.8 ft/degC rule of thumb, valid above 11 km too).
        density_alt_m = _density_altitude_m(rho)
        density_alt_ft = density_alt_m * M_TO_FT
        humidity_correction_ft = density_alt_ft - _density_altitude_m(rho_dry) * M_TO_FT

        result = {
            "input": {
                "pressure_altitude_ft": pressure_altitude_ft,
                "pressure_altitude_m": round(pressure_alt_m, 1),
                "temperature_c": temperature_c,
                "dewpoint_c": dewpoint_c,
            },
            "density_altitude_ft": round(density_alt_ft, 0),
            "density_altitude_m": round(density_alt_m, 0),
            "air_density_kg_m3": round(rho, 5),
            "density_ratio_sigma": round(rho / RHO0, 5),
            "pressure_ratio_delta": round(p / P0, 5),
            "isa_deviation_c": round(delta_isa, 1),
            "isa_temp_at_pressure_alt_c": round(isa_temp_k - 273.15, 1),
            "humidity_correction_ft": round(humidity_correction_ft, 0),
            "method": (
                "exact inversion of ISA density; humidity via virtual temperature"
            ),
        }
        if dewpoint_c is not None:
            result["vapor_pressure_hpa"] = round(vapor_pressure_pa / 100.0, 2)
            result["virtual_temperature_c"] = round(virtual_temp_k - 273.15, 2)

        dewpoint_str = (
            f"{dewpoint_c:.1f}°C" if dewpoint_c is not None else "Not specified"
        )

        output = f"""
DENSITY ALTITUDE CALCULATION
============================
Input Conditions:
  Pressure Altitude: {pressure_altitude_ft:,.0f} ft ({pressure_alt_m:,.0f} m)
  Temperature: {temperature_c:.1f}°C
  ISA Temperature at PA: {isa_temp_k - 273.15:.1f}°C
  ISA Deviation: {delta_isa:+.1f}°C
  Dewpoint: {dewpoint_str}

Results:
  Density Altitude: {density_alt_ft:,.0f} ft ({density_alt_m:,.0f} m)
  Air Density: {rho:.5f} kg/m³
  Density Ratio (σ): {rho / RHO0:.5f}
  Pressure Ratio (δ): {p / P0:.5f}

Performance Impact:
  {"⚠️ HIGH DENSITY ALTITUDE - Reduced engine and aerodynamic performance" if density_alt_ft > 5000 else "✓ Normal performance expected"}

{json.dumps(result, indent=2)}
"""
        return output.strip()

    except Exception as e:
        logger.error(f"Density altitude calculation error: {str(e)}", exc_info=True)
        return f"Error calculating density altitude: {str(e)}"


def true_airspeed_converter(
    speed_value: float,
    speed_type: Literal["IAS", "CAS", "EAS", "TAS", "MACH"],
    altitude_ft: float,
    temperature_c: float | None = None,
    position_error_kts: float = 0,
) -> str:
    """Convert between IAS, CAS, EAS, TAS, and Mach number.

    Args:
        speed_value: Input speed value (knots for airspeeds, dimensionless for Mach)
        speed_type: Input type - "IAS", "CAS", "EAS", "TAS", or "MACH"
        altitude_ft: Pressure altitude in feet
        temperature_c: Outside air temperature in Celsius (uses ISA if not provided)
        position_error_kts: Position error correction in knots (IAS to CAS)

    All conversions go through Mach number with the compressible ISA pitot
    relations (qc = P0 * ((1 + 0.2 (CAS/a0)^2)^3.5 - 1), M from qc and the
    static pressure at the pressure altitude; Rayleigh pitot formula above
    M 1), then TAS = M * a and EAS = TAS * sqrt(rho / rho0).  A non-ISA
    ``temperature_c`` changes the density and speed of sound (TAS, EAS for a
    given CAS/Mach) but not the CAS <-> Mach relation.

    Returns:
        Formatted string with all equivalent airspeeds (IAS, CAS, EAS, TAS, Mach),
        dynamic pressure, impact pressure, and atmospheric conditions.

    Raises:
        No exceptions are raised directly; errors are returned as formatted strings.
    """
    try:
        # Convert altitude to meters
        altitude_m = altitude_ft * 0.3048

        # Get atmospheric conditions
        isa = _get_isa_conditions(altitude_m)
        p = isa["pressure_pa"]
        rho = isa["density_kg_m3"]
        a = isa["speed_of_sound_ms"]

        # Use actual temperature if provided (non-ISA day).  Static pressure
        # still follows from the pressure altitude; only density and the
        # speed of sound change.
        if temperature_c is not None:
            temp_k = temperature_c + 273.15
            if temp_k <= 0:
                return "Error: temperature_c must be above absolute zero"
            # Recalculate density with actual temperature
            rho = p / (R_AIR * temp_k)
            a = math.sqrt(GAMMA * R_AIR * temp_k)
        else:
            temperature_c = isa["temperature_c"]
            temp_k = isa["temperature_k"]

        if speed_value < 0:
            return "Error: speed_value must be >= 0"

        # Conversion constants
        kts_to_ms = KTS_TO_MS
        ms_to_kts = MS_TO_KTS

        # Reduce every input to Mach number first.  CAS <-> Mach uses the
        # compressible pitot relations (qc from CAS at sea-level ISA, Mach
        # from qc and the static pressure p at the pressure altitude); these
        # do not depend on temperature.  TAS = M * a uses the actual (or ISA)
        # temperature, and EAS = TAS * sqrt(rho / rho0).
        if speed_type in ("IAS", "CAS"):
            # IAS + position error = CAS
            cas_in_kts = (
                speed_value + position_error_kts if speed_type == "IAS" else speed_value
            )
            if cas_in_kts < 0:
                return "Error: IAS + position_error_kts must be >= 0"
            mach = _cas_to_mach(cas_in_kts * kts_to_ms, p)
        elif speed_type == "EAS":
            # TAS = EAS / sqrt(sigma), sigma from the actual density
            mach = (speed_value * kts_to_ms) / math.sqrt(rho / RHO0) / a
        elif speed_type == "TAS":
            mach = (speed_value * kts_to_ms) / a
        elif speed_type == "MACH":
            mach = speed_value
        else:
            return f"Error: Unknown speed type '{speed_type}'"

        # Mach -> all other speeds
        tas_ms = mach * a
        tas_kts = tas_ms * ms_to_kts
        eas_ms = tas_ms * math.sqrt(rho / RHO0)
        eas_kts = eas_ms * ms_to_kts
        cas_kts = _mach_to_cas(mach, p) * ms_to_kts
        if speed_type in ("IAS", "CAS"):
            cas_kts = cas_in_kts  # exact round trip for the input value

        # CAS to IAS
        ias_kts = cas_kts - position_error_kts

        # Impact pressure (what the pitot-static system measures)
        qc = p * _qc_over_p(mach)

        # Dynamic pressure
        q = 0.5 * rho * tas_ms**2

        result = {
            "input": {
                "value": speed_value,
                "type": speed_type,
                "altitude_ft": altitude_ft,
                "temperature_c": round(temperature_c, 1),
                "position_error_kts": position_error_kts,
            },
            "IAS_kts": round(ias_kts, 1),
            "CAS_kts": round(cas_kts, 1),
            "EAS_kts": round(eas_kts, 1),
            "TAS_kts": round(tas_kts, 1),
            "TAS_ms": round(tas_ms, 2),
            "TAS_kmh": round(tas_ms * 3.6, 1),
            "MACH": round(mach, 4),
            "dynamic_pressure_pa": round(q, 1),
            "dynamic_pressure_psf": round(q * 0.020885, 2),
            "impact_pressure_pa": round(qc, 1),
            "method": "compressible ISA pitot relations via Mach",
            "atmospheric_conditions": {
                "pressure_pa": round(p, 0),
                "temperature_c": round(temperature_c, 1),
                "density_kg_m3": round(rho, 5),
                "speed_of_sound_ms": round(a, 1),
                "speed_of_sound_kts": round(a * ms_to_kts, 1),
            },
        }

        output = f"""
AIRSPEED CONVERSION
===================
Input: {speed_value:.1f} {speed_type} at {altitude_ft:,.0f} ft, {temperature_c:.1f}°C

Equivalent Airspeeds:
  IAS: {ias_kts:>8.1f} kts  (Indicated Airspeed)
  CAS: {cas_kts:>8.1f} kts  (Calibrated Airspeed)
  EAS: {eas_kts:>8.1f} kts  (Equivalent Airspeed)
  TAS: {tas_kts:>8.1f} kts  (True Airspeed)
  TAS: {tas_ms:>8.1f} m/s
  Mach: {mach:>7.4f}

Dynamic Pressure: {q:,.1f} Pa ({q * 0.020885:.2f} psf)
Impact Pressure (qc): {qc:,.1f} Pa

Atmospheric Conditions:
  Pressure: {p:,.0f} Pa
  Temperature: {temperature_c:.1f}°C ({temp_k:.1f} K)
  Density: {rho:.5f} kg/m³
  Speed of Sound: {a:.1f} m/s ({a * ms_to_kts:.1f} kts)

{json.dumps(result, indent=2)}
"""
        return output.strip()

    except Exception as e:
        logger.error(f"Airspeed conversion error: {str(e)}", exc_info=True)
        return f"Error converting airspeed: {str(e)}"


def stall_speed_calculator(
    weight_kg: float,
    wing_area_m2: float,
    cl_max_clean: float,
    cl_max_takeoff: float | None = None,
    cl_max_landing: float | None = None,
    altitude_ft: float = 0,
    load_factor: float = 1.0,
) -> str:
    """Calculate stall speeds for different aircraft configurations.

    Args:
        weight_kg: Aircraft weight in kg
        wing_area_m2: Wing reference area in m²
        cl_max_clean: Maximum lift coefficient in clean configuration
        cl_max_takeoff: Max CL with takeoff flaps (optional)
        cl_max_landing: Max CL with landing flaps (optional)
        altitude_ft: Pressure altitude in feet
        load_factor: Load factor (default 1.0 for level flight)

    Returns:
        Formatted string with stall speed calculations for each configuration,
        plus reference speeds (VREF, V2_min) and altitude correction.

    Raises:
        No exceptions are raised directly; errors are returned as formatted strings.
    """
    try:
        # Validate inputs: the lift equation divides by S and CL_max.
        positive = {
            "weight_kg": weight_kg,
            "wing_area_m2": wing_area_m2,
            "cl_max_clean": cl_max_clean,
        }
        if cl_max_takeoff is not None:
            positive["cl_max_takeoff"] = cl_max_takeoff
        if cl_max_landing is not None:
            positive["cl_max_landing"] = cl_max_landing
        for name, value in positive.items():
            if not value > 0:
                return f"Error: {name} must be > 0 (got {value})"
        if load_factor == 0:
            return "Error: load_factor must be non-zero"

        # Get atmospheric conditions
        altitude_m = altitude_ft * 0.3048
        isa = _get_isa_conditions(altitude_m)
        rho = isa["density_kg_m3"]

        # Weight in Newtons
        weight_n = weight_kg * G0

        # Stall speed from the lift equation at maximum CL:
        # L = W = 0.5 * rho * V^2 * S * CL_max  =>  V_s = sqrt(2*W*n / (rho*S*CL_max))
        def calc_vs(cl_max: float) -> float:
            """Calculate stall speed in m/s from the lift equation."""
            if cl_max <= 0 or rho <= 0 or wing_area_m2 <= 0:
                return 0
            return math.sqrt(
                (2 * weight_n * abs(load_factor)) / (rho * wing_area_m2 * cl_max)
            )

        # Default CL values if not provided
        if cl_max_takeoff is None:
            cl_max_takeoff = cl_max_clean * 1.2  # ~20% increase with takeoff flaps
        if cl_max_landing is None:
            cl_max_landing = cl_max_clean * 1.5  # ~50% increase with landing flaps

        # Calculate stall speeds
        vs1 = calc_vs(cl_max_clean)  # Clean stall speed
        vs_to = calc_vs(cl_max_takeoff)  # Takeoff config stall speed
        vs0 = calc_vs(cl_max_landing)  # Landing config stall speed (VS0)

        # Convert to knots
        ms_to_kts = 1.943844
        vs1_kts = vs1 * ms_to_kts
        vs_to_kts = vs_to * ms_to_kts
        vs0_kts = vs0 * ms_to_kts

        # Calculate at sea level for reference
        rho_sl = RHO0
        vs1_sl = math.sqrt(
            (2 * weight_n * abs(load_factor)) / (rho_sl * wing_area_m2 * cl_max_clean)
        )
        vs1_sl_kts = vs1_sl * ms_to_kts

        # Stall warning speeds (typically 5-10% above stall)
        vs_warning_kts = vs0_kts * 1.05

        # Reference speeds per FAR 25 certification requirements:
        vref = vs0_kts * 1.3  # VREF: Approach reference speed = 1.3 * VS0
        v2_min = vs_to_kts * 1.13  # V2 min: Takeoff safety speed = 1.13 * VS(TO config)

        result = {
            "input": {
                "weight_kg": weight_kg,
                "wing_area_m2": wing_area_m2,
                "cl_max_clean": cl_max_clean,
                "cl_max_takeoff": cl_max_takeoff,
                "cl_max_landing": cl_max_landing,
                "altitude_ft": altitude_ft,
                "load_factor": load_factor,
            },
            "stall_speeds": {
                "VS1_clean_kts": round(vs1_kts, 1),
                "VS1_clean_ms": round(vs1, 2),
                "VS_takeoff_kts": round(vs_to_kts, 1),
                "VS_takeoff_ms": round(vs_to, 2),
                "VS0_landing_kts": round(vs0_kts, 1),
                "VS0_landing_ms": round(vs0, 2),
            },
            "reference_speeds": {
                "VREF_kts": round(vref, 1),
                "V2_min_kts": round(v2_min, 1),
                "stall_warning_kts": round(vs_warning_kts, 1),
            },
            "sea_level_reference": {
                "VS1_sl_kts": round(vs1_sl_kts, 1),
                "altitude_correction_pct": (
                    round((vs1_kts / vs1_sl_kts - 1) * 100, 1) if vs1_sl_kts > 0 else 0
                ),
            },
            "atmospheric": {
                "altitude_ft": altitude_ft,
                "density_kg_m3": round(rho, 5),
                "density_ratio": round(rho / RHO0, 4),
            },
        }

        output = f"""
STALL SPEED CALCULATION
=======================
Aircraft: {weight_kg:,.0f} kg, {wing_area_m2:.1f} m² wing area
Altitude: {altitude_ft:,.0f} ft (ρ = {rho:.5f} kg/m³)
Load Factor: {load_factor:.1f}g

Stall Speeds:
  VS1 (Clean):      {vs1_kts:>6.1f} kts ({vs1:.1f} m/s) - CL_max = {cl_max_clean:.2f}
  VS (Takeoff):     {vs_to_kts:>6.1f} kts ({vs_to:.1f} m/s) - CL_max = {cl_max_takeoff:.2f}
  VS0 (Landing):    {vs0_kts:>6.1f} kts ({vs0:.1f} m/s) - CL_max = {cl_max_landing:.2f}

Reference Speeds (based on stall):
  VREF (1.3 VS0):   {vref:>6.1f} kts
  V2 min (1.13 VS): {v2_min:>6.1f} kts
  Stall Warning:    {vs_warning_kts:>6.1f} kts

Sea Level Reference:
  VS1 at SL:        {vs1_sl_kts:>6.1f} kts
  Altitude Effect:  +{(vs1_kts / vs1_sl_kts - 1) * 100:.1f}% increase

{json.dumps(result, indent=2)}
"""
        return output.strip()

    except Exception as e:
        logger.error(f"Stall speed calculation error: {str(e)}", exc_info=True)
        return f"Error calculating stall speed: {str(e)}"


def weight_and_balance(
    basic_empty_weight_kg: float,
    basic_empty_arm_m: float,
    fuel_kg: float,
    fuel_arm_m: float,
    payload_items: list[dict],
    forward_cg_limit_m: float | None = None,
    aft_cg_limit_m: float | None = None,
    max_takeoff_weight_kg: float | None = None,
    mac_m: float | None = None,
    lemac_m: float | None = None,
) -> str:
    """Calculate aircraft center of gravity and verify within limits.

    Args:
        basic_empty_weight_kg: Basic empty weight in kg
        basic_empty_arm_m: Basic empty weight CG arm (from datum) in meters
        fuel_kg: Fuel load in kg
        fuel_arm_m: Fuel tank CG arm in meters
        payload_items: List of payload items, each with keys:
            - weight_kg: Weight in kg
            - arm_m: CG arm in meters
            - name: Item name (optional)
        forward_cg_limit_m: Forward CG limit (optional)
        aft_cg_limit_m: Aft CG limit (optional)
        max_takeoff_weight_kg: Maximum takeoff weight (optional)
        mac_m: Mean aerodynamic chord length (optional, for %MAC calculation)
        lemac_m: Leading edge of MAC position (optional, for %MAC calculation)

    Returns:
        Formatted string with weight and balance calculation results.

    Raises:
        No exceptions are raised directly; errors are returned as formatted strings.
    """
    try:
        # Calculate moments
        items = []

        # Basic empty weight
        bew_moment = basic_empty_weight_kg * basic_empty_arm_m
        items.append(
            {
                "name": "Basic Empty Weight",
                "weight_kg": basic_empty_weight_kg,
                "arm_m": basic_empty_arm_m,
                "moment_kg_m": bew_moment,
            }
        )

        # Fuel
        fuel_moment = fuel_kg * fuel_arm_m
        items.append(
            {
                "name": "Fuel",
                "weight_kg": fuel_kg,
                "arm_m": fuel_arm_m,
                "moment_kg_m": fuel_moment,
            }
        )

        # Payload items
        for i, item in enumerate(payload_items):
            weight = item.get("weight_kg", 0)
            arm = item.get("arm_m", 0)
            name = item.get("name", f"Payload {i + 1}")
            moment = weight * arm
            items.append(
                {
                    "name": name,
                    "weight_kg": weight,
                    "arm_m": arm,
                    "moment_kg_m": moment,
                }
            )

        # Calculate totals
        total_weight = sum(item["weight_kg"] for item in items)
        total_moment = sum(item["moment_kg_m"] for item in items)

        # CG position: x_cg = total_moment / total_weight (moment arm from datum)
        cg_m = total_moment / total_weight if total_weight > 0 else 0

        # Compute CG as percentage of Mean Aerodynamic Chord (MAC) if provided.
        # %MAC = (CG_pos - LEMAC) / MAC * 100
        # Typical transport aircraft operate between 15-35% MAC.
        cg_pct_mac = None
        if mac_m is not None and lemac_m is not None and mac_m > 0:
            cg_pct_mac = ((cg_m - lemac_m) / mac_m) * 100

        # Check limits
        weight_ok = True
        cg_ok = True
        warnings = []

        if max_takeoff_weight_kg is not None:
            if total_weight > max_takeoff_weight_kg:
                weight_ok = False
                warnings.append(
                    f"OVERWEIGHT by {total_weight - max_takeoff_weight_kg:.1f} kg"
                )

        if forward_cg_limit_m is not None:
            if cg_m < forward_cg_limit_m:
                cg_ok = False
                warnings.append(
                    f"CG forward of limit by {forward_cg_limit_m - cg_m:.3f} m"
                )

        if aft_cg_limit_m is not None:
            if cg_m > aft_cg_limit_m:
                cg_ok = False
                warnings.append(f"CG aft of limit by {cg_m - aft_cg_limit_m:.3f} m")

        result = {
            "loading_items": items,
            "totals": {
                "total_weight_kg": round(total_weight, 2),
                "total_moment_kg_m": round(total_moment, 2),
                "cg_position_m": round(cg_m, 4),
                "cg_percent_mac": (
                    round(cg_pct_mac, 2) if cg_pct_mac is not None else None
                ),
            },
            "limits": {
                "max_takeoff_weight_kg": max_takeoff_weight_kg,
                "forward_cg_limit_m": forward_cg_limit_m,
                "aft_cg_limit_m": aft_cg_limit_m,
            },
            "status": {
                "weight_within_limits": weight_ok,
                "cg_within_limits": cg_ok,
                "all_ok": weight_ok and cg_ok,
                "warnings": warnings,
            },
        }

        # Format output
        items_str = "\n".join(
            f"  {item['name']:<25} {item['weight_kg']:>8.1f} kg   {item['arm_m']:>6.3f} m   {item['moment_kg_m']:>10.1f} kg·m"
            for item in items
        )

        status_str = "✓ WITHIN LIMITS" if (weight_ok and cg_ok) else "⚠️ LIMITS EXCEEDED"
        warnings_str = "\n  ".join(warnings) if warnings else "None"

        mac_str = (
            f"\n  CG as %MAC:    {cg_pct_mac:>8.2f}%" if cg_pct_mac is not None else ""
        )

        def limit_str(value: float | None) -> str:
            """Format an optional limit; a limit of 0 is still a limit."""
            return "Not specified" if value is None else f"{value}"

        output = f"""
WEIGHT AND BALANCE CALCULATION
==============================
Loading Items:
  {"Item":<25} {"Weight":>8}   {"Arm":>6}   {"Moment":>10}
  {"-" * 60}
{items_str}
  {"-" * 60}
  {"TOTALS":<25} {total_weight:>8.1f} kg   {cg_m:>6.3f} m   {total_moment:>10.1f} kg·m

Results:
  Total Weight:  {total_weight:>8.1f} kg
  CG Position:   {cg_m:>8.4f} m (from datum){mac_str}

Limits Check: {status_str}
  Max Weight:    {limit_str(max_takeoff_weight_kg):>8} kg
  Forward CG:    {limit_str(forward_cg_limit_m):>8} m
  Aft CG:        {limit_str(aft_cg_limit_m):>8} m

Warnings: {warnings_str}

{json.dumps(result, indent=2)}
"""
        return output.strip()

    except Exception as e:
        logger.error(f"Weight and balance calculation error: {str(e)}", exc_info=True)
        return f"Error calculating weight and balance: {str(e)}"


def _tas_to_cas_kts(
    tas_ms: float, speed_of_sound_ms: float, pressure_pa: float
) -> float:
    """Calibrated airspeed (kts) for a true airspeed at the given conditions."""
    return _mach_to_cas(tas_ms / speed_of_sound_ms, pressure_pa) * MS_TO_KTS


def _cas_kts_to_tas(
    cas_kts: float, speed_of_sound_ms: float, pressure_pa: float
) -> float:
    """True airspeed (m/s) for a calibrated airspeed at the given conditions."""
    return _cas_to_mach(cas_kts * KTS_TO_MS, pressure_pa) * speed_of_sound_ms


def _round_or_none(value: float | None, ndigits: int = 0) -> float | None:
    """Round a finite value; map None / inf / NaN to None (valid JSON)."""
    if value is None or not math.isfinite(value):
        return None
    return round(value, ndigits)


def _fmt_dist(value_m: float | None) -> str:
    """Format a distance in m and ft for the text report."""
    if value_m is None or not math.isfinite(value_m):
        return "   n/a"
    return f"{value_m:>6,.0f} m ({value_m * M_TO_FT:,.0f} ft)"


def takeoff_performance(
    weight_kg: float,
    pressure_altitude_ft: float,
    temperature_c: float,
    wind_kts: float = 0,
    runway_slope_pct: float = 0,
    runway_condition: Literal["dry", "wet", "contaminated"] = "dry",
    thrust_to_weight: float = 0.3,
    cl_max_takeoff: float = 1.8,
    wing_area_m2: float = 100,
    cd0: float = 0.025,
    oswald_e: float = 0.8,
    aspect_ratio: float = 9,
) -> str:
    """Calculate takeoff field length and V-speeds.

    Uses simplified performance equations for educational purposes:

    - V-speeds are reported as calibrated airspeed (KCAS, equal to KIAS for
      zero instrument/position error) -- the numbers a pilot would see --
      and, separately, as true airspeed (KTAS).  Stall speed follows from
      the lift equation at the local density; distances use TAS.
    - Ground roll: constant-average-force energy method to VLOF = 1.1 VS,
      with rolling friction set by the runway condition (dry 0.02, wet
      0.05, contaminated 0.10).
    - Air distance to the 35 ft screen: Raymer, "Aircraft Design: A
      Conceptual Approach", sec. 17.8 (takeoff transition): circular
      transition arc at V_TR = 1.15 VS with load factor n = 1.2, radius
      R = V_TR^2 / (0.2 g), climb angle sin(gamma) = (T - D) / W at V_TR.
      If the obstacle is cleared during the arc,
      S_air = sqrt(R^2 - (R - h_obs)^2); otherwise S_air = R sin(gamma) +
      (h_obs - h_TR) / tan(gamma) with h_TR = R (1 - cos(gamma)).
    - Factored distance: 14 CFR 25.113(a)(2), 115 % of the all-engines
      distance to 35 ft.  The runway condition enters only through the
      rolling-friction coefficient, not through an extra factor.

    Args:
        weight_kg: Takeoff weight in kg
        pressure_altitude_ft: Airport pressure altitude in feet
        temperature_c: Outside air temperature in Celsius
        wind_kts: Headwind (+) or tailwind (-) in knots
        runway_slope_pct: Runway slope in percent (+ uphill)
        runway_condition: "dry", "wet", or "contaminated"
        thrust_to_weight: Thrust-to-weight ratio
        cl_max_takeoff: Maximum lift coefficient in takeoff config
        wing_area_m2: Wing reference area in m²
        cd0: Zero-lift drag coefficient
        oswald_e: Oswald efficiency factor
        aspect_ratio: Wing aspect ratio

    Returns:
        Formatted string with takeoff performance calculations including V-speeds,
        ground roll, air distance to 35 ft, and factored distances.

    Raises:
        No exceptions are raised directly; errors are returned as formatted strings.
    """
    try:
        for name, value in (
            ("weight_kg", weight_kg),
            ("wing_area_m2", wing_area_m2),
            ("cl_max_takeoff", cl_max_takeoff),
            ("aspect_ratio", aspect_ratio),
            ("oswald_e", oswald_e),
        ):
            if not value > 0:
                return f"Error: {name} must be > 0 (got {value})"
        temp_k = temperature_c + 273.15
        if temp_k <= 0:
            return "Error: temperature_c must be above absolute zero"

        # Get atmospheric conditions
        altitude_m = pressure_altitude_ft * 0.3048

        # Calculate actual temperature effects
        isa = _get_isa_conditions(altitude_m)
        p = isa["pressure_pa"]
        rho = p / (R_AIR * temp_k)
        a = math.sqrt(GAMMA * R_AIR * temp_k)

        # Exact ISA density altitude for the actual density
        density_alt_ft = _density_altitude_m(rho) * M_TO_FT

        # Weight in Newtons
        W = weight_kg * G0

        # Thrust (simplified - assumes constant T/W)
        T = thrust_to_weight * W

        # Calculate V-speeds
        ms_to_kts = 1.943844

        # VS (stall speed in takeoff config) as TAS at the local density
        vs = math.sqrt((2 * W) / (rho * wing_area_m2 * cl_max_takeoff))

        # V-speed ratios to VS (simplified FAR 25 schedule):
        # V1 1.05 (decision), VR 1.08 (rotation), VLOF 1.1 (liftoff),
        # V2 1.13 (takeoff safety speed, minimum 1.13 VSR per 25.107).
        ratios = {"VS": 1.0, "V1": 1.05, "VR": 1.08, "VLOF": 1.1, "V2": 1.13}
        v_tas_kts = {k: r * vs * ms_to_kts for k, r in ratios.items()}
        v_cas_kts = {k: _tas_to_cas_kts(r * vs, a, p) for k, r in ratios.items()}
        vlof_kts = v_tas_kts["VLOF"]

        # Ground roll calculation using energy-balance method (simplified).
        # Rolling friction coefficient depends on runway condition.
        mu = {"dry": 0.02, "wet": 0.05, "contaminated": 0.10}[runway_condition]

        # Average velocity during ground roll (approx 70% of VLOF for force averaging)
        v_avg = vlof_kts * 0.5144 * 0.7

        # Drag coefficient during ground roll using the drag polar:
        # CD = CD0 + CL^2 / (pi * AR * e)  (parabolic drag polar)
        cl_ground = 0.5 * cl_max_takeoff  # Partial lift during ground roll
        k_induced = 1.0 / (math.pi * aspect_ratio * oswald_e)
        cd = cd0 + cl_ground**2 * k_induced

        # Average forces
        L_avg = 0.5 * rho * v_avg**2 * wing_area_m2 * cl_ground
        D_avg = 0.5 * rho * v_avg**2 * wing_area_m2 * cd

        # Rolling resistance
        R = mu * (W - L_avg)

        # Slope force
        slope_force = W * math.sin(math.atan(runway_slope_pct / 100))

        # Net accelerating force
        F_net = T - D_avg - R - slope_force

        # Wind effect on ground speed
        wind_ms = wind_kts * 0.5144

        # Ground roll distance
        vlof_ms = vlof_kts * 0.5144
        v_ground = vlof_ms - wind_ms  # Ground speed at liftoff

        warnings: list[str] = []
        if F_net > 0:
            # Kinematic equation: s = v^2 / (2*a), where a = F_net / m
            # This assumes approximately constant net force (average conditions).
            a_avg = F_net / weight_kg
            ground_roll_m = v_ground**2 / (2 * a_avg)
        else:
            ground_roll_m = float("inf")
            warnings.append("Net accelerating force <= 0: aircraft cannot reach VLOF")

        # Air distance to 35 ft: Raymer transition arc + climb.
        h_obs_m = 35 * 0.3048
        v_tr = 1.15 * vs  # transition speed, average of VLOF and V2-ish
        radius_m = v_tr**2 / (0.2 * G0)  # n = 1.2 pull-up: R = V^2 / (g (n - 1))
        cl_tr = W / (0.5 * rho * v_tr**2 * wing_area_m2)  # = CLmax / 1.15^2
        cd_tr = cd0 + cl_tr**2 * k_induced
        drag_tr = 0.5 * rho * v_tr**2 * wing_area_m2 * cd_tr
        sin_gamma = (T - drag_tr) / W
        if sin_gamma > 0:
            gamma = math.asin(min(sin_gamma, 1.0))
            h_tr = radius_m * (1 - math.cos(gamma))
            if h_tr >= h_obs_m:
                # Screen height reached while still in the transition arc
                air_distance_m = math.sqrt(radius_m**2 - (radius_m - h_obs_m) ** 2)
            else:
                s_tr = radius_m * math.sin(gamma)
                air_distance_m = s_tr + (h_obs_m - h_tr) / math.tan(gamma)
            # Climb gradient: height gained per horizontal distance
            climb_gradient = math.tan(gamma)
        else:
            gamma = 0.0
            air_distance_m = float("inf")
            climb_gradient = sin_gamma
            warnings.append(
                "Thrust below drag at V_TR: no climb gradient after liftoff"
            )

        # Total takeoff distance
        total_distance_m = ground_roll_m + air_distance_m

        # 14 CFR 25.113(a)(2): takeoff distance = 115% of the all-engines
        # horizontal distance to 35 ft.  The runway condition is already in
        # the rolling-friction coefficient, so no second condition factor.
        condition_factor = 1.15
        factored_distance_m = total_distance_m * condition_factor

        # Climb gradient (%)
        climb_gradient_pct = climb_gradient * 100

        def to_ft(value_m: float) -> float | None:
            return _round_or_none(value_m * M_TO_FT)

        result = {
            "input": {
                "weight_kg": weight_kg,
                "pressure_altitude_ft": pressure_altitude_ft,
                "temperature_c": temperature_c,
                "wind_kts": wind_kts,
                "runway_slope_pct": runway_slope_pct,
                "runway_condition": runway_condition,
            },
            "v_speeds_kts": {k: round(v, 1) for k, v in v_cas_kts.items()},
            "v_speeds_ktas": {k: round(v, 1) for k, v in v_tas_kts.items()},
            "speed_reference": (
                "v_speeds_kts are KCAS (= KIAS with zero position error); "
                "v_speeds_ktas are true airspeeds used for the distances"
            ),
            "distances_m": {
                "ground_roll": _round_or_none(ground_roll_m),
                "air_distance_to_35ft": _round_or_none(air_distance_m),
                "total_distance": _round_or_none(total_distance_m),
                "factored_distance": _round_or_none(factored_distance_m),
            },
            "distances_ft": {
                "ground_roll": to_ft(ground_roll_m),
                "air_distance_to_35ft": to_ft(air_distance_m),
                "total_distance": to_ft(total_distance_m),
                "factored_distance": to_ft(factored_distance_m),
            },
            "performance": {
                "climb_gradient_pct": round(climb_gradient_pct, 2),
                "climb_angle_deg": round(math.degrees(gamma), 2),
                "transition_radius_m": round(radius_m, 0),
                "density_altitude_ft": round(density_alt_ft, 0),
                "rolling_friction": mu,
                "condition_factor": condition_factor,
            },
            "method": (
                "ground roll: energy method; air distance: Raymer transition "
                "(V_TR = 1.15 VS, n = 1.2); factor: 14 CFR 25.113(a)(2)"
            ),
            "warnings": warnings,
        }

        output = f"""
TAKEOFF PERFORMANCE
===================
Conditions:
  Weight: {weight_kg:,.0f} kg
  Pressure Altitude: {pressure_altitude_ft:,.0f} ft
  Temperature: {temperature_c:.1f}°C (DA: {density_alt_ft:,.0f} ft)
  Wind: {wind_kts:+.0f} kts {"(headwind)" if wind_kts > 0 else "(tailwind)" if wind_kts < 0 else ""}
  Runway: {runway_slope_pct:+.1f}% slope, {runway_condition}

V-Speeds (KCAS, = KIAS with zero position error; KTAS in brackets):
  VS:   {v_cas_kts["VS"]:>6.1f} kts [{v_tas_kts["VS"]:.1f}]  (Stall)
  V1:   {v_cas_kts["V1"]:>6.1f} kts [{v_tas_kts["V1"]:.1f}]  (Decision)
  VR:   {v_cas_kts["VR"]:>6.1f} kts [{v_tas_kts["VR"]:.1f}]  (Rotation)
  VLOF: {v_cas_kts["VLOF"]:>6.1f} kts [{v_tas_kts["VLOF"]:.1f}]  (Liftoff)
  V2:   {v_cas_kts["V2"]:>6.1f} kts [{v_tas_kts["V2"]:.1f}]  (Safety)

Distances:
  Ground Roll:       {_fmt_dist(ground_roll_m)}
  Air Distance:      {_fmt_dist(air_distance_m)}
  Total Distance:    {_fmt_dist(total_distance_m)}
  Factored (x1.15):  {_fmt_dist(factored_distance_m)}

Climb Gradient: {climb_gradient_pct:.2f}%
{("Warnings: " + "; ".join(warnings)) if warnings else ""}
{json.dumps(result, indent=2)}
"""
        return output.strip()

    except Exception as e:
        logger.error(f"Takeoff performance calculation error: {str(e)}", exc_info=True)
        return f"Error calculating takeoff performance: {str(e)}"


def landing_performance(
    weight_kg: float,
    pressure_altitude_ft: float,
    temperature_c: float,
    wind_kts: float = 0,
    runway_slope_pct: float = 0,
    runway_condition: Literal["dry", "wet", "contaminated"] = "dry",
    cl_max_landing: float = 2.2,
    wing_area_m2: float = 100,
    vref_factor: float = 1.3,
    approach_angle_deg: float = 3.0,
) -> str:
    """Calculate landing distance for given conditions.

    - Speeds: VS0 from the lift equation, reported as KCAS (= KIAS with
      zero position error) with KTAS alongside; VREF = ``vref_factor`` x
      VS0 (CAS); VAPP = VREF + wind additive, where the additive is half
      the steady headwind component, at least 5 kt and at most 20 kt (no
      additive is taken for a tailwind beyond the 5 kt minimum); VTD =
      VREF - 5 kt.  The ground roll uses the touchdown TAS.
    - Ground roll: v_ground^2 / (2 a) with a = mu g + g sin(atan(slope))
      (slope positive uphill, so an uphill runway shortens the roll) and
      braking friction mu set by the runway condition (dry 0.4, wet 0.2,
      contaminated 0.1).
    - Factored (dispatch) distance -- the runway condition is applied once:
      dry: 14 CFR 121.195(b), dry landing distance / 0.6 (x1.67);
      wet: 14 CFR 121.195(d), 115 % of the dry requirement (x1.92 on the
      *dry* distance, since the regulatory wet factor replaces, rather than
      adds to, a wet-runway friction model);
      contaminated: no Part 121 dispatch factor exists; the FAA's
      recommended 15 % margin on the actual landing distance (SAFO 06012)
      is applied to the contaminated (low-mu) distance.

    Args:
        weight_kg: Landing weight in kg
        pressure_altitude_ft: Airport pressure altitude in feet
        temperature_c: Outside air temperature in Celsius
        wind_kts: Headwind (+) or tailwind (-) in knots
        runway_slope_pct: Runway slope in percent (+ uphill)
        runway_condition: "dry", "wet", or "contaminated"
        cl_max_landing: Maximum lift coefficient in landing config
        wing_area_m2: Wing reference area in m²
        vref_factor: Approach speed factor (typically 1.3)
        approach_angle_deg: Approach angle in degrees

    Returns:
        Formatted string with landing performance calculations including V-speeds,
        air distance from 50 ft, ground roll, and factored distances.

    Raises:
        No exceptions are raised directly; errors are returned as formatted strings.
    """
    try:
        for name, value in (
            ("weight_kg", weight_kg),
            ("wing_area_m2", wing_area_m2),
            ("cl_max_landing", cl_max_landing),
            ("vref_factor", vref_factor),
            ("approach_angle_deg", approach_angle_deg),
        ):
            if not value > 0:
                return f"Error: {name} must be > 0 (got {value})"
        temp_k = temperature_c + 273.15
        if temp_k <= 0:
            return "Error: temperature_c must be above absolute zero"

        # Get atmospheric conditions
        altitude_m = pressure_altitude_ft * 0.3048
        isa = _get_isa_conditions(altitude_m)
        p = isa["pressure_pa"]
        rho = p / (R_AIR * temp_k)
        a = math.sqrt(GAMMA * R_AIR * temp_k)

        # Weight in Newtons
        W = weight_kg * G0

        # Calculate V-speeds
        ms_to_kts = 1.943844

        # VS0 (stall speed in landing config): TAS at the local density,
        # then the calibrated airspeed the pilot sees.
        vs0_tas = math.sqrt((2 * W) / (rho * wing_area_m2 * cl_max_landing))
        vs0_kts = _tas_to_cas_kts(vs0_tas, a, p)

        # VREF (approach reference speed), CAS
        vref_kts = vs0_kts * vref_factor

        # VAPP (approach speed with wind additive): half the steady headwind
        # component, minimum 5 kt, maximum 20 kt; nothing extra for tailwind.
        headwind_kts = max(0.0, wind_kts)
        wind_additive = min(20.0, max(5.0, 0.5 * headwind_kts))
        vapp_kts = vref_kts + wind_additive

        # VTD (touchdown speed) - typically VREF - 5-10 kts
        vtd_kts = vref_kts - 5
        vtd_ms = _cas_kts_to_tas(vtd_kts, a, p)  # TAS for the ground roll

        # Wind effect
        wind_ms = wind_kts * 0.5144

        # Air distance (from 50ft to touchdown)
        # Using approach angle
        approach_rad = math.radians(approach_angle_deg)
        air_distance_m = 50 * 0.3048 / math.tan(approach_rad)

        # Ground roll calculation after touchdown.
        # Braking friction coefficient (mu) varies significantly with runway condition.
        mu_by_condition = {"dry": 0.4, "wet": 0.2, "contaminated": 0.1}
        mu_braking = mu_by_condition[runway_condition]

        # Ground speed at touchdown
        v_ground = vtd_ms - wind_ms

        def ground_roll(mu: float) -> float:
            """Braking distance; a = mu*g + g*sin(slope), uphill (+) helps."""
            g = G0
            decel = mu * g + g * math.sin(math.atan(runway_slope_pct / 100))
            if decel <= 0:
                return float("inf")  # cannot stop (e.g. steep icy downhill)
            return v_ground**2 / (2 * decel)

        ground_roll_m = ground_roll(mu_braking)
        dry_ground_roll_m = ground_roll(mu_by_condition["dry"])

        # Total (actual, unfactored) landing distance for this runway state
        total_distance_m = air_distance_m + ground_roll_m
        dry_total_m = air_distance_m + dry_ground_roll_m

        # Regulatory factored distance; see docstring for the references.
        dry_factor = 1 / 0.6  # 14 CFR 121.195(b): land within 60% of runway
        if runway_condition == "dry":
            condition_factor = dry_factor
            factored_distance_m = dry_total_m * condition_factor
            factored_basis = "dry landing distance (14 CFR 121.195(b))"
        elif runway_condition == "wet":
            condition_factor = dry_factor * 1.15  # 121.195(d): 115% of dry
            factored_distance_m = dry_total_m * condition_factor
            factored_basis = "dry landing distance x 1.15 (14 CFR 121.195(d))"
        else:
            condition_factor = 1.15
            factored_distance_m = total_distance_m * condition_factor
            factored_basis = (
                "actual contaminated landing distance + 15% (FAA SAFO 06012)"
            )

        warnings: list[str] = []
        if not math.isfinite(ground_roll_m):
            warnings.append(
                "Braking deceleration <= 0 (downhill slope exceeds friction): "
                "aircraft cannot stop"
            )

        def to_ft(value_m: float) -> float | None:
            return _round_or_none(value_m * M_TO_FT)

        result = {
            "input": {
                "weight_kg": weight_kg,
                "pressure_altitude_ft": pressure_altitude_ft,
                "temperature_c": temperature_c,
                "wind_kts": wind_kts,
                "runway_slope_pct": runway_slope_pct,
                "runway_condition": runway_condition,
            },
            "v_speeds_kts": {
                "VS0": round(vs0_kts, 1),
                "VREF": round(vref_kts, 1),
                "VAPP": round(vapp_kts, 1),
                "VTD": round(vtd_kts, 1),
            },
            "v_speeds_ktas": {
                "VS0": round(vs0_tas * ms_to_kts, 1),
                "VREF": round(_cas_kts_to_tas(vref_kts, a, p) * ms_to_kts, 1),
                "VAPP": round(_cas_kts_to_tas(vapp_kts, a, p) * ms_to_kts, 1),
                "VTD": round(vtd_ms * ms_to_kts, 1),
            },
            "speed_reference": (
                "v_speeds_kts are KCAS (= KIAS with zero position error); "
                "v_speeds_ktas are true airspeeds used for the distances"
            ),
            "distances_m": {
                "air_distance": _round_or_none(air_distance_m),
                "ground_roll": _round_or_none(ground_roll_m),
                "total_distance": _round_or_none(total_distance_m),
                "dry_total_distance": _round_or_none(dry_total_m),
                "factored_distance": _round_or_none(factored_distance_m),
            },
            "distances_ft": {
                "air_distance": to_ft(air_distance_m),
                "ground_roll": to_ft(ground_roll_m),
                "total_distance": to_ft(total_distance_m),
                "dry_total_distance": to_ft(dry_total_m),
                "factored_distance": to_ft(factored_distance_m),
            },
            "performance": {
                "braking_coefficient": mu_braking,
                "condition_factor": round(condition_factor, 4),
                "factored_basis": factored_basis,
                "wind_additive_kts": round(wind_additive, 1),
            },
            "warnings": warnings,
        }

        output = f"""
LANDING PERFORMANCE
===================
Conditions:
  Weight: {weight_kg:,.0f} kg
  Pressure Altitude: {pressure_altitude_ft:,.0f} ft
  Temperature: {temperature_c:.1f}°C
  Wind: {wind_kts:+.0f} kts {"(headwind)" if wind_kts > 0 else "(tailwind)" if wind_kts < 0 else ""}
  Runway: {runway_slope_pct:+.1f}% slope, {runway_condition}

V-Speeds (KCAS, = KIAS with zero position error):
  VS0:  {vs0_kts:>6.1f} kts  (Stall, landing config)
  VREF: {vref_kts:>6.1f} kts  (Reference)
  VAPP: {vapp_kts:>6.1f} kts  (Approach, +{wind_additive:.0f} kt additive)
  VTD:  {vtd_kts:>6.1f} kts  (Touchdown)

Distances:
  Air Distance:      {_fmt_dist(air_distance_m)}
  Ground Roll:       {_fmt_dist(ground_roll_m)}
  Total Distance:    {_fmt_dist(total_distance_m)}
  Factored (x{condition_factor:.2f}): {_fmt_dist(factored_distance_m)}
    basis: {factored_basis}

Braking: μ = {mu_braking} ({runway_condition})
{("Warnings: " + "; ".join(warnings)) if warnings else ""}
{json.dumps(result, indent=2)}
"""
        return output.strip()

    except Exception as e:
        logger.error(f"Landing performance calculation error: {str(e)}", exc_info=True)
        return f"Error calculating landing performance: {str(e)}"


# Regulation names accepted for backward compatibility, and what they mean.
_FUEL_RULE_ALIASES = {
    "FAR_91": "FAR_91_IFR",
    "FAR_121": "FAR_121_DOMESTIC",
}


def fuel_reserve_calculator(
    regulation: Literal[
        "FAR_91",
        "FAR_91_IFR",
        "FAR_91_VFR_DAY",
        "FAR_91_VFR_NIGHT",
        "FAR_121",
        "FAR_121_DOMESTIC",
        "FAR_121_FLAG",
        "JAR_OPS",
        "ICAO",
    ],
    trip_fuel_kg: float,
    cruise_fuel_flow_kg_hr: float,
    flight_time_min: float,
    alternate_fuel_kg: float = 0,
    holding_altitude_ft: float = 1500,
) -> str:
    """Calculate required fuel reserves per aviation regulations.

    Rules implemented (fuel beyond the trip fuel to the destination):

    - ``FAR_91_IFR`` (14 CFR 91.167): alternate (if required) + 45 min at
      normal cruising speed.  ``FAR_91`` is an alias for this rule.
    - ``FAR_91_VFR_DAY`` / ``FAR_91_VFR_NIGHT`` (14 CFR 91.151(a)): 30 min
      (day) or 45 min (night) at normal cruising speed; VFR needs no
      alternate, so ``alternate_fuel_kg`` is only added if supplied.
    - ``FAR_121_DOMESTIC`` (14 CFR 121.639): alternate + 45 min at normal
      cruise fuel consumption.  ``FAR_121`` is an alias for this rule.
    - ``FAR_121_FLAG`` (14 CFR 121.645(b)): 10 % of the total flight time
      (``flight_time_min``) at normal cruise fuel consumption + alternate +
      30 min holding at 1,500 ft above the alternate.
    - ``JAR_OPS`` / ``ICAO``: 5 % trip contingency (min. 5 min holding) +
      alternate + 30 min final reserve.

    Holding fuel flow is approximated as 80 % of the cruise fuel flow;
    ``holding_altitude_ft`` is echoed in the result for reference but does
    not change the holding fuel flow.

    Args:
        regulation: Regulatory framework (see above)
        trip_fuel_kg: Planned trip fuel in kg
        cruise_fuel_flow_kg_hr: Cruise fuel flow rate in kg/hr
        flight_time_min: Planned flight time in minutes (used by FAR_121_FLAG)
        alternate_fuel_kg: Fuel to fly to alternate airport in kg
        holding_altitude_ft: Expected holding altitude (informational only)

    Returns:
        Formatted string with fuel reserve breakdown per the selected regulation,
        including contingency, alternate, and final reserve components.

    Raises:
        No exceptions are raised directly; errors are returned as formatted strings.
    """
    try:
        for name, value in (
            ("trip_fuel_kg", trip_fuel_kg),
            ("cruise_fuel_flow_kg_hr", cruise_fuel_flow_kg_hr),
            ("flight_time_min", flight_time_min),
            ("alternate_fuel_kg", alternate_fuel_kg),
        ):
            if value < 0:
                return f"Error: {name} must be >= 0 (got {value})"

        reserves = {}
        notes = []
        rule = _FUEL_RULE_ALIASES.get(regulation, regulation)
        if rule != regulation:
            notes.append(f"{regulation} is treated as {rule}")

        cruise_kg_per_min = cruise_fuel_flow_kg_hr / 60
        holding_fuel_flow = cruise_fuel_flow_kg_hr * 0.8  # Lower at holding

        if rule == "FAR_91_IFR":
            # 14 CFR 91.167(a): destination + alternate + 45 min normal cruise
            reserves["contingency_kg"] = 0  # Not required under Part 91
            reserves["alternate_kg"] = alternate_fuel_kg
            reserves["final_reserve_kg"] = cruise_kg_per_min * 45
            notes.append(
                "14 CFR 91.167 (IFR): alternate (if required) + 45 min at "
                "normal cruising speed"
            )

        elif rule in ("FAR_91_VFR_DAY", "FAR_91_VFR_NIGHT"):
            # 14 CFR 91.151(a)(1)/(2): 30 min day, 45 min night
            minutes = 30 if rule == "FAR_91_VFR_DAY" else 45
            reserves["contingency_kg"] = 0
            reserves["alternate_kg"] = alternate_fuel_kg
            reserves["final_reserve_kg"] = cruise_kg_per_min * minutes
            period = "day" if minutes == 30 else "night"
            notes.append(
                f"14 CFR 91.151(a) (VFR {period}): {minutes} min at normal "
                "cruising speed; no alternate required"
            )

        elif rule == "FAR_121_DOMESTIC":
            # 14 CFR 121.639: destination + most distant alternate + 45 min
            # at normal cruise fuel consumption.  No percentage contingency.
            reserves["contingency_kg"] = 0
            reserves["alternate_kg"] = alternate_fuel_kg
            reserves["final_reserve_kg"] = cruise_kg_per_min * 45
            notes.append(
                "14 CFR 121.639 (domestic): alternate + 45 min at normal "
                "cruise fuel consumption"
            )

        elif rule == "FAR_121_FLAG":
            # 14 CFR 121.645(b): destination + 10% of the total flight time
            # at normal cruise consumption + alternate + 30 min holding at
            # 1,500 ft above the alternate.
            reserves["contingency_kg"] = 0.10 * flight_time_min * cruise_kg_per_min
            reserves["alternate_kg"] = alternate_fuel_kg
            reserves["final_reserve_kg"] = holding_fuel_flow * (30 / 60)
            notes.append(
                "14 CFR 121.645(b) (flag): 10% of flight time at cruise fuel "
                "flow + alternate + 30 min holding at 1,500 ft"
            )

        elif rule == "JAR_OPS":
            # JAR-OPS 1.255
            # 5% trip fuel contingency (or 5 min @ holding)
            # Alternate fuel
            # 30 min final reserve at 1500 ft over alternate
            reserves["contingency_kg"] = max(
                trip_fuel_kg * 0.05, holding_fuel_flow * (5 / 60)
            )
            reserves["alternate_kg"] = alternate_fuel_kg
            reserves["final_reserve_kg"] = holding_fuel_flow * (30 / 60)
            notes.append("JAR-OPS: 5% contingency + 30 min final reserve")

        elif rule == "ICAO":
            # ICAO Annex 6
            # 5% trip fuel or 5 min at holding
            # Alternate fuel
            # 30 min final reserve at 1500 ft
            reserves["contingency_kg"] = max(
                trip_fuel_kg * 0.05, holding_fuel_flow * (5 / 60)
            )
            reserves["alternate_kg"] = alternate_fuel_kg
            reserves["final_reserve_kg"] = holding_fuel_flow * (30 / 60)
            notes.append("ICAO Annex 6: 5% contingency + 30 min final reserve")

        else:
            return (
                f"Error: Unknown regulation '{regulation}'. Use one of: FAR_91_IFR, "
                "FAR_91_VFR_DAY, FAR_91_VFR_NIGHT, FAR_121_DOMESTIC, FAR_121_FLAG, "
                "JAR_OPS, ICAO (FAR_91 and FAR_121 are aliases)"
            )

        if rule in ("FAR_121_FLAG", "JAR_OPS", "ICAO"):
            notes.append(
                "Holding fuel flow approximated as 80% of cruise fuel flow; "
                "holding_altitude_ft is informational"
            )

        # Calculate totals
        total_reserves = sum(reserves.values())
        total_fuel = trip_fuel_kg + total_reserves

        # Extra fuel recommendation
        extra_fuel_recommended = trip_fuel_kg * 0.05  # Additional 5% margin

        result = {
            "regulation": regulation,
            "rule_applied": rule,
            "trip_fuel_kg": round(trip_fuel_kg, 1),
            "reserves": {k: round(v, 1) for k, v in reserves.items()},
            "total_reserves_kg": round(total_reserves, 1),
            "total_required_fuel_kg": round(total_fuel, 1),
            "extra_fuel_recommended_kg": round(extra_fuel_recommended, 1),
            "total_with_extra_kg": round(total_fuel + extra_fuel_recommended, 1),
            "flight_time_min": flight_time_min,
            "cruise_fuel_flow_kg_hr": cruise_fuel_flow_kg_hr,
            "holding_altitude_ft": holding_altitude_ft,
            "notes": notes,
        }

        # Format reserve breakdown
        reserve_lines = [
            f"  {k.replace('_', ' ').title()}: {v:>8.1f} kg"
            for k, v in reserves.items()
        ]
        reserve_str = "\n".join(reserve_lines)

        output = f"""
FUEL RESERVE CALCULATION
========================
Regulation: {regulation}{f" (as {rule})" if rule != regulation else ""}
Flight Time: {flight_time_min:.0f} min
Cruise Fuel Flow: {cruise_fuel_flow_kg_hr:.1f} kg/hr

Fuel Breakdown:
  Trip Fuel:             {trip_fuel_kg:>8.1f} kg

Reserves:
{reserve_str}
  ─────────────────────────────────
  Total Reserves:        {total_reserves:>8.1f} kg

TOTAL REQUIRED:          {total_fuel:>8.1f} kg

Recommended Extra:       {extra_fuel_recommended:>8.1f} kg
Total with Extra:        {total_fuel + extra_fuel_recommended:>8.1f} kg

Notes:
  • {chr(10) + "  • ".join(notes)}

{json.dumps(result, indent=2)}
"""
        return output.strip()

    except Exception as e:
        logger.error(f"Fuel reserve calculation error: {str(e)}", exc_info=True)
        return f"Error calculating fuel reserves: {str(e)}"
