"""Aircraft performance tools for the Aerospace MCP server.

Provides tools for aircraft performance calculations including:
- Weight and balance
- Takeoff and landing performance
- Stall speeds
- Fuel reserves
- Airspeed conversions
- Density altitude
"""

import json
import logging
import math
from typing import Literal

logger = logging.getLogger(__name__)

# Constants
G0 = 9.80665  # m/s² - standard gravity
R_AIR = 287.05  # J/(kg·K) - specific gas constant for dry air
GAMMA = 1.4  # ratio of specific heats for air
P0 = 101325.0  # Pa - sea level standard pressure
T0 = 288.15  # K - sea level standard temperature
RHO0 = 1.225  # kg/m³ - sea level standard density
A0 = 340.29  # m/s - sea level speed of sound
LAPSE_RATE = 0.0065  # K/m - temperature lapse rate in troposphere


def _get_isa_conditions(altitude_m: float) -> dict:
    """Get ISA atmospheric conditions at altitude."""
    if altitude_m < 0:
        altitude_m = 0

    if altitude_m <= 11000:
        # Troposphere
        T = T0 - LAPSE_RATE * altitude_m
        P = P0 * (T / T0) ** (G0 / (R_AIR * LAPSE_RATE))
    else:
        # Tropopause (simplified - isothermal at 216.65 K)
        T_trop = T0 - LAPSE_RATE * 11000
        P_trop = P0 * (T_trop / T0) ** (G0 / (R_AIR * LAPSE_RATE))
        T = T_trop
        P = P_trop * math.exp(-G0 * (altitude_m - 11000) / (R_AIR * T_trop))

    rho = P / (R_AIR * T)
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
        dewpoint_c: Optional dewpoint for humidity correction

    Returns:
        Formatted string with density altitude calculation results
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

        # Calculate actual pressure from pressure altitude
        if pressure_alt_m <= 11000:
            p = P0 * (1 - LAPSE_RATE * pressure_alt_m / T0) ** (
                G0 / (R_AIR * LAPSE_RATE)
            )
        else:
            p_trop = P0 * (1 - LAPSE_RATE * 11000 / T0) ** (G0 / (R_AIR * LAPSE_RATE))
            p = p_trop * math.exp(-G0 * (pressure_alt_m - 11000) / (R_AIR * 216.65))

        # Calculate actual density
        rho = p / (R_AIR * temp_k)

        # Humidity correction (optional)
        humidity_correction_ft = 0
        if dewpoint_c is not None:
            # Approximate humidity correction using dewpoint spread
            spread = temperature_c - dewpoint_c
            # Higher humidity (smaller spread) increases density altitude
            humidity_correction_ft = max(0, (30 - spread) * 3)  # Rough approximation

        # Find density altitude by solving for altitude where ISA density equals actual density
        # For troposphere: rho = rho0 * (T/T0)^(g0/(R*L) - 1)
        # Simplified approximation: each 1°C above ISA ≈ 120 ft increase
        density_alt_ft = (
            pressure_altitude_ft + (delta_isa * 118.8) + humidity_correction_ft
        )

        # Calculate density altitude in meters
        density_alt_m = density_alt_ft * 0.3048

        # Recalculate ISA conditions at density altitude for verification
        _isa_at_da = _get_isa_conditions(density_alt_m)

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
        }

        output = f"""
DENSITY ALTITUDE CALCULATION
============================
Input Conditions:
  Pressure Altitude: {pressure_altitude_ft:,.0f} ft ({pressure_alt_m:,.0f} m)
  Temperature: {temperature_c:.1f}°C
  ISA Temperature at PA: {isa_temp_k - 273.15:.1f}°C
  ISA Deviation: {delta_isa:+.1f}°C
  Dewpoint: {dewpoint_c if dewpoint_c else 'Not specified'}°C

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

    Returns:
        Formatted string with all airspeed conversions
    """
    try:
        # Convert altitude to meters
        altitude_m = altitude_ft * 0.3048

        # Get atmospheric conditions
        isa = _get_isa_conditions(altitude_m)
        p = isa["pressure_pa"]
        rho = isa["density_kg_m3"]
        a = isa["speed_of_sound_ms"]

        # Use actual temperature if provided
        if temperature_c is not None:
            temp_k = temperature_c + 273.15
            # Recalculate density with actual temperature
            rho = p / (R_AIR * temp_k)
            a = math.sqrt(GAMMA * R_AIR * temp_k)
        else:
            temperature_c = isa["temperature_c"]
            temp_k = isa["temperature_k"]

        # Conversion constants
        kts_to_ms = 0.5144444
        ms_to_kts = 1.943844

        # Start with converting to CAS (central reference)
        if speed_type == "IAS":
            # IAS + position error = CAS
            cas_kts = speed_value + position_error_kts
        elif speed_type == "CAS":
            cas_kts = speed_value
        elif speed_type == "EAS":
            # EAS to CAS (compressibility correction)
            eas_kts = speed_value
            eas_ms = eas_kts * kts_to_ms
            # At low speeds, CAS ≈ EAS * sqrt(p0/p) (simplified)
            _cas_ms = eas_ms / math.sqrt(P0 / p) if p > 0 else eas_ms  # noqa: F841
            # For subsonic flow, include compressibility
            mach_approx = eas_ms / (a * math.sqrt(RHO0 / rho)) if rho > 0 else 0
            if mach_approx < 0.3:
                # Low speed - minimal compressibility
                cas_kts = eas_kts / math.sqrt(rho / RHO0) if rho > 0 else eas_kts
            else:
                # Higher speed - include compressibility correction
                compressibility = 1 + mach_approx**2 / 4 + mach_approx**4 / 40
                cas_kts = eas_kts / math.sqrt(rho / RHO0) * compressibility
        elif speed_type == "TAS":
            # TAS to EAS to CAS
            tas_kts = speed_value
            tas_ms = tas_kts * kts_to_ms
            # EAS = TAS * sqrt(rho/rho0)
            eas_ms = tas_ms * math.sqrt(rho / RHO0)
            eas_kts = eas_ms * ms_to_kts
            # Then EAS to CAS (simplified)
            cas_kts = eas_kts  # Simplified - compressibility correction small at typical speeds
        elif speed_type == "MACH":
            # Mach to TAS to EAS to CAS
            mach = speed_value
            tas_ms = mach * a
            tas_kts = tas_ms * ms_to_kts
            eas_ms = tas_ms * math.sqrt(rho / RHO0)
            eas_kts = eas_ms * ms_to_kts
            cas_kts = eas_kts  # Simplified
        else:
            return f"Error: Unknown speed type '{speed_type}'"

        # Now convert CAS to all other speeds
        _cas_ms = cas_kts * kts_to_ms  # noqa: F841

        # CAS to IAS
        ias_kts = cas_kts - position_error_kts

        # CAS to EAS (simplified - exact requires iterative solution)
        eas_kts = cas_kts * math.sqrt(rho / RHO0) if rho > 0 else cas_kts
        eas_ms = eas_kts * kts_to_ms

        # EAS to TAS
        tas_ms = eas_ms / math.sqrt(rho / RHO0) if rho > 0 else eas_ms
        tas_kts = tas_ms * ms_to_kts

        # TAS to Mach
        mach = tas_ms / a if a > 0 else 0

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
        Formatted string with stall speed calculations
    """
    try:
        # Get atmospheric conditions
        altitude_m = altitude_ft * 0.3048
        isa = _get_isa_conditions(altitude_m)
        rho = isa["density_kg_m3"]

        # Weight in Newtons
        weight_n = weight_kg * G0

        # Calculate stall speed for a given CL_max
        def calc_vs(cl_max: float) -> float:
            """Calculate stall speed in m/s."""
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

        # Reference speeds
        vref = vs0_kts * 1.3  # Approach reference speed (1.3 VS0)
        v2_min = vs_to_kts * 1.13  # Minimum takeoff safety speed

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
        Formatted string with weight and balance calculation
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

        # Calculate CG
        cg_m = total_moment / total_weight if total_weight > 0 else 0

        # Calculate %MAC if parameters provided
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
  Max Weight:    {max_takeoff_weight_kg if max_takeoff_weight_kg else 'Not specified':>8} kg
  Forward CG:    {forward_cg_limit_m if forward_cg_limit_m else 'Not specified':>8} m
  Aft CG:        {aft_cg_limit_m if aft_cg_limit_m else 'Not specified':>8} m

Warnings: {warnings_str}

{json.dumps(result, indent=2)}
"""
        return output.strip()

    except Exception as e:
        logger.error(f"Weight and balance calculation error: {str(e)}", exc_info=True)
        return f"Error calculating weight and balance: {str(e)}"


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

    Uses simplified performance equations for educational purposes.

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
        Formatted string with takeoff performance calculations
    """
    try:
        # Get atmospheric conditions
        altitude_m = pressure_altitude_ft * 0.3048

        # Calculate actual temperature effects
        isa = _get_isa_conditions(altitude_m)
        temp_k = temperature_c + 273.15
        rho = isa["pressure_pa"] / (R_AIR * temp_k)

        # Calculate density altitude effect
        delta_isa = temp_k - isa["temperature_k"]
        density_alt_ft = pressure_altitude_ft + delta_isa * 118.8

        # Weight in Newtons
        W = weight_kg * G0

        # Thrust (simplified - assumes constant T/W)
        T = thrust_to_weight * W

        # Calculate V-speeds
        ms_to_kts = 1.943844

        # VS (stall speed in takeoff config)
        vs = math.sqrt((2 * W) / (rho * wing_area_m2 * cl_max_takeoff))
        vs_kts = vs * ms_to_kts

        # V1 (decision speed) - typically 1.05-1.1 VS
        v1_kts = vs_kts * 1.05

        # VR (rotation speed) - typically 1.05-1.1 VS
        vr_kts = vs_kts * 1.08

        # V2 (takeoff safety speed) - minimum 1.13 VS
        v2_kts = vs_kts * 1.13

        # VLOF (liftoff speed) - typically 1.1 VS
        vlof_kts = vs_kts * 1.1

        # Ground roll calculation (simplified)
        mu = {"dry": 0.02, "wet": 0.05, "contaminated": 0.10}[runway_condition]

        # Average velocity during ground roll
        v_avg = vlof_kts * 0.5144 * 0.7  # 70% of liftoff speed

        # Drag coefficient at liftoff
        cl_ground = 0.5 * cl_max_takeoff  # Partial lift during ground roll
        cd = cd0 + cl_ground**2 / (math.pi * aspect_ratio * oswald_e)

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

        if F_net > 0:
            # Using: s = v²/(2a) and a = F/m
            a_avg = F_net / weight_kg
            ground_roll_m = v_ground**2 / (2 * a_avg)
        else:
            ground_roll_m = float("inf")

        # Air distance to 35ft (simplified)
        # Assume constant climb angle
        _v2_ms = v2_kts * 0.5144  # noqa: F841
        excess_thrust = T - D_avg
        climb_gradient = excess_thrust / W if W > 0 else 0
        if climb_gradient > 0:
            air_distance_m = 35 * 0.3048 / climb_gradient
        else:
            air_distance_m = 1000  # Default if can't calculate

        # Total takeoff distance
        total_distance_m = ground_roll_m + air_distance_m

        # Apply runway condition factors
        condition_factors = {"dry": 1.0, "wet": 1.15, "contaminated": 1.25}
        factored_distance_m = total_distance_m * condition_factors[runway_condition]

        # Convert to feet
        ground_roll_ft = ground_roll_m * 3.281
        air_distance_ft = air_distance_m * 3.281
        total_distance_ft = total_distance_m * 3.281
        factored_distance_ft = factored_distance_m * 3.281

        # Climb gradient (%)
        climb_gradient_pct = climb_gradient * 100

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
                "VS": round(vs_kts, 1),
                "V1": round(v1_kts, 1),
                "VR": round(vr_kts, 1),
                "VLOF": round(vlof_kts, 1),
                "V2": round(v2_kts, 1),
            },
            "distances_m": {
                "ground_roll": round(ground_roll_m, 0),
                "air_distance_to_35ft": round(air_distance_m, 0),
                "total_distance": round(total_distance_m, 0),
                "factored_distance": round(factored_distance_m, 0),
            },
            "distances_ft": {
                "ground_roll": round(ground_roll_ft, 0),
                "air_distance_to_35ft": round(air_distance_ft, 0),
                "total_distance": round(total_distance_ft, 0),
                "factored_distance": round(factored_distance_ft, 0),
            },
            "performance": {
                "climb_gradient_pct": round(climb_gradient_pct, 2),
                "density_altitude_ft": round(density_alt_ft, 0),
                "condition_factor": condition_factors[runway_condition],
            },
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

V-Speeds (KIAS):
  VS:   {vs_kts:>6.1f} kts  (Stall)
  V1:   {v1_kts:>6.1f} kts  (Decision)
  VR:   {vr_kts:>6.1f} kts  (Rotation)
  VLOF: {vlof_kts:>6.1f} kts  (Liftoff)
  V2:   {v2_kts:>6.1f} kts  (Safety)

Distances:
  Ground Roll:       {ground_roll_m:>6,.0f} m ({ground_roll_ft:,.0f} ft)
  Air Distance:      {air_distance_m:>6,.0f} m ({air_distance_ft:,.0f} ft)
  Total Distance:    {total_distance_m:>6,.0f} m ({total_distance_ft:,.0f} ft)
  Factored ({runway_condition}):  {factored_distance_m:>6,.0f} m ({factored_distance_ft:,.0f} ft)

Climb Gradient: {climb_gradient_pct:.2f}%

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
        Formatted string with landing performance calculations
    """
    try:
        # Get atmospheric conditions
        altitude_m = pressure_altitude_ft * 0.3048
        isa = _get_isa_conditions(altitude_m)
        temp_k = temperature_c + 273.15
        rho = isa["pressure_pa"] / (R_AIR * temp_k)

        # Weight in Newtons
        W = weight_kg * G0

        # Calculate V-speeds
        ms_to_kts = 1.943844

        # VS0 (stall speed in landing config)
        vs0 = math.sqrt((2 * W) / (rho * wing_area_m2 * cl_max_landing))
        vs0_kts = vs0 * ms_to_kts

        # VREF (approach reference speed)
        vref_kts = vs0_kts * vref_factor
        _vref_ms = vref_kts / ms_to_kts  # noqa: F841

        # VAPP (approach speed with wind additive)
        wind_additive = max(0, -wind_kts * 0.5) + 5  # Add for gusts
        vapp_kts = vref_kts + wind_additive

        # VTD (touchdown speed) - typically VREF - 5-10 kts
        vtd_kts = vref_kts - 5
        vtd_ms = vtd_kts / ms_to_kts

        # Wind effect
        wind_ms = wind_kts * 0.5144

        # Air distance (from 50ft to touchdown)
        # Using approach angle
        approach_rad = math.radians(approach_angle_deg)
        air_distance_m = 50 * 0.3048 / math.tan(approach_rad)

        # Ground roll calculation
        # Deceleration from braking
        mu_braking = {"dry": 0.4, "wet": 0.2, "contaminated": 0.1}[runway_condition]

        # Average deceleration (simplified)
        g = G0
        decel = mu_braking * g - g * math.sin(math.atan(runway_slope_pct / 100))

        # Ground speed at touchdown
        v_ground = vtd_ms - wind_ms

        # Ground roll distance
        if decel > 0:
            ground_roll_m = v_ground**2 / (2 * decel)
        else:
            ground_roll_m = v_ground * 10  # Default if can't calculate

        # Total landing distance
        total_distance_m = air_distance_m + ground_roll_m

        # Apply safety factors
        condition_factors = {"dry": 1.0, "wet": 1.43, "contaminated": 1.92}
        factored_distance_m = total_distance_m * condition_factors[runway_condition]

        # Convert to feet
        air_distance_ft = air_distance_m * 3.281
        ground_roll_ft = ground_roll_m * 3.281
        total_distance_ft = total_distance_m * 3.281
        factored_distance_ft = factored_distance_m * 3.281

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
            "distances_m": {
                "air_distance": round(air_distance_m, 0),
                "ground_roll": round(ground_roll_m, 0),
                "total_distance": round(total_distance_m, 0),
                "factored_distance": round(factored_distance_m, 0),
            },
            "distances_ft": {
                "air_distance": round(air_distance_ft, 0),
                "ground_roll": round(ground_roll_ft, 0),
                "total_distance": round(total_distance_ft, 0),
                "factored_distance": round(factored_distance_ft, 0),
            },
            "performance": {
                "braking_coefficient": mu_braking,
                "condition_factor": condition_factors[runway_condition],
            },
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

V-Speeds (KIAS):
  VS0:  {vs0_kts:>6.1f} kts  (Stall, landing config)
  VREF: {vref_kts:>6.1f} kts  (Reference)
  VAPP: {vapp_kts:>6.1f} kts  (Approach)
  VTD:  {vtd_kts:>6.1f} kts  (Touchdown)

Distances:
  Air Distance:      {air_distance_m:>6,.0f} m ({air_distance_ft:,.0f} ft)
  Ground Roll:       {ground_roll_m:>6,.0f} m ({ground_roll_ft:,.0f} ft)
  Total Distance:    {total_distance_m:>6,.0f} m ({total_distance_ft:,.0f} ft)
  Factored ({runway_condition}):  {factored_distance_m:>6,.0f} m ({factored_distance_ft:,.0f} ft)

Braking: μ = {mu_braking} ({runway_condition})

{json.dumps(result, indent=2)}
"""
        return output.strip()

    except Exception as e:
        logger.error(f"Landing performance calculation error: {str(e)}", exc_info=True)
        return f"Error calculating landing performance: {str(e)}"


def fuel_reserve_calculator(
    regulation: Literal["FAR_91", "FAR_121", "JAR_OPS", "ICAO"],
    trip_fuel_kg: float,
    cruise_fuel_flow_kg_hr: float,
    flight_time_min: float,
    alternate_fuel_kg: float = 0,
    holding_altitude_ft: float = 1500,
) -> str:
    """Calculate required fuel reserves per aviation regulations.

    Args:
        regulation: Regulatory framework - "FAR_91", "FAR_121", "JAR_OPS", or "ICAO"
        trip_fuel_kg: Planned trip fuel in kg
        cruise_fuel_flow_kg_hr: Cruise fuel flow rate in kg/hr
        flight_time_min: Planned flight time in minutes
        alternate_fuel_kg: Fuel to fly to alternate airport in kg
        holding_altitude_ft: Expected holding altitude for reserve calculations

    Returns:
        Formatted string with fuel reserve breakdown
    """
    try:
        reserves = {}
        notes = []

        if regulation == "FAR_91":
            # FAR 91.151 - VFR: 30 min day, 45 min night
            # FAR 91.167 - IFR: 45 min at normal cruise
            reserves["final_reserve_kg"] = cruise_fuel_flow_kg_hr * (45 / 60)
            reserves["contingency_kg"] = 0  # Not required under Part 91
            reserves["alternate_kg"] = alternate_fuel_kg
            notes.append("FAR 91.167: 45 min fuel at normal cruise")

        elif regulation == "FAR_121":
            # FAR 121.639 - Domestic: 45 min at normal cruise
            # Plus alternate fuel if required
            # Plus 10% contingency
            reserves["contingency_kg"] = trip_fuel_kg * 0.10
            reserves["alternate_kg"] = alternate_fuel_kg
            reserves["final_reserve_kg"] = cruise_fuel_flow_kg_hr * (45 / 60)
            notes.append("FAR 121.639: 10% contingency + 45 min reserve")

        elif regulation == "JAR_OPS":
            # JAR-OPS 1.255
            # 5% trip fuel contingency (or 5 min @ holding)
            # Alternate fuel
            # 30 min final reserve at 1500 ft over alternate
            holding_fuel_flow = cruise_fuel_flow_kg_hr * 0.8  # Lower at holding
            reserves["contingency_kg"] = max(
                trip_fuel_kg * 0.05, holding_fuel_flow * (5 / 60)
            )
            reserves["alternate_kg"] = alternate_fuel_kg
            reserves["final_reserve_kg"] = holding_fuel_flow * (30 / 60)
            notes.append("JAR-OPS: 5% contingency + 30 min final reserve")

        elif regulation == "ICAO":
            # ICAO Annex 6
            # 5% trip fuel or 5 min at holding
            # Alternate fuel
            # 30 min final reserve at 1500 ft
            holding_fuel_flow = cruise_fuel_flow_kg_hr * 0.8
            reserves["contingency_kg"] = max(
                trip_fuel_kg * 0.05, holding_fuel_flow * (5 / 60)
            )
            reserves["alternate_kg"] = alternate_fuel_kg
            reserves["final_reserve_kg"] = holding_fuel_flow * (30 / 60)
            notes.append("ICAO Annex 6: 5% contingency + 30 min final reserve")

        else:
            return f"Error: Unknown regulation '{regulation}'"

        # Calculate totals
        total_reserves = sum(reserves.values())
        total_fuel = trip_fuel_kg + total_reserves

        # Extra fuel recommendation
        extra_fuel_recommended = trip_fuel_kg * 0.05  # Additional 5% margin

        result = {
            "regulation": regulation,
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
Regulation: {regulation}
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
  • {chr(10) + '  • '.join(notes)}

{json.dumps(result, indent=2)}
"""
        return output.strip()

    except Exception as e:
        logger.error(f"Fuel reserve calculation error: {str(e)}", exc_info=True)
        return f"Error calculating fuel reserves: {str(e)}"
