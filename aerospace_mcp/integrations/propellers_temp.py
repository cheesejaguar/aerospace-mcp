"""Propeller and UAV Energy Analysis Tools (temporary / standalone variant).

Provides propeller performance analysis using Blade Element Momentum Theory
(BEMT), UAV energy / endurance estimation, and motor-propeller matching.
Falls back to simplified momentum-theory methods when optional dependencies
(AeroSandbox, PyBEMT) are unavailable.

This module is a standalone variant of ``propellers.py`` with the
``update_availability`` calls removed for use outside the full integrations
package.

References:
    - Leishman, J.G., "Principles of Helicopter Aerodynamics" (2nd ed., 2006)
    - McCormick, B.W., "Aerodynamics, Aeronautics, and Flight Mechanics" (1995)

WARNING: This module is for educational and research purposes only.
Do NOT use for real flight planning, navigation, or aircraft operations.
"""

import math

import numpy as np
from pydantic import BaseModel, Field

# from . import update_availability

# ===========================================================================
# Optional Library Imports
# ===========================================================================

AEROSANDBOX_AVAILABLE = False
PYBEMT_AVAILABLE = False

try:
    import aerosandbox as asb

    AEROSANDBOX_AVAILABLE = True
    # update_availability removed
except ImportError:
    try:
        # import pybemt  # Available if needed

        PYBEMT_AVAILABLE = True
        # update_availability removed
    except ImportError:
        pass  # update_availability removed

# ===========================================================================
# Physical Constants
# ===========================================================================

# ISA sea-level air density (kg/m^3).
RHO_SEA_LEVEL = 1.225

# Standard gravitational acceleration (m/s^2).
GRAVITY = 9.80665

# ===========================================================================
# Propeller and Battery Reference Databases
# ===========================================================================

# Common propellers with basic aerodynamic characteristics.
PROPELLER_DATABASE = {
    "APC_10x7": {
        "diameter_m": 0.254,  # 10 inches
        "pitch_m": 0.178,  # 7 inches
        "num_blades": 2,
        "activity_factor": 100,
        "cl_design": 0.5,
        "cd_design": 0.02,
        "efficiency_max": 0.82,
    },
    "APC_12x8": {
        "diameter_m": 0.305,  # 12 inches
        "pitch_m": 0.203,  # 8 inches
        "num_blades": 2,
        "activity_factor": 110,
        "cl_design": 0.5,
        "cd_design": 0.02,
        "efficiency_max": 0.84,
    },
    "APC_15x10": {
        "diameter_m": 0.381,  # 15 inches
        "pitch_m": 0.254,  # 10 inches
        "num_blades": 2,
        "activity_factor": 120,
        "cl_design": 0.5,
        "cd_design": 0.02,
        "efficiency_max": 0.85,
    },
    "MULTISTAR_8045": {
        "diameter_m": 0.203,  # 8 inches
        "pitch_m": 0.114,  # 4.5 inches
        "num_blades": 3,
        "activity_factor": 90,
        "cl_design": 0.6,
        "cd_design": 0.025,
        "efficiency_max": 0.75,
    },
}

# Battery characteristics database.
BATTERY_DATABASE = {
    "LiPo_3S": {
        "nominal_voltage_v": 11.1,
        "energy_density_wh_kg": 150,
        "discharge_efficiency": 0.95,
        "cells": 3,
    },
    "LiPo_4S": {
        "nominal_voltage_v": 14.8,
        "energy_density_wh_kg": 150,
        "discharge_efficiency": 0.95,
        "cells": 4,
    },
    "LiPo_6S": {
        "nominal_voltage_v": 22.2,
        "energy_density_wh_kg": 150,
        "discharge_efficiency": 0.95,
        "cells": 6,
    },
    "Li-Ion_18650": {
        "nominal_voltage_v": 3.7,
        "energy_density_wh_kg": 200,
        "discharge_efficiency": 0.98,
        "cells": 1,
    },
}


# ===========================================================================
# Data Models
# ===========================================================================


class PropellerGeometry(BaseModel):
    """Propeller geometric parameters for BEMT analysis."""

    diameter_m: float = Field(..., gt=0, description="Propeller diameter in meters")
    pitch_m: float = Field(..., gt=0, description="Propeller pitch in meters")
    num_blades: int = Field(..., ge=2, le=6, description="Number of blades")
    hub_radius_m: float = Field(0.02, ge=0, description="Hub radius in meters")
    activity_factor: float = Field(
        100, ge=50, le=200, description="Propeller activity factor"
    )
    cl_design: float = Field(0.5, gt=0, le=1.5, description="Design lift coefficient")
    cd_design: float = Field(0.02, gt=0, le=0.1, description="Design drag coefficient")


class PropellerPerformancePoint(BaseModel):
    """Propeller performance at a single operating condition."""

    rpm: float = Field(..., description="Rotational speed in RPM")
    thrust_n: float = Field(..., description="Thrust in Newtons")
    torque_nm: float = Field(..., description="Torque in Newton-meters")
    power_w: float = Field(..., description="Power in Watts")
    efficiency: float = Field(..., description="Propulsive efficiency")
    advance_ratio: float = Field(..., description="Advance ratio J=V/(nD)")
    thrust_coefficient: float = Field(..., description="Thrust coefficient CT")
    power_coefficient: float = Field(..., description="Power coefficient CP")


class UAVConfiguration(BaseModel):
    """UAV configuration parameters."""

    mass_kg: float = Field(..., gt=0, description="Total UAV mass in kg")
    wing_area_m2: float | None = Field(
        None, gt=0, description="Wing area for fixed-wing"
    )
    disk_area_m2: float | None = Field(
        None, gt=0, description="Total rotor disk area for multirotor"
    )
    cd0: float = Field(0.03, ge=0, description="Zero-lift drag coefficient")
    cl_cruise: float | None = Field(
        None, description="Cruise lift coefficient for fixed-wing"
    )
    num_motors: int = Field(1, ge=1, le=8, description="Number of motors/propellers")
    motor_efficiency: float = Field(0.85, gt=0, le=1, description="Motor efficiency")
    esc_efficiency: float = Field(0.95, gt=0, le=1, description="ESC efficiency")


class BatteryConfiguration(BaseModel):
    """Battery pack configuration."""

    capacity_ah: float = Field(..., gt=0, description="Battery capacity in Amp-hours")
    voltage_nominal_v: float = Field(..., gt=0, description="Nominal voltage")
    mass_kg: float = Field(..., gt=0, description="Battery mass in kg")
    energy_density_wh_kg: float = Field(
        150, gt=0, description="Energy density in Wh/kg"
    )
    discharge_efficiency: float = Field(
        0.95, gt=0, le=1, description="Discharge efficiency"
    )


class EnergyAnalysisResult(BaseModel):
    """UAV energy analysis results."""

    flight_time_min: float = Field(..., description="Estimated flight time in minutes")
    range_km: float | None = Field(None, description="Range in km (for fixed-wing)")
    hover_time_min: float | None = Field(
        None, description="Hover time in minutes (for multirotor)"
    )
    power_required_w: float = Field(..., description="Average power required")
    energy_consumed_wh: float = Field(..., description="Energy consumed")
    battery_energy_wh: float = Field(..., description="Available battery energy")
    efficiency_overall: float = Field(..., description="Overall propulsive efficiency")


def _simple_propeller_analysis(
    geometry: PropellerGeometry,
    rpm_list: list[float],
    velocity_ms: float,
    altitude_m: float = 0,
) -> list[PropellerPerformancePoint]:
    """Simple propeller analysis using momentum theory and blade element methods.

    Fallback method when AeroSandbox / PyBEMT are unavailable.  Uses a
    single representative blade section at 75% radius.

    Args:
        geometry: Propeller geometry parameters.
        rpm_list: RPM values to evaluate.
        velocity_ms: Free-stream velocity (m/s).
        altitude_m: Geometric altitude for ISA model.

    Returns:
        Performance points at each RPM.
    """
    # --- ISA atmosphere model ---
    # Troposphere: T = 288.15 - 0.0065*h,  P = 101325*(T/288.15)^5.256
    if altitude_m < 11000:
        temp = 288.15 - 0.0065 * altitude_m
        pressure = 101325 * (temp / 288.15) ** 5.256
    else:
        # Stratosphere (isothermal): P = P_trop * exp(-g*dh/(R*T))
        temp = 216.65
        pressure = 22632 * math.exp(-0.0001577 * (altitude_m - 11000))

    # Ideal gas: rho = P / (R * T)
    rho = pressure / (287.04 * temp)

    results = []

    for rpm in rpm_list:
        n = rpm / 60.0  # Revolutions per second
        D = geometry.diameter_m

        # Advance ratio: J = V_inf / (n * D)
        J = velocity_ms / (n * D) if n > 0 else 0

        # --- Static / near-static regime (J < 0.1) ---
        if J < 0.1:
            # Momentum-theory static thrust coefficient (empirical)
            CT_static = 0.12 * geometry.num_blades / 2
            thrust_n = CT_static * rho * n**2 * D**4  # T = CT * rho * n^2 * D^4

            # CP from ideal momentum theory with 1.2x profile drag factor:
            # CP_ideal = CT^(3/2) / sqrt(2)
            CP_static = CT_static ** (3 / 2) / math.sqrt(2) * 1.2
            power_w = CP_static * rho * n**3 * D**5

            efficiency = 0.5 if power_w > 0 else 0

        else:
            # --- Forward flight: blade element at 75% radius ---
            # Geometric pitch angle: beta = arctan(pitch / (pi*D))
            beta = math.atan(geometry.pitch_m / (math.pi * D))

            # Effective AoA at 75%R station:
            # alpha = beta - arctan(J / (pi * 0.75))
            alpha_eff = beta - math.atan(J / (math.pi * 0.75))

            # Thin-airfoil section forces
            cl_eff = (
                geometry.cl_design * math.sin(2 * alpha_eff)
                if abs(alpha_eff) < math.pi / 4
                else 0
            )
            cd_eff = geometry.cd_design + 0.01 * cl_eff**2  # Quadratic drag polar

            # Blade element integration at r/R = 0.75
            CT = 0.5 * geometry.num_blades * cl_eff * (0.75**2) * (1 - 0.25)
            CP = 0.5 * geometry.num_blades * cd_eff * (0.75**2) * (
                1 - 0.25
            ) + CT * J / (2 * math.pi)  # Add induced power contribution

            # Tip-loss correction (simplified)
            CT *= min(1.0, geometry.num_blades / 2)
            CP *= min(1.0, geometry.num_blades / 2)

            thrust_n = CT * rho * n**2 * D**4
            power_w = CP * rho * n**3 * D**5

            # Propulsive efficiency: eta = J * CT / CP
            efficiency = J * CT / CP if CP > 0 else 0

        # Torque: Q = P / (2*pi*n)
        torque_nm = power_w / (2 * math.pi * n) if n > 0 else 0

        # Clamp to physical bounds
        efficiency = max(0, min(0.9, efficiency))
        thrust_n = max(0, thrust_n)
        power_w = max(1, power_w)  # Minimum accounts for bearing friction

        results.append(
            PropellerPerformancePoint(
                rpm=rpm,
                thrust_n=thrust_n,
                torque_nm=torque_nm,
                power_w=power_w,
                efficiency=efficiency,
                advance_ratio=J,
                thrust_coefficient=thrust_n / (rho * n**2 * D**4) if n > 0 else 0,
                power_coefficient=power_w / (rho * n**3 * D**5) if n > 0 else 0,
            )
        )

    return results


def propeller_bemt_analysis(
    geometry: PropellerGeometry,
    rpm_list: list[float],
    velocity_ms: float,
    altitude_m: float = 0,
) -> list[PropellerPerformancePoint]:
    """
    Blade Element Momentum Theory propeller analysis.

    Args:
        geometry: Propeller geometry parameters
        rpm_list: List of RPM values to analyze
        velocity_ms: Forward velocity in m/s
        altitude_m: Altitude for atmospheric conditions

    Returns:
        List of PropellerPerformancePoint objects
    """
    if AEROSANDBOX_AVAILABLE:
        try:
            return _aerosandbox_propeller_analysis(
                geometry, rpm_list, velocity_ms, altitude_m
            )
        except Exception:
            # Fall back to simple method
            pass

    # Use simple momentum theory + basic blade element method
    return _simple_propeller_analysis(geometry, rpm_list, velocity_ms, altitude_m)


def _aerosandbox_propeller_analysis(
    geometry: PropellerGeometry,
    rpm_list: list[float],
    velocity_ms: float,
    altitude_m: float,
) -> list[PropellerPerformancePoint]:
    """Full BEMT analysis using the AeroSandbox library.

    Args:
        geometry: Propeller geometry parameters.
        rpm_list: RPM values to analyze.
        velocity_ms: Free-stream velocity (m/s).
        altitude_m: Altitude for atmospheric model.

    Returns:
        Performance points at each RPM.
    """
    # Create atmosphere
    atmosphere = asb.Atmosphere(altitude=altitude_m)

    # Create simplified propeller geometry
    # AeroSandbox requires detailed blade geometry, so we'll approximate
    prop = asb.Propeller(
        name="TestProp",
        radius=geometry.diameter_m / 2,
        hub_radius=geometry.hub_radius_m,
        num_blades=geometry.num_blades,
        # Simplified blade sections
        sections=[
            asb.PropellerSection(
                radius_nondim=r_nd,
                chord=0.1
                * geometry.diameter_m
                * (1 - r_nd),  # Simplified chord distribution
                twist=math.degrees(
                    math.atan(geometry.pitch_m / (math.pi * geometry.diameter_m * r_nd))
                ),
                airfoil=asb.Airfoil("naca2412"),
            )
            for r_nd in np.linspace(0.2, 1.0, 5)  # 5 blade sections
        ],
    )

    results = []

    for rpm in rpm_list:
        try:
            # Create operating point
            op_point = asb.OperatingPoint(
                atmosphere=atmosphere,
                velocity=velocity_ms,
            )

            # Run propeller analysis
            analysis = prop.analyze_performance(op_point=op_point, rpm=rpm)

            thrust_n = float(analysis["thrust"])
            power_w = float(analysis["power"])
            torque_nm = power_w / (2 * math.pi * rpm / 60) if rpm > 0 else 0
            efficiency = float(analysis.get("efficiency", 0.0))

            n = rpm / 60.0
            D = geometry.diameter_m
            J = velocity_ms / (n * D) if n > 0 else 0

            results.append(
                PropellerPerformancePoint(
                    rpm=rpm,
                    thrust_n=thrust_n,
                    torque_nm=torque_nm,
                    power_w=power_w,
                    efficiency=efficiency,
                    advance_ratio=J,
                    thrust_coefficient=(
                        thrust_n / (atmosphere.density * n**2 * D**4) if n > 0 else 0
                    ),
                    power_coefficient=(
                        power_w / (atmosphere.density * n**3 * D**5) if n > 0 else 0
                    ),
                )
            )

        except Exception:
            # Fall back to simple method for this point
            simple_result = _simple_propeller_analysis(
                geometry, [rpm], velocity_ms, altitude_m
            )
            results.extend(simple_result)

    return results


def uav_energy_estimate(
    uav_config: UAVConfiguration,
    battery_config: BatteryConfiguration,
    mission_profile: dict[str, float],
    propeller_geometry: PropellerGeometry | None = None,
) -> EnergyAnalysisResult:
    """
    Estimate UAV endurance and energy consumption.

    Args:
        uav_config: UAV configuration parameters
        battery_config: Battery configuration
        mission_profile: Mission parameters (velocity, altitude, etc.)
        propeller_geometry: Optional propeller geometry for detailed analysis

    Returns:
        EnergyAnalysisResult with flight time and energy analysis
    """
    velocity_ms = mission_profile.get("velocity_ms", 15.0)
    altitude_m = mission_profile.get("altitude_m", 100.0)

    # Calculate atmospheric conditions
    if altitude_m < 11000:
        temp = 288.15 - 0.0065 * altitude_m
        pressure = 101325 * (temp / 288.15) ** 5.256
    else:
        temp = 216.65
        pressure = 22632 * math.exp(-0.0001577 * (altitude_m - 11000))

    rho = pressure / (287.04 * temp)

    # Determine aircraft type and power requirements
    if uav_config.disk_area_m2:
        # Multirotor analysis
        power_required_w = _multirotor_power_analysis(uav_config, velocity_ms, rho)
    elif uav_config.wing_area_m2 and uav_config.cl_cruise:
        # Fixed-wing analysis
        power_required_w = _fixed_wing_power_analysis(uav_config, velocity_ms, rho)
    else:
        # Generic power estimation
        power_required_w = _generic_power_estimate(uav_config, velocity_ms)

    # Apply system efficiencies
    power_electrical_w = power_required_w / (
        uav_config.motor_efficiency * uav_config.esc_efficiency
    )

    # Battery analysis
    battery_energy_wh = battery_config.capacity_ah * battery_config.voltage_nominal_v
    usable_energy_wh = (
        battery_energy_wh * battery_config.discharge_efficiency * 0.8
    )  # 80% depth of discharge

    # Flight time estimation
    flight_time_hours = usable_energy_wh / power_electrical_w
    flight_time_min = flight_time_hours * 60

    # Range estimation (for fixed-wing)
    range_km = None
    hover_time_min = None

    if uav_config.wing_area_m2:
        range_km = velocity_ms * flight_time_hours / 1000  # Convert to km
    else:
        hover_time_min = flight_time_min  # For multirotor, flight time = hover time

    return EnergyAnalysisResult(
        flight_time_min=flight_time_min,
        range_km=range_km,
        hover_time_min=hover_time_min,
        power_required_w=power_required_w,
        energy_consumed_wh=usable_energy_wh,
        battery_energy_wh=battery_energy_wh,
        efficiency_overall=uav_config.motor_efficiency * uav_config.esc_efficiency,
    )


def _multirotor_power_analysis(
    config: UAVConfiguration, velocity_ms: float, rho: float
) -> float:
    """Calculate power required for a multirotor in forward flight.

    Uses momentum theory for hover and Glauert model for forward flight.

    Args:
        config: UAV configuration.
        velocity_ms: Forward flight speed (m/s).
        rho: Air density (kg/m^3).

    Returns:
        Total mechanical power in Watts.
    """
    weight_n = config.mass_kg * GRAVITY

    # Hover power (momentum theory)
    disk_loading = weight_n / config.disk_area_m2
    power_hover_ideal = weight_n * math.sqrt(weight_n / (2 * rho * config.disk_area_m2))

    # Figure of merit for multirotor (typically 0.6-0.8)
    FM = 0.7

    # Forward flight power
    # Simplified analysis combining hover power and parasitic drag
    drag_parasitic = (
        0.5 * rho * velocity_ms**2 * config.cd0 * config.mass_kg ** (2 / 3) * 0.1
    )  # Rough frontal area
    power_parasitic = drag_parasitic * velocity_ms

    # Induced power reduction in forward flight
    mu = velocity_ms / math.sqrt(disk_loading / rho)  # Advance ratio
    power_induced_forward = power_hover_ideal * (1 + mu**2) ** 0.5  # Simplified

    total_power = (
        (power_induced_forward / FM) + power_parasitic + 50
    )  # Add 50W for electronics

    return total_power


def _fixed_wing_power_analysis(
    config: UAVConfiguration, velocity_ms: float, rho: float
) -> float:
    """Calculate power required for a fixed-wing aircraft in cruise.

    Uses a parabolic drag polar: CD = CD0 + k * CL^2.

    Args:
        config: UAV configuration.
        velocity_ms: Cruise airspeed (m/s).
        rho: Air density (kg/m^3).

    Returns:
        Total power in Watts.
    """
    weight_n = config.mass_kg * GRAVITY

    # Lift coefficient
    cl = config.cl_cruise or (2 * weight_n) / (
        rho * velocity_ms**2 * config.wing_area_m2
    )

    # Drag analysis
    # Assume simple drag polar: CD = CD0 + k*CL^2
    k = 0.05  # Typical induced drag factor
    cd = config.cd0 + k * cl**2

    # Drag force
    drag_n = 0.5 * rho * velocity_ms**2 * config.wing_area_m2 * cd

    # Power required
    power_aero = drag_n * velocity_ms

    # Add system power (electronics, payload)
    power_systems = 20  # Watts

    return power_aero + power_systems


def _generic_power_estimate(config: UAVConfiguration, velocity_ms: float) -> float:
    """Generic power estimate when detailed aerodynamic config is unknown.

    Args:
        config: UAV configuration (mass only is used).
        velocity_ms: Cruise speed (m/s).

    Returns:
        Estimated power in Watts.
    """
    # Empirical specific power scaling (W/kg)
    specific_power = 150  # W/kg typical for small UAVs
    base_power = config.mass_kg * specific_power

    # Velocity scaling
    velocity_factor = (
        velocity_ms / 15.0
    ) ** 2.5  # Power increases with velocity^2.5 roughly

    return base_power * velocity_factor


def motor_propeller_matching(
    motor_kv: float,
    battery_voltage: float,
    propeller_options: list[str],
    thrust_required_n: float,
    altitude_m: float = 0,
) -> dict[str, any]:
    """
    Analyze motor/propeller combinations for optimal matching.

    Args:
        motor_kv: Motor KV rating (RPM per volt)
        battery_voltage: Battery voltage under load
        propeller_options: List of propeller names from database
        thrust_required_n: Required thrust in Newtons
        altitude_m: Operating altitude

    Returns:
        Dictionary with analysis results for each propeller option
    """
    max_rpm = motor_kv * battery_voltage * 0.85  # Allow for voltage drop under load

    results = {}

    for prop_name in propeller_options:
        if prop_name not in PROPELLER_DATABASE:
            continue

        prop_data = PROPELLER_DATABASE[prop_name]

        # Create geometry object
        geometry = PropellerGeometry(
            diameter_m=prop_data["diameter_m"],
            pitch_m=prop_data["pitch_m"],
            num_blades=prop_data["num_blades"],
            activity_factor=prop_data["activity_factor"],
            cl_design=prop_data["cl_design"],
            cd_design=prop_data["cd_design"],
        )

        # Analyze at different RPM points
        rpm_points = [max_rpm * factor for factor in [0.5, 0.7, 0.85, 1.0]]

        # Static thrust analysis
        performance = propeller_bemt_analysis(geometry, rpm_points, 0.0, altitude_m)

        # Find operating point closest to required thrust
        best_point = None
        min_error = float("inf")

        for point in performance:
            error = abs(point.thrust_n - thrust_required_n)
            if error < min_error:
                min_error = error
                best_point = point

        if best_point:
            results[prop_name] = {
                "geometry": geometry.model_dump(),
                "operating_point": best_point.model_dump(),
                "thrust_error_n": min_error,
                "efficiency": best_point.efficiency,
                "power_required_w": best_point.power_w,
                "rpm_recommended": best_point.rpm,
                "suitable": min_error < thrust_required_n * 0.2,  # Within 20%
            }

    return results


def get_propeller_database() -> dict[str, dict[str, any]]:
    """Return a copy of the built-in propeller reference database.

    Returns:
        Dictionary mapping propeller names to parameters.
    """
    return PROPELLER_DATABASE.copy()


def get_battery_database() -> dict[str, dict[str, any]]:
    """Return a copy of the built-in battery reference database.

    Returns:
        Dictionary mapping battery type names to parameters.
    """
    return BATTERY_DATABASE.copy()
