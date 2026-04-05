"""Propeller and UAV Energy Analysis Tools.

Provides propeller performance analysis using Blade Element Momentum Theory
(BEMT), UAV energy / endurance estimation, and motor-propeller matching.
Falls back to simplified momentum-theory methods when optional dependencies
(AeroSandbox, PyBEMT) are unavailable.

Key capabilities:
    - BEMT propeller analysis (thrust, torque, power, efficiency vs. RPM)
    - UAV energy estimation for multirotor and fixed-wing configurations
    - Motor-propeller matching for optimal operating-point selection
    - Propeller and battery reference databases

References:
    - Leishman, J.G., "Principles of Helicopter Aerodynamics" (2nd ed., 2006)
    - McCormick, B.W., "Aerodynamics, Aeronautics, and Flight Mechanics" (1995)
    - Drela, M., "Flight Vehicle Aerodynamics" (2014)

WARNING: This module is for educational and research purposes only.
Do NOT use for real flight planning, navigation, or aircraft operations.
"""

import math

import numpy as np
from pydantic import BaseModel, Field

from . import update_availability

# ===========================================================================
# Optional Library Imports
# ===========================================================================

AEROSANDBOX_AVAILABLE = False
PYBEMT_AVAILABLE = False

try:
    import aerosandbox as asb

    AEROSANDBOX_AVAILABLE = True
    update_availability("propellers", True, {"aerosandbox": asb.__version__})
except ImportError:
    try:
        # import pybemt  # Available if needed

        PYBEMT_AVAILABLE = True
        update_availability("propellers", True, {"pybemt": "unknown"})
    except ImportError:
        update_availability(
            "propellers", True, {}
        )  # Still available with manual methods

# ===========================================================================
# Physical Constants
# ===========================================================================

# ISA sea-level air density (kg/m^3).
RHO_SEA_LEVEL = 1.225

# Standard gravitational acceleration (m/s^2).  NIST value.
GRAVITY = 9.80665

# ===========================================================================
# Propeller and Battery Reference Databases
# ===========================================================================

# Common propellers with basic aerodynamic characteristics.
# activity_factor: integrated blade width parameter (dimensionless, ~50-200).
# cl_design / cd_design: blade section lift/drag coefficients at design point.
# efficiency_max: peak propulsive efficiency (eta = J * CT / CP).
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
# energy_density_wh_kg: gravimetric energy density at cell level.
# discharge_efficiency: Coulombic efficiency accounting for internal losses.
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

    This is the fallback method when AeroSandbox / PyBEMT are unavailable.
    It uses simplified BEMT with a single representative blade section at
    75% radius (the "75% rule").

    Non-dimensional coefficients (dimensional analysis):
        - Thrust coefficient:  CT = T / (rho * n^2 * D^4)
        - Power coefficient:   CP = P / (rho * n^3 * D^5)
        - Advance ratio:       J  = V / (n * D)
        - Propulsive efficiency: eta = J * CT / CP

    Args:
        geometry: Propeller geometry parameters.
        rpm_list: List of rotational speeds to evaluate.
        velocity_ms: Free-stream (forward flight) velocity in m/s.
        altitude_m: Geometric altitude for atmospheric model.

    Returns:
        Performance points at each RPM operating condition.
    """
    # --- Atmospheric conditions (simplified ISA) ---
    # Troposphere (h < 11 km): T = T0 + L*h, P = P0*(T/T0)^(g/(R*L))
    if altitude_m < 11000:
        temp = 288.15 - 0.0065 * altitude_m  # ISA lapse rate: -6.5 K/km
        pressure = 101325 * (temp / 288.15) ** 5.256  # Barometric formula exponent
    else:
        # Stratosphere (isothermal at 216.65 K):
        # P = P_tropopause * exp(-g*dh / (R*T))
        temp = 216.65
        pressure = 22632 * math.exp(-0.0001577 * (altitude_m - 11000))

    # Ideal gas law: rho = P / (R_specific * T)
    rho = pressure / (287.04 * temp)

    results = []

    for rpm in rpm_list:
        n = rpm / 60.0  # Convert RPM to revolutions per second (rps)
        D = geometry.diameter_m

        # Advance ratio: J = V_inf / (n * D)
        # J = 0 corresponds to static (hovering) conditions.
        J = velocity_ms / (n * D) if n > 0 else 0

        # --- Static / near-static regime (J < 0.1) ---
        if J < 0.1:
            # Momentum theory gives an approximate static thrust coefficient.
            # Scale linearly with blade count (normalized to 2-blade baseline).
            CT_static = 0.12 * geometry.num_blades / 2  # Empirical approximation

            # T = CT * rho * n^2 * D^4  (dimensional analysis)
            thrust_n = CT_static * rho * n**2 * D**4

            # Power coefficient from ideal momentum theory:
            # CP_ideal = CT^(3/2) / sqrt(2)
            # Multiply by 1.2 to include profile (viscous) drag power losses.
            CP_static = CT_static ** (3 / 2) / math.sqrt(2) * 1.2
            power_w = CP_static * rho * n**3 * D**5

            # Static propulsive efficiency is inherently low (ideal = 0).
            efficiency = 0.5 if power_w > 0 else 0

        else:
            # --- Forward flight regime (blade element theory at 75% radius) ---

            # Geometric pitch angle at the blade tip:
            # beta = arctan(pitch / (pi * D))
            beta = math.atan(geometry.pitch_m / (math.pi * D))

            # Effective angle of attack at the 75% radial station:
            # alpha_eff = beta - arctan(J / (pi * 0.75))
            # The helix angle at 75%R is arctan(V / (pi * n * D * 0.75))
            #   = arctan(J / (pi * 0.75)).
            alpha_eff = beta - math.atan(J / (math.pi * 0.75))

            # Blade element lift and drag at the representative section.
            # Use thin-airfoil approximation: cl = cl_design * sin(2*alpha)
            # valid for |alpha| < 45 deg (avoid post-stall region).
            cl_eff = (
                geometry.cl_design * math.sin(2 * alpha_eff)
                if abs(alpha_eff) < math.pi / 4
                else 0
            )
            # Drag polar: cd = cd0 + k * cl^2  (simplified quadratic polar)
            cd_eff = geometry.cd_design + 0.01 * cl_eff**2

            # Blade element force integration (simplified single-station).
            # Thrust coefficient: CT ~ 0.5 * B * cl * r^2 * dr
            # evaluated at r/R = 0.75, with annular width factor (1 - 0.25).
            CT = 0.5 * geometry.num_blades * cl_eff * (0.75**2) * (1 - 0.25)
            # Power coefficient: CP ~ induced + profile
            # Profile contribution: same integral but with cd.
            # Induced contribution: CT * J / (2*pi) (from actuator disk theory).
            CP = 0.5 * geometry.num_blades * cd_eff * (0.75**2) * (
                1 - 0.25
            ) + CT * J / (2 * math.pi)

            # Prandtl tip-loss correction (simplified): reduce CT and CP
            # for low blade counts.  A 2-blade prop gets factor 1.0.
            CT *= min(1.0, geometry.num_blades / 2)
            CP *= min(1.0, geometry.num_blades / 2)

            # Convert non-dimensional coefficients to dimensional values
            thrust_n = CT * rho * n**2 * D**4
            power_w = CP * rho * n**3 * D**5

            # Propulsive efficiency: eta = J * CT / CP
            efficiency = J * CT / CP if CP > 0 else 0

        # Torque from P = 2*pi*n*Q  =>  Q = P / (2*pi*n)
        torque_nm = power_w / (2 * math.pi * n) if n > 0 else 0

        # Clamp to physical bounds
        efficiency = max(0, min(0.9, efficiency))
        thrust_n = max(0, thrust_n)
        power_w = max(1, power_w)  # Minimum power accounts for bearing losses

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

    Constructs a detailed blade geometry with 5 radial sections and uses
    AeroSandbox's built-in propeller analysis (which includes proper
    induction factor iteration and airfoil polars).

    Args:
        geometry: Propeller geometry parameters.
        rpm_list: RPM values to analyze.
        velocity_ms: Free-stream velocity (m/s).
        altitude_m: Altitude for atmospheric model.

    Returns:
        Performance points at each RPM.

    Raises:
        Exception: Falls back to simple method on any AeroSandbox error.
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
        battery_energy_wh * battery_config.discharge_efficiency * 0.9
    )  # 90% depth of discharge for UAV applications (more aggressive than automotive)

    # Flight time estimation
    flight_time_hours = usable_energy_wh / power_electrical_w
    flight_time_min = flight_time_hours * 60

    # Range estimation (for fixed-wing)
    range_km = None
    hover_time_min = None

    if uav_config.wing_area_m2:
        range_km = velocity_ms * flight_time_hours * 3.6  # Convert m/s·h to km
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


# ===========================================================================
# UAV Power Estimation Models
# ===========================================================================


def _multirotor_power_analysis(
    config: UAVConfiguration, velocity_ms: float, rho: float
) -> float:
    """Calculate power required for a multirotor in forward flight.

    Uses momentum theory for hover and a simplified Glauert model for
    forward-flight induced power variation.

    Power components:
        - Induced power: P_i = T * v_i / FM, where v_i is the induced
          velocity and FM is the figure of merit.
        - Parasite power: P_p = D_parasitic * V_inf.
        - Systems power: fixed 50 W for avionics/electronics.

    Args:
        config: UAV configuration (mass, disk area, drag coefficient).
        velocity_ms: Forward flight speed (m/s).
        rho: Air density (kg/m^3).

    Returns:
        Total mechanical power required in Watts.
    """
    weight_n = config.mass_kg * GRAVITY

    # Ideal hover power from momentum theory:
    # P_hover = T * sqrt(T / (2 * rho * A_disk))
    disk_loading = weight_n / config.disk_area_m2  # N/m^2
    power_hover_ideal = weight_n * math.sqrt(weight_n / (2 * rho * config.disk_area_m2))

    # Figure of Merit: accounts for non-ideal induced losses, tip losses,
    # and profile drag.  Typical multirotor FM = 0.6 to 0.8.
    FM = 0.7

    # Parasitic drag power: P_p = 0.5 * rho * V^2 * Cd0 * S_frontal * V
    # Frontal area approximated as 0.1 * m^(2/3) (dimensional heuristic).
    drag_parasitic = (
        0.5 * rho * velocity_ms**2 * config.cd0 * config.mass_kg ** (2 / 3) * 0.1
    )
    power_parasitic = drag_parasitic * velocity_ms

    # Induced power in forward flight (Glauert approximation):
    # P_i_fwd = P_hover * sqrt(1 + mu^2), where mu = V / v_hover.
    mu = velocity_ms / math.sqrt(disk_loading / rho)  # Advance ratio
    power_induced_forward = power_hover_ideal * (1 + mu**2) ** 0.5

    # Total power = induced/FM + parasitic + avionics
    total_power = (
        (power_induced_forward / FM) + power_parasitic + 50  # 50 W electronics overhead
    )

    return total_power


def _fixed_wing_power_analysis(
    config: UAVConfiguration, velocity_ms: float, rho: float
) -> float:
    """Calculate power required for a fixed-wing aircraft in cruise.

    Uses a simple parabolic drag polar: CD = CD0 + k * CL^2, where
    k = 1 / (pi * AR * e) is the induced drag factor (simplified here
    to a constant 0.015 for efficient UAVs).

    Power required: P = D * V / eta_prop.

    Args:
        config: UAV configuration (mass, wing area, CD0, CL_cruise).
        velocity_ms: Cruise airspeed (m/s).
        rho: Air density (kg/m^3).

    Returns:
        Total electrical power required in Watts.
    """
    weight_n = config.mass_kg * GRAVITY

    # Lift coefficient from level-flight equilibrium: L = W
    # CL = 2*W / (rho * V^2 * S)
    cl = config.cl_cruise or (2 * weight_n) / (
        rho * velocity_ms**2 * config.wing_area_m2
    )

    # Parabolic drag polar: CD = CD0 + k * CL^2
    # k ~ 1/(pi*AR*e) simplified to 0.015 for efficient UAVs.
    k = 0.015  # Induced drag factor
    effective_cd0 = min(config.cd0, 0.02)  # Cap CD0 for realistic small UAVs
    cd = effective_cd0 + k * cl**2

    # Drag force: D = 0.5 * rho * V^2 * S * CD
    drag_n = 0.5 * rho * velocity_ms**2 * config.wing_area_m2 * cd

    # Thrust power: P_thrust = D * V
    power_thrust = drag_n * velocity_ms

    # Account for propulsive efficiency (~0.8 for a well-matched propeller)
    power_aero = power_thrust / 0.8

    # Fixed overhead for avionics, servos, and payload
    power_systems = 5  # Watts

    return power_aero + power_systems


def _generic_power_estimate(config: UAVConfiguration, velocity_ms: float) -> float:
    """Generic power estimate when detailed aerodynamic config is unknown.

    Uses an empirical specific-power scaling (W/kg) with a velocity
    correction factor.  Appropriate only for order-of-magnitude estimates.

    Args:
        config: UAV configuration (mass only is used).
        velocity_ms: Cruise speed (m/s).

    Returns:
        Estimated power in Watts.
    """
    # Empirical specific power for efficient small fixed-wing UAVs (W/kg)
    specific_power = 15
    base_power = config.mass_kg * specific_power

    # Power scales roughly as V^2 (drag ~ V^2, P = D*V ~ V^3,
    # but propulsive efficiency improves, net ~V^2).
    velocity_factor = (velocity_ms / 15.0) ** 2.0

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
        Dictionary mapping propeller names to their geometric and
        aerodynamic parameters.
    """
    return PROPELLER_DATABASE.copy()


def get_battery_database() -> dict[str, dict[str, any]]:
    """Return a copy of the built-in battery reference database.

    Returns:
        Dictionary mapping battery type names to their electrical
        characteristics.
    """
    return BATTERY_DATABASE.copy()
