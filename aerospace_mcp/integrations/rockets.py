"""
Rocket trajectory analysis tools for aerospace MCP.

Provides 3DOF rocket ascent modeling with basic physics and atmosphere integration.
Uses lightweight manual calculations with optional RocketPy integration.
"""

import math
from dataclasses import dataclass
from typing import Any

from .atmosphere import get_atmosphere_profile


@dataclass
class RocketGeometry:
    """Rocket geometry parameters."""

    dry_mass_kg: float  # Rocket dry mass
    propellant_mass_kg: float  # Initial propellant mass
    diameter_m: float  # Rocket diameter
    length_m: float  # Total rocket length
    cd: float = 0.3  # Drag coefficient
    thrust_curve: list[list[float]] = (
        None  # [[time_s, thrust_N], ...] or constant thrust
    )


@dataclass
class RocketTrajectoryPoint:
    """Single point in rocket trajectory."""

    time_s: float
    altitude_m: float
    velocity_ms: float
    acceleration_ms2: float
    mass_kg: float
    thrust_n: float
    drag_n: float
    mach: float
    dynamic_pressure_pa: float


@dataclass
class RocketPerformance:
    """Rocket performance summary."""

    max_altitude_m: float
    apogee_time_s: float
    max_velocity_ms: float
    max_mach: float
    max_q_pa: float  # Max dynamic pressure
    burnout_altitude_m: float
    burnout_velocity_ms: float
    burnout_time_s: float
    total_impulse_ns: float
    specific_impulse_s: float


def get_thrust_at_time(thrust_curve: list[list[float]], time_s: float) -> float:
    """Get thrust at specified time from thrust curve."""
    if not thrust_curve:
        return 0.0

    # Handle constant thrust (single point)
    if len(thrust_curve) == 1:
        return thrust_curve[0][1] if time_s <= thrust_curve[0][0] else 0.0

    # Sort by time
    sorted_curve = sorted(thrust_curve, key=lambda x: x[0])

    # Before first point
    if time_s < sorted_curve[0][0]:
        return 0.0

    # After last point
    if time_s >= sorted_curve[-1][0]:
        return 0.0

    # Interpolate between points
    for i in range(len(sorted_curve) - 1):
        t1, thrust1 = sorted_curve[i]
        t2, thrust2 = sorted_curve[i + 1]

        if t1 <= time_s <= t2:
            if t2 == t1:
                return thrust1
            # Linear interpolation
            return thrust1 + (thrust2 - thrust1) * (time_s - t1) / (t2 - t1)

    return 0.0


def rocket_3dof_trajectory(
    geometry: RocketGeometry,
    dt_s: float = 0.1,
    max_time_s: float = 300.0,
    launch_angle_deg: float = 90.0,
) -> list[RocketTrajectoryPoint]:
    """
    Calculate 3DOF rocket trajectory using numerical integration.

    Args:
        geometry: Rocket geometry and mass properties
        dt_s: Time step for integration (seconds)
        max_time_s: Maximum simulation time (seconds)
        launch_angle_deg: Launch angle from horizontal (degrees)

    Returns:
        List of trajectory points
    """
    # Initial conditions
    trajectory = []
    time = 0.0
    altitude = 0.0  # m above sea level
    velocity_vertical = 0.0  # m/s
    velocity_horizontal = 0.0  # m/s
    mass = geometry.dry_mass_kg + geometry.propellant_mass_kg

    # Launch angle in radians
    launch_angle_rad = math.radians(launch_angle_deg)

    # Constants
    g0 = 9.80665  # m/s² standard gravity

    # Get atmosphere profile for drag calculations
    max_alt_m = 50000  # 50km max altitude for atmosphere
    alt_points = list(range(0, max_alt_m + 1000, 1000))
    atm_profile = get_atmosphere_profile(alt_points, "ISA")
    atm_dict = {point.altitude_m: point for point in atm_profile}

    while time <= max_time_s and altitude >= 0:
        # Get current thrust
        thrust = (
            get_thrust_at_time(geometry.thrust_curve, time)
            if geometry.thrust_curve
            else 0.0
        )

        # Get atmospheric properties at current altitude
        atm_alt = min(max_alt_m, max(0, int(altitude // 1000) * 1000))
        if atm_alt in atm_dict:
            atm = atm_dict[atm_alt]
            rho = atm.density_kg_m3
            speed_of_sound = atm.speed_of_sound_mps
        else:
            # Fallback for extreme altitudes
            rho = 1.225 * math.exp(-altitude / 8400)  # Simple exponential model
            speed_of_sound = 343.0 * math.sqrt(
                max(0.1, (288.15 - 0.0065 * altitude) / 288.15)
            )

        # Total velocity
        velocity_total = math.sqrt(velocity_vertical**2 + velocity_horizontal**2)

        # Mach number
        mach = velocity_total / speed_of_sound if speed_of_sound > 0 else 0.0

        # Dynamic pressure
        dynamic_pressure = 0.5 * rho * velocity_total**2

        # Drag force (opposing velocity direction)
        drag_area = math.pi * (geometry.diameter_m / 2) ** 2
        drag_force = geometry.cd * drag_area * dynamic_pressure

        # Drag acceleration components
        if velocity_total > 0:
            drag_accel_vertical = (
                -drag_force * (velocity_vertical / velocity_total) / mass
            )
            drag_accel_horizontal = (
                -drag_force * (velocity_horizontal / velocity_total) / mass
            )
        else:
            drag_accel_vertical = 0.0
            drag_accel_horizontal = 0.0

        # Thrust acceleration (in launch direction initially, then vertical)
        if time < 5.0:  # First 5 seconds follow launch angle
            thrust_accel_vertical = thrust * math.sin(launch_angle_rad) / mass
            thrust_accel_horizontal = thrust * math.cos(launch_angle_rad) / mass
        else:  # After 5 seconds, thrust is vertical only
            thrust_accel_vertical = thrust / mass
            thrust_accel_horizontal = 0.0

        # Total acceleration
        accel_vertical = thrust_accel_vertical - g0 + drag_accel_vertical
        accel_horizontal = thrust_accel_horizontal + drag_accel_horizontal

        # Record current state
        trajectory.append(
            RocketTrajectoryPoint(
                time_s=time,
                altitude_m=altitude,
                velocity_ms=velocity_total,
                acceleration_ms2=math.sqrt(accel_vertical**2 + accel_horizontal**2),
                mass_kg=mass,
                thrust_n=thrust,
                drag_n=drag_force,
                mach=mach,
                dynamic_pressure_pa=dynamic_pressure,
            )
        )

        # Integration (Euler method)
        velocity_vertical += accel_vertical * dt_s
        velocity_horizontal += accel_horizontal * dt_s
        altitude += velocity_vertical * dt_s

        # Update mass (simple propellant consumption)
        if thrust > 0 and geometry.thrust_curve:
            # Estimate mass flow rate from thrust curve
            # For a simplified model, assume constant mass flow rate during burn
            burn_time_total = max(
                (point[0] for point in geometry.thrust_curve if point[1] > 0),
                default=1.0,
            )
            if burn_time_total > 0:
                mass_flow_rate = geometry.propellant_mass_kg / burn_time_total
                mass = max(geometry.dry_mass_kg, mass - mass_flow_rate * dt_s)

        time += dt_s

        # Stop if apogee reached and descending
        if altitude > 0 and velocity_vertical < -1.0:
            break

    return trajectory


def analyze_rocket_performance(
    trajectory: list[RocketTrajectoryPoint],
) -> RocketPerformance:
    """Analyze rocket performance from trajectory data."""
    if not trajectory:
        return RocketPerformance(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

    # Find key performance metrics
    max_altitude = max(point.altitude_m for point in trajectory)
    max_velocity = max(point.velocity_ms for point in trajectory)
    max_mach = max(point.mach for point in trajectory)
    max_q = max(point.dynamic_pressure_pa for point in trajectory)

    # Find apogee time
    apogee_point = max(trajectory, key=lambda p: p.altitude_m)
    apogee_time = apogee_point.time_s

    # Find burnout point (when thrust drops to near zero)
    burnout_point = None
    for i, point in enumerate(trajectory):
        if i > 0 and trajectory[i - 1].thrust_n > 100 and point.thrust_n < 100:
            burnout_point = point
            break

    if burnout_point is None:
        burnout_point = trajectory[-1] if trajectory else trajectory[0]

    # Calculate total impulse
    total_impulse = 0.0
    for i in range(1, len(trajectory)):
        dt = trajectory[i].time_s - trajectory[i - 1].time_s
        avg_thrust = (trajectory[i].thrust_n + trajectory[i - 1].thrust_n) / 2
        total_impulse += avg_thrust * dt

    # Estimate specific impulse
    initial_prop_mass = trajectory[0].mass_kg - burnout_point.mass_kg
    specific_impulse = (
        total_impulse / (initial_prop_mass * 9.80665) if initial_prop_mass > 0 else 0
    )

    return RocketPerformance(
        max_altitude_m=max_altitude,
        apogee_time_s=apogee_time,
        max_velocity_ms=max_velocity,
        max_mach=max_mach,
        max_q_pa=max_q,
        burnout_altitude_m=burnout_point.altitude_m,
        burnout_velocity_ms=burnout_point.velocity_ms,
        burnout_time_s=burnout_point.time_s,
        total_impulse_ns=total_impulse,
        specific_impulse_s=specific_impulse,
    )


def estimate_rocket_sizing(
    target_altitude_m: float, payload_mass_kg: float, propellant_type: str = "solid"
) -> dict[str, Any]:
    """
    Estimate rocket sizing for target altitude and payload.

    Args:
        target_altitude_m: Target apogee altitude
        payload_mass_kg: Payload mass
        propellant_type: "solid" or "liquid"

    Returns:
        Dictionary with sizing estimates
    """
    # Rule-of-thumb ratios for different propellant types
    if propellant_type == "solid":
        isp_s = 250.0  # Specific impulse
        structural_ratio = 0.15  # Structure mass / propellant mass
        thrust_to_weight = 5.0  # Initial T/W ratio
    elif propellant_type == "liquid":
        isp_s = 300.0
        structural_ratio = 0.12
        thrust_to_weight = 4.0
    else:
        isp_s = 250.0
        structural_ratio = 0.15
        thrust_to_weight = 5.0

    # Estimate delta-V requirement (simplified)
    # For vertical flight with gravity and drag losses
    # Basic energy approach: need kinetic + potential energy
    potential_energy_per_kg = 9.80665 * target_altitude_m
    # Add gravity losses (roughly 1.5x theoretical for vertical flight)
    # Add drag losses (roughly 10-20% additional)
    delta_v_req = (
        math.sqrt(2 * potential_energy_per_kg) * 1.8
    )  # Factor accounts for losses

    # Rocket equation: delta_v = isp * g0 * ln(m_initial / m_final)
    g0 = 9.80665
    mass_ratio = math.exp(delta_v_req / (isp_s * g0))

    # Mass breakdown
    # m_initial = payload + structure + propellant
    # m_final = payload + structure
    # mass_ratio = m_initial / m_final = (payload + structure + propellant) / (payload + structure)

    # structure = structural_ratio * propellant
    # Let x = propellant mass
    # mass_ratio = (payload + structural_ratio * x + x) / (payload + structural_ratio * x)
    # mass_ratio * (payload + structural_ratio * x) = payload + structural_ratio * x + x
    # mass_ratio * payload + mass_ratio * structural_ratio * x = payload + structural_ratio * x + x
    # mass_ratio * payload - payload = x * (1 + structural_ratio - mass_ratio * structural_ratio)
    # x = (mass_ratio - 1) * payload / (1 + structural_ratio - mass_ratio * structural_ratio)

    denominator = 1 + structural_ratio - mass_ratio * structural_ratio
    if denominator <= 0:
        # Impossible mission - need staging
        propellant_mass = float("inf")
    else:
        propellant_mass = (mass_ratio - 1) * payload_mass_kg / denominator

    structure_mass = structural_ratio * propellant_mass
    total_mass = payload_mass_kg + structure_mass + propellant_mass

    # Thrust requirement
    thrust_n = thrust_to_weight * total_mass * g0

    # Rough geometry estimates
    # Assume cylindrical rocket with L/D = 8-12
    density_propellant = 1600.0  # kg/m³ typical solid propellant
    propellant_volume = propellant_mass / density_propellant

    # Assume propellant takes 70% of rocket volume
    total_volume = propellant_volume / 0.7

    # L/D ratio of 10
    ld_ratio = 10.0
    # V = π * r² * L = π * (D/2)² * L = π * D² * L / 4
    # L = ld_ratio * D
    # V = π * D² * (ld_ratio * D) / 4 = π * ld_ratio * D³ / 4
    # D³ = 4 * V / (π * ld_ratio)
    diameter = (4 * total_volume / (math.pi * ld_ratio)) ** (1 / 3)
    length = ld_ratio * diameter

    return {
        "total_mass_kg": total_mass,
        "propellant_mass_kg": propellant_mass,
        "structure_mass_kg": structure_mass,
        "payload_mass_kg": payload_mass_kg,
        "thrust_n": thrust_n,
        "specific_impulse_s": isp_s,
        "mass_ratio": mass_ratio,
        "delta_v_ms": delta_v_req,
        "diameter_m": diameter,
        "length_m": length,
        "thrust_to_weight": thrust_to_weight,
        "feasible": propellant_mass < float("inf"),
    }


# Update availability
try:
    from . import update_availability

    update_availability("rockets", True, {})
except ImportError:
    pass
