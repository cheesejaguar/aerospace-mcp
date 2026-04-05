"""Rocket trajectory analysis tools for aerospace MCP.

Provides 3-DOF (three degrees of freedom) rocket ascent modeling with:
    - Forward Euler numerical integration of the equations of motion
    - Atmospheric drag modeling via ISA density lookup table
    - Thrust curve interpolation with mass depletion
    - Performance metric extraction (apogee, max-Q, burnout, Isp)
    - Preliminary rocket sizing from target altitude / payload mass

The Tsiolkovsky rocket equation underpins the sizing calculations::

    delta_V = Isp * g0 * ln(m0 / mf)

where m0 is the initial mass, mf is the final (dry) mass, and Isp is the
specific impulse.

Uses NumPy for vectorized calculations with CuPy compatibility for GPU
acceleration via the ``_array_backend`` module.

References:
    - Sutton, G.P. & Biblarz, O., "Rocket Propulsion Elements" (9th ed., 2017)
    - Anderson, J.D., "Modern Compressible Flow" (3rd ed., 2003)

WARNING: This module is for educational and research purposes only.
Do NOT use for real flight planning, navigation, or aircraft operations.
"""

from dataclasses import dataclass
from typing import Any

from ._array_backend import np
from .atmosphere import get_atmosphere_profile

# ===========================================================================
# Physical Constants
# ===========================================================================

# Standard gravitational acceleration (m/s^2).  NIST reference value.
G0 = 9.80665

# Archimedes' constant.
PI = np.pi


# ===========================================================================
# Data Classes
# ===========================================================================


@dataclass
class RocketGeometry:
    """Rocket geometry and mass properties.

    Attributes:
        dry_mass_kg: Structural (empty) mass excluding propellant (kg).
        propellant_mass_kg: Initial propellant mass (kg).
        diameter_m: Maximum body diameter (m) -- used for drag area.
        length_m: Total rocket length (m).
        cd: Drag coefficient (dimensionless, default 0.3).
        thrust_curve: Time-thrust pairs ``[[t0, F0], [t1, F1], ...]``
            in seconds and Newtons.  ``None`` means zero thrust.
    """

    dry_mass_kg: float
    propellant_mass_kg: float
    diameter_m: float
    length_m: float
    cd: float = 0.3
    thrust_curve: list[list[float]] = None


@dataclass
class RocketTrajectoryPoint:
    """Single point along a rocket trajectory time history.

    Attributes:
        time_s: Time since launch (s).
        altitude_m: Altitude above sea level (m).
        velocity_ms: Total velocity magnitude (m/s).
        acceleration_ms2: Total acceleration magnitude (m/s^2).
        mass_kg: Current vehicle mass (kg).
        thrust_n: Current thrust (N).
        drag_n: Aerodynamic drag force (N).
        mach: Mach number (dimensionless).
        dynamic_pressure_pa: Dynamic pressure q = 0.5*rho*V^2 (Pa).
    """

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
    """Aggregate performance metrics extracted from a trajectory.

    Attributes:
        max_altitude_m: Maximum altitude (apogee) in meters.
        apogee_time_s: Time of apogee in seconds.
        max_velocity_ms: Maximum velocity in m/s.
        max_mach: Maximum Mach number.
        max_q_pa: Maximum dynamic pressure (max-Q) in Pascals.
        burnout_altitude_m: Altitude at motor burnout (m).
        burnout_velocity_ms: Velocity at burnout (m/s).
        burnout_time_s: Time of burnout (s).
        total_impulse_ns: Total impulse (N*s) via trapezoidal integration.
        specific_impulse_s: Effective specific impulse Isp = I_total / (m_prop * g0).
    """

    max_altitude_m: float
    apogee_time_s: float
    max_velocity_ms: float
    max_mach: float
    max_q_pa: float
    burnout_altitude_m: float
    burnout_velocity_ms: float
    burnout_time_s: float
    total_impulse_ns: float
    specific_impulse_s: float


# ===========================================================================
# Thrust and Atmosphere Utilities
# ===========================================================================


def get_thrust_at_time(thrust_curve: list[list[float]], time_s: float) -> float:
    """Get thrust at a specified time by linearly interpolating the thrust curve.

    Args:
        thrust_curve: List of ``[time_s, thrust_N]`` pairs.
        time_s: Query time in seconds.

    Returns:
        Thrust in Newtons (0.0 outside the thrust curve bounds).
    """
    if not thrust_curve:
        return 0.0

    # Convert to NumPy arrays for efficient interpolation
    thrust_array = np.array(thrust_curve)
    times = thrust_array[:, 0]
    thrusts = thrust_array[:, 1]

    # Handle constant thrust (single point)
    if len(thrust_curve) == 1:
        return float(thrusts[0]) if time_s <= times[0] else 0.0

    # Before first point or after last point
    if time_s < times[0] or time_s >= times[-1]:
        return 0.0

    # Use NumPy interpolation
    return float(np.interp(time_s, times, thrusts))


def _build_atmosphere_table(
    max_alt_m: int = 50000, step_m: int = 1000
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pre-compute an ISA atmosphere lookup table for fast interpolation.

    Builds arrays of density and speed-of-sound at regular altitude
    intervals, enabling ``np.interp`` during trajectory integration
    instead of per-step ISA calculations.

    Args:
        max_alt_m: Maximum altitude in meters.
        step_m: Altitude spacing in meters.

    Returns:
        Tuple of ``(altitudes, densities, speeds_of_sound)`` NumPy arrays.
    """
    alt_points = list(range(0, max_alt_m + step_m, step_m))
    atm_profile = get_atmosphere_profile(alt_points, "ISA")

    altitudes = np.array([p.altitude_m for p in atm_profile])
    densities = np.array([p.density_kg_m3 for p in atm_profile])
    speeds_of_sound = np.array([p.speed_of_sound_mps for p in atm_profile])

    return altitudes, densities, speeds_of_sound


# ===========================================================================
# 3-DOF Rocket Trajectory Simulation
# ===========================================================================


def rocket_3dof_trajectory(
    geometry: RocketGeometry,
    dt_s: float = 0.1,
    max_time_s: float = 300.0,
    launch_angle_deg: float = 90.0,
) -> list[RocketTrajectoryPoint]:
    """Simulate a 3-DOF rocket trajectory using Forward Euler integration.

    Equations of motion (vertical + horizontal, flat Earth)::

        a_vertical   = T_v/m - g + D_v/m
        a_horizontal = T_h/m     + D_h/m

    where T is thrust, g is gravity, D is aerodynamic drag (opposing
    velocity), and m is instantaneous mass.

    Drag force: D = Cd * A_ref * q, where q = 0.5 * rho * V^2 is the
    dynamic pressure and A_ref = pi * (d/2)^2 is the reference area.

    Mass depletion: dm/dt = -m_propellant / t_burn (constant flow rate).

    Uses a pre-computed ISA atmosphere lookup table for density and
    speed-of-sound interpolation at each time step.

    Args:
        geometry: Rocket geometry and mass properties.
        dt_s: Integration time step in seconds.
        max_time_s: Maximum simulation duration in seconds.
        launch_angle_deg: Launch elevation angle from horizontal (degrees).
            90 degrees = vertical launch.

    Returns:
        List of trajectory points from launch to apogee (or ground impact).
    """
    # Initial conditions
    trajectory = []
    time = 0.0
    altitude = 0.0  # m above sea level
    velocity_vertical = 0.0  # m/s
    velocity_horizontal = 0.0  # m/s
    mass = geometry.dry_mass_kg + geometry.propellant_mass_kg

    # Launch angle
    launch_angle_rad = np.radians(launch_angle_deg)
    sin_launch = np.sin(launch_angle_rad)
    cos_launch = np.cos(launch_angle_rad)

    # Reference (cross-sectional) area for drag: A = pi * (d/2)^2
    drag_area = PI * (geometry.diameter_m / 2) ** 2

    # Pre-compute mass flow rate if thrust curve exists
    mass_flow_rate = 0.0
    if geometry.thrust_curve:
        thrust_array = np.array(geometry.thrust_curve)
        burn_time_total = float(
            np.max(thrust_array[thrust_array[:, 1] > 0, 0], initial=1.0)
        )
        if burn_time_total > 0:
            mass_flow_rate = geometry.propellant_mass_kg / burn_time_total

    # Build atmosphere lookup table for fast interpolation
    max_alt_m = 50000
    atm_altitudes, atm_densities, atm_speeds_of_sound = _build_atmosphere_table(
        max_alt_m
    )

    while time <= max_time_s and altitude >= 0:
        # Get current thrust
        thrust = (
            get_thrust_at_time(geometry.thrust_curve, time)
            if geometry.thrust_curve
            else 0.0
        )

        # Get atmospheric properties via interpolation
        if 0 <= altitude <= max_alt_m:
            rho = float(np.interp(altitude, atm_altitudes, atm_densities))
            speed_of_sound = float(
                np.interp(altitude, atm_altitudes, atm_speeds_of_sound)
            )
        else:
            # Fallback for extreme altitudes
            rho = 1.225 * np.exp(-altitude / 8400)
            speed_of_sound = 343.0 * np.sqrt(
                max(0.1, (288.15 - 0.0065 * altitude) / 288.15)
            )

        # Total velocity (NumPy for efficiency)
        velocity_total = np.sqrt(velocity_vertical**2 + velocity_horizontal**2)

        # Mach number
        mach = velocity_total / speed_of_sound if speed_of_sound > 0 else 0.0

        # Dynamic pressure
        dynamic_pressure = 0.5 * rho * velocity_total**2

        # Drag force
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
            thrust_accel_vertical = thrust * sin_launch / mass
            thrust_accel_horizontal = thrust * cos_launch / mass
        else:  # After 5 seconds, thrust is vertical only
            thrust_accel_vertical = thrust / mass
            thrust_accel_horizontal = 0.0

        # Total acceleration
        accel_vertical = thrust_accel_vertical - G0 + drag_accel_vertical
        accel_horizontal = thrust_accel_horizontal + drag_accel_horizontal

        # Record current state
        trajectory.append(
            RocketTrajectoryPoint(
                time_s=time,
                altitude_m=altitude,
                velocity_ms=float(velocity_total),
                acceleration_ms2=float(
                    np.sqrt(accel_vertical**2 + accel_horizontal**2)
                ),
                mass_kg=mass,
                thrust_n=thrust,
                drag_n=drag_force,
                mach=float(mach),
                dynamic_pressure_pa=dynamic_pressure,
            )
        )

        # Forward Euler integration: v_{n+1} = v_n + a*dt,  h_{n+1} = h_n + v*dt
        velocity_vertical += accel_vertical * dt_s
        velocity_horizontal += accel_horizontal * dt_s
        altitude += velocity_vertical * dt_s

        # Update mass
        if thrust > 0 and mass_flow_rate > 0:
            mass = max(geometry.dry_mass_kg, mass - mass_flow_rate * dt_s)

        time += dt_s

        # Stop if apogee reached and descending
        if altitude > 0 and velocity_vertical < -1.0:
            break

    return trajectory


def analyze_rocket_performance(
    trajectory: list[RocketTrajectoryPoint],
) -> RocketPerformance:
    """
    Analyze rocket performance from trajectory data using NumPy.

    Args:
        trajectory: List of trajectory points

    Returns:
        RocketPerformance summary
    """
    if not trajectory:
        return RocketPerformance(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

    # Convert trajectory to NumPy arrays for vectorized analysis
    n = len(trajectory)
    times = np.array([p.time_s for p in trajectory])
    altitudes = np.array([p.altitude_m for p in trajectory])
    velocities = np.array([p.velocity_ms for p in trajectory])
    machs = np.array([p.mach for p in trajectory])
    dynamic_pressures = np.array([p.dynamic_pressure_pa for p in trajectory])
    thrusts = np.array([p.thrust_n for p in trajectory])

    # Find key performance metrics using NumPy
    max_altitude = float(np.max(altitudes))
    max_velocity = float(np.max(velocities))
    max_mach = float(np.max(machs))
    max_q = float(np.max(dynamic_pressures))

    # Find apogee time
    apogee_idx = int(np.argmax(altitudes))
    apogee_time = float(times[apogee_idx])

    # Find burnout point (when thrust drops to near zero)
    thrust_active = thrusts > 100
    if np.any(thrust_active):
        # Find last index where thrust is active, then next point is burnout
        last_thrust_idx = int(np.max(np.where(thrust_active)[0]))
        burnout_idx = min(last_thrust_idx + 1, n - 1)
    else:
        burnout_idx = n - 1

    burnout_point = trajectory[burnout_idx]

    # Calculate total impulse using trapezoidal integration
    if n > 1:
        dt = np.diff(times)
        avg_thrusts = (thrusts[:-1] + thrusts[1:]) / 2
        total_impulse = float(np.sum(avg_thrusts * dt))
    else:
        total_impulse = 0.0

    # Estimate specific impulse
    initial_prop_mass = trajectory[0].mass_kg - burnout_point.mass_kg
    specific_impulse = (
        total_impulse / (initial_prop_mass * G0) if initial_prop_mass > 0 else 0.0
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


# ===========================================================================
# Preliminary Rocket Sizing
# ===========================================================================


def estimate_rocket_sizing(
    target_altitude_m: float, payload_mass_kg: float, propellant_type: str = "solid"
) -> dict[str, Any]:
    """Estimate rocket sizing for a given target altitude and payload.

    Uses the Tsiolkovsky rocket equation for mass breakdown::

        delta_V = Isp * g0 * ln(m0 / mf)
        mass_ratio = exp(delta_V / (Isp * g0))

    Delta-V requirement is estimated from the energy needed to reach the
    target altitude, multiplied by a 1.8x factor to account for gravity
    losses, drag losses, and steering losses.

    Geometry is estimated from propellant volume assuming a length-to-
    diameter ratio of 10.

    Args:
        target_altitude_m: Desired apogee altitude above sea level (m).
        payload_mass_kg: Payload mass (kg).
        propellant_type: ``"solid"`` or ``"liquid"`` -- sets Isp and
            structural ratio.

    Returns:
        Dictionary with mass breakdown, thrust requirement, geometry
        estimates, and feasibility flag.
    """
    # Propellant-type-dependent performance parameters
    if propellant_type == "solid":
        isp_s = 250.0  # Specific impulse (s) -- typical APCP
        structural_ratio = 0.15  # epsilon = m_struct / m_prop
        thrust_to_weight = 5.0  # Initial thrust-to-weight ratio
    elif propellant_type == "liquid":
        isp_s = 300.0  # Specific impulse (s) -- LOX/RP-1 class
        structural_ratio = 0.12  # Liquid stages are more mass-efficient
        thrust_to_weight = 4.0
    else:
        isp_s = 250.0
        structural_ratio = 0.15
        thrust_to_weight = 5.0

    # Estimate required delta-V from energy balance:
    # V_ideal = sqrt(2 * g * h), then multiply by 1.8 for losses.
    potential_energy_per_kg = G0 * target_altitude_m
    delta_v_req = np.sqrt(2 * potential_energy_per_kg) * 1.8

    # Tsiolkovsky rocket equation: mass_ratio = exp(dV / (Isp * g0))
    mass_ratio = float(np.exp(delta_v_req / (isp_s * G0)))

    # Mass breakdown: solve for propellant mass from the structural ratio
    # and mass ratio.  If denominator <= 0, single-stage is infeasible.
    denominator = 1 + structural_ratio - mass_ratio * structural_ratio
    if denominator <= 0:
        # Single-stage solution impossible -- multi-staging required.
        propellant_mass = float("inf")
    else:
        propellant_mass = (mass_ratio - 1) * payload_mass_kg / denominator

    structure_mass = structural_ratio * propellant_mass
    total_mass = payload_mass_kg + structure_mass + propellant_mass

    # Thrust requirement
    thrust_n = thrust_to_weight * total_mass * G0

    # Rough geometry estimates
    density_propellant = 1600.0  # kg/m³ typical solid propellant
    propellant_volume = propellant_mass / density_propellant

    # Assume propellant takes 70% of rocket volume
    total_volume = propellant_volume / 0.7

    # L/D ratio of 10
    ld_ratio = 10.0
    diameter = float((4 * total_volume / (PI * ld_ratio)) ** (1 / 3))
    length = ld_ratio * diameter

    return {
        "total_mass_kg": total_mass,
        "propellant_mass_kg": propellant_mass,
        "structure_mass_kg": structure_mass,
        "payload_mass_kg": payload_mass_kg,
        "thrust_n": thrust_n,
        "specific_impulse_s": isp_s,
        "mass_ratio": mass_ratio,
        "delta_v_ms": float(delta_v_req),
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
