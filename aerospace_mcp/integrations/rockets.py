"""Rocket trajectory analysis tools for aerospace MCP.

Provides 3-DOF (three degrees of freedom) rocket ascent modeling with:
    - Forward Euler numerical integration of the equations of motion
    - Atmospheric drag modeling via ISA density lookup table
    - Thrust curve interpolation with mass depletion
    - Performance metric extraction (apogee, max-Q, burnout, Isp)
    - Preliminary rocket sizing from target altitude / payload mass

The trajectory integration, thrust interpolation and performance analysis
run in the native Rust core (``aerospace_mcp._native``, sources in
``rust/src/rocket.rs``); this module keeps the Python data model, builds the
atmosphere table (so the ISA / ambiance model stays in Python), and converts
results back into dataclasses.

The Tsiolkovsky rocket equation underpins the sizing calculations::

    delta_V = Isp * g0 * ln(m0 / mf)

where m0 is the initial mass, mf is the final (dry) mass, and Isp is the
specific impulse.

References:
    - Sutton, G.P. & Biblarz, O., "Rocket Propulsion Elements" (9th ed., 2017)
    - Anderson, J.D., "Modern Compressible Flow" (3rd ed., 2003)

WARNING: This module is for educational and research purposes only.
Do NOT use for real flight planning, navigation, or aircraft operations.
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from .. import _native
from ._array_backend import np, to_numpy
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
    return _native.thrust_at_time(thrust_curve, time_s)


@lru_cache(maxsize=4)
def _build_atmosphere_table(
    max_alt_m: int = 50000, step_m: int = 1000
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pre-compute an ISA atmosphere lookup table for fast interpolation.

    Builds arrays of density and speed-of-sound at regular altitude
    intervals, enabling ``np.interp`` during trajectory integration
    instead of per-step ISA calculations.

    The table is a pure function of its integer arguments, so results are
    cached (the ISA profile does not change).  This matters for callers
    like ``optimize_launch_angle`` that run many trajectories in a loop.
    The returned arrays are marked read-only to keep the cache safe.

    Args:
        max_alt_m: Maximum altitude in meters.
        step_m: Altitude spacing in meters.

    Returns:
        Tuple of ``(altitudes, densities, speeds_of_sound)`` NumPy arrays
        (read-only; copy before mutating).
    """
    alt_points = list(range(0, max_alt_m + step_m, step_m))
    atm_profile = get_atmosphere_profile(alt_points, "ISA")

    altitudes = np.array([p.altitude_m for p in atm_profile])
    densities = np.array([p.density_kg_m3 for p in atm_profile])
    speeds_of_sound = np.array([p.speed_of_sound_mps for p in atm_profile])

    for arr in (altitudes, densities, speeds_of_sound):
        arr.setflags(write=False)

    return altitudes, densities, speeds_of_sound


@lru_cache(maxsize=4)
def _native_atmosphere(max_alt_m: int = 50000) -> "_native.AtmosphereTable":
    """Atmosphere table in the form the native simulation consumes (cached)."""
    altitudes, densities, speeds_of_sound = _build_atmosphere_table(max_alt_m)
    return _native.AtmosphereTable(
        to_numpy(altitudes).tolist(),
        to_numpy(densities).tolist(),
        to_numpy(speeds_of_sound).tolist(),
    )


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
    speed-of-sound interpolation at each time step (exponential fallback
    above 50 km). Integration runs natively; see ``rust/src/rocket.rs``.

    Args:
        geometry: Rocket geometry and mass properties.
        dt_s: Integration time step in seconds.
        max_time_s: Maximum simulation duration in seconds.
        launch_angle_deg: Launch elevation angle from horizontal (degrees).
            90 degrees = vertical launch.

    Returns:
        List of trajectory points from launch to apogee (or ground impact).

    Raises:
        ValueError: If the time step or duration is out of bounds, or the
            resulting step count exceeds the safety cap.
    """
    points = _native.rocket_trajectory(
        geometry, _native_atmosphere(), dt_s, max_time_s, launch_angle_deg
    )
    return [RocketTrajectoryPoint(*p) for p in points]


def analyze_rocket_performance(
    trajectory: list[RocketTrajectoryPoint],
) -> RocketPerformance:
    """Extract apogee, max-Q, burnout and impulse metrics from a trajectory.

    Burnout is the sample right after the last one with thrust above 100 N;
    total impulse uses trapezoidal integration and Isp is
    ``I_total / (m_propellant_used * g0)``.

    Args:
        trajectory: List of trajectory points (any objects with the
            ``RocketTrajectoryPoint`` attributes).

    Returns:
        RocketPerformance summary (all zeros for an empty trajectory).
    """
    return RocketPerformance(*_native.analyze_performance(trajectory))


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
