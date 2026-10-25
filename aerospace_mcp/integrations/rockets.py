"""Rocket trajectory analysis tools for aerospace MCP.

Provides 3-DOF (three degrees of freedom) rocket ascent modeling with:
    - Forward Euler numerical integration of the equations of motion
    - Atmospheric drag modeling via ISA density lookup table
    - Thrust curve interpolation with thrust-proportional mass depletion
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

import math
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
        thrust_n: Thrust applied over the step starting at ``time_s`` (N):
            the thrust curve's mean over that step.
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
        total_impulse_ns: Total impulse (N*s) applied by the integrator
            (each sample's thrust held until the next sample).
        specific_impulse_s: Effective specific impulse
            Isp = I_total / (m_propellant_consumed * g0).
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


# Top of the rocket atmosphere table (m). The ISA model is valid to 86 km
# geometric (81 020 m with ambiance), so 80 km works with either backend.
# Above the table the native simulation continues the top row: density
# decays with the top interval's scale height (~6 km) and the speed of
# sound is held (see ``AtmosphereTable`` in ``rust/src/rocket.rs``).
ATMOSPHERE_TABLE_TOP_M = 80000


@lru_cache(maxsize=4)
def _build_atmosphere_table(
    max_alt_m: int = ATMOSPHERE_TABLE_TOP_M, step_m: int = 1000
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
def _native_atmosphere(
    max_alt_m: int = ATMOSPHERE_TABLE_TOP_M,
) -> "_native.AtmosphereTable":
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

    Each step applies the thrust curve's mean over the step (the curve is
    integrated exactly), so the delivered impulse equals the curve's impulse
    regardless of the time step. Mass flow is proportional to thrust,
    dm/dt = -F * m_propellant / I_total (constant effective exhaust
    velocity), so the propellant runs out exactly when the curve ends.

    Uses a pre-computed ISA atmosphere lookup table (0-80 km) for density
    and speed-of-sound interpolation at each time step, continued smoothly
    above the table (see ``ATMOSPHERE_TABLE_TOP_M``). Integration runs
    natively; see ``rust/src/rocket.rs``.

    Args:
        geometry: Rocket geometry and mass properties.
        dt_s: Integration time step in seconds.
        max_time_s: Maximum simulation duration in seconds.
        launch_angle_deg: Launch elevation angle from horizontal (degrees).
            90 degrees = vertical launch.

    Returns:
        List of trajectory points from launch to apogee (or ground impact).

    Raises:
        ValueError: If the time step or duration is out of bounds, the
            resulting step count exceeds the safety cap, or the geometry is
            invalid (non-positive masses or size, negative thrust, thrust
            curve times out of order).
    """
    points = _native.rocket_trajectory(
        geometry, _native_atmosphere(), dt_s, max_time_s, launch_angle_deg
    )
    return [RocketTrajectoryPoint(*p) for p in points]


def simulate_rocket(
    geometry: RocketGeometry,
    dt_s: float = 0.1,
    max_time_s: float = 300.0,
    launch_angle_deg: float = 90.0,
) -> tuple[RocketPerformance, "_native.RocketTrajectory"]:
    """Simulate and analyze in one native call, without per-point objects.

    Same model and arguments as :func:`rocket_3dof_trajectory`. Use this
    when only the performance and a few samples are needed: building a
    ``RocketTrajectoryPoint`` per step dominates the cost of small time
    steps (22 000 points at ``dt_s=0.001``).

    Returns:
        ``(performance, trajectory)``; ``trajectory`` supports ``len()``,
        indexing (negative indices count from the end) yielding tuples in
        ``RocketTrajectoryPoint`` field order, and ``points()`` for all of
        them.

    Raises:
        ValueError: As :func:`rocket_3dof_trajectory`.
    """
    trajectory = _native.simulate_rocket(
        geometry, _native_atmosphere(), dt_s, max_time_s, launch_angle_deg
    )
    return RocketPerformance(*trajectory.performance()), trajectory


def analyze_rocket_performance(
    trajectory: list[RocketTrajectoryPoint],
) -> RocketPerformance:
    """Extract apogee, max-Q, burnout and impulse metrics from a trajectory.

    Burnout is the sample right after the last one with thrust above 100 N.
    Total impulse sums each sample's thrust over the interval to the next
    sample (the impulse the forward-Euler integrator applies), and Isp is
    ``I_total / (m_propellant_consumed * g0)`` over the same samples.

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
    target_altitude_m: float,
    payload_mass_kg: float,
    propellant_type: str = "solid",
    design_margin: float = 1.0,
) -> dict[str, Any]:
    """Estimate single-stage rocket sizing for a target altitude and payload.

    Uses the Tsiolkovsky rocket equation for mass breakdown::

        delta_V = Isp * g0 * ln(m0 / mf)
        mass_ratio = exp(delta_V / (Isp * g0))

    The delta-V requirement is estimated from the energy needed to reach the
    target altitude, ``sqrt(2 * g0 * h)``, multiplied by a 1.8x factor for
    gravity, drag and steering losses, and then by ``design_margin``.

    The margin is applied to the delta-V budget rather than to the resulting
    masses because delta-V is the uncertain quantity here (the loss factor
    is a crude estimate) and a delta-V reserve is the usual way to carry
    margin in preliminary sizing: it propagates through the rocket equation,
    so the extra propellant also brings the structure needed to carry it and
    the vehicle still closes. Scaling the propellant mass afterwards would
    not.

    With propellant mass ``m_p``, structure ``eps * m_p`` and payload
    ``m_pl``, the rocket equation gives
    ``m_p = (R - 1) * m_pl / (1 + eps - R * eps)`` for mass ratio ``R``. A
    single stage cannot reach ``R >= (1 + eps) / eps``; such missions are
    reported as infeasible (``feasible: False``, masses and geometry
    ``None``) together with the largest delta-V a single stage can deliver.

    Geometry is estimated from propellant volume (bulk density per
    propellant type, propellant filling 70% of the body) assuming a
    length-to-diameter ratio of 10.

    Args:
        target_altitude_m: Desired apogee altitude above sea level (m), > 0.
        payload_mass_kg: Payload mass (kg), > 0.
        propellant_type: ``"solid"`` or ``"liquid"`` -- sets Isp, structural
            ratio, thrust-to-weight and propellant density.
        design_margin: Multiplier on the delta-V requirement, >= 1
            (1.0 = no margin).

    Returns:
        Dictionary with mass breakdown, thrust requirement, geometry
        estimates, the delta-V budget and margin, and a feasibility flag.
        All numbers are finite (or ``None``), so the result is valid JSON.

    Raises:
        ValueError: If an input is out of range or the propellant type is
            unknown.
    """
    if not (math.isfinite(target_altitude_m) and target_altitude_m > 0):
        raise ValueError("target_altitude_m must be > 0")
    if not (math.isfinite(payload_mass_kg) and payload_mass_kg > 0):
        raise ValueError("payload_mass_kg must be > 0")
    if not (math.isfinite(design_margin) and design_margin >= 1.0):
        raise ValueError("design_margin must be >= 1 (1.0 = no margin)")

    # Propellant-type-dependent parameters: specific impulse (s), structural
    # ratio eps = m_struct / m_prop, initial thrust-to-weight, and bulk
    # propellant density (kg/m^3).
    if propellant_type == "solid":
        # APCP composite: Isp ~250 s, density ~1600-1800 kg/m^3.
        isp_s, structural_ratio, thrust_to_weight, density = 250.0, 0.15, 5.0, 1600.0
    elif propellant_type == "liquid":
        # LOX/RP-1 class: Isp ~300 s; mixed bulk density ~1030 kg/m^3 at
        # O/F 2.56 (LOX 1141, RP-1 810 kg/m^3).
        isp_s, structural_ratio, thrust_to_weight, density = 300.0, 0.12, 4.0, 1030.0
    else:
        raise ValueError(
            f"Unknown propellant_type {propellant_type!r}; use 'solid' or 'liquid'"
        )

    # Energy balance: V_ideal = sqrt(2 * g * h), times 1.8 for losses, times
    # the design margin.
    delta_v_without_margin = math.sqrt(2 * G0 * target_altitude_m) * 1.8
    delta_v_req = delta_v_without_margin * design_margin
    exhaust_velocity = isp_s * G0

    # The largest mass ratio a single stage with this structural ratio can
    # reach (zero payload) is (1 + eps) / eps.
    max_mass_ratio = (1 + structural_ratio) / structural_ratio
    max_delta_v = exhaust_velocity * math.log(max_mass_ratio)

    result: dict[str, Any] = {
        "payload_mass_kg": payload_mass_kg,
        "propellant_type": propellant_type,
        "specific_impulse_s": isp_s,
        "structural_ratio": structural_ratio,
        "thrust_to_weight": thrust_to_weight,
        "design_margin": design_margin,
        "delta_v_without_margin_ms": delta_v_without_margin,
        "delta_v_ms": delta_v_req,
        "max_single_stage_delta_v_ms": max_delta_v,
    }

    exponent = delta_v_req / exhaust_velocity
    mass_ratio = math.exp(exponent) if exponent < 700 else math.inf
    denominator = 1 + structural_ratio - mass_ratio * structural_ratio
    if denominator <= 0:
        # Single-stage solution impossible -- multi-staging required.
        result.update(
            {
                "feasible": False,
                "reason": (
                    f"Required delta-V {delta_v_req:.0f} m/s exceeds the "
                    f"{max_delta_v:.0f} m/s a single {propellant_type} stage "
                    "can deliver; multi-staging is required"
                ),
                "mass_ratio": mass_ratio if math.isfinite(mass_ratio) else None,
                "total_mass_kg": None,
                "propellant_mass_kg": None,
                "structure_mass_kg": None,
                "thrust_n": None,
                "diameter_m": None,
                "length_m": None,
            }
        )
        return result

    # Mass breakdown from the structural ratio and mass ratio.
    propellant_mass = (mass_ratio - 1) * payload_mass_kg / denominator
    structure_mass = structural_ratio * propellant_mass
    total_mass = payload_mass_kg + structure_mass + propellant_mass

    # Thrust requirement
    thrust_n = thrust_to_weight * total_mass * G0

    # Rough geometry: propellant takes 70% of the volume, L/D = 10.
    total_volume = propellant_mass / density / 0.7
    ld_ratio = 10.0
    diameter = (4 * total_volume / (PI * ld_ratio)) ** (1 / 3)

    result.update(
        {
            "feasible": True,
            "mass_ratio": mass_ratio,
            "total_mass_kg": total_mass,
            "propellant_mass_kg": propellant_mass,
            "structure_mass_kg": structure_mass,
            "thrust_n": thrust_n,
            "diameter_m": float(diameter),
            "length_m": float(ld_ratio * diameter),
        }
    )
    return result


# Update availability
try:
    from . import update_availability

    update_availability("rockets", True, {})
except ImportError:
    pass
