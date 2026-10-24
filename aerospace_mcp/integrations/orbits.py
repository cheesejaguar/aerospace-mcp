"""Orbital mechanics and spacecraft trajectory analysis tools for aerospace MCP.

Provides orbital mechanics calculations including:
    - Classical orbital element conversions (Keplerian elements <-> state vectors)
    - Kepler equation solver (Newton-Raphson iteration for eccentric anomaly)
    - Orbit propagation with J2 secular perturbations (RAAN and argument of
      periapsis drift rates)
    - Hohmann transfer orbit delta-V computation (vis-viva equation)
    - Lambert problem solver (Izzo, 2015) for two-body trajectory design
    - Ground track calculation with Earth rotation correction
    - Orbital rendezvous planning (phasing orbits, circularization)
    - Porkchop plot analysis for interplanetary transfer windows
    - Ephemeris position lookup (simplified circular orbits or SPICE kernels)

The numerical kernels (Kepler's equation, element/state conversions, J2
propagation, ground tracks, Hohmann and Lambert) run in the native Rust core
(``aerospace_mcp._native``, sources in ``rust/src/orbit.rs`` and
``rust/src/lambert.rs``); this module keeps the Python data model, epoch
handling, and the higher-level planning functions. SPICE ephemerides are used
when ``spiceypy`` is installed.

References:
    - Bate, Mueller, White, "Fundamentals of Astrodynamics" (1971)
    - Vallado, "Fundamentals of Astrodynamics and Applications" (4th ed., 2013)
    - Curtis, "Orbital Mechanics for Engineering Students" (4th ed., 2020)

WARNING: This module is for educational and research purposes only.
Do NOT use for real flight planning, navigation, or spacecraft operations.
"""

import math
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from .. import _native

# ===========================================================================
# Physical Constants
# ===========================================================================

# Earth's standard gravitational parameter (mu = G * M_Earth).
# Source: IERS Conventions (2010), Table 1.1.
# Units: m^3 / s^2
MU_EARTH = 3.986004418e14

# Earth's equatorial radius per WGS-84 ellipsoid.
# Source: NIMA Technical Report TR8350.2, "Department of Defense World
# Geodetic System 1984" (3rd ed., 2000).
# Units: meters
R_EARTH = 6378137.0

# Earth's second zonal harmonic (oblateness) coefficient.
# Captures the dominant gravitational perturbation due to the equatorial
# bulge.  Causes secular drift in RAAN and argument of periapsis.
# Source: EGM-96 / JGM-3 gravity model.
# Dimensionless.
J2_EARTH = 1.08262668e-3

# Earth's mean sidereal rotation rate.
# Source: IERS Conventions.
# Units: rad / s
OMEGA_EARTH = 7.2921159e-5


# ===========================================================================
# Data Classes -- Orbital State Representations
# ===========================================================================


@dataclass
class OrbitElements:
    """Classical (Keplerian) orbital elements.

    The six classical orbital elements uniquely define a two-body conic
    orbit and the position of a body along that orbit at a given epoch.

    Attributes:
        semi_major_axis_m: Semi-major axis *a* (m). Defines orbit size.
        eccentricity: Eccentricity *e* (dimensionless, 0 = circle, <1 = ellipse).
        inclination_deg: Inclination *i* (degrees). Angle between the
            orbital plane and the equatorial plane.
        raan_deg: Right ascension of ascending node *Omega* (degrees).
            Angle in the equatorial plane from the vernal equinox to the
            ascending node.
        arg_periapsis_deg: Argument of periapsis *omega* (degrees). Angle
            in the orbital plane from the ascending node to periapsis.
        true_anomaly_deg: True anomaly *nu* (degrees). Angle in the
            orbital plane from periapsis to the current position.
        epoch_utc: Reference epoch in ISO-8601 UTC format.
    """

    semi_major_axis_m: float  # Semi-major axis (m)
    eccentricity: float  # Eccentricity (dimensionless)
    inclination_deg: float  # Inclination (degrees)
    raan_deg: float  # Right ascension of ascending node (degrees)
    arg_periapsis_deg: float  # Argument of periapsis (degrees)
    true_anomaly_deg: float  # True anomaly (degrees)
    epoch_utc: str  # Epoch in UTC ISO format


@dataclass
class StateVector:
    """Cartesian position and velocity state vector in an inertial frame.

    Attributes:
        position_m: Position vector [x, y, z] in meters.
        velocity_ms: Velocity vector [vx, vy, vz] in m/s.
        epoch_utc: Reference epoch in ISO-8601 UTC format.
        frame: Reference frame identifier (default ``"J2000"``).
    """

    position_m: list[float]  # Position vector [x, y, z] in meters
    velocity_ms: list[float]  # Velocity vector [vx, vy, vz] in m/s
    epoch_utc: str  # Epoch in UTC ISO format
    frame: str = "J2000"  # Reference frame


@dataclass
class OrbitProperties:
    """Derived physical properties of a two-body orbit.

    Attributes:
        period_s: Orbital period T = 2*pi*sqrt(a^3/mu) in seconds.
        apoapsis_m: Apoapsis altitude above Earth surface (m).
        periapsis_m: Periapsis altitude above Earth surface (m).
        energy_j_kg: Specific orbital energy epsilon = -mu/(2a) in J/kg.
        angular_momentum_m2s: Specific angular momentum magnitude
            h = sqrt(mu*a*(1-e^2)) in m^2/s.
    """

    period_s: float  # Orbital period (seconds)
    apoapsis_m: float  # Apoapsis altitude above Earth surface (m)
    periapsis_m: float  # Periapsis altitude above Earth surface (m)
    energy_j_kg: float  # Specific orbital energy (J/kg)
    angular_momentum_m2s: float  # Specific angular momentum magnitude (m²/s)


@dataclass
class GroundTrack:
    """Sub-satellite ground track point (latitude, longitude, altitude).

    Attributes:
        latitude_deg: Geodetic latitude in degrees (-90 to +90).
        longitude_deg: Geodetic longitude in degrees (-180 to +180).
        altitude_m: Altitude above Earth surface in meters.
        time_utc: UTC timestamp in ISO-8601 format.
    """

    latitude_deg: float
    longitude_deg: float
    altitude_m: float
    time_utc: str


@dataclass
class Maneuver:
    """Impulsive orbital maneuver definition.

    Attributes:
        delta_v_ms: Impulsive delta-V vector [dvx, dvy, dvz] in m/s.
        time_utc: Maneuver execution epoch in ISO-8601 UTC.
        description: Human-readable description of the maneuver.
    """

    delta_v_ms: list[float]  # Delta-V vector [x, y, z] in m/s
    time_utc: str  # Maneuver execution time
    description: str = ""  # Optional description


# ===========================================================================
# Vector Utility Functions
# ===========================================================================


def deg_to_rad(deg: float) -> float:
    """Convert degrees to radians.

    Args:
        deg: Angle in degrees.

    Returns:
        Angle in radians.
    """
    return deg * math.pi / 180.0


def rad_to_deg(rad: float) -> float:
    """Convert radians to degrees.

    Args:
        rad: Angle in radians.

    Returns:
        Angle in degrees.
    """
    return rad * 180.0 / math.pi


def vector_magnitude(vec: list[float]) -> float:
    """Calculate the Euclidean (L2) norm of a vector.

    Args:
        vec: Input vector of arbitrary dimension.

    Returns:
        Scalar magnitude ||vec||.
    """
    return math.sqrt(sum(x**2 for x in vec))


def vector_dot(a: list[float], b: list[float]) -> float:
    """Calculate the dot (inner) product of two vectors.

    Args:
        a: First vector.
        b: Second vector (same length as *a*).

    Returns:
        Scalar dot product a . b.
    """
    return sum(a[i] * b[i] for i in range(len(a)))


def vector_cross(a: list[float], b: list[float]) -> list[float]:
    """Calculate the cross product of two 3-D vectors.

    Args:
        a: First 3-D vector.
        b: Second 3-D vector.

    Returns:
        Cross product vector a x b.
    """
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def vector_normalize(vec: list[float]) -> list[float]:
    """Return the unit vector in the direction of *vec*.

    Args:
        vec: Input 3-D vector.

    Returns:
        Unit vector (or zero vector if magnitude is zero).
    """
    mag = vector_magnitude(vec)
    if mag == 0:
        return [0.0, 0.0, 0.0]
    return [x / mag for x in vec]


# ===========================================================================
# Kepler Equation Solver
# ===========================================================================


def kepler_equation_solver(
    mean_anomaly_rad: float,
    eccentricity: float,
    tolerance: float = 1e-8,
    max_iterations: int = 50,
) -> float:
    """Solve Kepler's equation for eccentric anomaly via Newton-Raphson.

    Kepler's equation relates mean anomaly *M* and eccentric anomaly *E*
    for an elliptical orbit::

        M = E - e * sin(E)

    This transcendental equation has no closed-form solution and must be
    solved iteratively.  The Newton-Raphson update is::

        E_{n+1} = E_n - f(E_n) / f'(E_n)

    where:
        f(E)  = E - e*sin(E) - M
        f'(E) = 1 - e*cos(E)

    Convergence is typically achieved in 3-6 iterations for e < 0.9.

    Args:
        mean_anomaly_rad: Mean anomaly *M* in radians.
        eccentricity: Orbital eccentricity *e* (0 <= e < 1 for ellipses).
        tolerance: Convergence tolerance on |E_{n+1} - E_n|.
        max_iterations: Maximum number of Newton-Raphson iterations.

    Returns:
        Eccentric anomaly *E* in radians.
    """
    return _native.kepler_solve(
        mean_anomaly_rad, eccentricity, tolerance, max_iterations
    )


# ===========================================================================
# Orbital Element <-> State Vector Conversions
# ===========================================================================


def elements_to_state_vector(
    elements: OrbitElements, mu: float = MU_EARTH
) -> StateVector:
    """Convert classical orbital elements to a Cartesian state vector.

    Algorithm (Vallado, 4th ed., Algorithm 10):
        1. Compute the orbital radius from the conic equation.
        2. Express position and velocity in the perifocal (PQW) frame.
        3. Rotate from PQW to the J2000 inertial frame using the
           3-1-3 Euler angle rotation sequence (RAAN, inclination,
           argument of periapsis).

    Args:
        elements: Classical (Keplerian) orbital elements.
        mu: Gravitational parameter of the central body (m^3/s^2).

    Returns:
        State vector in the J2000 Earth-centered inertial frame.
    """
    position, velocity = _native.elements_to_state(
        elements.semi_major_axis_m,
        elements.eccentricity,
        elements.inclination_deg,
        elements.raan_deg,
        elements.arg_periapsis_deg,
        elements.true_anomaly_deg,
        mu,
    )
    return StateVector(
        position_m=position,
        velocity_ms=velocity,
        epoch_utc=elements.epoch_utc,
        frame="J2000",
    )


def state_vector_to_elements(state: StateVector, mu: float = MU_EARTH) -> OrbitElements:
    """Convert a Cartesian state vector to classical orbital elements.

    Algorithm (Vallado, 4th ed., Algorithm 9):
        1. Compute specific angular momentum h = r x v.
        2. Compute specific orbital energy to get semi-major axis.
        3. Compute eccentricity vector from the vector identity.
        4. Determine inclination from h_z / |h|.
        5. Compute node vector n = K x h to find RAAN.
        6. Derive argument of periapsis and true anomaly using
           dot-product formulas with quadrant checks.

    Undefined angles are reported as zero (RAAN for equatorial orbits, the
    argument of periapsis for circular ones). For equatorial orbits the
    argument of periapsis holds the longitude of periapsis, and circular
    orbits measure the true anomaly from the node (or the x-axis when also
    equatorial), so the elements always round-trip through
    :func:`elements_to_state_vector`.

    Args:
        state: State vector in J2000 inertial frame.
        mu: Gravitational parameter of the central body (m^3/s^2).

    Returns:
        Classical (Keplerian) orbital elements.

    Raises:
        ValueError: For degenerate states (zero position, zero angular
            momentum, or exactly parabolic energy).
    """
    a, e, i, raan, arg_pe, nu = _native.state_to_elements(
        state.position_m, state.velocity_ms, mu
    )
    return OrbitElements(
        semi_major_axis_m=a,
        eccentricity=e,
        inclination_deg=i,
        raan_deg=raan,
        arg_periapsis_deg=arg_pe,
        true_anomaly_deg=nu,
        epoch_utc=state.epoch_utc,
    )


# ===========================================================================
# Orbit Property Calculations
# ===========================================================================


def calculate_orbit_properties(elements: OrbitElements) -> OrbitProperties:
    """Calculate derived physical properties of an orbit.

    Uses the vis-viva equation and Kepler's third law to compute
    period, apsides, energy, and angular momentum.

    Args:
        elements: Classical orbital elements.

    Returns:
        Computed orbital properties.
    """
    period, apoapsis, periapsis, energy, h = _native.orbit_properties(
        elements.semi_major_axis_m, elements.eccentricity, MU_EARTH, R_EARTH
    )
    return OrbitProperties(
        period_s=period,
        apoapsis_m=apoapsis,
        periapsis_m=periapsis,
        energy_j_kg=energy,
        angular_momentum_m2s=h,
    )


# ===========================================================================
# Orbit Propagation with J2 Perturbations
# ===========================================================================


def propagate_orbit_j2(
    initial_state: StateVector, time_span_s: float, time_step_s: float = 60.0
) -> list[StateVector]:
    """Propagate an orbit numerically with J2 oblateness perturbations.

    Uses a 4th-order Runge-Kutta (RK4) integrator with the equations
    of motion including the central body gravitational acceleration and
    the J2 zonal harmonic perturbation.

    The J2 perturbation acceleration in Cartesian coordinates is::

        a_J2_x = -(3/2) * J2 * mu * R_E^2 / r^5 * x * (1 - 5*(z/r)^2)
        a_J2_y = -(3/2) * J2 * mu * R_E^2 / r^5 * y * (1 - 5*(z/r)^2)
        a_J2_z = -(3/2) * J2 * mu * R_E^2 / r^5 * z * (3 - 5*(z/r)^2)

    This produces secular drift rates in RAAN and argument of periapsis:
        dOmega/dt = -(3/2) * n * J2 * (R_E/p)^2 * cos(i)
        domega/dt = -(3/2) * n * J2 * (R_E/p)^2 * (5/2 * sin^2(i) - 2)

    Integration runs natively (``rust/src/orbit.rs``). The final step is
    shortened so the last state lands exactly on ``time_span_s``.

    Args:
        initial_state: Initial state vector in the J2000 frame.
        time_span_s: Total propagation duration in seconds.
        time_step_s: Fixed integration time step in seconds.

    Returns:
        List of state vectors sampled at each time step, starting with
        ``initial_state``.

    Raises:
        ValueError: If the time step or span is out of bounds, or the
            resulting step count exceeds the safety cap.
    """
    samples = _native.propagate_j2(
        initial_state.position_m,
        initial_state.velocity_ms,
        time_span_s,
        time_step_s,
    )

    try:
        epoch = datetime.fromisoformat(initial_state.epoch_utc.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        epoch = datetime.now(UTC)

    states = [initial_state]
    for t, position, velocity in samples[1:]:
        states.append(
            StateVector(
                position_m=position,
                velocity_ms=velocity,
                epoch_utc=(epoch + timedelta(seconds=t)).isoformat(),
                frame=initial_state.frame,
            )
        )
    return states


# ===========================================================================
# Ground Track Computation
# ===========================================================================


def _elapsed_seconds(
    orbit_states: list[StateVector], time_step_s: float
) -> list[float]:
    """Seconds since the first state, from epochs if usable, else index * step."""
    try:
        epochs = [
            datetime.fromisoformat(s.epoch_utc.replace("Z", "+00:00"))
            for s in orbit_states
        ]
        elapsed = [(e - epochs[0]).total_seconds() for e in epochs]
    except (ValueError, TypeError, AttributeError):
        elapsed = []
    if elapsed and all(b > a for a, b in zip(elapsed, elapsed[1:], strict=False)):
        return elapsed
    return [i * time_step_s for i in range(len(orbit_states))]


def calculate_ground_track(
    orbit_states: list[StateVector], time_step_s: float = 60.0
) -> list[GroundTrack]:
    """Calculate the sub-satellite ground track from orbit state vectors.

    Converts ECI position to latitude/longitude (spherical Earth) by
    accounting for Earth's rotation.  Longitude is adjusted by subtracting
    ``OMEGA_EARTH * t`` (``t`` = time since the first state) to approximate
    the ECI-to-ECEF transformation (ignoring precession/nutation for
    simplicity) and wrapped into [-180, 180).

    ``t`` comes from the states' epochs when they parse and strictly
    increase (as states from :func:`propagate_orbit_j2` do, including its
    shortened final step); otherwise state ``i`` is taken to be at
    ``i * time_step_s``.

    Args:
        orbit_states: List of state vectors in the J2000 frame.
        time_step_s: Time step between consecutive states (seconds), used
            when the epochs can't provide the timing.

    Returns:
        List of ground track points with lat/lon/alt.

    Raises:
        ValueError: If the time step is below the minimum or the state
            list exceeds the safety cap.
    """
    if time_step_s < 0.1:
        raise ValueError("time_step_s must be >= 0.1 s")
    if len(orbit_states) > 100_001:
        raise ValueError("too many orbit states (max 100001)")
    track = _native.ground_track(
        [s.position_m for s in orbit_states],
        _elapsed_seconds(orbit_states, time_step_s),
    )
    return [
        GroundTrack(
            latitude_deg=lat,
            longitude_deg=lon,
            altitude_m=alt,
            time_utc=state.epoch_utc,
        )
        for (lat, lon, alt), state in zip(track, orbit_states, strict=True)
    ]


# ===========================================================================
# Orbital Maneuver Calculations
# ===========================================================================


def hohmann_transfer(
    r1_m: float, r2_m: float, mu: float = MU_EARTH
) -> dict[str, float]:
    """Calculate a Hohmann transfer between two circular orbits.

    The Hohmann transfer is the minimum-energy two-impulse transfer
    between coplanar circular orbits.  It uses an elliptical transfer
    orbit that is tangent to both the initial and final circular orbits.

    Delta-V derivation (vis-viva equation: v^2 = mu*(2/r - 1/a)):
        1. Transfer orbit semi-major axis: a_t = (r1 + r2) / 2
        2. At periapsis (r = r1):
           v_t1 = sqrt(mu * (2/r1 - 1/a_t))
           dv1 = v_t1 - v_circ1
        3. At apoapsis (r = r2):
           v_t2 = sqrt(mu * (2/r2 - 1/a_t))
           dv2 = v_circ2 - v_t2

    Args:
        r1_m: Initial circular orbit radius (m) -- measured from Earth center.
        r2_m: Final circular orbit radius (m) -- measured from Earth center.
        mu: Gravitational parameter of the central body (m^3/s^2).

    Returns:
        Dictionary with delta-V values, transfer time, and transfer
        semi-major axis.

    Raises:
        ValueError: If either radius is not positive.
    """
    dv1, dv2, dv_total, transfer_time, a_transfer = _native.hohmann_transfer(
        r1_m, r2_m, mu
    )
    return {
        "delta_v1_ms": dv1,
        "delta_v2_ms": dv2,
        "total_delta_v_ms": dv_total,
        "transfer_time_s": transfer_time,
        "transfer_time_h": transfer_time / 3600,
        "transfer_semi_major_axis_m": a_transfer,
    }


# ===========================================================================
# Lambert Problem Solver
# ===========================================================================


def solve_lambert(
    r1_vec: list[float],
    r2_vec: list[float],
    time_flight_s: float,
    mu: float = MU_EARTH,
    prograde: bool = True,
    revolutions: int = 0,
    low_path: bool = True,
) -> dict[str, Any]:
    """Solve Lambert's problem with Izzo's algorithm.

    Finds the two-body orbit connecting two position vectors in a given
    time of flight (D. Izzo, "Revisiting Lambert's problem", CMDA 2015).
    The solver runs natively (``rust/src/lambert.rs``) and replaces the
    archived poliastro implementation.

    Args:
        r1_vec: Departure position vector (m).
        r2_vec: Arrival position vector (m).
        time_flight_s: Time of flight (s), > 0.
        mu: Gravitational parameter of the central body (m^3/s^2).
        prograde: Counter-clockwise motion about +z if True.
        revolutions: Complete revolutions before arrival (0 = direct).
        low_path: For multi-revolution transfers, which of the two solutions
            to return (poliastro's ``lowpath`` convention; this selects a
            branch, it does not guarantee the lower-energy one). Ignored
            when ``revolutions == 0``.

    Returns:
        Dictionary with ``v1_ms`` / ``v2_ms`` (velocity vectors at departure
        and arrival) and ``transfer_angle_deg`` (0-360, the angle the
        solution actually traverses in the direction of motion).

    Raises:
        ValueError: Invalid input, collinear position vectors (0 or 180
            degree transfers have no unique plane), or more revolutions than
            the time of flight allows.
        RuntimeError: If the iteration fails to converge.
    """
    if not isinstance(revolutions, int) or revolutions < 0:
        raise ValueError("revolutions must be a non-negative integer")
    v1, v2, transfer_angle_rad = _native.lambert_izzo(
        mu, r1_vec, r2_vec, time_flight_s, revolutions, prograde, low_path
    )

    return {
        "v1_ms": v1,
        "v2_ms": v2,
        "transfer_angle_deg": rad_to_deg(transfer_angle_rad),
        "revolutions": revolutions,
        "prograde": prograde,
    }


# ===========================================================================
# Orbital Rendezvous Planning
# ===========================================================================


def orbital_rendezvous_planning(
    chaser_elements: OrbitElements, target_elements: OrbitElements
) -> dict[str, Any]:
    """Plan orbital rendezvous maneuvers between two spacecraft.

    Computes the relative geometry (phase angle, distance) between a
    chaser and target spacecraft, estimates the synodic period for
    phasing, and generates a simplified maneuver sequence including
    circularization and altitude-matching burns.

    Args:
        chaser_elements: Chaser spacecraft orbital elements.
        target_elements: Target spacecraft orbital elements.

    Returns:
        Rendezvous plan dictionary with relative distance, phase angle,
        phasing time, delta-V estimates, and maneuver list.
    """
    # Convert to state vectors for analysis
    chaser_state = elements_to_state_vector(chaser_elements)
    target_state = elements_to_state_vector(target_elements)

    # Calculate relative position
    rel_pos = [
        target_state.position_m[i] - chaser_state.position_m[i] for i in range(3)
    ]
    rel_distance = vector_magnitude(rel_pos)

    # Orbital properties
    chaser_props = calculate_orbit_properties(chaser_elements)
    target_props = calculate_orbit_properties(target_elements)

    # Phase angle between spacecraft
    chaser_r = vector_magnitude(chaser_state.position_m)
    target_r = vector_magnitude(target_state.position_m)

    cos_phase = vector_dot(chaser_state.position_m, target_state.position_m) / (
        chaser_r * target_r
    )
    cos_phase = max(-1, min(1, cos_phase))
    phase_angle_rad = math.acos(cos_phase)

    # Estimate phasing time
    if abs(chaser_props.period_s - target_props.period_s) > 1:
        synodic_period = abs(
            chaser_props.period_s
            * target_props.period_s
            / (chaser_props.period_s - target_props.period_s)
        )
        phasing_time_s = phase_angle_rad / (2 * math.pi) * synodic_period
    else:
        phasing_time_s = float("inf")  # Coplanar, similar orbits

    # Delta-V estimates for circularization if needed
    chaser_circ_dv = 0.0

    if chaser_elements.eccentricity > 0.01:
        # Circularization delta-V (rough estimate)
        v_ap = math.sqrt(
            MU_EARTH
            * (
                2
                / (
                    chaser_elements.semi_major_axis_m
                    * (1 + chaser_elements.eccentricity)
                )
                - 1 / chaser_elements.semi_major_axis_m
            )
        )
        v_circ = math.sqrt(
            MU_EARTH
            / (chaser_elements.semi_major_axis_m * (1 + chaser_elements.eccentricity))
        )
        chaser_circ_dv = abs(v_circ - v_ap)

    # Create sample maneuvers
    maneuvers = []

    # Add a simple phasing maneuver
    if chaser_circ_dv > 0:
        maneuvers.append(
            {
                "delta_v_ms": chaser_circ_dv,
                "type": "circularization",
                "description": "Circularize chaser orbit",
            }
        )

    # Add altitude matching maneuver if needed
    altitude_diff = abs(chaser_props.apoapsis_m - target_props.apoapsis_m)
    if altitude_diff > 1000:  # More than 1 km difference
        altitude_dv = 50.0  # Rough estimate for altitude adjustment
        maneuvers.append(
            {
                "delta_v_ms": altitude_dv,
                "type": "altitude_adjustment",
                "description": "Match target altitude",
            }
        )

    # Calculate total delta-V
    total_dv = sum(m["delta_v_ms"] for m in maneuvers)

    return {
        "relative_distance_m": rel_distance,
        "relative_distance_km": rel_distance / 1000,
        "phase_angle_deg": rad_to_deg(phase_angle_rad),
        "phasing_time_s": phasing_time_s,
        "phasing_time_h": (
            phasing_time_s / 3600 if phasing_time_s != float("inf") else float("inf")
        ),
        "time_to_rendezvous_s": phasing_time_s,  # Add expected key
        "chaser_period_s": chaser_props.period_s,
        "target_period_s": target_props.period_s,
        "period_difference_s": abs(chaser_props.period_s - target_props.period_s),
        "altitude_difference_m": abs(chaser_props.apoapsis_m - target_props.apoapsis_m),
        "estimated_circularization_dv_ms": chaser_circ_dv,
        "total_delta_v_ms": total_dv,  # Add expected key
        "maneuvers": maneuvers,  # Add expected key
        "feasibility": (
            "Good"
            if rel_distance < 100000
            and abs(chaser_elements.inclination_deg - target_elements.inclination_deg)
            < 5.0
            else "Challenging"
        ),
    }


# ===========================================================================
# Interplanetary Porkchop Plot Analysis
# ===========================================================================


def porkchop_plot_analysis(
    departure_body: str = "Earth",
    arrival_body: str = "Mars",
    departure_dates: list[str] = None,
    arrival_dates: list[str] = None,
    min_tof_days: int = 100,
    max_tof_days: int = 400,
) -> dict[str, Any]:
    """Generate porkchop plot data for interplanetary transfer windows.

    A porkchop plot maps departure date vs. arrival date (or time of
    flight) with contours of characteristic energy C3 or delta-V.
    Each cell solves Lambert's problem (Izzo) between the planets and
    reports C3 = |v1 - v_planet|^2 at departure. Planetary orbits are
    simplified to coplanar circles; production applications should use
    JPL SPICE ephemerides.

    Args:
        departure_body: Name of the departure celestial body.
        arrival_body: Name of the arrival celestial body.
        departure_dates: List of departure epoch strings (ISO-8601).
        arrival_dates: List of arrival epoch strings (ISO-8601).
        min_tof_days: Minimum allowed time of flight in days.
        max_tof_days: Maximum allowed time of flight in days.

    Returns:
        Dictionary containing the transfer grid, optimal transfer,
        and summary statistics.
    """
    # Default date ranges if not provided
    if departure_dates is None:
        # Earth-Mars synodic period is ~26 months, sample over 2 years
        departure_dates = [
            "2025-01-01T00:00:00",
            "2025-03-01T00:00:00",
            "2025-05-01T00:00:00",
            "2025-07-01T00:00:00",
            "2025-09-01T00:00:00",
            "2025-11-01T00:00:00",
            "2026-01-01T00:00:00",
            "2026-03-01T00:00:00",
            "2026-05-01T00:00:00",
            "2026-07-01T00:00:00",
            "2026-09-01T00:00:00",
            "2026-11-01T00:00:00",
        ]

    if arrival_dates is None:
        # Generate arrival dates based on TOF constraints
        arrival_dates = [
            "2025-06-01T00:00:00",
            "2025-08-01T00:00:00",
            "2025-10-01T00:00:00",
            "2025-12-01T00:00:00",
            "2026-02-01T00:00:00",
            "2026-04-01T00:00:00",
            "2026-06-01T00:00:00",
            "2026-08-01T00:00:00",
            "2026-10-01T00:00:00",
            "2026-12-01T00:00:00",
            "2027-02-01T00:00:00",
            "2027-04-01T00:00:00",
        ]

    # Simplified planetary positions (circular orbits for demonstration)
    # In production, would use SPICE kernels or JPL ephemeris
    earth_orbit_radius = 1.0  # AU
    mars_orbit_radius = 1.524  # AU
    earth_period_days = 365.25
    mars_period_days = 686.98

    mu_sun = 1.32712440018e20  # Sun's gravitational parameter (m^3/s^2)
    au_m = 1.496e11

    def get_planet_state(body: str, date_iso: str) -> tuple[list[float], list[float]]:
        """Simplified heliocentric position/velocity (circular, coplanar orbit)."""
        try:
            dt = datetime.fromisoformat(date_iso.replace("Z", "+00:00"))
        except Exception:
            dt = datetime.fromisoformat(date_iso)
        if dt.tzinfo is not None:
            dt = dt.astimezone(UTC).replace(tzinfo=None)

        days_since_j2000 = (dt - datetime(2000, 1, 1, 12, 0, 0)).total_seconds() / 86400

        # Unknown bodies default to Earth.
        if body.lower() == "mars":
            radius_au, period_days = mars_orbit_radius, mars_period_days
        else:
            radius_au, period_days = earth_orbit_radius, earth_period_days

        mean_anomaly = 2 * math.pi * days_since_j2000 / period_days
        r_m = radius_au * au_m
        v_circ = math.sqrt(mu_sun / r_m)
        position = [r_m * math.cos(mean_anomaly), r_m * math.sin(mean_anomaly), 0.0]
        velocity = [
            -v_circ * math.sin(mean_anomaly),
            v_circ * math.cos(mean_anomaly),
            0.0,
        ]
        return position, velocity

    # Generate porkchop data grid
    porkchop_data = []

    for dep_date in departure_dates:
        for arr_date in arrival_dates:
            dep_dt = datetime.fromisoformat(dep_date.replace("Z", "+00:00"))
            arr_dt = datetime.fromisoformat(arr_date.replace("Z", "+00:00"))
            tof_days = (arr_dt - dep_dt).total_seconds() / 86400

            # Filter by TOF constraints
            if tof_days < min_tof_days or tof_days > max_tof_days:
                continue

            r1, v_dep_planet = get_planet_state(departure_body, dep_date)
            r2, _ = get_planet_state(arrival_body, arr_date)

            try:
                lambert_result = solve_lambert(r1, r2, tof_days * 86400, mu_sun)

                # Hyperbolic excess velocity relative to the departure planet
                v1 = lambert_result["v1_ms"]
                v_inf_mag = vector_magnitude(
                    [v1[i] - v_dep_planet[i] for i in range(3)]
                )

                # Characteristic energy C3 = v_inf^2 (km^2/s^2)
                c3 = (v_inf_mag / 1000) ** 2

                porkchop_data.append(
                    {
                        "departure_date": dep_date,
                        "arrival_date": arr_date,
                        "time_of_flight_days": tof_days,
                        "c3_km2_s2": c3,
                        "delta_v_ms": v_inf_mag,
                        "departure_position_m": r1,
                        "arrival_position_m": r2,
                        "transfer_feasible": c3 < 100.0,  # Reasonable C3 limit
                    }
                )

            except (ValueError, RuntimeError) as e:
                # No Lambert solution (e.g. exactly 180-degree geometry)
                porkchop_data.append(
                    {
                        "departure_date": dep_date,
                        "arrival_date": arr_date,
                        "time_of_flight_days": tof_days,
                        "c3_km2_s2": None,
                        "delta_v_ms": None,
                        "departure_position_m": r1,
                        "arrival_position_m": r2,
                        "transfer_feasible": False,
                        "error": str(e),
                    }
                )

    # Find optimal transfer
    feasible_transfers = [t for t in porkchop_data if t["transfer_feasible"]]

    optimal_transfer = None
    if feasible_transfers:
        # Find minimum C3 transfer
        optimal_transfer = min(feasible_transfers, key=lambda x: x["c3_km2_s2"])

    # Generate summary statistics
    if feasible_transfers:
        c3_values = [t["c3_km2_s2"] for t in feasible_transfers]
        tof_values = [t["time_of_flight_days"] for t in feasible_transfers]

        summary_stats = {
            "feasible_transfers": len(feasible_transfers),
            "total_transfers_computed": len(porkchop_data),
            "min_c3_km2_s2": min(c3_values),
            "max_c3_km2_s2": max(c3_values),
            "mean_c3_km2_s2": sum(c3_values) / len(c3_values),
            "min_tof_days": min(tof_values),
            "max_tof_days": max(tof_values),
            "mean_tof_days": sum(tof_values) / len(tof_values),
        }
    else:
        summary_stats = {
            "feasible_transfers": 0,
            "total_transfers_computed": len(porkchop_data),
            "min_c3_km2_s2": None,
            "max_c3_km2_s2": None,
            "mean_c3_km2_s2": None,
            "min_tof_days": min_tof_days,
            "max_tof_days": max_tof_days,
            "mean_tof_days": (min_tof_days + max_tof_days) / 2,
        }

    return {
        "departure_body": departure_body,
        "arrival_body": arrival_body,
        "transfer_grid": porkchop_data,
        "optimal_transfer": optimal_transfer,
        "summary_statistics": summary_stats,
        "constraints": {
            "min_tof_days": min_tof_days,
            "max_tof_days": max_tof_days,
            "max_feasible_c3_km2_s2": 100.0,
        },
        "note": "Simplified analysis using circular planetary orbits. Use SPICE kernels for production applications.",
    }


# ===========================================================================
# Ephemeris Position Lookup
# ===========================================================================


def get_ephemeris_position(
    body: str, epoch_utc: str, frame: str = "J2000"
) -> dict[str, Any]:
    """Get the heliocentric position and velocity of a celestial body.

    This is a simplified implementation that models planetary orbits as
    circles with constant angular velocity.  For production mission
    design, use SPICE kernels via ``spiceypy``.

    Args:
        body: Celestial body name (``"Earth"``, ``"Mars"``, ``"Venus"``,
            or ``"Jupiter"``).
        epoch_utc: UTC epoch in ISO-8601 format.
        frame: Reference frame identifier (default ``"J2000"``).

    Returns:
        Dictionary with position (m), velocity (m/s), accuracy level,
        and SPICE recommendations.
    """
    # Import required for date parsing
    import datetime

    try:
        dt = datetime.datetime.fromisoformat(epoch_utc.replace("Z", "+00:00"))
    except Exception:
        dt = datetime.datetime.fromisoformat(epoch_utc)

    # Days since J2000.0 epoch
    j2000 = datetime.datetime(2000, 1, 1, 12, 0, 0)
    days_since_j2000 = (dt - j2000).total_seconds() / 86400

    # Simplified planetary orbital elements (circular orbits)
    orbital_data = {
        "earth": {
            "semi_major_axis_au": 1.0,
            "period_days": 365.25,
            "inclination_deg": 0.0,
        },
        "mars": {
            "semi_major_axis_au": 1.524,
            "period_days": 686.98,
            "inclination_deg": 1.85,
        },
        "venus": {
            "semi_major_axis_au": 0.723,
            "period_days": 224.70,
            "inclination_deg": 3.39,
        },
        "jupiter": {
            "semi_major_axis_au": 5.204,
            "period_days": 4332.82,
            "inclination_deg": 1.30,
        },
    }

    body_lower = body.lower()
    if body_lower not in orbital_data:
        return {
            "error": f"Body '{body}' not supported in simplified ephemeris",
            "supported_bodies": list(orbital_data.keys()),
            "recommendation": "Use SPICE kernels with spiceypy for accurate ephemeris data",
        }

    data = orbital_data[body_lower]

    # Calculate mean anomaly
    mean_anomaly = 2 * math.pi * days_since_j2000 / data["period_days"]

    # Position in heliocentric frame (simplified circular orbit)
    r_au = data["semi_major_axis_au"]
    x_au = r_au * math.cos(mean_anomaly)
    y_au = r_au * math.sin(mean_anomaly)
    z_au = 0.0  # Simplified: ignore inclination

    # Convert to meters
    AU_TO_M = 1.496e11
    position_m = [x_au * AU_TO_M, y_au * AU_TO_M, z_au * AU_TO_M]

    # Velocity (circular orbit)
    mu_sun = 1.32712440018e20  # m³/s²
    r_m = r_au * AU_TO_M
    v_circular = math.sqrt(mu_sun / r_m)  # m/s

    velocity_ms = [
        -v_circular * math.sin(mean_anomaly),
        v_circular * math.cos(mean_anomaly),
        0.0,
    ]

    return {
        "body": body,
        "epoch_utc": epoch_utc,
        "frame": frame,
        "position_m": position_m,
        "velocity_ms": velocity_ms,
        "accuracy": "low",
        "model": "circular_orbit_approximation",
        "warning": "This is a simplified model. Use SPICE kernels for mission-critical applications.",
        "spice_recommendation": {
            "kernels_needed": ["de430.bsp", "pck00010.tpc", "naif0012.tls"],
            "library": "spiceypy",
            "installation": "pip install spiceypy",
        },
    }


# ===========================================================================
# Optional SPICE Integration
# ===========================================================================

SPICE_AVAILABLE = False
try:
    import spiceypy as spice

    SPICE_AVAILABLE = True

    def get_ephemeris_position_spice(
        body: str, epoch_utc: str, frame: str = "J2000"
    ) -> dict[str, Any]:
        """
        Get accurate ephemeris position using SPICE kernels (if loaded).

        Args:
            body: SPICE body name or ID
            epoch_utc: UTC epoch in ISO format
            frame: SPICE reference frame

        Returns:
            High-accuracy position and velocity vectors
        """
        try:
            # Convert ISO time to ET (Ephemeris Time)
            et = spice.str2et(epoch_utc)

            # Get state vector (position and velocity)
            state, light_time = spice.spkezr(body, et, frame, "NONE", "SUN")

            # Convert km to m, km/s to m/s
            position_m = [state[i] * 1000 for i in range(3)]
            velocity_ms = [state[i + 3] * 1000 for i in range(3)]

            return {
                "body": body,
                "epoch_utc": epoch_utc,
                "frame": frame,
                "position_m": position_m,
                "velocity_ms": velocity_ms,
                "light_time_s": light_time,
                "accuracy": "high",
                "model": "SPICE_kernels",
                "source": "JPL_ephemeris",
            }

        except Exception as e:
            return {
                "error": f"SPICE error: {str(e)}",
                "fallback_available": True,
                "recommendation": "Check SPICE kernel loading and body names",
            }

except ImportError:

    def get_ephemeris_position_spice(
        body: str, epoch_utc: str, frame: str = "J2000"
    ) -> dict[str, Any]:
        """Placeholder when SPICE is not available."""
        return {
            "error": "SPICE not available",
            "install_command": "pip install spiceypy",
            "kernel_sources": [
                "https://naif.jpl.nasa.gov/naif/data_generic.html",
                "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/",
            ],
        }


try:
    from . import update_availability

    update_availability("orbits", True, {"spice_available": SPICE_AVAILABLE})
except ImportError:
    pass
