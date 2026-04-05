"""Orbital mechanics and spacecraft trajectory analysis tools for aerospace MCP.

Provides orbital mechanics calculations including:
    - Classical orbital element conversions (Keplerian elements <-> state vectors)
    - Kepler equation solver (Newton-Raphson iteration for eccentric anomaly)
    - Orbit propagation with J2 secular perturbations (RAAN and argument of
      periapsis drift rates)
    - Hohmann transfer orbit delta-V computation (vis-viva equation)
    - Simplified Lambert problem solver for two-body trajectory design
    - Ground track calculation with Earth rotation correction
    - Orbital rendezvous planning (phasing orbits, circularization)
    - Porkchop plot analysis for interplanetary transfer windows
    - Ephemeris position lookup (simplified circular orbits or SPICE kernels)

Falls back to manual implementations when optional libraries (poliastro,
astropy, spiceypy) are unavailable.

References:
    - Bate, Mueller, White, "Fundamentals of Astrodynamics" (1971)
    - Vallado, "Fundamentals of Astrodynamics and Applications" (4th ed., 2013)
    - Curtis, "Orbital Mechanics for Engineering Students" (4th ed., 2020)

WARNING: This module is for educational and research purposes only.
Do NOT use for real flight planning, navigation, or spacecraft operations.
"""

import math
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

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
    # Initial guess: for low eccentricity M is a good starting point;
    # for high eccentricity, pi avoids the nearly-flat region near E = 0.
    E = mean_anomaly_rad if eccentricity < 0.8 else math.pi

    for _ in range(max_iterations):
        # f(E)  = E - e*sin(E) - M   (Kepler's equation residual)
        f = E - eccentricity * math.sin(E) - mean_anomaly_rad
        # f'(E) = 1 - e*cos(E)       (derivative w.r.t. E)
        f_prime = 1 - eccentricity * math.cos(E)

        # Guard against near-zero derivative (degenerate case)
        if abs(f_prime) < 1e-12:
            break

        # Newton-Raphson update: E_{n+1} = E_n - f / f'
        E_new = E - f / f_prime

        if abs(E_new - E) < tolerance:
            return E_new

        E = E_new

    return E


# ===========================================================================
# Orbital Element <-> State Vector Conversions
# ===========================================================================


def elements_to_state_vector(elements: OrbitElements) -> StateVector:
    """Convert classical orbital elements to a Cartesian state vector.

    Algorithm (Vallado, 4th ed., Algorithm 10):
        1. Compute the orbital radius from the conic equation.
        2. Express position and velocity in the perifocal (PQW) frame.
        3. Rotate from PQW to the J2000 inertial frame using the
           3-1-3 Euler angle rotation sequence (RAAN, inclination,
           argument of periapsis).

    Args:
        elements: Classical (Keplerian) orbital elements.

    Returns:
        State vector in the J2000 Earth-centered inertial frame.
    """
    # Convert angles to radians
    i = deg_to_rad(elements.inclination_deg)
    raan = deg_to_rad(elements.raan_deg)
    arg_pe = deg_to_rad(elements.arg_periapsis_deg)
    nu = deg_to_rad(elements.true_anomaly_deg)

    # Semi-major axis and eccentricity
    a = elements.semi_major_axis_m
    e = elements.eccentricity

    # Conic equation: r = p / (1 + e*cos(nu)), where p = a*(1 - e^2)
    # is the semi-latus rectum.
    r = a * (1 - e**2) / (1 + e * math.cos(nu))

    # Position in perifocal (PQW) coordinates: x along periapsis, y
    # perpendicular in the orbital plane, z = 0 (by definition).
    r_peri = [r * math.cos(nu), r * math.sin(nu), 0.0]

    # Semi-latus rectum p = a*(1 - e^2)
    p = a * (1 - e**2)

    # Velocity in perifocal coordinates derived from the vis-viva
    # relations in the PQW frame:
    #   v_P = -sqrt(mu/p) * sin(nu)
    #   v_Q =  sqrt(mu/p) * (e + cos(nu))
    v_peri = [
        -math.sqrt(MU_EARTH / p) * math.sin(nu),
        math.sqrt(MU_EARTH / p) * (e + math.cos(nu)),
        0.0,
    ]

    # Pre-compute trigonometric values for the rotation matrix
    cos_raan, sin_raan = math.cos(raan), math.sin(raan)
    cos_i, sin_i = math.cos(i), math.sin(i)
    cos_arg, sin_arg = math.cos(arg_pe), math.sin(arg_pe)

    # Rotation matrix [R] = R3(-RAAN) * R1(-i) * R3(-omega)
    # Transforms perifocal (PQW) -> J2000 inertial frame.
    # See Vallado (2013), Eq. 4-44.
    R11 = cos_raan * cos_arg - sin_raan * sin_arg * cos_i
    R12 = -cos_raan * sin_arg - sin_raan * cos_arg * cos_i
    R13 = sin_raan * sin_i

    R21 = sin_raan * cos_arg + cos_raan * sin_arg * cos_i
    R22 = -sin_raan * sin_arg + cos_raan * cos_arg * cos_i
    R23 = -cos_raan * sin_i

    R31 = sin_arg * sin_i
    R32 = cos_arg * sin_i
    R33 = cos_i

    # Apply rotation: r_J2000 = [R] * r_PQW
    r_j2000 = [
        R11 * r_peri[0] + R12 * r_peri[1] + R13 * r_peri[2],
        R21 * r_peri[0] + R22 * r_peri[1] + R23 * r_peri[2],
        R31 * r_peri[0] + R32 * r_peri[1] + R33 * r_peri[2],
    ]

    # Apply same rotation to velocity: v_J2000 = [R] * v_PQW
    v_j2000 = [
        R11 * v_peri[0] + R12 * v_peri[1] + R13 * v_peri[2],
        R21 * v_peri[0] + R22 * v_peri[1] + R23 * v_peri[2],
        R31 * v_peri[0] + R32 * v_peri[1] + R33 * v_peri[2],
    ]

    return StateVector(
        position_m=r_j2000,
        velocity_ms=v_j2000,
        epoch_utc=elements.epoch_utc,
        frame="J2000",
    )


def state_vector_to_elements(state: StateVector) -> OrbitElements:
    """Convert a Cartesian state vector to classical orbital elements.

    Algorithm (Vallado, 4th ed., Algorithm 9):
        1. Compute specific angular momentum h = r x v.
        2. Compute specific orbital energy to get semi-major axis.
        3. Compute eccentricity vector from the vector identity.
        4. Determine inclination from h_z / |h|.
        5. Compute node vector n = K x h to find RAAN.
        6. Derive argument of periapsis and true anomaly using
           dot-product formulas with quadrant checks.

    Args:
        state: State vector in J2000 inertial frame.

    Returns:
        Classical (Keplerian) orbital elements.
    """
    r_vec = state.position_m
    v_vec = state.velocity_ms

    # Position and velocity magnitudes
    r = vector_magnitude(r_vec)
    v = vector_magnitude(v_vec)

    # Specific angular momentum: h = r x v
    h_vec = vector_cross(r_vec, v_vec)
    h = vector_magnitude(h_vec)

    # Specific orbital energy (vis-viva): epsilon = v^2/2 - mu/r
    # Semi-major axis from energy: a = -mu / (2*epsilon)
    energy = v**2 / 2 - MU_EARTH / r
    a = -MU_EARTH / (2 * energy)

    # Eccentricity vector: e_vec = (v x h)/mu - r_hat
    # Points from the focus toward periapsis; |e_vec| = eccentricity.
    v_cross_h = vector_cross(v_vec, h_vec)
    e_vec = [v_cross_h[i] / MU_EARTH - r_vec[i] / r for i in range(3)]
    e = vector_magnitude(e_vec)

    # Inclination: cos(i) = h_z / |h|  (angle between h and K-axis)
    i = math.acos(h_vec[2] / h)

    # Node vector: n = K x h (points toward the ascending node)
    k_vec = [0, 0, 1]
    n_vec = vector_cross(k_vec, h_vec)
    n = vector_magnitude(n_vec)

    # Right ascension of ascending node (RAAN): cos(Omega) = n_x / |n|
    # Quadrant check: if n_y < 0 then Omega is in [pi, 2*pi].
    if n > 1e-10:
        raan = math.acos(n_vec[0] / n)
        if n_vec[1] < 0:
            raan = 2 * math.pi - raan
    else:
        raan = 0.0  # Undefined for equatorial orbits; set to zero.

    # Argument of periapsis: cos(omega) = (n . e) / (|n| * |e|)
    # Quadrant check: if e_z < 0 then omega is in [pi, 2*pi].
    if n > 1e-10 and e > 1e-10:
        cos_arg_pe = vector_dot(n_vec, e_vec) / (n * e)
        cos_arg_pe = max(-1, min(1, cos_arg_pe))  # Clamp to [-1, 1]
        arg_pe = math.acos(cos_arg_pe)
        if e_vec[2] < 0:
            arg_pe = 2 * math.pi - arg_pe
    else:
        arg_pe = 0.0  # Undefined for circular or equatorial orbits.

    # True anomaly: cos(nu) = (e . r) / (|e| * |r|)
    # Quadrant check: if r . v < 0 then spacecraft is past apoapsis.
    if e > 1e-10:
        cos_nu = vector_dot(e_vec, r_vec) / (e * r)
        cos_nu = max(-1, min(1, cos_nu))  # Clamp for numerical safety
        nu = math.acos(cos_nu)
        if vector_dot(r_vec, v_vec) < 0:
            nu = 2 * math.pi - nu
    else:
        # For circular orbits, true anomaly measured from ascending node.
        if n > 1e-10:
            cos_nu = vector_dot(n_vec, r_vec) / (n * r)
            cos_nu = max(-1, min(1, cos_nu))
            nu = math.acos(cos_nu)
            if r_vec[2] < 0:
                nu = 2 * math.pi - nu
        else:
            # Circular equatorial orbit: use true longitude.
            nu = math.atan2(r_vec[1], r_vec[0])
            if nu < 0:
                nu += 2 * math.pi

    return OrbitElements(
        semi_major_axis_m=a,
        eccentricity=e,
        inclination_deg=rad_to_deg(i),
        raan_deg=rad_to_deg(raan),
        arg_periapsis_deg=rad_to_deg(arg_pe),
        true_anomaly_deg=rad_to_deg(nu),
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
    a = elements.semi_major_axis_m
    e = elements.eccentricity

    # Kepler's third law: T = 2*pi * sqrt(a^3 / mu)
    period = 2 * math.pi * math.sqrt(a**3 / MU_EARTH)

    # Apoapsis and periapsis *altitudes* (above Earth surface)
    # r_ap = a*(1+e),  r_pe = a*(1-e)  are orbital radii.
    r_ap = a * (1 + e) - R_EARTH
    r_pe = a * (1 - e) - R_EARTH

    # Specific orbital energy: epsilon = -mu / (2a)
    energy = -MU_EARTH / (2 * a)

    # Specific angular momentum: h = sqrt(mu * a * (1 - e^2))
    h = math.sqrt(MU_EARTH * a * (1 - e**2))

    return OrbitProperties(
        period_s=period,
        apoapsis_m=r_ap,
        periapsis_m=r_pe,
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

        a_J2_x = (3/2) * J2 * mu * R_E^2 / r^5 * x * (1 - 5*(z/r)^2)
        a_J2_y = (3/2) * J2 * mu * R_E^2 / r^5 * y * (1 - 5*(z/r)^2)
        a_J2_z = (3/2) * J2 * mu * R_E^2 / r^5 * z * (3 - 5*(z/r)^2)

    This produces secular drift rates in RAAN and argument of periapsis:
        dOmega/dt = -(3/2) * n * J2 * (R_E/p)^2 * cos(i)
        domega/dt = -(3/2) * n * J2 * (R_E/p)^2 * (5/2 * sin^2(i) - 2)

    Args:
        initial_state: Initial state vector in the J2000 frame.
        time_span_s: Total propagation duration in seconds.
        time_step_s: Fixed integration time step in seconds.

    Returns:
        List of state vectors sampled at each time step.
    """

    def acceleration_j2(r_vec: list[float]) -> list[float]:
        """Calculate total acceleration (central body + J2).

        Args:
            r_vec: Position vector [x, y, z] in meters.

        Returns:
            Acceleration vector [ax, ay, az] in m/s^2.
        """
        r = vector_magnitude(r_vec)

        # Two-body (Keplerian) central-force acceleration: a = -mu * r / |r|^3
        a_central = [-MU_EARTH * r_vec[i] / r**3 for i in range(3)]

        # J2 perturbation acceleration (see Vallado, Eq. 8-35).
        # factor = (3/2) * J2 * mu * R_E^2 / r^5
        factor = 1.5 * J2_EARTH * MU_EARTH * R_EARTH**2 / r**5
        # (z/r)^2 term determines latitude-dependent perturbation
        z2_r2 = (r_vec[2] / r) ** 2

        a_j2 = [
            factor * r_vec[0] * (1 - 5 * z2_r2),  # x-component
            factor * r_vec[1] * (1 - 5 * z2_r2),  # y-component
            factor * r_vec[2] * (3 - 5 * z2_r2),  # z-component (note: 3, not 1)
        ]

        return [a_central[i] + a_j2[i] for i in range(3)]

    # Initialize
    states = [initial_state]
    r = initial_state.position_m.copy()
    v = initial_state.velocity_ms.copy()

    # Parse initial epoch
    try:
        epoch = datetime.fromisoformat(initial_state.epoch_utc.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        epoch = datetime.now(UTC)

    # 4th-order Runge-Kutta (RK4) numerical integration.
    # State: [r, v]; derivatives: dr/dt = v, dv/dt = a(r).
    t = 0.0
    while t < time_span_s:
        dt = min(time_step_s, time_span_s - t)

        # RK4 stages -- k_r are velocity estimates, k_v are acceleration estimates
        k1_r = v
        k1_v = acceleration_j2(r)

        r2 = [r[i] + 0.5 * dt * k1_r[i] for i in range(3)]
        v2 = [v[i] + 0.5 * dt * k1_v[i] for i in range(3)]
        k2_r = v2
        k2_v = acceleration_j2(r2)

        r3 = [r[i] + 0.5 * dt * k2_r[i] for i in range(3)]
        v3 = [v[i] + 0.5 * dt * k2_v[i] for i in range(3)]
        k3_r = v3
        k3_v = acceleration_j2(r3)

        r4 = [r[i] + dt * k3_r[i] for i in range(3)]
        v4 = [v[i] + dt * k3_v[i] for i in range(3)]
        k4_r = v4
        k4_v = acceleration_j2(r4)

        # RK4 weighted update:
        # y_{n+1} = y_n + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
        r = [
            r[i] + dt / 6 * (k1_r[i] + 2 * k2_r[i] + 2 * k3_r[i] + k4_r[i])
            for i in range(3)
        ]
        v = [
            v[i] + dt / 6 * (k1_v[i] + 2 * k2_v[i] + 2 * k3_v[i] + k4_v[i])
            for i in range(3)
        ]

        t += dt

        # Create new state
        new_epoch = epoch.replace(microsecond=0) + type(epoch - epoch)(seconds=int(t))

        states.append(
            StateVector(
                position_m=r.copy(),
                velocity_ms=v.copy(),
                epoch_utc=new_epoch.isoformat(),
                frame=initial_state.frame,
            )
        )

    return states


# ===========================================================================
# Ground Track Computation
# ===========================================================================


def calculate_ground_track(
    orbit_states: list[StateVector], time_step_s: float = 60.0
) -> list[GroundTrack]:
    """Calculate the sub-satellite ground track from orbit state vectors.

    Converts ECI position to geodetic latitude/longitude by accounting
    for Earth's rotation.  Longitude is adjusted by subtracting
    ``OMEGA_EARTH * elapsed_time`` to approximate the ECI-to-ECEF
    transformation (ignoring precession/nutation for simplicity).

    Args:
        orbit_states: List of state vectors in the J2000 frame.
        time_step_s: Time step between consecutive states (seconds).

    Returns:
        List of ground track points with lat/lon/alt.
    """
    ground_track = []

    for i, state in enumerate(orbit_states):
        r_vec = state.position_m

        # Convert to latitude/longitude
        r = vector_magnitude(r_vec)
        lat = math.asin(r_vec[2] / r)

        # Account for Earth rotation
        try:
            datetime.fromisoformat(state.epoch_utc.replace("Z", "+00:00"))
            t_since_epoch = i * time_step_s
            lon = math.atan2(r_vec[1], r_vec[0]) - OMEGA_EARTH * t_since_epoch
        except Exception:
            lon = math.atan2(r_vec[1], r_vec[0])

        # Normalize longitude to [-180, 180] degrees
        while lon > math.pi:
            lon -= 2 * math.pi
        while lon < -math.pi:
            lon += 2 * math.pi

        altitude = r - R_EARTH

        ground_track.append(
            GroundTrack(
                latitude_deg=rad_to_deg(lat),
                longitude_deg=rad_to_deg(lon),
                altitude_m=altitude,
                time_utc=state.epoch_utc,
            )
        )

    return ground_track


# ===========================================================================
# Orbital Maneuver Calculations
# ===========================================================================


def hohmann_transfer(r1_m: float, r2_m: float) -> dict[str, float]:
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

    Returns:
        Dictionary with delta-V values, transfer time, and transfer
        semi-major axis.
    """
    # Transfer orbit semi-major axis (average of initial and final radii)
    a_transfer = (r1_m + r2_m) / 2

    # Circular orbit velocities: v_circ = sqrt(mu / r)
    v1_circular = math.sqrt(MU_EARTH / r1_m)
    v2_circular = math.sqrt(MU_EARTH / r2_m)

    # Transfer orbit velocities at periapsis and apoapsis (vis-viva equation)
    # v = sqrt(mu * (2/r - 1/a))
    v1_transfer = math.sqrt(MU_EARTH * (2 / r1_m - 1 / a_transfer))
    v2_transfer = math.sqrt(MU_EARTH * (2 / r2_m - 1 / a_transfer))

    # Delta-V at each impulse point (signed: positive = prograde burn)
    dv1 = v1_transfer - v1_circular  # Burn at departure orbit
    dv2 = v2_circular - v2_transfer  # Burn at arrival orbit
    dv_total = abs(dv1) + abs(dv2)  # Total delta-V magnitude

    # Transfer time = half the period of the transfer ellipse
    # T_transfer = pi * sqrt(a_t^3 / mu)
    transfer_time = math.pi * math.sqrt(a_transfer**3 / MU_EARTH)

    return {
        "delta_v1_ms": dv1,
        "delta_v2_ms": dv2,
        "total_delta_v_ms": dv_total,
        "transfer_time_s": transfer_time,
        "transfer_time_h": transfer_time / 3600,
        "transfer_semi_major_axis_m": a_transfer,
    }


# ===========================================================================
# Lambert Problem Solver (Simplified)
# ===========================================================================


def lambert_solver_simple(
    r1_vec: list[float], r2_vec: list[float], time_flight_s: float, mu: float = MU_EARTH
) -> dict[str, Any]:
    """Simplified Lambert problem solver for two-body trajectory design.

    The Lambert problem finds the orbit that connects two position
    vectors in a given time of flight.  This implementation uses a
    simplified approach rather than a full universal-variable or Gauss
    iterative method.

    Algorithm outline:
        1. Compute chord length c = |r2 - r1| and semi-perimeter
           s = (r1 + r2 + c) / 2.
        2. Check feasibility against minimum-energy transfer time
           t_min = pi * sqrt(a_min^3 / mu), where a_min = s/2.
        3. Estimate the transfer semi-major axis from Kepler's 3rd law.
        4. Approximate departure/arrival velocities using tangential
           (circular orbit) assumptions.

    Note:
        This is a first-order approximation.  A production Lambert solver
        (e.g., Izzo's or Gooding's algorithm) should be used for mission
        design.

    Args:
        r1_vec: Initial (departure) position vector in meters.
        r2_vec: Final (arrival) position vector in meters.
        time_flight_s: Time of flight between the two positions (seconds).
        mu: Gravitational parameter of the central body (m^3/s^2).

    Returns:
        Dictionary containing departure/arrival velocity vectors and
        transfer geometry.  Includes a ``"feasible"`` flag.
    """
    r1 = vector_magnitude(r1_vec)
    r2 = vector_magnitude(r2_vec)

    # Chord length between the two position vectors
    c = vector_magnitude([r2_vec[i] - r1_vec[i] for i in range(3)])

    # Semi-perimeter of the triangle formed by the focus and two positions
    s = (r1 + r2 + c) / 2

    # Minimum-energy transfer ellipse: a_min = s / 2
    a_min = s / 2

    # Minimum transfer time (parabolic / minimum-energy limit):
    # t_min = pi * sqrt(a_min^3 / mu)
    t_min = math.pi * math.sqrt(a_min**3 / mu)

    if time_flight_s < t_min:
        return {
            "feasible": False,
            "reason": f"Transfer time {time_flight_s:.1f}s less than minimum {t_min:.1f}s",
            "v1_ms": [0, 0, 0],
            "v2_ms": [0, 0, 0],
        }

    # Transfer angle from dot product: cos(dnu) = (r1 . r2) / (|r1| |r2|)
    cos_dnu = vector_dot(r1_vec, r2_vec) / (r1 * r2)
    cos_dnu = max(-1, min(1, cos_dnu))  # Clamp for numerical safety
    dnu = math.acos(cos_dnu)

    # Approximate semi-major axis from Kepler's 3rd law:
    # n = 2*pi / T  =>  a = (mu / n^2)^(1/3)
    n_approx = 2 * math.pi / time_flight_s  # Approximate mean motion
    a_approx = (mu / n_approx**2) ** (1 / 3)

    # Approximate velocity magnitudes (circular orbit assumption: v = sqrt(mu/r))
    v1_mag = math.sqrt(mu / r1)
    v2_mag = math.sqrt(mu / r2)

    # Unit vectors along each radius
    r1_unit = vector_normalize(r1_vec)
    r2_unit = vector_normalize(r2_vec)

    # Orbital plane normal from h = r1 x r2
    h_vec = vector_cross(r1_vec, r2_vec)
    h_unit = vector_normalize(h_vec)

    # Tangential velocity directions (perpendicular to radius in-plane)
    v1_dir = vector_normalize(vector_cross(h_unit, r1_unit))
    v2_dir = vector_normalize(vector_cross(h_unit, r2_unit))

    v1_vec = [v1_mag * v1_dir[i] for i in range(3)]
    v2_vec = [v2_mag * v2_dir[i] for i in range(3)]

    return {
        "feasible": True,
        "v1_ms": v1_vec,
        "v2_ms": v2_vec,
        "v1_vec_ms": v1_vec,  # Add expected key
        "initial_velocity_ms": v1_vec,  # Add expected key
        "transfer_angle_deg": rad_to_deg(dnu),
        "estimated_semi_major_axis_m": a_approx,
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
    This implementation uses simplified circular planetary orbits;
    production applications should use JPL SPICE ephemerides.

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

    def get_planet_position(body: str, date_iso: str) -> list[float]:
        """Get simplified planet position at given date."""
        # Parse date (simplified)
        import datetime

        try:
            dt = datetime.datetime.fromisoformat(date_iso.replace("Z", "+00:00"))
        except Exception:
            dt = datetime.datetime.fromisoformat(date_iso)

        # Days since J2000
        j2000 = datetime.datetime(2000, 1, 1, 12, 0, 0)
        days_since_j2000 = (dt - j2000).total_seconds() / 86400

        if body.lower() == "earth":
            # Earth mean anomaly
            mean_anomaly = 2 * math.pi * days_since_j2000 / earth_period_days
            x = earth_orbit_radius * math.cos(mean_anomaly)
            y = earth_orbit_radius * math.sin(mean_anomaly)
            return [x * 1.496e11, y * 1.496e11, 0.0]  # Convert AU to meters
        elif body.lower() == "mars":
            # Mars mean anomaly
            mean_anomaly = 2 * math.pi * days_since_j2000 / mars_period_days
            x = mars_orbit_radius * math.cos(mean_anomaly)
            y = mars_orbit_radius * math.sin(mean_anomaly)
            return [x * 1.496e11, y * 1.496e11, 0.0]  # Convert AU to meters
        else:
            # Default to Earth position
            mean_anomaly = 2 * math.pi * days_since_j2000 / earth_period_days
            x = earth_orbit_radius * math.cos(mean_anomaly)
            y = earth_orbit_radius * math.sin(mean_anomaly)
            return [x * 1.496e11, y * 1.496e11, 0.0]

    # Generate porkchop data grid
    porkchop_data = []

    for dep_date in departure_dates:
        for arr_date in arrival_dates:
            # Calculate time of flight
            import datetime

            try:
                dep_dt = datetime.datetime.fromisoformat(
                    dep_date.replace("Z", "+00:00")
                )
            except Exception:
                dep_dt = datetime.datetime.fromisoformat(dep_date)

            try:
                arr_dt = datetime.datetime.fromisoformat(
                    arr_date.replace("Z", "+00:00")
                )
            except Exception:
                arr_dt = datetime.datetime.fromisoformat(arr_date)

            tof_days = (arr_dt - dep_dt).total_seconds() / 86400

            # Filter by TOF constraints
            if tof_days < min_tof_days or tof_days > max_tof_days:
                continue

            # Get planetary positions
            r1 = get_planet_position(departure_body, dep_date)
            r2 = get_planet_position(arrival_body, arr_date)

            # Calculate transfer using simplified Lambert solver
            try:
                mu_sun = 1.32712440018e20  # Sun's gravitational parameter (m³/s²)
                tof_seconds = tof_days * 86400

                # Use simplified Lambert solver
                lambert_result = lambert_solver_simple(r1, r2, tof_seconds, mu_sun)

                # Calculate characteristic energy (C3)
                if "v1_vec_ms" in lambert_result:
                    v1 = lambert_result["v1_vec_ms"]
                elif "initial_velocity_ms" in lambert_result:
                    v1 = lambert_result["initial_velocity_ms"]
                else:
                    v1 = [0, 0, 0]  # Fallback

                # Earth's orbital velocity at 1 AU
                earth_v_orbit = math.sqrt(mu_sun / (1.496e11))  # m/s

                # Excess velocity magnitude
                v_inf_vec = [
                    v1[i] - earth_v_orbit if i == 1 else v1[i] for i in range(3)
                ]
                v_inf_mag = math.sqrt(sum(v**2 for v in v_inf_vec))

                # Characteristic energy C3 (km²/s²)
                c3 = (v_inf_mag / 1000) ** 2  # Convert to km/s then square

                # Delta-V estimate (simplified)
                delta_v_ms = v_inf_mag

                porkchop_data.append(
                    {
                        "departure_date": dep_date,
                        "arrival_date": arr_date,
                        "time_of_flight_days": tof_days,
                        "c3_km2_s2": c3,
                        "delta_v_ms": delta_v_ms,
                        "departure_position_m": r1,
                        "arrival_position_m": r2,
                        "transfer_feasible": c3 < 100.0,  # Reasonable C3 limit
                    }
                )

            except Exception as e:
                # Add failed transfer case
                porkchop_data.append(
                    {
                        "departure_date": dep_date,
                        "arrival_date": arr_date,
                        "time_of_flight_days": tof_days,
                        "c3_km2_s2": float("inf"),
                        "delta_v_ms": float("inf"),
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
