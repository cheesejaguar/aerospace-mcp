"""
Orbital mechanics and spacecraft trajectory analysis tools for aerospace MCP.

Provides orbital mechanics calculations with manual implementations and optional
advanced library integration for poliastro/astropy/spiceypy when available.
"""

import math
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

# Constants
MU_EARTH = 3.986004418e14  # m³/s² - Earth's gravitational parameter
R_EARTH = 6378137.0  # m - Earth's equatorial radius (WGS84)
J2_EARTH = 1.08262668e-3  # Earth's J2 perturbation coefficient
OMEGA_EARTH = 7.2921159e-5  # rad/s - Earth's rotation rate


@dataclass
class OrbitElements:
    """Classical orbital elements."""

    semi_major_axis_m: float  # Semi-major axis (m)
    eccentricity: float  # Eccentricity (dimensionless)
    inclination_deg: float  # Inclination (degrees)
    raan_deg: float  # Right ascension of ascending node (degrees)
    arg_periapsis_deg: float  # Argument of periapsis (degrees)
    true_anomaly_deg: float  # True anomaly (degrees)
    epoch_utc: str  # Epoch in UTC ISO format


@dataclass
class StateVector:
    """Position and velocity state vector."""

    position_m: list[float]  # Position vector [x, y, z] in meters
    velocity_ms: list[float]  # Velocity vector [vx, vy, vz] in m/s
    epoch_utc: str  # Epoch in UTC ISO format
    frame: str = "J2000"  # Reference frame


@dataclass
class OrbitProperties:
    """Computed orbital properties."""

    period_s: float  # Orbital period (seconds)
    apoapsis_m: float  # Apoapsis altitude above Earth surface (m)
    periapsis_m: float  # Periapsis altitude above Earth surface (m)
    energy_j_kg: float  # Specific orbital energy (J/kg)
    angular_momentum_m2s: float  # Specific angular momentum magnitude (m²/s)


@dataclass
class GroundTrack:
    """Ground track point."""

    latitude_deg: float
    longitude_deg: float
    altitude_m: float
    time_utc: str


@dataclass
class Maneuver:
    """Orbital maneuver definition."""

    delta_v_ms: list[float]  # Delta-V vector [x, y, z] in m/s
    time_utc: str  # Maneuver execution time
    description: str = ""  # Optional description


def deg_to_rad(deg: float) -> float:
    """Convert degrees to radians."""
    return deg * math.pi / 180.0


def rad_to_deg(rad: float) -> float:
    """Convert radians to degrees."""
    return rad * 180.0 / math.pi


def vector_magnitude(vec: list[float]) -> float:
    """Calculate vector magnitude."""
    return math.sqrt(sum(x**2 for x in vec))


def vector_dot(a: list[float], b: list[float]) -> float:
    """Calculate dot product of two vectors."""
    return sum(a[i] * b[i] for i in range(len(a)))


def vector_cross(a: list[float], b: list[float]) -> list[float]:
    """Calculate cross product of two 3D vectors."""
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def vector_normalize(vec: list[float]) -> list[float]:
    """Normalize a vector."""
    mag = vector_magnitude(vec)
    if mag == 0:
        return [0.0, 0.0, 0.0]
    return [x / mag for x in vec]


def kepler_equation_solver(
    mean_anomaly_rad: float,
    eccentricity: float,
    tolerance: float = 1e-8,
    max_iterations: int = 50,
) -> float:
    """
    Solve Kepler's equation for eccentric anomaly using Newton-Raphson method.

    Args:
        mean_anomaly_rad: Mean anomaly in radians
        eccentricity: Orbital eccentricity
        tolerance: Convergence tolerance
        max_iterations: Maximum iterations

    Returns:
        Eccentric anomaly in radians
    """
    # Initial guess
    E = mean_anomaly_rad if eccentricity < 0.8 else math.pi

    for _ in range(max_iterations):
        f = E - eccentricity * math.sin(E) - mean_anomaly_rad
        f_prime = 1 - eccentricity * math.cos(E)

        if abs(f_prime) < 1e-12:
            break

        E_new = E - f / f_prime

        if abs(E_new - E) < tolerance:
            return E_new

        E = E_new

    return E


def elements_to_state_vector(elements: OrbitElements) -> StateVector:
    """
    Convert orbital elements to state vector using manual calculations.

    Args:
        elements: Orbital elements

    Returns:
        State vector in J2000 frame
    """
    # Convert angles to radians
    i = deg_to_rad(elements.inclination_deg)
    raan = deg_to_rad(elements.raan_deg)
    arg_pe = deg_to_rad(elements.arg_periapsis_deg)
    nu = deg_to_rad(elements.true_anomaly_deg)

    # Semi-major axis and eccentricity
    a = elements.semi_major_axis_m
    e = elements.eccentricity

    # Calculate distance and flight path angle
    r = a * (1 - e**2) / (1 + e * math.cos(nu))

    # Position in perifocal coordinates
    r_peri = [r * math.cos(nu), r * math.sin(nu), 0.0]

    # Velocity in perifocal coordinates
    p = a * (1 - e**2)
    math.sqrt(MU_EARTH * p)

    v_peri = [
        -math.sqrt(MU_EARTH / p) * math.sin(nu),
        math.sqrt(MU_EARTH / p) * (e + math.cos(nu)),
        0.0,
    ]

    # Rotation matrices
    cos_raan, sin_raan = math.cos(raan), math.sin(raan)
    cos_i, sin_i = math.cos(i), math.sin(i)
    cos_arg, sin_arg = math.cos(arg_pe), math.sin(arg_pe)

    # Rotation matrix from perifocal to J2000
    R11 = cos_raan * cos_arg - sin_raan * sin_arg * cos_i
    R12 = -cos_raan * sin_arg - sin_raan * cos_arg * cos_i
    R13 = sin_raan * sin_i

    R21 = sin_raan * cos_arg + cos_raan * sin_arg * cos_i
    R22 = -sin_raan * sin_arg + cos_raan * cos_arg * cos_i
    R23 = -cos_raan * sin_i

    R31 = sin_arg * sin_i
    R32 = cos_arg * sin_i
    R33 = cos_i

    # Transform to J2000 frame
    r_j2000 = [
        R11 * r_peri[0] + R12 * r_peri[1] + R13 * r_peri[2],
        R21 * r_peri[0] + R22 * r_peri[1] + R23 * r_peri[2],
        R31 * r_peri[0] + R32 * r_peri[1] + R33 * r_peri[2],
    ]

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
    """
    Convert state vector to orbital elements using manual calculations.

    Args:
        state: State vector in J2000 frame

    Returns:
        Classical orbital elements
    """
    r_vec = state.position_m
    v_vec = state.velocity_ms

    # Position and velocity magnitudes
    r = vector_magnitude(r_vec)
    v = vector_magnitude(v_vec)

    # Specific angular momentum
    h_vec = vector_cross(r_vec, v_vec)
    h = vector_magnitude(h_vec)

    # Semi-major axis
    energy = v**2 / 2 - MU_EARTH / r
    a = -MU_EARTH / (2 * energy)

    # Eccentricity vector
    v_cross_h = vector_cross(v_vec, h_vec)
    e_vec = [v_cross_h[i] / MU_EARTH - r_vec[i] / r for i in range(3)]
    e = vector_magnitude(e_vec)

    # Inclination
    i = math.acos(h_vec[2] / h)

    # Node vector
    k_vec = [0, 0, 1]
    n_vec = vector_cross(k_vec, h_vec)
    n = vector_magnitude(n_vec)

    # RAAN
    if n > 1e-10:
        raan = math.acos(n_vec[0] / n)
        if n_vec[1] < 0:
            raan = 2 * math.pi - raan
    else:
        raan = 0.0

    # Argument of periapsis
    if n > 1e-10 and e > 1e-10:
        cos_arg_pe = vector_dot(n_vec, e_vec) / (n * e)
        cos_arg_pe = max(-1, min(1, cos_arg_pe))  # Clamp to [-1, 1]
        arg_pe = math.acos(cos_arg_pe)
        if e_vec[2] < 0:
            arg_pe = 2 * math.pi - arg_pe
    else:
        arg_pe = 0.0

    # True anomaly
    if e > 1e-10:
        cos_nu = vector_dot(e_vec, r_vec) / (e * r)
        cos_nu = max(-1, min(1, cos_nu))  # Clamp to [-1, 1]
        nu = math.acos(cos_nu)
        if vector_dot(r_vec, v_vec) < 0:
            nu = 2 * math.pi - nu
    else:
        # For circular orbits, use longitude of ascending node
        if n > 1e-10:
            cos_nu = vector_dot(n_vec, r_vec) / (n * r)
            cos_nu = max(-1, min(1, cos_nu))
            nu = math.acos(cos_nu)
            if r_vec[2] < 0:
                nu = 2 * math.pi - nu
        else:
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


def calculate_orbit_properties(elements: OrbitElements) -> OrbitProperties:
    """
    Calculate orbital properties from elements.

    Args:
        elements: Orbital elements

    Returns:
        Orbital properties
    """
    a = elements.semi_major_axis_m
    e = elements.eccentricity

    # Orbital period
    period = 2 * math.pi * math.sqrt(a**3 / MU_EARTH)

    # Apoapsis and periapsis altitudes
    r_ap = a * (1 + e) - R_EARTH
    r_pe = a * (1 - e) - R_EARTH

    # Specific orbital energy
    energy = -MU_EARTH / (2 * a)

    # Specific angular momentum
    h = math.sqrt(MU_EARTH * a * (1 - e**2))

    return OrbitProperties(
        period_s=period,
        apoapsis_m=r_ap,
        periapsis_m=r_pe,
        energy_j_kg=energy,
        angular_momentum_m2s=h,
    )


def propagate_orbit_j2(
    initial_state: StateVector, time_span_s: float, time_step_s: float = 60.0
) -> list[StateVector]:
    """
    Propagate orbit with J2 perturbations using numerical integration.

    Args:
        initial_state: Initial state vector
        time_span_s: Propagation time span (seconds)
        time_step_s: Integration time step (seconds)

    Returns:
        List of state vectors over time
    """

    def acceleration_j2(r_vec: list[float]) -> list[float]:
        """Calculate acceleration including J2 perturbations."""
        r = vector_magnitude(r_vec)

        # Central body acceleration
        a_central = [-MU_EARTH * r_vec[i] / r**3 for i in range(3)]

        # J2 perturbation
        factor = 1.5 * J2_EARTH * MU_EARTH * R_EARTH**2 / r**5
        z2_r2 = (r_vec[2] / r) ** 2

        a_j2 = [
            factor * r_vec[0] * (1 - 5 * z2_r2),
            factor * r_vec[1] * (1 - 5 * z2_r2),
            factor * r_vec[2] * (3 - 5 * z2_r2),
        ]

        return [a_central[i] + a_j2[i] for i in range(3)]

    # Initialize
    states = [initial_state]
    r = initial_state.position_m.copy()
    v = initial_state.velocity_ms.copy()

    # Parse initial epoch
    try:
        epoch = datetime.fromisoformat(initial_state.epoch_utc.replace("Z", "+00:00"))
    except:
        epoch = datetime.now(UTC)

    # Numerical integration (RK4)
    t = 0.0
    while t < time_span_s:
        dt = min(time_step_s, time_span_s - t)

        # RK4 integration
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

        # Update state
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


def calculate_ground_track(
    orbit_states: list[StateVector], time_step_s: float = 60.0
) -> list[GroundTrack]:
    """
    Calculate ground track from orbit state vectors.

    Args:
        orbit_states: List of state vectors
        time_step_s: Time step between states (seconds)

    Returns:
        List of ground track points
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
        except:
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


def hohmann_transfer(r1_m: float, r2_m: float) -> dict[str, float]:
    """
    Calculate Hohmann transfer orbit parameters.

    Args:
        r1_m: Initial circular orbit radius (m)
        r2_m: Final circular orbit radius (m)

    Returns:
        Transfer parameters including delta-V requirements
    """
    # Transfer orbit semi-major axis
    a_transfer = (r1_m + r2_m) / 2

    # Velocities
    v1_circular = math.sqrt(MU_EARTH / r1_m)
    v2_circular = math.sqrt(MU_EARTH / r2_m)

    v1_transfer = math.sqrt(MU_EARTH * (2 / r1_m - 1 / a_transfer))
    v2_transfer = math.sqrt(MU_EARTH * (2 / r2_m - 1 / a_transfer))

    # Delta-V requirements
    dv1 = abs(v1_transfer - v1_circular)
    dv2 = abs(v2_circular - v2_transfer)
    dv_total = dv1 + dv2

    # Transfer time
    transfer_time = math.pi * math.sqrt(a_transfer**3 / MU_EARTH)

    return {
        "delta_v_1_ms": dv1,
        "delta_v_2_ms": dv2,
        "delta_v_total_ms": dv_total,
        "transfer_time_s": transfer_time,
        "transfer_time_h": transfer_time / 3600,
        "semi_major_axis_m": a_transfer,
    }


def lambert_solver_simple(
    r1_vec: list[float], r2_vec: list[float], time_flight_s: float, mu: float = MU_EARTH
) -> dict[str, Any]:
    """
    Simple Lambert problem solver for two-body trajectory.

    Args:
        r1_vec: Initial position vector (m)
        r2_vec: Final position vector (m)
        time_flight_s: Time of flight (seconds)
        mu: Gravitational parameter (m³/s²)

    Returns:
        Dictionary with initial and final velocity vectors
    """
    r1 = vector_magnitude(r1_vec)
    r2 = vector_magnitude(r2_vec)

    # Chord length
    c = vector_magnitude([r2_vec[i] - r1_vec[i] for i in range(3)])

    # Semi-perimeter
    s = (r1 + r2 + c) / 2

    # Minimum energy ellipse semi-major axis
    a_min = s / 2

    # Check if transfer time is feasible
    t_min = math.pi * math.sqrt(a_min**3 / mu)

    if time_flight_s < t_min:
        return {
            "feasible": False,
            "reason": f"Transfer time {time_flight_s:.1f}s less than minimum {t_min:.1f}s",
            "v1_ms": [0, 0, 0],
            "v2_ms": [0, 0, 0],
        }

    # Simplified solution assuming elliptical transfer
    # This is an approximation - full Lambert solver requires iterative methods

    # Transfer angle (simplified)
    cos_dnu = vector_dot(r1_vec, r2_vec) / (r1 * r2)
    cos_dnu = max(-1, min(1, cos_dnu))  # Clamp
    dnu = math.acos(cos_dnu)

    # Approximate semi-major axis for given flight time
    # Using Kepler's 3rd law and approximation
    n_approx = 2 * math.pi / time_flight_s  # Approximate mean motion
    a_approx = (mu / n_approx**2) ** (1 / 3)

    # Energy and specific angular momentum approximations
    -mu / (2 * a_approx)
    math.sqrt(mu * a_approx * (1 - 0.1**2))  # Assume low eccentricity

    # Approximate velocities (simplified circular approximation)
    v1_mag = math.sqrt(mu / r1)
    v2_mag = math.sqrt(mu / r2)

    # Direction vectors (perpendicular to radius for circular approximation)
    r1_unit = vector_normalize(r1_vec)
    r2_unit = vector_normalize(r2_vec)

    # Simplified velocity directions (tangential)
    h_vec = vector_cross(r1_vec, r2_vec)
    h_unit = vector_normalize(h_vec)

    v1_dir = vector_normalize(vector_cross(h_unit, r1_unit))
    v2_dir = vector_normalize(vector_cross(h_unit, r2_unit))

    v1_vec = [v1_mag * v1_dir[i] for i in range(3)]
    v2_vec = [v2_mag * v2_dir[i] for i in range(3)]

    return {
        "feasible": True,
        "v1_ms": v1_vec,
        "v2_ms": v2_vec,
        "transfer_angle_deg": rad_to_deg(dnu),
        "estimated_semi_major_axis_m": a_approx,
    }


def orbital_rendezvous_planning(
    chaser_elements: OrbitElements, target_elements: OrbitElements
) -> dict[str, Any]:
    """
    Plan orbital rendezvous maneuvers between two spacecraft.

    Args:
        chaser_elements: Chaser spacecraft orbital elements
        target_elements: Target spacecraft orbital elements

    Returns:
        Rendezvous plan with phasing and approach maneuvers
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

    return {
        "relative_distance_m": rel_distance,
        "relative_distance_km": rel_distance / 1000,
        "phase_angle_deg": rad_to_deg(phase_angle_rad),
        "phasing_time_s": phasing_time_s,
        "phasing_time_h": phasing_time_s / 3600
        if phasing_time_s != float("inf")
        else float("inf"),
        "chaser_period_s": chaser_props.period_s,
        "target_period_s": target_props.period_s,
        "period_difference_s": abs(chaser_props.period_s - target_props.period_s),
        "altitude_difference_m": abs(chaser_props.apoapsis_m - target_props.apoapsis_m),
        "estimated_circularization_dv_ms": chaser_circ_dv,
        "feasibility": "Good"
        if rel_distance < 100000
        and abs(chaser_elements.inclination_deg - target_elements.inclination_deg) < 5.0
        else "Challenging",
    }


# Update availability
def porkchop_plot_analysis(
    departure_body: str = "Earth",
    arrival_body: str = "Mars",
    departure_dates: list[str] = None,
    arrival_dates: list[str] = None,
    min_tof_days: int = 100,
    max_tof_days: int = 400,
) -> dict[str, Any]:
    """
    Generate porkchop plot analysis for interplanetary transfers.

    Args:
        departure_body: Departure celestial body (default: Earth)
        arrival_body: Arrival celestial body (default: Mars)
        departure_dates: List of departure dates (ISO format)
        arrival_dates: List of arrival dates (ISO format)
        min_tof_days: Minimum time of flight (days)
        max_tof_days: Maximum time of flight (days)

    Returns:
        Dictionary containing transfer analysis grid
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
        except:
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
            except:
                dep_dt = datetime.datetime.fromisoformat(dep_date)

            try:
                arr_dt = datetime.datetime.fromisoformat(
                    arr_date.replace("Z", "+00:00")
                )
            except:
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


def get_ephemeris_position(
    body: str, epoch_utc: str, frame: str = "J2000"
) -> dict[str, Any]:
    """
    Get ephemeris position of celestial body (stub implementation).

    Args:
        body: Celestial body name (Earth, Mars, Venus, etc.)
        epoch_utc: UTC epoch in ISO format
        frame: Reference frame (default: J2000)

    Returns:
        Position and velocity vectors, with accuracy notes

    Note:
        This is a simplified implementation using circular orbits.
        For production use, install SPICE kernels and use spiceypy or similar.
    """
    # Import required for date parsing
    import datetime

    try:
        dt = datetime.datetime.fromisoformat(epoch_utc.replace("Z", "+00:00"))
    except:
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


# Optional SPICE integration (if available)
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
