"""Orbital mechanics tools for the Aerospace MCP server.

Provides tools for classical orbital mechanics computations including:
- Lambert's problem solver for transfer orbit determination.
- Conversion between orbital elements and Cartesian state vectors.
- Orbit propagation with J2 perturbation effects.
- Ground track calculation for satellite visibility.
- Hohmann transfer orbit delta-V computation.
- Orbital rendezvous maneuver planning.

Key equations used:
- Vis-viva equation: v^2 = mu * (2/r - 1/a)
- Kepler's equation: M = E - e*sin(E)  (relates mean and eccentric anomaly)
- Hohmann delta-V: computed from vis-viva at departure and arrival radii

WARNING: This module is for educational and research purposes only.
Do NOT use for real flight planning, navigation, or aircraft operations.
"""

import json
import logging
import math
from typing import Literal

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Gravitational Parameters (mu = G * M)
# ---------------------------------------------------------------------------
# Standard gravitational parameter mu [m^3/s^2] for each celestial body.
# These values determine orbital velocities via the vis-viva equation:
#   v^2 = mu * (2/r - 1/a)
# Source: IAU / JPL planetary ephemerides.

MU_EARTH = 3.986004418e14  # m^3/s^2 -- Earth's gravitational parameter (GM_E)
MU_SUN = 1.32712440018e20  # m^3/s^2 -- Sun's gravitational parameter (GM_Sun)

MU_BODIES = {
    "earth": MU_EARTH,
    "sun": MU_SUN,
    "moon": 4.9048695e12,  # m^3/s^2 -- Moon
    "mars": 4.282837e13,  # m^3/s^2 -- Mars
    "venus": 3.24859e14,  # m^3/s^2 -- Venus
    "jupiter": 1.26686534e17,  # m^3/s^2 -- Jupiter
}


def _vector(value) -> list[float]:
    """Accept ``[x, y, z]`` or ``{"x": ..., "y": ..., "z": ...}``."""
    if isinstance(value, dict):
        return [float(value["x"]), float(value["y"]), float(value["z"])]
    return [float(v) for v in value]


def _initial_state(orbital_state: dict):
    """Build a StateVector from a state-vector dict or an orbital-elements dict.

    State vectors use ``position_m`` / ``velocity_ms`` (lists or x/y/z dicts,
    as returned by ``elements_to_state_vector``); anything else is treated as
    classical elements (``semi_major_axis_m``, ``eccentricity``, ...).
    """
    from ..integrations.orbits import (
        OrbitElements,
        StateVector,
        elements_to_state_vector,
    )

    epoch = orbital_state.get("epoch_utc", "2000-01-01T12:00:00")
    if "position_m" in orbital_state:
        return StateVector(
            position_m=_vector(orbital_state["position_m"]),
            velocity_ms=_vector(orbital_state["velocity_ms"]),
            epoch_utc=epoch,
            frame=orbital_state.get("frame", "J2000"),
        )
    elements = {k: v for k, v in orbital_state.items() if k != "epoch_utc"}
    return elements_to_state_vector(OrbitElements(**elements, epoch_utc=epoch))


def lambert_problem_solver(
    r1_m: list[float],
    r2_m: list[float],
    tof_s: float,
    direction: Literal["prograde", "retrograde"] = "prograde",
    num_revolutions: int = 0,
    central_body: str = "earth",
) -> str:
    """Solve Lambert's orbital boundary value problem.

    Given two position vectors and time-of-flight, determine the orbit
    connecting them. This is foundational for interplanetary mission design
    and rendezvous trajectory planning.

    Args:
        r1_m: Initial position vector [x, y, z] in meters
        r2_m: Final position vector [x, y, z] in meters
        tof_s: Time of flight in seconds
        direction: Transfer direction - "prograde" or "retrograde"
        num_revolutions: Number of complete revolutions (default 0 for short path)
        central_body: Central body name for gravitational parameter

    Returns:
        JSON string with transfer orbit velocities, orbital elements,
        transfer angle, and timing information.

    Raises:
        No exceptions are raised directly; errors are returned as JSON strings.
    """
    try:
        mu = MU_BODIES.get(central_body.lower(), MU_EARTH)

        from ..integrations.orbits import solve_lambert

        try:
            solution = solve_lambert(
                r1_m,
                r2_m,
                tof_s,
                mu,
                prograde=direction == "prograde",
                revolutions=num_revolutions,
            )
        except (ValueError, RuntimeError) as e:
            return json.dumps(
                {
                    "success": False,
                    "error": str(e),
                    "v1_ms": [0, 0, 0],
                    "v2_ms": [0, 0, 0],
                },
                indent=2,
            )

        v1 = solution["v1_ms"]
        v2 = solution["v2_ms"]
        implementation = "native Izzo (2015) solver"

        # Calculate delta-V magnitudes
        def vec_mag(v):
            return math.sqrt(sum(x**2 for x in v))

        r1_mag = vec_mag(r1_m)
        r2_mag = vec_mag(r2_m)

        # Calculate classical orbital elements of the transfer orbit from
        # the initial position and velocity vectors.

        # Specific angular momentum vector: h = r x v
        h = [
            r1_m[1] * v1[2] - r1_m[2] * v1[1],
            r1_m[2] * v1[0] - r1_m[0] * v1[2],
            r1_m[0] * v1[1] - r1_m[1] * v1[0],
        ]
        h_mag = vec_mag(h)

        # Semi-latus rectum: p = h^2 / mu  (orbit shape parameter)
        p = h_mag**2 / mu

        # Eccentricity vector: e = (v x h)/mu - r_hat  (points toward periapsis)
        v1_mag = vec_mag(v1)
        e_vec = [
            (v1_mag**2 / mu - 1 / r1_mag) * r1_m[i]
            - sum(r1_m[j] * v1[j] for j in range(3)) / mu * v1[i]
            for i in range(3)
        ]
        e = vec_mag(e_vec)

        # Semi-major axis from semi-latus rectum: a = p / (1 - e^2)
        # For parabolic orbits (e = 1), semi-major axis is infinite.
        if abs(e - 1.0) > 1e-6:
            a = p / (1 - e**2)
        else:
            a = float("inf")  # Parabolic

        # Inclination: angle between orbit plane and equatorial plane
        # i = acos(h_z / |h|)
        i_rad = math.acos(max(-1, min(1, h[2] / h_mag))) if h_mag > 0 else 0
        i_deg = math.degrees(i_rad)

        # Transfer angle in the direction of motion (0-360 deg)
        transfer_angle_deg = solution["transfer_angle_deg"]

        # Orbital period from Kepler's third law: T = 2*pi * sqrt(a^3 / mu)
        # Only valid for elliptical orbits (e < 1, a > 0).
        if a > 0 and e < 1:
            period_s = 2 * math.pi * math.sqrt(a**3 / mu)
        else:
            period_s = float("inf")

        result = {
            "success": True,
            "v1_ms": [round(v, 6) for v in v1],
            "v2_ms": [round(v, 6) for v in v2],
            "v1_magnitude_ms": round(vec_mag(v1), 3),
            "v2_magnitude_ms": round(vec_mag(v2), 3),
            "transfer_orbit": {
                "semi_major_axis_m": round(a, 0) if a != float("inf") else "parabolic",
                "eccentricity": round(e, 6),
                "inclination_deg": round(i_deg, 3),
                "semi_latus_rectum_m": round(p, 0),
                "period_s": round(period_s, 1) if period_s != float("inf") else "n/a",
            },
            "transfer_parameters": {
                "transfer_angle_deg": round(transfer_angle_deg, 3),
                "time_of_flight_s": tof_s,
                "time_of_flight_hr": round(tof_s / 3600, 2),
                "direction": direction,
                "revolutions": num_revolutions,
            },
            "positions": {
                "r1_magnitude_m": round(r1_mag, 0),
                "r2_magnitude_m": round(r2_mag, 0),
                "r1_altitude_km": round((r1_mag - 6378137) / 1000, 1),
                "r2_altitude_km": round((r2_mag - 6378137) / 1000, 1),
            },
            "central_body": central_body,
            "mu_m3_s2": mu,
            "implementation": implementation,
        }

        output = f"""
LAMBERT PROBLEM SOLUTION
========================
Central Body: {central_body.title()} (μ = {mu:.4e} m³/s²)
Time of Flight: {tof_s:.1f} s ({tof_s / 3600:.2f} hr)
Direction: {direction}

Initial Position: r1 = [{r1_m[0]:.0f}, {r1_m[1]:.0f}, {r1_m[2]:.0f}] m
Final Position:   r2 = [{r2_m[0]:.0f}, {r2_m[1]:.0f}, {r2_m[2]:.0f}] m

Transfer Velocities:
  v1 = [{v1[0]:+.3f}, {v1[1]:+.3f}, {v1[2]:+.3f}] m/s  |v1| = {vec_mag(v1):.3f} m/s
  v2 = [{v2[0]:+.3f}, {v2[1]:+.3f}, {v2[2]:+.3f}] m/s  |v2| = {vec_mag(v2):.3f} m/s

Transfer Orbit:
  Semi-major axis: {a / 1000:,.0f} km
  Eccentricity: {e:.6f}
  Inclination: {i_deg:.3f}°
  Transfer angle: {transfer_angle_deg:.3f}°

Implementation: {implementation}

{json.dumps(result, indent=2)}
"""
        return output.strip()

    except Exception as e:
        logger.error(f"Lambert problem solver error: {str(e)}", exc_info=True)
        return json.dumps(
            {
                "success": False,
                "error": str(e),
            },
            indent=2,
        )


def elements_to_state_vector(orbital_elements: dict) -> str:
    """Convert orbital elements to state vector in J2000 frame.

    Args:
        orbital_elements: Dict with orbital elements (semi_major_axis_m, eccentricity, etc.)

    Returns:
        JSON string with position [x,y,z] in meters and velocity [vx,vy,vz]
        in m/s in the J2000 inertial frame.

    Raises:
        No exceptions are raised directly; errors are returned as formatted strings.
    """
    try:
        from ..integrations.orbits import (
            OrbitElements,
        )
        from ..integrations.orbits import (
            elements_to_state_vector as _elements_to_state,
        )

        elements = OrbitElements(**orbital_elements)
        result = _elements_to_state(elements)

        return json.dumps(
            {
                "position_m": {
                    "x": result.position_m[0],
                    "y": result.position_m[1],
                    "z": result.position_m[2],
                },
                "velocity_ms": {
                    "x": result.velocity_ms[0],
                    "y": result.velocity_ms[1],
                    "z": result.velocity_ms[2],
                },
                "reference_frame": "J2000",
                "units": {"position": "meters", "velocity": "m/s"},
            },
            indent=2,
        )

    except ImportError:
        return "Orbital mechanics not available - install orbital packages"
    except Exception as e:
        logger.error(f"Elements to state vector error: {str(e)}", exc_info=True)
        return f"Elements to state vector error: {str(e)}"


def state_vector_to_elements(state_vector: dict) -> str:
    """Convert state vector to classical orbital elements.

    Args:
        state_vector: Dict with position_m and velocity_ms arrays

    Returns:
        JSON string with classical orbital elements (a, e, i, RAAN, omega, nu).

    Raises:
        No exceptions are raised directly; errors are returned as formatted strings.
    """
    try:
        from ..integrations.orbits import (
            StateVector,
        )
        from ..integrations.orbits import (
            state_vector_to_elements as _state_to_elements,
        )

        state = StateVector(**state_vector)
        result = _state_to_elements(state)

        return json.dumps(
            {
                "semi_major_axis_m": result.semi_major_axis_m,
                "eccentricity": result.eccentricity,
                "inclination_deg": result.inclination_deg,
                "raan_deg": result.raan_deg,
                "arg_periapsis_deg": result.arg_periapsis_deg,
                "true_anomaly_deg": result.true_anomaly_deg,
                "epoch_utc": result.epoch_utc,
            },
            indent=2,
        )

    except ImportError:
        return "Orbital mechanics not available - install orbital packages"
    except Exception as e:
        logger.error(f"State vector to elements error: {str(e)}", exc_info=True)
        return f"State vector to elements error: {str(e)}"


def propagate_orbit_j2(
    initial_state: dict, propagation_time_s: float, time_step_s: float = 60.0
) -> str:
    """Propagate orbit with J2 perturbations using numerical integration.

    Args:
        initial_state: Initial orbital state: a state vector (``position_m``,
            ``velocity_ms``, optional ``epoch_utc``) or classical elements
        propagation_time_s: Propagation time in seconds
        time_step_s: Integration time step in seconds

    Returns:
        JSON string with propagated state vectors at each time step.

    Raises:
        No exceptions are raised directly; errors are returned as formatted strings.
    """
    try:
        from ..integrations.orbits import propagate_orbit_j2 as _propagate

        states = _propagate(
            _initial_state(initial_state), propagation_time_s, time_step_s
        )

        return json.dumps(
            {
                "num_states": len(states),
                "time_step_s": time_step_s,
                "propagation_time_s": propagation_time_s,
                "reference_frame": states[0].frame,
                "states": [
                    {
                        "epoch_utc": s.epoch_utc,
                        "position_m": s.position_m,
                        "velocity_ms": s.velocity_ms,
                    }
                    for s in states
                ],
            },
            indent=2,
        )

    except ImportError:
        return "Orbit propagation not available - install orbital packages"
    except Exception as e:
        logger.error(f"Orbit propagation error: {str(e)}", exc_info=True)
        return f"Orbit propagation error: {str(e)}"


def calculate_ground_track(
    orbital_state: dict, duration_s: float, time_step_s: float = 60.0
) -> str:
    """Calculate ground track from orbital state vectors.

    Args:
        orbital_state: Orbital state: a state vector (``position_m``,
            ``velocity_ms``, optional ``epoch_utc``) or classical elements
        duration_s: Duration for ground track calculation in seconds
        time_step_s: Time step for ground track points in seconds

    Returns:
        JSON string with ground track latitude/longitude coordinates.

    Raises:
        No exceptions are raised directly; errors are returned as formatted strings.
    """
    try:
        from ..integrations.orbits import calculate_ground_track as _ground_track
        from ..integrations.orbits import propagate_orbit_j2 as _propagate

        states = _propagate(_initial_state(orbital_state), duration_s, time_step_s)
        track = _ground_track(states, time_step_s)

        return json.dumps(
            {
                "num_points": len(track),
                "time_step_s": time_step_s,
                "duration_s": duration_s,
                "ground_track": [
                    {
                        "time_utc": p.time_utc,
                        "latitude_deg": p.latitude_deg,
                        "longitude_deg": p.longitude_deg,
                        "altitude_m": p.altitude_m,
                    }
                    for p in track
                ],
            },
            indent=2,
        )

    except ImportError:
        return "Ground track calculation not available - install orbital packages"
    except Exception as e:
        logger.error(f"Ground track error: {str(e)}", exc_info=True)
        return f"Ground track error: {str(e)}"


def hohmann_transfer(r1_m: float, r2_m: float) -> str:
    """Calculate Hohmann transfer orbit parameters between two circular orbits.

    Args:
        r1_m: Initial orbit radius in meters
        r2_m: Final orbit radius in meters

    Returns:
        JSON string with transfer orbit parameters including delta-V for each
        burn, transfer orbit semi-major axis, and time of flight.

    Raises:
        No exceptions are raised directly; errors are returned as formatted strings.

    Note:
        The Hohmann transfer is the minimum-energy two-impulse transfer between
        coplanar circular orbits. Using the vis-viva equation v^2 = mu*(2/r - 1/a):
          a_transfer = (r1 + r2) / 2
          delta_V1 = sqrt(mu/r1) * (sqrt(2*r2/(r1+r2)) - 1)   (departure burn)
          delta_V2 = sqrt(mu/r2) * (1 - sqrt(2*r1/(r1+r2)))    (arrival burn)
    """
    try:
        from ..integrations.orbits import hohmann_transfer as _hohmann

        result = _hohmann(r1_m, r2_m)

        return json.dumps(result, indent=2)

    except ImportError:
        return "Transfer orbit calculation not available - install orbital packages"
    except Exception as e:
        logger.error(f"Hohmann transfer error: {str(e)}", exc_info=True)
        return f"Hohmann transfer error: {str(e)}"


def orbital_rendezvous_planning(
    chaser_elements: dict, target_elements: dict, rendezvous_options: dict | None = None
) -> str:
    """Plan orbital rendezvous maneuvers between two spacecraft.

    Args:
        chaser_elements: Chaser spacecraft orbital elements
        target_elements: Target spacecraft orbital elements
        rendezvous_options: Reserved for future planning parameters
            (currently unused)

    Returns:
        JSON string with rendezvous plan including maneuver sequence and delta-V.

    Raises:
        No exceptions are raised directly; errors are returned as formatted strings.
    """
    try:
        from ..integrations.orbits import (
            OrbitElements,
        )
        from ..integrations.orbits import (
            orbital_rendezvous_planning as _rendezvous,
        )

        chaser = OrbitElements(**chaser_elements)
        target = OrbitElements(**target_elements)

        result = _rendezvous(chaser, target)

        return json.dumps(result, indent=2)

    except ImportError:
        return "Rendezvous planning not available - install orbital packages"
    except Exception as e:
        logger.error(f"Rendezvous planning error: {str(e)}", exc_info=True)
        return f"Rendezvous planning error: {str(e)}"
