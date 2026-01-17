"""Orbital mechanics tools for the Aerospace MCP server."""

import json
import logging
import math
from typing import Literal

logger = logging.getLogger(__name__)

# Constants
MU_EARTH = 3.986004418e14  # m³/s² - Earth's gravitational parameter
MU_SUN = 1.32712440018e20  # m³/s² - Sun's gravitational parameter

# Gravitational parameters for different central bodies
MU_BODIES = {
    "earth": MU_EARTH,
    "sun": MU_SUN,
    "moon": 4.9048695e12,
    "mars": 4.282837e13,
    "venus": 3.24859e14,
    "jupiter": 1.26686534e17,
}


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
        JSON string with transfer orbit velocities and orbital elements
    """
    try:
        mu = MU_BODIES.get(central_body.lower(), MU_EARTH)

        # Try to use poliastro if available for accurate solution
        try:
            import numpy as np
            from astropy import units as u
            from poliastro.iod import izzo

            r1_arr = np.array(r1_m)
            r2_arr = np.array(r2_m)

            # Solve Lambert's problem using Izzo's algorithm
            v1_solutions, v2_solutions = izzo.lambert(
                k=mu * u.m**3 / u.s**2,
                r0=r1_arr * u.m,
                r=r2_arr * u.m,
                tof=tof_s * u.s,
                M=num_revolutions,
                prograde=direction == "prograde",
            )

            # Get first solution - handle both single solution and array of solutions
            # poliastro may return a single Quantity or a tuple of Quantities
            if hasattr(v1_solutions, "__len__") and not isinstance(
                v1_solutions.value, int | float
            ):
                # It's an array-like with multiple elements
                if len(v1_solutions.shape) > 1:
                    # Multiple solutions, take first
                    v1 = v1_solutions[0].to(u.m / u.s).value.tolist()
                    v2 = v2_solutions[0].to(u.m / u.s).value.tolist()
                else:
                    # Single solution as 1D array
                    v1 = v1_solutions.to(u.m / u.s).value.tolist()
                    v2 = v2_solutions.to(u.m / u.s).value.tolist()
            else:
                # Single scalar (shouldn't happen for velocity vectors)
                v1 = v1_solutions.to(u.m / u.s).value.tolist()
                v2 = v2_solutions.to(u.m / u.s).value.tolist()

            # Ensure v1 and v2 are lists
            if not isinstance(v1, list):
                v1 = [float(v1)] if isinstance(v1, int | float) else list(v1)
            if not isinstance(v2, list):
                v2 = [float(v2)] if isinstance(v2, int | float) else list(v2)

            implementation = "poliastro (Izzo algorithm)"

        except ImportError:
            # Fallback to manual implementation
            from ..integrations.orbits import lambert_solver_simple

            result = lambert_solver_simple(r1_m, r2_m, tof_s, mu)

            if not result.get("feasible", True):
                return json.dumps(
                    {
                        "success": False,
                        "error": result.get("reason", "Transfer not feasible"),
                        "v1_ms": [0, 0, 0],
                        "v2_ms": [0, 0, 0],
                    },
                    indent=2,
                )

            v1 = result.get("v1_ms", [0, 0, 0])
            v2 = result.get("v2_ms", [0, 0, 0])
            implementation = "manual (simplified)"

        # Calculate delta-V magnitudes
        def vec_mag(v):
            return math.sqrt(sum(x**2 for x in v))

        r1_mag = vec_mag(r1_m)
        r2_mag = vec_mag(r2_m)

        # Calculate orbital elements of transfer orbit
        # Specific angular momentum
        h = [
            r1_m[1] * v1[2] - r1_m[2] * v1[1],
            r1_m[2] * v1[0] - r1_m[0] * v1[2],
            r1_m[0] * v1[1] - r1_m[1] * v1[0],
        ]
        h_mag = vec_mag(h)

        # Semi-latus rectum
        p = h_mag**2 / mu

        # Eccentricity vector
        v1_mag = vec_mag(v1)
        e_vec = [
            (v1_mag**2 / mu - 1 / r1_mag) * r1_m[i]
            - sum(r1_m[j] * v1[j] for j in range(3)) / mu * v1[i]
            for i in range(3)
        ]
        e = vec_mag(e_vec)

        # Semi-major axis
        if abs(e - 1.0) > 1e-6:
            a = p / (1 - e**2)
        else:
            a = float("inf")  # Parabolic

        # Inclination
        i_rad = math.acos(max(-1, min(1, h[2] / h_mag))) if h_mag > 0 else 0
        i_deg = math.degrees(i_rad)

        # Transfer angle
        cos_dnu = sum(r1_m[i] * r2_m[i] for i in range(3)) / (r1_mag * r2_mag)
        cos_dnu = max(-1, min(1, cos_dnu))
        transfer_angle_deg = math.degrees(math.acos(cos_dnu))

        # Determine if short or long way
        cross = [
            r1_m[1] * r2_m[2] - r1_m[2] * r2_m[1],
            r1_m[2] * r2_m[0] - r1_m[0] * r2_m[2],
            r1_m[0] * r2_m[1] - r1_m[1] * r2_m[0],
        ]
        if cross[2] < 0:
            transfer_angle_deg = 360 - transfer_angle_deg

        # Orbital period (if elliptical)
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
        JSON string with state vector components
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
        JSON string with orbital elements
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
                "arg_perigee_deg": result.arg_perigee_deg,
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
        initial_state: Initial orbital state (elements or state vector)
        propagation_time_s: Propagation time in seconds
        time_step_s: Integration time step in seconds

    Returns:
        JSON string with propagated state
    """
    try:
        from ..integrations.orbits import propagate_orbit_j2 as _propagate

        result = _propagate(initial_state, propagation_time_s, time_step_s)

        return json.dumps(result, indent=2)

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
        orbital_state: Orbital state (elements or state vector)
        duration_s: Duration for ground track calculation in seconds
        time_step_s: Time step for ground track points in seconds

    Returns:
        JSON string with ground track coordinates
    """
    try:
        from ..integrations.orbits import calculate_ground_track as _ground_track

        result = _ground_track(orbital_state, duration_s, time_step_s)

        return json.dumps(result, indent=2)

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
        JSON string with transfer orbit parameters
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
        rendezvous_options: Optional rendezvous planning parameters

    Returns:
        JSON string with rendezvous plan
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

        result = _rendezvous(chaser, target, rendezvous_options or {})

        return json.dumps(result, indent=2)

    except ImportError:
        return "Rendezvous planning not available - install orbital packages"
    except Exception as e:
        logger.error(f"Rendezvous planning error: {str(e)}", exc_info=True)
        return f"Rendezvous planning error: {str(e)}"
