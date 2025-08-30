"""Orbital mechanics tools for the Aerospace MCP server."""

import json
import logging

logger = logging.getLogger(__name__)


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

        return json.dumps({
            "position_m": {
                "x": result.position_m[0],
                "y": result.position_m[1],
                "z": result.position_m[2]
            },
            "velocity_ms": {
                "x": result.velocity_ms[0],
                "y": result.velocity_ms[1],
                "z": result.velocity_ms[2]
            },
            "reference_frame": "J2000",
            "units": {
                "position": "meters",
                "velocity": "m/s"
            }
        }, indent=2)

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

        return json.dumps({
            "semi_major_axis_m": result.semi_major_axis_m,
            "eccentricity": result.eccentricity,
            "inclination_deg": result.inclination_deg,
            "raan_deg": result.raan_deg,
            "arg_perigee_deg": result.arg_perigee_deg,
            "true_anomaly_deg": result.true_anomaly_deg,
            "epoch_utc": result.epoch_utc,
        }, indent=2)

    except ImportError:
        return "Orbital mechanics not available - install orbital packages"
    except Exception as e:
        logger.error(f"State vector to elements error: {str(e)}", exc_info=True)
        return f"State vector to elements error: {str(e)}"


def propagate_orbit_j2(
    initial_state: dict,
    propagation_time_s: float,
    time_step_s: float = 60.0
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
    orbital_state: dict,
    duration_s: float,
    time_step_s: float = 60.0
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
    chaser_elements: dict,
    target_elements: dict,
    rendezvous_options: dict | None = None
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
