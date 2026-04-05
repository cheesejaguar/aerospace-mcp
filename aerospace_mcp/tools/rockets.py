"""Rocket trajectory and performance analysis tools for the Aerospace MCP server.

Provides tools for rocket trajectory simulation (3DOF with atmospheric drag),
preliminary sizing estimation using the Tsiolkovsky rocket equation, and
launch angle optimization. The 3DOF model integrates equations of motion
using a 4th-order Runge-Kutta (RK4) method with altitude-dependent atmospheric
density for drag modeling.

WARNING: This module is for educational and research purposes only.
Do NOT use for real flight planning, navigation, or aircraft operations.
"""

import json
import logging
from typing import Literal

logger = logging.getLogger(__name__)


def rocket_3dof_trajectory(
    rocket_geometry: dict,
    launch_conditions: dict,
    simulation_options: dict | None = None,
) -> str:
    """Calculate 3DOF rocket trajectory using numerical integration.

    Args:
        rocket_geometry: Rocket geometry parameters
        launch_conditions: Launch conditions (launch_angle_deg, launch_site, etc.)
        simulation_options: Optional simulation settings

    Returns:
        Formatted string with trajectory analysis results including max altitude,
        max velocity, Mach number, apogee time, burnout time, max-Q, total
        impulse, and specific impulse.

    Raises:
        No exceptions are raised directly; errors are returned as formatted strings.
        ImportError is caught when rocketry packages are not installed.

    Note:
        The 3DOF equations of motion integrate:
            dv/dt = (T - D) / m - g * sin(gamma)     (along velocity)
            dgamma/dt = -(g / v) * cos(gamma)         (flight path angle)
            dx/dt = v * cos(gamma)                     (downrange)
            dh/dt = v * sin(gamma)                     (altitude)
        where T is thrust, D = 0.5 * rho(h) * v^2 * Cd * A_ref is aerodynamic
        drag with altitude-dependent density, m is instantaneous mass (decreasing
        during burn), and gamma is the flight path angle. Integration uses a
        4th-order Runge-Kutta (RK4) scheme for numerical stability.
    """
    try:
        from ..integrations.rockets import (
            RocketGeometry,
        )
        from ..integrations.rockets import (
            rocket_3dof_trajectory as _trajectory_analysis,
        )

        # Create geometry object
        geometry = RocketGeometry(**rocket_geometry)

        # Run trajectory analysis
        result = _trajectory_analysis(geometry, launch_conditions, simulation_options)

        # Format response
        result_lines = [
            "3DOF Rocket Trajectory Analysis",
            "=" * 50,
            f"Rocket: {geometry.length_m:.1f}m long, {geometry.diameter_m:.2f}m dia",
            f"Mass: {geometry.dry_mass_kg:.1f} kg dry + {geometry.propellant_mass_kg:.1f} kg prop",
            f"Launch Angle: {launch_conditions.get('launch_angle_deg', 90):.1f}°",
            "",
            "Performance Summary:",
            f"  Max Altitude: {result.performance.max_altitude_m:.0f} m ({result.performance.max_altitude_m / 1000:.2f} km)",
            f"  Max Velocity: {result.performance.max_velocity_ms:.1f} m/s (Mach {result.performance.max_mach:.2f})",
            f"  Apogee Time: {result.performance.apogee_time_s:.1f} s",
            f"  Burnout Time: {result.performance.burnout_time_s:.1f} s",
            f"  Max Q: {result.performance.max_q_pa / 1000:.1f} kPa",
            f"  Total Impulse: {result.performance.total_impulse_ns:.0f} N⋅s",
            f"  Specific Impulse: {result.performance.specific_impulse_s:.1f} s",
        ]

        # Add trajectory points summary
        if hasattr(result, "trajectory") and result.trajectory:
            result_lines.extend(
                [
                    "",
                    "Key Trajectory Points:",
                    f"{'Time (s)':>8} {'Alt (m)':>8} {'Vel (m/s)':>10} {'Accel (m/s²)':>12}",
                ]
            )
            result_lines.append("-" * 45)

            # Sample key points
            trajectory = result.trajectory
            sample_indices = [
                0,
                len(trajectory) // 4,
                len(trajectory) // 2,
                3 * len(trajectory) // 4,
                -1,
            ]

            for i in sample_indices:
                if i < len(trajectory):
                    point = trajectory[i]
                    result_lines.append(
                        f"{point.time_s:8.1f} {point.altitude_m:8.0f} {point.velocity_ms:10.1f} "
                        f"{point.acceleration_ms2:12.1f}"
                    )

        # Add JSON data
        json_data = json.dumps(
            {
                "performance": {
                    "max_altitude_m": result.performance.max_altitude_m,
                    "max_velocity_ms": result.performance.max_velocity_ms,
                    "apogee_time_s": result.performance.apogee_time_s,
                    "burnout_time_s": result.performance.burnout_time_s,
                    "max_q_pa": result.performance.max_q_pa,
                    "total_impulse_ns": result.performance.total_impulse_ns,
                    "specific_impulse_s": result.performance.specific_impulse_s,
                }
            },
            indent=2,
        )
        result_lines.extend(["", "JSON Performance Data:", json_data])

        return "\n".join(result_lines)

    except ImportError:
        return "Rocket trajectory analysis not available - install rocketry packages"
    except Exception as e:
        logger.error(f"Rocket trajectory error: {str(e)}", exc_info=True)
        return f"Rocket trajectory error: {str(e)}"


def estimate_rocket_sizing(
    target_altitude_m: float,
    payload_mass_kg: float,
    propellant_type: Literal["solid", "liquid"] = "solid",
    design_margin: float = 1.2,
) -> str:
    """Estimate rocket sizing requirements for target altitude and payload.

    Args:
        target_altitude_m: Target altitude in meters
        payload_mass_kg: Payload mass in kg
        propellant_type: Propellant type ('solid' or 'liquid')
        design_margin: Design margin factor

    Returns:
        JSON string with sizing estimates including propellant mass, dry mass,
        total mass, and structural dimensions.

    Raises:
        No exceptions are raised directly; errors are returned as formatted strings.

    Note:
        Sizing uses the Tsiolkovsky rocket equation (ideal rocket equation):
            delta_V = Isp * g0 * ln(m_initial / m_final)
        Rearranged to solve for propellant mass:
            m_prop = m_final * (exp(delta_V / (Isp * g0)) - 1)
        where Isp is specific impulse and g0 = 9.80665 m/s^2.
    """
    try:
        from ..integrations.rockets import estimate_rocket_sizing as _sizing

        result = _sizing(
            target_altitude_m, payload_mass_kg, propellant_type, design_margin
        )

        return json.dumps(result, indent=2)

    except ImportError:
        return "Rocket sizing not available - install rocketry packages"
    except Exception as e:
        logger.error(f"Rocket sizing error: {str(e)}", exc_info=True)
        return f"Rocket sizing error: {str(e)}"


def optimize_launch_angle(
    rocket_geometry: dict,
    target_range_m: float | None = None,
    optimize_for: Literal["altitude", "range"] = "altitude",
    angle_bounds_deg: tuple[float, float] = (45.0, 90.0),
) -> str:
    """Optimize rocket launch angle for maximum altitude or range.

    Args:
        rocket_geometry: Rocket geometry parameters
        target_range_m: Optional target range in meters
        optimize_for: Optimization objective ('altitude' or 'range')
        angle_bounds_deg: Launch angle bounds in degrees

    Returns:
        JSON string with optimization results including optimal angle and
        resulting performance metrics.

    Raises:
        No exceptions are raised directly; errors are returned as formatted strings.
    """
    try:
        from ..integrations.rockets import (
            RocketGeometry,
        )
        from ..integrations.rockets import (
            optimize_launch_angle as _optimize,
        )

        geometry = RocketGeometry(**rocket_geometry)

        result = _optimize(geometry, target_range_m, optimize_for, angle_bounds_deg)

        return json.dumps(result, indent=2)

    except ImportError:
        return "Launch optimization not available - install optimization packages"
    except Exception as e:
        logger.error(f"Launch optimization error: {str(e)}", exc_info=True)
        return f"Launch optimization error: {str(e)}"
