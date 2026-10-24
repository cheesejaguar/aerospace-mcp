"""Rocket trajectory and performance analysis tools for the Aerospace MCP server.

Provides tools for rocket trajectory simulation (3DOF with atmospheric drag),
preliminary sizing estimation using the Tsiolkovsky rocket equation, and
launch angle optimization. The trajectory model integrates the equations of
motion with forward Euler steps and altitude-dependent atmospheric density;
the simulation and optimizer run in the native Rust core
(``aerospace_mcp._native``).

WARNING: This module is for educational and research purposes only.
Do NOT use for real flight planning, navigation, or aircraft operations.
"""

import json
import logging
from dataclasses import asdict
from typing import Literal

logger = logging.getLogger(__name__)


def rocket_3dof_trajectory(
    rocket_geometry: dict,
    launch_conditions: dict,
    simulation_options: dict | None = None,
) -> str:
    """Calculate 3DOF rocket trajectory using numerical integration.

    Args:
        rocket_geometry: Rocket geometry parameters (dry_mass_kg,
            propellant_mass_kg, diameter_m, length_m, optional cd and
            thrust_curve as [[time_s, thrust_n], ...])
        launch_conditions: Launch conditions (launch_angle_deg, default 90)
        simulation_options: Optional simulation settings (dt_s, default 0.1;
            max_time_s, default 300)

    Returns:
        Formatted string with trajectory analysis results including max altitude,
        max velocity, Mach number, apogee time, burnout time, max-Q, total
        impulse, and specific impulse.

    Raises:
        No exceptions are raised directly; errors are returned as formatted strings.
        ImportError is caught when rocketry packages are not installed.

    Note:
        The 3DOF model integrates vertical and horizontal motion over a flat
        Earth with forward Euler steps:
            a_v = (T_v - D_v) / m - g,   a_h = (T_h - D_h) / m
        where T is thrust (along the launch angle for the first 5 s, then
        vertical), D = 0.5 * rho(h) * v^2 * Cd * A_ref is aerodynamic drag
        opposing the velocity with altitude-dependent density, and m is the
        instantaneous mass (mass flow proportional to thrust, so the
        propellant runs out when the thrust curve ends). The run ends at
        apogee, ground impact, or max_time_s.
    """
    try:
        from ..integrations.rockets import (
            RocketGeometry,
            RocketTrajectoryPoint,
            simulate_rocket,
        )

        # Create geometry object
        geometry = RocketGeometry(**rocket_geometry)
        options = simulation_options or {}

        # Performance and the few sample points printed below come straight
        # from the native trajectory, without a Python object per step.
        performance, trajectory = simulate_rocket(
            geometry,
            dt_s=options.get("dt_s", 0.1),
            max_time_s=options.get("max_time_s", 300.0),
            launch_angle_deg=launch_conditions.get("launch_angle_deg", 90.0),
        )

        # Format response
        result_lines = [
            "3DOF Rocket Trajectory Analysis",
            "=" * 50,
            f"Rocket: {geometry.length_m:.1f}m long, {geometry.diameter_m:.2f}m dia",
            f"Mass: {geometry.dry_mass_kg:.1f} kg dry + {geometry.propellant_mass_kg:.1f} kg prop",
            f"Launch Angle: {launch_conditions.get('launch_angle_deg', 90):.1f}°",
            "",
            "Performance Summary:",
            f"  Max Altitude: {performance.max_altitude_m:.0f} m ({performance.max_altitude_m / 1000:.2f} km)",
            f"  Max Velocity: {performance.max_velocity_ms:.1f} m/s (Mach {performance.max_mach:.2f})",
            f"  Apogee Time: {performance.apogee_time_s:.1f} s",
            f"  Burnout Time: {performance.burnout_time_s:.1f} s",
            f"  Max Q: {performance.max_q_pa / 1000:.1f} kPa",
            f"  Total Impulse: {performance.total_impulse_ns:.0f} N⋅s",
            f"  Specific Impulse: {performance.specific_impulse_s:.1f} s",
        ]

        # Add trajectory points summary
        if len(trajectory):
            result_lines.extend(
                [
                    "",
                    "Key Trajectory Points:",
                    f"{'Time (s)':>8} {'Alt (m)':>8} {'Vel (m/s)':>10} {'Accel (m/s²)':>12}",
                ]
            )
            result_lines.append("-" * 45)

            # Sample key points
            sample_indices = [
                0,
                len(trajectory) // 4,
                len(trajectory) // 2,
                3 * len(trajectory) // 4,
                -1,
            ]

            for i in sample_indices:
                if i < len(trajectory):
                    point = RocketTrajectoryPoint(*trajectory[i])
                    result_lines.append(
                        f"{point.time_s:8.1f} {point.altitude_m:8.0f} {point.velocity_ms:10.1f} "
                        f"{point.acceleration_ms2:12.1f}"
                    )

        # Add JSON data
        json_data = json.dumps(
            {
                "performance": {
                    "max_altitude_m": performance.max_altitude_m,
                    "max_velocity_ms": performance.max_velocity_ms,
                    "apogee_time_s": performance.apogee_time_s,
                    "burnout_time_s": performance.burnout_time_s,
                    "max_q_pa": performance.max_q_pa,
                    "total_impulse_ns": performance.total_impulse_ns,
                    "specific_impulse_s": performance.specific_impulse_s,
                }
            },
            indent=2,
            allow_nan=False,
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
        design_margin: Multiplier (>= 1) on the delta-V requirement; the
            default 1.2 sizes for 20% more delta-V than the loss-corrected
            estimate

    Returns:
        JSON string with sizing estimates including propellant mass, dry mass,
        total mass, structural dimensions, the delta-V budget with and
        without margin, and a feasibility flag (an infeasible single stage
        reports null masses and the reason).

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
            target_altitude_m,
            payload_mass_kg,
            propellant_type,
            design_margin=design_margin,
        )

        return json.dumps(result, indent=2, allow_nan=False)

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
        target_range_m: Optional target range in meters (not used yet:
            downrange distance is not modelled)
        optimize_for: Optimization objective ('altitude' or 'range')
        angle_bounds_deg: Launch angle bounds in degrees

    Returns:
        JSON string with optimization results including optimal angle and
        resulting performance metrics.

    Raises:
        No exceptions are raised directly; errors are returned as formatted strings.
    """
    try:
        from ..integrations.trajopt import (
            RocketGeometry,
        )
        from ..integrations.trajopt import (
            optimize_launch_angle as _optimize,
        )

        geometry = RocketGeometry(**rocket_geometry)
        objective = "max_range" if optimize_for == "range" else "max_altitude"

        result = _optimize(
            geometry, objective=objective, angle_bounds=tuple(angle_bounds_deg)
        )

        output = {
            "optimize_for": optimize_for,
            "optimal_launch_angle_deg": result.optimal_parameters["launch_angle_deg"],
            "optimal_objective": result.optimal_objective,
            "angle_bounds_deg": list(angle_bounds_deg),
            "performance": asdict(result.performance),
        }
        if optimize_for == "range":
            output["note"] = (
                "Downrange distance is not modelled by the 3DOF simulation; "
                "range optimization cannot distinguish launch angles."
            )

        return json.dumps(output, indent=2, allow_nan=False)

    except ImportError:
        return "Launch optimization not available - install optimization packages"
    except Exception as e:
        logger.error(f"Launch optimization error: {str(e)}", exc_info=True)
        return f"Launch optimization error: {str(e)}"
