"""Propeller and UAV energy analysis tools for the Aerospace MCP server.

Provides tools for propeller performance analysis using Blade Element Momentum
Theory (BEMT) and UAV mission energy estimation. BEMT combines blade element
theory (local 2D airfoil analysis along the blade span) with momentum theory
(global flow through the rotor disk) to predict thrust, torque, and efficiency.

WARNING: This module is for educational and research purposes only.
Do NOT use for real flight planning, navigation, or aircraft operations.
"""

import json
import logging

logger = logging.getLogger(__name__)


def propeller_bemt_analysis(
    propeller_geometry: dict,
    operating_conditions: dict,
    analysis_options: dict | None = None,
) -> str:
    """Analyze propeller performance using Blade Element Momentum Theory.

    Args:
        propeller_geometry: Propeller geometry (diameter_m, pitch_m, num_blades,
            and optionally hub_radius_m, activity_factor, cl_design, cd_design)
        operating_conditions: Operating conditions (rpm_list with each rpm > 0,
            velocity_ms >= 0, altitude_m)
        analysis_options: Optional analysis settings: n_elements (number of
            radial annuli, default 40)

    Returns:
        Formatted string with propeller performance analysis including thrust,
        torque, power, efficiency, and advance ratio at each RPM. Efficiency is
        J*CT/CP (0 for static thrust); negative thrust/power means windmilling.

    Raises:
        No exceptions are raised directly; errors are returned as formatted strings.
        ImportError is caught when propulsion packages are not installed.

    Note:
        BEMT iteratively solves for the inflow angle (phi) at each blade
        element by balancing:
        1. **Blade Element Theory**: Local lift and drag from 2D airfoil data
           at the effective angle of attack (alpha = phi - pitch_angle).
        2. **Momentum Theory**: Axial and tangential momentum changes through
           an annular ring of the rotor disk.
        Convergence is achieved when the induced velocity factors (a, a')
        satisfy both theories simultaneously. The advance ratio J = V / (n*D)
        characterizes the operating condition. The solver brackets the inflow
        angle and bisects (Ning 2014), so every annulus converges, and applies
        Prandtl tip and hub losses. The blade has constant geometric pitch and
        a typical small-propeller planform scaled to the activity factor.
    """
    try:
        from ..integrations.propellers import (
            PropellerGeometry,
        )
        from ..integrations.propellers import (
            propeller_bemt_analysis as _propeller_analysis,
        )

        # Create geometry object from the input dictionary
        geometry = PropellerGeometry(**propeller_geometry)

        rpm_list = operating_conditions.get("rpm_list", [2000, 2500, 3000])
        velocity_ms = operating_conditions.get("velocity_ms", 20.0)
        altitude_m = operating_conditions.get("altitude_m", 0.0)

        # Run analysis (n_elements only when requested, to keep the call
        # compatible with the integration's default)
        options = analysis_options or {}
        extra = {}
        if options.get("n_elements") is not None:
            extra["n_elements"] = int(options["n_elements"])
        results = _propeller_analysis(
            geometry, rpm_list, velocity_ms, altitude_m, **extra
        )

        # Format response
        result_lines = [
            "Propeller BEMT Analysis",
            "=" * 60,
            f"Propeller: {geometry.diameter_m:.3f}m dia, {geometry.pitch_m:.3f}m pitch, {geometry.num_blades} blades",
            f"Conditions: {velocity_ms:.1f} m/s @ {altitude_m:.0f}m altitude",
            "",
            f"{'RPM':>6} {'Thrust (N)':>10} {'Power (W)':>9} {'Efficiency':>10} {'Adv Ratio':>10}",
        ]
        result_lines.append("-" * 60)

        for result in results:
            result_lines.append(
                f"{result.rpm:6.0f} {result.thrust_n:10.2f} {result.power_w:9.1f} "
                f"{result.efficiency:10.3f} {result.advance_ratio:10.3f}"
            )

        # Add JSON data
        json_data = json.dumps(
            [
                {
                    "rpm": r.rpm,
                    "thrust_n": r.thrust_n,
                    "torque_nm": r.torque_nm,
                    "power_w": r.power_w,
                    "efficiency": r.efficiency,
                    "advance_ratio": r.advance_ratio,
                    "thrust_coefficient": r.thrust_coefficient,
                    "power_coefficient": r.power_coefficient,
                }
                for r in results
            ],
            indent=2,
        )
        result_lines.extend(["", "JSON Data:", json_data])

        return "\n".join(result_lines)

    except ImportError:
        return "Propeller analysis not available - install propulsion packages"
    except Exception as e:
        logger.error(f"Propeller analysis error: {str(e)}", exc_info=True)
        return f"Propeller analysis error: {str(e)}"


def uav_energy_estimate(
    uav_config: dict, battery_config: dict, mission_profile: dict | None = None
) -> str:
    """Estimate UAV flight time and energy consumption for mission planning.

    Args:
        uav_config: UAV configuration parameters (mass_kg; wing_area_m2 for a
            fixed-wing, disk_area_m2 for a multirotor, both for a VTOL; cd0,
            cl_cruise, num_motors, motor_efficiency, esc_efficiency, and
            optionally drag_area_m2, avionics_power_w). An optional
            ``propeller_geometry`` dict (diameter_m, pitch_m, num_blades, ...)
            makes the BEMT solver supply the propeller efficiency / rotor
            figure of merit instead of the defaults (0.8 / 0.7).
        battery_config: Battery configuration parameters
        mission_profile: Optional mission profile parameters (velocity_ms,
            altitude_m)

    Returns:
        Formatted string with energy analysis results including flight time,
        range, hover time, power required, and efficiency metrics. Flight time
        is at the cruise speed; hover time (rotorcraft) is in hover.

    Raises:
        No exceptions are raised directly; errors are returned as formatted strings.
    """
    try:
        from ..integrations.propellers import (
            BatteryConfiguration,
            UAVConfiguration,
        )
        from ..integrations.propellers import (
            uav_energy_estimate as _energy_estimate,
        )

        # Create configuration objects
        uav_config = dict(uav_config)
        propeller_config = uav_config.pop("propeller_geometry", None)
        uav = UAVConfiguration(**uav_config)
        battery = BatteryConfiguration(**battery_config)
        mission = mission_profile or {}

        # Run analysis
        if propeller_config:
            from ..integrations.propellers import PropellerGeometry

            result = _energy_estimate(
                uav, battery, mission, PropellerGeometry(**propeller_config)
            )
        else:
            result = _energy_estimate(uav, battery, mission)

        # Aircraft type as determined by the integration (older integrations
        # do not report it)
        type_labels = {
            "fixed_wing": "Fixed-Wing",
            "multirotor": "Multirotor",
            "vtol": "VTOL (Fixed-Wing + Rotors)",
            "generic": "Generic",
        }
        reported_type = getattr(result, "aircraft_type", None)
        aircraft_type = type_labels.get(
            reported_type, "Fixed-Wing" if uav.wing_area_m2 else "Multirotor"
        )

        # Format response
        result_lines = [
            f"UAV Energy Analysis ({aircraft_type})",
            "=" * 45,
            f"Aircraft Mass: {uav.mass_kg:.1f} kg",
            f"Battery: {battery.capacity_ah:.1f} Ah @ {battery.voltage_nominal_v:.1f}V ({battery.mass_kg:.2f} kg)",
        ]

        if uav.wing_area_m2:
            result_lines.extend(
                [
                    f"Wing Area: {uav.wing_area_m2:.2f} m²",
                    f"Wing Loading: {uav.mass_kg * 9.81 / uav.wing_area_m2:.1f} N/m²",
                ]
            )
        if uav.disk_area_m2:
            result_lines.extend(
                [
                    f"Rotor Disk Area: {uav.disk_area_m2:.2f} m²",
                    f"Disk Loading: {uav.mass_kg * 9.81 / uav.disk_area_m2:.1f} N/m²",
                ]
            )

        cruise_velocity = getattr(result, "cruise_velocity_ms", None)
        electrical_power = getattr(result, "electrical_power_w", None)
        result_lines.extend(["", "Energy Analysis:"])
        result_lines.extend(
            [
                f"  Battery Energy: {result.battery_energy_wh:.0f} Wh",
                f"  Usable Energy: {result.energy_consumed_wh:.0f} Wh",
                f"  Power Required: {result.power_required_w:.0f} W (shaft + avionics)",
            ]
        )
        if electrical_power is not None:
            result_lines.append(f"  Battery Power: {electrical_power:.0f} W")
        result_lines.extend(
            [
                f"  System Efficiency: {result.efficiency_overall:.1%}",
                "",
                "Mission Performance:",
            ]
        )
        if cruise_velocity is not None:
            result_lines.append(f"  Cruise Speed: {cruise_velocity:.1f} m/s")
        result_lines.append(
            f"  Flight Time: {result.flight_time_min:.1f} minutes ({result.flight_time_min / 60:.1f} hours)"
        )

        if result.range_km:
            result_lines.append(f"  Range: {result.range_km:.1f} km")

        if result.hover_time_min:
            result_lines.append(f"  Hover Time: {result.hover_time_min:.1f} minutes")

        warnings = getattr(result, "warnings", None) or []
        if warnings:
            result_lines.extend(["", "Warnings:"])
            result_lines.extend(f"  ⚠ {w}" for w in warnings)

        # Add recommendations
        result_lines.extend(["", "Recommendations:"])

        if result.flight_time_min < 10:
            result_lines.append(
                "  ⚠ Very short flight time - consider larger battery or lighter aircraft"
            )
        elif result.flight_time_min < 20:
            result_lines.append(
                "  ⚠ Short flight time - optimize for efficiency or add battery capacity"
            )
        elif result.flight_time_min > 120:
            result_lines.append(
                "  ✓ Excellent endurance - well optimized configuration"
            )
        else:
            result_lines.append("  ✓ Good flight time for mission requirements")

        if result.efficiency_overall < 0.7:
            result_lines.append("  ⚠ Low system efficiency - check motor/ESC selection")
        else:
            result_lines.append("  ✓ Good system efficiency")

        # Add JSON data
        json_output = {
            "flight_time_min": result.flight_time_min,
            "range_km": result.range_km,
            "hover_time_min": result.hover_time_min,
            "power_required_w": result.power_required_w,
            "energy_consumed_wh": result.energy_consumed_wh,
            "battery_energy_wh": result.battery_energy_wh,
            "efficiency_overall": result.efficiency_overall,
        }
        for key in (
            "aircraft_type",
            "cruise_velocity_ms",
            "electrical_power_w",
            "hover_power_w",
            "propeller_efficiency",
            "figure_of_merit",
            "warnings",
        ):
            if hasattr(result, key):
                json_output[key] = getattr(result, key)
        json_data = json.dumps(json_output, indent=2)
        result_lines.extend(["", "JSON Data:", json_data])

        return "\n".join(result_lines)

    except ImportError:
        return "UAV energy analysis not available - install propulsion packages"
    except Exception as e:
        logger.error(f"UAV energy analysis error: {str(e)}", exc_info=True)
        return f"UAV energy analysis error: {str(e)}"


def get_propeller_database() -> str:
    """Get available propeller database with geometric and performance data.

    Returns:
        JSON string with propeller database containing geometry and performance data.

    Raises:
        No exceptions are raised directly; errors are returned as formatted strings.
    """
    try:
        from ..integrations.propellers import PROPELLER_DATABASE

        return json.dumps(PROPELLER_DATABASE, indent=2)

    except ImportError:
        return "Propeller database not available"
    except Exception as e:
        logger.error(f"Propeller database error: {str(e)}", exc_info=True)
        return f"Propeller database error: {str(e)}"
