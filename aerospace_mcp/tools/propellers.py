"""Propeller and UAV energy analysis tools for the Aerospace MCP server."""

import json
import logging

logger = logging.getLogger(__name__)


def propeller_bemt_analysis(
    propeller_geometry: dict,
    operating_conditions: dict,
    analysis_options: dict | None = None
) -> str:
    """Analyze propeller performance using Blade Element Momentum Theory.

    Args:
        propeller_geometry: Propeller geometry (diameter_m, pitch_m, num_blades, etc.)
        operating_conditions: Operating conditions (rpm_list, velocity_ms, altitude_m)
        analysis_options: Optional analysis settings

    Returns:
        Formatted string with propeller performance analysis
    """
    try:
        from ..integrations.propellers import (
            PropellerGeometry,
        )
        from ..integrations.propellers import (
            propeller_bemt_analysis as _propeller_analysis,
        )

        # Create geometry object
        geometry = PropellerGeometry(**propeller_geometry)

        rpm_list = operating_conditions.get("rpm_list", [2000, 2500, 3000])
        velocity_ms = operating_conditions.get("velocity_ms", 20.0)
        altitude_m = operating_conditions.get("altitude_m", 0.0)

        # Run analysis
        results = _propeller_analysis(geometry, rpm_list, velocity_ms, altitude_m)

        # Format response
        result_lines = [
            "Propeller BEMT Analysis",
            "=" * 60,
            f"Propeller: {geometry.diameter_m:.2f}m dia, {geometry.pitch_m:.2f}m pitch, {geometry.num_blades} blades",
            f"Conditions: {velocity_ms:.1f} m/s @ {altitude_m:.0f}m altitude",
            "",
            f"{'RPM':>6} {'Thrust (N)':>10} {'Power (W)':>9} {'Efficiency':>10} {'Adv Ratio':>10}"
        ]
        result_lines.append("-" * 60)

        for result in results:
            result_lines.append(
                f"{result.rpm:6.0f} {result.thrust_n:10.1f} {result.power_w:9.0f} "
                f"{result.efficiency:10.3f} {result.advance_ratio:10.3f}"
            )

        # Add JSON data
        json_data = json.dumps([
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
        ], indent=2)
        result_lines.extend(["", "JSON Data:", json_data])

        return "\n".join(result_lines)

    except ImportError:
        return "Propeller analysis not available - install propulsion packages"
    except Exception as e:
        logger.error(f"Propeller analysis error: {str(e)}", exc_info=True)
        return f"Propeller analysis error: {str(e)}"


def uav_energy_estimate(
    uav_config: dict,
    battery_config: dict,
    mission_profile: dict | None = None
) -> str:
    """Estimate UAV flight time and energy consumption for mission planning.

    Args:
        uav_config: UAV configuration parameters
        battery_config: Battery configuration parameters
        mission_profile: Optional mission profile parameters

    Returns:
        Formatted string with energy analysis results
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
        uav = UAVConfiguration(**uav_config)
        battery = BatteryConfiguration(**battery_config)
        mission = mission_profile or {}

        # Run analysis
        result = _energy_estimate(uav, battery, mission)

        # Determine aircraft type
        aircraft_type = "Fixed-Wing" if uav.wing_area_m2 else "Multirotor"

        # Format response
        result_lines = [
            f"UAV Energy Analysis ({aircraft_type})",
            "=" * 45,
            f"Aircraft Mass: {uav.mass_kg:.1f} kg",
            f"Battery: {battery.capacity_ah:.1f} Ah @ {battery.voltage_nominal_v:.1f}V ({battery.mass_kg:.2f} kg)",
        ]

        if uav.wing_area_m2:
            result_lines.extend([
                f"Wing Area: {uav.wing_area_m2:.2f} m²",
                f"Wing Loading: {uav.mass_kg * 9.81 / uav.wing_area_m2:.1f} N/m²",
            ])
        elif uav.disk_area_m2:
            result_lines.extend([
                f"Rotor Disk Area: {uav.disk_area_m2:.2f} m²",
                f"Disk Loading: {uav.mass_kg * 9.81 / uav.disk_area_m2:.1f} N/m²",
            ])

        result_lines.extend([
            "",
            "Energy Analysis:",
            f"  Battery Energy: {result.battery_energy_wh:.0f} Wh",
            f"  Usable Energy: {result.energy_consumed_wh:.0f} Wh",
            f"  Power Required: {result.power_required_w:.0f} W",
            f"  System Efficiency: {result.efficiency_overall:.1%}",
            "",
            "Mission Performance:",
            f"  Flight Time: {result.flight_time_min:.1f} minutes ({result.flight_time_min / 60:.1f} hours)",
        ])

        if result.range_km:
            result_lines.append(f"  Range: {result.range_km:.1f} km")

        if result.hover_time_min:
            result_lines.append(f"  Hover Time: {result.hover_time_min:.1f} minutes")

        # Add recommendations
        result_lines.extend(["", "Recommendations:"])

        if result.flight_time_min < 10:
            result_lines.append("  ⚠ Very short flight time - consider larger battery or lighter aircraft")
        elif result.flight_time_min < 20:
            result_lines.append("  ⚠ Short flight time - optimize for efficiency or add battery capacity")
        elif result.flight_time_min > 120:
            result_lines.append("  ✓ Excellent endurance - well optimized configuration")
        else:
            result_lines.append("  ✓ Good flight time for mission requirements")

        if result.efficiency_overall < 0.7:
            result_lines.append("  ⚠ Low system efficiency - check motor/ESC selection")
        else:
            result_lines.append("  ✓ Good system efficiency")

        # Add JSON data
        json_data = json.dumps({
            "flight_time_min": result.flight_time_min,
            "range_km": result.range_km,
            "hover_time_min": result.hover_time_min,
            "power_required_w": result.power_required_w,
            "energy_consumed_wh": result.energy_consumed_wh,
            "battery_energy_wh": result.battery_energy_wh,
            "efficiency_overall": result.efficiency_overall,
        }, indent=2)
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
        JSON string with propeller database
    """
    try:
        from ..integrations.propellers import PROPELLER_DATABASE

        return json.dumps(PROPELLER_DATABASE, indent=2)

    except ImportError:
        return "Propeller database not available"
    except Exception as e:
        logger.error(f"Propeller database error: {str(e)}", exc_info=True)
        return f"Propeller database error: {str(e)}"
