"""Aerodynamics analysis tools for the Aerospace MCP server.

Provides tools for subsonic aerodynamic analysis including:
- Vortex Lattice Method (VLM) wing analysis for lift, drag, and moment prediction.
- Airfoil polar generation (CL, CD, CM vs. angle of attack).
- Longitudinal stability derivative calculation.
- Airfoil database lookup.

WARNING: This module is for educational and research purposes only.
Do NOT use for real flight planning, navigation, or aircraft operations.
"""

import json
import logging

logger = logging.getLogger(__name__)


def _append_method_and_warnings(result_lines: list[str], points) -> None:
    """Add the model name and the distinct warnings of ``points`` to the text."""
    methods = sorted({m for p in points if (m := getattr(p, "method", None))})
    if methods:
        result_lines.append(f"  Method: {', '.join(methods)}")
    seen = []
    for p in points:
        for w in getattr(p, "warnings", None) or []:
            if w not in seen:
                seen.append(w)
    if seen:
        result_lines.append("Warnings:")
        result_lines.extend(f"  ⚠ {w}" for w in seen)


def wing_vlm_analysis(
    wing_config: dict, flight_conditions: dict, analysis_options: dict | None = None
) -> str:
    """Analyze wing aerodynamics using Vortex Lattice Method or simplified lifting line theory.

    Args:
        wing_config: Wing configuration with keys:
            - span_m: Wing span in meters
            - chord_root_m: Root chord in meters
            - chord_tip_m: Tip chord in meters (optional, defaults to chord_root_m)
            - sweep_deg: Quarter-chord sweep in degrees (optional, default 0)
            - dihedral_deg: Dihedral angle in degrees (optional, default 0)
            - twist_deg: Tip twist in degrees, negative = washout (optional, default 0)
            - airfoil_root: Root airfoil name (optional, default 'NACA2412';
              database names or any NACA 4-digit, case-insensitive; unknown
              names are an error)
            - airfoil_tip: Tip airfoil name (optional, default matches root)
        flight_conditions: Flight conditions with keys:
            - alpha_deg_list: List of angles of attack to analyze (required)
            - mach: Mach number (optional, default 0.2; must be below 0.9)
            - reynolds: Reynolds number (optional)
            - x_ref_mac: Moment reference as a fraction of the MAC from its
              leading edge (optional, default 0.25 = MAC quarter chord)
        analysis_options: Optional analysis settings (currently unused)

    Returns:
        Formatted string with aerodynamic analysis results including CL, CD,
        CM (about x_ref_mac), and L/D ratio at each angle of attack, the
        method that ran (AeroSandbox VLM or lifting-line/DATCOM estimate)
        and any warnings.

    Raises:
        No exceptions are raised directly; errors are returned as formatted strings.
        ImportError is caught when aerodynamics packages are not installed.

    Note:
        The Vortex Lattice Method (VLM) discretizes the wing planform into
        panels, each modeled with a horseshoe vortex. Each horseshoe vortex
        consists of a bound vortex along the panel quarter-chord line and two
        trailing (semi-infinite) vortices extending downstream. The induced
        velocity at each panel's 3/4-chord control point is computed via the
        Biot-Savart law, and the no-penetration boundary condition is enforced
        to solve for the vortex strengths (circulation distribution).
    """
    try:
        from ..integrations.aero import WingGeometry
        from ..integrations.aero import wing_vlm_analysis as _wing_analysis

        # Build WingGeometry from config dict, applying defaults for optional fields
        geometry = WingGeometry(
            span_m=wing_config.get("span_m", 10.0),
            chord_root_m=wing_config.get(
                "chord_root_m", wing_config.get("chord_m", 2.0)
            ),
            chord_tip_m=wing_config.get(
                "chord_tip_m",
                wing_config.get("chord_root_m", wing_config.get("chord_m", 2.0)),
            ),
            sweep_deg=wing_config.get("sweep_deg", 0.0),
            dihedral_deg=wing_config.get("dihedral_deg", 0.0),
            twist_deg=wing_config.get("twist_deg", 0.0),
            airfoil_root=wing_config.get("airfoil_root", "NACA2412"),
            airfoil_tip=wing_config.get(
                "airfoil_tip", wing_config.get("airfoil_root", "NACA2412")
            ),
        )

        # Extract flight conditions
        alpha_deg_list = flight_conditions.get("alpha_deg_list", [0.0, 2.0, 5.0])
        mach = flight_conditions.get("mach", 0.2)
        reynolds = flight_conditions.get("reynolds", None)
        extra = {}
        if flight_conditions.get("x_ref_mac") is not None:
            extra["x_ref_mac"] = float(flight_conditions["x_ref_mac"])

        # Call integration function with correct signature
        analysis_results = _wing_analysis(
            geometry, alpha_deg_list, mach, reynolds, **extra
        )

        # Format response
        result_lines = ["Wing Aerodynamic Analysis (VLM)", "=" * 50]

        # Wing configuration
        result_lines.extend(
            [
                "Configuration:",
                f"  Span: {geometry.span_m:.1f} m",
                f"  Root Chord: {geometry.chord_root_m:.1f} m",
                f"  Tip Chord: {geometry.chord_tip_m:.1f} m",
                f"  Sweep: {geometry.sweep_deg:.1f}°",
                "",
                "Flight Conditions:",
                f"  Mach: {mach:.2f}",
                f"  Moment reference: {extra.get('x_ref_mac', 0.25):.3f} MAC",
            ]
        )
        _append_method_and_warnings(result_lines, analysis_results)
        result_lines.extend(
            ["", f"{'Alpha (°)':>10} {'CL':>8} {'CD':>8} {'CM':>8} {'L/D':>8}"]
        )
        result_lines.append("-" * 50)

        for point in analysis_results:
            result_lines.append(
                f"{point.alpha_deg:10.1f} {point.CL:8.4f} {point.CD:8.5f} "
                f"{point.CM:8.4f} {point.L_D_ratio:8.1f}"
            )

        # Add JSON data
        json_output = [
            {
                "alpha_deg": p.alpha_deg,
                "CL": p.CL,
                "CD": p.CD,
                "CM": p.CM,
                "L_D_ratio": p.L_D_ratio,
                "span_efficiency": p.span_efficiency,
                "method": getattr(p, "method", None),
                "warnings": getattr(p, "warnings", None) or [],
            }
            for p in analysis_results
        ]
        json_data = json.dumps(json_output, indent=2)
        result_lines.extend(["", "JSON Data:", json_data])

        return "\n".join(result_lines)

    except ImportError:
        return "Wing analysis not available - install aerodynamics packages"
    except Exception as e:
        logger.error(f"Wing analysis error: {str(e)}", exc_info=True)
        return f"Wing analysis error: {str(e)}"


def airfoil_polar_analysis(
    airfoil_name: str,
    reynolds_number: float = 1000000,
    mach_number: float = 0.1,
    alpha_range_deg: list[float] | None = None,
) -> str:
    """Generate airfoil polar data (CL, CD, CM vs alpha) using database or advanced methods.

    Args:
        airfoil_name: Airfoil name (e.g., 'NACA2412', 'NACA0012'; case-insensitive,
            database names or any NACA 4-digit; unknown names are an error)
        reynolds_number: Reynolds number (> 0)
        mach_number: Mach number (< 0.9)
        alpha_range_deg: Optional angle of attack range, defaults to [-10, 20] deg

    Returns:
        Formatted string with airfoil polar data (CL, CD, CM about c/4, L/D vs.
        alpha), the method that ran (NeuralFoil or database model) and any
        warnings. Lift is capped at cl_max; beyond stall the model blends into
        a flat plate (cl ~ 2 sin a cos a, cd ~ 2 sin^2 a).

    Raises:
        No exceptions are raised directly; errors are returned as formatted strings.
    """
    try:
        from ..integrations.aero import airfoil_polar_analysis as _airfoil_analysis

        alpha_range_deg = alpha_range_deg or list(range(-10, 21, 2))

        # Integration function signature: (airfoil_name, alpha_deg_list, reynolds, mach)
        polar_results = _airfoil_analysis(
            airfoil_name, alpha_range_deg, reynolds_number, mach_number
        )

        # Format response
        result_lines = [f"Airfoil Polar Analysis: {airfoil_name}", "=" * 60]
        result_lines.extend(
            [
                f"Reynolds Number: {reynolds_number:.0e}",
                f"Mach Number: {mach_number:.3f}",
            ]
        )
        _append_method_and_warnings(result_lines, polar_results)
        result_lines.extend(
            ["", f"{'Alpha (°)':>8} {'CL':>8} {'CD':>8} {'CM':>8} {'L/D':>8}"]
        )
        result_lines.append("-" * 50)

        # polar_results is a list of AirfoilPoint objects
        for point in polar_results:
            result_lines.append(
                f"{point.alpha_deg:8.1f} {point.cl:8.4f} "
                f"{point.cd:8.5f} {point.cm:8.4f} "
                f"{point.cl_cd_ratio:8.1f}"
            )

        # Add JSON data
        json_output = [
            {
                "alpha_deg": p.alpha_deg,
                "cl": p.cl,
                "cd": p.cd,
                "cm": p.cm,
                "cl_cd_ratio": p.cl_cd_ratio,
                "method": getattr(p, "method", None),
                "warnings": getattr(p, "warnings", None) or [],
            }
            for p in polar_results
        ]
        json_data = json.dumps(json_output, indent=2)
        result_lines.extend(["", "JSON Data:", json_data])

        return "\n".join(result_lines)

    except ImportError:
        return "Airfoil analysis not available - install aerodynamics packages"
    except Exception as e:
        logger.error(f"Airfoil analysis error: {str(e)}", exc_info=True)
        return f"Airfoil analysis error: {str(e)}"


def calculate_stability_derivatives(wing_config: dict, flight_conditions: dict) -> str:
    """Calculate basic longitudinal stability derivatives for a wing.

    Args:
        wing_config: Wing configuration with keys:
            - span_m: Wing span in meters
            - chord_root_m: Root chord in meters
            - chord_tip_m: Tip chord in meters (optional)
            - sweep_deg: Quarter-chord sweep (optional, default 0)
            - dihedral_deg: Dihedral angle (optional, default 0)
            - twist_deg: Tip twist (optional, default 0)
            - airfoil_root: Root airfoil name (optional, default 'NACA2412')
            - airfoil_tip: Tip airfoil name (optional)
        flight_conditions: Flight conditions with keys:
            - alpha_deg: Angle of attack at which the slopes are evaluated
              (optional, default 2.0); near and beyond stall the local slopes
              drop
            - mach: Mach number (optional, default 0.2; must be below 0.9)
            - x_ref_mac: Moment reference (e.g. the CG) as a fraction of the
              MAC from its leading edge (optional, default 0.25)

    Returns:
        JSON string with wing-alone stability derivatives:
        - CL_alpha: Local lift curve slope (dCL/dalpha) [1/rad] at alpha_deg
          (Helmbold-Diederich below stall). Positive for conventional aircraft.
        - CM_alpha: Pitching moment slope (dCM/dalpha) [1/rad] about x_ref_mac --
          must be negative for static longitudinal stability (nose-down
          restoring moment). With the wing aerodynamic centre at 0.25 MAC,
          CM_alpha = CL_alpha * (x_ref_mac - 0.25) in attached flow: zero about
          the default reference, negative for a reference ahead of it. The same
          reference is used for CM by wing_vlm_analysis.
        - CL_alpha_dot, CM_alpha_dot: not computed (null). For a conventional
          aircraft they come mainly from the horizontal tail (downwash lag),
          which this wing-alone model does not include.
        - alpha_deg, CL, stalled, x_ref_mac, method, notes: evaluation point
          and assumptions.

    Raises:
        No exceptions are raised directly; errors are returned as formatted strings.
    """
    try:
        from ..integrations.aero import WingGeometry
        from ..integrations.aero import calculate_stability_derivatives as _stability

        # Build WingGeometry from config dict, applying defaults for optional fields
        geometry = WingGeometry(
            span_m=wing_config.get("span_m", 10.0),
            chord_root_m=wing_config.get(
                "chord_root_m", wing_config.get("chord_m", 2.0)
            ),
            chord_tip_m=wing_config.get(
                "chord_tip_m",
                wing_config.get("chord_root_m", wing_config.get("chord_m", 2.0)),
            ),
            sweep_deg=wing_config.get("sweep_deg", 0.0),
            dihedral_deg=wing_config.get("dihedral_deg", 0.0),
            twist_deg=wing_config.get("twist_deg", 0.0),
            airfoil_root=wing_config.get("airfoil_root", "NACA2412"),
            airfoil_tip=wing_config.get(
                "airfoil_tip", wing_config.get("airfoil_root", "NACA2412")
            ),
        )

        # Extract flight conditions
        alpha_deg = flight_conditions.get("alpha_deg", 2.0)
        mach = flight_conditions.get("mach", 0.2)
        extra = {}
        if flight_conditions.get("x_ref_mac") is not None:
            extra["x_ref_mac"] = float(flight_conditions["x_ref_mac"])

        # Integration function signature: (geometry, alpha_deg, mach[, x_ref_mac])
        result = _stability(geometry, alpha_deg, mach, **extra)

        # Format output
        output = {
            "CL_alpha": result.CL_alpha,
            "CM_alpha": result.CM_alpha,
            "CL_alpha_dot": result.CL_alpha_dot,
            "CM_alpha_dot": result.CM_alpha_dot,
        }
        for key in ("alpha_deg", "CL", "stalled", "x_ref_mac", "method", "notes"):
            if hasattr(result, key):
                output[key] = getattr(result, key)
        return json.dumps(output, indent=2)

    except ImportError:
        return "Stability analysis not available - install aerodynamics packages"
    except Exception as e:
        logger.error(f"Stability analysis error: {str(e)}", exc_info=True)
        return f"Stability analysis error: {str(e)}"


def get_airfoil_database() -> str:
    """Get available airfoil database with aerodynamic coefficients.

    Returns:
        JSON string with airfoil database containing aerodynamic coefficients.

    Raises:
        No exceptions are raised directly; errors are returned as formatted strings.
    """
    try:
        from ..integrations.aero import AIRFOIL_DATABASE

        return json.dumps(AIRFOIL_DATABASE, indent=2)

    except ImportError:
        return "Airfoil database not available"
    except Exception as e:
        logger.error(f"Airfoil database error: {str(e)}", exc_info=True)
        return f"Airfoil database error: {str(e)}"
