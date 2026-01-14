"""
Aircraft Aerodynamics Tools

Provides aircraft aerodynamics analysis including VLM wing analysis,
airfoil polars, and basic aerodynamic calculations. Falls back to
simplified methods when optional dependencies are unavailable.

Uses NumPy for vectorized calculations with CuPy compatibility for GPU acceleration.
"""

from pydantic import BaseModel, Field

from . import update_availability
from ._array_backend import np, to_numpy

# Constants
PI = np.pi

# Optional library imports
AEROSANDBOX_AVAILABLE = False
MACHUPX_AVAILABLE = False

try:
    import aerosandbox as asb

    # import aerosandbox.numpy as anp  # Available if needed

    AEROSANDBOX_AVAILABLE = True
    update_availability("aero", True, {"aerosandbox": asb.__version__})
except ImportError:
    try:
        # import machupx as mx  # Available if needed

        MACHUPX_AVAILABLE = True
        update_availability("aero", True, {"machupx": "unknown"})
    except ImportError:
        update_availability("aero", True, {})  # Still available with manual methods

# Airfoil database - simplified coefficients for common airfoils
AIRFOIL_DATABASE = {
    "NACA0012": {
        "cl_alpha": 6.28,  # per radian
        "cl0": 0.0,  # zero-angle lift coefficient (symmetric)
        "cd0": 0.006,
        "cl_max": 1.4,
        "alpha_stall_deg": 15.0,
        "cm0": 0.0,  # symmetric airfoil
    },
    "NACA2412": {
        "cl_alpha": 6.28,
        "cl0": 0.25,  # zero-angle lift coefficient (cambered)
        "cd0": 0.007,
        "cl_max": 1.6,
        "alpha_stall_deg": 16.0,
        "cm0": -0.05,
    },
    "NACA4412": {
        "cl_alpha": 6.28,
        "cl0": 0.40,  # zero-angle lift coefficient (4% camber)
        "cd0": 0.008,
        "cl_max": 1.7,
        "alpha_stall_deg": 17.0,
        "cm0": -0.08,
    },
    "NACA6412": {
        "cl_alpha": 6.28,
        "cl0": 0.55,  # zero-angle lift coefficient (6% camber)
        "cd0": 0.007,
        "cl_max": 1.8,
        "alpha_stall_deg": 18.0,
        "cm0": -0.12,
    },
    "CLARKY": {
        "cl_alpha": 6.0,
        "cl0": 0.30,  # zero-angle lift coefficient (cambered)
        "cd0": 0.008,
        "cl_max": 1.5,
        "alpha_stall_deg": 14.0,
        "cm0": -0.06,
    },
}


# Data models
class AirfoilPoint(BaseModel):
    """Single airfoil polar point."""

    alpha_deg: float = Field(..., description="Angle of attack in degrees")
    cl: float = Field(..., description="Lift coefficient")
    cd: float = Field(..., description="Drag coefficient")
    cm: float | None = Field(None, description="Moment coefficient")
    cl_cd_ratio: float = Field(..., description="Lift to drag ratio")


class WingGeometry(BaseModel):
    """Wing planform geometry definition."""

    span_m: float = Field(..., gt=0, description="Wing span in meters")
    chord_root_m: float = Field(..., gt=0, description="Root chord in meters")
    chord_tip_m: float = Field(..., gt=0, description="Tip chord in meters")
    sweep_deg: float = Field(
        0.0, ge=-45, le=45, description="Quarter-chord sweep in degrees"
    )
    dihedral_deg: float = Field(
        0.0, ge=-15, le=15, description="Dihedral angle in degrees"
    )
    twist_deg: float = Field(
        0.0, ge=-10, le=10, description="Geometric twist (tip relative to root)"
    )
    airfoil_root: str = Field("NACA2412", description="Root airfoil name")
    airfoil_tip: str = Field("NACA2412", description="Tip airfoil name")


class WingAnalysisPoint(BaseModel):
    """Wing analysis results at a single condition."""

    alpha_deg: float = Field(..., description="Wing angle of attack")
    CL: float = Field(..., description="Wing lift coefficient")
    CD: float = Field(..., description="Wing drag coefficient")
    CM: float = Field(..., description="Wing pitching moment coefficient")
    L_D_ratio: float = Field(..., description="Lift to drag ratio")
    span_efficiency: float | None = Field(None, description="Span efficiency factor")


class StabilityDerivatives(BaseModel):
    """Longitudinal stability derivatives."""

    CL_alpha: float = Field(..., description="Lift curve slope (per radian)")
    CM_alpha: float = Field(..., description="Pitching moment curve slope")
    CL_alpha_dot: float | None = Field(None, description="CL due to alpha rate")
    CM_alpha_dot: float | None = Field(None, description="CM due to alpha rate")


def _simple_wing_analysis(
    geometry: WingGeometry, alpha_deg_list: list[float], mach: float = 0.2
) -> list[WingAnalysisPoint]:
    """
    Simple wing analysis using lifting line theory approximations.
    Uses vectorized NumPy calculations for efficiency.
    """
    # Convert to NumPy array
    alphas_deg = np.asarray(alpha_deg_list, dtype=np.float64)
    alphas_rad = np.radians(alphas_deg)

    # Wing parameters
    S = geometry.span_m * (geometry.chord_root_m + geometry.chord_tip_m) / 2  # Area
    AR = geometry.span_m**2 / S  # Aspect ratio

    # Get airfoil properties
    airfoil_data = AIRFOIL_DATABASE.get(
        geometry.airfoil_root, AIRFOIL_DATABASE["NACA2412"]
    )

    # Prandtl lifting line corrections
    e = 0.85  # Oswald efficiency
    CL_alpha_2d = airfoil_data["cl_alpha"]
    CL_alpha_3d = CL_alpha_2d / (1 + CL_alpha_2d / (PI * AR * e))

    # Mach number corrections
    beta = np.sqrt(max(0.01, 1 - mach**2))
    CL_alpha_3d = CL_alpha_3d / beta

    # Vectorized lift coefficient calculation
    cl0 = airfoil_data.get("cl0", 0.0)
    CL = cl0 + CL_alpha_3d * alphas_rad

    # Vectorized drag coefficient
    CD0 = airfoil_data["cd0"] * 1.1  # Wing CD0 slightly higher than airfoil
    CDi = CL**2 / (PI * AR * e)  # Induced drag
    CD = CD0 + CDi

    # Pitching moment
    CM = airfoil_data["cm0"] + 0.02 * CL

    # Apply stall model (vectorized)
    alpha_stall = airfoil_data["alpha_stall_deg"]
    stalled = np.abs(alphas_deg) > alpha_stall
    stall_factor = np.where(
        stalled,
        np.maximum(0.3, 1.0 - 0.1 * (np.abs(alphas_deg) - alpha_stall)),
        1.0,
    )
    CL = CL * stall_factor

    drag_multiplier = np.where(
        stalled, 1.5 + 0.1 * (np.abs(alphas_deg) - alpha_stall), 1.0
    )
    CD = CD * drag_multiplier

    # L/D ratio
    L_D = np.where(CD > 0.001, CL / CD, 0.0)

    # Convert to output format
    results = []
    alphas_np = to_numpy(alphas_deg)
    CL_np = to_numpy(CL)
    CD_np = to_numpy(CD)
    CM_np = to_numpy(CM)
    L_D_np = to_numpy(L_D)

    for i in range(len(alphas_np)):
        results.append(
            WingAnalysisPoint(
                alpha_deg=float(alphas_np[i]),
                CL=float(CL_np[i]),
                CD=float(CD_np[i]),
                CM=float(CM_np[i]),
                L_D_ratio=float(L_D_np[i]),
                span_efficiency=e,
            )
        )

    return results


def wing_vlm_analysis(
    geometry: WingGeometry,
    alpha_deg_list: list[float],
    mach: float = 0.2,
    reynolds: float | None = None,
) -> list[WingAnalysisPoint]:
    """
    Vortex Lattice Method wing analysis.

    Uses NumPy for vectorized calculations when using fallback methods.

    Args:
        geometry: Wing planform geometry
        alpha_deg_list: List of angles of attack to analyze (degrees)
        mach: Mach number
        reynolds: Reynolds number (optional, used for airfoil data if available)

    Returns:
        List of WingAnalysisPoint objects with CL, CD, CM data
    """
    if AEROSANDBOX_AVAILABLE:
        try:
            # Use AeroSandbox VLM
            return _aerosandbox_wing_analysis(geometry, alpha_deg_list, mach, reynolds)
        except Exception:
            # Fall back to simple method
            pass

    # Use simple lifting line approximation
    return _simple_wing_analysis(geometry, alpha_deg_list, mach)


def _aerosandbox_wing_analysis(
    geometry: WingGeometry,
    alpha_deg_list: list[float],
    mach: float,
    reynolds: float | None,
) -> list[WingAnalysisPoint]:
    """AeroSandbox-based VLM analysis."""
    import math  # AeroSandbox uses Python math, not numpy

    # Create wing geometry
    wing = asb.Wing(
        name="MainWing",
        xsecs=[
            asb.WingXSec(
                xyz_le=[0, 0, 0],
                chord=geometry.chord_root_m,
                airfoil=asb.Airfoil(geometry.airfoil_root),
            ),
            asb.WingXSec(
                xyz_le=[
                    geometry.span_m / 2 * math.tan(math.radians(geometry.sweep_deg)),
                    geometry.span_m / 2,
                    geometry.span_m / 2 * math.tan(math.radians(geometry.dihedral_deg)),
                ],
                chord=geometry.chord_tip_m,
                airfoil=asb.Airfoil(geometry.airfoil_tip),
                twist=math.radians(geometry.twist_deg),
            ),
        ],
    )

    # Create airplane
    airplane = asb.Airplane(name="TestAirplane", wings=[wing])

    results = []

    for alpha_deg in alpha_deg_list:
        # Create operating point
        op_point = asb.OperatingPoint(
            atmosphere=asb.Atmosphere(altitude=0),
            velocity=50.0,  # Arbitrary velocity
            alpha=math.radians(alpha_deg),
        )

        # Run VLM analysis
        vlm = asb.VortexLatticeMethod(
            airplane=airplane,
            op_point=op_point,
            chordwise_panels=10,
            spanwise_panels=20,
        )

        vlm_result = vlm.run()

        # Extract coefficients
        CL = vlm_result["CL"]
        CD = vlm_result.get("CD", 0.01)  # VLM doesn't compute viscous drag
        CM = vlm_result.get("CM", 0.0)

        L_D = CL / CD if CD > 0.001 else 0.0

        results.append(
            WingAnalysisPoint(
                alpha_deg=alpha_deg,
                CL=float(CL),
                CD=float(CD),
                CM=float(CM),
                L_D_ratio=L_D,
            )
        )

    return results


def airfoil_polar_analysis(
    airfoil_name: str,
    alpha_deg_list: list[float],
    reynolds: float = 1e6,
    mach: float = 0.1,
) -> list[AirfoilPoint]:
    """
    Generate airfoil polar data.

    Uses NumPy for vectorized calculations when using fallback methods.

    Args:
        airfoil_name: Airfoil designation (e.g., "NACA2412")
        alpha_deg_list: Angles of attack to analyze
        reynolds: Reynolds number
        mach: Mach number

    Returns:
        List of AirfoilPoint objects with cl, cd, cm data
    """
    if AEROSANDBOX_AVAILABLE:
        try:
            return _aerosandbox_airfoil_polar(
                airfoil_name, alpha_deg_list, reynolds, mach
            )
        except Exception:
            # Fall back to database method
            pass

    # Use simplified database method
    return _database_airfoil_polar(airfoil_name, alpha_deg_list, reynolds, mach)


def _database_airfoil_polar(
    airfoil_name: str, alpha_deg_list: list[float], reynolds: float, mach: float
) -> list[AirfoilPoint]:
    """Generate airfoil polar from database coefficients using vectorized NumPy."""
    # Get airfoil data from database
    airfoil_data = AIRFOIL_DATABASE.get(airfoil_name, AIRFOIL_DATABASE["NACA2412"])

    # Convert to NumPy arrays
    alphas_deg = np.asarray(alpha_deg_list, dtype=np.float64)
    alphas_rad = np.radians(alphas_deg)

    # Reynolds number corrections
    re_factor = min(1.2, (reynolds / 1e6) ** 0.1)

    # Mach number corrections
    beta = np.sqrt(max(0.01, 1 - mach**2))

    # Vectorized lift coefficient calculation
    cl0 = airfoil_data.get("cl0", 0.0)
    cl_alpha = airfoil_data["cl_alpha"]
    cl = (cl0 + cl_alpha * alphas_rad) / beta * re_factor

    # Apply stall model (vectorized)
    alpha_stall = airfoil_data["alpha_stall_deg"]
    stalled = np.abs(alphas_deg) > alpha_stall
    stall_factor = np.where(
        stalled,
        np.maximum(0.2, 1.0 - 0.15 * (np.abs(alphas_deg) - alpha_stall)),
        1.0,
    )
    cl = cl * stall_factor

    # Vectorized drag coefficient calculation
    cd0 = airfoil_data["cd0"]
    cd0 = cd0 * (1e6 / reynolds) ** 0.2  # Reynolds correction

    # Mach corrections for drag
    if mach > 0.3:
        cd0 = cd0 * (1 + 0.2 * (mach - 0.3) ** 2)

    cd = cd0 + 0.01 * cl**2  # Simplified induced drag approximation

    # Moment coefficient
    cm = airfoil_data["cm0"] + 0.01 * cl

    # L/D ratio
    cl_cd = np.where(cd > 0.001, cl / cd, 0.0)

    # Convert to output format
    results = []
    alphas_np = to_numpy(alphas_deg)
    cl_np = to_numpy(cl)
    cd_np = to_numpy(cd)
    cm_np = to_numpy(cm)
    cl_cd_np = to_numpy(cl_cd)

    for i in range(len(alphas_np)):
        results.append(
            AirfoilPoint(
                alpha_deg=float(alphas_np[i]),
                cl=float(cl_np[i]),
                cd=float(cd_np[i]),
                cm=float(cm_np[i]),
                cl_cd_ratio=float(cl_cd_np[i]),
            )
        )

    return results


def _aerosandbox_airfoil_polar(
    airfoil_name: str, alpha_deg_list: list[float], reynolds: float, mach: float
) -> list[AirfoilPoint]:
    """Generate airfoil polar using AeroSandbox XFoil integration."""
    try:
        # Create airfoil
        airfoil = asb.Airfoil(airfoil_name)

        results = []

        for alpha_deg in alpha_deg_list:
            try:
                # Run XFoil analysis
                result = airfoil.get_aero_from_neuralfoil(
                    alpha=alpha_deg, Re=reynolds, mach=mach
                )

                cl = result["CL"]
                cd = result["CD"]
                cm = result.get("CM", 0.0)

                # Use np.asarray().item() to extract scalar (NumPy 1.25+ deprecation fix)
                cl_scalar = float(np.asarray(cl).item())
                cd_scalar = float(np.asarray(cd).item())
                cm_scalar = float(np.asarray(cm).item())

                results.append(
                    AirfoilPoint(
                        alpha_deg=alpha_deg,
                        cl=cl_scalar,
                        cd=cd_scalar,
                        cm=cm_scalar,
                        cl_cd_ratio=cl_scalar / cd_scalar if cd_scalar > 0.001 else 0.0,
                    )
                )

            except Exception:
                # Fall back to database for this point
                db_result = _database_airfoil_polar(
                    airfoil_name, [alpha_deg], reynolds, mach
                )
                if db_result:
                    results.extend(db_result)

        return results

    except Exception:
        # Fall back to database method
        return _database_airfoil_polar(airfoil_name, alpha_deg_list, reynolds, mach)


def calculate_stability_derivatives(
    geometry: WingGeometry, alpha_deg: float = 2.0, mach: float = 0.2
) -> StabilityDerivatives:
    """
    Calculate basic longitudinal stability derivatives.

    Uses NumPy for efficient calculations.

    Args:
        geometry: Wing geometry
        alpha_deg: Reference angle of attack
        mach: Mach number

    Returns:
        StabilityDerivatives object
    """
    # Calculate wing area and aspect ratio
    S = geometry.span_m * (geometry.chord_root_m + geometry.chord_tip_m) / 2
    AR = geometry.span_m**2 / S

    # Get airfoil data
    airfoil_data = AIRFOIL_DATABASE.get(
        geometry.airfoil_root, AIRFOIL_DATABASE["NACA2412"]
    )

    # 3D lift curve slope using NumPy
    e = 0.85  # Oswald efficiency
    beta = float(np.sqrt(max(0.01, 1 - mach**2)))
    CL_alpha_2d = airfoil_data["cl_alpha"]
    CL_alpha = CL_alpha_2d / (1 + CL_alpha_2d / (PI * AR * e)) / beta

    # Pitching moment slope (simplified)
    CM_alpha = -0.1 * CL_alpha

    return StabilityDerivatives(
        CL_alpha=float(CL_alpha),
        CM_alpha=float(CM_alpha),
        CL_alpha_dot=None,
        CM_alpha_dot=None,
    )


def get_airfoil_database() -> dict[str, dict[str, float]]:
    """Get available airfoil database."""
    return AIRFOIL_DATABASE.copy()


def estimate_wing_area(geometry: WingGeometry) -> dict[str, float]:
    """
    Calculate wing geometric properties.

    Uses NumPy for efficient calculations.

    Args:
        geometry: Wing planform geometry

    Returns:
        Dictionary with wing area, aspect ratio, etc.
    """
    # Wing area (trapezoidal approximation)
    S = geometry.span_m * (geometry.chord_root_m + geometry.chord_tip_m) / 2

    # Aspect ratio
    AR = geometry.span_m**2 / S

    # Taper ratio
    taper_ratio = geometry.chord_tip_m / geometry.chord_root_m

    # Mean aerodynamic chord
    MAC = (
        (2 / 3)
        * geometry.chord_root_m
        * (1 + taper_ratio + taper_ratio**2)
        / (1 + taper_ratio)
    )

    # Sweep of mean aerodynamic chord (approximate)
    sweep_MAC_deg = geometry.sweep_deg - float(
        np.degrees(np.arctan(4 / AR * (0.25) * (1 - taper_ratio) / (1 + taper_ratio)))
    )

    return {
        "wing_area_m2": S,
        "aspect_ratio": AR,
        "taper_ratio": taper_ratio,
        "mean_aerodynamic_chord_m": MAC,
        "sweep_MAC_deg": sweep_MAC_deg,
        "span_m": geometry.span_m,
    }
