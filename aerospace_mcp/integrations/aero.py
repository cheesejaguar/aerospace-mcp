"""
Aircraft Aerodynamics Tools

Provides aircraft aerodynamics analysis including VLM wing analysis,
airfoil polars, and basic aerodynamic calculations. Falls back to 
simplified methods when optional dependencies are unavailable.
"""

import math
from typing import Dict, List, Optional, Tuple, Union
from pydantic import BaseModel, Field
import numpy as np

from . import update_availability

# Optional library imports
AEROSANDBOX_AVAILABLE = False
MACHUPX_AVAILABLE = False

try:
    import aerosandbox as asb
    import aerosandbox.numpy as anp
    AEROSANDBOX_AVAILABLE = True
    update_availability("aero", True, {"aerosandbox": asb.__version__})
except ImportError:
    try:
        import machupx as mx
        MACHUPX_AVAILABLE = True
        update_availability("aero", True, {"machupx": "unknown"})
    except ImportError:
        update_availability("aero", True, {})  # Still available with manual methods

# Airfoil database - simplified coefficients for common airfoils
AIRFOIL_DATABASE = {
    "NACA0012": {
        "cl_alpha": 6.28,  # per radian
        "cd0": 0.006,
        "cl_max": 1.4,
        "alpha_stall_deg": 15.0,
        "cm0": 0.0,  # symmetric airfoil
    },
    "NACA2412": {
        "cl_alpha": 6.28,
        "cd0": 0.007,
        "cl_max": 1.6,
        "alpha_stall_deg": 16.0,
        "cm0": -0.05,
    },
    "NACA4412": {
        "cl_alpha": 6.28,
        "cd0": 0.008,
        "cl_max": 1.7,
        "alpha_stall_deg": 17.0,
        "cm0": -0.08,
    },
    "NACA6412": {
        "cl_alpha": 6.28,
        "cd0": 0.007,
        "cl_max": 1.8,
        "alpha_stall_deg": 18.0,
        "cm0": -0.12,
    },
    "CLARKY": {
        "cl_alpha": 6.0,
        "cd0": 0.008,
        "cl_max": 1.5,
        "alpha_stall_deg": 14.0,
        "cm0": -0.06,
    }
}

# Data models
class AirfoilPoint(BaseModel):
    """Single airfoil polar point."""
    alpha_deg: float = Field(..., description="Angle of attack in degrees")
    cl: float = Field(..., description="Lift coefficient")
    cd: float = Field(..., description="Drag coefficient") 
    cm: Optional[float] = Field(None, description="Moment coefficient")
    cl_cd_ratio: float = Field(..., description="Lift to drag ratio")

class WingGeometry(BaseModel):
    """Wing planform geometry definition."""
    span_m: float = Field(..., gt=0, description="Wing span in meters")
    chord_root_m: float = Field(..., gt=0, description="Root chord in meters")
    chord_tip_m: float = Field(..., gt=0, description="Tip chord in meters")
    sweep_deg: float = Field(0.0, ge=-45, le=45, description="Quarter-chord sweep in degrees")
    dihedral_deg: float = Field(0.0, ge=-15, le=15, description="Dihedral angle in degrees")
    twist_deg: float = Field(0.0, ge=-10, le=10, description="Geometric twist (tip relative to root)")
    airfoil_root: str = Field("NACA2412", description="Root airfoil name")
    airfoil_tip: str = Field("NACA2412", description="Tip airfoil name")

class WingAnalysisPoint(BaseModel):
    """Wing analysis results at a single condition."""
    alpha_deg: float = Field(..., description="Wing angle of attack")
    CL: float = Field(..., description="Wing lift coefficient")
    CD: float = Field(..., description="Wing drag coefficient")
    CM: float = Field(..., description="Wing pitching moment coefficient")
    L_D_ratio: float = Field(..., description="Lift to drag ratio")
    span_efficiency: Optional[float] = Field(None, description="Span efficiency factor")

class StabilityDerivatives(BaseModel):
    """Longitudinal stability derivatives."""
    CL_alpha: float = Field(..., description="Lift curve slope (per radian)")
    CM_alpha: float = Field(..., description="Pitching moment curve slope")
    CL_alpha_dot: Optional[float] = Field(None, description="CL due to alpha rate")
    CM_alpha_dot: Optional[float] = Field(None, description="CM due to alpha rate")

def _simple_wing_analysis(geometry: WingGeometry, alpha_deg_list: List[float], mach: float = 0.2) -> List[WingAnalysisPoint]:
    """
    Simple wing analysis using lifting line theory approximations.
    Used as fallback when advanced libraries are unavailable.
    """
    results = []
    
    # Wing parameters
    S = geometry.span_m * (geometry.chord_root_m + geometry.chord_tip_m) / 2  # Wing area
    AR = geometry.span_m**2 / S  # Aspect ratio
    taper_ratio = geometry.chord_tip_m / geometry.chord_root_m
    
    # Get airfoil properties (assume root airfoil for simplicity)
    airfoil_data = AIRFOIL_DATABASE.get(geometry.airfoil_root, AIRFOIL_DATABASE["NACA2412"])
    
    # Prandtl lifting line corrections
    e = 0.85  # Oswald efficiency (typical for clean wing)
    CL_alpha_2d = airfoil_data["cl_alpha"]
    CL_alpha_3d = CL_alpha_2d / (1 + CL_alpha_2d / (math.pi * AR * e))
    
    # Mach number corrections (simplified)
    beta = math.sqrt(max(0.01, 1 - mach**2))
    CL_alpha_3d = CL_alpha_3d / beta
    
    for alpha_deg in alpha_deg_list:
        alpha_rad = math.radians(alpha_deg)
        
        # Basic lift coefficient
        CL = CL_alpha_3d * alpha_rad
        
        # Drag coefficient (simplified drag polar)
        CD0 = airfoil_data["cd0"] * 1.1  # Wing CD0 slightly higher than airfoil
        CDi = CL**2 / (math.pi * AR * e)  # Induced drag
        CD = CD0 + CDi
        
        # Pitching moment (very simplified)
        CM = airfoil_data["cm0"] + 0.02 * CL  # Approximate CM variation
        
        # Apply stall model
        if abs(alpha_deg) > airfoil_data["alpha_stall_deg"]:
            stall_factor = 1.0 - 0.1 * (abs(alpha_deg) - airfoil_data["alpha_stall_deg"])
            stall_factor = max(0.3, stall_factor)
            CL *= stall_factor
            CD *= (1.5 + 0.1 * (abs(alpha_deg) - airfoil_data["alpha_stall_deg"]))
        
        L_D = CL / CD if CD > 0.001 else 0.0
        
        results.append(WingAnalysisPoint(
            alpha_deg=alpha_deg,
            CL=CL,
            CD=CD, 
            CM=CM,
            L_D_ratio=L_D,
            span_efficiency=e
        ))
    
    return results

def wing_vlm_analysis(
    geometry: WingGeometry,
    alpha_deg_list: List[float],
    mach: float = 0.2,
    reynolds: Optional[float] = None
) -> List[WingAnalysisPoint]:
    """
    Vortex Lattice Method wing analysis.
    
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
        except Exception as e:
            # Fall back to simple method
            pass
    
    # Use simple lifting line approximation
    return _simple_wing_analysis(geometry, alpha_deg_list, mach)

def _aerosandbox_wing_analysis(geometry: WingGeometry, alpha_deg_list: List[float], 
                              mach: float, reynolds: Optional[float]) -> List[WingAnalysisPoint]:
    """AeroSandbox-based VLM analysis."""
    # Create wing geometry
    wing = asb.Wing(
        name="MainWing",
        xsecs=[
            asb.WingXSec(
                xyz_le=[0, 0, 0],
                chord=geometry.chord_root_m,
                airfoil=asb.Airfoil(geometry.airfoil_root)
            ),
            asb.WingXSec(
                xyz_le=[geometry.span_m/2 * math.tan(math.radians(geometry.sweep_deg)), 
                        geometry.span_m/2, 
                        geometry.span_m/2 * math.tan(math.radians(geometry.dihedral_deg))],
                chord=geometry.chord_tip_m,
                airfoil=asb.Airfoil(geometry.airfoil_tip),
                twist=math.radians(geometry.twist_deg)
            )
        ]
    )
    
    # Create airplane
    airplane = asb.Airplane(
        name="TestAirplane",
        wings=[wing]
    )
    
    results = []
    
    for alpha_deg in alpha_deg_list:
        # Create operating point
        op_point = asb.OperatingPoint(
            atmosphere=asb.Atmosphere(altitude=0),
            velocity=50.0,  # Arbitrary velocity
            alpha=math.radians(alpha_deg)
        )
        
        # Run VLM analysis
        vlm = asb.VortexLatticeMethod(
            airplane=airplane,
            op_point=op_point,
            chordwise_panels=10,
            spanwise_panels=20
        )
        
        vlm_result = vlm.run()
        
        # Extract coefficients
        CL = vlm_result["CL"]
        CD = vlm_result.get("CD", 0.01)  # VLM doesn't compute viscous drag
        CM = vlm_result.get("CM", 0.0)
        
        L_D = CL / CD if CD > 0.001 else 0.0
        
        results.append(WingAnalysisPoint(
            alpha_deg=alpha_deg,
            CL=float(CL),
            CD=float(CD),
            CM=float(CM),
            L_D_ratio=L_D
        ))
    
    return results

def airfoil_polar_analysis(
    airfoil_name: str,
    alpha_deg_list: List[float],
    reynolds: float = 1e6,
    mach: float = 0.1
) -> List[AirfoilPoint]:
    """
    Generate airfoil polar data.
    
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
            return _aerosandbox_airfoil_polar(airfoil_name, alpha_deg_list, reynolds, mach)
        except Exception as e:
            # Fall back to database method
            pass
    
    # Use simplified database method
    return _database_airfoil_polar(airfoil_name, alpha_deg_list, reynolds, mach)

def _database_airfoil_polar(airfoil_name: str, alpha_deg_list: List[float], 
                           reynolds: float, mach: float) -> List[AirfoilPoint]:
    """Generate airfoil polar from database coefficients."""
    # Get airfoil data from database
    airfoil_data = AIRFOIL_DATABASE.get(airfoil_name, AIRFOIL_DATABASE["NACA2412"])
    
    results = []
    
    for alpha_deg in alpha_deg_list:
        alpha_rad = math.radians(alpha_deg)
        
        # Reynolds number corrections (simplified)
        re_factor = min(1.2, (reynolds / 1e6) ** 0.1)
        
        # Mach number corrections
        beta = math.sqrt(max(0.01, 1 - mach**2))
        
        # Lift coefficient
        cl = (airfoil_data["cl_alpha"] * alpha_rad) / beta * re_factor
        
        # Apply stall model
        if abs(alpha_deg) > airfoil_data["alpha_stall_deg"]:
            stall_factor = 1.0 - 0.15 * (abs(alpha_deg) - airfoil_data["alpha_stall_deg"])
            stall_factor = max(0.2, stall_factor)
            cl *= stall_factor
        
        # Drag coefficient (simplified drag polar)
        cd0 = airfoil_data["cd0"]
        # Reynolds correction for cd0
        cd0 *= (1e6 / reynolds) ** 0.2
        
        # Mach corrections for drag
        if mach > 0.3:
            cd0 *= (1 + 0.2 * (mach - 0.3)**2)
            
        cd = cd0 + 0.01 * cl**2  # Simplified induced drag approximation
        
        # Moment coefficient
        cm = airfoil_data["cm0"] + 0.01 * cl
        
        # L/D ratio
        cl_cd = cl / cd if cd > 0.001 else 0.0
        
        results.append(AirfoilPoint(
            alpha_deg=alpha_deg,
            cl=cl,
            cd=cd,
            cm=cm,
            cl_cd_ratio=cl_cd
        ))
    
    return results

def _aerosandbox_airfoil_polar(airfoil_name: str, alpha_deg_list: List[float],
                              reynolds: float, mach: float) -> List[AirfoilPoint]:
    """Generate airfoil polar using AeroSandbox XFoil integration."""
    try:
        # Create airfoil
        airfoil = asb.Airfoil(airfoil_name)
        
        results = []
        
        for alpha_deg in alpha_deg_list:
            try:
                # Run XFoil analysis
                result = airfoil.get_aero_from_neuralfoil(
                    alpha=alpha_deg,
                    Re=reynolds,
                    mach=mach
                )
                
                cl = result["CL"]
                cd = result["CD"] 
                cm = result.get("CM", 0.0)
                
                results.append(AirfoilPoint(
                    alpha_deg=alpha_deg,
                    cl=float(cl),
                    cd=float(cd),
                    cm=float(cm),
                    cl_cd_ratio=float(cl/cd) if cd > 0.001 else 0.0
                ))
                
            except Exception as e:
                # Fall back to database for this point
                db_result = _database_airfoil_polar(airfoil_name, [alpha_deg], reynolds, mach)
                if db_result:
                    results.extend(db_result)
        
        return results
        
    except Exception as e:
        # Fall back to database method
        return _database_airfoil_polar(airfoil_name, alpha_deg_list, reynolds, mach)

def calculate_stability_derivatives(
    geometry: WingGeometry,
    alpha_deg: float = 2.0,
    mach: float = 0.2
) -> StabilityDerivatives:
    """
    Calculate basic longitudinal stability derivatives.
    
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
    airfoil_data = AIRFOIL_DATABASE.get(geometry.airfoil_root, AIRFOIL_DATABASE["NACA2412"])
    
    # 3D lift curve slope
    e = 0.85  # Oswald efficiency
    beta = math.sqrt(max(0.01, 1 - mach**2))
    CL_alpha_2d = airfoil_data["cl_alpha"]
    CL_alpha = CL_alpha_2d / (1 + CL_alpha_2d / (math.pi * AR * e)) / beta
    
    # Pitching moment slope (simplified)
    # Typically negative for stable aircraft
    CM_alpha = -0.1 * CL_alpha  # Rough approximation
    
    return StabilityDerivatives(
        CL_alpha=CL_alpha,
        CM_alpha=CM_alpha,
        CL_alpha_dot=None,  # Would need unsteady analysis
        CM_alpha_dot=None
    )

def get_airfoil_database() -> Dict[str, Dict[str, float]]:
    """Get available airfoil database."""
    return AIRFOIL_DATABASE.copy()

def estimate_wing_area(geometry: WingGeometry) -> Dict[str, float]:
    """
    Calculate wing geometric properties.
    
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
    MAC = (2/3) * geometry.chord_root_m * (1 + taper_ratio + taper_ratio**2) / (1 + taper_ratio)
    
    # Sweep of mean aerodynamic chord (approximate)
    sweep_MAC_deg = geometry.sweep_deg - math.degrees(
        math.atan(4/AR * (0.25) * (1 - taper_ratio) / (1 + taper_ratio))
    )
    
    return {
        "wing_area_m2": S,
        "aspect_ratio": AR,
        "taper_ratio": taper_ratio,
        "mean_aerodynamic_chord_m": MAC,
        "sweep_MAC_deg": sweep_MAC_deg,
        "span_m": geometry.span_m
    }