"""Aircraft Aerodynamics Tools.

Provides aircraft aerodynamics analysis including:
    - Wing analysis: Vortex Lattice Method via AeroSandbox when installed,
      otherwise a lifting-line / DATCOM estimate (Helmbold-Diederich lift
      slope with sweep, taper, twist, dihedral and compressibility)
    - Airfoil polar generation (NeuralFoil via AeroSandbox when installed,
      otherwise a database / thin-airfoil model with stall and a flat-plate
      post-stall model)
    - Longitudinal stability derivative estimation (wing alone)
    - Wing planform geometric property calculations (area, AR, MAC, taper)

Conventions:
    - Angles of attack are in degrees, measured at the wing root chord.
    - Section pitching moments are about the quarter chord (c/4).
    - Wing pitching moments are about a reference point ``x_ref_mac`` (a
      fraction of the mean aerodynamic chord, measured from the MAC leading
      edge; default 0.25, the quarter chord of the MAC).  The wing
      aerodynamic centre is taken at 0.25 MAC, so about the default
      reference the wing alone is neutrally stable (dCM/dCL = 0).
    - All models are subsonic: Mach numbers of 0.9 and above are rejected,
      and above 0.7 a warning notes that transonic effects are not modelled.

Results carry a ``method`` field naming the model that ran and a
``warnings`` list (for example when an optional library failed and the
built-in model was used instead).

Uses NumPy for vectorized calculations with CuPy compatibility for GPU
acceleration via the ``_array_backend`` module.

References:
    - Katz, J. & Plotkin, A., "Low-Speed Aerodynamics" (2nd ed., 2001)
    - Anderson, J.D., "Fundamentals of Aerodynamics" (6th ed., 2017)
    - Phillips, W.F., "Mechanics of Flight" (2nd ed., 2010)
    - USAF Stability and Control DATCOM (Hoak, 1978), sections 4.1.3, 4.1.4
    - Raymer, D.P., "Aircraft Design: A Conceptual Approach" (6th ed., 2018)
    - Viterna, L.A. & Corrigan, R.D., "Fixed pitch rotor performance of large
      horizontal axis wind turbines", NASA CP-2230 (1982)

WARNING: This module is for educational and research purposes only.
Do NOT use for real flight planning, navigation, or aircraft operations.
"""

import logging
import math
import re
import warnings as _py_warnings

from pydantic import BaseModel, Field

from . import update_availability
from ._array_backend import np, to_numpy

logger = logging.getLogger(__name__)

# ===========================================================================
# Constants
# ===========================================================================

PI = np.pi  # Archimedes' constant (3.14159...)

# Subsonic validity limits of the models in this module.
MACH_LIMIT = 0.9  # rejected at and above
MACH_WARNING = 0.7  # warning above (critical Mach of typical sections)

# Oswald span efficiency used for the induced drag of the fallback wing model.
OSWALD_EFFICIENCY = 0.85

# Wing aerodynamic centre as a fraction of the MAC (subsonic estimate).
WING_AC_MAC = 0.25

# Angle (rad) over which post-stall results blend into the flat-plate model.
STALL_BLEND_RAD = math.radians(10.0)

# ===========================================================================
# Optional Library Imports
# ===========================================================================

AEROSANDBOX_AVAILABLE = False
# MachUpX is not integrated (the flag used to be set without importing it).
MACHUPX_AVAILABLE = False

try:
    import aerosandbox as asb

    AEROSANDBOX_AVAILABLE = True
    update_availability("aero", True, {"aerosandbox": asb.__version__})
except ImportError:
    update_availability("aero", True, {})  # Still available with manual methods

# ===========================================================================
# Airfoil Coefficient Database
# ===========================================================================

# Simplified linearized coefficients for common airfoils.
# cl_alpha: lift-curve slope (per radian), ~2*pi from thin-airfoil theory.
# cl0: lift coefficient at zero angle of attack (camber contribution).
# cd0: minimum (zero-lift) drag coefficient.
# cl_max: maximum lift coefficient before stall.
# alpha_stall_deg: stall angle of attack (degrees).
# cm0: pitching moment coefficient about the quarter chord.
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

_NACA4_PATTERN = re.compile(r"^NACA(\d)(\d)(\d{2})$")


# ===========================================================================
# Data Models
# ===========================================================================


class AirfoilPoint(BaseModel):
    """Single point on an airfoil polar curve (CL, CD, CM vs. alpha)."""

    alpha_deg: float = Field(..., description="Angle of attack in degrees")
    cl: float = Field(..., description="Lift coefficient")
    cd: float = Field(..., description="Drag coefficient")
    cm: float | None = Field(
        None, description="Pitching moment coefficient about the quarter chord"
    )
    cl_cd_ratio: float = Field(..., description="Lift to drag ratio")
    method: str | None = Field(None, description="Model that produced the point")
    warnings: list[str] = Field(default_factory=list, description="Model warnings")


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
        0.0,
        ge=-10,
        le=10,
        description="Geometric twist, tip relative to root (negative = washout)",
    )
    airfoil_root: str = Field("NACA2412", description="Root airfoil name")
    airfoil_tip: str = Field("NACA2412", description="Tip airfoil name")


class WingAnalysisPoint(BaseModel):
    """Wing analysis results at a single condition."""

    alpha_deg: float = Field(..., description="Wing (root chord) angle of attack")
    CL: float = Field(..., description="Wing lift coefficient")
    CD: float = Field(..., description="Wing drag coefficient")
    CM: float = Field(
        ..., description="Wing pitching moment coefficient about x_ref_mac * MAC"
    )
    L_D_ratio: float = Field(..., description="Lift to drag ratio")
    span_efficiency: float | None = Field(None, description="Span efficiency factor")
    method: str | None = Field(None, description="Model that produced the point")
    warnings: list[str] = Field(default_factory=list, description="Model warnings")


class StabilityDerivatives(BaseModel):
    """Longitudinal stability derivatives (wing alone)."""

    CL_alpha: float = Field(..., description="Lift curve slope dCL/dalpha (per rad)")
    CM_alpha: float = Field(
        ..., description="Pitching moment slope dCM/dalpha about x_ref_mac (per rad)"
    )
    CL_alpha_dot: float | None = Field(
        None, description="CL due to alpha rate (not computed, see notes)"
    )
    CM_alpha_dot: float | None = Field(
        None, description="CM due to alpha rate (not computed, see notes)"
    )
    alpha_deg: float | None = Field(None, description="Angle of attack evaluated")
    CL: float | None = Field(None, description="Lift coefficient at alpha_deg")
    stalled: bool | None = Field(None, description="alpha_deg is beyond stall")
    x_ref_mac: float | None = Field(
        None, description="Moment reference, fraction of MAC from its leading edge"
    )
    method: str | None = Field(None, description="Model used")
    notes: list[str] = Field(default_factory=list, description="Assumptions")


# ===========================================================================
# Airfoil Data Resolution
# ===========================================================================


def normalize_airfoil_name(name: str) -> str:
    """Normalize an airfoil name: upper case without spaces, '-' or '_'.

    ``"naca 2412"``, ``"NACA-2412"`` and ``"naca2412"`` all become
    ``"NACA2412"``; ``"Clark Y"`` becomes ``"CLARKY"``.
    """
    return re.sub(r"[\s_\-]", "", str(name)).upper()


def _naca4_coefficients(m_digit: int, p_digit: int, t_digits: int) -> dict:
    """Section coefficients of a NACA 4-digit airfoil from thin-airfoil theory.

    Zero-lift angle and quarter-chord moment of the NACA mean line::

        alpha_L0 = -(1/pi) * integral_0^pi dz/dx (cos(theta) - 1) dtheta
        cm_c/4   = (pi/4) * (A2 - A1),  A_n = (2/pi) integral dz/dx cos(n theta)

    (NACA 2412: alpha_L0 = -2.08 deg, cm = -0.053.)  Lift slope 6.28 / rad;
    cd0, cl_max and stall angle are empirical fits to the database entries.
    """
    m = m_digit / 100.0
    p = p_digit / 10.0
    t = t_digits / 100.0
    if t <= 0:
        raise ValueError("NACA 4-digit thickness must be positive")
    if m > 0 and p <= 0:
        raise ValueError("NACA 4-digit camber position must be positive when cambered")

    alpha_l0 = 0.0
    cm0 = 0.0
    if m > 0:
        n = 4000
        theta = (np.arange(n) + 0.5) * math.pi / n
        x = (1 - np.cos(theta)) / 2
        dzdx = np.where(x < p, 2 * m / p**2 * (p - x), 2 * m / (1 - p) ** 2 * (p - x))
        dtheta = math.pi / n
        alpha_l0 = float(-np.sum(dzdx * (np.cos(theta) - 1)) * dtheta / math.pi)
        a1 = float(2 / math.pi * np.sum(dzdx * np.cos(theta)) * dtheta)
        a2 = float(2 / math.pi * np.sum(dzdx * np.cos(2 * theta)) * dtheta)
        cm0 = math.pi / 4 * (a2 - a1)

    cl_alpha = 6.28
    cl_max = min(1.0 + 3.33 * min(t, 0.15) + 7.5 * m, 1.9)
    return {
        "cl_alpha": cl_alpha,
        "cl0": -cl_alpha * alpha_l0,
        "cd0": 0.0045 + 0.0125 * t + 0.025 * m,
        "cl_max": cl_max,
        "alpha_stall_deg": math.degrees(alpha_l0 + cl_max / cl_alpha) + 3.0,
        "cm0": cm0,
    }


def get_airfoil_coefficients(name: str) -> dict[str, float]:
    """Return the section coefficients used for an airfoil name.

    Names are normalized (case, spaces, dashes).  Entries of
    ``AIRFOIL_DATABASE`` are used as given; any other NACA 4-digit
    designation is derived from thin-airfoil theory.

    Raises:
        ValueError: For unknown airfoil names (they used to silently become
            NACA2412).
    """
    key = normalize_airfoil_name(name)
    if key in AIRFOIL_DATABASE:
        return dict(AIRFOIL_DATABASE[key])
    match = _NACA4_PATTERN.match(key)
    if match:
        return _naca4_coefficients(*(int(g) for g in match.groups()))
    raise ValueError(
        f"Unknown airfoil '{name}'. Known airfoils: "
        f"{', '.join(sorted(AIRFOIL_DATABASE))}, or any NACA 4-digit designation"
    )


# ===========================================================================
# Flow Condition Validation
# ===========================================================================


def _check_flow(mach: float, reynolds: float | None = None) -> list[str]:
    """Validate Mach and Reynolds numbers; return warnings.

    Raises:
        ValueError: Mach outside [0, 0.9) or Reynolds number <= 0.
    """
    if not math.isfinite(mach) or mach < 0:
        raise ValueError("Mach number must be >= 0")
    if mach >= MACH_LIMIT:
        raise ValueError(
            f"Mach {mach:g} is outside the subsonic models (valid for M < "
            f"{MACH_LIMIT}); transonic and supersonic flow are not modelled"
        )
    if reynolds is not None and (not math.isfinite(reynolds) or reynolds <= 0):
        raise ValueError("Reynolds number must be positive")
    warnings = []
    if mach > MACH_WARNING:
        warnings.append(
            f"Mach {mach:.2f} is above ~{MACH_WARNING}: shock formation and drag "
            "divergence are not modelled (Prandtl-Glauert only)"
        )
    return warnings


def _smoothstep(t):
    """Smooth 0 -> 1 blend weight for t in [0, 1] (clipped)."""
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


def _rounded_lift_curve(s, slope, cl_peak, s_stall):
    """Lift up to stall: linear, then a parabola reaching ``cl_peak`` at
    ``s_stall`` with zero slope, then constant.

    ``s`` is the angle from zero lift (rad, >= 0).  The parabola starts at
    the knee ``s_k = 2 cl_peak / slope - s_stall`` where its slope equals the
    linear slope, so the curve and its slope are continuous.
    """
    s_lin = cl_peak / slope
    s_stall = np.clip(s_stall, s_lin, 2 * s_lin)
    s_knee = 2 * s_lin - s_stall
    width = np.maximum(s_stall - s_knee, 1e-12)
    parabola = cl_peak - slope / (2 * width) * (s_stall - s) ** 2
    return np.where(s <= s_knee, slope * s, np.where(s < s_stall, parabola, cl_peak))


# ===========================================================================
# Wing Analysis -- Lifting Line / DATCOM Fallback
# ===========================================================================


def _planform(geometry: WingGeometry) -> dict[str, float]:
    """Trapezoidal planform quantities used by the wing models."""
    b = geometry.span_m
    c_r = geometry.chord_root_m
    c_t = geometry.chord_tip_m
    taper = c_t / c_r
    S = b * (c_r + c_t) / 2
    AR = b**2 / S
    mac = (2 / 3) * c_r * (1 + taper + taper**2) / (1 + taper)
    # Spanwise MAC position as a fraction of the semi-span; also the
    # area-weighted mean of a quantity that varies linearly root -> tip.
    eta_mac = (1 + 2 * taper) / (3 * (1 + taper))
    tan_c4 = math.tan(math.radians(geometry.sweep_deg))
    # tan(Lambda_n) = tan(Lambda_m) - (4/AR) (n - m) (1 - taper) / (1 + taper)
    tan_le = tan_c4 + (1 / AR) * (1 - taper) / (1 + taper)
    tan_c2 = tan_c4 - (1 / AR) * (1 - taper) / (1 + taper)
    x_mac_le = eta_mac * (b / 2) * tan_le
    return {
        "S": S,
        "AR": AR,
        "taper": taper,
        "mac": mac,
        "eta_mac": eta_mac,
        "tan_c4": tan_c4,
        "tan_le": tan_le,
        "tan_c2": tan_c2,
        "x_mac_le": x_mac_le,
    }


def _blended_section(geometry: WingGeometry, mach: float) -> dict[str, float]:
    """Root/tip section properties blended with the MAC weighting.

    Properties vary linearly from root to tip; the wing uses their
    area-weighted mean ``(1 - eta) * root + eta * tip`` with
    ``eta = (1 + 2 taper) / (3 (1 + taper))``.  Prandtl-Glauert is applied to
    the section lift slope and moment.
    """
    root = get_airfoil_coefficients(geometry.airfoil_root)
    tip = get_airfoil_coefficients(geometry.airfoil_tip)
    eta = _planform(geometry)["eta_mac"]
    beta = math.sqrt(1 - mach**2)

    def blend(key):
        return (1 - eta) * root[key] + eta * tip[key]

    return {
        "cl_alpha": blend("cl_alpha"),  # incompressible section slope
        "alpha_l0": (1 - eta) * (-root["cl0"] / root["cl_alpha"])
        + eta * (-tip["cl0"] / tip["cl_alpha"]),
        "cl_max": blend("cl_max"),
        "cl_neg_max": (1 - eta) * max(root["cl_max"] - 2 * root["cl0"], 0.5)
        + eta * max(tip["cl_max"] - 2 * tip["cl0"], 0.5),
        "cd0": blend("cd0"),
        "cm0": blend("cm0") / beta,
    }


def _wing_lift_slope(geometry: WingGeometry, mach: float, cl_alpha_2d: float):
    """Helmbold-Diederich (DATCOM 4.1.3.2) subsonic wing lift-curve slope::

        CL_alpha = 2 pi A / (2 + sqrt(A^2 beta^2 / kappa^2
                                       (1 + tan^2(L_c/2) / beta^2) + 4))

    with beta = sqrt(1 - M^2), kappa = cl_alpha / (2 pi) and the half-chord
    sweep L_c/2; multiplied by cos^2(dihedral).  AR 8 rectangular wing with
    2*pi sections: 4.91 / rad unswept, 4.20 at 35 deg and 3.73 at 45 deg.
    """
    pf = _planform(geometry)
    A = pf["AR"]
    beta = math.sqrt(1 - mach**2)
    kappa = cl_alpha_2d / (2 * math.pi)
    root = math.sqrt((A * beta / kappa) ** 2 * (1 + pf["tan_c2"] ** 2 / beta**2) + 4)
    cl_alpha = 2 * math.pi * A / (2 + root)
    return cl_alpha * math.cos(math.radians(geometry.dihedral_deg)) ** 2


def _twist_moment(geometry: WingGeometry, cl_alpha_3d: float) -> float:
    """Pitching moment from the twist-induced (zero-net-lift) basic loading.

    The incidence change ``eps * (eta - eta_mac)`` (linear twist eps, zero
    area-weighted mean) gives an additional lift that acts at the local
    quarter chord ``eta * (b/2) * tan(L_c/4)``.  Following Schrenk's
    approximation, the basic loading is half the strip-theory loading
    ``cl_alpha_3d * eps * (eta - eta_mac)``.  Washout on an aft-swept wing
    unloads the (aft) tips and so gives a nose-up moment; it is zero
    without sweep.
    """
    pf = _planform(geometry)
    eps = math.radians(geometry.twist_deg)
    if eps == 0.0 or pf["tan_c4"] == 0.0:
        return 0.0
    half = geometry.span_m / 2
    eta = (np.arange(400) + 0.5) / 400
    chord = geometry.chord_root_m * (1 - (1 - pf["taper"]) * eta)
    delta_cl = 0.5 * cl_alpha_3d * eps * (eta - pf["eta_mac"])  # Schrenk
    x_c4 = eta * half * pf["tan_c4"]
    integral = float(np.sum(chord * delta_cl * x_c4)) / 400 * half
    return -2 * integral / (pf["S"] * pf["mac"])


def _wing_model(
    geometry: WingGeometry,
    alphas_deg,
    mach: float,
    reynolds: float | None = None,
    x_ref_mac: float = 0.25,
) -> dict:
    """Vectorized lifting-line / DATCOM wing model (see ``_simple_wing_analysis``)."""
    pf = _planform(geometry)
    sec = _blended_section(geometry, mach)
    A = pf["AR"]
    alphas_deg = np.asarray(alphas_deg, dtype=np.float64).reshape(-1)
    alphas = np.radians(alphas_deg)

    cl_alpha = _wing_lift_slope(geometry, mach, sec["cl_alpha"])

    # Zero-lift angle (root chord reference) with a linear-twist correction:
    # the area-weighted mean incidence of a linearly twisted wing is
    # eps * eta_mac above the root.
    eps = math.radians(geometry.twist_deg)
    alpha_l0 = sec["alpha_l0"] - eps * pf["eta_mac"]

    # Maximum lift (Raymer: CL_max ~ 0.9 cl_max cos(L_c/4)) and stall angle
    # (linear stall angle + 2 deg, cf. DATCOM delta alpha_CLmax)
    cos_c4 = 1 / math.sqrt(1 + pf["tan_c4"] ** 2)
    cl_max = 0.9 * sec["cl_max"] * cos_c4
    cl_neg = 0.9 * sec["cl_neg_max"] * cos_c4
    extra = math.radians(2.0)
    s_stall_pos = cl_max / cl_alpha + extra
    s_stall_neg = cl_neg / cl_alpha + extra

    s = alphas - alpha_l0
    cl_att = np.where(
        s >= 0,
        _rounded_lift_curve(np.abs(s), cl_alpha, cl_max, s_stall_pos),
        -_rounded_lift_curve(np.abs(s), cl_alpha, cl_neg, s_stall_neg),
    )

    # Profile drag (Reynolds correction when given) and Oswald induced drag
    cd0 = sec["cd0"] * 1.1  # wing CD0 ~10% above the section value
    if reynolds is not None:
        cd0 = cd0 * (1e6 / reynolds) ** 0.2
    if mach > 0.3:
        cd0 = cd0 * (1 + 0.2 * (mach - 0.3) ** 2)
    cd_att = cd0 + cl_att**2 / (PI * A * OSWALD_EFFICIENCY)

    # Moment about the reference point: CM = CM_ac + CL (x_ref - x_ac).
    # DATCOM 4.1.4.1: CM0 = [A cos^2 L / (A + 2 cos L)] * cm0_section + twist.
    cm_ac = (A * cos_c4**2 / (A + 2 * cos_c4)) * sec["cm0"] + _twist_moment(
        geometry, cl_alpha
    )
    cm_att = cm_ac + cl_att * (x_ref_mac - WING_AC_MAC)

    # Post-stall: blend into a finite flat plate (Viterna & Corrigan):
    # CD_max = 1.11 + 0.018 A, CL = CD_max/2 sin 2a, CD = CD_max sin^2 a,
    # normal force acting from 0.25 MAC (small a) to 0.5 MAC (90 deg).
    cd_max = min(1.11 + 0.018 * A, 2.0)
    cl_fp = cd_max / 2 * np.sin(2 * alphas)
    cd_fp = cd0 + cd_max * np.sin(alphas) ** 2
    cn_fp = cd_max * np.sin(alphas)
    x_cp = 0.25 + 0.25 * np.minimum(np.abs(alphas) / (PI / 2), 1.0)
    cm_fp = cm_ac - cn_fp * (x_cp - x_ref_mac)

    excess = np.maximum(s - s_stall_pos, 0) + np.maximum(-s_stall_neg - s, 0)
    w = _smoothstep(excess / STALL_BLEND_RAD)
    CL = (1 - w) * cl_att + w * cl_fp
    CD = (1 - w) * cd_att + w * np.maximum(cd_fp, cd_att)
    CM = (1 - w) * cm_att + w * cm_fp

    return {
        "alpha_deg": alphas_deg,
        "CL": CL,
        "CD": CD,
        "CM": CM,
        "stalled": (s > s_stall_pos) | (s < -s_stall_neg),
        "CL_alpha": cl_alpha,
        "alpha_l0_deg": math.degrees(alpha_l0),
        "alpha_stall_deg": math.degrees(alpha_l0 + s_stall_pos),
        "CL_max": cl_max,
        "cd0": cd0,
    }


def _simple_wing_analysis(
    geometry: WingGeometry,
    alpha_deg_list: list[float],
    mach: float = 0.2,
    reynolds: float | None = None,
    x_ref_mac: float = 0.25,
    warnings: list[str] | None = None,
) -> list[WingAnalysisPoint]:
    """Wing analysis using lifting-line / DATCOM estimates.

    - Lift slope: Helmbold-Diederich (DATCOM) with the half-chord sweep,
      Prandtl-Glauert compressibility inside the formula (beta), section
      slope from the airfoils and a cos^2(dihedral) factor.
    - Lift: ``CL = CL_alpha (alpha - alpha_0L)``, the zero-lift angle from
      the blended sections plus a linear-twist correction; the curve rounds
      off to ``CL_max = 0.9 cl_max cos(L_c/4)`` and blends into a
      finite-flat-plate model beyond stall.
    - Drag: ``CD = 1.1 cd0 + CL^2 / (pi A e)`` with e = 0.85 (cd0 Reynolds
      corrected when ``reynolds`` is given), flat-plate drag after stall.
    - Moment about ``x_ref_mac`` (fraction of MAC): DATCOM zero-lift moment
      plus ``CL (x_ref_mac - 0.25)``.

    Args:
        geometry: Wing planform geometry.
        alpha_deg_list: Angles of attack to evaluate (degrees).
        mach: Free-stream Mach number (< 0.9).
        reynolds: Optional Reynolds number (MAC based) for profile drag.
        x_ref_mac: Moment reference as a fraction of the MAC.
        warnings: Warnings to attach to every point.

    Returns:
        List of wing analysis results at each angle of attack.
    """
    warnings = list(warnings or [])
    model = _wing_model(geometry, alpha_deg_list, mach, reynolds, x_ref_mac)

    results = []
    alphas_np = to_numpy(model["alpha_deg"])
    CL_np = to_numpy(model["CL"])
    CD_np = to_numpy(model["CD"])
    CM_np = to_numpy(model["CM"])
    stalled_np = to_numpy(model["stalled"])
    for i in range(len(alphas_np)):
        point_warnings = list(warnings)
        if stalled_np[i]:
            point_warnings.append(
                f"Beyond the estimated stall angle ({model['alpha_stall_deg']:.1f} deg): "
                "post-stall values are approximate"
            )
        CL_i, CD_i = float(CL_np[i]), float(CD_np[i])
        results.append(
            WingAnalysisPoint(
                alpha_deg=float(alphas_np[i]),
                CL=CL_i,
                CD=CD_i,
                CM=float(CM_np[i]),
                L_D_ratio=CL_i / CD_i if CD_i > 1e-4 else 0.0,
                span_efficiency=OSWALD_EFFICIENCY,
                method="lifting_line_datcom",
                warnings=point_warnings,
            )
        )
    return results


# ===========================================================================
# VLM Wing Analysis (Primary Entry Point)
# ===========================================================================


def wing_vlm_analysis(
    geometry: WingGeometry,
    alpha_deg_list: list[float],
    mach: float = 0.2,
    reynolds: float | None = None,
    x_ref_mac: float = 0.25,
) -> list[WingAnalysisPoint]:
    """
    Vortex Lattice Method wing analysis.

    Uses AeroSandbox's VLM when installed, otherwise the lifting-line /
    DATCOM estimate of ``_simple_wing_analysis``.  If AeroSandbox fails the
    reason is logged and attached to the results as a warning; the
    ``method`` field of each point names the model that ran.

    Args:
        geometry: Wing planform geometry (airfoils from the database or any
            NACA 4-digit designation; names are case-insensitive)
        alpha_deg_list: List of angles of attack to analyze (degrees)
        mach: Mach number (must be below 0.9)
        reynolds: Reynolds number (optional, used for profile drag)
        x_ref_mac: Moment reference point as a fraction of the MAC from its
            leading edge (default 0.25, the MAC quarter chord)

    Returns:
        List of WingAnalysisPoint objects with CL, CD, CM data
    """
    warnings = _check_flow(mach, reynolds)
    # Resolve airfoils up front so unknown names fail on every path
    get_airfoil_coefficients(geometry.airfoil_root)
    get_airfoil_coefficients(geometry.airfoil_tip)

    if AEROSANDBOX_AVAILABLE and len(alpha_deg_list) > 0:
        try:
            return _aerosandbox_wing_analysis(
                geometry, alpha_deg_list, mach, reynolds, x_ref_mac, warnings
            )
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            logger.warning("AeroSandbox VLM failed, using lifting-line: %s", reason)
            warnings = warnings + [
                f"AeroSandbox VLM failed ({reason}); used the lifting-line/DATCOM estimate"
            ]

    return _simple_wing_analysis(
        geometry, alpha_deg_list, mach, reynolds, x_ref_mac, warnings
    )


def _load_asb_airfoil(name: str):
    """AeroSandbox airfoil by normalized (lower-case) name.

    AeroSandbox's own warning for unknown names is suppressed; callers check
    ``coordinates`` and raise a clear error instead.
    """
    with _py_warnings.catch_warnings():
        _py_warnings.simplefilter("ignore", UserWarning)
        return asb.Airfoil(normalize_airfoil_name(name).lower())


def _aerosandbox_wing_analysis(
    geometry: WingGeometry,
    alpha_deg_list: list[float],
    mach: float,
    reynolds: float | None,
    x_ref_mac: float = 0.25,
    warnings: list[str] | None = None,
) -> list[WingAnalysisPoint]:
    """VLM wing analysis using AeroSandbox (4.x API).

    Builds a symmetric wing (root and tip sections; tip leading edge placed
    so that the *quarter chord* has the given sweep; twist in degrees) and
    solves the VLM at each angle of attack (degrees).  Moments are about
    ``x_ref_mac`` of the MAC.  The VLM is inviscid and incompressible: CL and
    CM are scaled by the Helmbold-Diederich ratio CL_alpha(M) / CL_alpha(0)
    (induced drag by its square), CD is the induced drag plus the wing
    profile drag of the built-in model, and stall is not captured (points
    beyond the estimated stall angle carry a warning).

    Args:
        geometry: Wing planform geometry.
        alpha_deg_list: Angles of attack (degrees).
        mach: Free-stream Mach number.
        reynolds: Reynolds number (optional, for profile drag).
        x_ref_mac: Moment reference as a fraction of the MAC.
        warnings: Warnings to attach to every point.

    Returns:
        Wing analysis results at each angle of attack.
    """
    warnings = list(warnings or [])
    pf = _planform(geometry)
    half = geometry.span_m / 2

    def _asb_airfoil(name):
        airfoil = _load_asb_airfoil(name)
        if airfoil.coordinates is None:
            raise ValueError(f"AeroSandbox has no coordinates for airfoil '{name}'")
        return airfoil

    # Tip leading edge from the quarter-chord sweep
    x_tip_le = half * pf["tan_c4"] + 0.25 * (
        geometry.chord_root_m - geometry.chord_tip_m
    )
    wing = asb.Wing(
        name="MainWing",
        symmetric=True,
        xsecs=[
            asb.WingXSec(
                xyz_le=[0.0, 0.0, 0.0],
                chord=geometry.chord_root_m,
                twist=0.0,
                airfoil=_asb_airfoil(geometry.airfoil_root),
            ),
            asb.WingXSec(
                xyz_le=[
                    x_tip_le,
                    half,
                    half * math.tan(math.radians(geometry.dihedral_deg)),
                ],
                chord=geometry.chord_tip_m,
                twist=geometry.twist_deg,  # degrees
                airfoil=_asb_airfoil(geometry.airfoil_tip),
            ),
        ],
    )
    x_ref = pf["x_mac_le"] + x_ref_mac * pf["mac"]
    airplane = asb.Airplane(
        name="Wing",
        xyz_ref=[x_ref, 0.0, 0.0],
        wings=[wing],
        s_ref=pf["S"],
        c_ref=pf["mac"],
        b_ref=geometry.span_m,
    )

    # Profile drag and stall estimate from the built-in model
    model = _wing_model(geometry, alpha_deg_list, mach, reynolds, x_ref_mac)

    # Compressibility: scale the incompressible VLM by the Helmbold-Diederich
    # lift-slope ratio CL_alpha(M) / CL_alpha(0) (3-D Prandtl-Glauert)
    sec_cl_alpha = _blended_section(geometry, 0.0)["cl_alpha"]
    k_mach = _wing_lift_slope(geometry, mach, sec_cl_alpha) / _wing_lift_slope(
        geometry, 0.0, sec_cl_alpha
    )

    results = []
    for i, alpha_deg in enumerate(alpha_deg_list):
        vlm = asb.VortexLatticeMethod(
            airplane=airplane,
            op_point=asb.OperatingPoint(velocity=50.0, alpha=float(alpha_deg)),
            spanwise_resolution=12,
            chordwise_resolution=8,
        )
        vlm_result = vlm.run()

        CL = k_mach * float(np.asarray(vlm_result["CL"]).item())
        CDi = k_mach**2 * float(np.asarray(vlm_result["CD"]).item())
        CM = k_mach * float(np.asarray(vlm_result["Cm"]).item())
        CD = CDi + float(model["cd0"])

        point_warnings = list(warnings)
        if bool(to_numpy(model["stalled"])[i]):
            point_warnings.append(
                f"Beyond the estimated stall angle ({model['alpha_stall_deg']:.1f} deg): "
                "the linear VLM does not capture stall"
            )
        results.append(
            WingAnalysisPoint(
                alpha_deg=float(alpha_deg),
                CL=CL,
                CD=CD,
                CM=CM,
                L_D_ratio=CL / CD if CD > 1e-4 else 0.0,
                span_efficiency=(CL**2 / (PI * pf["AR"] * CDi) if CDi > 1e-9 else None),
                method="aerosandbox_vlm",
                warnings=point_warnings,
            )
        )

    return results


# ===========================================================================
# Airfoil Polar Analysis
# ===========================================================================


def airfoil_polar_analysis(
    airfoil_name: str,
    alpha_deg_list: list[float],
    reynolds: float = 1e6,
    mach: float = 0.1,
) -> list[AirfoilPoint]:
    """
    Generate airfoil polar data.

    Uses NeuralFoil (via AeroSandbox) when installed, otherwise the built-in
    database / thin-airfoil model (``_database_airfoil_polar``).  If NeuralFoil
    fails the reason is logged and attached to the points as a warning.

    Args:
        airfoil_name: Airfoil designation (e.g., "NACA2412"; case-insensitive)
        alpha_deg_list: Angles of attack to analyze
        reynolds: Reynolds number (> 0)
        mach: Mach number (< 0.9)

    Returns:
        List of AirfoilPoint objects with cl, cd, cm (about c/4) data
    """
    warnings = _check_flow(mach, reynolds)
    if len(alpha_deg_list) == 0:
        return []

    if AEROSANDBOX_AVAILABLE:
        try:
            return _aerosandbox_airfoil_polar(
                airfoil_name, alpha_deg_list, reynolds, mach, warnings
            )
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            logger.warning("NeuralFoil failed, using the database model: %s", reason)
            warnings = warnings + [
                f"NeuralFoil failed ({reason}); used the database model"
            ]

    # Use simplified database method
    return _database_airfoil_polar(
        airfoil_name, alpha_deg_list, reynolds, mach, warnings
    )


def _database_airfoil_polar(
    airfoil_name: str,
    alpha_deg_list: list[float],
    reynolds: float,
    mach: float,
    warnings: list[str] | None = None,
) -> list[AirfoilPoint]:
    """Generate an airfoil polar from database / thin-airfoil coefficients.

    Attached flow (Prandtl-Glauert, beta = sqrt(1 - M^2))::

        cl = (cl0 + cl_alpha * alpha) / beta      rounded off to cl_max
        cd = cd0 * (10^6 / Re)^0.2 + 0.01 * cl^2
        cm = cm0 / beta                            (about c/4, constant)

    The lift curve is linear up to a knee, then rounds off parabolically to
    ``cl_max`` at the database stall angle (negative stall mirrors it with
    ``|cl_min| = cl_max - 2 cl0``).  Beyond stall the coefficients blend over
    10 deg into the flat-plate model ``cl = 2 sin a cos a``,
    ``cd = cd0 + 2 sin^2 a``, ``cm = -2 sin a (x_cp - 1/4)`` with the centre of
    pressure moving from c/4 to c/2 at 90 deg.  The Reynolds number affects
    drag only.

    Args:
        airfoil_name: Airfoil name (database entry or NACA 4-digit).
        alpha_deg_list: Angles of attack (degrees).
        reynolds: Reynolds number (> 0).
        mach: Mach number (< 0.9).
        warnings: Warnings to attach to every point.

    Returns:
        Airfoil polar points.
    """
    warnings = list(warnings or [])
    data = get_airfoil_coefficients(airfoil_name)

    # Convert to NumPy arrays
    alphas_deg = np.asarray(alpha_deg_list, dtype=np.float64).reshape(-1)
    alphas = np.radians(alphas_deg)

    # Mach number corrections (Prandtl-Glauert); cl_max is not raised
    beta = math.sqrt(1 - mach**2)
    cl_alpha = data["cl_alpha"] / beta
    alpha_l0 = -data["cl0"] / data["cl_alpha"]
    cl_max = data["cl_max"]
    cl_neg = max(data["cl_max"] - 2 * data["cl0"], 0.5)
    s_stall = math.radians(data["alpha_stall_deg"]) - alpha_l0
    s_stall_neg = s_stall * cl_neg / cl_max

    s = alphas - alpha_l0
    cl_att = np.where(
        s >= 0,
        _rounded_lift_curve(np.abs(s), cl_alpha, cl_max, s_stall),
        -_rounded_lift_curve(np.abs(s), cl_alpha, cl_neg, s_stall_neg),
    )

    # Drag: Reynolds correction on cd0 only, Mach drag rise above M 0.3
    cd0 = data["cd0"] * (1e6 / reynolds) ** 0.2
    if mach > 0.3:
        cd0 = cd0 * (1 + 0.2 * (mach - 0.3) ** 2)
    cd_att = cd0 + 0.01 * cl_att**2

    # Quarter-chord moment is constant in attached flow (thin-airfoil theory)
    cm0 = data["cm0"] / beta

    # Flat-plate post-stall model
    cl_fp = 2 * np.sin(alphas) * np.cos(alphas)
    cd_fp = cd0 + 2 * np.sin(alphas) ** 2
    x_cp = 0.25 + 0.25 * np.minimum(np.abs(alphas) / (PI / 2), 1.0)
    cm_fp = -2 * np.sin(alphas) * (x_cp - 0.25)

    excess = np.maximum(s - s_stall, 0) + np.maximum(-s_stall_neg - s, 0)
    w = _smoothstep(excess / STALL_BLEND_RAD)
    cl = (1 - w) * cl_att + w * cl_fp
    cd = (1 - w) * cd_att + w * np.maximum(cd_fp, cd_att)
    cm = (1 - w) * cm0 + w * cm_fp
    stalled = (s > s_stall) | (s < -s_stall_neg)

    # Convert to output format
    results = []
    alphas_np = to_numpy(alphas_deg)
    cl_np = to_numpy(cl)
    cd_np = to_numpy(cd)
    cm_np = to_numpy(cm)
    stalled_np = to_numpy(stalled)
    for i in range(len(alphas_np)):
        point_warnings = list(warnings)
        if stalled_np[i]:
            point_warnings.append(
                "Beyond stall: flat-plate blend, values are approximate"
            )
        cl_i, cd_i = float(cl_np[i]), float(cd_np[i])
        results.append(
            AirfoilPoint(
                alpha_deg=float(alphas_np[i]),
                cl=cl_i,
                cd=cd_i,
                cm=float(cm_np[i]),
                cl_cd_ratio=cl_i / cd_i if cd_i > 1e-4 else 0.0,
                method="database_thin_airfoil",
                warnings=point_warnings,
            )
        )

    return results


def _aerosandbox_airfoil_polar(
    airfoil_name: str,
    alpha_deg_list: list[float],
    reynolds: float,
    mach: float,
    warnings: list[str] | None = None,
) -> list[AirfoilPoint]:
    """Generate airfoil polar using AeroSandbox's NeuralFoil surrogate model.

    NeuralFoil provides rapid airfoil performance predictions trained on
    XFoil data.  All angles are evaluated in one vectorized call.  Points
    with low NeuralFoil analysis confidence carry a warning.

    Args:
        airfoil_name: NACA or named airfoil designation.
        alpha_deg_list: Angles of attack (degrees).
        reynolds: Reynolds number.
        mach: Mach number.
        warnings: Warnings to attach to every point.

    Returns:
        Airfoil polar points.

    Raises:
        ValueError: AeroSandbox does not know the airfoil.
    """
    warnings = list(warnings or [])
    airfoil = _load_asb_airfoil(airfoil_name)
    if airfoil.coordinates is None:
        raise ValueError(f"AeroSandbox has no coordinates for airfoil '{airfoil_name}'")

    alphas = np.asarray(alpha_deg_list, dtype=np.float64).reshape(-1)
    result = airfoil.get_aero_from_neuralfoil(
        alpha=to_numpy(alphas), Re=reynolds, mach=mach
    )
    n = alphas.size
    cl = np.broadcast_to(np.asarray(result["CL"], dtype=np.float64), (n,))
    cd = np.broadcast_to(np.asarray(result["CD"], dtype=np.float64), (n,))
    cm = np.broadcast_to(np.asarray(result["CM"], dtype=np.float64), (n,))
    confidence = np.broadcast_to(
        np.asarray(result.get("analysis_confidence", 1.0), dtype=np.float64), (n,)
    )

    results = []
    for i in range(n):
        point_warnings = list(warnings)
        if confidence[i] < 0.5:
            point_warnings.append(
                f"Low NeuralFoil analysis confidence ({confidence[i]:.2f})"
            )
        cl_i, cd_i = float(cl[i]), float(cd[i])
        results.append(
            AirfoilPoint(
                alpha_deg=float(alphas[i]),
                cl=cl_i,
                cd=cd_i,
                cm=float(cm[i]),
                cl_cd_ratio=cl_i / cd_i if cd_i > 1e-4 else 0.0,
                method="aerosandbox_neuralfoil",
                warnings=point_warnings,
            )
        )
    return results


# ===========================================================================
# Stability Derivatives
# ===========================================================================


def calculate_stability_derivatives(
    geometry: WingGeometry,
    alpha_deg: float = 2.0,
    mach: float = 0.2,
    x_ref_mac: float = 0.25,
) -> StabilityDerivatives:
    """
    Calculate basic longitudinal stability derivatives of the wing alone.

    ``CL_alpha`` and ``CM_alpha`` are the local slopes (central differences
    of +-0.5 deg) of the lifting-line / DATCOM wing model used by
    ``wing_vlm_analysis``'s fallback, evaluated at ``alpha_deg``: below stall
    CL_alpha is the Helmbold-Diederich slope, near and beyond stall it drops
    (and can become negative).  Moments are about ``x_ref_mac`` of the MAC
    with the wing aerodynamic centre at 0.25 MAC, so in attached flow
    ``CM_alpha = CL_alpha * (x_ref_mac - 0.25)``: zero about the default
    reference, negative (stable) for a reference ahead of the aerodynamic
    centre.  ``CL_alpha_dot`` and ``CM_alpha_dot`` are not computed (None).

    Args:
        geometry: Wing geometry
        alpha_deg: Angle of attack at which the slopes are evaluated
        mach: Mach number (< 0.9)
        x_ref_mac: Moment reference (e.g. the CG) as a fraction of the MAC

    Returns:
        StabilityDerivatives object
    """
    _check_flow(mach)
    delta = 0.5
    model = _wing_model(
        geometry,
        [alpha_deg - delta, alpha_deg, alpha_deg + delta],
        mach,
        None,
        x_ref_mac,
    )
    CL = to_numpy(model["CL"])
    CM = to_numpy(model["CM"])
    d_alpha = math.radians(2 * delta)
    stalled = bool(to_numpy(model["stalled"])[1])

    notes = [
        "Wing alone; moments about x_ref_mac of the MAC (aerodynamic centre "
        "assumed at 0.25 MAC).",
        "CL_alpha_dot and CM_alpha_dot are not computed: for a conventional "
        "aircraft they come mainly from the horizontal tail (downwash lag), "
        "which this model does not include.",
    ]
    if stalled:
        notes.append(
            f"alpha_deg is beyond the estimated stall angle "
            f"({model['alpha_stall_deg']:.1f} deg); slopes are post-stall estimates."
        )

    return StabilityDerivatives(
        CL_alpha=float((CL[2] - CL[0]) / d_alpha),
        CM_alpha=float((CM[2] - CM[0]) / d_alpha),
        CL_alpha_dot=None,
        CM_alpha_dot=None,
        alpha_deg=float(alpha_deg),
        CL=float(CL[1]),
        stalled=stalled,
        x_ref_mac=float(x_ref_mac),
        method="lifting_line_datcom",
        notes=notes,
    )


# ===========================================================================
# Utility Functions
# ===========================================================================


def get_airfoil_database() -> dict[str, dict[str, float]]:
    """Return a copy of the built-in airfoil coefficient database.

    Returns:
        Dictionary mapping airfoil names to linearized coefficients.
    """
    return AIRFOIL_DATABASE.copy()


def estimate_wing_area(geometry: WingGeometry) -> dict[str, float]:
    """Calculate wing planform geometric properties.

    Computes area (trapezoidal), aspect ratio, taper ratio, mean
    aerodynamic chord (MAC), and MAC sweep angle.

    Mean aerodynamic chord formula (trapezoidal wing)::

        MAC = (2/3) * c_root * (1 + lambda + lambda^2) / (1 + lambda)

    where lambda = c_tip / c_root is the taper ratio.

    Args:
        geometry: Wing planform geometry definition.

    Returns:
        Dictionary with ``wing_area_m2``, ``aspect_ratio``,
        ``taper_ratio``, ``mean_aerodynamic_chord_m``,
        ``sweep_MAC_deg``, and ``span_m``.
    """
    # Trapezoidal wing area: S = b * (c_root + c_tip) / 2
    S = geometry.span_m * (geometry.chord_root_m + geometry.chord_tip_m) / 2

    # Aspect ratio: AR = b^2 / S
    AR = geometry.span_m**2 / S

    # Taper ratio: lambda = c_tip / c_root
    taper_ratio = geometry.chord_tip_m / geometry.chord_root_m

    # Mean aerodynamic chord (trapezoidal wing formula)
    MAC = (
        (2 / 3)
        * geometry.chord_root_m
        * (1 + taper_ratio + taper_ratio**2)
        / (1 + taper_ratio)
    )

    # Sweep of mean aerodynamic chord (approximate for trapezoidal wing)
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
