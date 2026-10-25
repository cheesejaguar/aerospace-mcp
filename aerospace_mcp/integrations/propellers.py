"""Propeller and UAV Energy Analysis Tools.

Provides propeller performance analysis using Blade Element Momentum Theory
(BEMT), UAV energy / endurance estimation, and motor-propeller matching.
Everything is implemented here with NumPy; no optional dependency is used
(AeroSandbox 4.x has no propeller model, and PyBEMT is not integrated).

Key capabilities:
    - BEMT propeller analysis (thrust, torque, power, efficiency vs. RPM)
    - UAV energy estimation for multirotor and fixed-wing configurations
    - Motor-propeller matching for optimal operating-point selection
    - Propeller and battery reference databases

References:
    - Leishman, J.G., "Principles of Helicopter Aerodynamics" (2nd ed., 2006)
    - McCormick, B.W., "Aerodynamics, Aeronautics, and Flight Mechanics" (1995)
    - Drela, M., "Flight Vehicle Aerodynamics" (2014)
    - Ning, S.A., "A simple solution method for the blade element momentum
      equations with guaranteed convergence", Wind Energy 17 (2014)
    - Brandt, J.B. & Selig, M.S., "Propeller Performance Data at Low Reynolds
      Numbers", AIAA 2011-1255 (UIUC propeller database)

WARNING: This module is for educational and research purposes only.
Do NOT use for real flight planning, navigation, or aircraft operations.
"""

import logging
import math

import numpy as np
from pydantic import BaseModel, Field

from . import update_availability
from .atmosphere import get_atmosphere_profile

logger = logging.getLogger(__name__)

# ===========================================================================
# Optional Library Imports
# ===========================================================================

# Kept for backwards compatibility.  The propeller analysis no longer uses
# optional libraries: AeroSandbox 4.x has no ``Propeller`` class (the old
# code path always failed and silently fell back), and PyBEMT was never
# actually imported.  Both flags are therefore always False.
AEROSANDBOX_AVAILABLE = False
PYBEMT_AVAILABLE = False

update_availability("propellers", True, {})  # Built-in BEMT, always available

# ===========================================================================
# Physical Constants
# ===========================================================================

# ISA sea-level air density (kg/m^3).
RHO_SEA_LEVEL = 1.225

# Standard gravitational acceleration (m/s^2).  NIST value.
GRAVITY = 9.80665

# ===========================================================================
# BEMT Blade Section and Planform Model
# ===========================================================================

# Section lift-curve slope (per radian): thin-airfoil 2*pi reduced by 10%
# for the thick, low-Reynolds-number sections of small propellers.
SECTION_LIFT_SLOPE = 0.9 * 2 * math.pi

# Section drag polar: cd = cd_design + K * (cl - cl_design)^2 (attached flow).
SECTION_DRAG_K = 0.025

# Stall: cl_max = min(cl_design + margin, 1.8); negative stall at cl = -0.6.
SECTION_CL_MAX_MARGIN = 0.7
SECTION_CL_MAX_LIMIT = 1.8
SECTION_CL_MIN = -0.6

# Angle range (rad) over which the post-stall model blends into the
# flat-plate model (cl = sin 2a, cd = 2 sin^2 a).
SECTION_STALL_BLEND_RAD = math.radians(10.0)

# Default blade planform: chord/radius vs. r/R of a typical small fixed-pitch
# propeller (APC-style: widest near 0.55R, rounded tip).  It is scaled so that
# its activity factor matches PropellerGeometry.activity_factor; unscaled it
# has AF = 118.
PLANFORM_X = np.array(
    [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55]
    + [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
)
PLANFORM_C_OVER_R = np.array(
    [0.109, 0.132, 0.156, 0.176, 0.193, 0.206, 0.216, 0.223, 0.226]
    + [0.225, 0.219, 0.210, 0.197, 0.179, 0.157, 0.130, 0.087, 0.042]
)


def _shape_activity_factor() -> float:
    """Activity factor of the unscaled planform (trapezoidal rule)."""
    xf = np.linspace(0.15, 1.0, 401)
    integrand = 0.5 * np.interp(xf, PLANFORM_X, PLANFORM_C_OVER_R) * xf**3
    integral = float(np.sum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(xf)))
    return 1e5 / 16 * integral


_PLANFORM_ACTIVITY_FACTOR = _shape_activity_factor()

# Default number of radial annuli used by the BEMT solver.
DEFAULT_BEMT_ELEMENTS = 40

# ===========================================================================
# Propeller and Battery Reference Databases
# ===========================================================================

# Common propellers with basic aerodynamic characteristics.
# activity_factor: integrated blade width parameter (dimensionless, ~50-200).
# cl_design / cd_design: blade section lift/drag coefficients at design point.
# efficiency_max: peak propulsive efficiency (eta = J * CT / CP).
PROPELLER_DATABASE = {
    "APC_10x7": {
        "diameter_m": 0.254,  # 10 inches
        "pitch_m": 0.178,  # 7 inches
        "num_blades": 2,
        "activity_factor": 100,
        "cl_design": 0.5,
        "cd_design": 0.02,
        "efficiency_max": 0.82,
    },
    "APC_12x8": {
        "diameter_m": 0.305,  # 12 inches
        "pitch_m": 0.203,  # 8 inches
        "num_blades": 2,
        "activity_factor": 110,
        "cl_design": 0.5,
        "cd_design": 0.02,
        "efficiency_max": 0.84,
    },
    "APC_15x10": {
        "diameter_m": 0.381,  # 15 inches
        "pitch_m": 0.254,  # 10 inches
        "num_blades": 2,
        "activity_factor": 120,
        "cl_design": 0.5,
        "cd_design": 0.02,
        "efficiency_max": 0.85,
    },
    "MULTISTAR_8045": {
        "diameter_m": 0.203,  # 8 inches
        "pitch_m": 0.114,  # 4.5 inches
        "num_blades": 3,
        "activity_factor": 90,
        "cl_design": 0.6,
        "cd_design": 0.025,
        "efficiency_max": 0.75,
    },
}

# Battery characteristics database.
# energy_density_wh_kg: gravimetric energy density at cell level.
# discharge_efficiency: Coulombic efficiency accounting for internal losses.
BATTERY_DATABASE = {
    "LiPo_3S": {
        "nominal_voltage_v": 11.1,
        "energy_density_wh_kg": 150,
        "discharge_efficiency": 0.95,
        "cells": 3,
    },
    "LiPo_4S": {
        "nominal_voltage_v": 14.8,
        "energy_density_wh_kg": 150,
        "discharge_efficiency": 0.95,
        "cells": 4,
    },
    "LiPo_6S": {
        "nominal_voltage_v": 22.2,
        "energy_density_wh_kg": 150,
        "discharge_efficiency": 0.95,
        "cells": 6,
    },
    "Li-Ion_18650": {
        "nominal_voltage_v": 3.7,
        "energy_density_wh_kg": 200,
        "discharge_efficiency": 0.98,
        "cells": 1,
    },
}


# ===========================================================================
# Data Models
# ===========================================================================


class PropellerGeometry(BaseModel):
    """Propeller geometric parameters for BEMT analysis."""

    diameter_m: float = Field(..., gt=0, description="Propeller diameter in meters")
    pitch_m: float = Field(..., gt=0, description="Propeller pitch in meters")
    num_blades: int = Field(..., ge=2, le=6, description="Number of blades")
    hub_radius_m: float | None = Field(
        None,
        ge=0,
        description="Hub radius in meters (default: 0.02 m, at most 0.2R)",
    )
    activity_factor: float = Field(
        100, ge=50, le=200, description="Propeller activity factor (per blade)"
    )
    cl_design: float = Field(0.5, gt=0, le=1.5, description="Design lift coefficient")
    cd_design: float = Field(0.02, gt=0, le=0.1, description="Design drag coefficient")


class PropellerPerformancePoint(BaseModel):
    """Propeller performance at a single operating condition."""

    rpm: float = Field(..., description="Rotational speed in RPM")
    thrust_n: float = Field(..., description="Thrust in Newtons")
    torque_nm: float = Field(..., description="Torque in Newton-meters")
    power_w: float = Field(..., description="Power in Watts")
    efficiency: float = Field(..., description="Propulsive efficiency")
    advance_ratio: float = Field(..., description="Advance ratio J=V/(nD)")
    thrust_coefficient: float = Field(..., description="Thrust coefficient CT")
    power_coefficient: float = Field(..., description="Power coefficient CP")


class UAVConfiguration(BaseModel):
    """UAV configuration parameters."""

    mass_kg: float = Field(..., gt=0, description="Total UAV mass in kg")
    wing_area_m2: float | None = Field(
        None, gt=0, description="Wing area for fixed-wing"
    )
    disk_area_m2: float | None = Field(
        None, gt=0, description="Total rotor disk area for multirotor"
    )
    cd0: float = Field(0.03, ge=0, description="Zero-lift drag coefficient")
    cl_cruise: float | None = Field(
        None,
        description=(
            "Cruise lift coefficient for fixed-wing; sets the cruise speed "
            "when the mission gives no velocity"
        ),
    )
    num_motors: int = Field(1, ge=1, le=8, description="Number of motors/propellers")
    motor_efficiency: float = Field(0.85, gt=0, le=1, description="Motor efficiency")
    esc_efficiency: float = Field(0.95, gt=0, le=1, description="ESC efficiency")
    drag_area_m2: float | None = Field(
        None,
        gt=0,
        description=(
            "Multirotor parasite drag area f = CD * A in m^2 "
            "(default cd0 * 0.1 * mass^(2/3))"
        ),
    )
    avionics_power_w: float | None = Field(
        None,
        ge=0,
        description=(
            "Electrical power of avionics/payload in W "
            "(default 50 W multirotor, 5 W fixed-wing)"
        ),
    )


class BatteryConfiguration(BaseModel):
    """Battery pack configuration."""

    capacity_ah: float = Field(..., gt=0, description="Battery capacity in Amp-hours")
    voltage_nominal_v: float = Field(..., gt=0, description="Nominal voltage")
    mass_kg: float = Field(..., gt=0, description="Battery mass in kg")
    energy_density_wh_kg: float = Field(
        150, gt=0, description="Energy density in Wh/kg"
    )
    discharge_efficiency: float = Field(
        0.95, gt=0, le=1, description="Discharge efficiency"
    )


class EnergyAnalysisResult(BaseModel):
    """UAV energy analysis results."""

    flight_time_min: float = Field(..., description="Estimated flight time in minutes")
    range_km: float | None = Field(
        None, description="Range in km at the cruise speed (None when hovering)"
    )
    hover_time_min: float | None = Field(
        None, description="Hover endurance in minutes (rotorcraft only)"
    )
    power_required_w: float = Field(
        ...,
        description=(
            "Power required at the cruise condition: propulsive shaft power "
            "plus avionics, before motor/ESC losses"
        ),
    )
    energy_consumed_wh: float = Field(..., description="Energy consumed")
    battery_energy_wh: float = Field(..., description="Available battery energy")
    efficiency_overall: float = Field(
        ..., description="Motor x ESC efficiency (propeller losses are in power)"
    )
    aircraft_type: str | None = Field(
        None, description="'fixed_wing', 'multirotor', 'vtol' or 'generic'"
    )
    cruise_velocity_ms: float | None = Field(
        None, description="Airspeed used for the cruise/forward-flight estimate"
    )
    electrical_power_w: float | None = Field(
        None, description="Battery power drawn at the cruise condition"
    )
    hover_power_w: float | None = Field(
        None, description="Hover shaft power plus avionics (rotorcraft only)"
    )
    propeller_efficiency: float | None = Field(
        None, description="Propeller efficiency used (fixed-wing cruise)"
    )
    figure_of_merit: float | None = Field(
        None, description="Rotor figure of merit used (rotorcraft hover)"
    )
    warnings: list[str] = Field(default_factory=list, description="Model warnings")


# ===========================================================================
# Atmosphere
# ===========================================================================


def _air_density(altitude_m: float) -> float:
    """ISA air density (kg/m^3) at a geometric altitude (0-86 km)."""
    return get_atmosphere_profile([float(altitude_m)])[0].density_kg_m3


# ===========================================================================
# BEMT Propeller Analysis
# ===========================================================================


def _section_coefficients(
    alpha: np.ndarray, cl_design: float, cd_design: float
) -> tuple[np.ndarray, np.ndarray]:
    """Blade section lift and drag coefficients (vectorized).

    Attached flow uses a linear lift curve and a parabolic drag polar centred
    on the design point::

        cl = a0 * (alpha - alpha_L0),  a0 = 0.9 * 2*pi
        cd = cd_design + K * (cl - cl_design)^2

    The pitch (blade angle) is measured from the attitude at which the
    section produces ``cl_design``, i.e. ``alpha_L0 = -cl_design / a0``
    (about -5 deg for cl_design = 0.5, typical of the flat-bottomed cambered
    sections of small propellers).  Lift is limited to [-0.6, cl_max] with
    ``cl_max = min(cl_design + 0.7, 1.8)``; beyond stall the coefficients
    blend smoothly (over 10 deg) into the flat-plate model
    ``cl = sin(2 alpha)``, ``cd = cd_design + 2 sin^2(alpha)``.

    Args:
        alpha: Angle of attack array (rad).
        cl_design: Section design lift coefficient.
        cd_design: Section minimum drag coefficient.

    Returns:
        Tuple ``(cl, cd)`` with the shape of ``alpha``.
    """
    a0 = SECTION_LIFT_SLOPE
    alpha_l0 = -cl_design / a0
    cl_max = min(cl_design + SECTION_CL_MAX_MARGIN, SECTION_CL_MAX_LIMIT)
    cl_min = SECTION_CL_MIN

    cl_att = np.clip(a0 * (alpha - alpha_l0), cl_min, cl_max)
    cd_att = cd_design + SECTION_DRAG_K * (cl_att - cl_design) ** 2

    # Flat-plate (fully separated) model
    cl_fp = np.sin(2 * alpha)
    cd_fp = cd_design + 2 * np.sin(alpha) ** 2

    # Smoothstep blend weight past positive / negative stall
    alpha_stall_pos = alpha_l0 + cl_max / a0
    alpha_stall_neg = alpha_l0 + cl_min / a0
    excess = np.maximum(alpha - alpha_stall_pos, 0.0) + np.maximum(
        alpha_stall_neg - alpha, 0.0
    )
    t = np.clip(excess / SECTION_STALL_BLEND_RAD, 0.0, 1.0)
    w = t * t * (3 - 2 * t)

    cl = (1 - w) * cl_att + w * cl_fp
    cd = (1 - w) * cd_att + w * np.maximum(cd_fp, cd_att)
    return cl, cd


def _blade_chord_m(x: np.ndarray, geometry: PropellerGeometry) -> np.ndarray:
    """Blade chord (m) at radial stations x = r/R.

    Uses the default planform shape (``PLANFORM_C_OVER_R``) scaled so that the
    activity factor matches the geometry::

        AF = (10^5 / 16) * integral_{0.15}^{1} (c/D) x^3 dx

    Args:
        x: Non-dimensional radial stations.
        geometry: Propeller geometry (diameter, activity factor).

    Returns:
        Chord in meters at each station.
    """
    scale = geometry.activity_factor / _PLANFORM_ACTIVITY_FACTOR
    radius = geometry.diameter_m / 2
    return scale * radius * np.interp(x, PLANFORM_X, PLANFORM_C_OVER_R)


def _hub_radius_m(geometry: PropellerGeometry) -> float:
    """Hub radius used by the solver.

    A given ``hub_radius_m`` is used as is (it must be below 0.6R).  When it
    is not given, 0.02 m is used, limited to 0.2R so that it stays sensible
    for very small propellers.
    """
    radius = geometry.diameter_m / 2
    if geometry.hub_radius_m is None:
        return min(0.02, 0.2 * radius)
    if geometry.hub_radius_m >= 0.6 * radius:
        raise ValueError(
            f"hub_radius_m ({geometry.hub_radius_m} m) must be smaller than "
            f"0.6 * radius ({0.6 * radius:.4f} m)"
        )
    return geometry.hub_radius_m


def _bemt_solve(
    geometry: PropellerGeometry,
    rpm: np.ndarray,
    velocity_ms: float,
    rho: float,
    n_elements: int = DEFAULT_BEMT_ELEMENTS,
) -> dict[str, np.ndarray]:
    """Solve the blade element momentum equations for several RPMs at once.

    The blade is split into ``n_elements`` annuli (cosine spacing, clustered
    at root and tip) from the hub (at least 0.1R) to the tip.  Blade angle is
    that of a constant-pitch helix, ``beta(r) = atan(p / (2 pi r))``.  At
    each annulus the inflow angle phi satisfies (Ning 2014, written for a
    propeller with induction factors a, a')::

        R(phi) = sin(phi) (1 - k) - (V / (Omega r)) cos(phi) (1 + k') = 0
        k  = sigma' cn / (4 F sin^2 phi),       a  = k / (1 - k)
        k' = sigma' ct / (4 F sin phi cos phi), a' = k' / (1 + k')

    with local solidity sigma' = B c / (2 pi r), cn = cl cos phi - cd sin
    phi, ct = cl sin phi + cd cos phi and the Prandtl tip and hub loss
    factor F.  The residual changes sign on (0, pi/2], so a vectorized
    bisection (60 halvings) converges for every annulus, including the
    static case V = 0 (where it reduces to sigma' cn = 4 F sin^2 phi) and
    the windmill state.  Should an annulus not bracket a root it falls back
    to the uninduced inflow angle and is reported as not converged.

    Args:
        geometry: Propeller geometry.
        rpm: Array of rotational speeds (rev/min), all > 0.
        velocity_ms: Axial free-stream velocity (m/s), >= 0.
        rho: Air density (kg/m^3).
        n_elements: Number of radial annuli.

    Returns:
        Dict of arrays (one entry per RPM): ``thrust_n``, ``torque_nm``,
        ``power_w`` and ``converged`` (bool).
    """
    D = geometry.diameter_m
    R = D / 2
    B = geometry.num_blades
    r_hub = _hub_radius_m(geometry)
    x_root = max(r_hub / R, 0.1)

    # Annulus edges with cosine spacing; elements at the annulus centres
    theta = np.linspace(0.0, math.pi, n_elements + 1)
    edges = x_root + (1 - x_root) * (1 - np.cos(theta)) / 2
    x = 0.5 * (edges[1:] + edges[:-1])
    r = x * R
    dr = np.diff(edges) * R

    chord = _blade_chord_m(x, geometry)
    beta = np.arctan(geometry.pitch_m / (2 * math.pi * r))
    sigma = B * chord / (2 * math.pi * r)

    omega = 2 * math.pi * np.asarray(rpm, dtype=np.float64)[:, None] / 60.0
    vy = omega * r[None, :]  # (n_rpm, n_elements) tangential velocity
    vx_over_vy = velocity_ms / vy

    def _loads(phi: np.ndarray):
        sphi = np.sin(phi)
        cphi = np.cos(phi)
        cl, cd = _section_coefficients(
            beta - phi, geometry.cl_design, geometry.cd_design
        )
        cn = cl * cphi - cd * sphi
        ct = cl * sphi + cd * cphi
        s_abs = np.maximum(np.abs(sphi), 1e-12)
        # Prandtl tip and hub loss factors
        f_tip = B / 2 * (R - r) / (r * s_abs)
        F = 2 / math.pi * np.arccos(np.exp(-np.minimum(f_tip, 50.0)))
        if r_hub > 0:
            f_hub = B / 2 * (r - r_hub) / (r_hub * s_abs)
            F = F * 2 / math.pi * np.arccos(np.exp(-np.minimum(f_hub, 50.0)))
        F = np.maximum(F, 1e-6)
        return sphi, cphi, cn, ct, F

    def _residual(phi: np.ndarray) -> np.ndarray:
        sphi, cphi, cn, ct, F = _loads(phi)
        # sin(phi)(1 - k) - (Vx/Vy) cos(phi)(1 + k'), written without the
        # 1/cos(phi) of k' so that phi = pi/2 is well defined
        return (
            sphi
            - sigma * cn / (4 * F * sphi)
            - vx_over_vy * (cphi + sigma * ct / (4 * F * sphi))
        )

    shape = vy.shape
    lo = np.full(shape, 1e-6)
    hi = np.full(shape, math.pi / 2)
    f_lo = _residual(lo)
    f_hi = _residual(hi)
    bracketed = np.sign(f_lo) != np.sign(f_hi)

    for _ in range(60):
        mid = 0.5 * (lo + hi)
        f_mid = _residual(mid)
        same = np.sign(f_mid) == np.sign(f_lo)
        lo = np.where(same, mid, lo)
        f_lo = np.where(same, f_mid, f_lo)
        hi = np.where(same, hi, mid)

    phi = 0.5 * (lo + hi)
    # Fallback for annuli without a sign change: uninduced inflow angle
    phi = np.where(bracketed, phi, np.arctan2(velocity_ms, vy))

    sphi, cphi, cn, ct, F = _loads(phi)
    # Relative velocity from the tangential component:
    # W = Vy (1 - a') / cos(phi) = Vy / (cos(phi) + sigma ct / (4 F sin(phi)))
    denom = cphi + sigma * ct / (4 * F * np.maximum(sphi, 1e-12))
    W = vy / np.where(np.abs(denom) < 1e-9, 1e-9, denom)

    dT = 0.5 * rho * W**2 * B * chord * cn * dr
    dQ = 0.5 * rho * W**2 * B * chord * ct * r * dr

    thrust = dT.sum(axis=1)
    torque = dQ.sum(axis=1)
    power = torque * omega[:, 0]
    converged = bracketed.all(axis=1)
    if not converged.all():
        logger.warning(
            "BEMT: %d annuli did not bracket a solution; used uninduced inflow",
            int((~bracketed).sum()),
        )
    return {
        "thrust_n": thrust,
        "torque_nm": torque,
        "power_w": power,
        "converged": converged,
    }


def _performance_points(
    geometry: PropellerGeometry,
    rpm_list: list[float],
    velocity_ms: float,
    rho: float,
    n_elements: int = DEFAULT_BEMT_ELEMENTS,
) -> list[PropellerPerformancePoint]:
    """Run the BEMT solver and package the results as performance points."""
    rpm = np.asarray(rpm_list, dtype=np.float64).reshape(-1)
    if rpm.size == 0:
        return []
    sol = _bemt_solve(geometry, rpm, velocity_ms, rho, n_elements)

    D = geometry.diameter_m
    results = []
    for i, rpm_i in enumerate(rpm):
        n = rpm_i / 60.0  # revolutions per second
        thrust = float(sol["thrust_n"][i])
        power = float(sol["power_w"][i])
        J = velocity_ms / (n * D)
        CT = thrust / (rho * n**2 * D**4)
        CP = power / (rho * n**3 * D**5)
        # Propulsive efficiency eta = J CT / CP (zero when static, and when
        # the propeller produces no thrust or windmills)
        efficiency = J * CT / CP if (thrust > 0 and power > 0) else 0.0
        results.append(
            PropellerPerformancePoint(
                rpm=float(rpm_i),
                thrust_n=thrust,
                torque_nm=float(sol["torque_nm"][i]),
                power_w=power,
                efficiency=efficiency,
                advance_ratio=J,
                thrust_coefficient=CT,
                power_coefficient=CP,
            )
        )
    return results


def _validate_operating_conditions(
    rpm_list: list[float], velocity_ms: float, n_elements: int
) -> None:
    """Reject non-physical inputs before running the solver."""
    rpm = np.asarray(rpm_list, dtype=np.float64).reshape(-1)
    if np.any(~np.isfinite(rpm)) or np.any(rpm <= 0):
        raise ValueError("rpm values must be positive")
    if not math.isfinite(velocity_ms) or velocity_ms < 0:
        raise ValueError("velocity_ms must be >= 0 (axial inflow only)")
    if int(n_elements) < 5:
        raise ValueError("n_elements must be at least 5")


def propeller_bemt_analysis(
    geometry: PropellerGeometry,
    rpm_list: list[float],
    velocity_ms: float,
    altitude_m: float = 0,
    n_elements: int = DEFAULT_BEMT_ELEMENTS,
) -> list[PropellerPerformancePoint]:
    """
    Blade Element Momentum Theory propeller analysis.

    Solves the BEM equations on ``n_elements`` radial annuli with Prandtl tip
    and hub losses (see ``_bemt_solve``), using a constant-pitch helical
    blade, the default planform scaled to the activity factor
    (``_blade_chord_m``) and a section model built from ``cl_design`` and
    ``cd_design`` (``_section_coefficients``).  Mach and Reynolds number
    effects on the sections are not modelled.

    Typical results for small propellers: static figure of merit 0.6-0.75,
    peak efficiency 0.6-0.75, zero thrust slightly above J = p/D.

    Args:
        geometry: Propeller geometry parameters
        rpm_list: List of RPM values to analyze (each > 0)
        velocity_ms: Axial forward velocity in m/s (>= 0)
        altitude_m: Geometric altitude for ISA density (0-86000 m)
        n_elements: Number of radial annuli (default 40)

    Returns:
        List of PropellerPerformancePoint objects.  Efficiency is
        ``J * CT / CP`` (zero when static or not producing thrust); thrust and
        power are negative when the propeller windmills.
    """
    _validate_operating_conditions(rpm_list, velocity_ms, n_elements)
    rho = _air_density(altitude_m)
    return _performance_points(geometry, rpm_list, velocity_ms, rho, int(n_elements))


def _propeller_at_thrust(
    geometry: PropellerGeometry, thrust_n: float, velocity_ms: float, rho: float
) -> PropellerPerformancePoint | None:
    """Find the operating point (RPM) at which the propeller gives ``thrust_n``.

    Thrust increases with RPM at fixed airspeed, so the RPM is bracketed on
    a log-spaced grid up to a tip speed of 250 m/s and refined on a linear
    grid.  Returns None if the thrust is out of reach.
    """
    D = geometry.diameter_m
    rpm_max = 250.0 / (math.pi * D) * 60.0
    grid = np.geomspace(rpm_max / 200.0, rpm_max, 48)
    for _ in range(3):
        thrust = _bemt_solve(geometry, grid, velocity_ms, rho)["thrust_n"]
        above = np.nonzero(thrust >= thrust_n)[0]
        if above.size == 0:
            return None
        i = int(above[0])
        if i == 0:
            return _performance_points(geometry, [grid[0]], velocity_ms, rho)[0]
        lo, hi = grid[i - 1], grid[i]
        t_lo, t_hi = thrust[i - 1], thrust[i]
        grid = np.linspace(lo, hi, 24)
    # Linear interpolation inside the final (narrow) bracket
    rpm = lo + (hi - lo) * (thrust_n - t_lo) / max(t_hi - t_lo, 1e-12)
    return _performance_points(geometry, [rpm], velocity_ms, rho)[0]


# ===========================================================================
# UAV Energy Estimation
# ===========================================================================


def uav_energy_estimate(
    uav_config: UAVConfiguration,
    battery_config: BatteryConfiguration,
    mission_profile: dict[str, float],
    propeller_geometry: PropellerGeometry | None = None,
) -> EnergyAnalysisResult:
    """
    Estimate UAV endurance and energy consumption.

    Aircraft type:
        - ``wing_area_m2`` given: fixed-wing cruise (``vtol`` when
          ``disk_area_m2`` is also given; hover time then uses the rotors).
        - only ``disk_area_m2``: multirotor.
        - neither: generic specific-power estimate (forward flight only).

    Power bookkeeping: ``power_required_w`` is the propulsive shaft power
    plus avionics; the battery supplies
    ``shaft / (motor_eff * esc_eff) + avionics``.  Flight time uses the
    cruise condition; ``hover_time_min`` uses hover power (rotorcraft).

    Args:
        uav_config: UAV configuration parameters
        battery_config: Battery configuration
        mission_profile: ``velocity_ms`` (default: from ``cl_cruise`` for a
            fixed-wing, else 15 m/s) and ``altitude_m`` (default 100 m)
        propeller_geometry: Optional propeller (one of ``num_motors``).  When
            given, the BEMT solver supplies the propeller efficiency at the
            fixed-wing cruise thrust and the rotor figure of merit in hover,
            replacing the defaults (0.8 and 0.7).

    Returns:
        EnergyAnalysisResult with flight time and energy analysis
    """
    altitude_m = mission_profile.get("altitude_m", 100.0)
    rho = _air_density(altitude_m)
    weight_n = uav_config.mass_kg * GRAVITY
    warnings: list[str] = []

    velocity_ms = mission_profile.get("velocity_ms")
    if velocity_ms is None:
        if uav_config.wing_area_m2 and uav_config.cl_cruise:
            # Speed at which the given cruise CL supports the weight
            velocity_ms = math.sqrt(
                2 * weight_n / (rho * uav_config.wing_area_m2 * uav_config.cl_cruise)
            )
        else:
            velocity_ms = 15.0
    velocity_ms = float(velocity_ms)
    if not math.isfinite(velocity_ms) or velocity_ms < 0:
        raise ValueError("velocity_ms must be >= 0")

    motor_chain_eff = uav_config.motor_efficiency * uav_config.esc_efficiency
    hover_shaft_w = None
    propeller_efficiency = None
    figure_of_merit = None

    if uav_config.wing_area_m2:
        aircraft_type = "vtol" if uav_config.disk_area_m2 else "fixed_wing"
        default_avionics = 5.0
        shaft_w, propeller_efficiency = _fixed_wing_power_analysis(
            uav_config, velocity_ms, rho, propeller_geometry, warnings
        )
        if uav_config.disk_area_m2:
            hover_shaft_w, figure_of_merit = _multirotor_power_analysis(
                uav_config, 0.0, rho, propeller_geometry, warnings
            )
    elif uav_config.disk_area_m2:
        aircraft_type = "multirotor"
        default_avionics = 50.0
        shaft_w, figure_of_merit = _multirotor_power_analysis(
            uav_config, velocity_ms, rho, propeller_geometry, warnings
        )
        hover_shaft_w, _ = _multirotor_power_analysis(
            uav_config, 0.0, rho, propeller_geometry, warnings
        )
    else:
        aircraft_type = "generic"
        default_avionics = 0.0
        shaft_w = _generic_power_estimate(uav_config, velocity_ms)
        warnings.append(
            "No wing or rotor area given: generic 15 W/kg specific-power estimate"
        )

    avionics_w = (
        uav_config.avionics_power_w
        if uav_config.avionics_power_w is not None
        else default_avionics
    )

    # Avionics power is electrical: it is not divided by motor/ESC efficiency
    electrical_w = shaft_w / motor_chain_eff + avionics_w

    # Battery analysis
    battery_energy_wh = battery_config.capacity_ah * battery_config.voltage_nominal_v
    usable_energy_wh = (
        battery_energy_wh * battery_config.discharge_efficiency * 0.9
    )  # 90% depth of discharge for UAV applications (more aggressive than automotive)

    # Flight time estimation at the cruise condition
    flight_time_hours = usable_energy_wh / electrical_w
    flight_time_min = flight_time_hours * 60

    range_km = velocity_ms * flight_time_hours * 3.6 if velocity_ms > 0 else None

    hover_time_min = None
    hover_power_w = None
    if hover_shaft_w is not None:
        hover_power_w = hover_shaft_w + avionics_w
        hover_time_min = (
            usable_energy_wh / (hover_shaft_w / motor_chain_eff + avionics_w) * 60
        )

    return EnergyAnalysisResult(
        flight_time_min=flight_time_min,
        range_km=range_km,
        hover_time_min=hover_time_min,
        power_required_w=shaft_w + avionics_w,
        energy_consumed_wh=usable_energy_wh,
        battery_energy_wh=battery_energy_wh,
        efficiency_overall=motor_chain_eff,
        aircraft_type=aircraft_type,
        cruise_velocity_ms=velocity_ms,
        electrical_power_w=electrical_w,
        hover_power_w=hover_power_w,
        propeller_efficiency=propeller_efficiency,
        figure_of_merit=figure_of_merit,
        warnings=warnings,
    )


# ===========================================================================
# UAV Power Estimation Models
# ===========================================================================

# Induced power factor kappa (non-uniform inflow, tip losses), Leishman ch. 2.
INDUCED_POWER_FACTOR = 1.15

# Default rotor figure of merit and propeller efficiency.
DEFAULT_FIGURE_OF_MERIT = 0.7
DEFAULT_PROPELLER_EFFICIENCY = 0.8


def _glauert_induced_velocity(v_hover: float, velocity_ms: float) -> float:
    """Solve Glauert's forward-flight inflow ``v_i = v_h^2 / sqrt(V^2 + v_i^2)``.

    Uses Newton iteration on ``f(v) = v^4 + V^2 v^2 - v_h^4``, which has a
    single positive root (``v_h`` in hover, ``~v_h^2 / V`` at high speed).
    """
    if velocity_ms <= 0:
        return v_hover
    vh4 = v_hover**4
    V2 = velocity_ms**2
    v = v_hover
    for _ in range(50):
        f = v**4 + V2 * v**2 - vh4
        df = 4 * v**3 + 2 * V2 * v
        step = f / df
        v = max(v - step, 1e-9)
        if abs(step) < 1e-12 * max(v, 1.0):
            break
    return v


def _multirotor_power_analysis(
    config: UAVConfiguration,
    velocity_ms: float,
    rho: float,
    propeller_geometry: PropellerGeometry | None = None,
    warnings: list[str] | None = None,
) -> tuple[float, float]:
    """Calculate shaft power required for a multirotor (hover or forward flight).

    Momentum theory with Glauert's forward-flight inflow::

        v_h = sqrt(T / (2 rho A)),   v_i = v_h^2 / sqrt(V^2 + v_i^2)
        P   = kappa T v_i + P0 + D V

    T = sqrt(W^2 + D^2) is the rotor thrust; D = 0.5 rho V^2 f is the
    parasite drag with drag area f (``drag_area_m2``, default
    ``cd0 * 0.1 * m^(2/3)``).  The profile power P0 is chosen so that the
    hover power equals the ideal power divided by the figure of merit,
    ``P0 = P_ideal (1/FM - kappa)`` (kappa = 1.15), and is held constant in
    forward flight (its growth with advance ratio needs the rotor tip speed).

    The figure of merit is 0.7 unless ``propeller_geometry`` is given, in
    which case it is taken from a BEMT hover solution at ``W / num_motors``.

    Args:
        config: UAV configuration (mass, disk area, drag coefficient).
        velocity_ms: Forward flight speed (m/s); 0 for hover.
        rho: Air density (kg/m^3).
        propeller_geometry: Optional rotor geometry for the figure of merit.
        warnings: List collecting model warnings.

    Returns:
        Tuple ``(shaft_power_w, figure_of_merit)``.
    """
    weight_n = config.mass_kg * GRAVITY
    disk_area = config.disk_area_m2

    figure_of_merit = DEFAULT_FIGURE_OF_MERIT
    if propeller_geometry is not None:
        rotor_thrust = weight_n / config.num_motors
        point = _propeller_at_thrust(propeller_geometry, rotor_thrust, 0.0, rho)
        if point is None or point.power_w <= 0:
            if warnings is not None:
                warnings.append(
                    "Propeller cannot hover this mass below 250 m/s tip speed; "
                    f"using figure of merit {DEFAULT_FIGURE_OF_MERIT}"
                )
        else:
            rotor_area = math.pi * (propeller_geometry.diameter_m / 2) ** 2
            p_ideal = rotor_thrust**1.5 / math.sqrt(2 * rho * rotor_area)
            figure_of_merit = p_ideal / point.power_w

    # Parasite drag: D = 0.5 * rho * V^2 * f
    drag_area = (
        config.drag_area_m2
        if config.drag_area_m2 is not None
        else config.cd0 * 0.1 * config.mass_kg ** (2 / 3)
    )
    drag_n = 0.5 * rho * velocity_ms**2 * drag_area
    thrust_n = math.hypot(weight_n, drag_n)

    # Hover induced velocity and ideal power from momentum theory
    v_hover = math.sqrt(thrust_n / (2 * rho * disk_area))
    p_ideal_hover = thrust_n * v_hover
    profile_power = p_ideal_hover * max(
        1.0 / figure_of_merit - INDUCED_POWER_FACTOR, 0.0
    )

    v_induced = _glauert_induced_velocity(v_hover, velocity_ms)
    induced_power = INDUCED_POWER_FACTOR * thrust_n * v_induced
    parasite_power = drag_n * velocity_ms

    return induced_power + profile_power + parasite_power, figure_of_merit


def _fixed_wing_power_analysis(
    config: UAVConfiguration,
    velocity_ms: float,
    rho: float,
    propeller_geometry: PropellerGeometry | None = None,
    warnings: list[str] | None = None,
) -> tuple[float, float]:
    """Calculate shaft power required for a fixed-wing aircraft in level cruise.

    Uses a parabolic drag polar CD = CD0 + k * CL^2 with the lift
    coefficient from level flight, CL = 2 W / (rho V^2 S); k = 1/(pi AR e)
    is taken as 0.015 (an efficient UAV wing, AR e ~ 21).  Shaft power is
    P = D V / eta_prop, with eta_prop = 0.8 unless ``propeller_geometry`` is
    given (then from BEMT at thrust D / num_motors).

    Args:
        config: UAV configuration (mass, wing area, CD0).
        velocity_ms: Cruise airspeed (m/s), > 0.
        rho: Air density (kg/m^3).
        propeller_geometry: Optional propeller geometry.
        warnings: List collecting model warnings.

    Returns:
        Tuple ``(shaft_power_w, propeller_efficiency)``.
    """
    if velocity_ms <= 0:
        raise ValueError("A fixed-wing estimate needs velocity_ms > 0")
    weight_n = config.mass_kg * GRAVITY

    # Lift coefficient from level-flight equilibrium: L = W
    cl = 2 * weight_n / (rho * velocity_ms**2 * config.wing_area_m2)
    if warnings is not None:
        if config.cl_cruise and abs(cl / config.cl_cruise - 1) > 0.05:
            warnings.append(
                f"cl_cruise={config.cl_cruise:g} not used: level flight at "
                f"{velocity_ms:.1f} m/s needs CL={cl:.3f}"
            )
        if cl > 1.4:
            warnings.append(
                f"Required CL={cl:.2f} exceeds a typical CL_max of ~1.4 "
                "(airspeed below stall)"
            )

    # Parabolic drag polar: CD = CD0 + k * CL^2
    k = 0.015  # Induced drag factor 1/(pi*AR*e)
    cd = config.cd0 + k * cl**2

    # Drag force: D = 0.5 * rho * V^2 * S * CD
    drag_n = 0.5 * rho * velocity_ms**2 * config.wing_area_m2 * cd

    # Thrust power: P_thrust = D * V
    power_thrust = drag_n * velocity_ms

    propeller_efficiency = DEFAULT_PROPELLER_EFFICIENCY
    if propeller_geometry is not None:
        point = _propeller_at_thrust(
            propeller_geometry, drag_n / config.num_motors, velocity_ms, rho
        )
        if point is None or point.efficiency <= 0:
            if warnings is not None:
                warnings.append(
                    "Propeller cannot produce the cruise thrust; using "
                    f"efficiency {DEFAULT_PROPELLER_EFFICIENCY}"
                )
        else:
            propeller_efficiency = point.efficiency

    return power_thrust / propeller_efficiency, propeller_efficiency


def _generic_power_estimate(config: UAVConfiguration, velocity_ms: float) -> float:
    """Generic power estimate when detailed aerodynamic config is unknown.

    Uses an empirical specific-power scaling (W/kg) with a velocity
    correction factor.  Appropriate only for order-of-magnitude estimates
    of forward flight.

    Args:
        config: UAV configuration (mass only is used).
        velocity_ms: Cruise speed (m/s), > 0.

    Returns:
        Estimated power in Watts.
    """
    if velocity_ms <= 0:
        raise ValueError(
            "The generic estimate needs velocity_ms > 0; give wing_area_m2 or "
            "disk_area_m2 for hover or detailed estimates"
        )
    # Empirical specific power for efficient small fixed-wing UAVs (W/kg)
    specific_power = 15
    base_power = config.mass_kg * specific_power

    # Power scales roughly as V^2 (drag ~ V^2, P = D*V ~ V^3,
    # but propulsive efficiency improves, net ~V^2).
    velocity_factor = (velocity_ms / 15.0) ** 2.0

    return base_power * velocity_factor


def motor_propeller_matching(
    motor_kv: float,
    battery_voltage: float,
    propeller_options: list[str],
    thrust_required_n: float,
    altitude_m: float = 0,
) -> dict[str, any]:
    """
    Analyze motor/propeller combinations for optimal matching.

    Static thrust and power at 50-100% of the loaded no-load speed come from
    the BEMT solver (``propeller_bemt_analysis``).

    Args:
        motor_kv: Motor KV rating (RPM per volt)
        battery_voltage: Battery voltage under load
        propeller_options: List of propeller names from database
        thrust_required_n: Required thrust in Newtons
        altitude_m: Operating altitude

    Returns:
        Dictionary with analysis results for each propeller option
    """
    max_rpm = motor_kv * battery_voltage * 0.85  # Allow for voltage drop under load

    results = {}

    for prop_name in propeller_options:
        if prop_name not in PROPELLER_DATABASE:
            continue

        prop_data = PROPELLER_DATABASE[prop_name]

        # Create geometry object
        geometry = PropellerGeometry(
            diameter_m=prop_data["diameter_m"],
            pitch_m=prop_data["pitch_m"],
            num_blades=prop_data["num_blades"],
            activity_factor=prop_data["activity_factor"],
            cl_design=prop_data["cl_design"],
            cd_design=prop_data["cd_design"],
        )

        # Analyze at different RPM points
        rpm_points = [max_rpm * factor for factor in [0.5, 0.7, 0.85, 1.0]]

        # Static thrust analysis
        performance = propeller_bemt_analysis(geometry, rpm_points, 0.0, altitude_m)

        # Find operating point closest to required thrust
        best_point = None
        min_error = float("inf")

        for point in performance:
            error = abs(point.thrust_n - thrust_required_n)
            if error < min_error:
                min_error = error
                best_point = point

        if best_point:
            results[prop_name] = {
                "geometry": geometry.model_dump(),
                "operating_point": best_point.model_dump(),
                "thrust_error_n": min_error,
                "efficiency": best_point.efficiency,
                "power_required_w": best_point.power_w,
                "rpm_recommended": best_point.rpm,
                "suitable": min_error < thrust_required_n * 0.2,  # Within 20%
            }

    return results


def get_propeller_database() -> dict[str, dict[str, any]]:
    """Return a copy of the built-in propeller reference database.

    Returns:
        Dictionary mapping propeller names to their geometric and
        aerodynamic parameters.
    """
    return PROPELLER_DATABASE.copy()


def get_battery_database() -> dict[str, dict[str, any]]:
    """Return a copy of the built-in battery reference database.

    Returns:
        Dictionary mapping battery type names to their electrical
        characteristics.
    """
    return BATTERY_DATABASE.copy()
