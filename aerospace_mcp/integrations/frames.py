"""Coordinate Frame Transformations.

Provides transformations between coordinate reference frames commonly used
in aerospace applications:
    - **ECEF** (Earth-Centered, Earth-Fixed) -- co-rotates with Earth
    - **ECI / GCRS** (Earth-Centered Inertial / Geocentric Celestial
      Reference System) -- non-rotating, aligned with vernal equinox
    - **ITRF** (International Terrestrial Reference Frame) -- equivalent to
      ECEF for this module
    - **GEODETIC** (latitude, longitude, altitude above WGS-84 ellipsoid)

The ECEF <-> Geodetic conversion uses a fixed-point iteration on the
WGS-84 ellipsoid that stays well conditioned on and near the polar axis.

ECI <-> ECEF accounts for Earth rotation.  When astropy is installed the
full IAU GCRS <-> ITRS reduction is used.  Otherwise the manual path
applies IAU-1976 precession (J2000 mean equator/equinox -> mean of date)
followed by a rotation about the z-axis by GMST (IAU-1982).  It neglects
nutation (up to ~17 arcsec), UT1-UTC (< 0.9 s) and polar motion
(~0.3 arcsec), so it agrees with a full reduction to roughly 1 km at
LEO radius.  In the manual path "GCRS" is treated as "ECI" (J2000 mean
equator and equinox; the ~23 mas GCRS frame bias is ignored) and "ITRF"
as "ECEF" (WGS-84 and ITRF agree at the centimetre level).  GEODETIC
<-> ECI conversions are chained through ECEF.

Uses NumPy for vectorized calculations with CuPy compatibility for GPU
acceleration via the ``_array_backend`` module.

References:
    - NIMA TR8350.2, "Department of Defense World Geodetic System 1984"
    - Vallado, D.A., "Fundamentals of Astrodynamics and Applications"
      (4th ed., 2013), Chapter 3 -- Coordinate systems
    - IAU SOFA Library documentation for precession/nutation
    - Lieske, J.H. et al., "Expressions for the Precession Quantities
      Based upon the IAU (1976) System of Astronomical Constants",
      A&A 58, 1-16 (1977)
    - Hofmann-Wellenhof, B., Lichtenegger, H. & Wasle, E., "GNSS --
      Global Navigation Satellite Systems" (2008) -- iterative
      Cartesian -> ellipsoidal coordinate conversion

WARNING: This module is for educational and research purposes only.
Do NOT use for real flight planning, navigation, or spacecraft operations.
"""

import math
from datetime import UTC, datetime

from pydantic import BaseModel, Field

from . import update_availability
from ._array_backend import np, to_numpy

# ===========================================================================
# WGS-84 Ellipsoid Parameters
# ===========================================================================

# Semi-major axis of the WGS-84 reference ellipsoid.
# Defines the equatorial radius of the Earth.
# Units: meters.
EARTH_A = 6378137.0

# Flattening of the WGS-84 ellipsoid: f = (a - b) / a.
# Quantifies how much the Earth is "squashed" at the poles.
# Dimensionless.
EARTH_F = 1.0 / 298.257223563

# Semi-minor axis (polar radius): b = a * (1 - f).
# Units: meters.
EARTH_B = EARTH_A * (1.0 - EARTH_F)

# First eccentricity squared: e^2 = 2f - f^2.
# Appears in the radius-of-curvature formula: N = a / sqrt(1 - e^2 * sin^2(lat)).
# Dimensionless.
EARTH_E2 = 2.0 * EARTH_F - EARTH_F**2


def _parse_epoch(epoch_iso: str) -> datetime:
    """Parse an ISO-8601 epoch into an aware UTC datetime.

    Naive timestamps are interpreted as UTC; a trailing "Z" is accepted.

    Raises:
        ValueError: If the string is not a valid ISO-8601 timestamp.
    """
    try:
        epoch = datetime.fromisoformat(str(epoch_iso).replace("Z", "+00:00"))
    except ValueError:
        raise ValueError(
            f"Invalid epoch {epoch_iso!r}; expected ISO-8601 such as "
            "'2024-03-20T12:00:00Z'"
        ) from None
    if epoch.tzinfo is None:
        epoch = epoch.replace(tzinfo=UTC)
    return epoch.astimezone(UTC)


def _days_since_j2000(epoch_iso: str) -> float:
    """Days elapsed since J2000.0 (JD 2451545.0) for a UTC epoch string."""
    # Julian Date from Unix time: JD = unix/86400 + 2440587.5
    jd = _parse_epoch(epoch_iso).timestamp() / 86400.0 + 2440587.5
    return jd - 2451545.0


def _gmst_rad(epoch_iso: str) -> float:
    """Greenwich Mean Sidereal Time in radians for an ISO-8601 epoch (UTC).

    Uses the IAU 1982 linear approximation:
        GMST_deg = 280.46061837 + 360.98564736629 * d
    where ``d`` is the number of UT1 days since J2000.0 (JD 2451545.0).
    UT1 is approximated by UTC (|UT1-UTC| < 0.9 s, i.e. < 0.004 deg).

    Args:
        epoch_iso: Epoch as an ISO-8601 string, e.g. "2000-01-01T12:00:00".
            Naive timestamps are interpreted as UTC.

    Returns:
        GMST angle in radians, wrapped to [0, 2*pi).
    """
    d = _days_since_j2000(epoch_iso)
    gmst_deg = (280.46061837 + 360.98564736629 * d) % 360.0
    return math.radians(gmst_deg)


# ===========================================================================
# Optional Library Imports
# ===========================================================================

ASTROPY_AVAILABLE = False
SKYFIELD_AVAILABLE = False

try:
    import astropy
    import astropy.units as u
    from astropy.coordinates import GCRS, ITRS, CartesianRepresentation
    from astropy.time import Time

    ASTROPY_AVAILABLE = True
    update_availability("frames", True, {"astropy": astropy.__version__})
except ImportError:
    pass

try:
    import skyfield

    # from skyfield.api import load, utc  # Available if needed
    # from skyfield.positionlib import position_of_radec  # Available if needed

    SKYFIELD_AVAILABLE = True
    if not ASTROPY_AVAILABLE:  # Only set if astropy not available
        try:
            version = skyfield.__version__
        except AttributeError:
            version = "unknown"
        update_availability("frames", True, {"skyfield": version})
except ImportError:
    if not ASTROPY_AVAILABLE:
        # Frames module is still available with manual calculations
        update_availability("frames", True, {})


# ===========================================================================
# Data Models
# ===========================================================================


class CoordinatePoint(BaseModel):
    """A point in 3-D space with reference frame and epoch metadata."""

    x: float = Field(..., description="X coordinate (m)")
    y: float = Field(..., description="Y coordinate (m)")
    z: float = Field(..., description="Z coordinate (m)")
    frame: str = Field(..., description="Coordinate frame")
    epoch: str | None = Field(None, description="Epoch (ISO format)")


class GeodeticPoint(BaseModel):
    """Geodetic coordinates."""

    latitude_deg: float = Field(..., description="Latitude in degrees")
    longitude_deg: float = Field(..., description="Longitude in degrees")
    altitude_m: float = Field(..., description="Height above ellipsoid (m)")


# ===========================================================================
# ECEF <-> Geodetic Conversion (Fixed-Point Iteration)
# ===========================================================================

# Maximum latitude iterations.  The fixed-point map below contracts by a
# factor of roughly e^2 * N / r per step (~0.0067 near the surface), so
# 1e-14 rad is reached in 6-8 steps; the cap only matters for points deep
# inside the Earth where geodetic coordinates stop being unique.
_GEODETIC_MAX_ITER = 30


def _manual_ecef_to_geodetic(
    x: float, y: float, z: float
) -> tuple[float, float, float]:
    """Convert ECEF Cartesian to geodetic coordinates (iterative method).

    Algorithm (fixed-point iteration on latitude):
        1. Compute longitude directly: lon = atan2(y, x).
        2. Compute distance from the Z-axis: p = sqrt(x^2 + y^2).
        3. Iterate on latitude:
           - N = a / sqrt(1 - e^2 * sin^2(lat))   (radius of curvature)
           - lat = atan2(z + e^2 * N * sin(lat), p)
        4. Height from the converged latitude, valid at every latitude
           (no division by cos(lat), so the poles are not singular):
           - h = p * cos(lat) + z * sin(lat) - a * sqrt(1 - e^2 * sin^2(lat))

    Args:
        x: ECEF X coordinate in meters.
        y: ECEF Y coordinate in meters.
        z: ECEF Z coordinate in meters.

    Returns:
        Tuple of ``(latitude_deg, longitude_deg, altitude_m)``.
    """
    # Longitude is computed directly (no iteration needed)
    lon_rad = math.atan2(y, x)

    # Distance from the Z-axis (projection onto equatorial plane)
    p = math.hypot(x, y)

    # Initial guess for geodetic latitude (h = 0 approximation)
    lat_rad = math.atan2(z, p * (1.0 - EARTH_E2))

    for _ in range(_GEODETIC_MAX_ITER):
        sin_lat = math.sin(lat_rad)
        # Radius of curvature in the prime vertical: N = a / sqrt(1 - e^2*sin^2(lat))
        N = EARTH_A / math.sqrt(1.0 - EARTH_E2 * sin_lat**2)
        lat_rad_new = math.atan2(z + EARTH_E2 * N * sin_lat, p)
        converged = abs(lat_rad_new - lat_rad) < 1e-14
        lat_rad = lat_rad_new
        if converged:
            break

    sin_lat = math.sin(lat_rad)
    # Height above the ellipsoid (pole-safe form)
    alt = (
        p * math.cos(lat_rad)
        + z * sin_lat
        - EARTH_A * math.sqrt(1.0 - EARTH_E2 * sin_lat**2)
    )

    return math.degrees(lat_rad), math.degrees(lon_rad), alt


def _manual_ecef_to_geodetic_vectorized(
    x: np.ndarray, y: np.ndarray, z: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorized ECEF to geodetic conversion for multiple points.

    Same fixed-point iteration as :func:`_manual_ecef_to_geodetic`.
    Returns (lat_deg, lon_deg, alt_m) as arrays.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)

    # Longitude
    lon_rad = np.arctan2(y, x)

    # Distance from z-axis
    p = np.sqrt(x**2 + y**2)

    # Initial guess for latitude
    lat_rad = np.arctan2(z, p * (1.0 - EARTH_E2))

    # Iterative solution: lat = atan2(z + e^2 * N * sin(lat), p)
    for _ in range(_GEODETIC_MAX_ITER):
        sin_lat = np.sin(lat_rad)
        N = EARTH_A / np.sqrt(1.0 - EARTH_E2 * sin_lat**2)
        lat_rad_new = np.arctan2(z + EARTH_E2 * N * sin_lat, p)
        converged = bool(np.all(np.abs(lat_rad_new - lat_rad) < 1e-14))
        lat_rad = lat_rad_new
        if converged:
            break

    # Height above the ellipsoid (pole-safe form)
    sin_lat = np.sin(lat_rad)
    alt = (
        p * np.cos(lat_rad)
        + z * sin_lat
        - EARTH_A * np.sqrt(1.0 - EARTH_E2 * sin_lat**2)
    )

    return np.degrees(lat_rad), np.degrees(lon_rad), alt


def _manual_geodetic_to_ecef(
    lat_deg: float, lon_deg: float, alt_m: float
) -> tuple[float, float, float]:
    """Convert geodetic coordinates to ECEF Cartesian.

    Uses the WGS-84 ellipsoid formulas::

        x = (N + h) * cos(lat) * cos(lon)
        y = (N + h) * cos(lat) * sin(lon)
        z = (N * (1 - e^2) + h) * sin(lat)

    where N = a / sqrt(1 - e^2 * sin^2(lat)) is the radius of curvature
    in the prime vertical.

    Args:
        lat_deg: Geodetic latitude in degrees.
        lon_deg: Geodetic longitude in degrees.
        alt_m: Height above the WGS-84 ellipsoid in meters.

    Returns:
        Tuple of ``(x, y, z)`` ECEF coordinates in meters.
    """
    lat_rad = np.radians(lat_deg)
    lon_rad = np.radians(lon_deg)

    sin_lat = float(np.sin(lat_rad))
    cos_lat = float(np.cos(lat_rad))
    sin_lon = float(np.sin(lon_rad))
    cos_lon = float(np.cos(lon_rad))

    # Radius of curvature in the prime vertical
    # N = a / sqrt(1 - e^2 * sin^2(lat))
    N = EARTH_A / float(np.sqrt(1.0 - EARTH_E2 * sin_lat**2))

    # ECEF coordinates from geodetic
    x = (N + alt_m) * cos_lat * cos_lon
    y = (N + alt_m) * cos_lat * sin_lon
    z = (N * (1.0 - EARTH_E2) + alt_m) * sin_lat

    return x, y, z


def _manual_geodetic_to_ecef_vectorized(
    lat_deg: np.ndarray, lon_deg: np.ndarray, alt_m: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorized geodetic to ECEF conversion.
    Returns (x, y, z) arrays in meters.
    """
    lat_deg = np.asarray(lat_deg)
    lon_deg = np.asarray(lon_deg)
    alt_m = np.asarray(alt_m)

    lat_rad = np.radians(lat_deg)
    lon_rad = np.radians(lon_deg)

    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    sin_lon = np.sin(lon_rad)
    cos_lon = np.cos(lon_rad)

    # Radius of curvature in prime vertical
    N = EARTH_A / np.sqrt(1.0 - EARTH_E2 * sin_lat**2)

    x = (N + alt_m) * cos_lat * cos_lon
    y = (N + alt_m) * cos_lat * sin_lon
    z = (N * (1.0 - EARTH_E2) + alt_m) * sin_lat

    return x, y, z


# ===========================================================================
# Frame Transformation Functions
# ===========================================================================


#: Default epoch used when a time-dependent transform is requested without
#: one (J2000.0).
DEFAULT_EPOCH = "2000-01-01T12:00:00"

#: Accepted frame names.  "ITRF" is an alias of "ECEF" and "GCRS" of "ECI"
#: in the manual (non-astropy) path; see the module docstring.
VALID_FRAMES = {"ECEF", "ECI", "ITRF", "GCRS", "GEODETIC"}

_FRAME_ALIASES = {"ITRF": "ECEF", "GCRS": "ECI"}

_ARCSEC_TO_RAD = math.pi / (180.0 * 3600.0)


def _rot3(angle: float) -> list[list[float]]:
    """Passive (frame) rotation matrix about the z-axis by ``angle`` radians."""
    c, s = math.cos(angle), math.sin(angle)
    return [[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]]


def _rot2(angle: float) -> list[list[float]]:
    """Passive (frame) rotation matrix about the y-axis by ``angle`` radians."""
    c, s = math.cos(angle), math.sin(angle)
    return [[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]]


def _matmul3(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    """Product of two 3x3 matrices given as nested lists."""
    return [
        [sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)
    ]


def _matvec3(m: list[list[float]], v: tuple[float, float, float]) -> list[float]:
    """Product of a 3x3 matrix and a 3-vector."""
    return [m[i][0] * v[0] + m[i][1] * v[1] + m[i][2] * v[2] for i in range(3)]


def _transpose3(m: list[list[float]]) -> list[list[float]]:
    """Transpose of a 3x3 matrix (the inverse of a rotation matrix)."""
    return [[m[j][i] for j in range(3)] for i in range(3)]


def _precession_matrix(epoch_iso: str) -> list[list[float]]:
    """IAU-1976 precession matrix from J2000 mean equator/equinox to mean of date.

    Uses the Lieske (1977) precession angles (Vallado 2013, Eq. 3-88)::

        zeta  = 2306.2181" T + 0.30188" T^2 + 0.017998" T^3
        theta = 2004.3109" T - 0.42665" T^2 - 0.041833" T^3
        z     = 2306.2181" T + 1.09468" T^2 + 0.018203" T^3

    with ``T`` in Julian centuries of TT since J2000.0 (UTC is used for TT;
    the ~69 s difference is negligible here), and::

        r_MOD = R3(-z) * R2(theta) * R3(-zeta) * r_J2000

    Args:
        epoch_iso: Epoch as an ISO-8601 string (UTC).

    Returns:
        3x3 rotation matrix as nested lists.
    """
    T = _days_since_j2000(epoch_iso) / 36525.0
    zeta = (2306.2181 * T + 0.30188 * T**2 + 0.017998 * T**3) * _ARCSEC_TO_RAD
    theta = (2004.3109 * T - 0.42665 * T**2 - 0.041833 * T**3) * _ARCSEC_TO_RAD
    z = (2306.2181 * T + 1.09468 * T**2 + 0.018203 * T**3) * _ARCSEC_TO_RAD
    return _matmul3(_matmul3(_rot3(-z), _rot2(theta)), _rot3(-zeta))


def _eci_to_ecef_matrix(epoch_iso: str) -> list[list[float]]:
    """Approximate ECI (J2000) -> ECEF rotation: R3(GMST) * P(epoch).

    Nutation, the equation of the equinoxes, UT1-UTC and polar motion are
    neglected (see the module docstring for the resulting accuracy).
    """
    return _matmul3(_rot3(_gmst_rad(epoch_iso)), _precession_matrix(epoch_iso))


def _coerce_xyz(
    xyz: list[float] | tuple[float, ...] | dict, from_frame: str
) -> tuple[float, float, float]:
    """Normalise coordinate input to an ``(x, y, z)`` tuple of floats.

    Accepted forms:
        - a sequence of three numbers ``[x, y, z]`` (for GEODETIC:
          ``[latitude_deg, longitude_deg, altitude_m]``);
        - a dict ``{"x": .., "y": .., "z": ..}`` (``x_m``/``y_m``/``z_m``
          are accepted as well);
        - for GEODETIC input, a dict ``{"latitude_deg": .., "longitude_deg":
          .., "altitude_m": ..}`` (``altitude_m`` defaults to 0).

    Raises:
        ValueError: If the input has the wrong shape or non-finite values.
    """
    if isinstance(xyz, dict):
        if from_frame == "GEODETIC" and "latitude_deg" in xyz:
            if "longitude_deg" not in xyz:
                raise ValueError("GEODETIC coordinates need 'longitude_deg'")
            values = (
                xyz["latitude_deg"],
                xyz["longitude_deg"],
                xyz.get("altitude_m", 0.0),
            )
        elif all(k in xyz for k in ("x", "y", "z")):
            values = (xyz["x"], xyz["y"], xyz["z"])
        elif all(k in xyz for k in ("x_m", "y_m", "z_m")):
            values = (xyz["x_m"], xyz["y_m"], xyz["z_m"])
        else:
            expected = "{'x', 'y', 'z'}"
            if from_frame == "GEODETIC":
                expected += " or {'latitude_deg', 'longitude_deg', 'altitude_m'}"
            raise ValueError(
                f"coordinates dict must contain keys {expected}; got {sorted(xyz)}"
            )
    else:
        try:
            values = tuple(xyz)
        except TypeError:
            raise ValueError("xyz must be a list of 3 coordinates") from None
        if len(values) != 3:
            raise ValueError("xyz must be a list of 3 coordinates")

    try:
        out = tuple(float(v) for v in values)
    except (TypeError, ValueError):
        raise ValueError(f"coordinates must be numeric, got {list(values)}") from None
    if not all(math.isfinite(v) for v in out):
        raise ValueError(f"coordinates must be finite, got {list(out)}")
    return out  # type: ignore[return-value]


def transform_frames(
    xyz: list[float] | dict,
    from_frame: str,
    to_frame: str,
    epoch_iso: str | None = DEFAULT_EPOCH,
) -> CoordinatePoint:
    """
    Transform coordinates between reference frames.

    Uses NumPy for efficient calculations.

    Args:
        xyz: [x, y, z] coordinates in meters, or a dict (see
            :func:`_coerce_xyz`).  For GEODETIC input the three values are
            latitude (deg), longitude (deg) and altitude (m).
        from_frame: Source frame ("ECEF", "ECI", "ITRF", "GCRS", "GEODETIC")
        to_frame: Target frame ("ECEF", "ECI", "ITRF", "GCRS", "GEODETIC")
        epoch_iso: Reference epoch in ISO format (UTC).  ``None`` selects
            J2000.0; only conversions involving ECI/GCRS depend on it.

    Returns:
        CoordinatePoint with transformed coordinates.  For a GEODETIC
        target, ``x``/``y``/``z`` hold latitude (deg), longitude (deg) and
        altitude (m).
    """
    # Validate frame names
    if from_frame not in VALID_FRAMES or to_frame not in VALID_FRAMES:
        raise ValueError(f"Frame must be one of {VALID_FRAMES}")

    x, y, z = _coerce_xyz(xyz, from_frame)
    if epoch_iso is None:
        epoch_iso = DEFAULT_EPOCH

    # Same frame - no transformation needed
    if from_frame == to_frame:
        return CoordinatePoint(x=x, y=y, z=z, frame=to_frame, epoch=epoch_iso)

    # GEODETIC is defined on the Earth-fixed ellipsoid: chain through ECEF.
    if from_frame == "GEODETIC":
        if not -90.0 <= x <= 90.0:
            raise ValueError("Latitude must be between -90 and +90 degrees")
        ecef = _manual_geodetic_to_ecef(x, y, z)
        if to_frame in ("ECEF", "ITRF"):
            x_e, y_e, z_e = ecef
            return CoordinatePoint(x=x_e, y=y_e, z=z_e, frame=to_frame, epoch=epoch_iso)
        return transform_frames(list(ecef), "ECEF", to_frame, epoch_iso)

    if to_frame == "GEODETIC":
        if from_frame in ("ECI", "GCRS"):
            ecef_pt = transform_frames([x, y, z], from_frame, "ECEF", epoch_iso)
            x, y, z = ecef_pt.x, ecef_pt.y, ecef_pt.z
        lat, lon, alt = _manual_ecef_to_geodetic(x, y, z)
        return CoordinatePoint(x=lat, y=lon, z=alt, frame="GEODETIC", epoch=epoch_iso)

    # From here both frames are Cartesian: ECI/GCRS (inertial) or ECEF/ITRF.
    # Use high-precision libraries if available
    if ASTROPY_AVAILABLE:
        try:
            # Parse epoch (astropy wants a naive UTC timestamp)
            time = Time(_parse_epoch(epoch_iso).replace(tzinfo=None), scale="utc")

            # Map frame names to astropy
            frame_map = {
                "ECI": GCRS,
                "GCRS": GCRS,
                "ECEF": ITRS,
                "ITRF": ITRS,
            }

            # Create coordinate object
            coords_from = frame_map[from_frame](
                CartesianRepresentation(x=x * u.m, y=y * u.m, z=z * u.m),
                obstime=time,
            )

            # Transform
            coords_to = coords_from.transform_to(frame_map[to_frame](obstime=time))

            return CoordinatePoint(
                x=float(coords_to.cartesian.x.to(u.m).value),
                y=float(coords_to.cartesian.y.to(u.m).value),
                z=float(coords_to.cartesian.z.to(u.m).value),
                frame=to_frame,
                epoch=epoch_iso,
            )
        except Exception:
            # Fall back to manual methods
            pass

    # Manual path: collapse the aliases (ITRF == ECEF, GCRS == ECI).
    src = _FRAME_ALIASES.get(from_frame, from_frame)
    dst = _FRAME_ALIASES.get(to_frame, to_frame)
    if src == dst:
        out = [x, y, z]
    elif src == "ECI":
        # ECI -> ECEF: precession to mean-of-date, then rotate by GMST.
        out = _matvec3(_eci_to_ecef_matrix(epoch_iso), (x, y, z))
    else:
        # ECEF -> ECI: inverse (transpose) of the rotation above.
        out = _matvec3(_transpose3(_eci_to_ecef_matrix(epoch_iso)), (x, y, z))

    return CoordinatePoint(
        x=float(out[0]),
        y=float(out[1]),
        z=float(out[2]),
        frame=to_frame,
        epoch=epoch_iso,
    )


def transform_frames_batch(
    xyz_list: list[list[float]],
    from_frame: str,
    to_frame: str,
    epoch_iso: str | None = DEFAULT_EPOCH,
) -> list[CoordinatePoint]:
    """
    Batch transform multiple coordinates between reference frames.

    Uses vectorized NumPy calculations for efficiency.

    Args:
        xyz_list: List of [x, y, z] coordinates in meters
        from_frame: Source frame
        to_frame: Target frame
        epoch_iso: Reference epoch in ISO format (``None`` selects J2000.0)

    Returns:
        List of CoordinatePoint with transformed coordinates
    """
    if not xyz_list:
        return []

    # Validate frame names
    if from_frame not in VALID_FRAMES or to_frame not in VALID_FRAMES:
        raise ValueError(f"Frame must be one of {VALID_FRAMES}")
    if epoch_iso is None:
        epoch_iso = DEFAULT_EPOCH

    # Convert to NumPy arrays
    coords = np.asarray(xyz_list, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("xyz_list must be a list of [x, y, z] triples")
    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]

    # Same frame - no transformation needed
    if from_frame == to_frame:
        return [
            CoordinatePoint(
                x=float(xi), y=float(yi), z=float(zi), frame=to_frame, epoch=epoch_iso
            )
            for xi, yi, zi in zip(to_numpy(x), to_numpy(y), to_numpy(z), strict=False)
        ]

    # Vectorized transformations for supported cases (ITRF == ECEF here)
    if from_frame in ("ECEF", "ITRF") and to_frame == "GEODETIC":
        lat, lon, alt = _manual_ecef_to_geodetic_vectorized(x, y, z)
        return [
            CoordinatePoint(
                x=float(lati),
                y=float(loni),
                z=float(alti),
                frame="GEODETIC",
                epoch=epoch_iso,
            )
            for lati, loni, alti in zip(
                to_numpy(lat), to_numpy(lon), to_numpy(alt), strict=False
            )
        ]

    elif from_frame == "GEODETIC" and to_frame in ("ECEF", "ITRF"):
        if bool(np.any(np.abs(x) > 90.0)):
            raise ValueError("Latitude must be between -90 and +90 degrees")
        x_new, y_new, z_new = _manual_geodetic_to_ecef_vectorized(x, y, z)
        return [
            CoordinatePoint(
                x=float(xi), y=float(yi), z=float(zi), frame=to_frame, epoch=epoch_iso
            )
            for xi, yi, zi in zip(
                to_numpy(x_new), to_numpy(y_new), to_numpy(z_new), strict=False
            )
        ]

    # Fall back to single-point transformation for other cases
    return [
        transform_frames(
            [float(xi), float(yi), float(zi)], from_frame, to_frame, epoch_iso
        )
        for xi, yi, zi in zip(to_numpy(x), to_numpy(y), to_numpy(z), strict=False)
    ]


def ecef_to_geodetic(x: float, y: float, z: float) -> GeodeticPoint:
    """
    Convert ECEF coordinates to geodetic (WGS84).

    Uses NumPy for efficient calculations.

    Args:
        x, y, z: ECEF coordinates in meters

    Returns:
        GeodeticPoint with latitude, longitude, altitude
    """
    x, y, z = _coerce_xyz([x, y, z], "ECEF")
    lat, lon, alt = _manual_ecef_to_geodetic(x, y, z)

    return GeodeticPoint(latitude_deg=lat, longitude_deg=lon, altitude_m=alt)


def geodetic_to_ecef(
    latitude_deg: float, longitude_deg: float, altitude_m: float
) -> CoordinatePoint:
    """
    Convert geodetic coordinates to ECEF.

    Uses NumPy for efficient calculations.

    Args:
        latitude_deg: Latitude in degrees (-90 to +90)
        longitude_deg: Longitude in degrees (-180 to +180)
        altitude_m: Height above WGS84 ellipsoid in meters

    Returns:
        CoordinatePoint with ECEF coordinates
    """
    latitude_deg, longitude_deg, altitude_m = _coerce_xyz(
        [latitude_deg, longitude_deg, altitude_m], "GEODETIC"
    )
    if not (-90 <= latitude_deg <= 90):
        raise ValueError("Latitude must be between -90 and +90 degrees")
    if not (-180 <= longitude_deg <= 180):
        raise ValueError("Longitude must be between -180 and +180 degrees")

    x, y, z = _manual_geodetic_to_ecef(latitude_deg, longitude_deg, altitude_m)

    return CoordinatePoint(x=x, y=y, z=z, frame="ECEF")


def get_frame_info() -> dict[str, any]:
    """
    Get information about available coordinate frames and capabilities.

    Returns:
        Dictionary with frame information and library status
    """
    return {
        "available_frames": ["ECEF", "ECI", "ITRF", "GCRS", "GEODETIC"],
        "high_precision_available": ASTROPY_AVAILABLE,
        "libraries": {
            "astropy": ASTROPY_AVAILABLE,
            "skyfield": SKYFIELD_AVAILABLE,
        },
        "manual_transforms": [
            "ECEF <-> GEODETIC",
            "ECI <-> ECEF (approximate)",
            "GEODETIC <-> ECI (via ECEF)",
        ],
        "frame_aliases": dict(_FRAME_ALIASES),
        "vectorized_batch": True,
        "notes": [
            "High-precision ECI/GCRS <-> ECEF/ITRF transformations require astropy",
            "Without astropy, ECI <-> ECEF applies IAU-1976 precession and a "
            "GMST (IAU-1982) Earth-rotation angle; nutation, UT1-UTC and polar "
            "motion are neglected (~1 km at LEO radius)",
            "Without astropy, GCRS is treated as ECI (J2000) and ITRF as ECEF (WGS-84)",
            "ECEF <-> GEODETIC uses the WGS-84 ellipsoid (exact to < 1 mm)",
            "Batch operations use vectorized NumPy for efficiency",
        ],
    }
