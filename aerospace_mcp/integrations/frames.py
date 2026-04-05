"""Coordinate Frame Transformations.

Provides transformations between coordinate reference frames commonly used
in aerospace applications:
    - **ECEF** (Earth-Centered, Earth-Fixed) -- co-rotates with Earth
    - **ECI / GCRS** (Earth-Centered Inertial / Geocentric Celestial
      Reference System) -- non-rotating, aligned with vernal equinox
    - **ITRF** (International Terrestrial Reference Frame) -- equivalent to
      ECEF for this module
    - **GEODETIC** (latitude, longitude, altitude above WGS-84 ellipsoid)

The ECEF <-> Geodetic conversion uses an iterative algorithm based on the
WGS-84 ellipsoid parameters.  ECI <-> ECEF requires Earth rotation
correction via GMST (Greenwich Mean Sidereal Time); this module provides
a simplified (identity) approximation when astropy is not available.

Uses NumPy for vectorized calculations with CuPy compatibility for GPU
acceleration via the ``_array_backend`` module.

References:
    - NIMA TR8350.2, "Department of Defense World Geodetic System 1984"
    - Vallado, D.A., "Fundamentals of Astrodynamics and Applications"
      (4th ed., 2013), Chapter 3 -- Coordinate systems
    - IAU SOFA Library documentation for precession/nutation

WARNING: This module is for educational and research purposes only.
Do NOT use for real flight planning, navigation, or spacecraft operations.
"""

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
# ECEF <-> Geodetic Conversion (Iterative Bowring Method)
# ===========================================================================


def _manual_ecef_to_geodetic(
    x: float, y: float, z: float
) -> tuple[float, float, float]:
    """Convert ECEF Cartesian to geodetic coordinates (iterative method).

    Algorithm (Bowring's iterative method):
        1. Compute longitude directly: lon = atan2(y, x).
        2. Compute distance from the Z-axis: p = sqrt(x^2 + y^2).
        3. Iterate on latitude using:
           - N = a / sqrt(1 - e^2 * sin^2(lat))   (radius of curvature)
           - alt = p / cos(lat) - N
           - lat = atan2(z, p * (1 - e^2 * N / (N + alt)))
        4. Convergence typically in 2-3 iterations (tolerance 1e-12 rad).

    Args:
        x: ECEF X coordinate in meters.
        y: ECEF Y coordinate in meters.
        z: ECEF Z coordinate in meters.

    Returns:
        Tuple of ``(latitude_deg, longitude_deg, altitude_m)``.
    """
    # Longitude is computed directly (no iteration needed)
    lon_rad = float(np.arctan2(y, x))

    # Distance from the Z-axis (projection onto equatorial plane)
    p = float(np.sqrt(x**2 + y**2))

    # Initial guess for geodetic latitude (spherical approximation)
    lat_rad = float(np.arctan2(z, p * (1.0 - EARTH_E2)))

    # Iterative refinement (Bowring's method -- converges in 2-3 steps)
    for _ in range(10):
        sin_lat = np.sin(lat_rad)
        # Radius of curvature in the prime vertical: N = a / sqrt(1 - e^2*sin^2(lat))
        N = EARTH_A / float(np.sqrt(1.0 - EARTH_E2 * sin_lat**2))
        # Altitude above the ellipsoid: h = p / cos(lat) - N
        alt = p / float(np.cos(lat_rad)) - N
        # Updated latitude incorporating the altitude correction
        lat_rad_new = float(np.arctan2(z, p * (1.0 - EARTH_E2 * N / (N + alt))))

        if abs(lat_rad_new - lat_rad) < 1e-12:
            break
        lat_rad = lat_rad_new

    return float(np.degrees(lat_rad)), float(np.degrees(lon_rad)), alt


def _manual_ecef_to_geodetic_vectorized(
    x: np.ndarray, y: np.ndarray, z: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorized ECEF to geodetic conversion for multiple points.
    Returns (lat_deg, lon_deg, alt_m) as arrays.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)

    # Longitude
    lon_rad = np.arctan2(y, x)

    # Distance from z-axis
    p = np.sqrt(x**2 + y**2)

    # Initial guess for latitude
    lat_rad = np.arctan2(z, p * (1.0 - EARTH_E2))

    # Iterative solution
    for _ in range(10):
        sin_lat = np.sin(lat_rad)
        N = EARTH_A / np.sqrt(1.0 - EARTH_E2 * sin_lat**2)
        cos_lat = np.cos(lat_rad)
        # Avoid division by zero at poles
        alt = np.where(np.abs(cos_lat) > 1e-10, p / cos_lat - N, np.abs(z) - EARTH_B)
        lat_rad_new = np.arctan2(z, p * (1.0 - EARTH_E2 * N / (N + alt)))

        if np.all(np.abs(lat_rad_new - lat_rad) < 1e-12):
            break
        lat_rad = lat_rad_new

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


def _simple_precession_matrix(epoch1: str, epoch2: str) -> list[list[float]]:
    """Compute a simple precession rotation matrix between two epochs.

    This is a placeholder that returns the identity matrix.  A real
    implementation would use IAU 2006/2000A precession theory to compute
    the 3x3 rotation matrix accounting for equinox precession between
    the two epochs.

    Args:
        epoch1: Source epoch in ISO-8601 format.
        epoch2: Target epoch in ISO-8601 format.

    Returns:
        3x3 rotation matrix as nested lists (currently identity).
    """
    # Parse epochs (assume they're close to J2000)
    # This is a placeholder - real implementation would use proper precession theory

    # Identity matrix for now - would implement IAU precession in real version
    return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


def transform_frames(
    xyz: list[float],
    from_frame: str,
    to_frame: str,
    epoch_iso: str = "2000-01-01T12:00:00",
) -> CoordinatePoint:
    """
    Transform coordinates between reference frames.

    Uses NumPy for efficient calculations.

    Args:
        xyz: [x, y, z] coordinates in meters
        from_frame: Source frame ("ECEF", "ECI", "ITRF", "GCRS")
        to_frame: Target frame ("ECEF", "ECI", "ITRF", "GCRS")
        epoch_iso: Reference epoch in ISO format

    Returns:
        CoordinatePoint with transformed coordinates
    """
    if len(xyz) != 3:
        raise ValueError("xyz must be a list of 3 coordinates")

    x, y, z = xyz

    # Validate frame names
    valid_frames = {"ECEF", "ECI", "ITRF", "GCRS", "GEODETIC"}
    if from_frame not in valid_frames or to_frame not in valid_frames:
        raise ValueError(f"Frame must be one of {valid_frames}")

    # Same frame - no transformation needed
    if from_frame == to_frame:
        return CoordinatePoint(x=x, y=y, z=z, frame=to_frame, epoch=epoch_iso)

    # Use high-precision libraries if available
    if ASTROPY_AVAILABLE:
        try:
            # Parse epoch
            time = Time(epoch_iso, format="isot")

            # Map frame names to astropy
            frame_map = {
                "ECI": GCRS,
                "GCRS": GCRS,
                "ECEF": ITRS,
                "ITRF": ITRS,
            }

            if from_frame in frame_map and to_frame in frame_map:
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

    # Manual transformations for basic cases
    if from_frame == "ECEF" and to_frame == "GEODETIC":
        lat, lon, alt = _manual_ecef_to_geodetic(x, y, z)
        return CoordinatePoint(x=lat, y=lon, z=alt, frame="GEODETIC", epoch=epoch_iso)
    elif from_frame == "GEODETIC" and to_frame == "ECEF":
        x_new, y_new, z_new = _manual_geodetic_to_ecef(x, y, z)
        return CoordinatePoint(x=x_new, y=y_new, z=z_new, frame="ECEF", epoch=epoch_iso)
    elif (from_frame in ["ECI", "GCRS"] and to_frame in ["ECEF", "ITRF"]) or (
        from_frame in ["ECEF", "ITRF"] and to_frame in ["ECI", "GCRS"]
    ):
        # Simple approximation: ECI ≈ ECEF (ignoring Earth rotation)
        # In real implementation, would apply rotation matrix based on GMST
        return CoordinatePoint(x=x, y=y, z=z, frame=to_frame, epoch=epoch_iso)

    raise NotImplementedError(
        f"Transformation from {from_frame} to {to_frame} not implemented. "
        f"Install astropy or skyfield for full functionality."
    )


def transform_frames_batch(
    xyz_list: list[list[float]],
    from_frame: str,
    to_frame: str,
    epoch_iso: str = "2000-01-01T12:00:00",
) -> list[CoordinatePoint]:
    """
    Batch transform multiple coordinates between reference frames.

    Uses vectorized NumPy calculations for efficiency.

    Args:
        xyz_list: List of [x, y, z] coordinates in meters
        from_frame: Source frame
        to_frame: Target frame
        epoch_iso: Reference epoch in ISO format

    Returns:
        List of CoordinatePoint with transformed coordinates
    """
    if not xyz_list:
        return []

    # Validate frame names
    valid_frames = {"ECEF", "ECI", "ITRF", "GCRS", "GEODETIC"}
    if from_frame not in valid_frames or to_frame not in valid_frames:
        raise ValueError(f"Frame must be one of {valid_frames}")

    # Convert to NumPy arrays
    coords = np.array(xyz_list)
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

    # Vectorized transformations for supported cases
    if from_frame == "ECEF" and to_frame == "GEODETIC":
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

    elif from_frame == "GEODETIC" and to_frame == "ECEF":
        x_new, y_new, z_new = _manual_geodetic_to_ecef_vectorized(x, y, z)
        return [
            CoordinatePoint(
                x=float(xi), y=float(yi), z=float(zi), frame="ECEF", epoch=epoch_iso
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
        "manual_transforms": ["ECEF <-> GEODETIC", "ECI <-> ECEF (approximate)"],
        "vectorized_batch": True,
        "notes": [
            "High-precision transformations require astropy",
            "Manual transforms use simplified models",
            "ECI/ECEF transforms ignore Earth rotation (approximate)",
            "Batch operations use vectorized NumPy for efficiency",
        ],
    }
