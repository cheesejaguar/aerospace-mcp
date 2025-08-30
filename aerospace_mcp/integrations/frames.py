"""
Coordinate Frame Transformations

Provides transformations between different coordinate reference frames
commonly used in aerospace applications (ECI, ECEF, ITRF, etc.).
"""

import math

from pydantic import BaseModel, Field

from . import update_availability

# Earth parameters (WGS84)
EARTH_A = 6378137.0  # Semi-major axis (m)
EARTH_F = 1.0 / 298.257223563  # Flattening
EARTH_B = EARTH_A * (1.0 - EARTH_F)  # Semi-minor axis
EARTH_E2 = 2.0 * EARTH_F - EARTH_F**2  # First eccentricity squared

# Optional library imports
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
    from skyfield.api import load, utc
    from skyfield.positionlib import position_of_radec

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


# Data models
class CoordinatePoint(BaseModel):
    """A point in 3D space with metadata."""

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


def _manual_ecef_to_geodetic(
    x: float, y: float, z: float
) -> tuple[float, float, float]:
    """
    Convert ECEF to geodetic coordinates using iterative method.
    Returns (lat_deg, lon_deg, alt_m).
    """
    # Longitude
    lon_rad = math.atan2(y, x)

    # Distance from z-axis
    p = math.sqrt(x**2 + y**2)

    # Initial guess for latitude
    lat_rad = math.atan2(z, p * (1.0 - EARTH_E2))

    # Iterative solution for latitude and altitude
    for _ in range(10):  # Usually converges in 2-3 iterations
        sin_lat = math.sin(lat_rad)
        N = EARTH_A / math.sqrt(1.0 - EARTH_E2 * sin_lat**2)
        alt = p / math.cos(lat_rad) - N
        lat_rad_new = math.atan2(z, p * (1.0 - EARTH_E2 * N / (N + alt)))

        if abs(lat_rad_new - lat_rad) < 1e-12:
            break
        lat_rad = lat_rad_new

    return math.degrees(lat_rad), math.degrees(lon_rad), alt


def _manual_geodetic_to_ecef(
    lat_deg: float, lon_deg: float, alt_m: float
) -> tuple[float, float, float]:
    """
    Convert geodetic to ECEF coordinates.
    Returns (x, y, z) in meters.
    """
    lat_rad = math.radians(lat_deg)
    lon_rad = math.radians(lon_deg)

    sin_lat = math.sin(lat_rad)
    cos_lat = math.cos(lat_rad)
    sin_lon = math.sin(lon_rad)
    cos_lon = math.cos(lon_rad)

    # Radius of curvature in prime vertical
    N = EARTH_A / math.sqrt(1.0 - EARTH_E2 * sin_lat**2)

    x = (N + alt_m) * cos_lat * cos_lon
    y = (N + alt_m) * cos_lat * sin_lon
    z = (N * (1.0 - EARTH_E2) + alt_m) * sin_lat

    return x, y, z


def _simple_precession_matrix(epoch1: str, epoch2: str) -> list[list[float]]:
    """
    Simple precession matrix for ECI frame transformations.
    Very approximate - for demonstration only.
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


def ecef_to_geodetic(x: float, y: float, z: float) -> GeodeticPoint:
    """
    Convert ECEF coordinates to geodetic (WGS84).

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
        "notes": [
            "High-precision transformations require astropy",
            "Manual transforms use simplified models",
            "ECI/ECEF transforms ignore Earth rotation (approximate)",
        ],
    }
