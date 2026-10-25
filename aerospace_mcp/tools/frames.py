"""Coordinate frame transformation tools for the Aerospace MCP server.

Provides tools for converting coordinates between common aerospace reference
frames including ECEF (Earth-Centered Earth-Fixed), ECI (Earth-Centered
Inertial), ITRF, GCRS, and geodetic (latitude/longitude/altitude). Geodetic
conversions use the WGS84 ellipsoid model parameters.

WARNING: This module is for educational and research purposes only.
Do NOT use for real flight planning, navigation, or aircraft operations.
"""

import json
import logging
from typing import Literal

logger = logging.getLogger(__name__)

# Inertial frames: transformations touching these depend on the epoch.
_INERTIAL = ("ECI", "GCRS")


def transform_frames(
    coordinates: dict,
    from_frame: Literal["ECEF", "ECI", "ITRF", "GCRS", "GEODETIC"],
    to_frame: Literal["ECEF", "ECI", "ITRF", "GCRS", "GEODETIC"],
    epoch_utc: str | None = None,
) -> str:
    """Transform coordinates between reference frames (ECEF, ECI, ITRF, GCRS, GEODETIC).

    Args:
        coordinates: Position in the source frame, as
            ``{"x": .., "y": .., "z": ..}`` in meters (a ``[x, y, z]`` list
            is also accepted).  For ``from_frame="GEODETIC"`` pass
            ``{"latitude_deg": .., "longitude_deg": .., "altitude_m": ..}``.
        from_frame: Source reference frame
        to_frame: Target reference frame
        epoch_utc: Optional epoch for time-dependent transformations (ISO
            format, UTC).  Only conversions involving ECI/GCRS depend on it;
            defaults to J2000.0 (2000-01-01T12:00:00).

    Returns:
        JSON string with transformed coordinates in the target frame:
        ``{"x", "y", "z"}`` in meters for Cartesian frames, or
        ``{"latitude_deg", "longitude_deg", "altitude_m"}`` for GEODETIC.

    Raises:
        No exceptions are raised directly; errors are returned as formatted strings.
        ImportError is caught when required frame transformation packages are missing.

    Note:
        Without astropy, ECI <-> ECEF applies IAU-1976 precession and a GMST
        rotation (nutation, UT1-UTC and polar motion neglected, ~1 km at LEO
        radius); GCRS is treated as ECI and ITRF as ECEF.
    """
    try:
        from ..integrations.frames import transform_frames as _transform

        # epoch_utc=None selects the integration's J2000.0 default
        result = _transform(coordinates, from_frame, to_frame, epoch_utc)

        if to_frame == "GEODETIC":
            output = {
                "latitude_deg": result.x,
                "longitude_deg": result.y,
                "altitude_m": result.z,
            }
            units = {
                "latitude": "degrees",
                "longitude": "degrees",
                "altitude": "meters",
            }
        else:
            output = {"x": result.x, "y": result.y, "z": result.z}
            units = {"position": "meters"}

        time_dependent = from_frame in _INERTIAL or to_frame in _INERTIAL
        return json.dumps(
            {
                "from_frame": from_frame,
                "to_frame": to_frame,
                "epoch_utc": result.epoch if time_dependent else None,
                "input": coordinates,
                "output": output,
                "units": units,
            },
            indent=2,
        )

    except ImportError as e:
        return f"Frame transformation not available - missing dependency: {str(e)}"
    except Exception as e:
        logger.error(f"Frame transformation error: {str(e)}", exc_info=True)
        return f"Frame transformation error: {str(e)}"


def geodetic_to_ecef(
    latitude_deg: float, longitude_deg: float, altitude_m: float = 0.0
) -> str:
    """Convert geodetic coordinates (lat/lon/alt) to Earth-centered Earth-fixed (ECEF) coordinates.

    Args:
        latitude_deg: Latitude in degrees (-90 to 90)
        longitude_deg: Longitude in degrees (-180 to 180)
        altitude_m: Altitude above WGS84 ellipsoid in meters

    Returns:
        JSON string with ECEF X, Y, Z coordinates in meters.

    Raises:
        No exceptions are raised directly; errors are returned as formatted strings.

    Note:
        The geodetic-to-ECEF conversion uses the WGS84 ellipsoid parameters:
            a = 6378137.0 m          (semi-major axis, equatorial radius)
            f = 1/298.257223563      (flattening)
            e^2 = 2f - f^2           (first eccentricity squared)

        The conversion equations are:
            N = a / sqrt(1 - e^2 * sin^2(lat))   (radius of curvature in prime vertical)
            X = (N + h) * cos(lat) * cos(lon)
            Y = (N + h) * cos(lat) * sin(lon)
            Z = (N * (1 - e^2) + h) * sin(lat)
        where lat, lon are geodetic latitude/longitude and h is altitude above ellipsoid.
    """
    try:
        from ..integrations.frames import geodetic_to_ecef as _geodetic_to_ecef

        result = _geodetic_to_ecef(latitude_deg, longitude_deg, altitude_m)

        return json.dumps(
            {
                "input": {
                    "latitude_deg": latitude_deg,
                    "longitude_deg": longitude_deg,
                    "altitude_m": altitude_m,
                },
                "output": {
                    "x_m": result.x,
                    "y_m": result.y,
                    "z_m": result.z,
                },
                "reference_frame": "WGS84 ECEF",
                "units": {"position": "meters"},
            },
            indent=2,
        )

    except ImportError:
        return "Coordinate conversion not available - geodetic module required"
    except Exception as e:
        logger.error(f"Geodetic to ECEF error: {str(e)}", exc_info=True)
        return f"Geodetic to ECEF error: {str(e)}"


def ecef_to_geodetic(x_m: float, y_m: float, z_m: float) -> str:
    """Convert ECEF coordinates to geodetic (lat/lon/alt) coordinates.

    Args:
        x_m: X coordinate in meters
        y_m: Y coordinate in meters
        z_m: Z coordinate in meters

    Returns:
        JSON string with geodetic latitude (deg), longitude (deg), and altitude (m).

    Raises:
        No exceptions are raised directly; errors are returned as formatted strings.
    """
    try:
        from ..integrations.frames import ecef_to_geodetic as _ecef_to_geodetic

        result = _ecef_to_geodetic(x_m, y_m, z_m)

        return json.dumps(
            {
                "input": {"x_m": x_m, "y_m": y_m, "z_m": z_m},
                "output": {
                    "latitude_deg": result.latitude_deg,
                    "longitude_deg": result.longitude_deg,
                    "altitude_m": result.altitude_m,
                },
                "reference_frame": "WGS84 Geodetic",
                "units": {
                    "latitude": "degrees",
                    "longitude": "degrees",
                    "altitude": "meters",
                },
            },
            indent=2,
        )

    except ImportError:
        return "Coordinate conversion not available - geodetic module required"
    except Exception as e:
        logger.error(f"ECEF to geodetic error: {str(e)}", exc_info=True)
        return f"ECEF to geodetic error: {str(e)}"
