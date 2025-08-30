"""Coordinate frame transformation tools for the Aerospace MCP server."""

import json
import logging
from typing import Literal

logger = logging.getLogger(__name__)


def transform_frames(
    coordinates: dict,
    from_frame: Literal["ECEF", "ECI", "ITRF", "GCRS", "GEODETIC"],
    to_frame: Literal["ECEF", "ECI", "ITRF", "GCRS", "GEODETIC"],
    epoch_utc: str | None = None
) -> str:
    """Transform coordinates between reference frames (ECEF, ECI, ITRF, GCRS, GEODETIC).

    Args:
        coordinates: Dict with coordinate data (format depends on frame)
        from_frame: Source reference frame
        to_frame: Target reference frame
        epoch_utc: Optional epoch for time-dependent transformations (ISO format)

    Returns:
        JSON string with transformed coordinates
    """
    try:
        from ..integrations.frames import transform_frames as _transform

        result = _transform(coordinates, from_frame, to_frame, epoch_utc)
        return json.dumps(result, indent=2)

    except ImportError as e:
        return f"Frame transformation not available - missing dependency: {str(e)}"
    except Exception as e:
        logger.error(f"Frame transformation error: {str(e)}", exc_info=True)
        return f"Frame transformation error: {str(e)}"


def geodetic_to_ecef(
    latitude_deg: float,
    longitude_deg: float,
    altitude_m: float = 0.0
) -> str:
    """Convert geodetic coordinates (lat/lon/alt) to Earth-centered Earth-fixed (ECEF) coordinates.

    Args:
        latitude_deg: Latitude in degrees (-90 to 90)
        longitude_deg: Longitude in degrees (-180 to 180)
        altitude_m: Altitude above WGS84 ellipsoid in meters

    Returns:
        JSON string with ECEF coordinates
    """
    try:
        from ..integrations.frames import geodetic_to_ecef as _geodetic_to_ecef

        result = _geodetic_to_ecef(latitude_deg, longitude_deg, altitude_m)

        return json.dumps({
            "input": {
                "latitude_deg": latitude_deg,
                "longitude_deg": longitude_deg,
                "altitude_m": altitude_m
            },
            "output": {
                "x_m": result["x_m"],
                "y_m": result["y_m"],
                "z_m": result["z_m"]
            },
            "reference_frame": "WGS84 ECEF",
            "units": {
                "position": "meters"
            }
        }, indent=2)

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
        JSON string with geodetic coordinates
    """
    try:
        from ..integrations.frames import ecef_to_geodetic as _ecef_to_geodetic

        result = _ecef_to_geodetic(x_m, y_m, z_m)

        return json.dumps({
            "input": {
                "x_m": x_m,
                "y_m": y_m,
                "z_m": z_m
            },
            "output": {
                "latitude_deg": result["latitude_deg"],
                "longitude_deg": result["longitude_deg"],
                "altitude_m": result["altitude_m"]
            },
            "reference_frame": "WGS84 Geodetic",
            "units": {
                "latitude": "degrees",
                "longitude": "degrees",
                "altitude": "meters"
            }
        }, indent=2)

    except ImportError:
        return "Coordinate conversion not available - geodetic module required"
    except Exception as e:
        logger.error(f"ECEF to geodetic error: {str(e)}", exc_info=True)
        return f"ECEF to geodetic error: {str(e)}"
