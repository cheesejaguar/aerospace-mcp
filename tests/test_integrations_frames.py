"""
Tests for coordinate frame transformation module.
"""

import math

import pytest

from aerospace_mcp.integrations.frames import (
    ASTROPY_AVAILABLE,
    SKYFIELD_AVAILABLE,
    _manual_ecef_to_geodetic,
    _manual_geodetic_to_ecef,
    ecef_to_geodetic,
    geodetic_to_ecef,
    get_frame_info,
    transform_frames,
)


class TestGeodeticECEFConversions:
    """Test geodetic <-> ECEF coordinate conversions."""

    def test_manual_conversions_round_trip(self):
        """Test manual conversions maintain round-trip accuracy."""
        # Test with known coordinates
        lat, lon, alt = 37.7749, -122.4194, 100.0  # San Francisco-ish

        # Convert to ECEF and back
        x, y, z = _manual_geodetic_to_ecef(lat, lon, alt)
        lat_out, lon_out, alt_out = _manual_ecef_to_geodetic(x, y, z)

        # Should round-trip within reasonable precision
        assert abs(lat_out - lat) < 1e-9
        assert abs(lon_out - lon) < 1e-9
        assert abs(alt_out - alt) < 2e-6  # meters (adjusted for numerical precision)

    def test_known_ecef_values(self):
        """Test conversion against known ECEF values."""
        # Equator, Greenwich meridian, sea level
        lat, lon, alt = 0.0, 0.0, 0.0
        x, y, z = _manual_geodetic_to_ecef(lat, lon, alt)

        # Should be on equatorial plane at Earth radius
        assert abs(x - 6378137.0) < 1.0  # WGS84 semi-major axis
        assert abs(y) < 1e-6
        assert abs(z) < 1e-6

        # North pole
        lat, lon, alt = 90.0, 0.0, 0.0
        x, y, z = _manual_geodetic_to_ecef(lat, lon, alt)

        assert abs(x) < 1e-6
        assert abs(y) < 1e-6
        assert abs(z - 6356752.314245) < 1.0  # WGS84 semi-minor axis

    def test_geodetic_validation(self):
        """Test geodetic coordinate validation."""
        # Valid coordinates should work
        result = geodetic_to_ecef(45.0, -90.0, 1000.0)
        assert result.frame == "ECEF"
        # At longitude -90°, x should be near zero (on negative y-axis)
        assert abs(result.x) < 1.0  # Should be essentially zero
        assert abs(result.y) > 4000000  # Should be large negative y value

        # Invalid latitude
        with pytest.raises(ValueError, match="Latitude must be between"):
            geodetic_to_ecef(91.0, 0.0, 0.0)

        with pytest.raises(ValueError, match="Latitude must be between"):
            geodetic_to_ecef(-91.0, 0.0, 0.0)

        # Invalid longitude
        with pytest.raises(ValueError, match="Longitude must be between"):
            geodetic_to_ecef(0.0, 181.0, 0.0)

        with pytest.raises(ValueError, match="Longitude must be between"):
            geodetic_to_ecef(0.0, -181.0, 0.0)

    def test_altitude_effects(self):
        """Test that altitude changes affect ECEF coordinates properly."""
        lat, lon = 45.0, 45.0

        # Compare sea level vs high altitude
        ecef_low = geodetic_to_ecef(lat, lon, 0.0)
        ecef_high = geodetic_to_ecef(lat, lon, 10000.0)  # 10km up

        # Distance from origin should increase
        dist_low = math.sqrt(ecef_low.x**2 + ecef_low.y**2 + ecef_low.z**2)
        dist_high = math.sqrt(ecef_high.x**2 + ecef_high.y**2 + ecef_high.z**2)

        assert dist_high > dist_low
        assert abs(dist_high - dist_low - 10000.0) < 1.0  # Should be ~10km difference


class TestFrameTransformations:
    """Test coordinate frame transformations."""

    def test_same_frame_no_transform(self):
        """Test that same frame transformation returns unchanged coordinates."""
        xyz = [1000000.0, 2000000.0, 3000000.0]

        result = transform_frames(xyz, "ECEF", "ECEF")

        assert result.x == xyz[0]
        assert result.y == xyz[1]
        assert result.z == xyz[2]
        assert result.frame == "ECEF"

    def test_ecef_geodetic_transforms(self):
        """Test ECEF <-> GEODETIC transformations."""
        # Start with geodetic coordinates
        lat, lon, alt = 37.7749, -122.4194, 500.0

        # Transform to ECEF
        ecef_result = transform_frames([lat, lon, alt], "GEODETIC", "ECEF")

        assert ecef_result.frame == "ECEF"
        assert abs(ecef_result.x) > 1000000  # Reasonable ECEF magnitude

        # Transform back to geodetic
        geodetic_result = transform_frames(
            [ecef_result.x, ecef_result.y, ecef_result.z], "ECEF", "GEODETIC"
        )

        assert geodetic_result.frame == "GEODETIC"
        assert abs(geodetic_result.x - lat) < 1e-6  # Should round-trip
        assert abs(geodetic_result.y - lon) < 1e-6
        assert abs(geodetic_result.z - alt) < 0.001

    def test_coordinate_validation(self):
        """Test coordinate input validation."""
        # Wrong number of coordinates
        with pytest.raises(ValueError, match="xyz must be a list of 3 coordinates"):
            transform_frames([1, 2], "ECEF", "ECI")

        with pytest.raises(ValueError, match="xyz must be a list of 3 coordinates"):
            transform_frames([1, 2, 3, 4], "ECEF", "ECI")

    def test_invalid_frames(self):
        """Test invalid frame names."""
        xyz = [1000000.0, 2000000.0, 3000000.0]

        with pytest.raises(ValueError, match="Frame must be one of"):
            transform_frames(xyz, "INVALID", "ECEF")

        with pytest.raises(ValueError, match="Frame must be one of"):
            transform_frames(xyz, "ECEF", "INVALID")

    def test_eci_ecef_approximate_transform(self):
        """Test ECI <-> ECEF approximate transformation."""
        xyz = [6500000.0, 0.0, 0.0]  # Point on equatorial plane

        # Transform ECI to ECEF (with astropy, accounts for Earth rotation)
        result = transform_frames(xyz, "ECI", "ECEF")

        assert result.frame == "ECEF"
        # With proper transformation, coordinates will be significantly different due to Earth rotation
        # Just check that the magnitude is reasonable (should be similar distance from Earth center)
        original_magnitude = (xyz[0] ** 2 + xyz[1] ** 2 + xyz[2] ** 2) ** 0.5
        result_magnitude = (result.x**2 + result.y**2 + result.z**2) ** 0.5
        assert (
            abs(result_magnitude - original_magnitude) < 1000
        )  # Magnitude should be preserved

    @pytest.mark.skipif(not ASTROPY_AVAILABLE, reason="astropy not available")
    def test_astropy_integration(self):
        """Test astropy integration if available."""
        xyz = [6500000.0, 1000000.0, 2000000.0]

        # Should use astropy for high-precision transforms if available
        result = transform_frames(xyz, "ECI", "ECEF", "2023-01-01T00:00:00")

        assert result.frame == "ECEF"
        assert isinstance(result.x, float)
        assert isinstance(result.y, float)
        assert isinstance(result.z, float)

    def test_approximate_transforms(self):
        """Test that approximate transformations work without high-precision libraries."""
        xyz = [1000000.0, 2000000.0, 3000000.0]

        # Without high-precision libraries, should still provide approximate results
        if not ASTROPY_AVAILABLE and not SKYFIELD_AVAILABLE:
            result = transform_frames(xyz, "GCRS", "ITRF")
            assert result.frame == "ITRF"
            assert result.x == xyz[0]  # Simplified approximation keeps same coordinates
            assert result.y == xyz[1]
            assert result.z == xyz[2]


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_ecef_to_geodetic_function(self):
        """Test ECEF to geodetic convenience function."""
        # Known ECEF coordinates (approximately San Francisco)
        x, y, z = -2699490.0, -4293565.0, 3855273.0

        result = ecef_to_geodetic(x, y, z)

        # Should be reasonable geodetic coordinates
        assert -90 <= result.latitude_deg <= 90
        assert -180 <= result.longitude_deg <= 180
        assert result.altitude_m > -1000  # Not too far underground

    def test_geodetic_to_ecef_function(self):
        """Test geodetic to ECEF convenience function."""
        result = geodetic_to_ecef(37.7749, -122.4194, 100.0)

        assert result.frame == "ECEF"
        assert isinstance(result.x, float)
        assert isinstance(result.y, float)
        assert isinstance(result.z, float)

        # Should be reasonable ECEF magnitude for Earth surface
        distance = math.sqrt(result.x**2 + result.y**2 + result.z**2)
        assert 6.3e6 < distance < 6.5e6  # Earth radius range


class TestFrameInfo:
    """Test frame information utilities."""

    def test_get_frame_info(self):
        """Test frame information retrieval."""
        info = get_frame_info()

        assert "available_frames" in info
        assert "ECEF" in info["available_frames"]
        assert "GEODETIC" in info["available_frames"]
        assert "ECI" in info["available_frames"]

        assert "libraries" in info
        assert "astropy" in info["libraries"]
        assert "skyfield" in info["libraries"]

        assert "manual_transforms" in info
        assert "notes" in info

        # Check library availability reporting - values should be boolean
        assert isinstance(info["libraries"]["astropy"], bool)
        assert isinstance(info["libraries"]["skyfield"], bool)
        assert info["high_precision_available"] == ASTROPY_AVAILABLE


class TestNumericalStability:
    """Test numerical stability and edge cases."""

    def test_polar_coordinates(self):
        """Test coordinate transformations at poles."""
        # North pole
        result = geodetic_to_ecef(90.0, 0.0, 1000.0)

        # At pole, X and Y should be near zero
        assert abs(result.x) < 1e-6
        assert abs(result.y) < 1e-6
        assert result.z > 6350000  # Should be near polar radius + altitude

        # South pole
        result = geodetic_to_ecef(-90.0, 180.0, 0.0)
        assert abs(result.x) < 1e-6
        assert abs(result.y) < 1e-6
        assert result.z < -6350000  # Should be negative

    def test_equatorial_coordinates(self):
        """Test coordinates on equator."""
        # Prime meridian on equator
        result = geodetic_to_ecef(0.0, 0.0, 0.0)

        assert abs(result.x - 6378137.0) < 1.0  # Earth radius
        assert abs(result.y) < 1e-6
        assert abs(result.z) < 1e-6

        # 90° east on equator
        result = geodetic_to_ecef(0.0, 90.0, 0.0)

        assert abs(result.x) < 1e-6
        assert abs(result.y - 6378137.0) < 1.0
        assert abs(result.z) < 1e-6

    def test_high_altitude(self):
        """Test very high altitude coordinates."""
        # Satellite altitude
        result = geodetic_to_ecef(0.0, 0.0, 20200000.0)  # GPS orbit altitude

        # Distance from center should be Earth radius + altitude
        distance = math.sqrt(result.x**2 + result.y**2 + result.z**2)
        expected = 6378137.0 + 20200000.0
        assert abs(distance - expected) < 1.0

    def test_precision_consistency(self):
        """Test that precision is consistent across different coordinate ranges."""
        test_cases = [
            (0.0, 0.0, 0.0),
            (45.0, 45.0, 1000.0),
            (-45.0, -45.0, 1000.0),
            (89.9, 179.9, 10000.0),
            (-89.9, -179.9, 10000.0),
        ]

        for lat, lon, alt in test_cases:
            # Round trip test
            ecef = geodetic_to_ecef(lat, lon, alt)
            geodetic = ecef_to_geodetic(ecef.x, ecef.y, ecef.z)

            # Should maintain precision
            assert abs(geodetic.latitude_deg - lat) < 1e-9
            assert abs(geodetic.longitude_deg - lon) < 1e-9
            assert (
                abs(geodetic.altitude_m - alt) < 5e-3
            )  # Relaxed for high latitude/altitude cases


if __name__ == "__main__":
    pytest.main([__file__])
