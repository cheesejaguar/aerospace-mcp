"""
Tests for coordinate frame transformation module.
"""

import math

import pytest

from aerospace_mcp.integrations._array_backend import np
from aerospace_mcp.integrations.frames import (
    ASTROPY_AVAILABLE,
    SKYFIELD_AVAILABLE,
    _manual_ecef_to_geodetic,
    _manual_ecef_to_geodetic_vectorized,
    _manual_geodetic_to_ecef,
    _manual_geodetic_to_ecef_vectorized,
    ecef_to_geodetic,
    geodetic_to_ecef,
    get_frame_info,
    transform_frames,
    transform_frames_batch,
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
        """GMST-based ECI<->ECEF works without high-precision libraries."""
        xyz = [1000000.0, 2000000.0, 3000000.0]

        if not ASTROPY_AVAILABLE and not SKYFIELD_AVAILABLE:
            result = transform_frames(xyz, "GCRS", "ITRF")
            assert result.frame == "ITRF"
            # z-axis rotation: z unchanged, xy-magnitude preserved
            assert result.z == xyz[2]
            xy_mag_in = (xyz[0] ** 2 + xyz[1] ** 2) ** 0.5
            xy_mag_out = (result.x**2 + result.y**2) ** 0.5
            assert abs(xy_mag_out - xy_mag_in) < 1e-3

    @pytest.mark.skipif(ASTROPY_AVAILABLE, reason="tests the manual GMST path")
    def test_eci_ecef_round_trip(self):
        """ECI -> ECEF -> ECI recovers the input to sub-millimetre level."""
        xyz = [6500000.0, 1000000.0, 2000000.0]
        epoch = "2024-03-20T12:00:00"

        ecef = transform_frames(xyz, "ECI", "ECEF", epoch)
        back = transform_frames([ecef.x, ecef.y, ecef.z], "ECEF", "ECI", epoch)

        assert abs(back.x - xyz[0]) < 1e-3
        assert abs(back.y - xyz[1]) < 1e-3
        assert abs(back.z - xyz[2]) < 1e-3

    @pytest.mark.skipif(ASTROPY_AVAILABLE, reason="tests the manual GMST path")
    def test_eci_ecef_rotation_advances_with_time(self):
        """Six hours of Earth rotation shifts the transform by ~90.25 deg."""
        import math

        xyz = [6500000.0, 0.0, 0.0]
        epoch1 = "2024-01-01T00:00:00"
        epoch2 = "2024-01-01T06:00:00"

        r1 = transform_frames(xyz, "ECI", "ECEF", epoch1)
        r2 = transform_frames(xyz, "ECI", "ECEF", epoch2)

        angle1 = math.atan2(r1.y, r1.x)
        angle2 = math.atan2(r2.y, r2.x)
        delta_deg = math.degrees(angle2 - angle1) % 360.0
        # Sidereal rate: 360.9856 deg/day -> 90.246 deg per 6 h.  The ECI->ECEF
        # rotation angle is -GMST, so the frame angle decreases; check the
        # magnitude of the advance.
        assert (
            abs((360.0 - delta_deg) - 90.246) < 0.01 or abs(delta_deg - 90.246) < 0.01
        )


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

        # Check library availability reporting
        assert info["libraries"]["astropy"] == ASTROPY_AVAILABLE
        assert info["libraries"]["skyfield"] == SKYFIELD_AVAILABLE
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


class TestVectorizedCoordinateTransforms:
    """Test vectorized coordinate transformation functions."""

    def test_vectorized_ecef_to_geodetic(self):
        """Test vectorized ECEF to geodetic conversion."""
        # Test points (San Francisco, Tokyo, London approximately)
        x = np.array([-2699490.0, -3946792.0, 3979261.0])
        y = np.array([-4293565.0, 3366090.0, -14416.0])
        z = np.array([3855273.0, 3698041.0, 4969426.0])

        lat, lon, alt = _manual_ecef_to_geodetic_vectorized(x, y, z)

        # Check correct array lengths
        assert len(lat) == 3
        assert len(lon) == 3
        assert len(alt) == 3

        # All latitudes should be in valid range
        assert all(-90 <= lat_val <= 90 for lat_val in lat)
        # All longitudes should be in valid range
        assert all(-180 <= lon_val <= 180 for lon_val in lon)

    def test_vectorized_geodetic_to_ecef(self):
        """Test vectorized geodetic to ECEF conversion."""
        lat = np.array([37.7749, 35.6762, 51.5074])
        lon = np.array([-122.4194, 139.6503, -0.1278])
        alt = np.array([0.0, 0.0, 0.0])

        x, y, z = _manual_geodetic_to_ecef_vectorized(lat, lon, alt)

        # Check correct array lengths
        assert len(x) == 3
        assert len(y) == 3
        assert len(z) == 3

        # All coordinates should be within Earth surface range
        for i in range(3):
            dist = math.sqrt(float(x[i]) ** 2 + float(y[i]) ** 2 + float(z[i]) ** 2)
            assert 6.3e6 < dist < 6.5e6

    def test_vectorized_consistency_with_scalar(self):
        """Test that vectorized transforms match scalar transforms."""
        # Test ECEF to geodetic
        x, y, z = -2699490.0, -4293565.0, 3855273.0
        scalar_result = _manual_ecef_to_geodetic(x, y, z)
        vector_result = _manual_ecef_to_geodetic_vectorized(
            np.array([x]), np.array([y]), np.array([z])
        )

        assert abs(scalar_result[0] - float(vector_result[0][0])) < 1e-10
        assert abs(scalar_result[1] - float(vector_result[1][0])) < 1e-10
        assert abs(scalar_result[2] - float(vector_result[2][0])) < 1e-6

        # Test geodetic to ECEF
        lat, lon, alt = 37.7749, -122.4194, 100.0
        scalar_result = _manual_geodetic_to_ecef(lat, lon, alt)
        vector_result = _manual_geodetic_to_ecef_vectorized(
            np.array([lat]), np.array([lon]), np.array([alt])
        )

        assert abs(scalar_result[0] - float(vector_result[0][0])) < 1e-6
        assert abs(scalar_result[1] - float(vector_result[1][0])) < 1e-6
        assert abs(scalar_result[2] - float(vector_result[2][0])) < 1e-6


class TestBatchTransformations:
    """Test batch coordinate transformation function."""

    def test_transform_frames_batch_same_frame(self):
        """Test batch transformation with same source and target frame."""
        xyz_list = [[1e6, 2e6, 3e6], [4e6, 5e6, 6e6], [7e6, 8e6, 9e6]]
        results = transform_frames_batch(xyz_list, "ECEF", "ECEF")

        assert len(results) == 3
        for i, result in enumerate(results):
            assert result.frame == "ECEF"
            assert result.x == xyz_list[i][0]
            assert result.y == xyz_list[i][1]
            assert result.z == xyz_list[i][2]

    def test_transform_frames_batch_ecef_to_geodetic(self):
        """Test batch ECEF to GEODETIC conversion."""
        xyz_list = [
            [-2699490.0, -4293565.0, 3855273.0],  # San Francisco
            [3979261.0, -14416.0, 4969426.0],  # London
        ]
        results = transform_frames_batch(xyz_list, "ECEF", "GEODETIC")

        assert len(results) == 2
        for result in results:
            assert result.frame == "GEODETIC"
            assert -90 <= result.x <= 90  # latitude
            assert -180 <= result.y <= 180  # longitude

    def test_transform_frames_batch_geodetic_to_ecef(self):
        """Test batch GEODETIC to ECEF conversion."""
        # Lat, Lon, Alt as x, y, z
        xyz_list = [
            [37.7749, -122.4194, 0.0],  # San Francisco
            [51.5074, -0.1278, 0.0],  # London
        ]
        results = transform_frames_batch(xyz_list, "GEODETIC", "ECEF")

        assert len(results) == 2
        for result in results:
            assert result.frame == "ECEF"
            # Should be valid ECEF coordinates
            dist = math.sqrt(result.x**2 + result.y**2 + result.z**2)
            assert 6.3e6 < dist < 6.5e6

    def test_transform_frames_batch_empty_list(self):
        """Test batch transformation with empty list."""
        results = transform_frames_batch([], "ECEF", "GEODETIC")
        assert results == []


class TestPolarAxisGeodetic:
    """ECEF -> geodetic on and near the polar axis (formerly divided by cos(lat))."""

    POLAR_RADIUS = 6356752.314245  # WGS-84 b (NIMA TR8350.2, Table 3.1)

    def test_pole_on_ellipsoid(self):
        lat, lon, alt = _manual_ecef_to_geodetic(0.0, 0.0, self.POLAR_RADIUS)
        assert lat == 90.0
        assert abs(alt) < 1e-6

    def test_pole_above_ellipsoid_and_south_pole(self):
        _, _, alt = _manual_ecef_to_geodetic(0.0, 0.0, self.POLAR_RADIUS + 1000.0)
        assert abs(alt - 1000.0) < 1e-6
        lat, _, alt = _manual_ecef_to_geodetic(0.0, 0.0, -self.POLAR_RADIUS)
        assert lat == -90.0
        assert abs(alt) < 1e-6

    def test_one_millimetre_off_axis(self):
        # Old algorithm: alt = -1.36 m here (p / cos(lat) cancellation).
        _, _, alt = _manual_ecef_to_geodetic(1e-3, 0.0, self.POLAR_RADIUS)
        assert abs(alt) < 1e-6

    def test_near_pole_round_trip(self):
        x, y, z = _manual_geodetic_to_ecef(89.9999, 10.0, 1000.0)
        lat, lon, alt = _manual_ecef_to_geodetic(x, y, z)
        assert abs(lat - 89.9999) < 1e-10
        assert abs(lon - 10.0) < 1e-10
        assert abs(alt - 1000.0) < 1e-6  # old algorithm: 999.954 m

    def test_origin_does_not_crash(self):
        lat, lon, alt = _manual_ecef_to_geodetic(0.0, 0.0, 0.0)
        assert math.isfinite(lat) and math.isfinite(alt)

    def test_vectorized_matches_scalar_at_pole(self):
        pts = [(0.0, 0.0, self.POLAR_RADIUS), (1e-3, 0.0, -self.POLAR_RADIUS)]
        lat, _, alt = _manual_ecef_to_geodetic_vectorized(*zip(*pts, strict=True))
        for i, p in enumerate(pts):
            s_lat, _, s_alt = _manual_ecef_to_geodetic(*p)
            assert abs(float(lat[i]) - s_lat) < 1e-12
            assert abs(float(alt[i]) - s_alt) < 1e-6


class TestFrameAliasesAndChaining:
    """Frames documented as supported must not raise NotImplementedError."""

    EPOCH = "2024-03-20T12:00:00Z"

    def test_identity_aliases(self):
        xyz = [7e6, 1e6, 2e6]
        for src, dst in [("ECI", "GCRS"), ("ITRF", "ECEF"), ("ECEF", "ITRF")]:
            r = transform_frames(xyz, src, dst, self.EPOCH)
            if not ASTROPY_AVAILABLE:
                assert [r.x, r.y, r.z] == xyz
            assert r.frame == dst

    def test_itrf_geodetic(self):
        r = transform_frames([6378137.0, 0.0, 0.0], "ITRF", "GEODETIC")
        assert (r.x, r.y) == (0.0, 0.0) and abs(r.z) < 1e-9
        back = transform_frames([0.0, 0.0, 0.0], "GEODETIC", "ITRF")
        assert abs(back.x - 6378137.0) < 1e-9 and back.frame == "ITRF"

    def test_geodetic_eci_round_trip(self):
        geo = {"latitude_deg": 10.0, "longitude_deg": 20.0, "altitude_m": 500e3}
        eci = transform_frames(geo, "GEODETIC", "ECI", self.EPOCH)
        # Earth rotation preserves the radius and the z component
        ecef = transform_frames(geo, "GEODETIC", "ECEF")
        assert (
            abs(
                math.dist([eci.x, eci.y, eci.z], [0, 0, 0])
                - math.dist([ecef.x, ecef.y, ecef.z], [0, 0, 0])
            )
            < 1e-6
        )
        back = transform_frames([eci.x, eci.y, eci.z], "ECI", "GEODETIC", self.EPOCH)
        assert abs(back.x - 10.0) < 1e-9
        assert abs(back.y - 20.0) < 1e-9
        assert abs(back.z - 500e3) < 1e-6

    def test_dict_inputs(self):
        a = transform_frames({"x": 1.0, "y": 2.0, "z": 3.0}, "ECEF", "ECEF")
        b = transform_frames({"x_m": 1.0, "y_m": 2.0, "z_m": 3.0}, "ECEF", "ECEF")
        assert (a.x, a.y, a.z) == (b.x, b.y, b.z) == (1.0, 2.0, 3.0)
        with pytest.raises(ValueError, match="coordinates dict must contain"):
            transform_frames({"lat": 1}, "GEODETIC", "ECEF")
        with pytest.raises(ValueError, match="finite"):
            transform_frames([float("nan"), 0, 0], "ECEF", "GEODETIC")
        with pytest.raises(ValueError, match="Latitude"):
            transform_frames([91.0, 0, 0], "GEODETIC", "ECI")

    def test_none_epoch_means_j2000(self):
        a = transform_frames([7e6, 0, 0], "ECI", "ECEF", None)
        b = transform_frames([7e6, 0, 0], "ECI", "ECEF", "2000-01-01T12:00:00")
        assert (a.x, a.y, a.z) == (b.x, b.y, b.z)
        assert a.epoch == "2000-01-01T12:00:00"

    def test_invalid_epoch(self):
        with pytest.raises(ValueError, match="Invalid epoch"):
            transform_frames([7e6, 0, 0], "ECI", "ECEF", "yesterday")

    def test_vallado_example_3_15(self):
        """Vallado (2013) Example 3-15 IAU-76/FK5 reduction, 2004-04-06 07:51:28.386 UTC.

        r_ITRF = (-1033.4793830, 7901.2952754, 6380.3565958) km and
        r_GCRF = (5102.508958, 6123.011401, 6378.136928) km.  The manual path
        neglects nutation, UT1-UTC and polar motion (~140 m here); without the
        precession term the error would be ~10 km.
        """
        epoch = "2004-04-06T07:51:28.386009"
        r_itrf = [-1033.4793830e3, 7901.2952754e3, 6380.3565958e3]
        r_gcrf = [5102.508958e3, 6123.011401e3, 6378.136928e3]
        out = transform_frames(r_itrf, "ITRF", "GCRS", epoch)
        assert math.dist([out.x, out.y, out.z], r_gcrf) < 300.0
        back = transform_frames(r_gcrf, "GCRS", "ITRF", epoch)
        assert math.dist([back.x, back.y, back.z], r_itrf) < 300.0

    def test_frame_info_notes_are_current(self):
        notes = " ".join(get_frame_info()["notes"])
        assert "ignore Earth rotation" not in notes
        assert "GMST" in notes


if __name__ == "__main__":
    pytest.main([__file__])
