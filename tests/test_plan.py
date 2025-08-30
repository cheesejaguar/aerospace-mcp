"""Tests for flight planning functionality."""

import pytest
from unittest.mock import patch, MagicMock
import math
from typing import Tuple, List

from aerospace_mcp.core import (
    great_circle_points,
    estimates_openap,
    SegmentEst,
    OpenAPError,
    NM_PER_KM,
    KM_PER_NM,
    PlanRequest,
    PlanResponse
)


class TestGreatCirclePoints:
    """Tests for great circle route calculation."""

    @pytest.mark.unit
    def test_sjc_to_nrt_distance(self):
        """Test great circle calculation from SJC to NRT."""
        # SJC coordinates
        lat1, lon1 = 37.3626, -121.929
        # NRT coordinates
        lat2, lon2 = 35.7647, 140.386

        polyline, distance_km = great_circle_points(lat1, lon1, lat2, lon2, 100.0)

        # Approximate distance SJC->NRT is ~8,800-9,200 km
        assert 8500 < distance_km < 9500

        # Should have multiple points based on step size
        expected_points = int(math.ceil(distance_km / 100.0)) + 1
        assert len(polyline) == expected_points

        # First point should be SJC
        assert abs(polyline[0][0] - lat1) < 0.01
        assert abs(polyline[0][1] - lon1) < 0.01

        # Last point should be NRT
        assert abs(polyline[-1][0] - lat2) < 0.01
        assert abs(polyline[-1][1] - lon2) < 0.01

    @pytest.mark.unit
    def test_short_distance_calculation(self):
        """Test great circle calculation for short distance."""
        # SJC to SFO (nearby airports)
        lat1, lon1 = 37.3626, -121.929  # SJC
        lat2, lon2 = 37.6213, -122.379  # SFO

        polyline, distance_km = great_circle_points(lat1, lon1, lat2, lon2, 10.0)

        # Distance should be around 65-75 km
        assert 50 < distance_km < 100

        # Should have at least 2 points (start and end)
        assert len(polyline) >= 2

    @pytest.mark.unit
    def test_same_point_calculation(self):
        """Test great circle calculation for same point."""
        lat, lon = 37.3626, -121.929

        polyline, distance_km = great_circle_points(lat, lon, lat, lon, 10.0)

        # Distance should be 0
        assert distance_km == 0.0

        # Should have 1 point
        assert len(polyline) == 1
        assert polyline[0] == (lat, lon)

    @pytest.mark.unit
    @pytest.mark.parametrize("step_km", [1.0, 25.0, 50.0, 100.0, 500.0])
    def test_different_step_sizes(self, step_km):
        """Test different step sizes for polyline generation."""
        lat1, lon1 = 37.3626, -121.929  # SJC
        lat2, lon2 = 35.7647, 140.386   # NRT

        polyline, distance_km = great_circle_points(lat1, lon1, lat2, lon2, step_km)

        # Distance should be consistent regardless of step size
        assert 8500 < distance_km < 9500

        # Number of points should be inversely related to step size
        expected_points = max(1, int(math.ceil(distance_km / step_km))) + 1
        assert len(polyline) == expected_points

    @pytest.mark.unit
    def test_antipodal_points(self):
        """Test great circle calculation for antipodal points."""
        # Roughly antipodal points
        lat1, lon1 = 0.0, 0.0
        lat2, lon2 = 0.0, 180.0

        polyline, distance_km = great_circle_points(lat1, lon1, lat2, lon2, 1000.0)

        # Distance should be approximately half the earth's circumference
        earth_circumference = 40075.0  # km at equator
        expected_distance = earth_circumference / 2
        assert abs(distance_km - expected_distance) < 100  # 100km tolerance


class TestOpenAPEstimates:
    """Tests for OpenAP flight performance estimates."""

    @pytest.mark.unit
    def test_openap_unavailable_error(self):
        """Test error when OpenAP is not available."""
        with patch('aerospace_mcp.core.OPENAP_AVAILABLE', False):
            with pytest.raises(OpenAPError, match="OpenAP backend unavailable"):
                estimates_openap("A320", 35000, None, 1000.0)

    @pytest.mark.unit
    @patch('aerospace_mcp.core.OPENAP_AVAILABLE', True)
    def test_openap_estimates_a359(self, mock_openap_flight_generator, mock_openap_fuel_flow, mock_openap_props):
        """Test OpenAP estimates for A359."""
        with patch('aerospace_mcp.core.FlightGenerator', return_value=mock_openap_flight_generator):
            with patch('aerospace_mcp.core.FuelFlow', return_value=mock_openap_fuel_flow):
                with patch('aerospace_mcp.core.prop.aircraft', return_value=mock_openap_props):

                    estimates, engine_name = estimates_openap("A359", 35000, None, 9000.0)

                    assert engine_name == "openap"
                    assert "block" in estimates
                    assert "climb" in estimates
                    assert "cruise" in estimates
                    assert "descent" in estimates
                    assert "assumptions" in estimates

                    # Check block estimates
                    block = estimates["block"]
                    assert "time_min" in block
                    assert "fuel_kg" in block
                    assert block["time_min"] > 0
                    assert block["fuel_kg"] > 0

                    # Check segment estimates structure
                    for segment_name in ["climb", "cruise", "descent"]:
                        segment = estimates[segment_name]
                        assert "time_min" in segment
                        assert "distance_km" in segment
                        assert "avg_gs_kts" in segment
                        assert "fuel_kg" in segment

                    # Check assumptions
                    assumptions = estimates["assumptions"]
                    assert assumptions["zero_wind"] is True
                    assert assumptions["cruise_alt_ft"] == 35000
                    assert "mass_kg" in assumptions

    @pytest.mark.unit
    @patch('aerospace_mcp.core.OPENAP_AVAILABLE', True)
    def test_openap_with_explicit_mass(self, mock_openap_flight_generator, mock_openap_fuel_flow):
        """Test OpenAP estimates with explicit mass."""
        with patch('aerospace_mcp.core.FlightGenerator', return_value=mock_openap_flight_generator):
            with patch('aerospace_mcp.core.FuelFlow', return_value=mock_openap_fuel_flow):

                test_mass = 75000.0  # kg
                estimates, _ = estimates_openap("A320", 35000, test_mass, 5000.0)

                assert estimates["assumptions"]["mass_kg"] == test_mass

    @pytest.mark.unit
    @patch('aerospace_mcp.core.OPENAP_AVAILABLE', True)
    def test_openap_fallback_mass(self, mock_openap_flight_generator, mock_openap_fuel_flow):
        """Test OpenAP estimates with fallback mass when aircraft props fail."""
        with patch('aerospace_mcp.core.FlightGenerator', return_value=mock_openap_flight_generator):
            with patch('aerospace_mcp.core.FuelFlow', return_value=mock_openap_fuel_flow):
                with patch('aerospace_mcp.core.prop.aircraft', side_effect=Exception("Aircraft not found")):

                    estimates, _ = estimates_openap("UNKNOWN", 35000, None, 5000.0)

                    # Should use fallback mass
                    assert estimates["assumptions"]["mass_kg"] == 60000.0

    @pytest.mark.unit
    @patch('aerospace_mcp.core.OPENAP_AVAILABLE', True)
    def test_openap_cruise_altitude_handling(self, mock_openap_flight_generator, mock_openap_fuel_flow):
        """Test OpenAP estimates with different cruise altitudes."""
        mock_gen = mock_openap_flight_generator
        mock_gen.climb.side_effect = [TypeError("alt_cr not supported"), mock_gen.climb.return_value]
        mock_gen.descent.side_effect = [TypeError("alt_cr not supported"), mock_gen.descent.return_value]

        with patch('aerospace_mcp.core.FlightGenerator', return_value=mock_gen):
            with patch('aerospace_mcp.core.FuelFlow', return_value=mock_openap_fuel_flow):

                # Should handle TypeError gracefully and call without alt_cr
                estimates, _ = estimates_openap("A320", 45000, None, 5000.0)

                # Verify both climb and descent were called twice (first with alt_cr, then without)
                assert mock_gen.climb.call_count == 2
                assert mock_gen.descent.call_count == 2

    @pytest.mark.unit
    @patch('aerospace_mcp.core.OPENAP_AVAILABLE', True)
    def test_openap_short_route(self, mock_openap_fuel_flow):
        """Test OpenAP estimates for very short routes."""
        # Create a mock generator with long climb/descent distances
        mock_gen = MagicMock()

        import pandas as pd

        # Mock segments where climb + descent > total route distance
        climb_data = pd.DataFrame({
            't': [60], 's': [100000], 'altitude': [25000], 'groundspeed': [300], 'vertical_rate': [1500]
        })
        cruise_data = pd.DataFrame({
            't': [30], 's': [15000], 'altitude': [35000], 'groundspeed': [450], 'vertical_rate': [0]
        })
        descent_data = pd.DataFrame({
            't': [60], 's': [120000], 'altitude': [15000], 'groundspeed': [350], 'vertical_rate': [-1200]
        })

        mock_gen.climb.return_value = climb_data
        mock_gen.cruise.return_value = cruise_data
        mock_gen.descent.return_value = descent_data

        with patch('aerospace_mcp.core.FlightGenerator', return_value=mock_gen):
            with patch('aerospace_mcp.core.FuelFlow', return_value=mock_openap_fuel_flow):

                # Very short route - 200km, but climb+descent = 220km
                estimates, _ = estimates_openap("A320", 35000, None, 200.0)

                # Cruise distance should be 0
                assert estimates["cruise"]["distance_km"] == 0.0
                assert estimates["cruise"]["time_min"] == 0.0


class TestSegmentEst:
    """Tests for SegmentEst model."""

    @pytest.mark.unit
    def test_segment_est_creation(self):
        """Test SegmentEst model creation."""
        segment = SegmentEst(
            time_min=120.5,
            distance_km=850.0,
            avg_gs_kts=420.0,
            fuel_kg=2500.0
        )

        assert segment.time_min == 120.5
        assert segment.distance_km == 850.0
        assert segment.avg_gs_kts == 420.0
        assert segment.fuel_kg == 2500.0

    @pytest.mark.unit
    def test_segment_est_serialization(self):
        """Test SegmentEst model serialization."""
        segment = SegmentEst(
            time_min=60.0,
            distance_km=500.0,
            avg_gs_kts=400.0,
            fuel_kg=1200.0
        )

        data = segment.model_dump()
        assert isinstance(data, dict)
        assert data["time_min"] == 60.0
        assert data["distance_km"] == 500.0
        assert data["avg_gs_kts"] == 400.0
        assert data["fuel_kg"] == 1200.0


class TestConstants:
    """Tests for unit conversion constants."""

    @pytest.mark.unit
    def test_nm_km_conversion_constants(self):
        """Test nautical mile to kilometer conversion constants."""
        # 1 NM = 1.852 km (exact definition)
        expected_nm_per_km = 1.0 / 1.852
        expected_km_per_nm = 1.852

        assert abs(NM_PER_KM - expected_nm_per_km) < 1e-6
        assert abs(KM_PER_NM - expected_km_per_nm) < 1e-6

        # Test that they are inverses
        assert abs(NM_PER_KM * KM_PER_NM - 1.0) < 1e-10

    @pytest.mark.unit
    def test_distance_conversions(self):
        """Test practical distance conversions."""
        # Test round trip conversions
        km_values = [100, 500, 1000, 5000]

        for km in km_values:
            nm = km * NM_PER_KM
            km_back = nm * KM_PER_NM
            assert abs(km - km_back) < 1e-10


class TestPlanRequestValidation:
    """Tests for PlanRequest model validation."""

    @pytest.mark.unit
    def test_valid_plan_request(self):
        """Test creating a valid plan request."""
        request = PlanRequest(
            depart_city="San Jose",
            arrive_city="Tokyo",
            depart_country="US",
            arrive_country="JP",
            ac_type="A359",
            cruise_alt_ft=35000,
            route_step_km=25.0
        )

        assert request.depart_city == "San Jose"
        assert request.arrive_city == "Tokyo"
        assert request.ac_type == "A359"
        assert request.cruise_alt_ft == 35000
        assert request.route_step_km == 25.0

    @pytest.mark.unit
    def test_plan_request_defaults(self):
        """Test PlanRequest default values."""
        request = PlanRequest(
            depart_city="San Jose",
            arrive_city="Tokyo",
            ac_type="A320"
        )

        assert request.cruise_alt_ft == 35000  # default
        assert request.route_step_km == 25.0   # default
        assert request.backend == "openap"     # default
        assert request.depart_country is None # default
        assert request.arrive_country is None # default

    @pytest.mark.unit
    def test_plan_request_altitude_validation(self):
        """Test altitude validation in PlanRequest."""
        # Valid altitude
        request = PlanRequest(
            depart_city="San Jose",
            arrive_city="Tokyo",
            ac_type="A320",
            cruise_alt_ft=35000
        )
        assert request.cruise_alt_ft == 35000

        # Test boundary values
        request_min = PlanRequest(
            depart_city="San Jose",
            arrive_city="Tokyo",
            ac_type="A320",
            cruise_alt_ft=8000
        )
        assert request_min.cruise_alt_ft == 8000

        request_max = PlanRequest(
            depart_city="San Jose",
            arrive_city="Tokyo",
            ac_type="A320",
            cruise_alt_ft=45000
        )
        assert request_max.cruise_alt_ft == 45000

    @pytest.mark.unit
    def test_plan_request_step_validation(self):
        """Test route step validation in PlanRequest."""
        # Valid step
        request = PlanRequest(
            depart_city="San Jose",
            arrive_city="Tokyo",
            ac_type="A320",
            route_step_km=50.0
        )
        assert request.route_step_km == 50.0


class TestFlightPlanningIntegration:
    """Integration tests for flight planning."""

    @pytest.mark.integration
    @pytest.mark.slow
    def test_complete_flight_plan_sjc_nrt(self, mock_airports_iata, mock_openap_flight_generator, mock_openap_fuel_flow, mock_openap_props):
        """Test complete flight planning from SJC to NRT."""
        with patch('aerospace_mcp.core.OPENAP_AVAILABLE', True):
            with patch('aerospace_mcp.core.FlightGenerator', return_value=mock_openap_flight_generator):
                with patch('aerospace_mcp.core.FuelFlow', return_value=mock_openap_fuel_flow):
                    with patch('aerospace_mcp.core.prop.aircraft', return_value=mock_openap_props):

                        # Test the individual components that would be used in the full API
                        from aerospace_mcp.core import _resolve_endpoint

                        # Resolve airports
                        dep = _resolve_endpoint("San Jose", "US", None, "departure")
                        arr = _resolve_endpoint("Tokyo", "JP", None, "arrival")

                        assert dep.iata == "SJC"
                        assert arr.iata == "NRT"

                        # Calculate route
                        polyline, distance_km = great_circle_points(
                            dep.lat, dep.lon, arr.lat, arr.lon, 25.0
                        )

                        assert 8500 < distance_km < 9500  # Reasonable distance
                        assert len(polyline) > 300       # Should have many points

                        # Get performance estimates
                        estimates, engine_name = estimates_openap("A359", 35000, None, distance_km)

                        assert engine_name == "openap"
                        assert estimates["block"]["time_min"] > 0
                        assert estimates["block"]["fuel_kg"] > 0

    @pytest.mark.integration
    def test_distance_reasonableness_check(self):
        """Test that calculated distances are reasonable for known routes."""
        # Known approximate distances
        test_routes = [
            # (lat1, lon1, lat2, lon2, expected_km_range)
            (37.3626, -121.929, 35.7647, 140.386, (8500, 9500)),  # SJC-NRT
            (37.3626, -121.929, 37.6213, -122.379, (50, 100)),    # SJC-SFO
            (40.6398, -73.7789, 51.4700, -0.4543, (5500, 5600)),  # JFK-LHR
        ]

        for lat1, lon1, lat2, lon2, (min_km, max_km) in test_routes:
            polyline, distance_km = great_circle_points(lat1, lon1, lat2, lon2, 100.0)
            assert min_km < distance_km < max_km, f"Distance {distance_km} not in range {min_km}-{max_km}"
