"""Tests for flight planning functionality."""

import math
from unittest.mock import MagicMock, patch

import pytest

from aerospace_mcp.core import (
    KM_PER_NM,
    NM_PER_KM,
    OpenAPError,
    PlanRequest,
    SegmentEst,
    estimates_openap,
    great_circle_points,
)


def _set_cruise_tas(mock_gen, tas_kts: float = 450.0, alt_ft: float = 35000.0):
    """Give a mocked FlightGenerator a WRAP cruise Mach worth ``tas_kts``.

    estimates_openap derives cruise TAS as M * a(h) at the requested cruise
    altitude from ``fgen.wrap.cruise_mach()``, so the mock needs a Mach
    number rather than a canned cruise DataFrame.
    """
    from openap.extra import aero

    mach = tas_kts * aero.kts / aero.vsound(alt_ft * aero.ft)
    mock_gen.wrap.cruise_mach.return_value = {"default": mach}


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
        assert 8200 < distance_km < 8400

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

        # Distance should be around 49-65 km
        assert 45 < distance_km < 65

        # Should have at least 2 points (start and end)
        assert len(polyline) >= 2

    @pytest.mark.unit
    def test_same_point_calculation(self):
        """Test great circle calculation for same point."""
        lat, lon = 37.3626, -121.929

        polyline, distance_km = great_circle_points(lat, lon, lat, lon, 10.0)

        # Distance should be 0
        assert distance_km == 0.0

        # Should have at least 1 point (may have start and end point)
        assert len(polyline) >= 1
        assert polyline[0] == (lat, lon)

    @pytest.mark.unit
    @pytest.mark.parametrize("step_km", [1.0, 25.0, 50.0, 100.0, 500.0])
    def test_different_step_sizes(self, step_km):
        """Test different step sizes for polyline generation."""
        lat1, lon1 = 37.3626, -121.929  # SJC
        lat2, lon2 = 35.7647, 140.386  # NRT

        polyline, distance_km = great_circle_points(lat1, lon1, lat2, lon2, step_km)

        # Distance should be consistent regardless of step size
        assert 8200 < distance_km < 8400

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
        with patch("aerospace_mcp.core.OPENAP_AVAILABLE", False):
            with pytest.raises(OpenAPError, match="OpenAP backend unavailable"):
                estimates_openap("A320", 35000, None, 1000.0)

    @pytest.mark.unit
    @patch("aerospace_mcp.core.OPENAP_AVAILABLE", True)
    def test_openap_estimates_a359(
        self, mock_openap_flight_generator, mock_openap_fuel_flow, mock_openap_props
    ):
        """Test OpenAP estimates for A359."""
        with patch(
            "aerospace_mcp.core.FlightGenerator",
            return_value=mock_openap_flight_generator,
        ):
            with patch(
                "aerospace_mcp.core.FuelFlow", return_value=mock_openap_fuel_flow
            ):
                with patch(
                    "aerospace_mcp.core.prop.aircraft", return_value=mock_openap_props
                ):
                    estimates, engine_name = estimates_openap(
                        "A359", 35000, None, 9000.0
                    )

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
    @patch("aerospace_mcp.core.OPENAP_AVAILABLE", True)
    def test_cruise_time_unit_conversion(
        self, mock_openap_flight_generator, mock_openap_fuel_flow, mock_openap_props
    ):
        """Regression: cruise time must convert knots -> km/s via KM_PER_NM.

        A prior bug used NM_PER_KM (the reciprocal), inflating cruise time
        and fuel by ~3.43x.
        """
        with patch(
            "aerospace_mcp.core.FlightGenerator",
            return_value=mock_openap_flight_generator,
        ):
            with patch(
                "aerospace_mcp.core.FuelFlow", return_value=mock_openap_fuel_flow
            ):
                with patch(
                    "aerospace_mcp.core.prop.aircraft", return_value=mock_openap_props
                ):
                    route_km = 9000.0
                    _set_cruise_tas(mock_openap_flight_generator, 450.0)
                    estimates, _ = estimates_openap("A359", 35000, None, route_km)

                    # Mock fixtures: climb covers 10 km, descent 12 km,
                    # cruise TAS 450 kts (M * a at FL350).
                    d_remaining = route_km - (10.0 + 12.0)
                    expected_min = d_remaining / (450 * KM_PER_NM) * 60.0
                    assert estimates["cruise"]["time_min"] == pytest.approx(
                        expected_min, rel=1e-6
                    )
                    # The buggy conversion produced ~3.43x this value; make
                    # sure we are nowhere near it.
                    assert estimates["cruise"]["time_min"] < expected_min * 2

    @pytest.mark.unit
    @patch("aerospace_mcp.core.OPENAP_AVAILABLE", True)
    def test_headwind_increases_cruise_time_and_fuel(
        self, mock_openap_flight_generator, mock_openap_fuel_flow, mock_openap_props
    ):
        """Headwind slows the cruise; tailwind speeds it up; default unchanged."""
        with patch(
            "aerospace_mcp.core.FlightGenerator",
            return_value=mock_openap_flight_generator,
        ):
            with patch(
                "aerospace_mcp.core.FuelFlow", return_value=mock_openap_fuel_flow
            ):
                with patch(
                    "aerospace_mcp.core.prop.aircraft", return_value=mock_openap_props
                ):
                    _set_cruise_tas(mock_openap_flight_generator, 450.0)
                    calm, _ = estimates_openap("A359", 35000, None, 9000.0)
                    head, _ = estimates_openap(
                        "A359", 35000, None, 9000.0, headwind_kts=50.0
                    )
                    tail, _ = estimates_openap(
                        "A359", 35000, None, 9000.0, headwind_kts=-50.0
                    )

                    assert head["cruise"]["time_min"] > calm["cruise"]["time_min"]
                    assert tail["cruise"]["time_min"] < calm["cruise"]["time_min"]
                    assert head["cruise"]["fuel_kg"] > calm["cruise"]["fuel_kg"]

                    # Cruise GS 450 kts, 50 kt headwind -> 400 kts effective.
                    assert head["cruise"]["avg_gs_kts"] == pytest.approx(400.0)
                    assert head["assumptions"]["zero_wind"] is False
                    assert calm["assumptions"]["zero_wind"] is True

    @pytest.mark.unit
    @patch("aerospace_mcp.core.OPENAP_AVAILABLE", True)
    def test_fuel_flow_failure_raises(
        self, mock_openap_flight_generator, mock_openap_props
    ):
        """A broken fuel-flow model must raise, not silently report 0 kg."""
        failing_ff = MagicMock()
        failing_ff.enroute.side_effect = RuntimeError("model exploded")
        with patch(
            "aerospace_mcp.core.FlightGenerator",
            return_value=mock_openap_flight_generator,
        ):
            with patch("aerospace_mcp.core.FuelFlow", return_value=failing_ff):
                with patch(
                    "aerospace_mcp.core.prop.aircraft", return_value=mock_openap_props
                ):
                    with pytest.raises(OpenAPError, match="Fuel flow"):
                        estimates_openap("A359", 35000, None, 9000.0)

    @pytest.mark.unit
    @patch("aerospace_mcp.core.OPENAP_AVAILABLE", True)
    def test_mass_source_tracking(
        self, mock_openap_flight_generator, mock_openap_fuel_flow, mock_openap_props
    ):
        """assumptions.mass_source records how the mass was determined."""
        with patch(
            "aerospace_mcp.core.FlightGenerator",
            return_value=mock_openap_flight_generator,
        ):
            with patch(
                "aerospace_mcp.core.FuelFlow", return_value=mock_openap_fuel_flow
            ):
                with patch(
                    "aerospace_mcp.core.prop.aircraft", return_value=mock_openap_props
                ):
                    est, _ = estimates_openap("A359", 35000, 75000.0, 9000.0)
                    assert est["assumptions"]["mass_source"] == "user"

                    est, _ = estimates_openap("A359", 35000, None, 9000.0)
                    assert est["assumptions"]["mass_source"] == "mtow_85pct"

                with patch(
                    "aerospace_mcp.core.FuelFlow", return_value=mock_openap_fuel_flow
                ):
                    with patch(
                        "aerospace_mcp.core.prop.aircraft",
                        side_effect=Exception("Aircraft not found"),
                    ):
                        est, _ = estimates_openap("UNKNOWN", 35000, None, 9000.0)
                        assert est["assumptions"]["mass_source"] == "default_60t"

    @pytest.mark.unit
    @patch("aerospace_mcp.core.OPENAP_AVAILABLE", True)
    def test_openap_with_explicit_mass(
        self, mock_openap_flight_generator, mock_openap_fuel_flow
    ):
        """Test OpenAP estimates with explicit mass."""
        with patch(
            "aerospace_mcp.core.FlightGenerator",
            return_value=mock_openap_flight_generator,
        ):
            with patch(
                "aerospace_mcp.core.FuelFlow", return_value=mock_openap_fuel_flow
            ):
                test_mass = 75000.0  # kg
                estimates, _ = estimates_openap("A320", 35000, test_mass, 5000.0)

                assert estimates["assumptions"]["mass_kg"] == test_mass

    @pytest.mark.unit
    @patch("aerospace_mcp.core.OPENAP_AVAILABLE", True)
    def test_openap_fallback_mass(
        self, mock_openap_flight_generator, mock_openap_fuel_flow
    ):
        """Test OpenAP estimates with fallback mass when aircraft props fail."""
        with patch(
            "aerospace_mcp.core.FlightGenerator",
            return_value=mock_openap_flight_generator,
        ):
            with patch(
                "aerospace_mcp.core.FuelFlow", return_value=mock_openap_fuel_flow
            ):
                with patch(
                    "aerospace_mcp.core.prop.aircraft",
                    side_effect=Exception("Aircraft not found"),
                ):
                    estimates, _ = estimates_openap("UNKNOWN", 35000, None, 5000.0)

                    # Should use fallback mass
                    assert estimates["assumptions"]["mass_kg"] == 60000.0

    @pytest.mark.unit
    @patch("aerospace_mcp.core.OPENAP_AVAILABLE", True)
    def test_openap_cruise_altitude_handling(
        self, mock_openap_flight_generator, mock_openap_fuel_flow
    ):
        """Test OpenAP estimates with different cruise altitudes."""
        mock_gen = mock_openap_flight_generator
        mock_gen.climb.side_effect = [
            TypeError("alt_cr not supported"),
            mock_gen.climb.return_value,
        ]
        mock_gen.descent.side_effect = [
            TypeError("alt_cr not supported"),
            mock_gen.descent.return_value,
        ]

        with patch("aerospace_mcp.core.FlightGenerator", return_value=mock_gen):
            with patch(
                "aerospace_mcp.core.FuelFlow", return_value=mock_openap_fuel_flow
            ):
                # Should handle TypeError gracefully and call without alt_cr
                estimates, _ = estimates_openap("A320", 45000, None, 5000.0)

                # Verify both climb and descent were called twice (first with alt_cr, then without)
                assert mock_gen.climb.call_count == 2
                assert mock_gen.descent.call_count == 2

    @pytest.mark.unit
    @patch("aerospace_mcp.core.OPENAP_AVAILABLE", True)
    def test_openap_short_route(self, mock_openap_fuel_flow):
        """Short routes cap the top of climb so segments sum to the route.

        Previously a route shorter than climb + descent was billed the full
        climb and descent anyway (segments summed to more than the route).
        """
        import pandas as pd

        mock_gen = MagicMock()
        _set_cruise_tas(mock_gen, 450.0)
        # Climb: 0 -> 30,000 ft over 150 km; descent: 30,000 -> 0 ft, 150 km.
        mock_gen.climb.return_value = pd.DataFrame(
            {
                "t": [0, 100, 200, 300, 400],
                "s": [0, 20000, 60000, 100000, 150000],
                "altitude": [0, 5000, 15000, 22000, 30000],
                "groundspeed": [150, 250, 300, 350, 420],
                "vertical_rate": [2000, 2000, 1800, 1500, 1200],
            }
        )
        mock_gen.descent.return_value = pd.DataFrame(
            {
                "t": [0, 100, 200, 300, 400],
                "s": [0, 50000, 90000, 130000, 150000],
                "altitude": [30000, 22000, 15000, 5000, 0],
                "groundspeed": [420, 350, 300, 250, 140],
                "vertical_rate": [-1500, -1500, -1500, -1000, -700],
            }
        )

        with patch("aerospace_mcp.core.FlightGenerator", return_value=mock_gen):
            with patch(
                "aerospace_mcp.core.FuelFlow", return_value=mock_openap_fuel_flow
            ):
                # 190 km route, but climb + descent need 300 km.
                estimates, _ = estimates_openap("A320", 35000, None, 190.0)

        seg_km = sum(
            estimates[k]["distance_km"] for k in ("climb", "cruise", "descent")
        )
        assert seg_km == pytest.approx(190.0)
        a = estimates["assumptions"]
        # Highest climb sample whose climb + matching descent fits: 15,000 ft
        # (60 km climb + 60 km descent from 15,000 ft; 22,000 ft would need
        # 100 + 100 km), leaving 70 km flown level.
        assert a["top_of_climb_ft"] == 15000
        assert estimates["climb"]["distance_km"] == pytest.approx(60.0)
        assert estimates["descent"]["distance_km"] == pytest.approx(60.0)
        assert estimates["cruise"]["distance_km"] == pytest.approx(70.0)
        assert estimates["cruise"]["avg_gs_kts"] == pytest.approx(300.0)
        assert "capped" in a["profile_note"]
        assert a["cruise_alt_ft"] == 35000  # requested altitude is echoed

    @pytest.mark.unit
    @patch("aerospace_mcp.core.OPENAP_AVAILABLE", True)
    def test_openap_tiny_route_scales_profile(self, mock_openap_fuel_flow):
        """A route shorter than even the lowest profile is scaled to fit."""
        import pandas as pd

        mock_gen = MagicMock()
        _set_cruise_tas(mock_gen, 450.0)
        mock_gen.climb.return_value = pd.DataFrame(
            {
                "t": [0, 100, 200],
                "s": [0, 10000, 40000],
                "altitude": [0, 2000, 30000],
                "groundspeed": [150, 250, 420],
                "vertical_rate": [2000, 2000, 1200],
            }
        )
        mock_gen.descent.return_value = pd.DataFrame(
            {
                "t": [0, 100, 200],
                "s": [0, 30000, 40000],
                "altitude": [30000, 2000, 0],
                "groundspeed": [420, 250, 140],
                "vertical_rate": [-1500, -1000, -700],
            }
        )
        with patch("aerospace_mcp.core.FlightGenerator", return_value=mock_gen):
            with patch(
                "aerospace_mcp.core.FuelFlow", return_value=mock_openap_fuel_flow
            ):
                # Lowest airborne profile (2,000 ft) needs 10 + 10 = 20 km.
                estimates, _ = estimates_openap("A320", 35000, None, 5.0)

        seg_km = sum(
            estimates[k]["distance_km"] for k in ("climb", "cruise", "descent")
        )
        assert seg_km == pytest.approx(5.0)
        assert estimates["cruise"]["distance_km"] == pytest.approx(0.0)
        assert "scaled" in estimates["assumptions"]["profile_note"]

    @pytest.mark.unit
    @patch("aerospace_mcp.core.OPENAP_AVAILABLE", True)
    def test_cruise_tas_follows_requested_altitude(
        self, mock_openap_flight_generator, mock_openap_fuel_flow, mock_openap_props
    ):
        """Cruise TAS = M * a(h) at the *requested* altitude, not FL358.

        Regression: the cruise speed came from OpenAP's default-altitude
        cruise() segment, so every cruise altitude reported ~447 kt.
        """
        from openap.extra import aero

        mock_openap_flight_generator.wrap.cruise_mach.return_value = {"default": 0.78}
        with patch(
            "aerospace_mcp.core.FlightGenerator",
            return_value=mock_openap_flight_generator,
        ):
            with patch(
                "aerospace_mcp.core.FuelFlow", return_value=mock_openap_fuel_flow
            ):
                with patch(
                    "aerospace_mcp.core.prop.aircraft", return_value=mock_openap_props
                ):
                    low, _ = estimates_openap("A359", 25000, None, 9000.0)
                    high, _ = estimates_openap("A359", 35000, None, 9000.0)

        for est, alt_ft in ((low, 25000), (high, 35000)):
            expected = 0.78 * aero.vsound(alt_ft * aero.ft) / aero.kts
            assert est["cruise"]["avg_gs_kts"] == pytest.approx(expected, rel=1e-9)
        # Speed of sound falls with altitude in the troposphere.
        assert low["cruise"]["avg_gs_kts"] > high["cruise"]["avg_gs_kts"]
        mock_openap_flight_generator.cruise.assert_not_called()

    @pytest.mark.unit
    @patch("aerospace_mcp.core.OPENAP_AVAILABLE", True)
    def test_non_positive_mass_rejected(self):
        """mass_kg <= 0 is an input error (0 used to mean 'missing')."""
        with pytest.raises(ValueError, match="mass_kg"):
            estimates_openap("A320", 35000, 0.0, 1000.0)
        with pytest.raises(ValueError, match="mass_kg"):
            estimates_openap("A320", 35000, -5.0, 1000.0)


class TestSegmentEst:
    """Tests for SegmentEst model."""

    @pytest.mark.unit
    def test_segment_est_creation(self):
        """Test SegmentEst model creation."""
        segment = SegmentEst(
            time_min=120.5, distance_km=850.0, avg_gs_kts=420.0, fuel_kg=2500.0
        )

        assert segment.time_min == 120.5
        assert segment.distance_km == 850.0
        assert segment.avg_gs_kts == 420.0
        assert segment.fuel_kg == 2500.0

    @pytest.mark.unit
    def test_segment_est_serialization(self):
        """Test SegmentEst model serialization."""
        segment = SegmentEst(
            time_min=60.0, distance_km=500.0, avg_gs_kts=400.0, fuel_kg=1200.0
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
            route_step_km=25.0,
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
            depart_city="San Jose", arrive_city="Tokyo", ac_type="A320"
        )

        assert request.cruise_alt_ft == 35000  # default
        assert request.route_step_km == 25.0  # default
        assert request.backend == "openap"  # default
        assert request.depart_country is None  # default
        assert request.arrive_country is None  # default

    @pytest.mark.unit
    def test_plan_request_altitude_validation(self):
        """Test altitude validation in PlanRequest."""
        # Valid altitude
        request = PlanRequest(
            depart_city="San Jose",
            arrive_city="Tokyo",
            ac_type="A320",
            cruise_alt_ft=35000,
        )
        assert request.cruise_alt_ft == 35000

        # Test boundary values
        request_min = PlanRequest(
            depart_city="San Jose",
            arrive_city="Tokyo",
            ac_type="A320",
            cruise_alt_ft=8000,
        )
        assert request_min.cruise_alt_ft == 8000

        request_max = PlanRequest(
            depart_city="San Jose",
            arrive_city="Tokyo",
            ac_type="A320",
            cruise_alt_ft=45000,
        )
        assert request_max.cruise_alt_ft == 45000

    @pytest.mark.unit
    def test_plan_request_mass_must_be_positive(self):
        """mass_kg must be > 0 when given."""
        from pydantic import ValidationError

        for bad in (0, -5):
            with pytest.raises(ValidationError):
                PlanRequest(
                    depart_city="San Jose",
                    arrive_city="Tokyo",
                    ac_type="A320",
                    mass_kg=bad,
                )

    @pytest.mark.unit
    def test_plan_request_step_validation(self):
        """Test route step validation in PlanRequest."""
        # Valid step
        request = PlanRequest(
            depart_city="San Jose",
            arrive_city="Tokyo",
            ac_type="A320",
            route_step_km=50.0,
        )
        assert request.route_step_km == 50.0


class TestFlightPlanningIntegration:
    """Integration tests for flight planning."""

    @pytest.mark.integration
    @pytest.mark.slow
    def test_complete_flight_plan_sjc_nrt(
        self,
        mock_airports_iata,
        mock_openap_flight_generator,
        mock_openap_fuel_flow,
        mock_openap_props,
    ):
        """Test complete flight planning from SJC to NRT."""
        with patch("aerospace_mcp.core.OPENAP_AVAILABLE", True):
            with patch(
                "aerospace_mcp.core.FlightGenerator",
                return_value=mock_openap_flight_generator,
            ):
                with patch(
                    "aerospace_mcp.core.FuelFlow", return_value=mock_openap_fuel_flow
                ):
                    with patch(
                        "aerospace_mcp.core.prop.aircraft",
                        return_value=mock_openap_props,
                    ):
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

                        assert 8200 < distance_km < 8400  # Reasonable distance
                        assert len(polyline) > 300  # Should have many points

                        # Get performance estimates
                        estimates, engine_name = estimates_openap(
                            "A359", 35000, None, distance_km
                        )

                        assert engine_name == "openap"
                        assert estimates["block"]["time_min"] > 0
                        assert estimates["block"]["fuel_kg"] > 0

    @pytest.mark.integration
    def test_distance_reasonableness_check(self):
        """Test that calculated distances are reasonable for known routes."""
        # Known approximate distances
        test_routes = [
            # (lat1, lon1, lat2, lon2, expected_km_range)
            (37.3626, -121.929, 35.7647, 140.386, (8200, 8400)),  # SJC-NRT
            (37.3626, -121.929, 37.6213, -122.379, (45, 65)),  # SJC-SFO
            (40.6398, -73.7789, 51.4700, -0.4543, (5500, 5600)),  # JFK-LHR
        ]

        for lat1, lon1, lat2, lon2, (min_km, max_km) in test_routes:
            polyline, distance_km = great_circle_points(lat1, lon1, lat2, lon2, 100.0)
            assert min_km < distance_km < max_km, (
                f"Distance {distance_km} not in range {min_km}-{max_km}"
            )
