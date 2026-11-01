"""Tests for flight planning functionality."""

import math

import pandas as pd
import pytest

import aerospace_mcp.core as core
from aerospace_mcp.core import (
    KM_PER_NM,
    NM_PER_KM,
    OpenAPError,
    PlanRequest,
    SegmentEst,
    _fit_profiles_to_route,
    _Profile,
    estimates_openap,
    great_circle_points,
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
    """Tests for OpenAP flight performance estimates (real OpenAP models)."""

    @pytest.mark.unit
    def test_openap_unavailable_error(self, monkeypatch):
        """Test error when OpenAP is not installed."""
        monkeypatch.setattr(core, "OPENAP_AVAILABLE", False)
        with pytest.raises(OpenAPError, match="OpenAP backend unavailable"):
            estimates_openap("A320", 35000, None, 1000.0)

    @pytest.mark.unit
    def test_openap_estimates_a359(self):
        """A359 long-haul estimate: structure, consistency and mass default."""
        estimates, engine_name = estimates_openap("A359", 35000, None, 9000.0)

        assert engine_name == "openap"
        segments = [estimates[k] for k in ("climb", "cruise", "descent")]
        for segment in segments:
            assert segment["time_min"] > 0
            assert segment["distance_km"] > 0
            assert segment["avg_gs_kts"] > 0
            assert segment["fuel_kg"] > 0

        # Segments add up to the route and to the block figures.
        assert sum(s["distance_km"] for s in segments) == pytest.approx(9000.0)
        assert estimates["block"]["time_min"] == pytest.approx(
            sum(s["time_min"] for s in segments)
        )
        assert estimates["block"]["fuel_kg"] == pytest.approx(
            sum(s["fuel_kg"] for s in segments)
        )
        # A 9,000 km A350 sector: roughly 9-11 h block, 60-110 t of fuel.
        assert 540 < estimates["block"]["time_min"] < 660
        assert 60_000 < estimates["block"]["fuel_kg"] < 110_000

        assumptions = estimates["assumptions"]
        assert assumptions["zero_wind"] is True
        assert assumptions["cruise_alt_ft"] == 35000
        assert assumptions["top_of_climb_ft"] == 35000
        assert "profile_note" not in assumptions
        # Default mass is 85 % of OpenAP's MTOW for the type.
        mtow = core._aircraft_props("A359")["limits"]["MTOW"]
        assert assumptions["mass_source"] == "mtow_85pct"
        assert assumptions["mass_kg"] == pytest.approx(0.85 * mtow)

    @pytest.mark.unit
    def test_cruise_time_unit_conversion(self):
        """Regression: cruise time must convert knots -> km/s via KM_PER_NM.

        A prior bug used NM_PER_KM (the reciprocal), inflating cruise time
        and fuel by ~3.43x.
        """
        estimates, _ = estimates_openap("A359", 35000, None, 9000.0)
        cruise = estimates["cruise"]

        expected_min = cruise["distance_km"] / (cruise["avg_gs_kts"] * KM_PER_NM) * 60
        assert cruise["time_min"] == pytest.approx(expected_min, rel=1e-9)
        buggy_min = cruise["distance_km"] / (cruise["avg_gs_kts"] * NM_PER_KM) * 60
        assert cruise["time_min"] < buggy_min / 3

    @pytest.mark.unit
    def test_headwind_increases_cruise_time_and_fuel(self):
        """Headwind slows the cruise; tailwind speeds it up; default unchanged."""
        calm, _ = estimates_openap("A359", 35000, None, 9000.0)
        head, _ = estimates_openap("A359", 35000, None, 9000.0, headwind_kts=50.0)
        tail, _ = estimates_openap("A359", 35000, None, 9000.0, headwind_kts=-50.0)

        assert head["cruise"]["time_min"] > calm["cruise"]["time_min"]
        assert tail["cruise"]["time_min"] < calm["cruise"]["time_min"]
        assert head["cruise"]["fuel_kg"] > calm["cruise"]["fuel_kg"]
        assert tail["cruise"]["fuel_kg"] < calm["cruise"]["fuel_kg"]

        # Ground speed = TAS - headwind; climb and descent are unaffected.
        calm_gs = calm["cruise"]["avg_gs_kts"]
        assert head["cruise"]["avg_gs_kts"] == pytest.approx(calm_gs - 50.0)
        assert tail["cruise"]["avg_gs_kts"] == pytest.approx(calm_gs + 50.0)
        assert head["climb"] == calm["climb"]
        assert head["descent"] == calm["descent"]
        # Fuel flow depends on TAS, not ground speed: fuel scales with time.
        assert head["cruise"]["fuel_kg"] / calm["cruise"]["fuel_kg"] == pytest.approx(
            head["cruise"]["time_min"] / calm["cruise"]["time_min"]
        )
        assert head["assumptions"]["zero_wind"] is False
        assert head["assumptions"]["headwind_kts"] == 50.0
        assert calm["assumptions"]["zero_wind"] is True

    @pytest.mark.unit
    def test_fuel_flow_failure_raises(self, monkeypatch):
        """A broken fuel-flow model must raise, not silently report 0 kg.

        No real input makes OpenAP's fuel model fail, so the failure is
        injected into an otherwise real FuelFlow.
        """
        real_fuel_flow = core._openap("FuelFlow")

        class FailingFuelFlow(real_fuel_flow):
            def enroute(self, *args, **kwargs):
                raise RuntimeError("model exploded")

        monkeypatch.setattr(core, "FuelFlow", FailingFuelFlow)
        with pytest.raises(OpenAPError, match="Fuel flow computation failed"):
            estimates_openap("A359", 35000, None, 9000.0)

    @pytest.mark.unit
    def test_openap_with_explicit_mass(self):
        """An explicit mass is used as given and drives fuel burn."""
        light, _ = estimates_openap("A320", 35000, 55000.0, 1500.0)
        heavy, _ = estimates_openap("A320", 35000, 75000.0, 1500.0)

        assert light["assumptions"]["mass_kg"] == 55000.0
        assert light["assumptions"]["mass_source"] == "user"
        assert heavy["assumptions"]["mass_kg"] == 75000.0
        assert heavy["block"]["fuel_kg"] > light["block"]["fuel_kg"]

    @pytest.mark.unit
    def test_openap_fallback_mass(self, monkeypatch):
        """Without an MTOW from OpenAP's property tables, 60 t is assumed.

        Every type OpenAP can fly also has property data, so the lookup
        failure is injected.
        """
        prop = core._openap("prop")

        def missing(*args, **kwargs):
            raise ValueError("Aircraft not found")

        monkeypatch.setattr(prop, "aircraft", missing)
        estimates, _ = estimates_openap("A320", 35000, None, 5000.0)

        assert estimates["assumptions"]["mass_kg"] == 60000.0
        assert estimates["assumptions"]["mass_source"] == "default_60t"

    @pytest.mark.unit
    def test_unsupported_aircraft_raises(self):
        """A type OpenAP has no model or synonym for is a clean domain error."""
        with pytest.raises(OpenAPError, match="not supported by OpenAP") as exc:
            estimates_openap("C172", 8000, None, 300.0)
        assert exc.value.kind == "unsupported_aircraft"

    @pytest.mark.unit
    def test_descent_starts_at_requested_cruise_altitude(self):
        """The requested altitude is passed to OpenAP's profile generator."""
        fgen = core._cached_model(core._openap("FlightGenerator"), "A320")
        for alt_ft in (25000, 35000):
            descent = core._cached_profile(fgen, "descent", alt_ft, core._SIM_DT_S)
            assert descent.alt[0] == pytest.approx(alt_ft)
            assert descent.alt[-1] == pytest.approx(0.0)

    @pytest.mark.unit
    def test_openap_short_route(self):
        """Short routes cap the top of climb so segments sum to the route.

        Previously a route shorter than climb + descent was billed the full
        climb and descent anyway (segments summed to more than the route).
        """
        estimates, _ = estimates_openap("A320", 35000, None, 190.0)

        seg_km = sum(
            estimates[k]["distance_km"] for k in ("climb", "cruise", "descent")
        )
        assert seg_km == pytest.approx(190.0)
        a = estimates["assumptions"]
        assert 1000 <= a["top_of_climb_ft"] < 35000
        assert "capped" in a["profile_note"]
        assert "scaled" not in a["profile_note"]
        assert a["cruise_alt_ft"] == 35000  # requested altitude is echoed

    @pytest.mark.unit
    def test_openap_tiny_route_scales_profile(self):
        """A route shorter than even the lowest profile is scaled to fit."""
        estimates, _ = estimates_openap("A320", 35000, None, 5.0)

        seg_km = sum(
            estimates[k]["distance_km"] for k in ("climb", "cruise", "descent")
        )
        assert seg_km == pytest.approx(5.0)
        assert estimates["cruise"]["distance_km"] == pytest.approx(0.0)
        assert "scaled" in estimates["assumptions"]["profile_note"]

    @pytest.mark.unit
    def test_cruise_tas_follows_requested_altitude(self):
        """Cruise TAS = M * a(h) at the *requested* altitude, not FL358.

        Regression: the cruise speed came from OpenAP's default-altitude
        cruise() segment, so every cruise altitude reported ~447 kt.
        """
        from aerospace_mcp.core import _openap_aero

        aero = _openap_aero()
        fgen = core._cached_model(core._openap("FlightGenerator"), "A359")
        mach = fgen.wrap.cruise_mach()["default"]

        low, _ = estimates_openap("A359", 25000, None, 9000.0)
        high, _ = estimates_openap("A359", 35000, None, 9000.0)

        for est, alt_ft in ((low, 25000), (high, 35000)):
            expected = mach * aero.vsound(alt_ft * aero.ft) / aero.kts
            assert est["cruise"]["avg_gs_kts"] == pytest.approx(expected, rel=1e-9)
        # Speed of sound falls with altitude in the troposphere.
        assert low["cruise"]["avg_gs_kts"] > high["cruise"]["avg_gs_kts"]

    @pytest.mark.unit
    def test_non_positive_mass_rejected(self):
        """mass_kg <= 0 is an input error (0 used to mean 'missing')."""
        with pytest.raises(ValueError, match="mass_kg"):
            estimates_openap("A320", 35000, 0.0, 1000.0)
        with pytest.raises(ValueError, match="mass_kg"):
            estimates_openap("A320", 35000, -5.0, 1000.0)


def _profile(**cols) -> _Profile:
    return _Profile.from_frame(pd.DataFrame(cols))


class TestFitProfilesToRoute:
    """Tests for the short-route capping of climb/descent profiles."""

    # Climb 0 -> 30,000 ft over 150 km; descent 30,000 -> 0 ft over 150 km.
    CLIMB = {
        "t": [0, 100, 200, 300, 400],
        "s": [0, 20000, 60000, 100000, 150000],
        "altitude": [0, 5000, 15000, 22000, 30000],
        "groundspeed": [150, 250, 300, 350, 420],
        "vertical_rate": [2000, 2000, 1800, 1500, 1200],
    }
    DESCENT = {
        "t": [0, 100, 200, 300, 400],
        "s": [0, 50000, 90000, 130000, 150000],
        "altitude": [30000, 22000, 15000, 5000, 0],
        "groundspeed": [420, 350, 300, 250, 140],
        "vertical_rate": [-1500, -1500, -1500, -1000, -700],
    }

    @pytest.mark.unit
    def test_long_route_is_untouched(self):
        climb, descent = _profile(**self.CLIMB), _profile(**self.DESCENT)
        out = _fit_profiles_to_route(climb, descent, 1000.0)
        assert out == (climb, descent, None, None, 1.0)

    @pytest.mark.unit
    def test_short_route_caps_top_of_climb(self):
        climb, descent = _profile(**self.CLIMB), _profile(**self.DESCENT)
        # 190 km route, but climb + descent need 300 km.
        c, d, top_ft, level_gs, scale = _fit_profiles_to_route(climb, descent, 190.0)

        # Highest climb sample whose climb + matching descent fits: 15,000 ft
        # (60 km climb + 60 km descent from 15,000 ft; 22,000 ft would need
        # 100 + 100 km), leaving 70 km flown level at the 15,000 ft speed.
        assert top_ft == 15000
        assert level_gs == 300
        assert scale == 1.0
        assert c.s[-1] / 1000 == pytest.approx(60.0)
        assert d.s[-1] / 1000 == pytest.approx(60.0)
        assert d.alt[0] == 15000 and d.t[0] == 0 and d.s[0] == 0

    @pytest.mark.unit
    def test_tiny_route_scales_lowest_profile(self):
        climb = _profile(
            t=[0, 100, 200],
            s=[0, 10000, 40000],
            altitude=[0, 2000, 30000],
            groundspeed=[150, 250, 420],
            vertical_rate=[2000, 2000, 1200],
        )
        descent = _profile(
            t=[0, 100, 200],
            s=[0, 30000, 40000],
            altitude=[30000, 2000, 0],
            groundspeed=[420, 250, 140],
            vertical_rate=[-1500, -1000, -700],
        )
        # Lowest airborne profile (2,000 ft) needs 10 + 10 = 20 km.
        c, d, top_ft, _, scale = _fit_profiles_to_route(climb, descent, 5.0)

        assert top_ft == 2000
        assert scale == pytest.approx(0.25)
        assert (c.s[-1] + d.s[-1]) / 1000 == pytest.approx(5.0)
        # Time is scaled with distance, so speeds are preserved.
        assert c.t[-1] == pytest.approx(25.0)


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

        # Just outside the 8,000-45,000 ft range is rejected
        from pydantic import ValidationError

        for bad in (7999, 45001):
            with pytest.raises(ValidationError, match="cruise_alt_ft"):
                PlanRequest(
                    depart_city="San Jose",
                    arrive_city="Tokyo",
                    ac_type="A320",
                    cruise_alt_ft=bad,
                )

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

        # Must be > 1 km and <= 1000 km
        from pydantic import ValidationError

        for bad in (1.0, -10.0, 1000.5):
            with pytest.raises(ValidationError, match="route_step_km"):
                PlanRequest(
                    depart_city="San Jose",
                    arrive_city="Tokyo",
                    ac_type="A320",
                    route_step_km=bad,
                )


class TestFlightPlanningIntegration:
    """Integration tests for flight planning."""

    @pytest.mark.integration
    @pytest.mark.slow
    def test_complete_flight_plan_sjc_nrt(self):
        """Resolve SJC and NRT, route between them and estimate the flight."""
        from aerospace_mcp.core import _resolve_endpoint

        dep = _resolve_endpoint("San Jose", "US", None, "departure")
        arr = _resolve_endpoint("Tokyo", "JP", "NRT", "arrival")

        assert dep.iata == "SJC"
        assert arr.iata == "NRT"

        polyline, distance_km = great_circle_points(
            dep.lat, dep.lon, arr.lat, arr.lon, 25.0
        )

        assert 8200 < distance_km < 8400
        assert len(polyline) == int(math.ceil(distance_km / 25.0)) + 1

        estimates, engine_name = estimates_openap("A359", 35000, None, distance_km)

        assert engine_name == "openap"
        # SJC-NRT is scheduled at ~11 h westbound; zero wind is a bit faster.
        assert 9 * 60 < estimates["block"]["time_min"] < 12 * 60
        seg_km = sum(
            estimates[k]["distance_km"] for k in ("climb", "cruise", "descent")
        )
        assert seg_km == pytest.approx(distance_km)

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
