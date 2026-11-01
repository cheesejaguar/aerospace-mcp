"""Tests for multi-leg planning, wind-aware estimates, and the aircraft DB tool.

All tests use the real airport database, great-circle routing and OpenAP.
"""

from __future__ import annotations

import json
import math

import pytest

import aerospace_mcp.tools.core as core_tools
from aerospace_mcp.core import (
    AirportResolutionError,
    FlightPlanError,
    estimates_openap,
    great_circle_points,
    plan_multi_leg,
)


class TestPlanMultiLeg:
    def test_two_legs_aggregate(self):
        result = plan_multi_leg(
            [{"iata": "SFO"}, {"iata": "LAX"}, {"city": "Seattle", "country": "US"}],
            ac_type="A320",
        )

        legs = result["legs"]
        assert [leg["leg"] for leg in legs] == [1, 2]
        assert [leg["depart"]["iata"] for leg in legs] == ["SFO", "LAX"]
        assert [leg["arrive"]["iata"] for leg in legs] == ["LAX", "SEA"]

        # Each leg is planned exactly like a single-leg route
        sfo, lax = legs[0]["depart"], legs[0]["arrive"]
        _, sfo_lax_km = great_circle_points(
            sfo["lat"], sfo["lon"], lax["lat"], lax["lon"], 25.0
        )
        assert legs[0]["distance_km"] == pytest.approx(sfo_lax_km)
        single, _ = estimates_openap("A320", 35000, None, sfo_lax_km)
        assert legs[0]["estimates"]["block"] == pytest.approx(single["block"])

        totals = result["totals"]
        assert totals["legs"] == 2
        assert totals["distance_km"] == pytest.approx(
            sum(leg["distance_km"] for leg in legs)
        )
        assert totals["time_min"] == pytest.approx(
            sum(leg["estimates"]["block"]["time_min"] for leg in legs)
        )
        assert totals["fuel_kg"] == pytest.approx(
            sum(leg["estimates"]["block"]["fuel_kg"] for leg in legs)
        )
        # SFO-LAX ~540 km, LAX-SEA ~1,540 km
        assert 2000 < totals["distance_km"] < 2150
        assert result["assumptions"]["constant_mass_between_legs"] is True

    def test_too_few_waypoints(self):
        with pytest.raises(FlightPlanError, match="at least 2"):
            plan_multi_leg([{"iata": "SFO"}], ac_type="A320")

    def test_too_many_waypoints(self):
        with pytest.raises(FlightPlanError, match="at most"):
            plan_multi_leg([{"iata": "SFO"}] * 11, ac_type="A320")

    def test_missing_city_and_iata(self):
        with pytest.raises(FlightPlanError, match="city.*iata"):
            plan_multi_leg([{"iata": "SFO"}, {}], ac_type="A320")

    def test_repeated_consecutive_airport(self):
        with pytest.raises(FlightPlanError, match="both SFO"):
            plan_multi_leg([{"iata": "SFO"}, {"iata": "SFO"}], ac_type="A320")

    def test_unresolvable_waypoint_names_the_stop(self):
        with pytest.raises(AirportResolutionError, match="waypoint 1"):
            plan_multi_leg([{"iata": "SFO"}, {"city": "Atlantis"}], ac_type="A320")

    def test_invalid_aircraft_parameters(self):
        with pytest.raises(FlightPlanError, match="cruise_alt_ft"):
            plan_multi_leg(
                [{"iata": "SFO"}, {"iata": "LAX"}], ac_type="A320", cruise_alt_ft=60000
            )


class TestPlanMultiLegTool:
    def test_tool_returns_json(self):
        out = core_tools.plan_multi_leg_flight([{"iata": "SFO"}, {"iata": "LAX"}])
        data = json.loads(out)
        assert data["totals"]["legs"] == 1
        assert data["legs"][0]["arrive"]["iata"] == "LAX"

    def test_tool_stringifies_errors(self):
        out = core_tools.plan_multi_leg_flight([])
        assert "at least 2" in out
        assert not out.startswith("{")


def _east_west_plan(wind=None):
    # JFK -> LHR departs on a north-easterly initial bearing (~51 deg)
    return json.loads(
        core_tools.plan_flight(
            {"iata": "JFK"}, {"iata": "LHR"}, {"ac_type": "A320"}, None, wind=wind
        )
    )


class TestWindAwarePlanning:
    def test_headwind_passed_to_estimates(self):
        calm = _east_west_plan()
        # Wind FROM 270 (a westerly) is a tailwind on an eastbound route.
        windy = _east_west_plan({"wind_speed_kts": 50.0, "wind_direction_deg": 270.0})

        bearing = windy["route"]["initial_bearing_deg"]
        expected = 50.0 * math.cos(math.radians(270.0 - bearing))
        assert expected < 0  # tailwind
        assert windy["wind"]["headwind_component_kts"] == pytest.approx(expected)

        # The component reaches the performance estimate
        assert windy["performance"]["assumptions"]["headwind_kts"] == pytest.approx(
            expected
        )
        assert windy["performance"]["cruise"]["avg_gs_kts"] == pytest.approx(
            calm["performance"]["cruise"]["avg_gs_kts"] - expected
        )
        assert (
            windy["performance"]["block"]["time_min"]
            < calm["performance"]["block"]["time_min"]
        )

    def test_no_wind_default_unchanged(self):
        data = _east_west_plan()

        assert "wind" not in data
        assert data["performance"]["assumptions"]["headwind_kts"] == 0.0
        assert data["performance"]["assumptions"]["zero_wind"] is True


class TestAircraftDatabase:
    def test_openap_unavailable(self, monkeypatch):
        monkeypatch.setattr(core_tools, "OPENAP_AVAILABLE", False)
        out = core_tools.get_aircraft_database()
        assert "not available" in out

    def test_list_and_search(self):
        from openap import prop

        listing = json.loads(core_tools.get_aircraft_database())
        assert listing["total_aircraft"] == len(prop.available_aircraft())
        assert listing["aircraft_types"] == sorted(listing["aircraft_types"])
        assert {"A320", "B738"} <= set(listing["aircraft_types"])

        search = json.loads(core_tools.get_aircraft_database("a32"))
        assert search["matches"] == ["A320", "A321"]
        assert "aircraft" not in search  # several matches: no detail

        exact = json.loads(core_tools.get_aircraft_database("A320"))
        assert exact["aircraft"]["mtow_kg"] == 78000
        assert exact["aircraft"]["engine"] == "CFM56-5B4"
        assert exact["aircraft"]["cruise_mach"] == 0.78

    def test_unknown_aircraft(self):
        out = json.loads(core_tools.get_aircraft_database("ZZZZ"))
        assert out["matches"] == []
        assert "aircraft" not in out
