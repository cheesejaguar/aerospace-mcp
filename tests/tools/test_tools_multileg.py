"""Tests for multi-leg planning, wind-aware estimates, and the aircraft DB tool."""

from __future__ import annotations

import json
import math

import pytest

import aerospace_mcp.core as core
import aerospace_mcp.tools.core as core_tools
from aerospace_mcp.core import AirportOut, FlightPlanError, plan_multi_leg


def _airport(iata: str, lon: float = 0.0) -> AirportOut:
    return AirportOut(
        iata=iata,
        icao="K" + iata,
        name=f"{iata} International Airport",
        city=f"City{iata}",
        country="US",
        lat=0.0,
        lon=lon,
        tz=None,
    )


@pytest.fixture
def stub_planning(monkeypatch):
    """Stub airport resolution, routing, and performance in core."""
    airports = {
        "AAA": _airport("AAA", 0.0),
        "BBB": _airport("BBB", 5.0),
        "CCC": _airport("CCC", 10.0),
    }

    def resolve(city, country, prefer_iata, role):
        key = prefer_iata or city[-3:]
        return airports[key]

    monkeypatch.setattr(core, "_resolve_endpoint", resolve)
    monkeypatch.setattr(
        core,
        "great_circle_points",
        lambda lat1, lon1, lat2, lon2, step: ([(lat1, lon1), (lat2, lon2)], 500.0),
    )
    monkeypatch.setattr(
        core,
        "estimates_openap",
        lambda ac, alt, mass, dist, headwind_kts=0.0: (
            {"block": {"time_min": 60.0, "fuel_kg": 2000.0}},
            "openap",
        ),
    )
    return airports


class TestPlanMultiLeg:
    def test_two_legs_aggregate(self, stub_planning):
        result = plan_multi_leg(
            [{"iata": "AAA"}, {"iata": "BBB"}, {"iata": "CCC"}],
            ac_type="A320",
        )

        assert result["totals"]["legs"] == 2
        assert result["totals"]["distance_km"] == pytest.approx(1000.0)
        assert result["totals"]["time_min"] == pytest.approx(120.0)
        assert result["totals"]["fuel_kg"] == pytest.approx(4000.0)
        assert [leg["leg"] for leg in result["legs"]] == [1, 2]
        assert result["legs"][0]["depart"]["iata"] == "AAA"
        assert result["legs"][1]["arrive"]["iata"] == "CCC"

    def test_too_few_waypoints(self):
        with pytest.raises(FlightPlanError, match="at least 2"):
            plan_multi_leg([{"iata": "AAA"}], ac_type="A320")

    def test_too_many_waypoints(self):
        with pytest.raises(FlightPlanError, match="at most"):
            plan_multi_leg([{"iata": "AAA"}] * 11, ac_type="A320")

    def test_missing_city_and_iata(self):
        with pytest.raises(FlightPlanError, match="city.*iata"):
            plan_multi_leg([{"iata": "AAA"}, {}], ac_type="A320")

    def test_repeated_consecutive_airport(self, stub_planning):
        with pytest.raises(FlightPlanError, match="both AAA"):
            plan_multi_leg([{"iata": "AAA"}, {"iata": "AAA"}], ac_type="A320")


class TestPlanMultiLegTool:
    def test_tool_returns_json(self, monkeypatch):
        monkeypatch.setattr(
            core,
            "plan_multi_leg",
            lambda waypoints, ac_type, cruise_alt_ft, mass_kg, route_step_km: {
                "legs": [],
                "totals": {"legs": 0},
            },
        )
        out = core_tools.plan_multi_leg_flight([{"iata": "AAA"}, {"iata": "BBB"}])
        assert json.loads(out)["totals"]["legs"] == 0

    def test_tool_stringifies_errors(self, monkeypatch):
        def boom(*args, **kwargs):
            raise FlightPlanError("waypoints must be a list of at least 2 entries")

        monkeypatch.setattr(core, "plan_multi_leg", boom)
        out = core_tools.plan_multi_leg_flight([])
        assert "at least 2" in out


class TestWindAwarePlanning:
    def test_headwind_passed_to_estimates(self, monkeypatch):
        captured = {}

        monkeypatch.setattr(
            core_tools,
            "_resolve_endpoint",
            lambda city, country, iata=None, role=None: _airport(iata or "AAA"),
        )
        monkeypatch.setattr(
            core_tools,
            "great_circle_points",
            lambda a, b, c, d, step_km: ([(0.0, 0.0), (0.0, 5.0)], 500.0),
        )
        monkeypatch.setattr(core_tools, "OPENAP_AVAILABLE", True)

        def fake_estimates(ac, alt, mass, dist, headwind_kts=0.0):
            captured["headwind"] = headwind_kts
            return {"block": {"time_min": 1.0, "fuel_kg": 1.0}}, "openap"

        monkeypatch.setattr(core_tools, "estimates_openap", fake_estimates)

        out = core_tools.plan_flight(
            {"city": "CityAAA", "iata": "AAA"},
            {"city": "CityBBB", "iata": "BBB"},
            {"ac_type": "A320"},
            None,
            wind={"wind_speed_kts": 50.0, "wind_direction_deg": 90.0},
        )
        data = json.loads(out)

        # Route is due east (bearing 90); wind FROM 090 is a pure headwind.
        bearing = data["route"]["initial_bearing_deg"]
        expected = 50.0 * math.cos(math.radians(90.0 - bearing))
        assert captured["headwind"] == pytest.approx(expected)
        assert data["wind"]["headwind_component_kts"] == pytest.approx(expected)

    def test_no_wind_default_unchanged(self, monkeypatch):
        captured = {}

        monkeypatch.setattr(
            core_tools,
            "_resolve_endpoint",
            lambda city, country, iata=None, role=None: _airport(iata or "AAA"),
        )
        monkeypatch.setattr(
            core_tools,
            "great_circle_points",
            lambda a, b, c, d, step_km: ([(0.0, 0.0), (0.0, 5.0)], 500.0),
        )
        monkeypatch.setattr(core_tools, "OPENAP_AVAILABLE", True)

        def fake_estimates(ac, alt, mass, dist, headwind_kts=0.0):
            captured["headwind"] = headwind_kts
            return {"block": {"time_min": 1.0, "fuel_kg": 1.0}}, "openap"

        monkeypatch.setattr(core_tools, "estimates_openap", fake_estimates)

        out = core_tools.plan_flight(
            {"city": "CityAAA", "iata": "AAA"},
            {"city": "CityBBB", "iata": "BBB"},
            {"ac_type": "A320"},
        )
        data = json.loads(out)

        assert captured["headwind"] == 0.0
        assert "wind" not in data


class TestAircraftDatabase:
    def test_openap_unavailable(self, monkeypatch):
        monkeypatch.setattr(core_tools, "OPENAP_AVAILABLE", False)
        out = core_tools.get_aircraft_database()
        assert "not available" in out

    def test_list_and_search(self, monkeypatch):
        class FakeProp:
            @staticmethod
            def available_aircraft():
                return ["a320", "a321", "b738"]

            @staticmethod
            def aircraft(ac_type, use_synonym=False):
                assert ac_type == "A320"
                return {
                    "limits": {"MTOW": 78000.0, "OEW": 42600.0, "MFC": 23859.0},
                    "engine": {"default": "CFM56-5B4"},
                    "wing": {"area": 122.6, "span": 35.8},
                    "cruise": {"mach": 0.78, "height": 11000.0},
                }

        monkeypatch.setattr(core_tools, "OPENAP_AVAILABLE", True)
        monkeypatch.setattr(core, "prop", FakeProp, raising=False)

        listing = json.loads(core_tools.get_aircraft_database())
        assert listing["total_aircraft"] == 3
        assert "A320" in listing["aircraft_types"]

        search = json.loads(core_tools.get_aircraft_database("a32"))
        assert search["matches"] == ["A320", "A321"]

        exact = json.loads(core_tools.get_aircraft_database("A320"))
        assert exact["aircraft"]["mtow_kg"] == 78000.0
        assert exact["aircraft"]["engine"] == "CFM56-5B4"
