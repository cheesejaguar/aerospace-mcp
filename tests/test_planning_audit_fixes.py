"""End-to-end regression tests for the flight-planning audit fixes.

Unlike most planning tests these run unstubbed, against the real bundled
airport database and the real OpenAP models, so they exercise airport
ranking, the MCP tool wrappers, the HTTP layer and the OpenAP caches
together.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time

import pytest
from fastapi.testclient import TestClient

import aerospace_mcp.core as core
import aerospace_mcp.http_app as main_module
from aerospace_mcp.core import (
    FlightPlanError,
    PlanRequest,
    _find_city_airports,
    create_flight_plan,
    estimates_openap,
    plan_multi_leg,
)
from aerospace_mcp.tools import core as core_tools

pytestmark = pytest.mark.skipif(
    not core.OPENAP_AVAILABLE, reason="OpenAP is required for these tests"
)


def _plan(*args, **kwargs) -> dict:
    out = core_tools.plan_flight(*args, **kwargs)
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        pytest.fail(f"plan_flight returned an error: {out}")


def _segments_km(est: dict) -> float:
    return sum(est[k]["distance_km"] for k in ("climb", "cruise", "descent"))


# ---------------------------------------------------------------------------
# Airport ranking
# ---------------------------------------------------------------------------


class TestCityRanking:
    @pytest.mark.parametrize(
        "city, country, expected",
        [
            ("Nice", None, {"NCE"}),  # was CND (Constanta) via "Eunice"
            ("Rome", None, {"FCO"}),  # was Rome, NY
            ("Paris", None, {"CDG"}),
            ("Tokyo", None, {"HND", "NRT"}),
            ("New York", None, {"JFK", "LGA", "EWR"}),
            ("Male", None, {"MLE"}),  # accent folding: "Malé"; was DAV (Panama)
            ("Delhi", None, {"DEL"}),  # city "New Delhi"; was no result
            ("San Jose", None, {"SJC"}),  # namesake + International tie-break
            ("San Jose", "CR", {"SJO"}),
            ("San Jose", "us", {"SJC"}),
            ("Rome", "US", {"RME"}),
            ("Berlin", None, {"BER"}),  # not the closed TXL
            ("Toronto", None, {"YYZ"}),  # filed under Mississauga
        ],
    )
    def test_top_match(self, city, country, expected):
        results = _find_city_airports(city, country)
        assert results, f"no airport for {city}"
        assert results[0].iata in expected

    def test_london_resolves_to_great_britain(self):
        results = _find_city_airports("London")
        assert results[0].iata == "LHR"
        assert results[0].country == "GB"
        # Groton (New London), CT used to win on a name-substring match
        assert "GON" not in [a.iata for a in results]

    def test_substring_inside_word_does_not_match(self):
        assert "UCE" not in [a.iata for a in _find_city_airports("Nice")]

    def test_name_word_match_fallback(self):
        """With no city hit, whole words in the airport name still match."""
        assert _find_city_airports("Heathrow")[0].iata == "LHR"

    def test_deterministic(self):
        first = [a.iata for a in _find_city_airports("Portland")]
        assert first == [a.iata for a in _find_city_airports("Portland")]
        assert first[0] == "PDX"


# ---------------------------------------------------------------------------
# plan_flight MCP tool
# ---------------------------------------------------------------------------


class TestPlanFlightTool:
    def test_iata_only_endpoints(self):
        """Was "Flight planning error: 'city'"."""
        data = _plan({"iata": "SJC"}, {"iata": "DEN"})
        assert data["departure"]["airport"]["iata"] == "SJC"
        assert data["arrival"]["airport"]["iata"] == "DEN"

    def test_mass_kg_is_used(self):
        data = _plan(
            {"iata": "SJC"}, {"iata": "DEN"}, {"ac_type": "A320", "mass_kg": 50000}
        )
        assumptions = data["performance"]["assumptions"]
        assert assumptions["mass_kg"] == 50000
        assert assumptions["mass_source"] == "user"

    def test_aircraft_keys_kept_without_ac_type(self):
        data = _plan(
            {"iata": "SJC"},
            {"iata": "DEN"},
            {"cruise_alt_ft": 25000, "route_step_km": 100},
        )
        assert data["performance"]["assumptions"]["cruise_alt_ft"] == 25000
        # 1526 km at <= 100 km spacing -> 17 points
        assert len(data["route"]["points"]) == 17

    def test_route_options_step_km(self):
        data = _plan({"iata": "SJC"}, {"iata": "DEN"}, None, {"step_km": 500})
        assert len(data["route"]["points"]) == 5

    def test_same_airport_rejected(self):
        out = core_tools.plan_flight({"iata": "SJC"}, {"iata": "SJC"})
        assert out.startswith("Invalid request")
        out = core_tools.plan_flight({"city": "Denver"}, {"city": "denver "})
        assert out.startswith("Invalid request")

    def test_same_city_name_different_countries(self):
        data = _plan(
            {"city": "San Jose", "country": "CR"},
            {"city": "San Jose", "country": "US"},
        )
        assert data["departure"]["airport"]["iata"] == "SJO"
        assert data["arrival"]["airport"]["iata"] == "SJC"

    def test_short_route_profile_fits(self):
        """SFO-SJC (49 km) used to be billed a full FL350 profile:
        50.7 min, 2132 kg, segments summing to 514 km."""
        data = _plan({"iata": "SFO"}, {"iata": "SJC"})
        perf = data["performance"]
        assert _segments_km(perf) == pytest.approx(data["route"]["distance_km"])
        assert perf["assumptions"]["top_of_climb_ft"] < 10000
        assert "profile_note" in perf["assumptions"]
        assert perf["block"]["time_min"] < 15
        assert perf["block"]["fuel_kg"] < 700

    def test_long_route_not_capped(self):
        perf = _plan({"iata": "SJC"}, {"iata": "DEN"})["performance"]
        assert perf["assumptions"]["top_of_climb_ft"] == 35000
        assert "profile_note" not in perf["assumptions"]
        assert _segments_km(perf) == pytest.approx(1525.75, abs=0.1)

    def test_cruise_tas_depends_on_altitude(self):
        """Cruise TAS = M * a(h) at the requested altitude (was always the
        FL358 value, ~447 kt).  A320 WRAP default cruise Mach is 0.78."""
        from aerospace_mcp.core import _openap_aero

        aero = _openap_aero()

        for alt in (25000, 35000):
            perf = _plan(
                {"iata": "SJC"},
                {"iata": "DEN"},
                {"ac_type": "A320", "cruise_alt_ft": alt},
            )["performance"]
            expected = 0.78 * aero.vsound(alt * aero.ft) / aero.kts
            assert perf["cruise"]["avg_gs_kts"] == pytest.approx(expected, rel=1e-6)


class TestOpenAPSynonyms:
    def test_every_openap_type_plans(self):
        """21 of 37 OpenAP types (A20N, B38M, A359, ...) used to fail."""
        from openap import prop

        failures = []
        for ac in prop.available_aircraft():
            try:
                est, _ = estimates_openap(ac.upper(), 35000, None, 1500.0)
                assert est["block"]["fuel_kg"] > 0
            except Exception as e:  # pragma: no cover - reported below
                failures.append((ac, str(e)))
        assert not failures

    @pytest.mark.parametrize(
        "ac", ["A20N", "A21N", "B38M", "B772", "B773", "A359", "E195"]
    )
    def test_named_synonym_types(self, ac):
        est, _ = estimates_openap(ac, 35000, None, 2000.0)
        assert est["assumptions"]["mass_source"] == "mtow_85pct"

    def test_system_status_lists_supported_types(self):
        status = json.loads(core_tools.get_system_status())
        supported = status["openap_info"]["supported_aircraft"]
        for ac in ("A20N", "B38M", "A359", "E195"):
            assert ac in supported


# ---------------------------------------------------------------------------
# Multi-leg planning
# ---------------------------------------------------------------------------


class TestMultiLeg:
    def test_none_aircraft_values_use_defaults(self):
        out = core_tools.plan_multi_leg_flight(
            [{"iata": "SJC"}, {"iata": "DEN"}],
            {"ac_type": "A320", "cruise_alt_ft": None, "route_step_km": None},
        )
        data = json.loads(out)
        assert data["assumptions"]["cruise_alt_ft"] == 35000

    @pytest.mark.parametrize(
        "aircraft, field",
        [
            ({"cruise_alt_ft": 60000}, "cruise_alt_ft"),
            ({"cruise_alt_ft": 1000}, "cruise_alt_ft"),
            ({"route_step_km": 0.5}, "route_step_km"),
            ({"mass_kg": 0}, "mass_kg"),
        ],
    )
    def test_bounds_validated(self, aircraft, field):
        with pytest.raises(FlightPlanError, match=field):
            plan_multi_leg(
                [{"iata": "SJC"}, {"iata": "DEN"}], ac_type="A320", **aircraft
            )
        out = core_tools.plan_multi_leg_flight(
            [{"iata": "SJC"}, {"iata": "DEN"}], aircraft
        )
        assert "invalid aircraft parameters" in out

    def test_multi_leg_totals_match_single_legs(self):
        legs = plan_multi_leg(
            [{"iata": "SJC"}, {"iata": "DEN"}, {"iata": "ORD"}], "A320"
        )
        single = [
            create_flight_plan(
                PlanRequest(
                    depart_city="-",
                    arrive_city="-",
                    prefer_depart_iata=a,
                    prefer_arrive_iata=b,
                    ac_type="A320",
                )
            )
            for a, b in (("SJC", "DEN"), ("DEN", "ORD"))
        ]
        assert legs["totals"]["fuel_kg"] == pytest.approx(
            sum(p.estimates["block"]["fuel_kg"] for p in single)
        )


# ---------------------------------------------------------------------------
# Core dict / typed APIs
# ---------------------------------------------------------------------------


class TestCorePlanners:
    def test_dict_api_rejects_same_airport(self):
        with pytest.raises(ValueError, match="identical"):
            core.plan_flight(
                {
                    "depart_city": "x",
                    "arrive_city": "y",
                    "prefer_depart_iata": "SJC",
                    "prefer_arrive_iata": "SJC",
                    "ac_type": "A320",
                }
            )

    def test_dict_api_rejects_non_positive_mass(self):
        with pytest.raises(ValueError, match="mass_kg"):
            core.plan_flight(
                {
                    "depart_city": "San Jose",
                    "arrive_city": "Denver",
                    "ac_type": "A320",
                    "mass_kg": 0,
                }
            )


# ---------------------------------------------------------------------------
# Caching and lazy import
# ---------------------------------------------------------------------------


class TestCachingAndImport:
    def test_repeat_calls_identical(self):
        first = _plan({"iata": "SJC"}, {"iata": "NRT"}, {"ac_type": "B789"})
        second = _plan({"iata": "SJC"}, {"iata": "NRT"}, {"ac_type": "B789"})
        assert first == second

    def test_cached_profiles_are_read_only(self):
        estimates_openap("A320", 35000, None, 1500.0)
        fgen = core._cached_model(core._openap("FlightGenerator"), "A320")
        profile = core._cached_profile(fgen, "climb", 35000, core._SIM_DT_S)
        with pytest.raises(ValueError):
            profile.s[0] = 1.0

    def test_aircraft_props_are_copies(self):
        props = core._aircraft_props("A320")
        props["limits"]["MTOW"] = 1.0
        assert core._aircraft_props("A320")["limits"]["MTOW"] != 1.0

    def test_warm_plan_is_fast(self):
        """OpenAP models and profiles are cached (was ~38 ms per plan)."""
        args = ({"iata": "SJC"}, {"iata": "DEN"})
        core_tools.plan_flight(*args)  # warm the caches
        start = time.perf_counter()
        for _ in range(20):
            core_tools.plan_flight(*args)
        per_call_ms = (time.perf_counter() - start) / 20 * 1000
        assert per_call_ms < 25  # ~1-2 ms locally; generous for slow CI

    def test_openap_imported_lazily(self):
        code = (
            "import sys, aerospace_mcp.core as c; "
            "print(c.OPENAP_AVAILABLE, 'openap' in sys.modules); "
            "c.estimates_openap('A320', 35000, None, 1000.0); "
            "print('openap' in sys.modules)"
        )
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        ).stdout.split()
        assert out == ["True", "False", "True"]


# ---------------------------------------------------------------------------
# HTTP layer
# ---------------------------------------------------------------------------


@pytest.fixture
def http_client(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_RPM", "0")
    return TestClient(main_module.app)


class TestHTTP:
    def test_unsupported_aircraft_is_400(self, http_client):
        """Was 501 (reserved for "OpenAP not installed")."""
        r = http_client.post(
            "/plan",
            json={
                "depart_city": "x",
                "arrive_city": "y",
                "prefer_depart_iata": "SJC",
                "prefer_arrive_iata": "DEN",
                "ac_type": "ZZZZ",
            },
        )
        assert r.status_code == 400
        assert "not supported" in r.json()["detail"]

    def test_same_airport_is_400(self, http_client):
        r = http_client.post(
            "/plan",
            json={
                "depart_city": "x",
                "arrive_city": "y",
                "prefer_depart_iata": "SJC",
                "prefer_arrive_iata": "SJC",
                "ac_type": "A320",
            },
        )
        assert r.status_code == 400

    def test_same_city_name_different_countries_is_200(self, http_client):
        r = http_client.post(
            "/plan",
            json={
                "depart_city": "San Jose",
                "depart_country": "CR",
                "arrive_city": "San Jose",
                "arrive_country": "US",
                "ac_type": "A320",
            },
        )
        assert r.status_code == 200
        assert (r.json()["depart"]["iata"], r.json()["arrive"]["iata"]) == (
            "SJO",
            "SJC",
        )

    def test_mass_zero_is_422(self, http_client):
        r = http_client.post(
            "/plan",
            json={
                "depart_city": "San Jose",
                "arrive_city": "Denver",
                "ac_type": "A320",
                "mass_kg": 0,
            },
        )
        assert r.status_code == 422

    def test_chunked_body_over_limit_is_413(self, http_client, monkeypatch):
        """A body without Content-Length is metered while streaming."""
        monkeypatch.setenv("MAX_BODY_BYTES", "100")
        body = json.dumps(
            {"depart_city": "x" * 90, "arrive_city": "y", "ac_type": "A320"}
        )

        def chunks():
            for i in range(0, len(body), 32):
                yield body[i : i + 32].encode()

        r = http_client.post(
            "/plan", content=chunks(), headers={"content-type": "application/json"}
        )
        assert r.status_code == 413
        assert "too large" in r.json()["detail"]

    def test_chunked_body_under_limit_passes(self, http_client, monkeypatch):
        monkeypatch.setenv("MAX_BODY_BYTES", "4096")
        body = json.dumps({"depart_city": "San Jose"}).encode()

        def chunks():
            yield body[:10]
            yield body[10:]

        r = http_client.post(
            "/plan", content=chunks(), headers={"content-type": "application/json"}
        )
        assert r.status_code == 422  # reaches validation (missing fields)
