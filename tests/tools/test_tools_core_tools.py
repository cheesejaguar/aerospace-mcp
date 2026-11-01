"""Core flight-planning tools against the real airport database and OpenAP."""

from __future__ import annotations

import json

import pytest

import aerospace_mcp.tools.core as core_tools


def test_search_airports_and_distance():
    out = core_tools.search_airports("SJC", "US", "auto")
    assert out.startswith("Found 1 airport(s):")
    assert "SJC (KSJC)" in out
    out2 = core_tools.search_airports("Unknown", "US", "city")
    assert out2 == "No airports found for city 'Unknown' in country 'US'"
    out3 = core_tools.search_airports("  ")
    assert out3 == "Error: Query parameter is required"

    # Quarter of the equator: a * pi / 2 on the WGS-84 ellipsoid
    data = json.loads(core_tools.calculate_distance(0, 0, 0, 90))
    assert data["distance_km"] == pytest.approx(10018.754, abs=0.01)
    assert data["distance_nm"] == pytest.approx(10018.754 / 1.852, abs=0.01)
    assert data["initial_bearing_deg"] == pytest.approx(90.0)


def test_plan_and_status():
    out = core_tools.plan_flight(
        {"city": "San Jose", "iata": "SJC"},
        {"city": "Tokyo", "iata": "NRT"},
        {"ac_type": "A320", "cruise_alt_ft": 41000, "route_step_km": 25.0},
        {},
    )
    data = json.loads(out)
    assert data["departure"]["airport"]["iata"] == "SJC"
    assert data["arrival"]["airport"]["iata"] == "NRT"
    assert 8200 < data["route"]["distance_km"] < 8400
    assert data["performance"]["assumptions"]["cruise_alt_ft"] == 41000
    assert data["engine"] == "openap"
    # Polyline sampled at the requested step
    points = data["route"]["points"]
    assert len(points) == int(-(-data["route"]["distance_km"] // 25.0)) + 1
    # Geodesic sampling round-trips through floating point, so compare closely
    # rather than bit-for-bit (last-ulp differences vary by platform).
    assert points[0] == pytest.approx(
        data["departure"]["airport"]["coordinates"], abs=1e-9
    )

    status = json.loads(core_tools.get_system_status())
    assert status["status"] == "operational"
    assert status["capabilities"]["airport_search"] is True


def test_plan_optional_fields_and_resolution_error():
    out_err = core_tools.plan_flight(
        {"city": "Atlantis", "country": "US"},
        {"city": "Tokyo", "country": "JP"},
        {"ac_type": "A320"},
    )
    assert out_err.startswith("Airport resolution error")
    assert "Atlantis" in out_err

    # Country codes and route_options.step_km are honoured
    out_ok = json.loads(
        core_tools.plan_flight(
            {"city": "San Jose", "country": "CR"},
            {"city": "Tokyo", "country": "JP"},
            {"ac_type": "A320"},
            {"step_km": 500.0},
        )
    )
    assert out_ok["departure"]["airport"]["iata"] == "SJO"
    assert out_ok["arrival"]["airport"]["iata"] == "HND"
    distance = out_ok["route"]["distance_km"]
    assert len(out_ok["route"]["points"]) == int(-(-distance // 500.0)) + 1


def test_search_airports_auto_detects_city():
    """Queries that are not three letters are searched as cities."""
    out = core_tools.search_airports("Tokyo", "JP")
    assert out.startswith("Found 2 airport(s):")
    assert out.index("HND") < out.index("NRT")


def test_plan_invalid_request():
    out = core_tools.plan_flight(
        {"iata": "SJC"},
        {"iata": "NRT"},
        {"ac_type": "A320", "route_step_km": -1.0},
    )
    assert out.startswith("Invalid request")
    assert "route_step_km" in out


def test_plan_same_airport_rejected():
    out = core_tools.plan_flight({"iata": "SJC"}, {"city": "San Jose", "country": "US"})
    assert out.startswith("Invalid request: departure and arrival both resolve to SJC")


def test_plan_openap_error_and_unavailable(monkeypatch):
    # An aircraft type OpenAP cannot model: route still returned, with a note
    out = json.loads(
        core_tools.plan_flight({"iata": "SJC"}, {"iata": "LAX"}, {"ac_type": "C172"})
    )
    assert "performance" not in out
    assert out["performance_note"].startswith("Performance estimation failed")
    assert out["route"]["distance_km"] > 400

    # OpenAP not installed
    monkeypatch.setattr(core_tools, "OPENAP_AVAILABLE", False)
    out2 = json.loads(
        core_tools.plan_flight({"iata": "SJC"}, {"iata": "LAX"}, {"ac_type": "A320"})
    )
    assert out2["performance_note"] == (
        "OpenAP not available - no performance estimates for A320"
    )


def test_distance_invalid_input():
    out = core_tools.calculate_distance("a", 0, 0, 0)
    assert out.startswith("Distance calculation error")


def test_get_aircraft_performance_branches(monkeypatch):
    # Success
    data = json.loads(core_tools.get_aircraft_performance("A320", 1000))
    assert data["engine"] == "openap"
    assert data["performance"]["assumptions"]["cruise_alt_ft"] == 35000
    # Unsupported type -> OpenAPError branch
    assert core_tools.get_aircraft_performance("C172", 300).startswith(
        "Performance estimation error"
    )
    # Unavailable
    monkeypatch.setattr(core_tools, "OPENAP_AVAILABLE", False)
    assert "OpenAP library is not available" in core_tools.get_aircraft_performance(
        "A320", 1000
    )


def test_system_status_openap_false(monkeypatch):
    monkeypatch.setattr(core_tools, "OPENAP_AVAILABLE", False)
    out = core_tools.get_system_status()
    assert "OpenAP not available" in out
