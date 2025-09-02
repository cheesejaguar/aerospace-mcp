from __future__ import annotations

import json

import aerospace_mcp.tools.core as core_tools


def test_search_airports_and_distance(monkeypatch):
    class A:
        def __init__(self, iata: str):
            self.iata = iata
            self.icao = "K" + iata
            self.name = "Name"
            self.city = "City"
            self.country = "US"
            self.lat = 0.0
            self.lon = 0.0
            self.tz = None

    monkeypatch.setattr(
        core_tools, "_airport_from_iata", lambda q: A(q) if q.upper() == "SJC" else None
    )
    monkeypatch.setattr(
        core_tools,
        "_find_city_airports",
        lambda q, c: [A("SJC")] if q.lower() == "san jose" else [],
    )

    out = core_tools.search_airports("SJC", "US", "auto")
    assert "Found" in out
    out2 = core_tools.search_airports("Unknown", "US", "city")
    assert "No airports" in out2
    out3 = core_tools.search_airports("  ")
    assert "required" in out3

    # calculate_distance expects great_circle_points providing mapping
    monkeypatch.setattr(
        core_tools,
        "great_circle_points",
        lambda a, b, c, d, step_km: {
            "distance_km": 1000.0,
            "distance_nm": 539.9,
            "initial_bearing_deg": 10.0,
            "final_bearing_deg": 20.0,
        },
    )
    dist = core_tools.calculate_distance(0, 0, 1, 1)
    data = json.loads(dist)
    assert data["distance_km"] == 1000.0


def test_plan_and_status(monkeypatch):
    # Stubs for endpoints and routing
    class A:
        def __init__(self, iata: str):
            self.iata = iata
            self.icao = "K" + iata
            self.name = "Name"
            self.city = "City"
            self.country = "US"
            self.lat = 0.0
            self.lon = 0.0
            self.tz = None

    monkeypatch.setattr(
        core_tools,
        "_resolve_endpoint",
        lambda city, country, iata=None: A(iata or "SJC"),
    )
    monkeypatch.setattr(
        core_tools,
        "great_circle_points",
        lambda a, b, c, d, step_km: {
            "distance_km": 1000.0,
            "distance_nm": 539.9,
            "initial_bearing_deg": 10.0,
            "final_bearing_deg": 20.0,
            "points": [(0.0, 0.0, 0.0)],
        },
    )
    # estimates_openap may not be available; return simple tuple
    monkeypatch.setattr(core_tools, "OPENAP_AVAILABLE", True)
    monkeypatch.setattr(
        core_tools,
        "estimates_openap",
        lambda *a, **k: (
            {"assumptions": {"cruise_alt_ft": a[1], "mass_kg": 70000.0}},
            "engine",
        ),
    )

    out = core_tools.plan_flight(
        {"city": "City", "iata": "SJC"},
        {"city": "City", "iata": "NRT"},
        {"ac_type": "A320", "cruise_alt_ft": 41000, "route_step_km": 25.0},
        {},
    )
    data = json.loads(out)
    assert data["route"]["distance_km"] == 1000.0

    status = core_tools.get_system_status()
    assert "system" in status


def test_plan_optional_fields_and_resolution_error(monkeypatch):
    import aerospace_mcp.tools.core as core_tools
    from aerospace_mcp.core import AirportResolutionError

    # Resolution error path
    def raise_res(*a, **k):
        raise AirportResolutionError("nope")

    monkeypatch.setattr(core_tools, "_resolve_endpoint", raise_res)
    out_err = core_tools.plan_flight(
        {"city": "City", "country": "US"},
        {"city": "City", "country": "JP"},
        {"ac_type": "A320"},
    )
    assert "Airport resolution error" in out_err

    # Optional fields executed (with successful resolve)
    class A:
        def __init__(self, iata: str, country: str):
            self.iata = iata
            self.icao = "K" + iata
            self.name = "Name"
            self.city = "City"
            self.country = country
            self.lat = 0.0
            self.lon = 0.0
            self.tz = None

    monkeypatch.setattr(
        core_tools,
        "_resolve_endpoint",
        lambda city, country, iata=None: A(iata or "SJC", country or "US"),
    )
    monkeypatch.setattr(
        core_tools,
        "great_circle_points",
        lambda *a, **k: {
            "distance_km": 1.0,
            "distance_nm": 1.0,
            "initial_bearing_deg": 0.0,
            "final_bearing_deg": 0.0,
            "points": [(0, 0, 0)],
        },
    )
    out_ok = core_tools.plan_flight(
        {"city": "City", "country": "US"},
        {"city": "City", "country": "JP"},
        {"ac_type": "A320"},
    )
    assert "departure" in out_ok


def test_search_airports_exception(monkeypatch):
    import aerospace_mcp.tools.core as core_tools

    def boom(q):
        raise RuntimeError("boom")

    monkeypatch.setattr(core_tools, "_airport_from_iata", boom)
    out = core_tools.search_airports("SJC", None, "iata")
    assert "Search error" in out


def test_plan_invalid_request(monkeypatch):
    import aerospace_mcp.tools.core as core_tools

    # Valid endpoints
    class A:
        def __init__(self, iata: str):
            self.iata = iata
            self.icao = "K" + iata
            self.name = "Name"
            self.city = "City"
            self.country = "US"
            self.lat = 0.0
            self.lon = 0.0
            self.tz = None

    monkeypatch.setattr(core_tools, "_resolve_endpoint", lambda *a, **k: A("SJC"))
    # Build invalid PlanRequest by negative route_step_km
    out = core_tools.plan_flight(
        {"city": "City"},
        {"city": "City"},
        {"ac_type": "A320", "route_step_km": -1.0},
    )
    assert "Invalid request" in out


def test_plan_openap_error_and_unavailable(monkeypatch):
    import aerospace_mcp.tools.core as core_tools

    class A:
        def __init__(self, iata: str):
            self.iata = iata
            self.icao = "K" + iata
            self.name = "Name"
            self.city = "City"
            self.country = "US"
            self.lat = 0.0
            self.lon = 0.0
            self.tz = None

    monkeypatch.setattr(core_tools, "_resolve_endpoint", lambda *a, **k: A("SJC"))
    monkeypatch.setattr(
        core_tools,
        "great_circle_points",
        lambda *a, **k: {
            "distance_km": 1.0,
            "distance_nm": 1.0,
            "initial_bearing_deg": 0.0,
            "final_bearing_deg": 0.0,
            "points": [(0, 0, 0)],
        },
    )
    # OpenAP available but raises
    monkeypatch.setattr(core_tools, "OPENAP_AVAILABLE", True)
    from aerospace_mcp.core import OpenAPError

    def raise_openap(*a, **k):
        raise OpenAPError("bad")

    monkeypatch.setattr(core_tools, "estimates_openap", raise_openap)
    out = core_tools.plan_flight(
        {"city": "City"}, {"city": "City"}, {"ac_type": "A320"}
    )
    assert "Performance estimation failed" in out
    # OpenAP unavailable path
    monkeypatch.setattr(core_tools, "OPENAP_AVAILABLE", False)
    out2 = core_tools.plan_flight(
        {"city": "City"}, {"city": "City"}, {"ac_type": "A320"}
    )
    assert "OpenAP not available" in out2


def test_plan_and_distance_exception(monkeypatch):
    import aerospace_mcp.tools.core as core_tools

    def boom(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(core_tools, "great_circle_points", boom)
    out = core_tools.plan_flight(
        {"city": "City"}, {"city": "City"}, {"ac_type": "A320"}
    )
    assert "Flight planning error" in out
    out2 = core_tools.calculate_distance(0, 0, 0, 0)
    assert "Distance calculation error" in out2


def test_get_aircraft_performance_branches(monkeypatch):
    import aerospace_mcp.tools.core as core_tools

    # Unavailable
    monkeypatch.setattr(core_tools, "OPENAP_AVAILABLE", False)
    assert "OpenAP library is not available" in core_tools.get_aircraft_performance(
        "A320", 1000
    )
    # Available success
    monkeypatch.setattr(core_tools, "OPENAP_AVAILABLE", True)
    monkeypatch.setattr(core_tools, "estimates_openap", lambda *a, **k: {"ok": True})
    assert "ok" in core_tools.get_aircraft_performance("A320", 1000)
    # OpenAPError
    from aerospace_mcp.core import OpenAPError

    def raise_openap(*a, **k):
        raise OpenAPError("bad")

    monkeypatch.setattr(core_tools, "estimates_openap", raise_openap)
    assert "Performance estimation error" in core_tools.get_aircraft_performance(
        "A320", 1000
    )
    # Unexpected error
    monkeypatch.setattr(
        core_tools,
        "estimates_openap",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    assert "Unexpected error" in core_tools.get_aircraft_performance("A320", 1000)


def test_system_status_openap_false(monkeypatch):
    import aerospace_mcp.tools.core as core_tools

    monkeypatch.setattr(core_tools, "OPENAP_AVAILABLE", False)
    out = core_tools.get_system_status()
    assert "OpenAP not available" in out
