from __future__ import annotations

import json
import math

import pytest

from tests.tools.test_tools_common import without_integrations


def test_orbits_elements_to_state_circular_equatorial():
    """Circular equatorial orbit at periapsis: r = a along x, v = sqrt(mu/a)
    along y."""
    from aerospace_mcp.tools.orbits import elements_to_state_vector

    out = json.loads(elements_to_state_vector(CIRCULAR))
    assert out["position_m"] == {"x": 7.0e6, "y": 0.0, "z": 0.0}
    assert out["velocity_ms"]["x"] == pytest.approx(0.0, abs=1e-9)
    assert out["velocity_ms"]["y"] == pytest.approx(math.sqrt(MU_EARTH / 7.0e6))
    assert out["reference_frame"] == "J2000"


def test_orbits_state_to_elements_round_trip():
    from aerospace_mcp.tools.orbits import (
        elements_to_state_vector,
        state_vector_to_elements,
    )

    state = json.loads(elements_to_state_vector(ELEMENTS))
    out = json.loads(
        state_vector_to_elements(
            {
                "position_m": list(state["position_m"].values()),
                "velocity_ms": list(state["velocity_ms"].values()),
                "epoch_utc": ELEMENTS["epoch_utc"],
            }
        )
    )
    assert out["semi_major_axis_m"] == pytest.approx(7.0e6, rel=1e-9)
    assert out["eccentricity"] == pytest.approx(0.001, rel=1e-6)
    assert out["inclination_deg"] == pytest.approx(51.6)
    assert out["raan_deg"] == pytest.approx(30.0)
    assert out["epoch_utc"] == ELEMENTS["epoch_utc"]


def test_orbits_hohmann_leo_to_geo():
    """LEO (300 km) to GEO: the textbook ~2.43 + ~1.47 = ~3.9 km/s, 5.3 h."""
    from aerospace_mcp.tools.orbits import hohmann_transfer

    out = json.loads(hohmann_transfer(6678e3, 42164e3))
    assert out["delta_v1_ms"] == pytest.approx(2426, abs=5)
    assert out["delta_v2_ms"] == pytest.approx(1467, abs=5)
    assert out["total_delta_v_ms"] == pytest.approx(
        out["delta_v1_ms"] + out["delta_v2_ms"]
    )
    assert out["transfer_time_h"] == pytest.approx(5.28, abs=0.01)
    assert out["transfer_semi_major_axis_m"] == (6678e3 + 42164e3) / 2


def test_orbits_rendezvous_same_period_is_valid_json():
    """Equal periods never close the phase angle: phasing time is null.

    It used to be float('inf'), which the tool serialised as the invalid
    JSON token ``Infinity``.
    """
    from aerospace_mcp.tools.orbits import orbital_rendezvous_planning

    target = {**CIRCULAR, "true_anomaly_deg": 10.0}
    raw = orbital_rendezvous_planning(CIRCULAR, target)
    assert "Infinity" not in raw
    out = json.loads(raw)
    assert out["phase_angle_deg"] == pytest.approx(10.0)
    # Chord between two points 10 deg apart on a 7,000 km circle
    assert out["relative_distance_m"] == pytest.approx(
        2 * 7.0e6 * math.sin(math.radians(5.0))
    )
    assert out["phasing_time_s"] is None
    assert out["phasing_time_h"] is None


def test_orbits_rendezvous_different_altitudes():
    from aerospace_mcp.tools.orbits import orbital_rendezvous_planning

    target = {**CIRCULAR, "semi_major_axis_m": 7.1e6, "true_anomaly_deg": 30.0}
    out = json.loads(orbital_rendezvous_planning(CIRCULAR, target))
    assert out["phasing_time_s"] > 0
    assert out["altitude_difference_m"] == pytest.approx(1.0e5)
    assert [m["type"] for m in out["maneuvers"]] == ["altitude_adjustment"]


CIRCULAR = {
    "semi_major_axis_m": 7.0e6,
    "eccentricity": 0.0,
    "inclination_deg": 0.0,
    "raan_deg": 0.0,
    "arg_periapsis_deg": 0.0,
    "true_anomaly_deg": 0.0,
    "epoch_utc": "2026-01-01T00:00:00+00:00",
}
MU_EARTH = 3.986004418e14

ELEMENTS = {
    "semi_major_axis_m": 7.0e6,
    "eccentricity": 0.001,
    "inclination_deg": 51.6,
    "raan_deg": 30.0,
    "arg_periapsis_deg": 0.0,
    "true_anomaly_deg": 0.0,
    "epoch_utc": "2026-01-01T00:00:00+00:00",
}


def test_propagate_accepts_state_vectors_and_elements():
    """Real native propagation through the tool layer (no stubs)."""
    import json

    from aerospace_mcp.tools.orbits import elements_to_state_vector, propagate_orbit_j2

    from_elements = json.loads(propagate_orbit_j2(ELEMENTS, 600.0, 60.0))
    assert from_elements["num_states"] == 11
    assert from_elements["states"][-1]["epoch_utc"] == "2026-01-01T00:10:00+00:00"

    # The x/y/z dict form returned by elements_to_state_vector chains directly.
    state = json.loads(elements_to_state_vector(ELEMENTS))
    state["epoch_utc"] = ELEMENTS["epoch_utc"]
    from_state = json.loads(propagate_orbit_j2(state, 600.0, 60.0))
    assert (
        from_state["states"][-1]["position_m"]
        == from_elements["states"][-1]["position_m"]
    )


def test_ground_track_end_to_end():
    import json

    from aerospace_mcp.tools.orbits import calculate_ground_track

    out = json.loads(calculate_ground_track(ELEMENTS, 5400.0, 60.0))
    assert out["num_points"] == 91
    lats = [p["latitude_deg"] for p in out["ground_track"]]
    # Ground track latitude is bounded by the inclination.
    assert max(lats) <= 51.7 and min(lats) >= -51.7
    assert max(lats) > 45.0


WEEK_AT_10S = (7 * 86400.0, 10.0)  # 60 481 states


def test_propagate_returns_a_bounded_summary():
    """A week at 10 s steps used to return all 60k states (17 MB of JSON)."""
    import json

    from aerospace_mcp.tools.orbits import propagate_orbit_j2

    text = propagate_orbit_j2(ELEMENTS, *WEEK_AT_10S)
    out = json.loads(text)
    assert len(text) < 100_000
    assert out["num_states"] == 60481
    assert out["max_points"] == 200 and out["num_samples"] == 200
    states = out["states"]
    assert len(states) == 200
    # First and last always included, evenly spaced in between.
    assert states[0]["elapsed_s"] == 0.0
    assert states[-1]["elapsed_s"] == 7 * 86400.0
    assert states[0] == {"elapsed_s": 0.0, **out["initial_state"]}
    assert states[-1] == {"elapsed_s": 7 * 86400.0, **out["final_state"]}
    assert out["final_state"]["epoch_utc"] == "2026-01-08T00:00:00+00:00"
    gaps = {
        round(b["elapsed_s"] - a["elapsed_s"])
        for a, b in zip(states, states[1:], strict=False)
    }
    assert gaps <= {3030, 3040}


def test_propagate_samples_match_the_full_propagation():
    import json

    from aerospace_mcp.integrations.orbits import propagate_orbit_j2 as full
    from aerospace_mcp.tools.orbits import _initial_state, propagate_orbit_j2

    every = full(_initial_state(ELEMENTS), 6000.0, 60.0)
    out = json.loads(propagate_orbit_j2(ELEMENTS, 6000.0, 60.0, max_points=11))
    assert out["num_states"] == len(every) == 101
    for k, state in enumerate(out["states"]):
        assert state["elapsed_s"] == 600.0 * k
        assert state["position_m"] == every[10 * k].position_m
        assert state["epoch_utc"] == every[10 * k].epoch_utc
    # Small runs return every state.
    small = json.loads(propagate_orbit_j2(ELEMENTS, 600.0, 60.0, max_points=5000))
    assert small["num_samples"] == small["num_states"] == 11


def test_ground_track_returns_a_bounded_summary_with_extents():
    import json

    from aerospace_mcp.tools.orbits import calculate_ground_track

    text = calculate_ground_track(ELEMENTS, *WEEK_AT_10S, max_points=50)
    out = json.loads(text)
    assert len(text) < 20_000
    assert out["num_points"] == 60481 and out["num_samples"] == 50
    extents = out["extents"]
    # Latitude spans +-inclination over a week; longitude wraps all the way.
    assert 51.0 < extents["latitude_deg"]["max"] <= 51.7
    assert -51.7 <= extents["latitude_deg"]["min"] < -51.0
    assert (
        extents["longitude_deg"]["min"] < -179 and extents["longitude_deg"]["max"] > 179
    )
    assert out["ground_track"][0]["elapsed_s"] == 0.0
    assert out["ground_track"][-1]["elapsed_s"] == 7 * 86400.0
    assert out["ground_track"][-1]["time_utc"] == "2026-01-08T00:00:00+00:00"


def test_ground_track_samples_match_the_full_track():
    import json

    from aerospace_mcp.integrations.orbits import calculate_ground_track as full_track
    from aerospace_mcp.integrations.orbits import propagate_orbit_j2 as full
    from aerospace_mcp.tools.orbits import _initial_state, calculate_ground_track

    track = full_track(full(_initial_state(ELEMENTS), 5400.0, 60.0), 60.0)
    out = json.loads(calculate_ground_track(ELEMENTS, 5400.0, 60.0, max_points=10))
    assert out["num_points"] == len(track) == 91
    for k, point in enumerate(out["ground_track"]):
        expected = track[10 * k]
        assert point["latitude_deg"] == expected.latitude_deg
        assert abs(point["longitude_deg"] - expected.longitude_deg) < 1e-9
        assert point["time_utc"] == expected.time_utc
    lats = [p.latitude_deg for p in track]
    assert out["extents"]["latitude_deg"] == {"min": min(lats), "max": max(lats)}


def test_trajectory_tools_validate_max_points():
    from aerospace_mcp.tools.orbits import calculate_ground_track, propagate_orbit_j2

    for bad in (1, 5001, 2.5, True):
        assert "max_points" in propagate_orbit_j2(ELEMENTS, 600.0, 60.0, bad)
        assert "max_points" in calculate_ground_track(ELEMENTS, 600.0, 60.0, bad)


def test_propagate_rejects_out_of_range_step():
    from aerospace_mcp.tools.orbits import propagate_orbit_j2

    assert "error" in propagate_orbit_j2(ELEMENTS, 600.0, 0.01).lower()


def test_orbits_import_errors(monkeypatch):
    from aerospace_mcp.tools import orbits as orbits_tools

    without_integrations(monkeypatch, "orbits")
    assert "not available" in orbits_tools.elements_to_state_vector({}).lower()
    assert "not available" in orbits_tools.state_vector_to_elements({}).lower()
    assert "not available" in orbits_tools.propagate_orbit_j2({}, 10.0).lower()
    assert "not available" in orbits_tools.calculate_ground_track({}, 10.0).lower()
    assert "not available" in orbits_tools.hohmann_transfer(1.0, 2.0).lower()
    assert "not available" in orbits_tools.orbital_rendezvous_planning({}, {}).lower()


def test_orbits_invalid_input_errors():
    """Bad inputs come back as error strings naming the problem."""
    from aerospace_mcp.tools import orbits as otools

    out = otools.elements_to_state_vector({})
    assert out.startswith("Elements to state vector error") and "semi_major" in out
    out = otools.state_vector_to_elements({})
    assert out.startswith("State vector to elements error") and "position_m" in out
    assert otools.propagate_orbit_j2({}, 10.0).startswith("Orbit propagation error")
    assert otools.calculate_ground_track({}, 10.0).startswith("Ground track error")
    assert otools.hohmann_transfer(-1.0, 2.0) == (
        "Hohmann transfer error: orbit radii must be > 0"
    )
    assert otools.orbital_rendezvous_planning({}, {}).startswith(
        "Rendezvous planning error"
    )
