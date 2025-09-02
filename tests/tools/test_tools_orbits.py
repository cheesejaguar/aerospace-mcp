from __future__ import annotations

import sys

from tests.tools.test_tools_common import make_module


class OrbitElements:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class StateVector:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def test_orbits_elements_to_state_success():
    class Result:
        position_m = [1.0, 2.0, 3.0]
        velocity_ms = [4.0, 5.0, 6.0]

    sys.modules["aerospace_mcp.integrations.orbits"] = make_module(
        OrbitElements=OrbitElements, elements_to_state_vector=lambda e: Result()
    )
    from aerospace_mcp.tools.orbits import elements_to_state_vector

    out = elements_to_state_vector({"semi_major_axis_m": 7000e3})
    assert "position_m" in out


def test_orbits_state_to_elements_success():
    class Result:
        semi_major_axis_m = 1.0
        eccentricity = 0.1
        inclination_deg = 45.0
        raan_deg = 0.0
        arg_perigee_deg = 0.0
        true_anomaly_deg = 0.0
        epoch_utc = "2020-01-01T00:00:00Z"

    sys.modules["aerospace_mcp.integrations.orbits"] = make_module(
        StateVector=StateVector, state_vector_to_elements=lambda s: Result()
    )
    from aerospace_mcp.tools.orbits import state_vector_to_elements

    out = state_vector_to_elements(
        {"position_m": [1, 0, 0], "velocity_ms": [0, 1, 0], "epoch_utc": "..."}
    )
    assert "eccentricity" in out


def test_orbits_propagate_and_groundtrack_success():
    sys.modules["aerospace_mcp.integrations.orbits"] = make_module(
        propagate_orbit_j2=lambda *a, **k: {"ok": True},
        calculate_ground_track=lambda *a, **k: {"points": []},
        hohmann_transfer=lambda *a, **k: {"dv": [1.0, 2.0]},
        orbital_rendezvous_planning=lambda *a, **k: {"plan": []},
        OrbitElements=OrbitElements,
    )
    from aerospace_mcp.tools.orbits import (
        calculate_ground_track,
        hohmann_transfer,
        orbital_rendezvous_planning,
        propagate_orbit_j2,
    )

    assert "ok" in propagate_orbit_j2({"state": 1}, 10.0)
    assert "points" in calculate_ground_track({"state": 1}, 10.0)
    assert "dv" in hohmann_transfer(7000e3, 8000e3)
    assert "plan" in orbital_rendezvous_planning({}, {})


def test_orbits_import_errors():
    import sys

    from aerospace_mcp.tools import orbits as orbits_tools

    sys.modules["aerospace_mcp.integrations.orbits"] = make_module()
    assert "not available" in orbits_tools.elements_to_state_vector({}).lower()
    assert "not available" in orbits_tools.state_vector_to_elements({}).lower()
    assert "not available" in orbits_tools.propagate_orbit_j2({}, 10.0).lower()
    assert "not available" in orbits_tools.calculate_ground_track({}, 10.0).lower()
    assert "not available" in orbits_tools.hohmann_transfer(1.0, 2.0).lower()
    assert "not available" in orbits_tools.orbital_rendezvous_planning({}, {}).lower()


def test_orbits_exception_branches():
    import sys

    def _boom(*a, **k):
        raise RuntimeError("boom")

    sys.modules["aerospace_mcp.integrations.orbits"] = make_module(
        OrbitElements=OrbitElements,
        StateVector=StateVector,
        elements_to_state_vector=_boom,
        state_vector_to_elements=_boom,
        propagate_orbit_j2=_boom,
        calculate_ground_track=_boom,
        hohmann_transfer=_boom,
        orbital_rendezvous_planning=_boom,
    )
    from aerospace_mcp.tools import orbits as otools

    assert "error" in otools.elements_to_state_vector({}).lower()
    assert "error" in otools.state_vector_to_elements({}).lower()
    assert "error" in otools.propagate_orbit_j2({}, 10.0).lower()
    assert "error" in otools.calculate_ground_track({}, 10.0).lower()
    assert "error" in otools.hohmann_transfer(1.0, 2.0).lower()
    assert "error" in otools.orbital_rendezvous_planning({}, {}).lower()
