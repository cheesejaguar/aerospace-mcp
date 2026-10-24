"""Regression tests for defects found auditing the native (Rust) port.

Each test reproduces a case that previously crashed the server, leaked
NaN/Infinity into tool JSON, or returned an inconsistent answer.
"""

import json
import math

import pytest

from aerospace_mcp.integrations import orbits as O
from aerospace_mcp.integrations import rockets as R
from aerospace_mcp.tools import optimization as opt_tools
from aerospace_mcp.tools import orbits as orbit_tools
from aerospace_mcp.tools import rockets as rocket_tools

GEOMETRY = {
    "dry_mass_kg": 20.0,
    "propellant_mass_kg": 80.0,
    "diameter_m": 0.2,
    "length_m": 2.0,
    "cd": 0.4,
}


def _strict_json(text: str):
    """Parse JSON, rejecting NaN/Infinity (which strict JSON clients reject)."""

    def _reject(token):
        raise ValueError(f"non-finite value in tool output: {token}")

    return json.loads(text, parse_constant=_reject)


@pytest.mark.parametrize("n_segments", [2**62, 2**40, 51, 0, -1])
def test_thrust_profile_segment_count_is_bounded(n_segments):
    """Huge counts used to panic (PanicException escapes `except Exception`
    and drops the MCP connection) or abort the process on allocation."""
    out = opt_tools.optimize_thrust_profile(GEOMETRY, 5.0, 10000.0, n_segments)
    assert "n_segments" in out and "error" in out.lower()


def test_elements_outside_conic_domain_report_errors_not_nan():
    for a, e in [(7e6, 1.5), (7e6, 1.0)]:
        out = orbit_tools.elements_to_state_vector(
            {
                "semi_major_axis_m": a,
                "eccentricity": e,
                "inclination_deg": 10.0,
                "raan_deg": 0.0,
                "arg_periapsis_deg": 0.0,
                "true_anomaly_deg": 0.0,
                "epoch_utc": "2026-01-01T00:00:00Z",
            }
        )
        assert "error" in out.lower(), out
        assert "NaN" not in out and "Infinity" not in out


def test_orbit_properties_reject_non_elliptic():
    elements = O.OrbitElements(
        semi_major_axis_m=-7e6,
        eccentricity=1.5,
        inclination_deg=0.0,
        raan_deg=0.0,
        arg_periapsis_deg=0.0,
        true_anomaly_deg=0.0,
        epoch_utc="2026-01-01T00:00:00Z",
    )
    with pytest.raises(ValueError, match="elliptical"):
        O.calculate_orbit_properties(elements)


def test_state_vector_tool_accepts_elements_tool_output():
    """elements_to_state_vector emits {x, y, z} dicts; they must chain."""
    elements = {
        "semi_major_axis_m": 7e6,
        "eccentricity": 0.01,
        "inclination_deg": 51.6,
        "raan_deg": 30.0,
        "arg_periapsis_deg": 45.0,
        "true_anomaly_deg": 10.0,
        "epoch_utc": "2026-01-01T00:00:00Z",
    }
    state = _strict_json(orbit_tools.elements_to_state_vector(elements))
    back = _strict_json(orbit_tools.state_vector_to_elements(state))
    assert back["semi_major_axis_m"] == pytest.approx(7e6, rel=1e-9)
    assert back["inclination_deg"] == pytest.approx(51.6, rel=1e-9)


def test_lambert_plane_through_z_axis_reports_consistent_short_way():
    """(r1 x r2)_z is exactly 0 here; rounding used to flip the solver onto
    the 351-degree long way while the reported angle said 8.6 degrees."""
    result = O.solve_lambert(
        [17e6, 13e6, 739122.8681162545], [17e6, 13e6, -2495907.938505691], 3000.0
    )
    assert result["transfer_angle_deg"] == pytest.approx(8.63, abs=0.01)
    assert O.vector_magnitude(result["v1_ms"]) == pytest.approx(1649.96, abs=0.1)


def test_lambert_rejects_negative_revolutions():
    with pytest.raises(ValueError, match="revolutions"):
        O.solve_lambert([7e6, 0, 0], [0, 7e6, 0], 3000.0, revolutions=-1)


def test_ground_track_uses_the_shortened_final_step():
    """100 s at 30 s steps ends at t = 100, not 4 * 30 = 120."""
    initial = O.StateVector(
        position_m=[7e6, 0.0, 0.0],
        velocity_ms=[0.0, 7546.0, 0.0],
        epoch_utc="2026-01-01T00:00:00Z",
    )
    states = O.propagate_orbit_j2(initial, 100.0, 30.0)
    assert states[-1].epoch_utc == "2026-01-01T00:01:40+00:00"
    track = O.calculate_ground_track(states, 30.0)
    x, y, _ = states[-1].position_m
    expected = math.degrees(math.atan2(y, x) - O.OMEGA_EARTH * 100.0)
    assert track[-1].longitude_deg == pytest.approx(expected, abs=1e-9)


def test_ground_track_falls_back_to_index_times_without_usable_epochs():
    states = [
        O.StateVector(position_m=[7e6, 0, 0], velocity_ms=[0, 0, 0], epoch_utc="n/a")
    ] * 3
    track = O.calculate_ground_track(states, 60.0)
    assert track[2].longitude_deg == pytest.approx(
        math.degrees(-O.OMEGA_EARTH * 120.0), abs=1e-9
    )


def test_degenerate_rocket_geometry_is_an_error_not_infinity():
    geometry = dict(GEOMETRY, dry_mass_kg=0.0, propellant_mass_kg=0.0)
    out = rocket_tools.rocket_3dof_trajectory(geometry, {})
    assert "dry_mass_kg" in out and "Infinity" not in out
    with pytest.raises(ValueError, match="propellant_mass_kg"):
        R.rocket_3dof_trajectory(
            R.RocketGeometry(**dict(GEOMETRY, propellant_mass_kg=-1))
        )
