"""Parity tests: the native (Rust) core against the legacy pure-Python code.

``tests/data/legacy_python_reference.json`` was recorded from the pure-Python
trajectory, optimizer and orbit implementations (commit ``source_commit`` in
the file) before they were moved to ``rust/``. These tests check the port
reproduces those results to floating-point rounding, except for two
intentional fixes covered at the end of this module:

* The J2 perturbation sign (the legacy code advanced the node of prograde
  orbits instead of regressing it). Legacy propagation results are compared
  by running the native propagator with the legacy (negated) J2 coefficient.
* Ground-track longitude wrapping uses a modulo instead of repeated
  subtraction; longitudes are compared modulo 360 degrees.
"""

import json
import math
from dataclasses import asdict
from pathlib import Path

import pytest

from aerospace_mcp import _native
from aerospace_mcp.integrations import orbits as O
from aerospace_mcp.integrations import rockets as R
from aerospace_mcp.integrations import trajopt as T

REFERENCE = json.loads(
    (Path(__file__).parent / "data" / "legacy_python_reference.json").read_text()
)

GEOMETRIES = {
    "small_vertical": {
        "dry_mass_kg": 15.0,
        "propellant_mass_kg": 60.0,
        "diameter_m": 0.18,
        "length_m": 1.5,
        "cd": 0.35,
        "thrust_curve": [[0.0, 1800.0], [6.0, 1800.0], [6.1, 0.0]],
    },
    "heavy": {
        "dry_mass_kg": 25.0,
        "propellant_mass_kg": 100.0,
        "diameter_m": 0.25,
        "length_m": 2.0,
        "cd": 0.4,
        "thrust_curve": [[0.0, 3000.0], [8.0, 3000.0], [8.1, 0.0]],
    },
    "high_alt": {
        "dry_mass_kg": 50.0,
        "propellant_mass_kg": 450.0,
        "diameter_m": 0.3,
        "length_m": 6.0,
        "cd": 0.3,
        "thrust_curve": [[0.0, 40000.0], [2.0, 38000.0], [20.0, 30000.0], [25.0, 0.0]],
    },
    "no_thrust": {
        "dry_mass_kg": 10.0,
        "propellant_mass_kg": 20.0,
        "diameter_m": 0.1,
        "length_m": 1.0,
        "cd": 0.3,
        "thrust_curve": [],
    },
}

REL = 1e-9


def approx(value):
    return pytest.approx(value, rel=REL, abs=1e-9)


def _case_id(case):
    return f"{case['geometry']}-{case['launch_angle_deg']}deg-dt{case['dt_s']}"


@pytest.mark.parametrize("case", REFERENCE["trajectories"], ids=_case_id)
def test_trajectory_parity(case):
    geometry = R.RocketGeometry(**GEOMETRIES[case["geometry"]])
    trajectory = R.rocket_3dof_trajectory(
        geometry, dt_s=case["dt_s"], launch_angle_deg=case["launch_angle_deg"]
    )

    assert len(trajectory) == case["n_points"]
    for index, expected in case["samples"].items():
        assert asdict(trajectory[int(index)]) == approx(expected)
    assert asdict(R.analyze_rocket_performance(trajectory)) == approx(
        case["performance"]
    )


def test_thrust_interpolation_parity():
    for case in REFERENCE["thrust_at_time"]:
        assert R.get_thrust_at_time(case["curve"], case["t"]) == case["thrust"]


@pytest.mark.parametrize(
    "case",
    REFERENCE["launch_angle"],
    ids=lambda c: f"{c['geometry']}-{c['bounds']}",
)
def test_launch_angle_optimizer_parity(case):
    result = T.optimize_launch_angle(
        R.RocketGeometry(**GEOMETRIES[case["geometry"]]),
        angle_bounds=tuple(case["bounds"]),
    )
    assert result.optimal_parameters["launch_angle_deg"] == approx(case["angle"])
    assert result.optimal_objective == approx(case["objective"])


@pytest.mark.parametrize(
    "case",
    REFERENCE["thrust_profile"],
    ids=lambda c: f"{c['objective']}-{c['n_segments']}seg-{c['burn_time_s']}s",
)
def test_thrust_profile_optimizer_parity(case):
    base = {
        "dry_mass_kg": 20.0,
        "propellant_mass_kg": 80.0,
        "diameter_m": 0.2,
        "length_m": 2.0,
        "cd": 0.4,
    }
    result = T.optimize_thrust_profile(
        R.RocketGeometry(**base),
        case["burn_time_s"],
        case["total_impulse_target"],
        case["n_segments"],
        case["objective"],
    )

    assert result.iterations == case["iterations"]
    assert result.converged == case["converged"]
    # 100 finite-difference gradient steps amplify rounding differences, so
    # the tolerance here is looser than for single simulations.
    expected_params = dict(case["optimal_parameters"])
    expected_curve = expected_params.pop("thrust_curve")
    actual_params = dict(result.optimal_parameters)
    actual_curve = actual_params.pop("thrust_curve")
    assert actual_params == pytest.approx(expected_params, rel=1e-6)
    assert len(actual_curve) == len(expected_curve)
    for actual, expected in zip(actual_curve, expected_curve, strict=True):
        assert actual == pytest.approx(expected, rel=1e-6)
    assert result.optimal_objective == pytest.approx(
        case["optimal_objective"], rel=1e-6
    )
    assert result.performance.max_altitude_m == pytest.approx(
        case["max_altitude_m"], rel=1e-6, abs=1e-6
    )


def test_elements_to_state_parity():
    for case in REFERENCE["orbits"]["elements_to_state"]:
        state = O.elements_to_state_vector(O.OrbitElements(**case["elements"]))
        assert state.position_m == approx(case["position_m"])
        assert state.velocity_ms == approx(case["velocity_ms"])


def test_state_to_elements_parity():
    for case in REFERENCE["orbits"]["state_to_elements"]:
        state = O.StateVector(
            position_m=case["position_m"],
            velocity_ms=case["velocity_ms"],
            epoch_utc="2026-01-01T00:00:00Z",
        )
        assert asdict(O.state_vector_to_elements(state)) == approx(case["elements"])


def test_orbit_properties_and_kepler_parity():
    for case in REFERENCE["orbits"]["properties"]:
        props = O.calculate_orbit_properties(O.OrbitElements(**case["elements"]))
        assert asdict(props) == approx(case["properties"])
    for case in REFERENCE["orbits"]["kepler"]:
        assert O.kepler_equation_solver(case["M"], case["e"]) == approx(case["E"])


def test_hohmann_parity():
    for case in REFERENCE["orbits"]["hohmann"]:
        assert O.hohmann_transfer(case["r1"], case["r2"]) == approx(case["result"])


@pytest.mark.parametrize(
    "case",
    REFERENCE["orbits"]["propagate"],
    ids=lambda c: f"{c['span']}s-step{c['step']}",
)
def test_propagation_parity_with_legacy_j2_sign(case):
    samples = _native.propagate_j2(
        case["initial"]["position_m"],
        case["initial"]["velocity_ms"],
        case["span"],
        case["step"],
        j2=-_native.J2_EARTH,  # the legacy sign
    )
    assert len(samples) == case["n_states"]
    _, final_r, final_v = samples[-1]
    _, mid_r, mid_v = samples[len(samples) // 2]
    # Errors grow along the trajectory; a day of RK4 steps keeps ~1e-9.
    assert final_r == pytest.approx(case["final"]["position_m"], rel=1e-8)
    assert final_v == pytest.approx(case["final"]["velocity_ms"], rel=1e-8)
    assert mid_r == pytest.approx(case["middle"]["position_m"], rel=1e-8)
    assert mid_v == pytest.approx(case["middle"]["velocity_ms"], rel=1e-8)


def test_propagation_epochs_match_legacy():
    case = REFERENCE["orbits"]["propagate"][2]  # 1000 s at 7 s steps
    initial = O.StateVector(**case["initial"])
    states = O.propagate_orbit_j2(initial, case["span"], case["step"])
    assert states[-1].epoch_utc == case["final"]["epoch_utc"]
    assert states[len(states) // 2].epoch_utc == case["middle"]["epoch_utc"]


def test_ground_track_parity():
    ref = REFERENCE["orbits"]["ground_track"]
    states = [
        O.StateVector(position_m=p, velocity_ms=[0, 0, 0], epoch_utc="2026-01-01")
        for p in ref["positions"]
    ]
    track = O.calculate_ground_track(states, ref["time_step_s"])
    assert len(track) == len(ref["track"])
    for point, expected in zip(track, ref["track"], strict=True):
        assert point.latitude_deg == approx(expected["latitude_deg"])
        assert point.altitude_m == approx(expected["altitude_m"])
        dlon = (point.longitude_deg - expected["longitude_deg"] + 180) % 360 - 180
        assert abs(dlon) < 1e-9


# ---------------------------------------------------------------------------
# Intentional differences from the legacy implementation
# ---------------------------------------------------------------------------


def test_j2_regresses_the_node_of_prograde_orbits():
    """dRAAN/dt = -1.5 n J2 (R/p)^2 cos(i): about -4.5 deg/day here."""
    elements = O.OrbitElements(
        semi_major_axis_m=7.0e6,
        eccentricity=0.001,
        inclination_deg=51.6,
        raan_deg=30.0,
        arg_periapsis_deg=0.0,
        true_anomaly_deg=0.0,
        epoch_utc="2026-01-01T00:00:00Z",
    )
    a, e, i = 7.0e6, 0.001, math.radians(51.6)
    n = math.sqrt(O.MU_EARTH / a**3)
    p = a * (1 - e**2)
    expected_rate = -1.5 * n * O.J2_EARTH * (O.R_EARTH / p) ** 2 * math.cos(i)

    states = O.propagate_orbit_j2(O.elements_to_state_vector(elements), 86400.0, 30.0)
    final = O.state_vector_to_elements(states[-1])
    drift_deg = final.raan_deg - 30.0

    assert drift_deg == pytest.approx(math.degrees(expected_rate * 86400.0), rel=0.02)
    assert drift_deg < 0


def test_ground_track_handles_huge_time_steps():
    """The legacy while-loop longitude wrap never finished for huge steps."""
    states = [
        O.StateVector(position_m=[7e6, 0, 0], velocity_ms=[0, 0, 0], epoch_utc="x")
    ] * 2
    track = O.calculate_ground_track(states, 1e300)
    assert all(-180.0 <= p.longitude_deg < 180.0 for p in track)
