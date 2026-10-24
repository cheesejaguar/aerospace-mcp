"""Regression tests for defects found auditing the native (Rust) port.

Each test reproduces a case that previously crashed the server, leaked
NaN/Infinity into tool JSON, or returned an inconsistent answer. The second
half covers the model bugs inherited from the Python implementation and
fixed after the port (see ``tests/test_native_parity.py``).
"""

import json
import math

import pytest

from aerospace_mcp import _native
from aerospace_mcp.integrations import orbits as O
from aerospace_mcp.integrations import rockets as R
from aerospace_mcp.integrations import trajopt as T
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


# ---------------------------------------------------------------------------
# Inherited model bugs fixed after the port
# ---------------------------------------------------------------------------

PROFILE_CASES = [
    # (dry kg, propellant kg, diameter m, cd, burn s, impulse N s)
    (15.0, 60.0, 0.18, 0.35, 6.0, 10800.0),
    (20.0, 80.0, 0.2, 0.4, 8.0, 15000.0),
    (20.0, 80.0, 0.2, 0.4, 4.0, 8000.0),
    (50.0, 450.0, 0.3, 0.3, 25.0, 765000.0),
]


def _constant_thrust_performance(geometry, burn_time_s, impulse):
    thrust = impulse / burn_time_s
    constant = R.RocketGeometry(
        **{**geometry.__dict__, "thrust_curve": [[0.0, thrust], [burn_time_s, thrust]]}
    )
    return R.analyze_rocket_performance(R.rocket_3dof_trajectory(constant))


@pytest.mark.parametrize("case", PROFILE_CASES, ids=lambda c: f"{c[4]}s-{c[5]}Ns")
@pytest.mark.parametrize("n_segments", [2, 5])
def test_thrust_profile_never_worse_than_constant_thrust(case, n_segments):
    """max_altitude used to end at 0 m with every multiplier on a bound."""
    dry, prop, diameter, cd, burn, impulse = case
    geometry = R.RocketGeometry(dry, prop, diameter, 2.0, cd)
    baseline = _constant_thrust_performance(geometry, burn, impulse)

    best = T.optimize_thrust_profile(geometry, burn, impulse, n_segments)
    assert best.initial_objective == pytest.approx(baseline.max_altitude_m)
    assert best.performance.max_altitude_m >= baseline.max_altitude_m
    assert best.optimal_objective == best.performance.max_altitude_m

    q = T.optimize_thrust_profile(geometry, burn, impulse, n_segments, "min_max_q")
    assert q.performance.max_q_pa <= baseline.max_q_pa * (1 + 1e-12)

    loss = T.optimize_thrust_profile(
        geometry, burn, impulse, n_segments, "min_gravity_loss"
    )
    assert loss.optimal_objective <= R.G0 * baseline.burnout_time_s * (1 + 1e-12)


def test_thrust_profile_improves_the_audit_case():
    """15 kg + 60 kg, 6 s, 10800 N s: the legacy optimum was 0 m."""
    geometry = R.RocketGeometry(15.0, 60.0, 0.18, 1.5, 0.35)
    result = T.optimize_thrust_profile(geometry, 6.0, 10800.0, 5)
    assert result.optimal_objective > result.initial_objective + 100.0
    multipliers = [
        m for k, m in result.optimal_parameters.items() if k.startswith("thrust_mult")
    ]
    assert not all(m in (0.1, 3.0) for m in multipliers)


@pytest.mark.parametrize("n_segments", [1, 2, 5, 7])
def test_thrust_profile_delivers_the_impulse_target(n_segments):
    """Delivered 5400 / 5574 / 9461 N s of a 10800 N s target for n = 1/2/5."""
    geometry = R.RocketGeometry(15.0, 60.0, 0.18, 1.5, 0.35)
    result = T.optimize_thrust_profile(geometry, 6.0, 10800.0, n_segments)
    assert result.performance.total_impulse_ns == pytest.approx(10800.0, rel=1e-9)
    curve = result.optimal_parameters["thrust_curve"]
    assert len(curve) == 2 * n_segments
    assert curve[-1][0] == pytest.approx(6.0)


def test_single_segment_profile_burns_for_the_whole_burn_time():
    """n = 1 used to empty the tank at t = 1 s of a 6 s burn (Isp 9.2 s)."""
    geometry = R.RocketGeometry(15.0, 60.0, 0.18, 1.5, 0.35)
    result = T.optimize_thrust_profile(geometry, 6.0, 10800.0, 1)
    points = result.trajectory_points
    empty_at = next(p.time_s for p in points if p.mass_kg <= 15.0 + 1e-9)
    assert empty_at == pytest.approx(6.0, abs=1e-6)
    assert result.performance.specific_impulse_s == pytest.approx(
        10800.0 / (60.0 * R.G0)
    )


@pytest.mark.parametrize(
    "curve", [[[0.0, 10800.0], [6.0, 0.0]], [[0.0, 4000.0], [5.0, 0.0]]]
)
def test_propellant_lasts_until_the_thrust_curve_ends(curve):
    """Both curves used to burn all 60 kg in the first 1.0 s."""
    geometry = R.RocketGeometry(15.0, 60.0, 0.18, 1.5, 0.35, curve)
    trajectory = R.rocket_3dof_trajectory(geometry)
    empty_at = next(p.time_s for p in trajectory if p.mass_kg <= 15.0 + 1e-9)
    assert empty_at == pytest.approx(curve[-1][0], abs=1e-6)
    impulse = 0.5 * curve[0][1] * curve[-1][0]
    performance = R.analyze_rocket_performance(trajectory)
    assert performance.total_impulse_ns == pytest.approx(impulse, rel=1e-12)
    assert performance.specific_impulse_s == pytest.approx(impulse / (60 * R.G0))


def test_production_atmosphere_is_continuous_up_high():
    """Mach 15.3 -> 46.4 and q 13 -> 40 kPa used to happen across 50 km."""
    geometry = R.RocketGeometry(
        50.0, 450.0, 0.3, 6.0, 0.3, [[0, 40000.0], [2, 38000.0], [20, 30000.0], [25, 0]]
    )
    trajectory = R.rocket_3dof_trajectory(geometry)
    assert max(p.altitude_m for p in trajectory) > 100_000
    for a, b in zip(trajectory, trajectory[1:], strict=False):
        if a.altitude_m > 10_000:
            # Both change smoothly up here: a few percent per 0.1 s step.
            assert b.mach == pytest.approx(a.mach, rel=0.02), (a, b)
            assert b.dynamic_pressure_pa == pytest.approx(
                a.dynamic_pressure_pa, rel=0.07
            ), (a, b)


def test_gradient_is_not_halved_at_a_bound():
    params, value, _, _ = _native.gradient_descent(
        lambda p: p[0], [1.0], [(0.0, 1.0)], 0.1, 1e-12, 1
    )
    assert params == [pytest.approx(0.9)]
    assert value == pytest.approx(0.9)


@pytest.mark.parametrize("tof", [329.87, 1684.0])
@pytest.mark.parametrize("angle_deg", [1e-4, 1e-3])
def test_lambert_near_zero_angle_round_trip(angle_deg, tof):
    """Needed > 35 Householder iterations (RuntimeError: no convergence)."""
    r = 7e6
    th = math.radians(angle_deg)
    r1, r2 = [r, 0.0, 0.0], [r * math.cos(th), r * math.sin(th), 0.0]
    solution = O.solve_lambert(r1, r2, tof)
    step = max(tof / 20000, 0.1)
    samples = _native.propagate_j2(r1, solution["v1_ms"], tof, step, j2=0.0)
    _, r_end, v_end = samples[-1]
    assert r_end == pytest.approx(r2, abs=1e-6 * r)
    assert v_end == pytest.approx(solution["v2_ms"], abs=1e-6 * 8000)


def test_sizing_tool_accepts_design_margin():
    """Always failed: 'takes from 2 to 3 positional arguments but 4 were given'."""
    out = _strict_json(rocket_tools.estimate_rocket_sizing(30000.0, 10.0))
    assert out["design_margin"] == 1.2
    assert out["delta_v_ms"] == pytest.approx(1.2 * out["delta_v_without_margin_ms"])
    infeasible = _strict_json(rocket_tools.estimate_rocket_sizing(500000.0, 10.0))
    assert infeasible["feasible"] is False and infeasible["total_mass_kg"] is None
    assert "error" in rocket_tools.estimate_rocket_sizing(30000.0, 10.0, "solid", 0.5)


def test_sensitivity_with_zero_baseline_has_no_division_errors():
    """A rocket that never lifts off has baseline 0; dividing by it failed."""
    grounded = dict(GEOMETRY, thrust_curve=[])
    out = _strict_json(
        opt_tools.trajectory_sensitivity_analysis(
            grounded, {"cd": [0.3, 0.5], "propellant_mass_kg": [0.0, 40.0]}
        )
    )
    assert out["baseline_value"] == 0.0
    for points in out["parameter_sensitivities"].values():
        for point in points:
            assert point["objective_value"] == 0.0
            assert point["percent_change_objective"] is None
            assert point["sensitivity"] is None


def test_sensitivity_with_zero_baseline_parameter():
    """cd = 0 as the baseline: the parameter's percent change is undefined."""
    out = _strict_json(
        opt_tools.trajectory_sensitivity_analysis(
            dict(GEOMETRY, cd=0.0, thrust_curve=[[0, 2000.0], [5, 2000.0]]),
            {"cd": [0.0, 0.4]},
        )
    )
    unchanged, changed = out["parameter_sensitivities"]["cd"]
    assert unchanged["percent_change_param"] is None
    assert changed["percent_change_param"] is None and changed["sensitivity"] is None
    assert changed["percent_change_objective"] < 0


def test_sensitivity_rejects_unknown_parameters_and_objectives():
    geometry = R.RocketGeometry(**GEOMETRY)
    with pytest.raises(ValueError, match="length_m"):
        T.trajectory_sensitivity_analysis(geometry, {"length_m": [1.0]})
    with pytest.raises(ValueError, match="objective"):
        T.trajectory_sensitivity_analysis(geometry, {"cd": [0.3]}, "max_range")


def test_optimizers_reject_inputs_that_used_to_give_infinities():
    geometry = R.RocketGeometry(**dict(GEOMETRY, thrust_curve=[[0, 2000.0], [5, 0]]))
    with pytest.raises(ValueError, match="objective"):
        T.optimize_launch_angle(geometry, objective="max_speed")  # was -Infinity
    with pytest.raises(ValueError, match="finite"):
        T.optimize_launch_angle(geometry, angle_bounds=(math.nan, 90.0))
    with pytest.raises(ValueError, match="objective"):
        T.optimize_thrust_profile(geometry, 5.0, 10000.0, 3, "fastest")
    bad = R.RocketGeometry(**dict(GEOMETRY, dry_mass_kg=0.0))
    with pytest.raises(ValueError, match="dry_mass_kg"):
        T.optimize_thrust_profile(bad, 5.0, 10000.0, 3)
    out = opt_tools.optimize_thrust_profile(GEOMETRY, 5.0, 10000.0, 3, "fastest")
    assert "error" in out.lower() and "Infinity" not in out


def test_simulate_rocket_matches_the_dataclass_path():
    geometry = R.RocketGeometry(**dict(GEOMETRY, thrust_curve=[[0, 2500.0], [5, 0]]))
    points = R.rocket_3dof_trajectory(geometry, dt_s=0.01)
    performance, trajectory = R.simulate_rocket(geometry, dt_s=0.01)

    assert performance == R.analyze_rocket_performance(points)
    assert len(trajectory) == len(points)
    assert R.RocketTrajectoryPoint(*trajectory[-1]) == points[-1]
    middle = len(points) // 2
    assert R.RocketTrajectoryPoint(*trajectory[middle]) == points[middle]
    assert [R.RocketTrajectoryPoint(*p) for p in trajectory.points()] == points
    with pytest.raises(IndexError):
        trajectory[len(points)]
    with pytest.raises(IndexError):
        trajectory[-len(points) - 1]


def test_rocket_tool_reports_the_native_performance():
    geometry = dict(GEOMETRY, thrust_curve=[[0, 2500.0], [5, 0]])
    out = rocket_tools.rocket_3dof_trajectory(geometry, {}, {"dt_s": 0.001})
    reported = json.loads(out.split("JSON Performance Data:")[1])["performance"]
    expected = R.analyze_rocket_performance(
        R.rocket_3dof_trajectory(R.RocketGeometry(**geometry), dt_s=0.001)
    )
    for key, value in reported.items():
        assert value == getattr(expected, key)
