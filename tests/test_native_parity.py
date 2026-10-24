"""Parity tests: the native (Rust) core against the legacy pure-Python code.

``tests/data/legacy_python_reference.json`` was recorded from the pure-Python
trajectory, optimizer and orbit implementations (commit ``source_commit`` in
the file) before they were moved to ``rust/``. The port reproduced those
results to floating-point rounding. Model bugs inherited from the Python
code were then fixed on purpose, so only part of the reference still
applies.

Still compared against the legacy reference (unaffected by the fixes):

* thrust-curve interpolation (``thrust_at_time``);
* trajectories and launch-angle optimizations of the unpowered geometry;
* every orbit kernel: element/state conversions, orbit properties, Kepler's
  equation, Hohmann transfers, J2 propagation (with the legacy J2 sign, see
  below) and ground tracks.

Fixes that break parity with the legacy reference, and why:

* Propellant flow (``rust/src/rocket.rs``). The legacy model burned
  ``m_prop / t_burn`` with ``t_burn`` the last curve sample with positive
  thrust, so ``[[0, F], [6, 0]]`` emptied the tank after 1 s and the
  ``high_alt`` geometry flew its last 5 s of thrust on dry mass. Mass flow
  is now proportional to thrust (``mdot = F * m_prop / I_total``), so the
  tank empties exactly when the curve ends.
* Thrust quadrature (same file). Each Euler step applied the thrust sampled
  at its start, so the delivered impulse depended on how curve knots fell
  on the time grid (+0.8% for ``small_vertical`` at ``dt = 0.1``). Each step
  now applies the curve's mean over the step, so the delivered impulse is
  the curve's impulse, and ``analyze`` reports that impulse (each sample's
  thrust held until the next sample, not a trapezoid over samples) and
  computes Isp from the propellant consumed over the same samples.
* Atmosphere above the table. The legacy fallback restarted from sea-level
  formulas above 50 km (speed of sound 330 -> 108 m/s, density x3), so
  ``high_alt``'s max Mach was 31.7. Density now decays from the table's top
  row with the top interval's scale height and the speed of sound is held;
  the production table also goes up to 80 km instead of 50 km.
* Thrust-profile optimizer (``rust/src/optimize.rs``). The impulse target was
  met with a rectangle rule over a linearly interpolated curve (5400 N s
  delivered for a 10800 N s target with one segment), and the fixed 0.05
  step on raw meters drove every ``max_altitude`` case to the 1 m floor. The
  profile is now a step curve that delivers the target exactly, optimized by
  a line search on the normalized objective.
* Projected gradient descent divided one-sided differences at a bound by
  ``2h`` instead of ``h`` (``gradient_descent`` is not in the reference).

Together these move the legacy rockets' apogees by -1.7% to -4.2%
(``small_vertical`` and ``heavy``; the legacy sim over-delivered impulse and
burned propellant early) and by -30% for ``high_alt`` (which also no longer
reaches the legacy fallback's Mach 31). Affected cases are checked here
instead against ``_corrected_reference``: the legacy Python loop with exactly
the fixes above applied, plus physical invariants (propellant and impulse
accounting, continuity at the table top, optimizer never worse than its
start).

Intentional orbit fixes made during the port (reference still used):

* The J2 perturbation sign (the legacy code advanced the node of prograde
  orbits instead of regressing it). Legacy propagation results are compared
  by running the native propagator with the legacy (negated) J2 coefficient.
* Ground-track longitude wrapping uses a modulo instead of repeated
  subtraction; longitudes are compared modulo 360 degrees.

The rocket simulation's atmosphere table depends on whether the optional
``ambiance`` package is installed, so the tests feed the native simulation
the exact table the reference was recorded with (``atmosphere_table``).
"""

import bisect
import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from aerospace_mcp import _native
from aerospace_mcp.integrations import orbits as O
from aerospace_mcp.integrations import rockets as R
from aerospace_mcp.integrations import trajopt as T

REFERENCE = json.loads(
    (Path(__file__).parent / "data" / "legacy_python_reference.json").read_text()
)
TABLE = REFERENCE["atmosphere_table"]

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
POWERED = [c for c in REFERENCE["trajectories"] if c["geometry"] != "no_thrust"]
UNPOWERED = [c for c in REFERENCE["trajectories"] if c["geometry"] == "no_thrust"]

REL = 1e-9


@pytest.fixture(autouse=True)
def reference_atmosphere(monkeypatch):
    """Simulate with the atmosphere the reference was recorded with."""
    table = _native.AtmosphereTable(
        TABLE["altitudes_m"], TABLE["densities_kg_m3"], TABLE["speeds_of_sound_ms"]
    )
    monkeypatch.setattr(R, "_native_atmosphere", lambda *_: table)
    monkeypatch.setattr(T, "_native_atmosphere", lambda *_: table)


def approx(value):
    return pytest.approx(value, rel=REL, abs=1e-9)


def _case_id(case):
    return f"{case['geometry']}-{case['launch_angle_deg']}deg-dt{case['dt_s']}"


# ---------------------------------------------------------------------------
# Corrected reference model: the legacy Python loop plus the documented fixes
# ---------------------------------------------------------------------------


def _impulse_function(curve):
    """``(I, I_total)``: impulse of the interpolated curve from t = 0."""
    if not curve:
        return (lambda t: 0.0), 0.0
    if len(curve) == 1:
        t0, f0 = curve[0]
        return (lambda t: f0 * max(min(t, t0), 0.0)), f0 * max(t0, 0.0)
    times = [t for t, _ in curve]
    prefix = [0.0]
    for (ta, fa), (tb, fb) in zip(curve, curve[1:], strict=False):
        prefix.append(prefix[-1] + 0.5 * (fa + fb) * (tb - ta))

    def since_first(t):
        if t <= times[0]:
            return 0.0
        if t >= times[-1]:
            return prefix[-1]
        j = bisect.bisect_right(times, t) - 1
        (ta, fa), (tb, fb) = curve[j], curve[j + 1]
        f = fa + (fb - fa) * ((t - ta) / (tb - ta))
        return prefix[j] + 0.5 * (fa + f) * (t - ta)

    before = since_first(0.0)
    return (lambda t: since_first(t) - before), prefix[-1] - before


def _corrected_reference(geometry, dt_s, launch_angle_deg, max_time_s=300.0):
    """Legacy ``rocket_3dof_trajectory`` with the fixes in the module docstring."""
    altitudes = TABLE["altitudes_m"]
    densities = TABLE["densities_kg_m3"]
    speeds = TABLE["speeds_of_sound_ms"]
    h_top, rho_top, a_top = altitudes[-1], densities[-1], speeds[-1]
    scale_height = (altitudes[-1] - altitudes[-2]) / math.log(
        densities[-2] / densities[-1]
    )
    impulse_to, total_impulse = _impulse_function(geometry["thrust_curve"])
    dry, prop = geometry["dry_mass_kg"], geometry["propellant_mass_kg"]

    trajectory = []
    time = altitude = velocity_vertical = velocity_horizontal = impulse = 0.0
    mass = dry + prop
    launch_angle_rad = np.radians(launch_angle_deg)
    sin_launch, cos_launch = np.sin(launch_angle_rad), np.cos(launch_angle_rad)
    drag_area = math.pi * (geometry["diameter_m"] / 2) ** 2

    while time <= max_time_s and altitude >= 0:
        next_time = time + dt_s
        next_impulse = impulse_to(next_time)
        thrust = max((next_impulse - impulse) / dt_s, 0.0)  # fix: step mean
        if altitude <= h_top:
            rho = float(np.interp(altitude, altitudes, densities))
            speed_of_sound = float(np.interp(altitude, altitudes, speeds))
        else:  # fix: continue from the top row
            rho = rho_top * math.exp(-(altitude - h_top) / scale_height)
            speed_of_sound = a_top

        velocity_total = math.sqrt(velocity_vertical**2 + velocity_horizontal**2)
        mach = velocity_total / speed_of_sound
        dynamic_pressure = 0.5 * rho * velocity_total**2
        drag_force = geometry["cd"] * drag_area * dynamic_pressure
        if velocity_total > 0:
            drag_v = -drag_force * (velocity_vertical / velocity_total) / mass
            drag_h = -drag_force * (velocity_horizontal / velocity_total) / mass
        else:
            drag_v = drag_h = 0.0
        if time < 5.0:
            thrust_v, thrust_h = thrust * sin_launch / mass, thrust * cos_launch / mass
        else:
            thrust_v, thrust_h = thrust / mass, 0.0
        accel_v = thrust_v - R.G0 + drag_v
        accel_h = thrust_h + drag_h

        trajectory.append(
            [
                time,
                altitude,
                velocity_total,
                math.sqrt(accel_v**2 + accel_h**2),
                mass,
                thrust,
                drag_force,
                mach,
                dynamic_pressure,
            ]
        )
        velocity_vertical += accel_v * dt_s
        velocity_horizontal += accel_h * dt_s
        altitude += velocity_vertical * dt_s
        if total_impulse > 0:  # fix: mass flow proportional to thrust
            mass = dry + prop * max(1.0 - next_impulse / total_impulse, 0.0)
        impulse = next_impulse
        time = next_time
        if altitude > 0 and velocity_vertical < -1.0:
            break
    return trajectory


def _reference_performance(trajectory):
    """Legacy ``analyze_rocket_performance`` with the impulse/Isp fix."""
    cols = list(zip(*trajectory, strict=True))
    times, alts, vels, _, masses, thrusts, _, machs, qs = cols
    apogee = int(np.argmax(alts))
    active = [i for i, f in enumerate(thrusts) if f > 100]
    burnout = min(active[-1] + 1, len(times) - 1) if active else len(times) - 1
    # fix: each sample's thrust held until the next sample
    impulse = sum(
        f * (b - a) for f, a, b in zip(thrusts, times, times[1:], strict=False)
    )
    used = masses[0] - masses[-1]  # fix: over the same samples
    return {
        "max_altitude_m": alts[apogee],
        "apogee_time_s": times[apogee],
        "max_velocity_ms": max(vels),
        "max_mach": max(machs),
        "max_q_pa": max(qs),
        "burnout_altitude_m": alts[burnout],
        "burnout_velocity_ms": vels[burnout],
        "burnout_time_s": times[burnout],
        "total_impulse_ns": impulse,
        "specific_impulse_s": impulse / (used * R.G0) if used > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# Unaffected by the fixes: compared against the legacy reference
# ---------------------------------------------------------------------------


def test_thrust_interpolation_parity():
    for case in REFERENCE["thrust_at_time"]:
        assert R.get_thrust_at_time(case["curve"], case["t"]) == case["thrust"]


@pytest.mark.parametrize("case", UNPOWERED, ids=_case_id)
def test_unpowered_trajectory_parity(case):
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


@pytest.mark.parametrize(
    "case",
    [c for c in REFERENCE["launch_angle"] if c["geometry"] == "no_thrust"],
    ids=lambda c: f"{c['geometry']}-{c['bounds']}",
)
def test_unpowered_launch_angle_optimizer_parity(case):
    result = T.optimize_launch_angle(
        R.RocketGeometry(**GEOMETRIES[case["geometry"]]),
        angle_bounds=tuple(case["bounds"]),
    )
    assert result.optimal_parameters["launch_angle_deg"] == approx(case["angle"])
    assert result.optimal_objective == approx(case["objective"])


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
# Powered flight: corrected reference model and invariants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", POWERED, ids=_case_id)
def test_powered_trajectory_matches_corrected_reference(case):
    """Every sample of the native run equals the legacy loop plus the fixes."""
    spec = GEOMETRIES[case["geometry"]]
    expected = _corrected_reference(spec, case["dt_s"], case["launch_angle_deg"])
    trajectory = R.rocket_3dof_trajectory(
        R.RocketGeometry(**spec),
        dt_s=case["dt_s"],
        launch_angle_deg=case["launch_angle_deg"],
    )

    assert len(trajectory) == len(expected)
    for point, row in zip(trajectory, expected, strict=True):
        assert list(asdict(point).values()) == approx(row)
    assert asdict(R.analyze_rocket_performance(trajectory)) == approx(
        _reference_performance(expected)
    )


@pytest.mark.parametrize("case", POWERED, ids=_case_id)
def test_powered_trajectory_invariants(case):
    """Propellant and impulse accounting of the corrected model."""
    spec = GEOMETRIES[case["geometry"]]
    curve = spec["thrust_curve"]
    burn_end = curve[-1][0]
    _, i_total = _impulse_function(curve)
    dry, prop = spec["dry_mass_kg"], spec["propellant_mass_kg"]

    performance, trajectory = R.simulate_rocket(
        R.RocketGeometry(**spec),
        dt_s=case["dt_s"],
        launch_angle_deg=case["launch_angle_deg"],
    )
    points = [R.RocketTrajectoryPoint(*p) for p in trajectory.points()]
    masses = [p.mass_kg for p in points]

    # Mass never increases; the tank empties exactly when the curve ends.
    assert all(b <= a for a, b in zip(masses, masses[1:], strict=False))
    for p in points:
        if p.time_s < burn_end - 1e-9:
            assert p.mass_kg > dry
        else:
            assert p.mass_kg == pytest.approx(dry, abs=1e-9)
    # Delivered impulse is the curve's impulse; constant effective Isp.
    assert performance.total_impulse_ns == pytest.approx(i_total, rel=1e-12)
    assert performance.specific_impulse_s == pytest.approx(
        i_total / (prop * R.G0), rel=1e-12
    )
    # Burnout is the first sample at or after the end of the curve.
    end = performance.burnout_time_s
    assert burn_end - 1e-9 <= end < burn_end + case["dt_s"] - 1e-9


@pytest.mark.parametrize("case", POWERED, ids=_case_id)
def test_fixes_move_legacy_results_by_documented_amounts(case):
    legacy = case["performance"]
    performance, _ = R.simulate_rocket(
        R.RocketGeometry(**GEOMETRIES[case["geometry"]]),
        dt_s=case["dt_s"],
        launch_angle_deg=case["launch_angle_deg"],
    )
    change = performance.max_altitude_m / legacy["max_altitude_m"] - 1
    if case["geometry"] == "high_alt":
        assert -0.32 < change < -0.28
        # No more Mach 31 from the 108 m/s fallback speed of sound.
        assert legacy["max_mach"] > 30 and performance.max_mach < 10
    else:
        assert -0.045 < change < -0.015


def test_atmosphere_is_continuous_at_the_table_top():
    """Density and speed of sound used to jump x3 when crossing 50 km."""
    trajectory = R.rocket_3dof_trajectory(R.RocketGeometry(**GEOMETRIES["high_alt"]))
    top = TABLE["altitudes_m"][-1]
    crossing = [
        (a, b)
        for a, b in zip(trajectory, trajectory[1:], strict=False)
        if a.altitude_m <= top < b.altitude_m
    ]
    assert len(crossing) == 1
    a, b = crossing[0]
    # Across one 0.1 s step (~350 m of climb) both stay within 5%.
    assert b.mach == pytest.approx(a.mach, rel=0.01)
    assert b.dynamic_pressure_pa == pytest.approx(a.dynamic_pressure_pa, rel=0.05)


@pytest.mark.parametrize(
    "case",
    [c for c in REFERENCE["launch_angle"] if c["geometry"] != "no_thrust"],
    ids=lambda c: f"{c['geometry']}-{c['bounds']}",
)
def test_powered_launch_angle_optimizer_matches_corrected_reference(case):
    """Golden-section search over the corrected reference model."""
    spec = GEOMETRIES[case["geometry"]]

    def objective(angle):
        trajectory = _corrected_reference(spec, 0.1, angle)
        return -_reference_performance(trajectory)["max_altitude_m"]

    angle, value = _native.golden_section_search(objective, *case["bounds"], 0.1, 50)
    result = T.optimize_launch_angle(
        R.RocketGeometry(**spec), angle_bounds=tuple(case["bounds"])
    )
    assert result.optimal_parameters["launch_angle_deg"] == approx(angle)
    assert result.optimal_objective == approx(-value)
    # Same golden-section path as the legacy run: only the objective changed.
    assert angle == approx(case["angle"])


@pytest.mark.parametrize(
    "case",
    REFERENCE["thrust_profile"],
    ids=lambda c: f"{c['objective']}-{c['n_segments']}seg-{c['burn_time_s']}s",
)
def test_thrust_profile_optimizer_on_legacy_cases(case):
    """The legacy cases, checked for the properties the legacy run violated."""
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
    performance = result.performance

    # The impulse target is met (the legacy curve under-delivered it).
    assert performance.total_impulse_ns == pytest.approx(
        case["total_impulse_target"], rel=1e-9
    )
    # Never worse than constant thrust, and the reported objective is the
    # returned trajectory's metric.
    metric = {
        "max_altitude": performance.max_altitude_m,
        "min_max_q": performance.max_q_pa,
        "min_gravity_loss": R.G0 * performance.burnout_time_s,
    }[case["objective"]]
    assert result.optimal_objective == pytest.approx(metric, rel=1e-9)
    if case["objective"] == "max_altitude":
        assert result.optimal_objective >= result.initial_objective
        # The legacy optimizer reported the 1 m floor and a 0 m apogee.
        assert case["optimal_objective"] == 1.0 and case["max_altitude_m"] == 0.0
        assert performance.max_altitude_m > 500.0
    else:
        assert result.optimal_objective <= result.initial_objective


# ---------------------------------------------------------------------------
# Intentional differences from the legacy orbit implementation
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
