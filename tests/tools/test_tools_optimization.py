from __future__ import annotations

import json
import math

import pytest

from tests.tools.test_tools_common import without_integrations

ROCKET = {
    "dry_mass_kg": 20.0,
    "propellant_mass_kg": 80.0,
    "diameter_m": 0.2,
    "length_m": 2.0,
    "cd": 0.4,
    "thrust_curve": [[0.0, 3000.0], [5.0, 3000.0], [5.1, 0.0]],
}


def test_optimization_import_errors(monkeypatch):
    from aerospace_mcp.tools import optimization as opt_tools

    # GA / PSO / Monte Carlo live in integrations.gnc (the tools used to
    # import them from integrations.trajopt, where they never existed).
    without_integrations(monkeypatch, "trajopt", "orbits", "gnc")
    assert "not available" in opt_tools.optimize_thrust_profile({}, 1.0, 1.0).lower()
    assert "not available" in opt_tools.trajectory_sensitivity_analysis({}, {}).lower()
    assert "not available" in opt_tools.genetic_algorithm_optimization({}).lower()
    assert "not available" in opt_tools.particle_swarm_optimization({}).lower()
    assert (
        "not available"
        in opt_tools.porkchop_plot_analysis("Earth", "Mars", [], []).lower()
    )
    assert (
        "not available" in opt_tools.monte_carlo_uncertainty_analysis({}, {}, 1).lower()
    )


def test_optimization_invalid_input_errors():
    from aerospace_mcp.tools import optimization as opt

    out = opt.optimize_thrust_profile({}, 1.0, 1.0)
    assert out.startswith("Thrust profile optimization error") and "dry_mass_kg" in out
    out = opt.trajectory_sensitivity_analysis({}, {})
    assert out.startswith("Sensitivity analysis error") and "dry_mass_kg" in out
    assert opt.porkchop_plot_analysis(
        "Earth", "Mars", ["notadate"], ["2025-08-01"]
    ).startswith("Porkchop plot error")


def test_porkchop_rejects_unmodelled_bodies():
    """Only Earth and Mars orbits are modelled; other names used to fall back
    to Earth silently and return a meaningless Earth-to-Earth grid."""
    from aerospace_mcp.tools.optimization import porkchop_plot_analysis

    out = porkchop_plot_analysis("Earth", "Venus", ["2025-01-01"], ["2025-08-01"])
    assert out == (
        "Porkchop plot error: Unsupported body 'Venus': only 'Earth' and 'Mars' "
        "are modelled"
    )
    # Case-insensitive for the supported bodies
    grid = json.loads(
        porkchop_plot_analysis("earth", "MARS", ["2025-01-01"], ["2025-08-01"])
    )
    assert len(grid["transfer_grid"]) == 1


def test_sensitivity_objective_is_read_from_options():
    """The objective comes from analysis_options and changes the result."""
    from aerospace_mcp.tools.optimization import trajectory_sensitivity_analysis

    variations = {"cd": [0.3, 0.5]}
    altitude = json.loads(trajectory_sensitivity_analysis(ROCKET, variations))
    velocity = json.loads(
        trajectory_sensitivity_analysis(
            ROCKET, variations, {"objective": "max_velocity"}
        )
    )
    assert altitude["objective"] == "max_altitude"
    assert velocity["objective"] == "max_velocity"
    assert velocity["baseline_value"] != altitude["baseline_value"]
    # Less drag -> higher apogee; more drag -> lower
    low_cd, high_cd = altitude["parameter_sensitivities"]["cd"]
    assert low_cd["objective_value"] > altitude["baseline_value"]
    assert high_cd["objective_value"] < altitude["baseline_value"]
    assert low_cd["percent_change_param"] == pytest.approx(-25.0)


def test_thrust_profile_and_porkchop_end_to_end():
    """Real native-backed run through the tool layer (no stubs)."""
    import json

    import pytest

    from aerospace_mcp.tools.optimization import (
        optimize_thrust_profile,
        porkchop_plot_analysis,
    )

    out = json.loads(
        optimize_thrust_profile(
            {
                "dry_mass_kg": 20.0,
                "propellant_mass_kg": 80.0,
                "diameter_m": 0.2,
                "length_m": 2.0,
                "cd": 0.4,
            },
            5.0,
            10000.0,
            3,
            "min_max_q",
        )
    )
    assert out["iterations"] >= 1
    # A step profile: start and end sample per segment.
    assert len(out["optimal_parameters"]["thrust_curve"]) == 6
    # Never worse than the constant-thrust start; the impulse target is met.
    assert out["optimal_objective"] <= out["initial_objective"]
    assert out["performance"]["total_impulse_ns"] == pytest.approx(10000.0)

    grid = json.loads(
        porkchop_plot_analysis(
            "Earth",
            "Mars",
            ["2025-04-01T00:00:00Z", "2025-05-01T00:00:00Z"],
            ["2026-01-01T00:00:00Z", "2026-02-01T00:00:00Z"],
            {"min_tof_days": 200, "max_tof_days": 400},
        )
    )
    assert len(grid["transfer_grid"]) == 4
    # Earth-Mars Hohmann C3 is ~8.7 km^2/s^2 for circular coplanar orbits.
    assert 8.0 < grid["optimal_transfer"]["c3_km2_s2"] < 15.0


# ---------------------------------------------------------------------------
# End-to-end GA / PSO / Monte Carlo through the tool layer (no stubs)
# ---------------------------------------------------------------------------

G0 = 9.80665
ISP = 300.0


def _strict_json(text: str):
    """Parse JSON rejecting NaN/Infinity (not valid JSON for MCP clients)."""

    def _reject(token):
        raise ValueError(f"non-standard JSON constant {token}: {text[:200]}")

    try:
        return json.loads(text, parse_constant=_reject)
    except json.JSONDecodeError:
        raise AssertionError(f"tool did not return JSON: {text[:300]}") from None


def _trajectory(thrust_n=(100.0, 0.0, 0.0), n=5, dt=100.0):
    return [
        {
            "time_s": dt * i,
            "position_m": [6.9e6, 0.0, 0.0],
            "velocity_ms": [0.0, 7500.0, 0.0],
            "thrust_n": list(thrust_n),
            "mass_kg": 1000.0,
        }
        for i in range(n)
    ]


def _fuel(trajectory):
    """dm = |F| dt / (Isp g0) per segment (thrust of the segment's end point)."""
    return sum(
        math.dist(b["thrust_n"], [0, 0, 0]) * (b["time_s"] - a["time_s"]) / (ISP * G0)
        for a, b in zip(trajectory, trajectory[1:], strict=False)
    )


@pytest.mark.parametrize("tool_name", ["genetic", "pso"])
def test_optimizers_end_to_end_improve_fuel(tool_name):
    from aerospace_mcp.tools import optimization as opt

    problem = {"trajectory": _trajectory(), "objective": "minimize_fuel"}
    if tool_name == "genetic":
        out = opt.genetic_algorithm_optimization(
            problem, {"population_size": 20, "generations": 60, "seed": 7}
        )
    else:
        out = opt.particle_swarm_optimization(
            problem, {"n_particles": 20, "iterations": 60, "seed": 7}
        )
    result = _strict_json(out)
    # 100 N for 4 x 100 s at Isp 300 s: 100 * 400 / (300 * 9.80665) kg
    assert result["initial_cost"] == pytest.approx(100 * 400 / (ISP * G0))
    assert result["optimal_cost"] < result["initial_cost"]
    assert result["fuel_mass_kg"] == pytest.approx(
        _fuel(result["optimal_trajectory"]), rel=1e-12
    )
    # Only thrust is optimised: times come from the input
    assert [wp["time_s"] for wp in result["optimal_trajectory"]] == [
        0.0,
        100.0,
        200.0,
        300.0,
        400.0,
    ]
    assert 1 <= result["iterations"] <= 60


def test_ga_reports_stagnation_as_converged():
    """Zero thrust is already optimal (cost 0): the GA stops after 21
    generations without improvement and reports converged=True (the flag
    used to be inverted and iterations always equalled 'generations')."""
    from aerospace_mcp.tools import optimization as opt

    problem = {"trajectory": _trajectory(thrust_n=(0, 0, 0))}
    result = _strict_json(
        opt.genetic_algorithm_optimization(
            problem, {"population_size": 10, "generations": 200, "seed": 1}
        )
    )
    assert result["optimal_cost"] == 0.0
    assert result["converged"] is True
    assert result["iterations"] == 22


def test_pso_runs_until_converged():
    """PSO convergence uses a real history of the global best (it used to
    stop after 12 iterations whatever happened)."""
    from aerospace_mcp.tools import optimization as opt

    # Optimal start: the global best never improves -> stops after the window
    flat = _strict_json(
        opt.particle_swarm_optimization(
            {"trajectory": _trajectory(thrust_n=(0, 0, 0))},
            {"n_particles": 5, "iterations": 100, "seed": 1},
        )
    )
    assert flat["converged"] is True and flat["iterations"] == 11

    # Improving run: allowed to use more than 12 iterations
    busy = _strict_json(
        opt.particle_swarm_optimization(
            {"trajectory": _trajectory()},
            {"n_particles": 10, "iterations": 40, "seed": 2},
        )
    )
    assert busy["iterations"] > 12


def test_optimizers_validate_inputs_and_bounds():
    from aerospace_mcp.tools import optimization as opt

    assert "needs a 'trajectory'" in opt.genetic_algorithm_optimization({})
    bad_times = _trajectory()
    bad_times[2]["time_s"] = 0.0
    assert "strictly increasing" in opt.particle_swarm_optimization(
        {"trajectory": bad_times}
    )
    assert "objective must be one of" in opt.genetic_algorithm_optimization(
        {"trajectory": _trajectory(), "objective": "fastest"}
    )
    assert "population_size must be between" in opt.genetic_algorithm_optimization(
        {"trajectory": _trajectory()}, {"population_size": 10_000}
    )
    assert "exceeds the limit" in opt.particle_swarm_optimization(
        {"trajectory": _trajectory(n=100)},
        {"num_particles": 200, "max_iterations": 500},
    )
    assert "Unknown GA parameters" in opt.genetic_algorithm_optimization(
        {"trajectory": _trajectory()}, {"popsize": 3}
    )


def test_monte_carlo_end_to_end_known_answers():
    """Position errors ~ N(0, s^2 I3): |e| is Maxwell distributed with mean
    2 s sqrt(2/pi).  Without thrust errors delta-V equals the nominal
    sum(|F|/m dt) = 100 / 1000 * 400 = 40 m/s exactly."""
    from aerospace_mcp.tools import optimization as opt

    sigma = 100.0
    args = ({"trajectory": _trajectory()}, {"position_m": {"std": sigma}}, 4000)
    result = _strict_json(opt.monte_carlo_uncertainty_analysis(*args, {"seed": 11}))
    assert result["n_samples"] == 4000
    assert result["delta_v_statistics"]["mean_ms"] == pytest.approx(40.0)
    assert result["delta_v_statistics"]["std_ms"] == pytest.approx(0.0, abs=1e-12)
    expected_mean = 2 * sigma * math.sqrt(2 / math.pi)
    assert result["position_error_statistics"]["mean_m"] == pytest.approx(
        expected_mean, rel=0.05
    )
    lo, hi = result["confidence_intervals"]["position_95_m"]
    assert 0 < lo < expected_mean < hi
    # Same seed -> identical results
    again = _strict_json(opt.monte_carlo_uncertainty_analysis(*args, {"seed": 11}))
    assert again["position_error_statistics"] == result["position_error_statistics"]


def test_monte_carlo_validates_inputs():
    from aerospace_mcp.tools import optimization as opt

    nominal = {"trajectory": _trajectory()}
    assert "n_samples must be between" in opt.monte_carlo_uncertainty_analysis(
        nominal, {}, 1
    )
    assert "Unsupported uncertainty" in opt.monte_carlo_uncertainty_analysis(
        nominal, {"mass_kg": 1.0}, 10
    )
    assert "nominal_trajectory must be" in opt.monte_carlo_uncertainty_analysis(
        {}, {}, 10
    )
