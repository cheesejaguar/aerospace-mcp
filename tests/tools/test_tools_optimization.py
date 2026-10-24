from __future__ import annotations

import sys

from tests.tools.test_tools_common import StubPerf, make_module


class RocketGeometry:
    def __init__(self, **kwargs):
        pass


def test_optimization_success():
    class OptResult:
        iterations = 3
        converged = True
        optimal_objective = 1.0
        optimal_parameters = {"a": 1}
        performance = StubPerf()

    sys.modules["aerospace_mcp.integrations.trajopt"] = make_module(
        RocketGeometry=RocketGeometry,
        optimize_thrust_profile=lambda *a, **k: OptResult(),
        trajectory_sensitivity_analysis=lambda *a, **k: {"sensitivity": []},
        genetic_algorithm_optimization=lambda *a, **k: {"ga": True},
        particle_swarm_optimization=lambda *a, **k: {"pso": True},
        monte_carlo_uncertainty_analysis=lambda *a, **k: {"mc": True},
    )
    sys.modules["aerospace_mcp.integrations.orbits"] = make_module(
        porkchop_plot_analysis=lambda *a, **k: {"porkchop": True},
    )

    from aerospace_mcp.tools.optimization import (
        genetic_algorithm_optimization,
        monte_carlo_uncertainty_analysis,
        optimize_thrust_profile,
        particle_swarm_optimization,
        porkchop_plot_analysis,
        trajectory_sensitivity_analysis,
    )

    out = optimize_thrust_profile({}, 10, 1000)
    assert '"converged": true' in out and "optimal_objective" in out
    assert "sensitivity" in trajectory_sensitivity_analysis({}, {})
    assert "ga" in genetic_algorithm_optimization({})
    assert "pso" in particle_swarm_optimization({})
    assert "porkchop" in porkchop_plot_analysis(
        "Earth", "Mars", ["2025-01-01"], ["2025-12-01"]
    )
    assert "mc" in monte_carlo_uncertainty_analysis({}, {}, 10)


def test_optimization_import_errors():
    import sys

    from aerospace_mcp.tools import optimization as opt_tools

    sys.modules["aerospace_mcp.integrations.trajopt"] = make_module()
    sys.modules["aerospace_mcp.integrations.orbits"] = make_module()
    assert "not available" in opt_tools.optimize_thrust_profile({}, 1.0, 1.0).lower()
    assert "not available" in opt_tools.trajectory_sensitivity_analysis({}, {}).lower()
    assert "not available" in opt_tools.genetic_algorithm_optimization({}).lower()
    assert "not available" in opt_tools.particle_swarm_optimization({}).lower()
    assert "not available" in opt_tools.porkchop_plot_analysis("A", "B", [], []).lower()
    assert (
        "not available" in opt_tools.monte_carlo_uncertainty_analysis({}, {}, 1).lower()
    )


def test_optimization_exception_branches():
    import sys

    def _boom(*a, **k):
        raise RuntimeError("boom")

    sys.modules["aerospace_mcp.integrations.trajopt"] = make_module(
        RocketGeometry=type("R", (), {}),
        optimize_thrust_profile=_boom,
        trajectory_sensitivity_analysis=_boom,
        genetic_algorithm_optimization=_boom,
        particle_swarm_optimization=_boom,
        monte_carlo_uncertainty_analysis=_boom,
    )
    sys.modules["aerospace_mcp.integrations.orbits"] = make_module(
        porkchop_plot_analysis=_boom,
    )
    from aerospace_mcp.tools import optimization as opt

    assert "error" in opt.optimize_thrust_profile({}, 1.0, 1.0).lower()
    assert "error" in opt.trajectory_sensitivity_analysis({}, {}).lower()
    assert "error" in opt.genetic_algorithm_optimization({}).lower()
    assert "error" in opt.particle_swarm_optimization({}).lower()
    assert "error" in opt.porkchop_plot_analysis("A", "B", [], []).lower()
    assert "error" in opt.monte_carlo_uncertainty_analysis({}, {}, 1).lower()


def test_sensitivity_objective_is_read_from_options():
    calls = {}

    def _sensitivity(geometry, variations, objective="max_altitude"):
        calls["objective"] = objective
        return {"ok": True}

    sys.modules["aerospace_mcp.integrations.trajopt"] = make_module(
        RocketGeometry=RocketGeometry,
        trajectory_sensitivity_analysis=_sensitivity,
    )
    from aerospace_mcp.tools.optimization import trajectory_sensitivity_analysis

    trajectory_sensitivity_analysis({}, {"cd": [0.3]}, {"objective": "max_velocity"})
    assert calls["objective"] == "max_velocity"


def test_thrust_profile_and_porkchop_end_to_end():
    """Real native-backed run through the tool layer (no stubs)."""
    import json

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
    assert len(out["optimal_parameters"]["thrust_curve"]) == 4

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
