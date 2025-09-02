from __future__ import annotations

import sys

from tests.tools.test_tools_common import StubPerf, make_module


class RocketGeometry:
    def __init__(self, **kwargs):
        pass


def test_optimization_success():
    class OptResult:
        optimization_result = "ok"
        optimal_objective = 1.0
        optimal_parameters = {"a": 1}
        performance = StubPerf()

    sys.modules["aerospace_mcp.integrations.trajopt"] = make_module(
        RocketGeometry=RocketGeometry,
        optimize_thrust_profile=lambda *a, **k: OptResult(),
        trajectory_sensitivity_analysis=lambda *a, **k: {"sensitivity": []},
        genetic_algorithm_optimization=lambda *a, **k: {"ga": True},
        particle_swarm_optimization=lambda *a, **k: {"pso": True},
        porkchop_plot_analysis=lambda *a, **k: {"porkchop": True},
        monte_carlo_uncertainty_analysis=lambda *a, **k: {"mc": True},
    )

    from aerospace_mcp.tools.optimization import (
        genetic_algorithm_optimization,
        monte_carlo_uncertainty_analysis,
        optimize_thrust_profile,
        particle_swarm_optimization,
        porkchop_plot_analysis,
        trajectory_sensitivity_analysis,
    )

    assert "optimization_result" in optimize_thrust_profile({}, 10, 1000)
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
        porkchop_plot_analysis=_boom,
        monte_carlo_uncertainty_analysis=_boom,
    )
    from aerospace_mcp.tools import optimization as opt

    assert "error" in opt.optimize_thrust_profile({}, 1.0, 1.0).lower()
    assert "error" in opt.trajectory_sensitivity_analysis({}, {}).lower()
    assert "error" in opt.genetic_algorithm_optimization({}).lower()
    assert "error" in opt.particle_swarm_optimization({}).lower()
    assert "error" in opt.porkchop_plot_analysis("A", "B", [], []).lower()
    assert "error" in opt.monte_carlo_uncertainty_analysis({}, {}, 1).lower()
