from __future__ import annotations

from tests.tools.test_tools_common import make_module


class RocketGeometry:
    def __init__(self, **kwargs):
        self.length_m = kwargs.get("length_m", 2.0)
        self.diameter_m = kwargs.get("diameter_m", 0.3)
        self.dry_mass_kg = kwargs.get("dry_mass_kg", 10.0)
        self.propellant_mass_kg = kwargs.get("propellant_mass_kg", 5.0)


GEOMETRY = {
    "length_m": 2.0,
    "diameter_m": 0.2,
    "dry_mass_kg": 10,
    "propellant_mass_kg": 5,
    "thrust_curve": [[0.0, 800.0], [3.0, 800.0], [3.1, 0.0]],
}


def test_rocket_3dof_trajectory_success():
    """Runs the real (native) simulation through the tool layer."""
    from aerospace_mcp.tools.rockets import rocket_3dof_trajectory

    out = rocket_3dof_trajectory(
        GEOMETRY, {"launch_angle_deg": 80}, {"dt_s": 0.05, "max_time_s": 120}
    )
    assert "3DOF Rocket Trajectory Analysis" in out
    assert "Launch Angle: 80.0" in out
    assert "Key Trajectory Points" in out
    assert '"max_altitude_m"' in out


def test_rocket_3dof_trajectory_invalid_options():
    from aerospace_mcp.tools.rockets import rocket_3dof_trajectory

    out = rocket_3dof_trajectory(GEOMETRY, {}, {"dt_s": 0.0})
    assert "error" in out.lower() and "dt_s" in out


def test_rocket_importerror():
    import sys

    from aerospace_mcp.tools import rockets as r_tools

    sys.modules["aerospace_mcp.integrations.rockets"] = make_module()
    sys.modules["aerospace_mcp.integrations.trajopt"] = make_module()
    assert "not available" in r_tools.rocket_3dof_trajectory({}, {}).lower()
    assert "not available" in r_tools.estimate_rocket_sizing(1000, 5).lower()
    assert (
        "not available" in r_tools.optimize_launch_angle({}, None, "altitude").lower()
    )


def test_estimate_rocket_sizing_success():
    import sys

    sys.modules["aerospace_mcp.integrations.rockets"] = make_module(
        estimate_rocket_sizing=lambda *a, **k: {"ok": True}
    )
    from aerospace_mcp.tools.rockets import estimate_rocket_sizing

    out = estimate_rocket_sizing(1000, 5)
    assert "ok" in out


def test_optimize_launch_angle_success():
    """Runs the real (native) optimizer through the tool layer."""
    import json

    from aerospace_mcp.tools.rockets import optimize_launch_angle

    out = json.loads(optimize_launch_angle(GEOMETRY, None, "altitude", (80.0, 90.0)))
    assert 80.0 <= out["optimal_launch_angle_deg"] <= 90.0
    assert out["optimal_objective"] == out["performance"]["max_altitude_m"]
    assert "note" not in out

    ranged = json.loads(optimize_launch_angle(GEOMETRY, None, "range", (80.0, 90.0)))
    assert "not modelled" in ranged["note"]


def test_rockets_exception_branches():
    import sys

    def _boom(*a, **k):
        raise RuntimeError("boom")

    sys.modules["aerospace_mcp.integrations.rockets"] = make_module(
        RocketGeometry=RocketGeometry,
        analyze_rocket_performance=_boom,
        rocket_3dof_trajectory=_boom,
        estimate_rocket_sizing=_boom,
    )
    sys.modules["aerospace_mcp.integrations.trajopt"] = make_module(
        RocketGeometry=RocketGeometry,
        optimize_launch_angle=_boom,
    )
    from aerospace_mcp.tools.rockets import (
        estimate_rocket_sizing as _size,
    )
    from aerospace_mcp.tools.rockets import (
        optimize_launch_angle as _opt,
    )
    from aerospace_mcp.tools.rockets import (
        rocket_3dof_trajectory as _traj,
    )

    assert "error" in _traj({}, {}).lower()
    assert "error" in _size(1000, 5).lower()
    assert "error" in _opt({}, None, "altitude").lower()
