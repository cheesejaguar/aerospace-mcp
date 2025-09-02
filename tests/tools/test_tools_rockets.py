from __future__ import annotations

import sys

from tests.tools.test_tools_common import (
    StubPerf,
    StubPoint,
    make_module,
)


class RocketGeometry:
    def __init__(self, **kwargs):
        self.length_m = kwargs.get("length_m", 2.0)
        self.diameter_m = kwargs.get("diameter_m", 0.3)
        self.dry_mass_kg = kwargs.get("dry_mass_kg", 10.0)
        self.propellant_mass_kg = kwargs.get("propellant_mass_kg", 5.0)


def test_rocket_3dof_trajectory_success():
    class Result:
        performance = StubPerf()
        trajectory = [
            StubPoint(time_s=0, altitude_m=0, velocity_ms=0, acceleration_ms2=0),
            StubPoint(time_s=10, altitude_m=100, velocity_ms=50, acceleration_ms2=5),
        ]

    def _traj(geometry, launch_conditions, sim_opts):
        return Result()

    sys.modules["aerospace_mcp.integrations.rockets"] = make_module(
        RocketGeometry=RocketGeometry, rocket_3dof_trajectory=_traj
    )

    from aerospace_mcp.tools.rockets import rocket_3dof_trajectory

    out = rocket_3dof_trajectory(
        {
            "length_m": 2.0,
            "diameter_m": 0.2,
            "dry_mass_kg": 10,
            "propellant_mass_kg": 5,
        },
        {"launch_angle_deg": 80},
        {},
    )
    assert "Rocket Trajectory" in out or "3DOF" in out


def test_rocket_importerror():
    import sys

    from aerospace_mcp.tools import rockets as r_tools

    sys.modules["aerospace_mcp.integrations.rockets"] = make_module()
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
    sys.modules["aerospace_mcp.integrations.rockets"] = make_module(
        RocketGeometry=RocketGeometry,
        optimize_launch_angle=lambda *a, **k: {"ok": True},
    )
    from aerospace_mcp.tools.rockets import optimize_launch_angle

    out = optimize_launch_angle({"length_m": 2.0}, None, "altitude", (45.0, 90.0))
    assert "ok" in out


def test_rockets_exception_branches():
    import sys

    def _boom(*a, **k):
        raise RuntimeError("boom")

    sys.modules["aerospace_mcp.integrations.rockets"] = make_module(
        RocketGeometry=RocketGeometry,
        rocket_3dof_trajectory=_boom,
        estimate_rocket_sizing=_boom,
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
