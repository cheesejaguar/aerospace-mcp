from __future__ import annotations

import pytest

from tests.tools.test_tools_common import without_integrations

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


def test_rocket_importerror(monkeypatch):
    from aerospace_mcp.tools import rockets as r_tools

    without_integrations(monkeypatch, "rockets", "trajopt")
    assert "not available" in r_tools.rocket_3dof_trajectory({}, {}).lower()
    assert "not available" in r_tools.estimate_rocket_sizing(1000, 5).lower()
    assert (
        "not available" in r_tools.optimize_launch_angle({}, None, "altitude").lower()
    )


def test_estimate_rocket_sizing_success():
    """Tsiolkovsky: the mass ratio is exp(delta_v / (Isp g0))."""
    import json
    import math

    from aerospace_mcp.tools.rockets import estimate_rocket_sizing

    out = json.loads(estimate_rocket_sizing(1000, 5))
    assert out["payload_mass_kg"] == 5
    assert out["specific_impulse_s"] == 250.0  # solid propellant default
    assert out["delta_v_ms"] == pytest.approx(1.2 * out["delta_v_without_margin_ms"])
    assert out["mass_ratio"] == pytest.approx(
        math.exp(out["delta_v_ms"] / (250.0 * 9.80665)), rel=1e-6
    )
    assert out["total_mass_kg"] == pytest.approx(
        5 + out["propellant_mass_kg"] + out["structure_mass_kg"]
    )
    assert out["feasible"] is True


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


def test_rockets_invalid_input_errors():
    from aerospace_mcp.tools.rockets import (
        estimate_rocket_sizing,
        optimize_launch_angle,
        rocket_3dof_trajectory,
    )

    out = rocket_3dof_trajectory({}, {})
    assert out.startswith("Rocket trajectory error") and "dry_mass_kg" in out
    assert estimate_rocket_sizing(-1000, 5) == (
        "Rocket sizing error: target_altitude_m must be > 0"
    )
    out = optimize_launch_angle({}, None, "altitude")
    assert out.startswith("Launch optimization error") and "dry_mass_kg" in out
