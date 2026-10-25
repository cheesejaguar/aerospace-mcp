from __future__ import annotations

import sys

from tests.tools.test_tools_common import StubModel, make_module


def test_wing_airfoil_and_stability_success():
    # Create mock WingGeometry class
    class MockWingGeometry:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    # Mock return types matching integration functions
    def mock_wing_vlm_analysis(geometry, alpha_deg_list, mach=0.2, reynolds=None):
        return [
            StubModel(
                alpha_deg=a,
                CL=0.5,
                CD=0.02,
                CM=-0.05,
                L_D_ratio=25.0,
                span_efficiency=0.85,
            )
            for a in alpha_deg_list
        ]

    def mock_airfoil_polar_analysis(name, alpha_deg_list, reynolds=1e6, mach=0.1):
        return [
            StubModel(alpha_deg=a, cl=0.1 * a, cd=0.01, cm=-0.05, cl_cd_ratio=10.0)
            for a in alpha_deg_list
        ]

    def mock_calculate_stability(geometry, alpha_deg=2.0, mach=0.2):
        return StubModel(
            CL_alpha=4.5, CM_alpha=-0.5, CL_alpha_dot=None, CM_alpha_dot=None
        )

    sys.modules["aerospace_mcp.integrations.aero"] = make_module(
        WingGeometry=MockWingGeometry,
        wing_vlm_analysis=mock_wing_vlm_analysis,
        airfoil_polar_analysis=mock_airfoil_polar_analysis,
        calculate_stability_derivatives=mock_calculate_stability,
        AIRFOIL_DATABASE={"NACA0012": {"thickness": 0.12}},
    )

    from aerospace_mcp.tools.aerodynamics import (
        airfoil_polar_analysis,
        calculate_stability_derivatives,
        get_airfoil_database,
        wing_vlm_analysis,
    )

    assert "Aerodynamic" in wing_vlm_analysis({}, {"alpha_deg_list": [0.0, 2.0]}, {})
    assert "Polar Analysis" in airfoil_polar_analysis("NACA0012")
    assert "CL_alpha" in calculate_stability_derivatives({}, {})
    assert "NACA0012" in get_airfoil_database()


def test_aero_import_errors():
    import sys

    from aerospace_mcp.tools import aerodynamics as aero_tools

    sys.modules["aerospace_mcp.integrations.aero"] = make_module()
    assert "not available" in aero_tools.wing_vlm_analysis({}, {}, {}).lower()
    assert "not available" in aero_tools.airfoil_polar_analysis("NACA0012").lower()
    assert "not available" in aero_tools.calculate_stability_derivatives({}, {}).lower()
    assert "not available" in aero_tools.get_airfoil_database().lower()


def test_aero_exception_branches():
    import sys

    def _boom(*a, **k):
        raise RuntimeError("boom")

    class Bad:
        pass

    # Need to include WingGeometry to avoid ImportError
    class MockWingGeometry:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    sys.modules["aerospace_mcp.integrations.aero"] = make_module(
        WingGeometry=MockWingGeometry,
        wing_vlm_analysis=_boom,
        airfoil_polar_analysis=_boom,
        calculate_stability_derivatives=_boom,
        AIRFOIL_DATABASE=Bad(),
    )
    from aerospace_mcp.tools import aerodynamics as aero_tools

    assert (
        "error"
        in aero_tools.wing_vlm_analysis({}, {"alpha_deg_list": [0.0]}, {}).lower()
    )
    assert "error" in aero_tools.airfoil_polar_analysis("NACA0012").lower()
    assert "error" in aero_tools.calculate_stability_derivatives({}, {}).lower()
    assert "error" in aero_tools.get_airfoil_database().lower()


# ---------------------------------------------------------------------------
# End-to-end tests against the real integration module (no stubs)
# ---------------------------------------------------------------------------


def _use_real_integration(monkeypatch):
    """Drop a stub left in sys.modules and force the built-in models."""
    name = "aerospace_mcp.integrations.aero"
    if getattr(sys.modules.get(name), "__file__", None) is None:
        sys.modules.pop(name, None)
    import importlib

    aero = importlib.import_module(name)
    monkeypatch.setattr(aero, "AEROSANDBOX_AVAILABLE", False)
    return aero


def _json_part(out: str):
    import json

    return json.loads(out.split("JSON Data:")[1])


def test_wing_vlm_analysis_end_to_end(monkeypatch):
    import math

    _use_real_integration(monkeypatch)
    from aerospace_mcp.tools.aerodynamics import wing_vlm_analysis

    cfg = {"span_m": 8.0, "chord_root_m": 1.0, "airfoil_root": "naca0012"}
    out = wing_vlm_analysis(cfg, {"alpha_deg_list": [0.0, 4.0], "mach": 0.0})
    assert "Method: lifting_line_datcom" in out
    data = _json_part(out)
    cl_alpha = (data[1]["CL"] - data[0]["CL"]) / math.radians(4)
    assert abs(cl_alpha - 4.91) < 0.01  # Helmbold-Diederich, AR 8
    assert data[0]["method"] == "lifting_line_datcom"

    # Moment reference is honoured: dCM/dCL = x_ref - 0.25
    data = _json_part(
        wing_vlm_analysis(
            cfg, {"alpha_deg_list": [0.0, 4.0], "mach": 0.2, "x_ref_mac": 0.1}
        )
    )
    slope = (data[1]["CM"] - data[0]["CM"]) / (data[1]["CL"] - data[0]["CL"])
    assert abs(slope + 0.15) < 1e-9

    assert "subsonic" in wing_vlm_analysis(cfg, {"alpha_deg_list": [0.0], "mach": 1.2})
    assert "Unknown airfoil" in wing_vlm_analysis(
        {"span_m": 8.0, "chord_root_m": 1.0, "airfoil_root": "bogus"},
        {"alpha_deg_list": [0.0]},
    )


def test_airfoil_polar_analysis_end_to_end(monkeypatch):
    _use_real_integration(monkeypatch)
    from aerospace_mcp.tools.aerodynamics import airfoil_polar_analysis

    data = _json_part(airfoil_polar_analysis("NACA0012", 1e6, 0.0, [0, 15, 25, 90]))
    assert max(p["cl"] for p in data) <= 1.4 + 1e-9  # capped at cl_max
    assert data[2]["cl_cd_ratio"] < 5  # post-stall
    assert abs(data[3]["cd"] - 2.006) < 1e-6  # flat plate at 90 deg
    assert data[3]["warnings"]
    assert "Reynolds number must be positive" in airfoil_polar_analysis(
        "NACA0012", 0, 0.1, [0, 5]
    )


def test_calculate_stability_derivatives_end_to_end(monkeypatch):
    import json

    _use_real_integration(monkeypatch)
    from aerospace_mcp.tools.aerodynamics import calculate_stability_derivatives

    cfg = {"span_m": 8.0, "chord_root_m": 1.0, "airfoil_root": "NACA0012"}
    low = json.loads(
        calculate_stability_derivatives(cfg, {"alpha_deg": 4, "mach": 0.0})
    )
    high = json.loads(
        calculate_stability_derivatives(cfg, {"alpha_deg": 14, "mach": 0.0})
    )
    assert high["CL_alpha"] < low["CL_alpha"]  # alpha used to be ignored
    assert low["CM_alpha"] == 0.0  # about the MAC quarter chord (default)
    assert low["CL_alpha_dot"] is None
    assert any("not computed" in n for n in low["notes"])

    fwd = json.loads(
        calculate_stability_derivatives(
            cfg, {"alpha_deg": 4, "mach": 0.0, "x_ref_mac": 0.15}
        )
    )
    assert fwd["CM_alpha"] < 0
    assert fwd["x_ref_mac"] == 0.15
