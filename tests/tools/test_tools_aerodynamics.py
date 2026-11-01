from __future__ import annotations

from tests.tools.test_tools_common import without_integrations


def test_aero_import_errors(monkeypatch):
    from aerospace_mcp.tools import aerodynamics as aero_tools

    without_integrations(monkeypatch, "aero")
    assert "not available" in aero_tools.wing_vlm_analysis({}, {}, {}).lower()
    assert "not available" in aero_tools.airfoil_polar_analysis("NACA0012").lower()
    assert "not available" in aero_tools.calculate_stability_derivatives({}, {}).lower()
    assert "not available" in aero_tools.get_airfoil_database().lower()


def test_aero_invalid_input_errors():
    from aerospace_mcp.tools import aerodynamics as aero_tools

    out = aero_tools.airfoil_polar_analysis("NACA0012", 1e6, 0.1, ["x"])
    assert out.startswith("Airfoil analysis error")


def test_get_airfoil_database():
    import json

    from aerospace_mcp.tools.aerodynamics import get_airfoil_database

    data = json.loads(get_airfoil_database())
    assert data["NACA0012"]["cl0"] == 0.0  # symmetric section
    assert data["NACA0012"]["cl_max"] == 1.4
    assert data["NACA2412"]["cl0"] > 0  # cambered section


# ---------------------------------------------------------------------------
# End-to-end tests against the real integration module
# ---------------------------------------------------------------------------


def _use_real_integration(monkeypatch):
    """Force the built-in models (AeroSandbox's VLM gives different numbers)."""
    import aerospace_mcp.integrations.aero as aero

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
