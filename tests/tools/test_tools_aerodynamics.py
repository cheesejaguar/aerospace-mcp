from __future__ import annotations

import sys

from tests.tools.test_tools_common import make_module


def test_wing_airfoil_and_stability_success():
    sys.modules["aerospace_mcp.integrations.aero"] = make_module(
        wing_vlm_analysis=lambda wing, cond, opts: {
            "aspect_ratio": 10.0,
            "cl": 0.5,
            "cd": 0.02,
            "cl_cd_ratio": 25.0,
            "lift_n": 1000.0,
            "drag_n": 40.0,
        },
        airfoil_polar_analysis=lambda name, Re, M, alphas: {
            "polar_data": [
                {
                    "alpha_deg": a,
                    "cl": 0.1 * a,
                    "cd": 0.01,
                    "cm": -0.05,
                    "cl_cd_ratio": 10,
                }
                for a in alphas
            ]
        },
        calculate_stability_derivatives=lambda wing, cond: {"cm_alpha": -0.5},
        AIRFOIL_DATABASE={"NACA0012": {"thickness": 0.12}},
    )

    from aerospace_mcp.tools.aerodynamics import (
        airfoil_polar_analysis,
        calculate_stability_derivatives,
        get_airfoil_database,
        wing_vlm_analysis,
    )

    assert "Aerodynamic" in wing_vlm_analysis({}, {}, {})
    assert "Polar Analysis" in airfoil_polar_analysis("NACA0012")
    assert "cm_alpha" in calculate_stability_derivatives({}, {})
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

    sys.modules["aerospace_mcp.integrations.aero"] = make_module(
        wing_vlm_analysis=_boom,
        airfoil_polar_analysis=_boom,
        calculate_stability_derivatives=_boom,
        AIRFOIL_DATABASE=Bad(),
    )
    from aerospace_mcp.tools import aerodynamics as aero_tools

    assert "error" in aero_tools.wing_vlm_analysis({}, {}, {}).lower()
    assert "error" in aero_tools.airfoil_polar_analysis("NACA0012").lower()
    assert "error" in aero_tools.calculate_stability_derivatives({}, {}).lower()
    assert "error" in aero_tools.get_airfoil_database().lower()
