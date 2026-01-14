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
