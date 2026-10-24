from __future__ import annotations

import sys

from tests.tools.test_tools_common import StubModel, make_module


def test_get_atmosphere_profile_success(monkeypatch):
    def _get_profile(alts, model):
        return [
            StubModel(
                altitude_m=a,
                pressure_pa=101325 - a,
                temperature_k=288.15 - 0.0065 * a,
                density_kg_m3=1.225,
                speed_of_sound_mps=340.0,
            )
            for a in alts
        ]

    mod = make_module(get_atmosphere_profile=_get_profile)
    sys.modules["aerospace_mcp.integrations.atmosphere"] = mod

    from aerospace_mcp.tools.atmosphere import get_atmosphere_profile

    out = get_atmosphere_profile([0, 1000], "ISA")
    assert "Atmospheric Profile" in out
    assert "JSON Data:" in out


def test_get_atmosphere_profile_importerror(monkeypatch):
    sys.modules["aerospace_mcp.integrations.atmosphere"] = make_module()
    from aerospace_mcp.tools.atmosphere import get_atmosphere_profile

    out = get_atmosphere_profile([0], "ISA")
    assert "not available" in out.lower()


def test_wind_model_simple_success():
    def _wind_model(*args, **kwargs):
        return [
            StubModel(
                altitude_m=a,
                wind_speed_mps=5.0 + i,  # Correct attribute name
                wind_direction_deg=270.0,
            )
            for i, a in enumerate([0.0, 50.0])
        ]

    sys.modules["aerospace_mcp.integrations.atmosphere"] = make_module(
        wind_model_simple=_wind_model
    )

    from aerospace_mcp.tools.atmosphere import wind_model_simple

    out = wind_model_simple([0.0, 50.0])
    assert "Wind Profile" in out


def test_atmosphere_exception_handling(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("boom")

    sys.modules["aerospace_mcp.integrations.atmosphere"] = make_module(
        get_atmosphere_profile=_boom, wind_model_simple=_boom
    )
    from aerospace_mcp.tools.atmosphere import (
        get_atmosphere_profile as _gp,
    )
    from aerospace_mcp.tools.atmosphere import (
        wind_model_simple as _wm,
    )

    assert "error" in _gp([0]).lower()
    assert "error" in _wm([0.0]).lower()


def test_wind_model_simple_importerror():
    sys.modules["aerospace_mcp.integrations.atmosphere"] = make_module()
    from aerospace_mcp.tools.atmosphere import wind_model_simple

    out = wind_model_simple([0.0])
    assert "not available" in out.lower()


# ---------------------------------------------------------------------------
# End-to-end tests against the real integration module (no stubs)
# ---------------------------------------------------------------------------


def _use_real_integration():
    """Drop a stub left in sys.modules so the real module is imported."""
    name = "aerospace_mcp.integrations.atmosphere"
    if getattr(sys.modules.get(name), "__file__", None) is None:
        sys.modules.pop(name, None)


def _json_part(out: str):
    import json

    return json.loads(out.split("JSON Data:")[1])


def test_get_atmosphere_profile_end_to_end():
    _use_real_integration()
    from aerospace_mcp.tools.atmosphere import get_atmosphere_profile

    data = _json_part(get_atmosphere_profile([0.0, 80000.0], "COESA"))
    # US Standard Atmosphere 1976 at 80 km geometric: 198.64 K, 1.0525 Pa
    assert abs(data[1]["temperature_k"] - 198.639) < 0.01
    assert abs(data[1]["pressure_pa"] / 1.0525 - 1) < 1e-3
    assert data[0]["viscosity_pa_s"] is not None

    data = _json_part(get_atmosphere_profile([0.0, 1000.0], "ISA"))
    assert abs(data[0]["pressure_pa"] - 101325.0) < 1.0

    # 'enhanced' was offered but never implemented; the error is explicit.
    assert "Unknown model type" in get_atmosphere_profile([0.0], "enhanced")


def test_wind_model_simple_end_to_end():
    _use_real_integration()
    from aerospace_mcp.tools.atmosphere import wind_model_simple

    # power_law used to always fail ("Unknown wind model: power_law")
    data = _json_part(wind_model_simple([5.0, 10.0, 100.0], 10.0, 270.0, "power_law"))
    assert abs(data[1]["wind_speed_mps"] - 10.0) < 1e-9
    assert abs(data[2]["wind_speed_mps"] - 10.0 * 10.0**0.143) < 1e-9
    assert data[0]["wind_direction_deg"] == 270.0

    # Log law below the reference height: 5.00 and 8.49 m/s (z0 = 0.1 m)
    data = _json_part(wind_model_simple([1.0, 5.0], 10.0, 270.0, "logarithmic", 0.1))
    assert abs(data[0]["wind_speed_mps"] - 5.0) < 1e-9
    assert abs(data[1]["wind_speed_mps"] - 8.4949) < 1e-3

    # z0 >= z_ref is rejected instead of returning inf / 0
    assert "error" in wind_model_simple([5.0, 50.0], 10.0, 270.0, "logarithmic", 20.0)
