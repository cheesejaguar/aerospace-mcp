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
                wind_speed_ms=5.0 + i,
                wind_direction_deg=270.0,
                gust_factor=1.0,
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
