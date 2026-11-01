from __future__ import annotations

from tests.tools.test_tools_common import without_integrations


def test_atmosphere_import_errors(monkeypatch):
    from aerospace_mcp.tools.atmosphere import get_atmosphere_profile, wind_model_simple

    without_integrations(monkeypatch, "atmosphere")
    assert "not available" in get_atmosphere_profile([0], "ISA").lower()
    assert "not available" in wind_model_simple([0.0]).lower()


def test_atmosphere_invalid_input_errors():
    from aerospace_mcp.tools.atmosphere import get_atmosphere_profile, wind_model_simple

    assert get_atmosphere_profile([-1e6], "ISA") == (
        "Atmosphere profile error: Altitude out of ISA range (0-86000m)"
    )
    assert get_atmosphere_profile(["x"], "ISA").startswith("Atmosphere profile error")
    assert wind_model_simple([0.0], 10.0, 270.0, "bogus").startswith(
        "Wind model error: Unknown wind model: bogus"
    )


# ---------------------------------------------------------------------------
# End-to-end tests against the real integration module
# ---------------------------------------------------------------------------


def _json_part(out: str):
    import json

    return json.loads(out.split("JSON Data:")[1])


def test_get_atmosphere_profile_end_to_end():
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
