from __future__ import annotations

import json
import math

from tests.tools.test_tools_common import without_integrations


def test_frames_import_errors(monkeypatch):
    from aerospace_mcp.tools import frames as tools

    without_integrations(monkeypatch, "frames")
    assert "not available" in tools.transform_frames({}, "ECEF", "ECI", None).lower()
    assert "not available" in tools.geodetic_to_ecef(0, 0, 0).lower()
    assert "not available" in tools.ecef_to_geodetic(0, 0, 0).lower()


def test_frames_invalid_input_errors():
    from aerospace_mcp.tools import frames as tools

    assert tools.geodetic_to_ecef(100, 0, 0) == (
        "Geodetic to ECEF error: Latitude must be between -90 and +90 degrees"
    )
    assert tools.geodetic_to_ecef("a", 0, 0).startswith("Geodetic to ECEF error")
    assert tools.ecef_to_geodetic("a", 0, 0).startswith("ECEF to geodetic error")


# ---------------------------------------------------------------------------
# End-to-end tests against the real integration module.  Earlier stubbed
# tests returned dicts where the integration returns pydantic models, which
# is how every frames tool shipped broken.
# ---------------------------------------------------------------------------


def _real_frames_tools():
    """Import the tool module (it imports integrations.frames on each call)."""
    from aerospace_mcp.tools import frames as tools

    return tools


def test_geodetic_to_ecef_end_to_end():
    """WGS-84 known answer: N = a / sqrt(1 - e^2 sin^2(lat)) (NIMA TR8350.2)."""
    tools = _real_frames_tools()
    out = json.loads(tools.geodetic_to_ecef(37.0, -122.0, 100.0))["output"]
    assert abs(out["x_m"] - -2702626.9216) < 1e-3
    assert abs(out["y_m"] - -4325107.1818) < 1e-3
    assert abs(out["z_m"] - 3817453.3419) < 1e-3


def test_ecef_to_geodetic_end_to_end():
    tools = _real_frames_tools()
    out = json.loads(tools.ecef_to_geodetic(-2702626.9216, -4325107.1818, 3817453.3419))
    geo = out["output"]
    assert abs(geo["latitude_deg"] - 37.0) < 1e-8
    assert abs(geo["longitude_deg"] - -122.0) < 1e-8
    assert abs(geo["altitude_m"] - 100.0) < 1e-3

    # North pole on the ellipsoid: lat 90, h 0 (b = 6356752.314245 m)
    pole = json.loads(tools.ecef_to_geodetic(0.0, 0.0, 6356752.314245))["output"]
    assert pole["latitude_deg"] == 90.0
    assert abs(pole["altitude_m"]) < 1e-6


def test_transform_frames_end_to_end_dict_and_list_inputs(monkeypatch):
    tools = _real_frames_tools()
    import aerospace_mcp.integrations.frames as frames_integration

    # The ECI known answer below is the manual GMST + precession path; with
    # astropy installed the full IAU model (nutation, polar motion) differs
    # by ~200 m, so pin the manual path.
    monkeypatch.setattr(frames_integration, "ASTROPY_AVAILABLE", False)

    geo = json.loads(
        tools.transform_frames(
            {"x": 6378137.0, "y": 0.0, "z": 0.0}, "ECEF", "GEODETIC", None
        )
    )
    assert geo["output"] == {
        "latitude_deg": 0.0,
        "longitude_deg": 0.0,
        "altitude_m": 0.0,
    }

    ecef = json.loads(
        tools.transform_frames(
            {"latitude_deg": 37.0, "longitude_deg": -122.0, "altitude_m": 100.0},
            "GEODETIC",
            "ITRF",
        )
    )
    assert abs(ecef["output"]["x"] - -2702626.9216) < 1e-3
    assert ecef["epoch_utc"] is None  # Earth-fixed only: epoch irrelevant

    # List input and the default (None) epoch for a time-dependent transform:
    # at J2000.0 precession is the identity and GMST = 280.46061837 deg.
    eci = json.loads(tools.transform_frames([7e6, 0.0, 0.0], "ECI", "ECEF"))
    gmst = math.radians(280.46061837)
    assert eci["epoch_utc"] == "2000-01-01T12:00:00"
    assert abs(eci["output"]["x"] - 7e6 * math.cos(gmst)) < 1e-6
    assert abs(eci["output"]["y"] - -7e6 * math.sin(gmst)) < 1e-6


def test_transform_frames_end_to_end_bad_input_reports_error():
    tools = _real_frames_tools()
    out = tools.transform_frames({"a": 1}, "ECEF", "GEODETIC")
    assert out.startswith("Frame transformation error")
    assert "'x', 'y', 'z'" in out
