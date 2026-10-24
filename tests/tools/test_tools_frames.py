from __future__ import annotations

import json
import math
import sys

from tests.tools.test_tools_common import StubModel, make_module


def test_transform_frames_success():
    # The integration returns a CoordinatePoint-like object (attributes,
    # not dict keys); the stub mirrors that.
    def _transform(coords, from_f, to_f, epoch):
        return StubModel(x=1.0, y=2.0, z=3.0, frame=to_f, epoch=epoch or "J2000")

    sys.modules["aerospace_mcp.integrations.frames"] = make_module(
        transform_frames=_transform
    )
    from aerospace_mcp.tools.frames import transform_frames

    out = json.loads(transform_frames({"x": 1}, "ECEF", "ECI", None))
    assert out["output"] == {"x": 1.0, "y": 2.0, "z": 3.0}
    assert out["epoch_utc"] == "J2000"


def test_transform_frames_importerror():
    sys.modules["aerospace_mcp.integrations.frames"] = make_module()
    from aerospace_mcp.tools.frames import transform_frames

    out = transform_frames({}, "ECEF", "ECI", None)
    assert "not available" in out.lower()


def test_geodetic_to_ecef_success():
    sys.modules["aerospace_mcp.integrations.frames"] = make_module(
        geodetic_to_ecef=lambda lat, lon, alt: StubModel(x=1.0, y=2.0, z=3.0)
    )
    from aerospace_mcp.tools.frames import geodetic_to_ecef

    out = json.loads(geodetic_to_ecef(0, 0, 0))
    assert out["output"] == {"x_m": 1.0, "y_m": 2.0, "z_m": 3.0}


def test_geodetic_to_ecef_importerror():
    sys.modules["aerospace_mcp.integrations.frames"] = make_module()
    from aerospace_mcp.tools.frames import geodetic_to_ecef

    out = geodetic_to_ecef(0, 0, 0)
    assert "not available" in out.lower()


def test_ecef_to_geodetic_success():
    sys.modules["aerospace_mcp.integrations.frames"] = make_module(
        ecef_to_geodetic=lambda x, y, z: StubModel(
            latitude_deg=0.0, longitude_deg=0.0, altitude_m=0.0
        )
    )
    from aerospace_mcp.tools.frames import ecef_to_geodetic

    out = json.loads(ecef_to_geodetic(1, 2, 3))
    assert out["reference_frame"] == "WGS84 Geodetic"
    assert out["output"]["altitude_m"] == 0.0


def test_frames_exception_branches():
    import sys as _sys

    def _boom(*a, **k):
        raise RuntimeError("boom")

    _sys.modules["aerospace_mcp.integrations.frames"] = make_module(
        transform_frames=_boom, geodetic_to_ecef=_boom, ecef_to_geodetic=_boom
    )
    from aerospace_mcp.tools.frames import (
        ecef_to_geodetic as _e2g,
    )
    from aerospace_mcp.tools.frames import (
        geodetic_to_ecef as _g2e,
    )
    from aerospace_mcp.tools.frames import (
        transform_frames as _tf,
    )

    assert "error" in _tf({}, "ECEF", "ECI", None).lower()
    assert "error" in _g2e(0, 0, 0).lower()
    assert "error" in _e2g(0, 0, 0).lower()


def test_ecef_to_geodetic_importerror():
    sys.modules["aerospace_mcp.integrations.frames"] = make_module()
    from aerospace_mcp.tools.frames import ecef_to_geodetic

    out = ecef_to_geodetic(0, 0, 0)
    assert "not available" in out.lower()


# ---------------------------------------------------------------------------
# End-to-end tests against the real integration module (no stubs).  The
# stubbed tests above returned dicts where the integration returns pydantic
# models, which is how every frames tool shipped broken.
# ---------------------------------------------------------------------------


def _real_frames_tools():
    """Import the tool module with the real integrations.frames module."""
    sys.modules.pop("aerospace_mcp.integrations.frames", None)
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


def test_transform_frames_end_to_end_dict_and_list_inputs():
    tools = _real_frames_tools()

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
