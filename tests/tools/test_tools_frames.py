from __future__ import annotations

import sys

from tests.tools.test_tools_common import make_module


def test_transform_frames_success():
    def _transform(coords, from_f, to_f, epoch):
        return {"ok": True, "from": from_f, "to": to_f, "coords": coords}

    sys.modules["aerospace_mcp.integrations.frames"] = make_module(
        transform_frames=_transform
    )
    from aerospace_mcp.tools.frames import transform_frames

    out = transform_frames({"x": 1}, "ECEF", "ECI", None)
    assert "ok" in out


def test_transform_frames_importerror():
    import sys as _sys

    _sys.modules["aerospace_mcp.integrations.frames"] = make_module()
    from aerospace_mcp.tools.frames import transform_frames

    out = transform_frames({}, "ECEF", "ECI", None)
    assert "not available" in out.lower()


def test_geodetic_to_ecef_success():
    sys.modules["aerospace_mcp.integrations.frames"] = make_module(
        geodetic_to_ecef=lambda lat, lon, alt: {"x_m": 1.0, "y_m": 2.0, "z_m": 3.0}
    )
    from aerospace_mcp.tools.frames import geodetic_to_ecef

    out = geodetic_to_ecef(0, 0, 0)
    assert "ECEF" in out


def test_geodetic_to_ecef_importerror():
    import sys as _sys

    _sys.modules["aerospace_mcp.integrations.frames"] = make_module()
    from aerospace_mcp.tools.frames import geodetic_to_ecef

    out = geodetic_to_ecef(0, 0, 0)
    assert "not available" in out.lower()


def test_ecef_to_geodetic_success():
    sys.modules["aerospace_mcp.integrations.frames"] = make_module(
        ecef_to_geodetic=lambda x, y, z: {
            "latitude_deg": 0.0,
            "longitude_deg": 0.0,
            "altitude_m": 0.0,
        }
    )
    from aerospace_mcp.tools.frames import ecef_to_geodetic

    out = ecef_to_geodetic(1, 2, 3)
    assert "Geodetic" in out


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
    import sys as _sys

    _sys.modules["aerospace_mcp.integrations.frames"] = make_module()
    from aerospace_mcp.tools.frames import ecef_to_geodetic

    out = ecef_to_geodetic(0, 0, 0)
    assert "not available" in out.lower()
