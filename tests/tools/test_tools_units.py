"""Tests for the convert_units tool."""

from __future__ import annotations

import json

import pytest

from aerospace_mcp.tools.units import convert_units


def _result(raw: str) -> float:
    data = json.loads(raw)
    return data["result"]


@pytest.mark.parametrize(
    "value,from_unit,to_unit,expected",
    [
        (100.0, "kts", "mps", 51.4444),
        (35000.0, "ft", "m", 10668.0),
        (1.0, "nm", "km", 1.852),
        (1000.0, "lb", "kg", 453.59237),
        (1.0, "atm", "hpa", 1013.25),
        (29.92, "inhg", "hpa", 1013.208),
        (180.0, "deg", "rad", 3.14159265),
        (1.0, "mach", "kts", 661.478),
    ],
)
def test_conversions(value, from_unit, to_unit, expected):
    assert _result(convert_units(value, from_unit, to_unit)) == pytest.approx(
        expected, rel=1e-4
    )


def test_temperature_affine():
    assert _result(convert_units(0.0, "c", "f")) == pytest.approx(32.0)
    assert _result(convert_units(15.0, "c", "k")) == pytest.approx(288.15)
    assert _result(convert_units(-40.0, "f", "c")) == pytest.approx(-40.0)


def test_case_insensitive_and_whitespace():
    assert _result(convert_units(1.0, " KM ", "m")) == pytest.approx(1000.0)


def test_round_trip():
    there = _result(convert_units(12345.6, "ft", "m"))
    back = _result(convert_units(there, "m", "ft"))
    assert back == pytest.approx(12345.6)


def test_cross_dimension_rejected():
    out = convert_units(1.0, "kg", "ft")
    assert out.startswith("Error")
    assert "mass" in out and "length" in out


def test_unknown_unit_rejected():
    out = convert_units(1.0, "parsec", "m")
    assert out.startswith("Error")
    assert "parsec" in out


def test_dimension_reported():
    data = json.loads(convert_units(250.0, "kts", "kmh"))
    assert data["dimension"] == "speed"
    assert data["result"] == pytest.approx(463.0)
