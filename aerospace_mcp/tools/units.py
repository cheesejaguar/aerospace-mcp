"""General-purpose aerospace unit conversion tool.

Supports the unit families that come up constantly in aerospace work:
length, speed, mass, pressure, temperature, and angle.  Conversions go
through a canonical SI base unit per dimension; temperature is handled
separately because its conversions are affine (offset + scale), not
purely multiplicative.

WARNING: For educational and research purposes only.
"""

import json
import math

# Multiplicative factor to the dimension's base unit (metres, m/s, kg, Pa,
# radians).  Unit keys are lowercase; lookup is case-insensitive.
_LENGTH_TO_M = {
    "m": 1.0,
    "km": 1000.0,
    "ft": 0.3048,
    "nm": 1852.0,  # nautical mile
    "mi": 1609.344,  # statute mile
    "in": 0.0254,
    "cm": 0.01,
    "mm": 0.001,
}

_SPEED_TO_MPS = {
    "mps": 1.0,  # metres per second
    "m/s": 1.0,
    "kmh": 1000.0 / 3600.0,
    "km/h": 1000.0 / 3600.0,
    "kts": 1852.0 / 3600.0,  # knots
    "kt": 1852.0 / 3600.0,
    "mph": 1609.344 / 3600.0,
    "fps": 0.3048,  # feet per second
    "ft/s": 0.3048,
    "fpm": 0.3048 / 60.0,  # feet per minute (vertical speed)
    "mach": 340.294,  # ISA sea-level speed of sound
}

_MASS_TO_KG = {
    "kg": 1.0,
    "g": 0.001,
    "lb": 0.45359237,
    "lbs": 0.45359237,
    "t": 1000.0,  # metric tonne
    "tonne": 1000.0,
    "slug": 14.59390,
    "oz": 0.028349523125,
}

_PRESSURE_TO_PA = {
    "pa": 1.0,
    "hpa": 100.0,
    "kpa": 1000.0,
    "mbar": 100.0,
    "bar": 100000.0,
    "inhg": 3386.389,  # inches of mercury (altimeter settings)
    "mmhg": 133.3224,
    "psi": 6894.757,
    "psf": 47.88026,  # pounds per square foot (dynamic pressure)
    "atm": 101325.0,
}

_ANGLE_TO_RAD = {
    "rad": 1.0,
    "deg": math.pi / 180.0,
    "grad": math.pi / 200.0,
    "arcmin": math.pi / (180.0 * 60.0),
    "arcsec": math.pi / (180.0 * 3600.0),
}

_TEMPERATURE_UNITS = ("k", "c", "f", "r")  # Kelvin, Celsius, Fahrenheit, Rankine

_DIMENSIONS: dict[str, dict[str, float]] = {
    "length": _LENGTH_TO_M,
    "speed": _SPEED_TO_MPS,
    "mass": _MASS_TO_KG,
    "pressure": _PRESSURE_TO_PA,
    "angle": _ANGLE_TO_RAD,
}


def _find_dimension(unit: str) -> str | None:
    """Return the dimension name a unit belongs to, or None if unknown."""
    if unit in _TEMPERATURE_UNITS:
        return "temperature"
    for dim, table in _DIMENSIONS.items():
        if unit in table:
            return dim
    return None


def _to_kelvin(value: float, unit: str) -> float:
    if unit == "k":
        return value
    if unit == "c":
        return value + 273.15
    if unit == "f":
        return (value - 32.0) * 5.0 / 9.0 + 273.15
    # Rankine
    return value * 5.0 / 9.0


def _from_kelvin(value_k: float, unit: str) -> float:
    if unit == "k":
        return value_k
    if unit == "c":
        return value_k - 273.15
    if unit == "f":
        return (value_k - 273.15) * 9.0 / 5.0 + 32.0
    # Rankine
    return value_k * 9.0 / 5.0


def convert_units(value: float, from_unit: str, to_unit: str) -> str:
    """Convert a value between aerospace units.

    Supported dimensions and units:
        - length: m, km, ft, nm (nautical mile), mi, in, cm, mm
        - speed: mps (m/s), kmh, kts (knots), mph, fps, fpm, mach (ISA SL)
        - mass: kg, g, lb, t (tonne), slug, oz
        - pressure: pa, hpa, kpa, mbar, bar, inhg, mmhg, psi, psf, atm
        - temperature: k, c, f, r (Kelvin, Celsius, Fahrenheit, Rankine)
        - angle: deg, rad, grad, arcmin, arcsec

    Cross-dimension conversions (e.g. kg to ft) are rejected with a clear
    error message.

    Args:
        value: Numeric value to convert.
        from_unit: Source unit symbol (case-insensitive, e.g. "kts").
        to_unit: Target unit symbol (case-insensitive, e.g. "mps").

    Returns:
        JSON string with the converted value, both units, and the dimension,
        or an error message string for unknown/incompatible units.
    """
    try:
        src = from_unit.strip().lower()
        dst = to_unit.strip().lower()

        src_dim = _find_dimension(src)
        dst_dim = _find_dimension(dst)

        if src_dim is None:
            return (
                f"Error: unknown unit '{from_unit}'. Supported dimensions: "
                f"{', '.join([*list(_DIMENSIONS), 'temperature'])}"
            )
        if dst_dim is None:
            return (
                f"Error: unknown unit '{to_unit}'. Supported dimensions: "
                f"{', '.join([*list(_DIMENSIONS), 'temperature'])}"
            )
        if src_dim != dst_dim:
            return (
                f"Error: cannot convert {src_dim} ('{from_unit}') to "
                f"{dst_dim} ('{to_unit}')"
            )

        if src_dim == "temperature":
            result = _from_kelvin(_to_kelvin(float(value), src), dst)
        else:
            table = _DIMENSIONS[src_dim]
            result = float(value) * table[src] / table[dst]

        return json.dumps(
            {
                "value": value,
                "from_unit": src,
                "to_unit": dst,
                "dimension": src_dim,
                "result": result,
            },
            indent=2,
        )
    except Exception as e:
        return f"Unit conversion error: {e}"
