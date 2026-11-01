from __future__ import annotations

import json

from aerospace_mcp.tools import propellers as prop_tools
from tests.tools.test_tools_common import without_integrations

BATTERY = {"capacity_ah": 2.5, "voltage_nominal_v": 14.8, "mass_kg": 0.3}
QUAD = {"mass_kg": 1.5, "disk_area_m2": 0.2}


def test_propellers_import_errors(monkeypatch):
    without_integrations(monkeypatch, "propellers")
    assert "not available" in prop_tools.propeller_bemt_analysis({}, {}, {}).lower()
    assert "not available" in prop_tools.uav_energy_estimate({}, {}).lower()
    assert "not available" in prop_tools.get_propeller_database().lower()


def test_propellers_invalid_input_errors():
    """Missing required geometry/config fields are reported, not raised."""
    out = prop_tools.propeller_bemt_analysis({}, {"rpm_list": [3000]}, {})
    assert out.startswith("Propeller analysis error")
    assert "diameter_m" in out
    out = prop_tools.uav_energy_estimate({}, BATTERY)
    assert out.startswith("UAV energy analysis error")
    assert "mass_kg" in out


def test_get_propeller_database():
    data = json.loads(prop_tools.get_propeller_database())
    assert data["APC_10x7"]["diameter_m"] == 0.254
    assert data["APC_10x7"]["num_blades"] == 2


def test_uav_energy_recommendations_branches():
    """Flight-time and efficiency advice follows the real energy estimate."""
    tiny_battery = {"capacity_ah": 1.0, "voltage_nominal_v": 11.1, "mass_kg": 0.1}
    very_short = prop_tools.uav_energy_estimate(
        {"mass_kg": 3.0, "disk_area_m2": 0.1}, tiny_battery
    )
    assert _json_part(very_short)["flight_time_min"] < 10
    assert "Very short flight time" in very_short
    assert "Rotor Disk Area: 0.10" in very_short  # multirotor branch

    short = prop_tools.uav_energy_estimate(QUAD, BATTERY)
    assert 10 <= _json_part(short)["flight_time_min"] < 20
    assert "Short flight time" in short
    assert "Good system efficiency" in short

    # Motor 60 % x ESC 95 % = 57 % overall
    inefficient = prop_tools.uav_energy_estimate(
        {**QUAD, "motor_efficiency": 0.6}, BATTERY
    )
    assert _json_part(inefficient)["efficiency_overall"] == 0.6 * 0.95
    assert "Low system efficiency" in inefficient

    glider = prop_tools.uav_energy_estimate(
        {"mass_kg": 2.0, "wing_area_m2": 1.0},
        {"capacity_ah": 20.0, "voltage_nominal_v": 22.2, "mass_kg": 2.0},
        {"velocity_ms": 12.0},
    )
    assert _json_part(glider)["flight_time_min"] > 120
    assert "Excellent endurance" in glider
    assert "Rotor Disk Area" not in glider


# ---------------------------------------------------------------------------
# End-to-end tests against the real integration module
# ---------------------------------------------------------------------------


def _json_part(out: str):
    import json

    return json.loads(out.split("JSON Data:")[1])


def test_propeller_bemt_analysis_end_to_end():
    from aerospace_mcp.tools.propellers import propeller_bemt_analysis

    geo = {"diameter_m": 0.254, "pitch_m": 0.1143, "num_blades": 2}
    # Audit case: 10x4.5 at 10 m/s used to give ~0 N of thrust
    data = _json_part(
        propeller_bemt_analysis(geo, {"rpm_list": [5000, 7000], "velocity_ms": 10.0})
    )
    for point in data:
        assert point["thrust_n"] > 0.5
        assert 0.4 < point["efficiency"] < 0.8
        ratio = point["advance_ratio"] * point["thrust_coefficient"]
        assert abs(point["efficiency"] - ratio / point["power_coefficient"]) < 1e-9

    # Static: efficiency 0 (was 0.5); n_elements option is honoured
    data = _json_part(
        propeller_bemt_analysis(
            geo, {"rpm_list": [5000], "velocity_ms": 0.0}, {"n_elements": 20}
        )
    )
    assert data[0]["efficiency"] == 0.0
    assert data[0]["thrust_n"] > 0

    assert "rpm values must be positive" in propeller_bemt_analysis(
        geo, {"rpm_list": [-3000], "velocity_ms": 10.0}
    )


def test_uav_energy_estimate_end_to_end():
    from aerospace_mcp.tools.propellers import uav_energy_estimate

    battery = {"capacity_ah": 5.0, "voltage_nominal_v": 14.8, "mass_kg": 0.5}

    # Fixed-wing power depends on mass (it did not when cl_cruise was given)
    light = _json_part(
        uav_energy_estimate(
            {"mass_kg": 2.0, "wing_area_m2": 0.5, "cl_cruise": 0.5},
            battery,
            {"velocity_ms": 15.0, "altitude_m": 0.0},
        )
    )
    heavy = _json_part(
        uav_energy_estimate(
            {"mass_kg": 5.0, "wing_area_m2": 0.5, "cl_cruise": 0.5},
            battery,
            {"velocity_ms": 15.0, "altitude_m": 0.0},
        )
    )
    assert abs(light["power_required_w"] - 45.3) < 0.2
    assert abs(heavy["power_required_w"] - 53.6) < 0.2
    assert light["aircraft_type"] == "fixed_wing"

    # Both areas: labelled VTOL, with hover time from the rotors
    out = uav_energy_estimate(
        {"mass_kg": 2.0, "wing_area_m2": 0.5, "disk_area_m2": 0.2},
        battery,
        {"velocity_ms": 15.0},
    )
    assert "VTOL" in out
    assert _json_part(out)["hover_time_min"] > 0

    # Propeller geometry inside uav_config feeds the BEMT efficiency
    data = _json_part(
        uav_energy_estimate(
            {
                "mass_kg": 2.0,
                "wing_area_m2": 0.5,
                "propeller_geometry": {
                    "diameter_m": 0.254,
                    "pitch_m": 0.1143,
                    "num_blades": 2,
                },
            },
            battery,
            {"velocity_ms": 15.0},
        )
    )
    assert data["propeller_efficiency"] != 0.8

    # Generic config at zero speed: clear error instead of ZeroDivisionError
    assert "velocity_ms > 0" in uav_energy_estimate(
        {"mass_kg": 2.0}, battery, {"velocity_ms": 0.0}
    )
