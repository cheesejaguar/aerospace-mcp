from __future__ import annotations

import sys

from tests.tools.test_tools_common import StubModel, make_module


class PropellerGeometry:
    def __init__(self, **kwargs):
        self.diameter_m = kwargs.get("diameter_m", 0.3)
        self.pitch_m = kwargs.get("pitch_m", 0.2)
        self.num_blades = kwargs.get("num_blades", 2)


def test_propeller_and_uav_success():
    def _bemt(geom, rpm_list, vel, alt):
        return [
            StubModel(
                rpm=rpm,
                thrust_n=10.0,
                torque_nm=1.0,
                power_w=100.0,
                efficiency=0.8,
                advance_ratio=0.5,
                thrust_coefficient=0.1,
                power_coefficient=0.05,
            )
            for rpm in rpm_list
        ]

    class UAVConfig:
        def __init__(self, **kw):
            self.mass_kg = kw.get("mass_kg", 5.0)
            self.wing_area_m2 = kw.get("wing_area_m2", 1.0)
            self.disk_area_m2 = kw.get("disk_area_m2", 0.0)

    class Battery:
        def __init__(self, **kw):
            self.capacity_ah = kw.get("capacity_ah", 10.0)
            self.voltage_nominal_v = kw.get("voltage_nominal_v", 22.2)
            self.mass_kg = kw.get("mass_kg", 1.0)

    class EnergyResult:
        flight_time_min = 30.0
        range_km = 10.0
        hover_time_min = None
        power_required_w = 200.0
        energy_consumed_wh = 300.0
        battery_energy_wh = 400.0
        efficiency_overall = 0.8

    sys.modules["aerospace_mcp.integrations.propellers"] = make_module(
        PropellerGeometry=PropellerGeometry,
        propeller_bemt_analysis=_bemt,
        UAVConfiguration=UAVConfig,
        BatteryConfiguration=Battery,
        uav_energy_estimate=lambda *a, **k: EnergyResult(),
        PROPELLER_DATABASE={"APC_10x4.7": {"diameter": 0.254}},
    )

    from aerospace_mcp.tools.propellers import (
        get_propeller_database,
        propeller_bemt_analysis,
        uav_energy_estimate,
    )

    assert "BEMT Analysis" in propeller_bemt_analysis(
        {"diameter_m": 0.3, "pitch_m": 0.2, "num_blades": 2},
        {"rpm_list": [2000, 3000], "velocity_ms": 20.0, "altitude_m": 0.0},
        {},
    )
    assert "UAV Energy" in uav_energy_estimate(
        {"mass_kg": 5.0, "wing_area_m2": 1.0},
        {"capacity_ah": 10.0, "voltage_nominal_v": 22.2, "mass_kg": 1.0},
        {},
    )
    assert "APC_10x4.7" in get_propeller_database()


def test_propellers_import_errors():
    import sys

    from aerospace_mcp.tools import propellers as prop_tools

    sys.modules["aerospace_mcp.integrations.propellers"] = make_module()
    assert "not available" in prop_tools.propeller_bemt_analysis({}, {}, {}).lower()
    assert "not available" in prop_tools.uav_energy_estimate({}, {}).lower()
    assert "not available" in prop_tools.get_propeller_database().lower()


def test_propellers_exception_branches():
    import sys

    def _boom(*a, **k):
        raise RuntimeError("boom")

    sys.modules["aerospace_mcp.integrations.propellers"] = make_module(
        PropellerGeometry=PropellerGeometry,
        propeller_bemt_analysis=_boom,
        UAVConfiguration=type("U", (), {}),
        BatteryConfiguration=type("B", (), {}),
        uav_energy_estimate=_boom,
        PROPELLER_DATABASE={"x": 1},
    )
    from aerospace_mcp.tools import propellers as ptools

    assert "error" in ptools.propeller_bemt_analysis({}, {}, {}).lower()
    assert "error" in ptools.uav_energy_estimate({}, {}).lower()

    # Force exception via database access
    class BadDB:
        def __iter__(self):
            raise RuntimeError("boom")

    sys.modules["aerospace_mcp.integrations.propellers"] = make_module(
        PROPELLER_DATABASE=BadDB()
    )
    assert "error" in ptools.get_propeller_database().lower()


def test_uav_energy_recommendations_branches():
    import sys

    class UAV:
        def __init__(self, **kw):
            self.mass_kg = 5.0
            self.wing_area_m2 = kw.get("wing_area_m2", 1.0)
            self.disk_area_m2 = kw.get("disk_area_m2", 0.0)

    class Bat:
        def __init__(self, **kw):
            self.capacity_ah = 10.0
            self.voltage_nominal_v = 22.2
            self.mass_kg = 1.0

    # Case: very short (<10) and low efficiency
    class R1:
        flight_time_min = 5.0
        range_km = None
        hover_time_min = 2.0
        power_required_w = 100
        energy_consumed_wh = 100
        battery_energy_wh = 200
        efficiency_overall = 0.5

    # Case: short (<20)
    class R2(R1):
        flight_time_min = 15.0
        efficiency_overall = 0.9

    # Case: excellent (>120)
    class R3(R1):
        flight_time_min = 130.0
        efficiency_overall = 0.9

    sys.modules["aerospace_mcp.integrations.propellers"] = make_module(
        UAVConfiguration=UAV,
        BatteryConfiguration=Bat,
        uav_energy_estimate=lambda *a, **k: R1(),
    )
    from aerospace_mcp.tools.propellers import uav_energy_estimate as uav

    out1 = uav(
        {"wing_area_m2": 1.0},
        {"capacity_ah": 10.0, "voltage_nominal_v": 22.2, "mass_kg": 1.0},
    )
    assert "Very short" in out1
    assert "Low system efficiency" in out1

    sys.modules["aerospace_mcp.integrations.propellers"] = make_module(
        UAVConfiguration=UAV,
        BatteryConfiguration=Bat,
        uav_energy_estimate=lambda *a, **k: R2(),
    )
    out2 = uav(
        {"wing_area_m2": 1.0},
        {"capacity_ah": 10.0, "voltage_nominal_v": 22.2, "mass_kg": 1.0},
    )
    assert "Short flight time" in out2

    sys.modules["aerospace_mcp.integrations.propellers"] = make_module(
        UAVConfiguration=UAV,
        BatteryConfiguration=Bat,
        uav_energy_estimate=lambda *a, **k: R3(),
    )
    out3 = uav(
        {"wing_area_m2": 1.0},
        {"capacity_ah": 10.0, "voltage_nominal_v": 22.2, "mass_kg": 1.0},
    )
    assert "Excellent endurance" in out3

    # Multirotor branch (disk area)
    class UAV2(UAV):
        def __init__(self, **kw):
            super().__init__(**kw)
            self.wing_area_m2 = 0.0
            self.disk_area_m2 = 2.0

    sys.modules["aerospace_mcp.integrations.propellers"] = make_module(
        UAVConfiguration=UAV2,
        BatteryConfiguration=Bat,
        uav_energy_estimate=lambda *a, **k: R2(),
    )
    out4 = uav(
        {"disk_area_m2": 2.0},
        {"capacity_ah": 10.0, "voltage_nominal_v": 22.2, "mass_kg": 1.0},
    )
    assert "Rotor Disk Area" in out4
