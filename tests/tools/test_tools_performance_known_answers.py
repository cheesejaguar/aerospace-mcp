"""Known-answer tests for the aircraft performance tools.

These call the tools unstubbed (the performance module has no integration
dependencies) and check them against independent references: the ISA
compressible-flow (pitot) relations, the ISA density profile, Raymer's
takeoff transition model and the 14 CFR distance / fuel rules.
"""

import json
import math

import pytest

from aerospace_mcp.tools.performance import (
    density_altitude_calculator,
    fuel_reserve_calculator,
    landing_performance,
    stall_speed_calculator,
    takeoff_performance,
    true_airspeed_converter,
    weight_and_balance,
)


def extract_json(result: str) -> dict:
    """Return the JSON block that ends every tool report."""
    return json.loads(result[result.index("\n{") + 1 :])


def _isa_ref(h_m: float) -> tuple[float, float, float]:
    """Independent ISA reference: (T [K], p [Pa], rho [kg/m^3])."""
    t0, p0, lapse, r, g = 288.15, 101325.0, 0.0065, 287.05287, 9.80665
    if h_m <= 11000:
        t = t0 - lapse * h_m
        p = p0 * (t / t0) ** (g / (r * lapse))
    else:
        t = 216.65
        p = 22632.06 * math.exp(-g * (h_m - 11000) / (r * t))
    return t, p, p / (r * t)


def _speeds(value, kind, alt_ft, **kw) -> dict:
    return extract_json(true_airspeed_converter(value, kind, alt_ft, **kw))


class TestAirspeedKnownAnswers:
    """Compressible ISA airspeed relations.

    Reference values computed independently from
    qc = P0 ((1 + 0.2 (CAS/a0)^2)^3.5 - 1), M = sqrt(5 ((qc/p + 1)^(2/7) - 1)),
    TAS = M a, EAS = TAS sqrt(rho/rho0) (subsonic St. Venant pitot formula,
    e.g. Anderson, Introduction to Flight, ch. 4).  Before the fix every
    input returned TAS = CAS.
    """

    def test_cas_250_at_10000ft(self):
        d = _speeds(250, "CAS", 10000)
        assert d["TAS_kts"] == pytest.approx(288.7, abs=0.15)
        assert d["EAS_kts"] == pytest.approx(248.1, abs=0.15)
        assert d["MACH"] == pytest.approx(0.4523, abs=2e-4)
        assert d["CAS_kts"] == 250

    def test_mach_078_at_fl350(self):
        d = _speeds(0.78, "MACH", 35000)
        assert d["MACH"] == pytest.approx(0.78)
        assert d["TAS_kts"] == pytest.approx(449.6, abs=0.15)
        assert d["CAS_kts"] == pytest.approx(264.4, abs=0.15)
        assert d["EAS_kts"] == pytest.approx(250.3, abs=0.15)

    def test_tas_450_at_fl350(self):
        d = _speeds(450, "TAS", 35000)
        assert d["TAS_kts"] == pytest.approx(450.0, abs=0.05)
        assert d["CAS_kts"] == pytest.approx(264.7, abs=0.15)
        assert d["MACH"] == pytest.approx(0.7807, abs=2e-4)

    def test_eas_250_at_10000ft(self):
        d = _speeds(250, "EAS", 10000)
        assert d["EAS_kts"] == pytest.approx(250.0, abs=0.05)
        assert d["CAS_kts"] == pytest.approx(251.9, abs=0.15)
        assert d["TAS_kts"] == pytest.approx(290.9, abs=0.15)

    def test_cas_300_at_fl350(self):
        d = _speeds(300, "CAS", 35000)
        assert d["MACH"] == pytest.approx(0.8736, abs=2e-4)
        assert d["TAS_kts"] == pytest.approx(503.5, abs=0.15)

    def test_ias_position_error(self):
        """IAS + position error = CAS; IAS is reported back unchanged."""
        d = _speeds(245, "IAS", 10000, position_error_kts=5)
        assert d["CAS_kts"] == pytest.approx(250.0)
        assert d["IAS_kts"] == pytest.approx(245.0)
        assert d["TAS_kts"] == pytest.approx(288.7, abs=0.15)

    def test_sea_level_isa_all_equal(self):
        """At sea-level ISA, CAS = EAS = TAS."""
        d = _speeds(200, "CAS", 0)
        assert d["EAS_kts"] == pytest.approx(200.0, abs=0.1)
        assert d["TAS_kts"] == pytest.approx(200.0, abs=0.1)

    @pytest.mark.parametrize("kind", ["CAS", "EAS", "TAS", "MACH"])
    @pytest.mark.parametrize("alt_ft", [0, 10000, 35000, 45000])
    def test_round_trip_consistency(self, kind, alt_ft):
        """Feeding any output back in reproduces the same Mach."""
        base = _speeds(0.6, "MACH", alt_ft)
        value = base["MACH"] if kind == "MACH" else base[f"{kind}_kts"]
        again = _speeds(value, kind, alt_ft)
        assert again["MACH"] == pytest.approx(base["MACH"], abs=2e-4)

    def test_non_isa_temperature(self):
        """CAS -> Mach is temperature-independent; TAS scales with sqrt(T)."""
        isa = _speeds(280, "CAS", 20000)
        t_isa_c = isa["input"]["temperature_c"]
        hot = _speeds(280, "CAS", 20000, temperature_c=t_isa_c + 20)
        assert hot["MACH"] == pytest.approx(isa["MACH"], abs=1e-4)
        t_isa = t_isa_c + 273.15
        ratio = math.sqrt((t_isa + 20) / t_isa)
        assert hot["TAS_kts"] == pytest.approx(isa["TAS_kts"] * ratio, abs=0.2)
        # EAS = M sqrt(gamma p / rho0) does not depend on temperature either
        assert hot["EAS_kts"] == pytest.approx(isa["EAS_kts"], abs=0.1)

    def test_supersonic_round_trip(self):
        """Above M 1 the Rayleigh pitot formula is used.

        M 2 at 40,000 ft: qc/p = 166.92 M^7 / (7 M^2 - 1)^2.5 - 1 = 4.64,
        qc / P0 = 0.859 (subsonic at sea level) -> CAS ~ 651 kt.
        """
        d = _speeds(2.0, "MACH", 40000)
        assert d["CAS_kts"] == pytest.approx(651.1, abs=0.5)
        back = _speeds(d["CAS_kts"], "CAS", 40000)
        assert back["MACH"] == pytest.approx(2.0, abs=1e-3)

    def test_negative_speed_rejected(self):
        assert true_airspeed_converter(-5, "TAS", 10000).startswith("Error")


class TestDensityAltitudeKnownAnswers:
    """Exact inversion of the ISA density profile (was 118.8 ft/degC)."""

    @pytest.mark.parametrize(
        "pa_ft, oat_c, expected_ft",
        [(5000, 30, 7801), (8000, 35, 11937), (40000, -40, 41527), (0, 15, 0)],
    )
    def test_dry_air(self, pa_ft, oat_c, expected_ft):
        data = extract_json(density_altitude_calculator(pa_ft, oat_c))
        assert data["density_altitude_ft"] == pytest.approx(expected_ft, abs=2)

    def test_matches_independent_isa_density(self):
        """The ISA density at the reported DA equals the actual density."""
        data = extract_json(density_altitude_calculator(6000, 28))
        _, _, rho_at_da = _isa_ref(data["density_altitude_m"])
        assert rho_at_da == pytest.approx(data["air_density_kg_m3"], rel=2e-4)

    def test_humidity_virtual_temperature(self):
        """Dewpoint 25 C at 5000 ft / 30 C adds ~460 ft (was +75 ft)."""
        data = extract_json(density_altitude_calculator(5000, 30, dewpoint_c=25))
        assert 440 <= data["humidity_correction_ft"] <= 480
        assert data["density_altitude_ft"] == pytest.approx(8262, abs=5)
        assert data["virtual_temperature_c"] > 30

    def test_dewpoint_zero_is_reported(self):
        out = density_altitude_calculator(5000, 30, dewpoint_c=0)
        assert "Dewpoint: 0.0°C" in out
        assert "Not specified" not in out


class TestInputValidationAndDisplay:
    def test_stall_zero_cl_rejected(self):
        out = stall_speed_calculator(70000, 122.6, 0)
        assert out.startswith("Error: cl_max_clean must be > 0")

    def test_stall_zero_wing_area_rejected(self):
        out = stall_speed_calculator(70000, 0, 1.5)
        assert out.startswith("Error: wing_area_m2 must be > 0")

    def test_weight_balance_zero_limit_displayed(self):
        out = weight_and_balance(
            1000, 2.0, 100, 2.5, [], forward_cg_limit_m=0.0, aft_cg_limit_m=3.0
        )
        forward = [line for line in out.splitlines() if "Forward CG:" in line][0]
        assert "0.0" in forward
        assert "Not specified" not in forward


A320_TAKEOFF = {
    "weight_kg": 70000,
    "pressure_altitude_ft": 0,
    "temperature_c": 15,
    "thrust_to_weight": 0.3,
    "cl_max_takeoff": 2.0,
    "wing_area_m2": 122.6,
}


class TestTakeoffKnownAnswers:
    def test_raymer_air_distance(self):
        """Raymer, Aircraft Design, sec. 17.8 transition: V_TR = 1.15 VS,
        R = V_TR^2 / (0.2 g), sin(gamma) = (T - D) / W; obstacle cleared
        within the arc -> S = sqrt(R^2 - (R - h)^2) = 256 m (was 38 m)."""
        data = extract_json(takeoff_performance(**A320_TAKEOFF))
        assert data["distances_m"]["air_distance_to_35ft"] == pytest.approx(256, abs=1)
        w = 70000 * 9.80665
        vs = math.sqrt(2 * w / (1.225 * 122.6 * 2.0))
        v_tr = 1.15 * vs
        cl = w / (0.5 * 1.225 * v_tr**2 * 122.6)
        cd = 0.025 + cl**2 / (math.pi * 9 * 0.8)
        gamma = math.degrees(math.asin(0.3 - cd / cl))
        assert data["performance"]["climb_angle_deg"] == pytest.approx(gamma, abs=0.05)
        assert data["performance"]["transition_radius_m"] == pytest.approx(
            v_tr**2 / (0.2 * 9.80665), abs=1
        )

    @pytest.mark.parametrize("condition", ["dry", "wet", "contaminated"])
    def test_factored_is_far25_113(self, condition):
        """14 CFR 25.113(a)(2): 115% of the all-engines distance to 35 ft;
        the runway condition acts only through rolling friction."""
        data = extract_json(
            takeoff_performance(**A320_TAKEOFF, runway_condition=condition)
        )
        d = data["distances_m"]
        assert d["factored_distance"] == pytest.approx(
            d["total_distance"] * 1.15, abs=1
        )

    def test_v_speeds_are_calibrated(self):
        """KCAS V-speeds stay near their sea-level values at a hot/high
        airport while the KTAS ones grow (TAS used to be labelled KIAS)."""
        sl = extract_json(takeoff_performance(**A320_TAKEOFF))
        hot_high = extract_json(
            takeoff_performance(
                **{**A320_TAKEOFF, "pressure_altitude_ft": 5000, "temperature_c": 30}
            )
        )
        assert sl["v_speeds_kts"]["VS"] == pytest.approx(
            sl["v_speeds_ktas"]["VS"], abs=0.2
        )
        assert hot_high["v_speeds_kts"]["VS"] == pytest.approx(
            sl["v_speeds_kts"]["VS"], abs=0.5
        )
        assert hot_high["v_speeds_ktas"]["VS"] > hot_high["v_speeds_kts"]["VS"] + 10
        assert hot_high["performance"]["density_altitude_ft"] == pytest.approx(
            7801, abs=2
        )


A320_LANDING = {
    "weight_kg": 64000,
    "pressure_altitude_ft": 0,
    "temperature_c": 15,
    "cl_max_landing": 2.8,
    "wing_area_m2": 122.6,
}


class TestLandingKnownAnswers:
    def test_slope_sign(self):
        """a = mu g + g sin(atan(slope/100)), slope positive uphill: uphill
        shortens the roll, downhill lengthens it (was inverted)."""
        level = extract_json(landing_performance(**A320_LANDING))
        up = extract_json(landing_performance(**A320_LANDING, runway_slope_pct=2))
        down = extract_json(landing_performance(**A320_LANDING, runway_slope_pct=-2))
        roll = level["distances_m"]["ground_roll"]
        assert up["distances_m"]["ground_roll"] < roll
        assert down["distances_m"]["ground_roll"] > roll
        g = 9.80665
        expected = roll * 0.4 * g / (0.4 * g - g * math.sin(math.atan(0.02)))
        assert down["distances_m"]["ground_roll"] == pytest.approx(expected, abs=1.5)

    def test_regulatory_factors(self):
        """14 CFR 121.195(b) dry: / 0.6; 121.195(d) wet: 1.15 x the dry
        requirement; contaminated: actual distance + 15% (SAFO 06012).
        The condition is applied once (wet used to get both a low mu and a
        1.43 factor)."""
        dry = extract_json(landing_performance(**A320_LANDING))
        wet = extract_json(landing_performance(**A320_LANDING, runway_condition="wet"))
        cont = extract_json(
            landing_performance(**A320_LANDING, runway_condition="contaminated")
        )
        dry_total = dry["distances_m"]["total_distance"]
        assert dry["distances_m"]["factored_distance"] == pytest.approx(
            dry_total / 0.6, abs=1
        )
        assert dry["performance"]["condition_factor"] == pytest.approx(1.6667, abs=1e-3)
        assert wet["distances_m"]["factored_distance"] == pytest.approx(
            dry_total / 0.6 * 1.15, abs=1
        )
        assert wet["performance"]["condition_factor"] == pytest.approx(1.9167, abs=1e-3)
        assert cont["distances_m"]["factored_distance"] == pytest.approx(
            cont["distances_m"]["total_distance"] * 1.15, abs=1
        )

    @pytest.mark.parametrize(
        "wind, additive", [(0, 5), (-10, 5), (20, 10), (30, 15), (60, 20)]
    )
    def test_vapp_wind_additive(self, wind, additive):
        """Half the headwind, min 5 kt, max 20 kt; nothing extra for a
        tailwind (it used to add speed in a tailwind)."""
        d = extract_json(landing_performance(**A320_LANDING, wind_kts=wind))
        v = d["v_speeds_kts"]
        assert v["VAPP"] - v["VREF"] == pytest.approx(additive, abs=0.11)

    def test_speeds_are_calibrated(self):
        sl = extract_json(landing_performance(**A320_LANDING))
        high = extract_json(
            landing_performance(
                **{**A320_LANDING, "pressure_altitude_ft": 5000, "temperature_c": 30}
            )
        )
        assert high["v_speeds_kts"]["VS0"] == pytest.approx(
            sl["v_speeds_kts"]["VS0"], abs=0.5
        )
        assert high["v_speeds_ktas"]["VS0"] > high["v_speeds_kts"]["VS0"] + 10

    def test_cannot_stop_reports_none(self):
        out = landing_performance(
            **A320_LANDING, runway_slope_pct=-50, runway_condition="contaminated"
        )
        data = extract_json(out)
        assert data["distances_m"]["ground_roll"] is None
        assert data["warnings"]


def _reserves(reg: str, **kw) -> dict:
    args = {
        "trip_fuel_kg": 10000,
        "cruise_fuel_flow_kg_hr": 2400,
        "flight_time_min": 240,
        "alternate_fuel_kg": 1500,
    }
    args.update(kw)
    return extract_json(fuel_reserve_calculator(reg, **args))


class TestFuelReserveRules:
    """14 CFR 91.151 / 91.167 / 121.639 / 121.645 reserve rules."""

    def test_far121_domestic(self):
        """121.639: alternate + 45 min at normal cruise fuel flow."""
        d = _reserves("FAR_121_DOMESTIC")
        assert d["reserves"] == {
            "contingency_kg": 0,
            "alternate_kg": 1500,
            "final_reserve_kg": 1800.0,
        }

    def test_far121_flag_uses_flight_time(self):
        """121.645(b): 10% of flight time at cruise flow + alternate + 30 min
        hold (holding flow approximated as 80% of cruise)."""
        d4 = _reserves("FAR_121_FLAG", flight_time_min=240)
        d10 = _reserves("FAR_121_FLAG", flight_time_min=600)
        assert d4["reserves"]["contingency_kg"] == pytest.approx(0.1 * 4 * 2400)
        assert d10["reserves"]["contingency_kg"] == pytest.approx(0.1 * 10 * 2400)
        assert d4["reserves"]["final_reserve_kg"] == pytest.approx(0.8 * 2400 * 0.5)
        assert d4["reserves"]["alternate_kg"] == 1500

    @pytest.mark.parametrize(
        "reg, minutes",
        [("FAR_91_VFR_DAY", 30), ("FAR_91_VFR_NIGHT", 45), ("FAR_91_IFR", 45)],
    )
    def test_far91_variants(self, reg, minutes):
        d = _reserves(reg)
        assert d["reserves"]["final_reserve_kg"] == pytest.approx(2400 * minutes / 60)
        assert d["rule_applied"] == reg

    def test_far91_alias_is_ifr(self):
        d = _reserves("FAR_91")
        assert d["rule_applied"] == "FAR_91_IFR"
        assert any("treated as FAR_91_IFR" in n for n in d["notes"])

    def test_unknown_regulation(self):
        out = fuel_reserve_calculator("EASA_X", 1000, 100, 60)
        assert out.startswith("Error: Unknown regulation")

    def test_negative_input_rejected(self):
        out = fuel_reserve_calculator("ICAO", -1, 100, 60)
        assert out.startswith("Error: trip_fuel_kg")
