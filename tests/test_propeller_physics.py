"""Known-answer and sanity checks for the propeller BEMT and UAV power models.

References:
    - Momentum theory (Leishman, "Principles of Helicopter Aerodynamics",
      ch. 2): ideal hover power P_ideal = T^1.5 / sqrt(2 rho A); with CT, CP
      based on n and D, CP_ideal = CT^1.5 / sqrt(pi / 2).  The figure of
      merit FM = P_ideal / P of small propellers is 0.5-0.75.
    - Glauert forward-flight inflow: v_i = v_h^2 / sqrt(V^2 + v_i^2).
    - UIUC propeller database (Brandt & Selig, AIAA 2011-1255): an APC
      10x4.7 at J ~ 0.34 has CT ~ 0.07-0.08 and efficiency ~ 0.6; thrust
      vanishes somewhat above J = p/D.
"""

import math

import pytest

from aerospace_mcp.integrations.atmosphere import get_atmosphere_profile
from aerospace_mcp.integrations.propellers import (
    GRAVITY,
    BatteryConfiguration,
    PropellerGeometry,
    UAVConfiguration,
    _bemt_solve,
    _glauert_induced_velocity,
    motor_propeller_matching,
    propeller_bemt_analysis,
    uav_energy_estimate,
)

RHO_SL = 1.225  # ISA sea level density (kg/m^3)
# 10 x 4.5 in two-blade propeller from the audit
PROP_10x45 = PropellerGeometry(diameter_m=0.254, pitch_m=0.1143, num_blades=2)


def _rho(alt):
    return get_atmosphere_profile([alt])[0].density_kg_m3


class TestStaticThrust:
    @pytest.mark.parametrize(
        "D,p,B",
        [
            (0.254, 0.1143, 2),
            (0.254, 0.178, 2),
            (0.3048, 0.2032, 2),
            (0.2032, 0.1143, 3),
        ],
    )
    def test_figure_of_merit_vs_momentum_theory(self, D, p, B):
        """Static FM = P_ideal / P is 0.5-0.75 (momentum theory)."""
        geo = PropellerGeometry(diameter_m=D, pitch_m=p, num_blades=B)
        pt = propeller_bemt_analysis(geo, [5000.0], 0.0)[0]
        area = math.pi * D**2 / 4
        p_ideal = pt.thrust_n**1.5 / math.sqrt(2 * RHO_SL * area)
        fm = p_ideal / pt.power_w
        assert 0.5 <= fm <= 0.75
        # Same statement in coefficient form: CP_ideal = CT^1.5 / sqrt(pi/2)
        fm_coeff = pt.thrust_coefficient**1.5 / (
            math.sqrt(math.pi / 2) * pt.power_coefficient
        )
        assert fm_coeff == pytest.approx(fm, rel=1e-6)

    def test_static_efficiency_is_zero(self):
        """eta = J CT / CP is 0 at V = 0 (was reported as 0.5)."""
        for pt in propeller_bemt_analysis(PROP_10x45, [3000, 5000, 7000], 0.0):
            assert pt.efficiency == 0.0
            assert pt.thrust_n > 0
            assert pt.power_w > 0

    def test_static_coefficients_independent_of_rpm(self):
        """Without Re/Mach effects CT and CP do not depend on rpm."""
        pts = propeller_bemt_analysis(PROP_10x45, [3000, 7000], 0.0)
        assert pts[0].thrust_coefficient == pytest.approx(pts[1].thrust_coefficient)
        assert pts[0].power_coefficient == pytest.approx(pts[1].power_coefficient)
        # Small-propeller range (UIUC static CT ~0.08-0.13)
        assert 0.07 < pts[0].thrust_coefficient < 0.14

    def test_blades_and_activity_factor_matter(self):
        """Solidity (blades, activity factor) changes thrust (was ignored)."""
        base = propeller_bemt_analysis(PROP_10x45, [6000.0], 0.0)[0]
        three = propeller_bemt_analysis(
            PropellerGeometry(diameter_m=0.254, pitch_m=0.1143, num_blades=3),
            [6000.0],
            0.0,
        )[0]
        wide = propeller_bemt_analysis(
            PropellerGeometry(
                diameter_m=0.254, pitch_m=0.1143, num_blades=2, activity_factor=150
            ),
            [6000.0],
            0.0,
        )[0]
        assert three.thrust_n > base.thrust_n
        assert wide.thrust_n > base.thrust_n


class TestForwardFlight:
    def test_10x45_at_10ms_produces_thrust(self):
        """The audit case gave ~0 N; UIUC APC 10x4.7 data give CT ~0.07 at J~0.34."""
        pts = propeller_bemt_analysis(PROP_10x45, [5000, 6000, 7000], 10.0)
        for pt in pts:
            assert pt.thrust_n > 0.5
        at_7000 = pts[-1]  # J = 0.34
        assert at_7000.advance_ratio == pytest.approx(0.3375, abs=1e-3)
        assert 0.045 < at_7000.thrust_coefficient < 0.09
        assert 0.5 < at_7000.efficiency < 0.75

    def test_efficiency_definition_and_bounds(self):
        """eta = J CT / CP exactly, never above 0.9 (no clamping needed)."""
        n = 6000 / 60
        D = PROP_10x45.diameter_m
        for J in [0.1, 0.2, 0.3, 0.4, 0.5, 0.55]:
            pt = propeller_bemt_analysis(PROP_10x45, [6000.0], J * n * D)[0]
            assert pt.efficiency == pytest.approx(
                pt.advance_ratio * pt.thrust_coefficient / pt.power_coefficient
            )
            assert 0.0 < pt.efficiency <= 0.9
            # Power includes the useful power T V: P > T V
            assert pt.power_w > pt.thrust_n * J * n * D

    def test_no_discontinuity_at_low_advance_ratio(self):
        """The old model switched regimes at J = 0.1 (65% thrust jump)."""
        n = 5000 / 60
        D = PROP_10x45.diameter_m
        lo = propeller_bemt_analysis(PROP_10x45, [5000.0], 0.099 * n * D)[0]
        hi = propeller_bemt_analysis(PROP_10x45, [5000.0], 0.101 * n * D)[0]
        assert hi.thrust_n == pytest.approx(lo.thrust_n, rel=0.01)

    def test_zero_thrust_slightly_above_geometric_pitch(self):
        """Thrust vanishes near J = p/D (somewhat above it with camber)."""
        n = 6000 / 60
        D = PROP_10x45.diameter_m
        p_over_d = PROP_10x45.pitch_m / D
        js = [0.02 * i for i in range(5, 60)]
        pts = [propeller_bemt_analysis(PROP_10x45, [6000.0], j * n * D)[0] for j in js]
        j_zero = next(p.advance_ratio for p in pts if p.thrust_n <= 0)
        assert p_over_d < j_zero < 1.6 * p_over_d
        # Thrust falls monotonically with J
        thrusts = [p.thrust_n for p in pts]
        assert all(b < a for a, b in zip(thrusts, thrusts[1:], strict=False))

    def test_windmilling_reports_negative_thrust(self):
        """Far above zero-thrust J the propeller windmills (no clamping)."""
        pt = propeller_bemt_analysis(PROP_10x45, [2000.0], 30.0)[0]
        assert pt.thrust_n < 0
        assert pt.efficiency == 0.0
        # Torque and power have the same sign: P = 2 pi n Q
        assert pt.power_w == pytest.approx(2 * math.pi * 2000 / 60 * pt.torque_nm)

    def test_solver_converges_everywhere(self):
        import numpy as np

        sol = _bemt_solve(PROP_10x45, np.array([1000.0, 5000.0, 9000.0]), 15.0, RHO_SL)
        assert sol["converged"].all()
        sol = _bemt_solve(PROP_10x45, np.array([5000.0]), 0.0, RHO_SL)
        assert sol["converged"].all()


class TestValidation:
    def test_rpm_must_be_positive(self):
        with pytest.raises(ValueError, match="rpm"):
            propeller_bemt_analysis(PROP_10x45, [0.0], 10.0)
        with pytest.raises(ValueError, match="rpm"):
            propeller_bemt_analysis(PROP_10x45, [-3000.0], 10.0)

    def test_velocity_must_be_non_negative(self):
        with pytest.raises(ValueError, match="velocity"):
            propeller_bemt_analysis(PROP_10x45, [3000.0], -5.0)

    def test_hub_radius_checked(self):
        geo = PropellerGeometry(
            diameter_m=0.254, pitch_m=0.1143, num_blades=2, hub_radius_m=0.1
        )
        with pytest.raises(ValueError, match="hub_radius_m"):
            propeller_bemt_analysis(geo, [3000.0], 0.0)
        # The 0.02 m default is limited to 0.2R on tiny propellers, also
        # after a model_dump() round trip
        tiny = PropellerGeometry(diameter_m=0.05, pitch_m=0.025, num_blades=2)
        assert propeller_bemt_analysis(tiny, [20000.0], 0.0)[0].thrust_n > 0
        tiny = PropellerGeometry(**tiny.model_dump())
        assert propeller_bemt_analysis(tiny, [20000.0], 0.0)[0].thrust_n > 0

    def test_empty_rpm_list(self):
        assert propeller_bemt_analysis(PROP_10x45, [], 0.0) == []


class TestMultirotorPower:
    MASS = 1.5
    AREA = 4 * math.pi * 0.127**2  # four 10-inch rotors, 0.2027 m^2

    @pytest.mark.parametrize("V,p_ideal", [(5.0, 65.2), (10.0, 41.9), (15.0, 28.8)])
    def test_glauert_induced_power(self, V, p_ideal):
        """Ideal induced power T v_i from Glauert's inflow (audit values)."""
        T = self.MASS * GRAVITY
        v_h = math.sqrt(T / (2 * RHO_SL * self.AREA))
        v_i = _glauert_induced_velocity(v_h, V)
        assert v_i == pytest.approx(v_h**2 / math.sqrt(V**2 + v_i**2), rel=1e-9)
        assert T * v_i == pytest.approx(p_ideal, abs=0.1)

    def _run(self, V, **kw):
        uav = UAVConfiguration(
            mass_kg=self.MASS, disk_area_m2=self.AREA, num_motors=4, **kw
        )
        bat = BatteryConfiguration(capacity_ah=5.0, voltage_nominal_v=14.8, mass_kg=0.5)
        return uav_energy_estimate(uav, bat, {"velocity_ms": V, "altitude_m": 0.0})

    def test_hover_power_is_ideal_over_figure_of_merit(self):
        """Hover shaft power = T^1.5/sqrt(2 rho A)/FM; avionics added as is."""
        r = self._run(0.0)
        T = self.MASS * GRAVITY
        p_ideal = T**1.5 / math.sqrt(2 * _rho(0.0) * self.AREA)
        assert r.hover_power_w == pytest.approx(p_ideal / 0.7 + 50.0, rel=1e-6)
        # Avionics power is electrical: not divided by motor/ESC efficiency
        eff = 0.85 * 0.95
        assert r.electrical_power_w == pytest.approx(p_ideal / 0.7 / eff + 50.0)
        assert r.flight_time_min == pytest.approx(r.hover_time_min)

    def test_power_decreases_into_forward_flight(self):
        """Induced power falls with speed (it used to rise as sqrt(1+mu^2))."""
        p = [self._run(V).power_required_w for V in (0.0, 5.0, 10.0, 15.0)]
        assert p[0] > p[1] > p[2] > p[3]

    def test_hover_time_independent_of_cruise_speed(self):
        """hover_time_min is evaluated in hover, not at the mission speed."""
        assert self._run(5.0).hover_time_min == pytest.approx(
            self._run(15.0).hover_time_min
        )

    def test_parasite_drag_area(self):
        """A larger drag area raises fast-forward-flight power."""
        clean = self._run(15.0)
        draggy = self._run(15.0, drag_area_m2=0.05)
        drag_power = 0.5 * _rho(0.0) * 15.0**3 * 0.05
        assert draggy.power_required_w > clean.power_required_w + 0.8 * drag_power

    def test_bemt_figure_of_merit_with_propeller(self):
        uav = UAVConfiguration(mass_kg=self.MASS, disk_area_m2=self.AREA, num_motors=4)
        bat = BatteryConfiguration(capacity_ah=5.0, voltage_nominal_v=14.8, mass_kg=0.5)
        r = uav_energy_estimate(uav, bat, {"velocity_ms": 0.0}, PROP_10x45)
        assert 0.5 < r.figure_of_merit < 0.8
        assert r.figure_of_merit != pytest.approx(0.7)


class TestFixedWingPower:
    def _run(self, mass, cd0=0.03, V=15.0, cl_cruise=None, alt=0.0, prop=None):
        uav = UAVConfiguration(
            mass_kg=mass, wing_area_m2=0.5, cd0=cd0, cl_cruise=cl_cruise
        )
        bat = BatteryConfiguration(capacity_ah=5.0, voltage_nominal_v=14.8, mass_kg=0.5)
        mission = {"altitude_m": alt}
        if V is not None:
            mission["velocity_ms"] = V
        return uav_energy_estimate(uav, bat, mission, prop)

    @staticmethod
    def _expected(mass, cd0=0.03, V=15.0, S=0.5, k=0.015):
        """P = D V / 0.8 + 5 W with CL = 2W/(rho V^2 S), CD = CD0 + k CL^2."""
        q = 0.5 * _rho(0.0) * V**2
        cl = mass * GRAVITY / (q * S)
        drag = q * S * (cd0 + k * cl**2)
        return drag * V / 0.8 + 5.0

    @pytest.mark.parametrize("mass,watts", [(2.0, 45.3), (5.0, 53.6)])
    def test_power_depends_on_mass(self, mass, watts):
        """Audit expectation: ~45 W (2 kg) and ~54 W (5 kg) before efficiencies.

        The user's cl_cruise no longer replaces the level-flight CL.
        """
        r = self._run(mass, cl_cruise=0.5)
        assert r.power_required_w == pytest.approx(self._expected(mass), rel=1e-9)
        assert r.power_required_w == pytest.approx(watts, abs=0.2)
        assert any("cl_cruise" in w for w in r.warnings)

    def test_cd0_not_capped(self):
        """cd0 used to be silently capped at 0.02."""
        p = [self._run(2.0, cd0=c).power_required_w for c in (0.02, 0.05, 0.1)]
        assert p[0] < p[1] < p[2]
        assert p[2] == pytest.approx(self._expected(2.0, cd0=0.1), rel=1e-9)

    def test_cl_cruise_sets_speed_when_velocity_missing(self):
        r = self._run(2.0, V=None, cl_cruise=0.6, alt=100.0)
        v_expected = math.sqrt(2 * 2.0 * GRAVITY / (_rho(100.0) * 0.5 * 0.6))
        assert r.cruise_velocity_ms == pytest.approx(v_expected)
        assert r.warnings == []

    def test_stall_warning(self):
        r = self._run(5.0, V=6.0)
        assert any("CL_max" in w for w in r.warnings)

    def test_zero_velocity_rejected(self):
        with pytest.raises(ValueError, match="velocity"):
            self._run(2.0, V=0.0)

    def test_bemt_propeller_efficiency(self):
        """With a propeller, eta comes from BEMT at thrust = drag."""
        r = self._run(2.0, prop=PROP_10x45)
        assert 0.4 < r.propeller_efficiency < 0.8
        assert r.propeller_efficiency != pytest.approx(0.8)


class TestOtherConfigurations:
    BAT = BatteryConfiguration(capacity_ah=5.0, voltage_nominal_v=14.8, mass_kg=0.5)

    def test_vtol_has_cruise_and_hover(self):
        uav = UAVConfiguration(mass_kg=2.0, wing_area_m2=0.5, disk_area_m2=0.2)
        r = uav_energy_estimate(uav, self.BAT, {"velocity_ms": 15.0})
        assert r.aircraft_type == "vtol"
        assert r.range_km > 0
        assert r.hover_time_min < r.flight_time_min

    def test_generic_zero_velocity_is_an_error(self):
        """Used to raise ZeroDivisionError."""
        with pytest.raises(ValueError, match="velocity_ms > 0"):
            uav_energy_estimate(
                UAVConfiguration(mass_kg=2.0), self.BAT, {"velocity_ms": 0.0}
            )


def test_motor_matching_uses_bemt_static_power():
    """motor_propeller_matching reports the BEMT static operating point."""
    out = motor_propeller_matching(1000, 11.1, ["APC_10x7"], 10.0)["APC_10x7"]
    op = out["operating_point"]
    geo = PropellerGeometry(**out["geometry"])
    again = propeller_bemt_analysis(geo, [op["rpm"]], 0.0)[0]
    assert op["power_w"] == pytest.approx(again.power_w)
    fm = (
        again.thrust_n**1.5 / math.sqrt(2 * RHO_SL * math.pi * 0.127**2) / again.power_w
    )
    assert 0.5 < fm < 0.8
