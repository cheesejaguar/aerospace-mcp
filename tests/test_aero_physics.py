"""Known-answer checks for the built-in (fallback) aerodynamics models.

References:
    - DATCOM 4.1.3.2 / Helmbold-Diederich lift slope:
      CL_alpha = 2 pi A / (2 + sqrt(A^2 beta^2 / kappa^2 (1 + tan^2 L_c/2 /
      beta^2) + 4)); AR 8 rectangular wing with 2*pi sections: 4.91 / rad
      unswept, 4.20 at 35 deg sweep, 3.73 at 45 deg.
    - Thin-airfoil theory (Anderson, "Fundamentals of Aerodynamics", ex. 4.6):
      NACA 2412 alpha_L0 = -2.077 deg, cm_c/4 = -0.053; cm about c/4 is
      constant in attached flow.
    - Flat plate at high incidence: cl = 2 sin a cos a, cd = 2 sin^2 a.
"""

import math

import pytest

from aerospace_mcp.integrations import aero
from aerospace_mcp.integrations.aero import (
    AIRFOIL_DATABASE,
    WingGeometry,
    airfoil_polar_analysis,
    calculate_stability_derivatives,
    get_airfoil_coefficients,
    normalize_airfoil_name,
    wing_vlm_analysis,
)


@pytest.fixture(autouse=True)
def builtin_models(monkeypatch):
    """Exercise the built-in models even when AeroSandbox is installed."""
    monkeypatch.setattr(aero, "AEROSANDBOX_AVAILABLE", False)


def _rect(span=8.0, airfoil="NACA0012", **kw):
    return WingGeometry(
        span_m=span,
        chord_root_m=1.0,
        chord_tip_m=1.0,
        airfoil_root=airfoil,
        airfoil_tip=kw.pop("airfoil_tip", airfoil),
        **kw,
    )


def _slope(geometry, mach=0.0, a=(0.0, 4.0)):
    r = wing_vlm_analysis(geometry, list(a), mach)
    return (r[1].CL - r[0].CL) / math.radians(a[1] - a[0])


def _helmbold(A, mach=0.0, sweep_c2_deg=0.0, kappa=6.28 / (2 * math.pi)):
    beta = math.sqrt(1 - mach**2)
    t2 = math.tan(math.radians(sweep_c2_deg)) ** 2
    return (
        2
        * math.pi
        * A
        / (2 + math.sqrt((A * beta / kappa) ** 2 * (1 + t2 / beta**2) + 4))
    )


class TestWingLiftSlope:
    @pytest.mark.parametrize(
        "sweep,expected", [(0.0, 4.91), (35.0, 4.20), (45.0, 3.73)]
    )
    def test_helmbold_diederich_sweep(self, sweep, expected):
        """Sweep used to be ignored (4.85 / rad at any sweep)."""
        assert _slope(_rect(sweep_deg=sweep)) == pytest.approx(expected, abs=0.01)

    @pytest.mark.parametrize("mach", [0.3, 0.6, 0.7])
    def test_compressibility_inside_formula(self, mach):
        """beta enters Helmbold-Diederich (not a 1/beta factor on the slope)."""
        assert _slope(_rect(), mach) == pytest.approx(_helmbold(8.0, mach), rel=1e-6)

    def test_taper_uses_half_chord_sweep(self):
        """Tapered wing: quarter-chord sweep converted to half-chord sweep."""
        g = WingGeometry(
            span_m=10.0,
            chord_root_m=2.0,
            chord_tip_m=1.0,
            sweep_deg=30.0,
            airfoil_root="NACA0012",
            airfoil_tip="NACA0012",
        )
        A = 10.0**2 / 15.0
        tan_c2 = math.tan(math.radians(30.0)) - (1 / A) * (1 - 0.5) / (1 + 0.5)
        expected = _helmbold(A, sweep_c2_deg=math.degrees(math.atan(tan_c2)))
        assert _slope(g) == pytest.approx(expected, rel=1e-6)

    def test_dihedral_cos_squared(self):
        ratio = _slope(_rect(dihedral_deg=10.0)) / _slope(_rect())
        assert ratio == pytest.approx(math.cos(math.radians(10.0)) ** 2, rel=1e-9)

    @pytest.mark.parametrize("mach", [0.9, 0.95, 1.2, 2.0])
    def test_transonic_and_supersonic_rejected(self, mach):
        """M >= 1 used to give a 10x lift slope (beta clamped to 0.1)."""
        with pytest.raises(ValueError, match="subsonic"):
            wing_vlm_analysis(_rect(), [0.0, 4.0], mach)
        with pytest.raises(ValueError, match="subsonic"):
            calculate_stability_derivatives(_rect(), 2.0, mach)
        with pytest.raises(ValueError, match="subsonic"):
            airfoil_polar_analysis("NACA0012", [0.0], 1e6, mach)

    def test_high_subsonic_warns(self):
        r = wing_vlm_analysis(_rect(), [2.0], 0.8)
        assert any("not modelled" in w for w in r[0].warnings)


class TestWingLift:
    def test_cambered_zero_alpha_lift_is_3d(self):
        """CL(0) = CL_alpha * (-alpha_0L), not the 2-D cl0 (0.25)."""
        r = wing_vlm_analysis(_rect(airfoil="NACA2412"), [0.0], 0.0)
        alpha_l0 = -0.25 / 6.28
        assert r[0].CL == pytest.approx(_helmbold(8.0) * -alpha_l0, rel=1e-6)
        assert r[0].CL == pytest.approx(0.195, abs=0.003)

    def test_linear_twist_shifts_zero_lift_angle(self):
        """Washout raises alpha_0L by |eps| * eta_mac (0.5 for a rectangle)."""
        base = wing_vlm_analysis(_rect(), [4.0], 0.0)[0].CL
        washed = wing_vlm_analysis(_rect(twist_deg=-4.0), [4.0], 0.0)[0].CL
        assert base - washed == pytest.approx(
            _helmbold(8.0) * math.radians(2.0), rel=1e-6
        )

    def test_tip_airfoil_matters(self):
        same = wing_vlm_analysis(_rect(airfoil="NACA2412"), [4.0], 0.2)[0].CL
        mixed = wing_vlm_analysis(
            _rect(airfoil="NACA2412", airfoil_tip="NACA0012"), [4.0], 0.2
        )[0].CL
        assert mixed < same

    def test_stall_bounded_and_smooth(self):
        """NACA4412 wing used to reach CL 1.87 (> 2-D cl_max 1.7), CD jumped
        0.172 -> 0.279 between 17 and 17.5 deg and CD was 27.6 at 90 deg."""
        alphas = [a * 0.25 for a in range(0, 361)]  # 0 .. 90 deg
        r = wing_vlm_analysis(_rect(airfoil="NACA4412"), alphas, 0.2)
        cl = [p.CL for p in r]
        cd = [p.CD for p in r]
        assert max(cl) <= 0.9 * 1.7 + 1e-9  # CL_max = 0.9 cl_max cos(sweep)
        # Continuous in 0.25 deg steps (the post-stall lift drop of ~1.0 is
        # spread over the 10 deg blend)
        assert max(abs(b - a) for a, b in zip(cd, cd[1:], strict=False)) < 0.03
        assert max(abs(b - a) for a, b in zip(cl, cl[1:], strict=False)) < 0.05
        # Finite flat plate at 90 deg: CD_max = 1.11 + 0.018 A, CL ~ 0
        assert cd[-1] == pytest.approx(1.11 + 0.018 * 8 + 1.1 * 0.008, abs=1e-3)
        assert abs(cl[-1]) < 1e-9
        assert any("stall" in w for w in r[-1].warnings)


class TestPitchingMoment:
    @pytest.mark.parametrize("x_ref", [0.1, 0.25, 0.4])
    def test_wing_and_stability_tools_agree(self, x_ref):
        """dCM/dCL from the wing analysis equals CM_alpha / CL_alpha, and both
        equal x_ref - 0.25 (aerodynamic centre at the MAC quarter chord)."""
        g = WingGeometry(
            span_m=9.0,
            chord_root_m=1.8,
            chord_tip_m=1.2,
            sweep_deg=5.0,
            airfoil_root="NACA2412",
            airfoil_tip="NACA2412",
        )
        r = wing_vlm_analysis(g, [0.0, 4.0], 0.2, x_ref_mac=x_ref)
        dcm_dcl = (r[1].CM - r[0].CM) / (r[1].CL - r[0].CL)
        s = calculate_stability_derivatives(g, 2.0, 0.2, x_ref_mac=x_ref)
        assert dcm_dcl == pytest.approx(x_ref - 0.25, abs=1e-9)
        assert s.CM_alpha / s.CL_alpha == pytest.approx(x_ref - 0.25, abs=1e-6)

    def test_forward_reference_is_stable(self):
        s = calculate_stability_derivatives(_rect(), 2.0, 0.2, x_ref_mac=0.15)
        assert s.CM_alpha < 0

    def test_washout_on_swept_wing_is_nose_up(self):
        base = WingGeometry(
            span_m=10.0,
            chord_root_m=2.0,
            chord_tip_m=1.0,
            sweep_deg=30.0,
            airfoil_root="NACA0012",
            airfoil_tip="NACA0012",
        )
        washed = base.model_copy(update={"twist_deg": -4.0})
        assert wing_vlm_analysis(base, [0.0], 0.2)[0].CM == pytest.approx(
            0.0, abs=1e-12
        )
        assert wing_vlm_analysis(washed, [0.0], 0.2)[0].CM > 0

    def test_section_moment_constant_about_quarter_chord(self):
        """Thin-airfoil theory: cm_c/4 is constant (was cm0 + 0.01 cl)."""
        r = airfoil_polar_analysis("NACA2412", [-4, 0, 4, 8], 1e6, 0.0)
        assert all(p.cm == pytest.approx(-0.05) for p in r)


class TestStabilityDerivatives:
    def test_alpha_matters_near_stall(self):
        """Identical results at 4 and 14 deg used to be returned."""
        low = calculate_stability_derivatives(_rect(), 4.0, 0.0)
        high = calculate_stability_derivatives(_rect(), 14.0, 0.0)
        assert low.CL_alpha == pytest.approx(_helmbold(8.0), rel=1e-6)
        assert high.CL_alpha < 0.8 * low.CL_alpha
        post = calculate_stability_derivatives(_rect(), 25.0, 0.0)
        assert post.stalled is True
        assert low.stalled is False

    def test_alpha_rate_derivatives_documented_as_not_computed(self):
        s = calculate_stability_derivatives(_rect())
        assert s.CL_alpha_dot is None and s.CM_alpha_dot is None
        assert any("not computed" in n for n in s.notes)


class TestAirfoilPolar:
    @pytest.mark.parametrize("name", list(AIRFOIL_DATABASE))
    def test_cl_never_exceeds_cl_max(self, name):
        """NACA0012 used to reach 1.644 (cl_max 1.4), NACA2412 2.00 (1.6)."""
        alphas = [a * 0.5 for a in range(-60, 181)]  # -30 .. 90 deg
        r = airfoil_polar_analysis(name, alphas, 1e6, 0.0)
        cl_max = AIRFOIL_DATABASE[name]["cl_max"]
        assert max(p.cl for p in r) <= cl_max + 1e-9
        assert max(p.cl for p in r) == pytest.approx(cl_max, abs=0.01)
        # Continuous: no jumps between 0.5 deg steps
        cl = [p.cl for p in r]
        cd = [p.cd for p in r]
        assert max(abs(b - a) for a, b in zip(cl, cl[1:], strict=False)) < 0.08
        assert max(abs(b - a) for a, b in zip(cd, cd[1:], strict=False)) < 0.05

    def test_flat_plate_post_stall(self):
        """L/D used to be 60.9 at 25 deg; cl 1.97, cd 0.045 at 90 deg."""
        r = {
            p.alpha_deg: p
            for p in airfoil_polar_analysis("NACA0012", [25, 45, 90], 1e6, 0.0)
        }
        assert r[25].cl_cd_ratio < 5
        assert r[45].cl == pytest.approx(1.0, abs=1e-6)  # 2 sin a cos a
        assert r[90].cl == pytest.approx(0.0, abs=1e-9)
        assert r[90].cd == pytest.approx(2.0 + 0.006, abs=1e-6)  # 2 sin^2 a + cd0
        assert r[90].cm < 0  # centre of pressure moves to mid-chord

    @pytest.mark.parametrize("re", [1e5, 1e6, 1e7])
    def test_reynolds_affects_drag_only(self, re):
        """The lift slope reached 7.54 / rad (> 2 pi) at Re = 1e7."""
        r = airfoil_polar_analysis("NACA0012", [0.0, 5.0], re, 0.0)
        assert (r[1].cl - r[0].cl) / math.radians(5) == pytest.approx(6.28, rel=1e-9)
        assert r[0].cd == pytest.approx(0.006 * (1e6 / re) ** 0.2, rel=1e-9)

    @pytest.mark.parametrize("re", [0.0, -1e6])
    def test_non_positive_reynolds_rejected(self, re):
        with pytest.raises(ValueError, match="Reynolds"):
            airfoil_polar_analysis("NACA0012", [0.0], re, 0.1)

    def test_empty_alpha_list(self):
        assert airfoil_polar_analysis("NACA0012", [], 1e6, 0.1) == []


class TestAirfoilNames:
    def test_normalization(self):
        assert normalize_airfoil_name(" naca 2412") == "NACA2412"
        assert normalize_airfoil_name("NACA-2412") == "NACA2412"
        assert normalize_airfoil_name("Clark Y") == "CLARKY"
        lower = airfoil_polar_analysis("naca0012", [5.0], 1e6, 0.0)[0]
        upper = airfoil_polar_analysis("NACA0012", [5.0], 1e6, 0.0)[0]
        assert lower.cl == upper.cl
        # lower-case 'naca0012' used to become NACA2412 (CL(0) = 0.25)
        assert wing_vlm_analysis(_rect(airfoil="naca0012"), [0.0], 0.2)[0].CL == 0.0

    def test_unknown_name_rejected(self):
        with pytest.raises(ValueError, match="Unknown airfoil"):
            get_airfoil_coefficients("XYZ123")

    def test_naca_4_digit_thin_airfoil_theory(self):
        """Non-database NACA 4-digit sections from thin-airfoil theory."""
        c = aero._naca4_coefficients(2, 4, 12)  # NACA 2412
        alpha_l0_deg = math.degrees(-c["cl0"] / c["cl_alpha"])
        assert alpha_l0_deg == pytest.approx(-2.077, abs=0.005)
        assert c["cm0"] == pytest.approx(-0.053, abs=0.001)
        sym = get_airfoil_coefficients("naca0015")
        assert sym["cl0"] == 0.0 and sym["cm0"] == 0.0
        assert 1.3 < sym["cl_max"] < 1.6


class TestOptionalBackends:
    def test_machupx_not_reported(self):
        from aerospace_mcp.integrations import get_domain_status

        assert aero.MACHUPX_AVAILABLE is False
        assert "aero_machupx" not in get_domain_status()["aero"]["libraries"]

    def test_aerosandbox_failure_is_reported(self, monkeypatch):
        """A failing AeroSandbox path is logged and reported, not swallowed."""

        class _BrokenAsb:
            """Every AeroSandbox call raises."""

            def __getattr__(self, name):
                def _raise(*a, **k):
                    raise RuntimeError("asb exploded")

                return _raise

        monkeypatch.setattr(aero, "AEROSANDBOX_AVAILABLE", True)
        monkeypatch.setattr(aero, "asb", _BrokenAsb(), raising=False)

        r = wing_vlm_analysis(_rect(), [0.0, 4.0], 0.2)
        assert r[0].method == "lifting_line_datcom"
        assert any("asb exploded" in w for w in r[0].warnings)

        p = airfoil_polar_analysis("NACA0012", [0.0, 4.0], 1e6, 0.1)
        assert p[0].method == "database_thin_airfoil"
        assert any("NeuralFoil failed" in w for w in p[0].warnings)
