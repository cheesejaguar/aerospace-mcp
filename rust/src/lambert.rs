//! Lambert's problem via Izzo's algorithm.
//!
//! Given two position vectors and a time of flight, find the two-body orbit
//! connecting them. Implements D. Izzo, "Revisiting Lambert's problem",
//! Celestial Mechanics and Dynamical Astronomy 121 (2015), following the
//! structure of the (now archived) poliastro implementation that
//! aerospace-mcp previously used, including its correction to the
//! single-revolution initial guess (poliastro issue #1362). One deliberate
//! difference: the Halley iteration that locates the minimum time of flight
//! for multi-revolution transfers re-evaluates T(x) at every iterate, as in
//! pykep, rather than reusing the starting value.
//!
//! Supports single- and multi-revolution transfers, prograde or retrograde
//! motion, and both solution branches for multi-revolution cases.

use std::f64::consts::PI;
use std::fmt;

use crate::vec3::{Vec3, add, cross, norm, scale, sub};

/// Options for [`izzo`].
#[derive(Debug, Clone, Copy)]
pub struct LambertOptions {
    /// Number of complete revolutions (0 = direct transfer).
    pub revolutions: u32,
    /// Counter-clockwise motion about +z (prograde) or clockwise.
    pub prograde: bool,
    /// For multi-revolution transfers, which of the two solutions to return
    /// (poliastro's `lowpath` convention: `true` takes the larger initial
    /// guess `x`). This is a branch selector, not "lowest energy"; compare
    /// both branches if energy matters. Ignored when `revolutions == 0`.
    pub low_path: bool,
    pub max_iterations: usize,
    pub tolerance: f64,
}

impl Default for LambertOptions {
    fn default() -> Self {
        Self {
            revolutions: 0,
            prograde: true,
            low_path: true,
            max_iterations: 35,
            tolerance: 1e-8,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum LambertError {
    InvalidInput(&'static str),
    /// r1 and r2 are parallel or anti-parallel, so the transfer plane is undefined.
    Collinear,
    /// Not enough time of flight for the requested number of revolutions.
    TooManyRevolutions {
        requested: u32,
        max_feasible: u32,
    },
    NoConvergence,
}

impl fmt::Display for LambertError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LambertError::InvalidInput(msg) => f.write_str(msg),
            LambertError::Collinear => f.write_str(
                "position vectors are collinear (0 or 180 degree transfer); \
                 the transfer plane is undefined",
            ),
            LambertError::TooManyRevolutions {
                requested,
                max_feasible,
            } => write!(
                f,
                "no solution with {requested} revolution(s) for this time of flight \
                 (maximum feasible: {max_feasible})"
            ),
            LambertError::NoConvergence => f.write_str("Lambert iteration failed to converge"),
        }
    }
}

impl std::error::Error for LambertError {}

/// A Lambert solution.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LambertSolution {
    /// Velocity at `r1`.
    pub v1: Vec3,
    /// Velocity at `r2`.
    pub v2: Vec3,
    /// Transfer angle actually traversed, in `(0, 2*pi)` radians, measured in
    /// the direction of motion (excluding complete revolutions).
    pub transfer_angle_rad: f64,
}

/// Solve Lambert's problem for the velocities at `r1` and `r2`.
///
/// Units are whatever `mu`, the positions and `tof` use consistently
/// (e.g. m^3/s^2, m, s -> m/s).
pub fn izzo(
    mu: f64,
    r1: Vec3,
    r2: Vec3,
    tof: f64,
    opts: LambertOptions,
) -> Result<LambertSolution, LambertError> {
    if !(mu > 0.0 && mu.is_finite()) {
        return Err(LambertError::InvalidInput(
            "gravitational parameter must be > 0",
        ));
    }
    if !(tof > 0.0 && tof.is_finite()) {
        return Err(LambertError::InvalidInput("time of flight must be > 0"));
    }
    if r1.iter().chain(&r2).any(|x| !x.is_finite()) {
        return Err(LambertError::InvalidInput(
            "position vectors must be finite",
        ));
    }
    let (r1_n, r2_n) = (norm(r1), norm(r2));
    if r1_n == 0.0 || r2_n == 0.0 {
        return Err(LambertError::InvalidInput(
            "position vectors must be non-zero",
        ));
    }

    let c = sub(r2, r1);
    let c_n = norm(c);
    let s = (r1_n + r2_n + c_n) * 0.5;

    let i_r1 = scale(r1, 1.0 / r1_n);
    let i_r2 = scale(r2, 1.0 / r2_n);
    let h = cross(i_r1, i_r2);
    let h_n = norm(h);
    if h_n <= 1e-12 {
        return Err(LambertError::Collinear);
    }
    let i_h = scale(h, 1.0 / h_n);

    // lambda: transfer geometry parameter, |lambda| < 1. The motion sense
    // is taken from the unnormalized r1 x r2: for transfer planes containing
    // the z-axis its z-component is exactly 0, while the normalized i_h[2]
    // can pick up a rounding-level sign that would flip the branch.
    let mut ll = (1.0 - (c_n / s).min(1.0)).sqrt();
    let (mut i_t1, mut i_t2) = if cross(r1, r2)[2] < 0.0 {
        ll = -ll;
        (cross(i_r1, i_h), cross(i_r2, i_h))
    } else {
        (cross(i_h, i_r1), cross(i_h, i_r2))
    };
    if !opts.prograde {
        ll = -ll;
        i_t1 = scale(i_t1, -1.0);
        i_t2 = scale(i_t2, -1.0);
    }

    // Non-dimensional time of flight.
    let t = (2.0 * mu / s.powi(3)).sqrt() * tof;
    let (x, y) = find_xy(ll, t, opts)?;

    let gamma = (mu * s / 2.0).sqrt();
    let rho = (r1_n - r2_n) / c_n;
    let sigma = (1.0 - rho.powi(2)).sqrt();

    let v_r1 = gamma * ((ll * y - x) - rho * (ll * y + x)) / r1_n;
    let v_r2 = -gamma * ((ll * y - x) + rho * (ll * y + x)) / r2_n;
    let v_t1 = gamma * sigma * (y + ll * x) / r1_n;
    let v_t2 = gamma * sigma * (y + ll * x) / r2_n;

    let v1 = add(scale(i_r1, v_r1), scale(i_t1, v_t1));
    let v2 = add(scale(i_r2, v_r2), scale(i_t2, v_t2));

    // lambda < 0 exactly when the motion goes the long way round.
    let short_angle = h_n.atan2(i_r1[0] * i_r2[0] + i_r1[1] * i_r2[1] + i_r1[2] * i_r2[2]);
    let transfer_angle_rad = if ll < 0.0 {
        2.0 * PI - short_angle
    } else {
        short_angle
    };
    Ok(LambertSolution {
        v1,
        v2,
        transfer_angle_rad,
    })
}

fn find_xy(ll: f64, t: f64, opts: LambertOptions) -> Result<(f64, f64), LambertError> {
    let m = opts.revolutions;
    let mut m_max = (t / PI).floor();
    let t_00 = ll.acos() + ll * (1.0 - ll.powi(2)).sqrt();

    // The floor estimate may admit one revolution too many.
    if t < t_00 + m_max * PI && m_max > 0.0 {
        let t_min = compute_t_min(ll, m_max as u32, opts)?;
        if t < t_min {
            m_max -= 1.0;
        }
    }
    if f64::from(m) > m_max {
        return Err(LambertError::TooManyRevolutions {
            requested: m,
            max_feasible: m_max.max(0.0) as u32,
        });
    }

    let x0 = initial_guess(t, ll, m, opts.low_path);
    let x = householder(x0, t, ll, m, opts)?;
    Ok((x, compute_y(x, ll)))
}

fn compute_y(x: f64, ll: f64) -> f64 {
    (1.0 - ll.powi(2) * (1.0 - x.powi(2))).sqrt()
}

fn compute_psi(x: f64, y: f64, ll: f64) -> f64 {
    if (-1.0..1.0).contains(&x) {
        // Elliptic.
        (x * y + ll * (1.0 - x.powi(2))).clamp(-1.0, 1.0).acos()
    } else if x > 1.0 {
        // Hyperbolic.
        ((y - x * ll) * (x.powi(2) - 1.0).sqrt()).asinh()
    } else {
        // Parabolic.
        0.0
    }
}

/// Gauss hypergeometric function 2F1(3, 1; 5/2; x) by its series (Battin).
fn hyp2f1b(x: f64) -> f64 {
    if x >= 1.0 {
        return f64::INFINITY;
    }
    let (mut res, mut term) = (1.0, 1.0);
    for ii in 0..10_000 {
        let k = f64::from(ii);
        term = term * (3.0 + k) * (1.0 + k) / (2.5 + k) * x / (k + 1.0);
        let previous = res;
        res += term;
        if res == previous {
            break;
        }
    }
    res
}

/// Non-dimensional time-of-flight residual T(x) - t0.
fn tof_equation_y(x: f64, y: f64, t0: f64, ll: f64, m: u32) -> f64 {
    let t = if m == 0 && 0.6_f64.sqrt() < x && x < 1.4_f64.sqrt() {
        // Battin series near the parabolic case, where Lancaster's form
        // loses precision.
        let eta = y - ll * x;
        let s1 = (1.0 - ll - x * eta) * 0.5;
        let q = 4.0 / 3.0 * hyp2f1b(s1);
        (eta.powi(3) * q + 4.0 * ll * eta) * 0.5
    } else {
        let psi = compute_psi(x, y, ll);
        ((psi + f64::from(m) * PI) / (1.0 - x.powi(2)).abs().sqrt() - x + ll * y)
            / (1.0 - x.powi(2))
    };
    t - t0
}

fn tof_equation(x: f64, t0: f64, ll: f64, m: u32) -> f64 {
    tof_equation_y(x, compute_y(x, ll), t0, ll, m)
}

/// dT/dx, d2T/dx2, d3T/dx3 at `x` given T(x) = `t`.
fn tof_derivatives(x: f64, y: f64, t: f64, ll: f64) -> (f64, f64, f64) {
    let umx2 = 1.0 - x.powi(2);
    let l2 = ll.powi(2);
    let l3 = l2 * ll;
    let d1 = (3.0 * t * x - 2.0 + 2.0 * l3 * x / y) / umx2;
    let d2 = (3.0 * t + 5.0 * x * d1 + 2.0 * (1.0 - l2) * l3 / y.powi(3)) / umx2;
    let d3 = (7.0 * x * d2 + 8.0 * d1 - 6.0 * (1.0 - l2) * l3 * l2 * x / y.powi(5)) / umx2;
    (d1, d2, d3)
}

/// Minimum non-dimensional time of flight for `m >= 1` revolutions.
fn compute_t_min(ll: f64, m: u32, opts: LambertOptions) -> Result<f64, LambertError> {
    if ll == 1.0 {
        return Ok(tof_equation(0.0, 0.0, ll, m));
    }
    if m == 0 {
        return Ok(0.0);
    }
    // Halley iterations on dT/dx = 0, starting at x > 0 to avoid ll = -1 issues.
    let mut x = 0.1;
    let mut t = tof_equation(x, 0.0, ll, m);
    for _ in 0..opts.max_iterations {
        let y = compute_y(x, ll);
        let (d1, d2, d3) = tof_derivatives(x, y, t, ll);
        if d2 == 0.0 {
            return Err(LambertError::NoConvergence);
        }
        let next = x - 2.0 * d1 * d2 / (2.0 * d2.powi(2) - d1 * d3);
        if !next.is_finite() {
            return Err(LambertError::NoConvergence);
        }
        if (next - x).abs() < opts.tolerance {
            return Ok(tof_equation(next, 0.0, ll, m));
        }
        x = next;
        t = tof_equation(x, 0.0, ll, m);
    }
    Err(LambertError::NoConvergence)
}

fn initial_guess(t: f64, ll: f64, m: u32, low_path: bool) -> f64 {
    if m == 0 {
        let t_0 = ll.acos() + ll * (1.0 - ll.powi(2)).sqrt();
        let t_1 = 2.0 * (1.0 - ll.powi(3)) / 3.0;
        if t >= t_0 {
            (t_0 / t).powf(2.0 / 3.0) - 1.0
        } else if t < t_1 {
            5.0 / 2.0 * t_1 / t * (t_1 - t) / (1.0 - ll.powi(5)) + 1.0
        } else {
            // Corrected form of the expression after Eq. (30) in the paper.
            ((2.0_f64).ln() * (t / t_0).ln() / (t_1 / t_0).ln()).exp() - 1.0
        }
    } else {
        let mpi = f64::from(m) * PI;
        let a = ((mpi + PI) / (8.0 * t)).powf(2.0 / 3.0);
        let x_0l = (a - 1.0) / (a + 1.0);
        let b = ((8.0 * t) / mpi).powf(2.0 / 3.0);
        let x_0r = (b - 1.0) / (b + 1.0);
        if low_path {
            x_0l.max(x_0r)
        } else {
            x_0l.min(x_0r)
        }
    }
}

/// Householder (quartic) iterations for T(x) = t0.
fn householder(
    mut x: f64,
    t0: f64,
    ll: f64,
    m: u32,
    opts: LambertOptions,
) -> Result<f64, LambertError> {
    for _ in 0..opts.max_iterations {
        let y = compute_y(x, ll);
        let fval = tof_equation_y(x, y, t0, ll, m);
        // Already on the solution: the derivatives are singular at x = 1
        // (exactly parabolic), so don't evaluate them.
        if fval.abs() <= 4.0 * f64::EPSILON * t0.abs() {
            return Ok(x);
        }
        let t = fval + t0;
        let (d1, d2, d3) = tof_derivatives(x, y, t, ll);
        let next = x - fval
            * ((d1.powi(2) - fval * d2 / 2.0)
                / (d1 * (d1.powi(2) - fval * d2) + d3 * fval.powi(2) / 6.0));
        if !next.is_finite() {
            return Err(LambertError::NoConvergence);
        }
        if (next - x).abs() < opts.tolerance {
            return Ok(next);
        }
        x = next;
    }
    Err(LambertError::NoConvergence)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::orbit::{Gravity, MU_EARTH, propagate};

    /// `(v1, v2)` view of [`super::izzo`], for the velocity-only tests.
    fn izzo(
        mu: f64,
        r1: Vec3,
        r2: Vec3,
        tof: f64,
        opts: LambertOptions,
    ) -> Result<(Vec3, Vec3), LambertError> {
        super::izzo(mu, r1, r2, tof, opts).map(|s| (s.v1, s.v2))
    }

    fn assert_vec_close(a: Vec3, b: Vec3, tol: f64) {
        for k in 0..3 {
            assert!((a[k] - b[k]).abs() <= tol, "{a:?} vs {b:?}");
        }
    }

    /// Curtis, *Orbital Mechanics for Engineering Students*, Example 5.2 (km, s).
    #[test]
    fn curtis_example_5_2() {
        let (v1, v2) = izzo(
            398600.0,
            [5000.0, 10000.0, 2100.0],
            [-14600.0, 2500.0, 7000.0],
            3600.0,
            LambertOptions::default(),
        )
        .unwrap();
        assert_vec_close(v1, [-5.9925, 1.9254, 3.2456], 1e-4);
        assert_vec_close(v2, [-3.3125, -4.1966, -0.38529], 1e-4);
    }

    /// Vallado, *Fundamentals of Astrodynamics and Applications*, Example 7-5 (km, s).
    #[test]
    fn vallado_example_7_5() {
        let (v1, v2) = izzo(
            3.986004418e5,
            [15945.34, 0.0, 0.0],
            [12214.83899, 10249.46731, 0.0],
            76.0 * 60.0,
            LambertOptions::default(),
        )
        .unwrap();
        assert_vec_close(v1, [2.058913, 2.915965, 0.0], 1e-5);
        assert_vec_close(v2, [-3.451565, 0.910315, 0.0], 1e-5);
    }

    /// Propagating (r1, v1) for the time of flight must arrive at r2 with v2.
    /// Uses fine-step two-body RK4 so hyperbolic solutions are covered too.
    fn check_round_trip(r1: Vec3, r2: Vec3, tof: f64, opts: LambertOptions) {
        let (v1, v2) = izzo(MU_EARTH, r1, r2, tof, opts).unwrap();
        let two_body = Gravity {
            j2: 0.0,
            ..Gravity::EARTH_J2
        };
        let samples = propagate(r1, v1, tof, (tof / 50_000.0).max(0.1), two_body).unwrap();
        let (_, r_end, v_end) = *samples.last().unwrap();
        assert_vec_close(r_end, r2, 1e-6 * norm(r2));
        assert_vec_close(v_end, v2, 1e-6 * norm(v2));
    }

    #[test]
    fn round_trips_across_geometries() {
        let r1 = [7.0e6, 0.0, 0.0];
        check_round_trip(r1, [0.0, 8.0e6, 0.0], 3600.0, LambertOptions::default());
        // Retrograde (270 degree) transfer.
        check_round_trip(
            r1,
            [0.0, 7.0e6, 0.0],
            3000.0,
            LambertOptions {
                prograde: false,
                ..LambertOptions::default()
            },
        );
        // Out-of-plane, short and long times of flight.
        for tof in [600.0, 2000.0, 9000.0] {
            check_round_trip(r1, [1.0e6, 6.0e6, 3.0e6], tof, LambertOptions::default());
        }
    }

    #[test]
    fn multi_revolution_branches() {
        let (r1, r2) = ([7.0e6, 0.0, 0.0], [0.0, 7.5e6, 1.0e6]);
        let tof = 4.0 * 3600.0;
        for low_path in [true, false] {
            check_round_trip(
                r1,
                r2,
                tof,
                LambertOptions {
                    revolutions: 1,
                    low_path,
                    ..LambertOptions::default()
                },
            );
        }
        let (low, _) = izzo(
            MU_EARTH,
            r1,
            r2,
            tof,
            LambertOptions {
                revolutions: 1,
                ..LambertOptions::default()
            },
        )
        .unwrap();
        let (high, _) = izzo(
            MU_EARTH,
            r1,
            r2,
            tof,
            LambertOptions {
                revolutions: 1,
                low_path: false,
                ..LambertOptions::default()
            },
        )
        .unwrap();
        assert!(
            (norm(low) - norm(high)).abs() > 1.0,
            "branches should differ"
        );
        assert!(matches!(
            izzo(
                MU_EARTH,
                r1,
                r2,
                tof,
                LambertOptions {
                    revolutions: 5,
                    ..LambertOptions::default()
                }
            ),
            Err(LambertError::TooManyRevolutions { requested: 5, .. })
        ));
    }

    #[test]
    fn rejects_degenerate_inputs() {
        let r1 = [7.0e6, 0.0, 0.0];
        let opts = LambertOptions::default();
        assert_eq!(
            izzo(MU_EARTH, r1, [-7.0e6, 0.0, 0.0], 2700.0, opts),
            Err(LambertError::Collinear)
        );
        assert_eq!(
            izzo(MU_EARTH, r1, [8.0e6, 0.0, 0.0], 2700.0, opts),
            Err(LambertError::Collinear)
        );
        assert!(izzo(MU_EARTH, r1, [0.0, 7.0e6, 0.0], 0.0, opts).is_err());
        assert!(izzo(0.0, r1, [0.0, 7.0e6, 0.0], 10.0, opts).is_err());
        assert!(izzo(MU_EARTH, [0.0; 3], [0.0, 7.0e6, 0.0], 10.0, opts).is_err());
    }

    #[test]
    fn hyperbolic_short_transfer() {
        // Very short time of flight forces a hyperbolic solution.
        let r1 = [7.0e6, 0.0, 0.0];
        let r2 = [0.0, 7.0e6, 0.0];
        let (v1, _) = izzo(MU_EARTH, r1, r2, 300.0, LambertOptions::default()).unwrap();
        let energy = norm(v1).powi(2) / 2.0 - MU_EARTH / 7.0e6;
        assert!(energy > 0.0);
    }

    #[test]
    fn reports_the_transfer_angle_it_solved() {
        let r1 = [7.0e6, 0.0, 0.0];
        let pro = super::izzo(
            MU_EARTH,
            r1,
            [0.0, 7.0e6, 0.0],
            3000.0,
            LambertOptions::default(),
        )
        .unwrap();
        assert!((pro.transfer_angle_rad - PI / 2.0).abs() < 1e-12);
        let retro = super::izzo(
            MU_EARTH,
            r1,
            [0.0, 7.0e6, 0.0],
            3000.0,
            LambertOptions {
                prograde: false,
                ..LambertOptions::default()
            },
        )
        .unwrap();
        assert!((retro.transfer_angle_rad - 1.5 * PI).abs() < 1e-12);
    }

    /// Transfer plane containing the z-axis: (r1 x r2)_z is exactly 0, but
    /// the normalized normal's z picks up a -5.6e-17 rounding error that
    /// used to flip "prograde" onto the long (351 degree) way.
    #[test]
    fn plane_through_z_axis_takes_the_short_way() {
        let r1 = [17e6, 13e6, 739122.8681162545];
        let r2 = [17e6, 13e6, -2495907.938505691];
        let sol = super::izzo(MU_EARTH, r1, r2, 3000.0, LambertOptions::default()).unwrap();
        assert!(sol.transfer_angle_rad < PI, "{}", sol.transfer_angle_rad);
        assert!((norm(sol.v1) - 1649.96).abs() < 0.1, "{:?}", sol.v1);
        check_round_trip(r1, r2, 3000.0, LambertOptions::default());
    }

    /// Exactly the parabolic time of flight makes the first guess x = 1,
    /// where the time-of-flight derivatives are singular.
    #[test]
    fn exactly_parabolic_time_of_flight() {
        let (r1, r2) = ([7e6, 0.0, 0.0], [0.0, 8e6, 0.0]);
        let tof = 1006.9374781471273;
        let (v1, _) = izzo(MU_EARTH, r1, r2, tof, LambertOptions::default()).unwrap();
        let energy = norm(v1).powi(2) / 2.0 - MU_EARTH / 7e6;
        assert!(energy.abs() < 1e-3 * MU_EARTH / 7e6, "energy {energy}");
        check_round_trip(r1, r2, tof, LambertOptions::default());
    }
}
