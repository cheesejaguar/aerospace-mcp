//! Two-body orbital mechanics: Kepler's equation, classical elements <->
//! Cartesian state, derived orbit properties, RK4 propagation with J2,
//! ground tracks, and Hohmann transfers.
//!
//! Ported from the original pure-Python `aerospace_mcp.integrations.orbits`
//! with the same algorithms (Vallado, *Fundamentals of Astrodynamics and
//! Applications*, 4th ed.); the gravitational parameter is now an argument
//! instead of being fixed to Earth.

use std::f64::consts::PI;
use std::fmt;

use crate::vec3::{Vec3, cross, dot, norm};

// Powers that were `x ** 3` / `x ** 5` in Python use `powf` (C `pow`, as
// CPython does) rather than repeated multiplication, keeping results
// rounding-identical to the original implementation.

/// Earth's gravitational parameter (m^3/s^2), IERS 2010.
pub const MU_EARTH: f64 = 3.986004418e14;
/// WGS-84 equatorial radius (m).
pub const R_EARTH: f64 = 6378137.0;
/// Earth's J2 zonal harmonic (EGM-96 / JGM-3).
pub const J2_EARTH: f64 = 1.08262668e-3;
/// Earth's sidereal rotation rate (rad/s).
pub const OMEGA_EARTH: f64 = 7.2921159e-5;

fn deg_to_rad(deg: f64) -> f64 {
    deg * PI / 180.0
}

fn rad_to_deg(rad: f64) -> f64 {
    rad * 180.0 / PI
}

/// Invalid orbital-mechanics inputs.
#[derive(Debug, Clone, PartialEq)]
pub struct OrbitError(pub String);

impl fmt::Display for OrbitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for OrbitError {}

fn err<T>(msg: &str) -> Result<T, OrbitError> {
    Err(OrbitError(msg.to_string()))
}

/// Classical orbital elements (angles in degrees).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Elements {
    pub semi_major_axis_m: f64,
    pub eccentricity: f64,
    pub inclination_deg: f64,
    pub raan_deg: f64,
    pub arg_periapsis_deg: f64,
    pub true_anomaly_deg: f64,
}

/// Solve Kepler's equation `M = E - e sin E` for `E` by Newton-Raphson.
///
/// `M` is reduced to `[-pi, pi]` before iterating (the result is shifted
/// back by the same multiple of `2*pi`); without that, high-eccentricity
/// cases with large or negative `M` diverge.
pub fn kepler_solve(
    mean_anomaly_rad: f64,
    eccentricity: f64,
    tolerance: f64,
    max_iterations: usize,
) -> f64 {
    let turns = ((mean_anomaly_rad + PI) / (2.0 * PI)).floor();
    let offset = turns * 2.0 * PI;
    let mean_anomaly_rad = mean_anomaly_rad - offset;
    let mut e_anom = if eccentricity < 0.8 {
        mean_anomaly_rad
    } else {
        PI.copysign(mean_anomaly_rad)
    };
    for _ in 0..max_iterations {
        let f = e_anom - eccentricity * e_anom.sin() - mean_anomaly_rad;
        let f_prime = 1.0 - eccentricity * e_anom.cos();
        if f_prime.abs() < 1e-12 {
            break;
        }
        let next = e_anom - f / f_prime;
        if (next - e_anom).abs() < tolerance {
            return next + offset;
        }
        e_anom = next;
    }
    e_anom + offset
}

/// Classical elements -> inertial `(position_m, velocity_ms)`.
///
/// Perifocal position/velocity rotated by `R3(-RAAN) R1(-i) R3(-omega)`
/// (Vallado Algorithm 10). Elliptic (`a > 0, 0 <= e < 1`) and hyperbolic
/// (`a < 0, e > 1`) orbits are supported; parabolic orbits (`e == 1`) and
/// true anomalies beyond a hyperbola's asymptotes are rejected rather than
/// producing NaN/infinite states.
pub fn elements_to_state(el: &Elements, mu: f64) -> Result<(Vec3, Vec3), OrbitError> {
    let values = [
        el.semi_major_axis_m,
        el.eccentricity,
        el.inclination_deg,
        el.raan_deg,
        el.arg_periapsis_deg,
        el.true_anomaly_deg,
        mu,
    ];
    if values.iter().any(|x| !x.is_finite()) {
        return err("orbital elements must be finite");
    }
    if !(mu > 0.0) {
        return err("gravitational parameter must be > 0");
    }
    if el.eccentricity < 0.0 {
        return err("eccentricity must be >= 0");
    }
    if !(el.semi_major_axis_m * (1.0 - el.eccentricity.powi(2)) > 0.0) {
        return err(
            "invalid conic: need a > 0 with 0 <= e < 1 (ellipse) or a < 0 with e > 1 \
             (hyperbola); parabolic orbits (e = 1) are not supported",
        );
    }
    if !(1.0 + el.eccentricity * deg_to_rad(el.true_anomaly_deg).cos() > 0.0) {
        return err("true anomaly lies beyond the hyperbola's asymptotes");
    }

    let i = deg_to_rad(el.inclination_deg);
    let raan = deg_to_rad(el.raan_deg);
    let arg_pe = deg_to_rad(el.arg_periapsis_deg);
    let nu = deg_to_rad(el.true_anomaly_deg);
    let a = el.semi_major_axis_m;
    let e = el.eccentricity;

    let r = a * (1.0 - e.powi(2)) / (1.0 + e * nu.cos());
    let r_peri = [r * nu.cos(), r * nu.sin(), 0.0];
    let p = a * (1.0 - e.powi(2));
    let v_peri = [
        -(mu / p).sqrt() * nu.sin(),
        (mu / p).sqrt() * (e + nu.cos()),
        0.0,
    ];

    let (cos_raan, sin_raan) = (raan.cos(), raan.sin());
    let (cos_i, sin_i) = (i.cos(), i.sin());
    let (cos_arg, sin_arg) = (arg_pe.cos(), arg_pe.sin());

    let rot = [
        [
            cos_raan * cos_arg - sin_raan * sin_arg * cos_i,
            -cos_raan * sin_arg - sin_raan * cos_arg * cos_i,
            sin_raan * sin_i,
        ],
        [
            sin_raan * cos_arg + cos_raan * sin_arg * cos_i,
            -sin_raan * sin_arg + cos_raan * cos_arg * cos_i,
            -cos_raan * sin_i,
        ],
        [sin_arg * sin_i, cos_arg * sin_i, cos_i],
    ];
    let apply = |v: Vec3| -> Vec3 {
        [
            rot[0][0] * v[0] + rot[0][1] * v[1] + rot[0][2] * v[2],
            rot[1][0] * v[0] + rot[1][1] * v[1] + rot[1][2] * v[2],
            rot[2][0] * v[0] + rot[2][1] * v[1] + rot[2][2] * v[2],
        ]
    };
    Ok((apply(r_peri), apply(v_peri)))
}

/// Angle from `a` to `b` in `[0, 2*pi)`, measured counter-clockwise about
/// `h`. Uses `atan2`, which unlike `acos` keeps full precision near 0 and pi.
fn angle_about(a: Vec3, b: Vec3, h: Vec3) -> f64 {
    let sin_part = dot(cross(a, b), h) / norm(h);
    // `+ 0.0` turns a -0.0 result into 0.0.
    sin_part.atan2(dot(a, b)).rem_euclid(2.0 * PI) + 0.0
}

/// Inertial state -> classical elements (Vallado Algorithm 9).
///
/// Undefined angles are reported as zero: RAAN for equatorial orbits and
/// the argument of periapsis for circular ones. Equatorial orbits report the
/// longitude of periapsis as the argument of periapsis; circular orbits
/// measure the true anomaly from the node (or the x-axis when also
/// equatorial).
pub fn state_to_elements(r_vec: Vec3, v_vec: Vec3, mu: f64) -> Result<Elements, OrbitError> {
    if r_vec.iter().chain(&v_vec).any(|x| !x.is_finite()) {
        return err("state vector must be finite");
    }
    if !(mu > 0.0 && mu.is_finite()) {
        return err("gravitational parameter must be > 0");
    }
    let r = norm(r_vec);
    let v = norm(v_vec);
    if r == 0.0 {
        return err("position vector must be non-zero");
    }
    let h_vec = cross(r_vec, v_vec);
    let h = norm(h_vec);
    if h == 0.0 {
        return err("degenerate state: zero angular momentum (rectilinear motion)");
    }

    let energy = v.powi(2) / 2.0 - mu / r;
    if energy == 0.0 {
        return err("parabolic state: semi-major axis is undefined");
    }
    let a = -mu / (2.0 * energy);

    let v_cross_h = cross(v_vec, h_vec);
    let e_vec: Vec3 = std::array::from_fn(|k| v_cross_h[k] / mu - r_vec[k] / r);
    let e = norm(e_vec);

    let n_vec = cross([0.0, 0.0, 1.0], h_vec);
    let n = norm(n_vec);
    // Equatorial when the node vector is negligible relative to |h|
    // (n = |h| sin i); an absolute threshold misclassifies i = 180 deg.
    let equatorial = n <= 1e-11 * h;

    let i = n.atan2(h_vec[2]);
    let raan = if equatorial {
        0.0
    } else {
        n_vec[1].atan2(n_vec[0]).rem_euclid(2.0 * PI) + 0.0
    };

    // Equatorial orbits have no node; there the "argument of periapsis" is
    // the longitude of periapsis (measured in the direction of motion, so it
    // round-trips through `elements_to_state` with RAAN = 0).
    let sense = h_vec[2].signum();
    let arg_pe = if e <= 1e-10 {
        0.0
    } else if !equatorial {
        angle_about(n_vec, e_vec, h_vec)
    } else {
        (sense * e_vec[1]).atan2(e_vec[0]).rem_euclid(2.0 * PI) + 0.0
    };

    let nu = if e > 1e-10 {
        angle_about(e_vec, r_vec, h_vec)
    } else if !equatorial {
        // Circular inclined: argument of latitude, from the node.
        angle_about(n_vec, r_vec, h_vec)
    } else {
        // Circular equatorial: true longitude.
        (sense * r_vec[1]).atan2(r_vec[0]).rem_euclid(2.0 * PI) + 0.0
    };

    Ok(Elements {
        semi_major_axis_m: a,
        eccentricity: e,
        inclination_deg: rad_to_deg(i),
        raan_deg: rad_to_deg(raan),
        arg_periapsis_deg: rad_to_deg(arg_pe),
        true_anomaly_deg: rad_to_deg(nu),
    })
}

/// Derived orbit properties.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Properties {
    pub period_s: f64,
    /// Apoapsis altitude above the body's surface (m).
    pub apoapsis_m: f64,
    /// Periapsis altitude above the body's surface (m).
    pub periapsis_m: f64,
    pub energy_j_kg: f64,
    pub angular_momentum_m2s: f64,
}

/// Period, apsis altitudes, energy and angular momentum of an elliptical
/// orbit (`a > 0`, `0 <= e < 1`); other conics have no period and are
/// rejected.
pub fn orbit_properties(
    semi_major_axis_m: f64,
    eccentricity: f64,
    mu: f64,
    body_radius_m: f64,
) -> Result<Properties, OrbitError> {
    let (a, e) = (semi_major_axis_m, eccentricity);
    if [a, e, mu, body_radius_m].iter().any(|x| !x.is_finite()) {
        return err("orbit parameters must be finite");
    }
    if !(a > 0.0 && (0.0..1.0).contains(&e) && mu > 0.0) {
        return err("orbit properties require an elliptical orbit (a > 0, 0 <= e < 1)");
    }
    Ok(Properties {
        period_s: 2.0 * PI * (a.powf(3.0) / mu).sqrt(),
        apoapsis_m: a * (1.0 + e) - body_radius_m,
        periapsis_m: a * (1.0 - e) - body_radius_m,
        energy_j_kg: -mu / (2.0 * a),
        angular_momentum_m2s: (mu * a * (1.0 - e.powi(2))).sqrt(),
    })
}

/// Central-body plus J2 acceleration (Vallado, ch. 8):
///
/// ```text
/// a_J2 = -(3/2) J2 mu R^2 / r^5 * [x (1 - 5 z^2/r^2), y (1 - 5 z^2/r^2), z (3 - 5 z^2/r^2)]
/// ```
///
/// The original Python implementation omitted the leading minus sign, which
/// made the node of prograde orbits advance instead of regress.
fn acceleration_j2(r_vec: Vec3, mu: f64, j2: f64, r_eq: f64) -> Vec3 {
    let r = norm(r_vec);
    let r3 = r.powf(3.0);
    let factor = -1.5 * j2 * mu * r_eq.powi(2) / r.powf(5.0);
    let z2_r2 = (r_vec[2] / r).powi(2);
    [
        -mu * r_vec[0] / r3 + factor * r_vec[0] * (1.0 - 5.0 * z2_r2),
        -mu * r_vec[1] / r3 + factor * r_vec[1] * (1.0 - 5.0 * z2_r2),
        -mu * r_vec[2] / r3 + factor * r_vec[2] * (3.0 - 5.0 * z2_r2),
    ]
}

/// One propagated sample: `(seconds since start, position_m, velocity_ms)`.
pub type StateSample = (f64, Vec3, Vec3);

/// Gravity model for [`propagate`].
#[derive(Debug, Clone, Copy)]
pub struct Gravity {
    pub mu: f64,
    pub j2: f64,
    pub equatorial_radius_m: f64,
}

impl Gravity {
    pub const EARTH_J2: Gravity = Gravity {
        mu: MU_EARTH,
        j2: J2_EARTH,
        equatorial_radius_m: R_EARTH,
    };
}

/// Fixed-step RK4 propagation. The last step is shortened to land exactly
/// on `time_span_s`; the initial state is included as the first sample.
pub fn propagate(
    r0: Vec3,
    v0: Vec3,
    time_span_s: f64,
    time_step_s: f64,
    gravity: Gravity,
) -> Result<Vec<StateSample>, OrbitError> {
    if !(time_step_s >= 0.1) {
        return err("time_step_s must be >= 0.1 s");
    }
    if !(time_span_s > 0.0 && time_span_s <= 30.0 * 86400.0) {
        return err("time_span_s must be > 0 and <= 30 days (2592000 s)");
    }
    if time_span_s / time_step_s > 100_000.0 {
        return err("too many integration steps (max 100000); increase time_step_s");
    }

    let Gravity {
        mu,
        j2,
        equatorial_radius_m: r_eq,
    } = gravity;
    let accel = |r: Vec3| acceleration_j2(r, mu, j2, r_eq);
    let lin = |base: Vec3, h: f64, k: Vec3| -> Vec3 { std::array::from_fn(|i| base[i] + h * k[i]) };

    let mut samples = Vec::with_capacity((time_span_s / time_step_s).ceil() as usize + 2);
    samples.push((0.0, r0, v0));
    let (mut r, mut v) = (r0, v0);
    // A remainder this small is accumulated rounding in `t`, not a real
    // step; stepping it would emit a near-duplicate final sample.
    let negligible = 1e-9 * time_step_s;
    let mut t = 0.0;
    while time_span_s - t > negligible {
        let dt = time_step_s.min(time_span_s - t);

        let (k1_r, k1_v) = (v, accel(r));
        let (r2, v2) = (lin(r, 0.5 * dt, k1_r), lin(v, 0.5 * dt, k1_v));
        let (k2_r, k2_v) = (v2, accel(r2));
        let (r3, v3) = (lin(r, 0.5 * dt, k2_r), lin(v, 0.5 * dt, k2_v));
        let (k3_r, k3_v) = (v3, accel(r3));
        let (r4, v4) = (lin(r, dt, k3_r), lin(v, dt, k3_v));
        let (k4_r, k4_v) = (v4, accel(r4));

        r = std::array::from_fn(|i| {
            r[i] + dt / 6.0 * (k1_r[i] + 2.0 * k2_r[i] + 2.0 * k3_r[i] + k4_r[i])
        });
        v = std::array::from_fn(|i| {
            v[i] + dt / 6.0 * (k1_v[i] + 2.0 * k2_v[i] + 2.0 * k3_v[i] + k4_v[i])
        });
        t += dt;
        samples.push((t, r, v));
    }
    if let Some(last) = samples.last_mut() {
        last.0 = time_span_s;
    }
    Ok(samples)
}

/// Most samples [`ground_track`] accepts: one more than [`propagate`]'s
/// step cap, since propagation also returns the initial state.
pub const MAX_GROUND_TRACK_POINTS: usize = 100_001;

/// Sub-satellite `(latitude_deg, longitude_deg, altitude_m)` for inertial
/// positions at `elapsed_s` seconds after the first sample, correcting
/// longitude for Earth rotation (precession/nutation ignored) and a
/// spherical Earth.
pub fn ground_track(
    positions: &[Vec3],
    elapsed_s: &[f64],
) -> Result<Vec<(f64, f64, f64)>, OrbitError> {
    if positions.len() != elapsed_s.len() {
        return err("need one elapsed time per position");
    }
    if positions.len() > MAX_GROUND_TRACK_POINTS {
        return err("too many orbit states (max 100001)");
    }
    if elapsed_s.iter().any(|t| !t.is_finite()) {
        return err("elapsed times must be finite");
    }
    positions
        .iter()
        .zip(elapsed_s)
        .map(|(&r_vec, &t)| {
            let r = norm(r_vec);
            if !(r > 0.0 && r.is_finite()) {
                return err("position vector must be finite and non-zero");
            }
            let lat = (r_vec[2] / r).asin();
            let lon = r_vec[1].atan2(r_vec[0]) - OMEGA_EARTH * t;
            // Wrap into [-pi, pi).
            let lon = (lon + PI).rem_euclid(2.0 * PI) - PI;
            Ok((rad_to_deg(lat), rad_to_deg(lon), r - R_EARTH))
        })
        .collect()
}

/// Hohmann transfer between coplanar circular orbits of radius `r1`, `r2`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Hohmann {
    /// Signed departure burn (positive = prograde).
    pub delta_v1_ms: f64,
    /// Signed arrival burn (positive = prograde).
    pub delta_v2_ms: f64,
    pub total_delta_v_ms: f64,
    pub transfer_time_s: f64,
    pub transfer_semi_major_axis_m: f64,
}

pub fn hohmann(r1_m: f64, r2_m: f64, mu: f64) -> Result<Hohmann, OrbitError> {
    if !(r1_m > 0.0 && r2_m > 0.0) {
        return err("orbit radii must be > 0");
    }
    let a_t = (r1_m + r2_m) / 2.0;
    let v1_circ = (mu / r1_m).sqrt();
    let v2_circ = (mu / r2_m).sqrt();
    let v1_t = (mu * (2.0 / r1_m - 1.0 / a_t)).sqrt();
    let v2_t = (mu * (2.0 / r2_m - 1.0 / a_t)).sqrt();
    let dv1 = v1_t - v1_circ;
    let dv2 = v2_circ - v2_t;
    Ok(Hohmann {
        delta_v1_ms: dv1,
        delta_v2_ms: dv2,
        total_delta_v_ms: dv1.abs() + dv2.abs(),
        transfer_time_s: PI * (a_t.powf(3.0) / mu).sqrt(),
        transfer_semi_major_axis_m: a_t,
    })
}

/// Two-body state after `dt` seconds from `(r, v)` via Kepler's equation
/// (elliptic orbits only); an independent reference for the RK4 propagator.
#[cfg(test)]
fn kepler_propagate(r: Vec3, v: Vec3, dt: f64, mu: f64) -> Result<(Vec3, Vec3), OrbitError> {
    let mut el = state_to_elements(r, v, mu)?;
    let e = el.eccentricity;
    if e >= 1.0 {
        return err("kepler_propagate supports elliptic orbits only");
    }
    let nu = deg_to_rad(el.true_anomaly_deg);
    let e_anom =
        2.0 * ((1.0 - e).sqrt() * (nu / 2.0).sin()).atan2((1.0 + e).sqrt() * (nu / 2.0).cos());
    let m0 = e_anom - e * e_anom.sin();
    let n = (mu / el.semi_major_axis_m.powf(3.0)).sqrt();
    let m = (m0 + n * dt).rem_euclid(2.0 * PI);
    let e1 = kepler_solve(m, e, 1e-14, 100);
    let nu1 =
        2.0 * ((1.0 + e).sqrt() * (e1 / 2.0).sin()).atan2((1.0 - e).sqrt() * (e1 / 2.0).cos());
    el.true_anomaly_deg = rad_to_deg(nu1);
    elements_to_state(&el, mu)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, rel: f64) -> bool {
        (a - b).abs() <= rel * a.abs().max(b.abs()).max(1.0)
    }

    #[test]
    fn kepler_equation_residual_is_small() {
        for &e in &[0.0, 0.3, 0.85, 0.97] {
            for &m in &[0.1, 1.0, 3.0, 5.5] {
                let ea = kepler_solve(m, e, 1e-12, 50);
                assert!((ea - e * ea.sin() - m).abs() < 1e-9, "e={e} M={m}");
            }
        }
    }

    #[test]
    fn element_state_round_trip() {
        let el = Elements {
            semi_major_axis_m: 2.6e7,
            eccentricity: 0.7,
            inclination_deg: 63.4,
            raan_deg: 200.0,
            arg_periapsis_deg: 270.0,
            true_anomaly_deg: 170.0,
        };
        let (r, v) = elements_to_state(&el, MU_EARTH).unwrap();
        let back = state_to_elements(r, v, MU_EARTH).unwrap();
        assert!(close(back.semi_major_axis_m, el.semi_major_axis_m, 1e-9));
        assert!(close(back.eccentricity, el.eccentricity, 1e-9));
        assert!(close(back.inclination_deg, el.inclination_deg, 1e-9));
        assert!(close(back.raan_deg, el.raan_deg, 1e-9));
        assert!(close(back.arg_periapsis_deg, el.arg_periapsis_deg, 1e-9));
        assert!(close(back.true_anomaly_deg, el.true_anomaly_deg, 1e-9));
    }

    #[test]
    fn equatorial_eccentric_orbits_round_trip() {
        for inclination_deg in [0.0, 180.0] {
            let el = Elements {
                semi_major_axis_m: 9.0e6,
                eccentricity: 0.2,
                inclination_deg,
                raan_deg: 0.0,
                arg_periapsis_deg: 40.0,
                true_anomaly_deg: 100.0,
            };
            let (r, v) = elements_to_state(&el, MU_EARTH).unwrap();
            let back = state_to_elements(r, v, MU_EARTH).unwrap();
            let (r2, v2) = elements_to_state(&back, MU_EARTH).unwrap();
            for k in 0..3 {
                assert!((r[k] - r2[k]).abs() < 1e-6 * 9.0e6, "i={inclination_deg}");
                assert!((v[k] - v2[k]).abs() < 1e-6 * 1e4, "i={inclination_deg}");
            }
            assert!(
                close(back.arg_periapsis_deg, 40.0, 1e-9),
                "i={inclination_deg}: {back:?}"
            );
        }
    }

    #[test]
    fn degenerate_states_are_rejected() {
        assert!(state_to_elements([0.0; 3], [1.0, 0.0, 0.0], MU_EARTH).is_err());
        assert!(state_to_elements([7e6, 0.0, 0.0], [1.0, 0.0, 0.0], MU_EARTH).is_err());
    }

    #[test]
    fn two_body_propagation_matches_kepler() {
        let (r0, v0) = ([7.0e6, 0.0, 1.0e5], [0.0, 6000.0, 4500.0]);
        let two_body = Gravity {
            j2: 0.0,
            ..Gravity::EARTH_J2
        };
        let samples = propagate(r0, v0, 5000.0, 1.0, two_body).unwrap();
        let (t, r, _) = *samples.last().unwrap();
        assert_eq!(t, 5000.0);
        let (rk, _) = kepler_propagate(r0, v0, 5000.0, MU_EARTH).unwrap();
        for k in 0..3 {
            assert!((r[k] - rk[k]).abs() < 1e-3, "{r:?} vs {rk:?}");
        }
    }

    #[test]
    fn propagation_bounds() {
        let g = Gravity::EARTH_J2;
        let (r, v) = ([7.0e6, 0.0, 0.0], [0.0, 7546.0, 0.0]);
        assert!(propagate(r, v, 3600.0, 0.01, g).is_err());
        assert!(propagate(r, v, 3600.0, f64::NAN, g).is_err());
        assert!(propagate(r, v, 0.0, 60.0, g).is_err());
        assert!(propagate(r, v, 31.0 * 86400.0, 60.0, g).is_err());
        assert!(propagate(r, v, 86400.0, 0.5, g).is_err());
        assert_eq!(propagate(r, v, 3600.0, 60.0, g).unwrap().len(), 61);
        // Uneven final step lands exactly on the span.
        let s = propagate(r, v, 100.0, 30.0, g).unwrap();
        assert_eq!(s.len(), 5);
        assert_eq!(s[4].0, 100.0);
    }

    #[test]
    fn j2_regresses_the_node_of_a_prograde_orbit() {
        let el = Elements {
            semi_major_axis_m: 7.0e6,
            eccentricity: 0.001,
            inclination_deg: 51.6,
            raan_deg: 30.0,
            arg_periapsis_deg: 0.0,
            true_anomaly_deg: 0.0,
        };
        let (r, v) = elements_to_state(&el, MU_EARTH).unwrap();
        let samples = propagate(r, v, 86400.0, 30.0, Gravity::EARTH_J2).unwrap();
        let (_, rf, vf) = *samples.last().unwrap();
        let raan_f = state_to_elements(rf, vf, MU_EARTH).unwrap().raan_deg;
        // Secular rate: -1.5 n J2 (R/p)^2 cos i ~ -5 deg/day for this orbit.
        let drift = raan_f - 30.0;
        assert!((-5.5..-4.5).contains(&drift), "drift {drift}");
    }

    #[test]
    fn ground_track_wraps_longitude() {
        let pts = ground_track(
            &[[7e6, 0.0, 0.0], [-7e6, 1e-9, 0.0], [0.0, 0.0, 7e6]],
            &[0.0, 60.0, 120.0],
        )
        .unwrap();
        for (lat, lon, _) in &pts {
            assert!((-90.0..=90.0).contains(lat));
            assert!((-180.0..180.0).contains(lon));
        }
        assert!((pts[2].0 - 90.0).abs() < 1e-12);
        assert!(ground_track(&[[7e6, 0.0, 0.0]], &[]).is_err());
        assert!(ground_track(&[[7e6, 0.0, 0.0]], &[f64::NAN]).is_err());
        // Huge times used to spin the Python while-loop normalization forever.
        assert!(ground_track(&[[7e6, 0.0, 0.0]], &[1e300]).is_ok());
    }

    #[test]
    fn ground_track_uses_actual_sample_times() {
        // 100 s at 30 s steps: the last sample is at t = 100, not 4 * 30.
        let samples = propagate(
            [7e6, 0.0, 0.0],
            [0.0, 7546.0, 0.0],
            100.0,
            30.0,
            Gravity::EARTH_J2,
        )
        .unwrap();
        let positions: Vec<Vec3> = samples.iter().map(|s| s.1).collect();
        let times: Vec<f64> = samples.iter().map(|s| s.0).collect();
        assert_eq!(times, vec![0.0, 30.0, 60.0, 90.0, 100.0]);
        let track = ground_track(&positions, &times).unwrap();
        let (_, last_lon, _) = track[4];
        let p = positions[4];
        let expected = (p[1].atan2(p[0]) - OMEGA_EARTH * 100.0).to_degrees();
        assert!(
            (last_lon - expected).abs() < 1e-9,
            "{last_lon} vs {expected}"
        );
        // A full-length propagation (100k steps + initial state) is accepted.
        let positions = vec![[7e6, 0.0, 0.0]; MAX_GROUND_TRACK_POINTS];
        let times = vec![0.0; MAX_GROUND_TRACK_POINTS];
        assert!(ground_track(&positions, &times).is_ok());
    }

    #[test]
    fn propagation_does_not_emit_rounding_level_final_steps() {
        // 10 x 0.1 accumulates to 0.9999999999999999, which used to add an
        // 11th step of 1e-16 s.
        let s = propagate(
            [7e6, 0.0, 0.0],
            [0.0, 7546.0, 0.0],
            1.0,
            0.1,
            Gravity::EARTH_J2,
        )
        .unwrap();
        assert_eq!(s.len(), 11);
        assert_eq!(s[10].0, 1.0);
    }

    #[test]
    fn invalid_conics_are_rejected_instead_of_returning_nan() {
        let base = Elements {
            semi_major_axis_m: 7e6,
            eccentricity: 0.1,
            inclination_deg: 30.0,
            raan_deg: 0.0,
            arg_periapsis_deg: 0.0,
            true_anomaly_deg: 0.0,
        };
        let with = |a: f64, e: f64, nu: f64| Elements {
            semi_major_axis_m: a,
            eccentricity: e,
            true_anomaly_deg: nu,
            ..base
        };
        assert!(elements_to_state(&with(7e6, 1.5, 0.0), MU_EARTH).is_err()); // a > 0, e > 1
        assert!(elements_to_state(&with(7e6, 1.0, 0.0), MU_EARTH).is_err()); // parabolic
        assert!(elements_to_state(&with(7e6, -0.1, 0.0), MU_EARTH).is_err());
        assert!(elements_to_state(&with(f64::NAN, 0.1, 0.0), MU_EARTH).is_err());
        // Hyperbola: asymptote at acos(-1/1.5) = 131.8 deg.
        assert!(elements_to_state(&with(-7e6, 1.5, 150.0), MU_EARTH).is_err());
        let (r, v) = elements_to_state(&with(-7e6, 1.5, 100.0), MU_EARTH).unwrap();
        let back = state_to_elements(r, v, MU_EARTH).unwrap();
        assert!(close(back.eccentricity, 1.5, 1e-9) && close(back.semi_major_axis_m, -7e6, 1e-9));

        assert!(orbit_properties(-7e6, 1.5, MU_EARTH, R_EARTH).is_err());
        assert!(orbit_properties(7e6, 1.0, MU_EARTH, R_EARTH).is_err());
        assert!(state_to_elements([f64::NAN, 0.0, 0.0], [0.0, 7e3, 0.0], MU_EARTH).is_err());
    }

    #[test]
    fn retrograde_equatorial_is_treated_as_equatorial() {
        // sin(pi) is 1.2e-16, which an absolute node threshold called inclined.
        let el = Elements {
            semi_major_axis_m: 9e6,
            eccentricity: 0.2,
            inclination_deg: 180.0,
            raan_deg: 30.0,
            arg_periapsis_deg: 20.0,
            true_anomaly_deg: 60.0,
        };
        let (r, v) = elements_to_state(&el, MU_EARTH).unwrap();
        let back = state_to_elements(r, v, MU_EARTH).unwrap();
        assert_eq!(back.raan_deg, 0.0);
        let (r2, v2) = elements_to_state(&back, MU_EARTH).unwrap();
        for k in 0..3 {
            assert!((r[k] - r2[k]).abs() < 1e-6, "{r:?} vs {r2:?}");
            assert!((v[k] - v2[k]).abs() < 1e-9);
        }
    }

    #[test]
    fn small_angles_keep_precision() {
        // acos lost ~1e-8 rad here (1e-6 deg came back as 1.479e-6 deg).
        let el = Elements {
            semi_major_axis_m: 7e6,
            eccentricity: 0.01,
            inclination_deg: 1e-6,
            raan_deg: 1e-6,
            arg_periapsis_deg: 1e-6,
            true_anomaly_deg: 1e-6,
        };
        let (r, v) = elements_to_state(&el, MU_EARTH).unwrap();
        let back = state_to_elements(r, v, MU_EARTH).unwrap();
        assert!((back.inclination_deg - 1e-6).abs() < 1e-12, "{back:?}");
        let (r2, _) = elements_to_state(&back, MU_EARTH).unwrap();
        for k in 0..3 {
            assert!((r[k] - r2[k]).abs() < 1e-5, "{r:?} vs {r2:?}");
        }
    }

    #[test]
    fn kepler_handles_large_and_negative_mean_anomaly() {
        for &(m, e) in &[(10.0, 0.999), (-3.0, 0.99), (100.0, 0.5), (-50.0, 0.97)] {
            let ea = kepler_solve(m, e, 1e-12, 50);
            assert!((ea - e * ea.sin() - m).abs() < 1e-9, "M={m} e={e}: E={ea}");
        }
    }

    #[test]
    fn hohmann_leo_to_geo() {
        let h = hohmann(6.778e6, 4.2164e7, MU_EARTH).unwrap();
        assert!(close(h.total_delta_v_ms, 3854.0, 1e-3));
        assert!(close(h.transfer_time_s / 3600.0, 5.29, 1e-3));
        assert!(hohmann(0.0, 1.0, MU_EARTH).is_err());
    }
}
