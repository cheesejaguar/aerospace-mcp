//! 3-DOF rocket ascent simulation and performance analysis.
//!
//! Ported from the original pure-Python `aerospace_mcp.integrations.rockets`;
//! the integrator and force model are unchanged from that implementation:
//!
//! - Forward Euler integration, flat Earth, constant gravity.
//! - Thrust along the launch angle for the first 5 s, vertical afterwards.
//! - Drag `D = Cd * A_ref * q` opposing the velocity, `q = 0.5 * rho * V^2`.
//! - Density and speed of sound interpolated from an atmosphere table built
//!   on the Python side (so the ISA / ambiance model stays single-sourced).
//!
//! Three parts of the model were corrected after the port, so results no
//! longer match the Python original for thrusted flights (see
//! `tests/test_native_parity.py`):
//!
//! - Thrust and propellant flow come from the exact integral of the
//!   interpolated thrust curve over each step ([`ThrustProfile`]), so the
//!   impulse delivered equals the curve's impulse however its knots fall on
//!   the time grid.
//! - Mass flow is proportional to thrust, `mdot = F * m_prop / I_total`
//!   (constant effective exhaust velocity), so exactly the propellant mass
//!   is consumed by the end of the burn. The original used a constant
//!   `m_prop / t_burn` with `t_burn` the last sample with positive thrust,
//!   which emptied the tank after 1 s for curves like `[[0, F], [6, 0]]`.
//! - Above the atmosphere table, density decays exponentially from the top
//!   row and the speed of sound stays at the top row's value. The original
//!   restarted from sea-level formulas there, so density and Mach number
//!   jumped by a factor of ~3 at the table top.

use std::f64::consts::PI;
use std::fmt;

use crate::interp::{interp, interp_pairs};

/// Standard gravitational acceleration (m/s^2).
pub const G0: f64 = 9.80665;

/// Thrust is applied along the launch angle for this long, then vertically.
const GUIDED_PHASE_S: f64 = 5.0;

/// Rocket mass and aerodynamic properties needed by the simulation.
#[derive(Debug, Clone)]
pub struct Geometry {
    pub dry_mass_kg: f64,
    pub propellant_mass_kg: f64,
    pub diameter_m: f64,
    pub cd: f64,
    /// `(time_s, thrust_n)` samples; empty means no thrust.
    pub thrust_curve: Vec<(f64, f64)>,
}

/// Density scale height (m) used above the atmosphere table when the table's
/// top interval cannot provide one (a single row, or density not decreasing
/// there). 7 km is typical of the 50-80 km layer where such tables end.
pub const FALLBACK_SCALE_HEIGHT_M: f64 = 7000.0;

/// Tabulated atmosphere used for density / speed-of-sound lookups.
///
/// Inside the table, values are interpolated linearly. Above it, density
/// continues from the top row as `rho_top * exp(-(h - h_top) / H)` with `H`
/// the local density scale height of the table's top interval,
/// `(h_n - h_{n-1}) / ln(rho_{n-1} / rho_n)` (about 6 km for an ISA table
/// ending at 80 km), and the speed of sound stays at the top row's value.
/// Both are continuous at the table top, so max-Q and max-Mach do not jump
/// when a trajectory leaves the table. Below the first row the first row's
/// values are used.
#[derive(Debug, Clone)]
pub struct AtmosphereTable {
    altitudes_m: Vec<f64>,
    densities_kg_m3: Vec<f64>,
    speeds_of_sound_ms: Vec<f64>,
    scale_height_m: f64,
}

impl AtmosphereTable {
    pub fn new(
        altitudes_m: Vec<f64>,
        densities_kg_m3: Vec<f64>,
        speeds_of_sound_ms: Vec<f64>,
    ) -> Result<Self, String> {
        if altitudes_m.is_empty() {
            return Err("atmosphere table must not be empty".into());
        }
        if altitudes_m.len() != densities_kg_m3.len()
            || altitudes_m.len() != speeds_of_sound_ms.len()
        {
            return Err("atmosphere table columns must have equal length".into());
        }
        if altitudes_m.iter().any(|h| !h.is_finite()) {
            return Err("atmosphere table altitudes must be finite".into());
        }
        if altitudes_m.windows(2).any(|w| !(w[0] < w[1])) {
            return Err("atmosphere table altitudes must be strictly increasing".into());
        }
        if densities_kg_m3
            .iter()
            .chain(&speeds_of_sound_ms)
            .any(|v| !(v.is_finite() && *v > 0.0))
        {
            return Err("atmosphere table densities and speeds of sound must be > 0".into());
        }

        let n = altitudes_m.len();
        let scale_height_m = if n >= 2 && densities_kg_m3[n - 2] > densities_kg_m3[n - 1] {
            (altitudes_m[n - 1] - altitudes_m[n - 2])
                / (densities_kg_m3[n - 2] / densities_kg_m3[n - 1]).ln()
        } else {
            FALLBACK_SCALE_HEIGHT_M
        };
        Ok(Self {
            altitudes_m,
            densities_kg_m3,
            speeds_of_sound_ms,
            scale_height_m,
        })
    }

    fn top(&self) -> usize {
        self.altitudes_m.len() - 1
    }

    /// `(density, speed_of_sound)` at `altitude_m`.
    pub fn lookup(&self, altitude_m: f64) -> (f64, f64) {
        let top = self.top();
        let h_top = self.altitudes_m[top];
        if altitude_m <= h_top {
            (
                interp(altitude_m, &self.altitudes_m, &self.densities_kg_m3),
                interp(altitude_m, &self.altitudes_m, &self.speeds_of_sound_ms),
            )
        } else {
            let rho =
                self.densities_kg_m3[top] * (-(altitude_m - h_top) / self.scale_height_m).exp();
            (rho, self.speeds_of_sound_ms[top])
        }
    }
}

/// One sample of the trajectory time history.
///
/// State (time, altitude, velocity, mass) is at the start of a step; the
/// forces and the acceleration are the ones applied over that step.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrajectoryPoint {
    pub time_s: f64,
    pub altitude_m: f64,
    pub velocity_ms: f64,
    pub acceleration_ms2: f64,
    pub mass_kg: f64,
    /// Mean thrust of the curve over the step starting at `time_s`.
    pub thrust_n: f64,
    pub drag_n: f64,
    pub mach: f64,
    pub dynamic_pressure_pa: f64,
}

/// Aggregate metrics extracted from a trajectory.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Performance {
    pub max_altitude_m: f64,
    pub apogee_time_s: f64,
    pub max_velocity_ms: f64,
    pub max_mach: f64,
    pub max_q_pa: f64,
    pub burnout_altitude_m: f64,
    pub burnout_velocity_ms: f64,
    pub burnout_time_s: f64,
    /// Impulse the integrator applied: each sample's thrust held until the
    /// next sample.
    pub total_impulse_ns: f64,
    /// `total_impulse / (propellant consumed * g0)`, both over the whole
    /// trajectory.
    pub specific_impulse_s: f64,
}

/// Integration settings.
#[derive(Debug, Clone, Copy)]
pub struct SimOptions {
    pub dt_s: f64,
    pub max_time_s: f64,
    pub launch_angle_deg: f64,
}

impl Default for SimOptions {
    fn default() -> Self {
        Self {
            dt_s: 0.1,
            max_time_s: 300.0,
            launch_angle_deg: 90.0,
        }
    }
}

/// Invalid simulation settings (the messages match the Python originals).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimError {
    TimeStep,
    Duration,
    TooManySteps,
    LaunchAngle,
    Geometry(&'static str),
}

impl fmt::Display for SimError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            SimError::TimeStep => "dt_s must be between 0.001 and 10 s",
            SimError::Duration => "max_time_s must be > 0 and <= 3600 s",
            SimError::TooManySteps => "too many integration steps (max 500000); increase dt_s",
            SimError::LaunchAngle => "launch_angle_deg must be finite",
            SimError::Geometry(msg) => msg,
        })
    }
}

impl std::error::Error for SimError {}

/// Thrust at `time_s`, linearly interpolated from `curve`.
///
/// Zero before the first sample and at/after the last one. A single-sample
/// curve is treated as constant thrust up to (and including) its time.
pub fn thrust_at_time(curve: &[(f64, f64)], time_s: f64) -> f64 {
    match curve {
        [] => 0.0,
        [(t0, f0)] => {
            if time_s <= *t0 {
                *f0
            } else {
                0.0
            }
        }
        _ => {
            let (first, last) = (curve[0].0, curve[curve.len() - 1].0);
            if time_s < first || time_s >= last {
                return 0.0;
            }
            interp_pairs(time_s, curve)
        }
    }
}

/// Cumulative impulse of a thrust curve, `I(t)` = integral of
/// [`thrust_at_time`] from 0 to `t`.
///
/// Thrust before `t = 0` is never simulated, so it is not counted. Curve
/// times must be non-decreasing (as [`Geometry::validate`] requires);
/// duplicate times are steps, as in [`thrust_at_time`].
#[derive(Debug, Clone)]
pub struct ThrustProfile<'a> {
    curve: &'a [(f64, f64)],
    /// Integral of the curve from its first sample to each sample.
    prefix: Vec<f64>,
    /// The same integral up to `t = 0`, subtracted so `I(0) = 0`.
    before_zero: f64,
}

impl<'a> ThrustProfile<'a> {
    pub fn new(curve: &'a [(f64, f64)]) -> Self {
        let mut prefix = Vec::with_capacity(curve.len());
        let mut total = 0.0;
        for (i, &(t, f)) in curve.iter().enumerate() {
            if i > 0 {
                let (t0, f0) = curve[i - 1];
                total += 0.5 * (f0 + f) * (t - t0);
            }
            prefix.push(total);
        }
        let mut profile = Self {
            curve,
            prefix,
            before_zero: 0.0,
        };
        if curve.len() >= 2 {
            profile.before_zero = profile.integral_since_first_sample(0.0);
        }
        profile
    }

    /// Integral from the first sample to `t`, for curves of 2+ samples.
    fn integral_since_first_sample(&self, t: f64) -> f64 {
        let n = self.curve.len();
        if t <= self.curve[0].0 {
            return 0.0;
        }
        if t >= self.curve[n - 1].0 {
            return self.prefix[n - 1];
        }
        // t lies strictly inside segment j (so t1 > t0).
        let j = self.curve.partition_point(|&(ti, _)| ti <= t) - 1;
        let (t0, f0) = self.curve[j];
        let (t1, f1) = self.curve[j + 1];
        let f = f0 + (f1 - f0) * ((t - t0) / (t1 - t0));
        self.prefix[j] + 0.5 * (f0 + f) * (t - t0)
    }

    /// Impulse delivered from `t = 0` to `t` (N s); `t` must be >= 0.
    pub fn impulse_to(&self, t: f64) -> f64 {
        match self.curve {
            [] => 0.0,
            // Constant thrust up to the sample time.
            [(t0, f0)] => f0 * t.min(*t0).max(0.0),
            _ => self.integral_since_first_sample(t) - self.before_zero,
        }
    }

    /// Total impulse of the curve over `t >= 0` (N s).
    pub fn total_impulse(&self) -> f64 {
        match self.curve {
            [] => 0.0,
            [(t0, f0)] => f0 * t0.max(0.0),
            _ => self.prefix[self.curve.len() - 1] - self.before_zero,
        }
    }
}

impl Geometry {
    /// Reject inputs that would make the simulation divide by zero or
    /// produce NaN/infinite results.
    pub fn validate(&self) -> Result<(), SimError> {
        let finite = [
            self.dry_mass_kg,
            self.propellant_mass_kg,
            self.diameter_m,
            self.cd,
        ]
        .iter()
        .chain(self.thrust_curve.iter().flat_map(|(t, f)| [t, f]))
        .all(|x| x.is_finite());
        if !finite {
            return Err(SimError::Geometry("rocket geometry values must be finite"));
        }
        if !(self.dry_mass_kg > 0.0) {
            return Err(SimError::Geometry("dry_mass_kg must be > 0"));
        }
        if self.propellant_mass_kg < 0.0 {
            return Err(SimError::Geometry("propellant_mass_kg must be >= 0"));
        }
        if !(self.diameter_m > 0.0) {
            return Err(SimError::Geometry("diameter_m must be > 0"));
        }
        if self.cd < 0.0 {
            return Err(SimError::Geometry("cd must be >= 0"));
        }
        if self.thrust_curve.iter().any(|(_, f)| *f < 0.0) {
            return Err(SimError::Geometry(
                "thrust_curve thrust values must be >= 0",
            ));
        }
        if self.thrust_curve.windows(2).any(|w| w[1].0 < w[0].0) {
            return Err(SimError::Geometry(
                "thrust_curve times must be non-decreasing",
            ));
        }
        Ok(())
    }
}

/// Validate settings the same way the Python implementation did.
pub fn validate(opts: &SimOptions) -> Result<(), SimError> {
    if !(0.001..=10.0).contains(&opts.dt_s) {
        return Err(SimError::TimeStep);
    }
    if !(opts.max_time_s > 0.0 && opts.max_time_s <= 3600.0) {
        return Err(SimError::Duration);
    }
    if opts.max_time_s / opts.dt_s > 500_000.0 {
        return Err(SimError::TooManySteps);
    }
    if !opts.launch_angle_deg.is_finite() {
        return Err(SimError::LaunchAngle);
    }
    Ok(())
}

/// Simulate from launch to apogee (or ground impact / `max_time_s`).
pub fn simulate(
    geometry: &Geometry,
    atmosphere: &AtmosphereTable,
    opts: &SimOptions,
) -> Result<Vec<TrajectoryPoint>, SimError> {
    let mut out = Vec::new();
    simulate_into(geometry, atmosphere, opts, &mut out)?;
    Ok(out)
}

/// Like [`simulate`], but reuses `out`'s allocation (it is cleared first).
/// Optimizers call this in their inner loop.
pub fn simulate_into(
    geometry: &Geometry,
    atmosphere: &AtmosphereTable,
    opts: &SimOptions,
    out: &mut Vec<TrajectoryPoint>,
) -> Result<(), SimError> {
    validate(opts)?;
    geometry.validate()?;
    out.clear();

    let dt = opts.dt_s;
    let mut time = 0.0_f64;
    let mut altitude = 0.0_f64;
    let mut velocity_vertical = 0.0_f64;
    let mut velocity_horizontal = 0.0_f64;
    let mut mass = geometry.dry_mass_kg + geometry.propellant_mass_kg;

    // Same expression as numpy.radians: deg * (pi / 180).
    let launch_angle_rad = opts.launch_angle_deg * (PI / 180.0);
    let sin_launch = launch_angle_rad.sin();
    let cos_launch = launch_angle_rad.cos();

    let drag_area = PI * (geometry.diameter_m / 2.0).powi(2);

    // Each step applies the curve's mean thrust over the step, so the impulse
    // delivered telescopes to exactly I(t). Propellant flows in proportion to
    // thrust (mdot = F * m_prop / I_total, a constant effective exhaust
    // velocity), so the mass after impulse I is
    // m_dry + m_prop * (1 - I / I_total): the tank is empty exactly when the
    // curve ends.
    let profile = ThrustProfile::new(&geometry.thrust_curve);
    let total_impulse = profile.total_impulse();
    let mut impulse = 0.0_f64;

    while time <= opts.max_time_s && altitude >= 0.0 {
        let next_time = time + dt;
        let next_impulse = profile.impulse_to(next_time);
        // (Clamped: rounding in I(t) can leave -1e-12 N where the curve is 0.)
        let thrust = ((next_impulse - impulse) / dt).max(0.0);
        let (rho, speed_of_sound) = atmosphere.lookup(altitude);

        let velocity_total = (velocity_vertical.powi(2) + velocity_horizontal.powi(2)).sqrt();
        let mach = if speed_of_sound > 0.0 {
            velocity_total / speed_of_sound
        } else {
            0.0
        };
        let dynamic_pressure = 0.5 * rho * velocity_total.powi(2);
        let drag_force = geometry.cd * drag_area * dynamic_pressure;

        let (drag_accel_vertical, drag_accel_horizontal) = if velocity_total > 0.0 {
            (
                -drag_force * (velocity_vertical / velocity_total) / mass,
                -drag_force * (velocity_horizontal / velocity_total) / mass,
            )
        } else {
            (0.0, 0.0)
        };

        let (thrust_accel_vertical, thrust_accel_horizontal) = if time < GUIDED_PHASE_S {
            (thrust * sin_launch / mass, thrust * cos_launch / mass)
        } else {
            (thrust / mass, 0.0)
        };

        let accel_vertical = thrust_accel_vertical - G0 + drag_accel_vertical;
        let accel_horizontal = thrust_accel_horizontal + drag_accel_horizontal;

        out.push(TrajectoryPoint {
            time_s: time,
            altitude_m: altitude,
            velocity_ms: velocity_total,
            acceleration_ms2: (accel_vertical.powi(2) + accel_horizontal.powi(2)).sqrt(),
            mass_kg: mass,
            thrust_n: thrust,
            drag_n: drag_force,
            mach,
            dynamic_pressure_pa: dynamic_pressure,
        });

        // Forward Euler (velocity first, then position with the new velocity).
        velocity_vertical += accel_vertical * dt;
        velocity_horizontal += accel_horizontal * dt;
        altitude += velocity_vertical * dt;

        if total_impulse > 0.0 {
            let remaining = (1.0 - next_impulse / total_impulse).max(0.0);
            mass = geometry.dry_mass_kg + geometry.propellant_mass_kg * remaining;
        }

        impulse = next_impulse;
        time = next_time;

        // Past apogee and descending.
        if altitude > 0.0 && velocity_vertical < -1.0 {
            break;
        }
    }
    Ok(())
}

/// Extract apogee, max-Q, burnout and impulse metrics from a trajectory.
pub fn analyze(trajectory: &[TrajectoryPoint]) -> Performance {
    let Some(first) = trajectory.first() else {
        return Performance::default();
    };
    let n = trajectory.len();

    // First occurrence of each maximum, as numpy.max / argmax report.
    let mut apogee_idx = 0;
    let mut max_velocity = first.velocity_ms;
    let mut max_mach = first.mach;
    let mut max_q = first.dynamic_pressure_pa;
    let mut last_thrust_idx = None;
    let mut total_impulse = 0.0;
    for (i, p) in trajectory.iter().enumerate() {
        if p.altitude_m > trajectory[apogee_idx].altitude_m {
            apogee_idx = i;
        }
        max_velocity = max_velocity.max(p.velocity_ms);
        max_mach = max_mach.max(p.mach);
        max_q = max_q.max(p.dynamic_pressure_pa);
        if p.thrust_n > 100.0 {
            last_thrust_idx = Some(i);
        }
        // A sample's thrust acts until the next sample, as in the Euler
        // update, so this is the impulse the integrator applied.
        if i > 0 {
            let prev = &trajectory[i - 1];
            total_impulse += prev.thrust_n * (p.time_s - prev.time_s);
        }
    }

    // Burnout is the sample right after the last one with meaningful thrust.
    let burnout_idx = match last_thrust_idx {
        Some(i) => (i + 1).min(n - 1),
        None => n - 1,
    };
    let burnout = &trajectory[burnout_idx];

    // Propellant consumed over the same samples as the impulse.
    let propellant_used = first.mass_kg - trajectory[n - 1].mass_kg;
    let specific_impulse = if propellant_used > 0.0 {
        total_impulse / (propellant_used * G0)
    } else {
        0.0
    };

    Performance {
        max_altitude_m: trajectory[apogee_idx].altitude_m,
        apogee_time_s: trajectory[apogee_idx].time_s,
        max_velocity_ms: max_velocity,
        max_mach,
        max_q_pa: max_q,
        burnout_altitude_m: burnout.altitude_m,
        burnout_velocity_ms: burnout.velocity_ms,
        burnout_time_s: burnout.time_s,
        total_impulse_ns: total_impulse,
        specific_impulse_s: specific_impulse,
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Two-point exponential-ish table good enough for unit tests.
    pub(crate) fn test_atmosphere() -> AtmosphereTable {
        let altitudes: Vec<f64> = (0..=50).map(|k| k as f64 * 1000.0).collect();
        let densities = altitudes
            .iter()
            .map(|h| 1.225 * (-h / 8500.0).exp())
            .collect();
        let sos = altitudes.iter().map(|_| 340.0).collect();
        AtmosphereTable::new(altitudes, densities, sos).unwrap()
    }

    pub(crate) fn test_rocket() -> Geometry {
        Geometry {
            dry_mass_kg: 15.0,
            propellant_mass_kg: 60.0,
            diameter_m: 0.18,
            cd: 0.35,
            thrust_curve: vec![(0.0, 1800.0), (6.0, 1800.0), (6.1, 0.0)],
        }
    }

    #[test]
    fn thrust_curve_lookup() {
        let curve = [(0.0, 1000.0), (2.0, 3000.0), (4.0, 0.0)];
        assert_eq!(thrust_at_time(&curve, -0.1), 0.0);
        assert_eq!(thrust_at_time(&curve, 0.0), 1000.0);
        assert_eq!(thrust_at_time(&curve, 1.0), 2000.0);
        assert_eq!(thrust_at_time(&curve, 4.0), 0.0);
        assert_eq!(thrust_at_time(&[(1.0, 500.0)], 1.0), 500.0);
        assert_eq!(thrust_at_time(&[(1.0, 500.0)], 1.5), 0.0);
        assert_eq!(thrust_at_time(&[], 1.0), 0.0);
    }

    #[test]
    fn rejects_bad_options() {
        let (g, atm) = (test_rocket(), test_atmosphere());
        let bad = |dt_s, max_time_s| SimOptions {
            dt_s,
            max_time_s,
            launch_angle_deg: 90.0,
        };
        assert_eq!(simulate(&g, &atm, &bad(0.0, 10.0)), Err(SimError::TimeStep));
        assert_eq!(
            simulate(&g, &atm, &bad(f64::NAN, 10.0)),
            Err(SimError::TimeStep)
        );
        assert_eq!(simulate(&g, &atm, &bad(0.1, 0.0)), Err(SimError::Duration));
        assert_eq!(
            simulate(&g, &atm, &bad(0.001, 3600.0)),
            Err(SimError::TooManySteps)
        );
    }

    #[test]
    fn vertical_flight_reaches_apogee_and_conserves_impulse() {
        let traj = simulate(&test_rocket(), &test_atmosphere(), &SimOptions::default()).unwrap();
        let perf = analyze(&traj);
        assert!(perf.max_altitude_m > 1000.0);
        assert!(perf.apogee_time_s > perf.burnout_time_s);
        // 1800 N for 6 s plus the 0.1 s ramp-down, delivered exactly.
        assert!((perf.total_impulse_ns - 1800.0 * 6.05).abs() < 1e-6);
        // Mass never drops below dry mass.
        assert!(traj.iter().all(|p| p.mass_kg >= 15.0));
    }

    #[test]
    fn thrust_profile_integrates_the_interpolated_curve() {
        let curve = [(0.0, 1000.0), (2.0, 3000.0), (4.0, 0.0)];
        let p = ThrustProfile::new(&curve);
        assert_eq!(p.total_impulse(), 4000.0 + 3000.0);
        assert_eq!(p.impulse_to(0.0), 0.0);
        assert_eq!(p.impulse_to(1.0), 1500.0);
        assert_eq!(p.impulse_to(2.0), 4000.0);
        assert_eq!(p.impulse_to(10.0), 7000.0);

        // Duplicate knots are a step; samples before t = 0 don't count.
        let step = [(-1.0, 500.0), (1.0, 500.0), (1.0, 100.0), (3.0, 100.0)];
        let p = ThrustProfile::new(&step);
        assert_eq!(p.total_impulse(), 500.0 + 200.0);
        assert_eq!(p.impulse_to(2.0), 600.0);

        // One sample: constant thrust up to its time.
        assert_eq!(ThrustProfile::new(&[(4.0, 250.0)]).total_impulse(), 1000.0);
        assert_eq!(ThrustProfile::new(&[(4.0, 250.0)]).impulse_to(2.0), 500.0);
        assert_eq!(ThrustProfile::new(&[]).total_impulse(), 0.0);

        // Matches a fine midpoint sum of thrust_at_time.
        let curve = [(0.3, 40000.0), (2.0, 38000.0), (20.0, 30000.0), (25.0, 0.0)];
        let n = 250_000;
        let h = 25.0 / f64::from(n);
        let sum: f64 = (0..n)
            .map(|k| thrust_at_time(&curve, (f64::from(k) + 0.5) * h) * h)
            .sum();
        let total = ThrustProfile::new(&curve).total_impulse();
        assert!((sum - total).abs() < 1e-6 * total, "{sum} vs {total}");
    }

    /// Mass flow proportional to thrust: the tank is empty exactly at the
    /// end of the curve, whatever its shape, and Isp = I / (m_prop * g0).
    #[test]
    fn propellant_is_consumed_over_the_whole_burn() {
        let atm = test_atmosphere();
        for (curve, burn_end) in [
            (vec![(0.0, 10800.0), (6.0, 0.0)], 6.0),
            (vec![(0.0, 4000.0), (5.0, 0.0)], 5.0),
            (vec![(0.0, 1800.0), (6.0, 1800.0), (6.1, 0.0)], 6.1),
            (
                vec![(0.0, 900.0), (3.0, 2500.0), (3.0, 1000.0), (7.25, 0.0)],
                7.25,
            ),
        ] {
            let mut g = test_rocket();
            g.thrust_curve = curve;
            let i_total = ThrustProfile::new(&g.thrust_curve).total_impulse();
            for dt_s in [0.1, 0.03] {
                let opts = SimOptions {
                    dt_s,
                    ..SimOptions::default()
                };
                let traj = simulate(&g, &atm, &opts).unwrap();
                for p in &traj {
                    if p.time_s < burn_end - 1e-9 {
                        assert!(
                            p.mass_kg > 15.0,
                            "{:?} empty at {}",
                            g.thrust_curve,
                            p.time_s
                        );
                    } else {
                        assert!((p.mass_kg - 15.0).abs() < 1e-9, "{p:?}");
                    }
                }
                let perf = analyze(&traj);
                assert!((perf.total_impulse_ns - i_total).abs() < 1e-6 * i_total);
                let isp = i_total / (60.0 * G0);
                assert!((perf.specific_impulse_s - isp).abs() < 1e-9 * isp);
            }
        }
    }

    #[test]
    fn atmosphere_is_continuous_at_the_table_top() {
        let atm = test_atmosphere();
        let (rho_top, a_top) = atm.lookup(50_000.0);
        let (rho_above, a_above) = atm.lookup(50_000.0 + 1e-6);
        assert!((rho_above / rho_top - 1.0).abs() < 1e-9);
        assert_eq!(a_above, a_top);
        // The test table is exactly exponential with H = 8500 m, so the
        // extrapolation reproduces it.
        let (rho, _) = atm.lookup(70_000.0);
        assert!((rho / (1.225 * (-70_000.0_f64 / 8500.0).exp()) - 1.0).abs() < 1e-9);
        // A one-row table falls back to the documented scale height.
        let flat = AtmosphereTable::new(vec![0.0], vec![1.0], vec![300.0]).unwrap();
        let (rho, a) = flat.lookup(FALLBACK_SCALE_HEIGHT_M);
        assert!((rho - (-1.0_f64).exp()).abs() < 1e-12);
        assert_eq!(a, 300.0);
        assert!(AtmosphereTable::new(vec![0.0], vec![0.0], vec![300.0]).is_err());
    }

    #[test]
    fn no_thrust_stops_immediately() {
        let mut g = test_rocket();
        g.thrust_curve.clear();
        let traj = simulate(&g, &test_atmosphere(), &SimOptions::default()).unwrap();
        assert_eq!(traj.len(), 1);
        assert_eq!(analyze(&traj).max_altitude_m, 0.0);
        assert_eq!(analyze(&[]), Performance::default());
    }

    #[test]
    fn rejects_degenerate_geometry() {
        let atm = test_atmosphere();
        let opts = SimOptions::default();
        let mut g = test_rocket();
        g.dry_mass_kg = 0.0;
        g.propellant_mass_kg = 0.0;
        assert!(matches!(
            simulate(&g, &atm, &opts),
            Err(SimError::Geometry(_))
        ));
        let mut g = test_rocket();
        g.propellant_mass_kg = -1.0;
        assert!(simulate(&g, &atm, &opts).is_err());
        let mut g = test_rocket();
        g.thrust_curve.push((7.0, f64::INFINITY));
        assert!(simulate(&g, &atm, &opts).is_err());
        let mut g = test_rocket();
        g.thrust_curve[1].1 = -1.0;
        assert!(simulate(&g, &atm, &opts).is_err());
        let mut g = test_rocket();
        g.thrust_curve.swap(0, 1);
        assert!(simulate(&g, &atm, &opts).is_err());
        let nan_angle = SimOptions {
            launch_angle_deg: f64::NAN,
            ..opts
        };
        assert_eq!(
            simulate(&test_rocket(), &atm, &nan_angle),
            Err(SimError::LaunchAngle)
        );
    }
}
