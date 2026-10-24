//! 3-DOF rocket ascent simulation and performance analysis.
//!
//! Ported from the original pure-Python `aerospace_mcp.integrations.rockets`
//! with the same model and operation order, so results match the Python
//! implementation to rounding:
//!
//! - Forward Euler integration, flat Earth, constant gravity.
//! - Thrust along the launch angle for the first 5 s, vertical afterwards.
//! - Drag `D = Cd * A_ref * q` opposing the velocity, `q = 0.5 * rho * V^2`.
//! - Constant mass flow `m_prop / t_burn` while thrust is non-zero.
//! - Density and speed of sound interpolated from an atmosphere table built
//!   on the Python side (so the ISA / ambiance model stays single-sourced),
//!   with an exponential fallback above the table.

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

/// Tabulated atmosphere used for density / speed-of-sound lookups.
#[derive(Debug, Clone)]
pub struct AtmosphereTable {
    altitudes_m: Vec<f64>,
    densities_kg_m3: Vec<f64>,
    speeds_of_sound_ms: Vec<f64>,
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
        if altitudes_m.windows(2).any(|w| !(w[0] < w[1])) {
            return Err("atmosphere table altitudes must be strictly increasing".into());
        }
        Ok(Self {
            altitudes_m,
            densities_kg_m3,
            speeds_of_sound_ms,
        })
    }

    fn max_altitude_m(&self) -> f64 {
        self.altitudes_m[self.altitudes_m.len() - 1]
    }

    /// `(density, speed_of_sound)` at `altitude_m`.
    fn lookup(&self, altitude_m: f64) -> (f64, f64) {
        if (0.0..=self.max_altitude_m()).contains(&altitude_m) {
            (
                interp(altitude_m, &self.altitudes_m, &self.densities_kg_m3),
                interp(altitude_m, &self.altitudes_m, &self.speeds_of_sound_ms),
            )
        } else {
            // Fallback for altitudes beyond the table.
            let rho = 1.225 * (-altitude_m / 8400.0).exp();
            let a = 343.0 * ((288.15 - 0.0065 * altitude_m) / 288.15).max(0.1).sqrt();
            (rho, a)
        }
    }
}

/// One sample of the trajectory time history.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrajectoryPoint {
    pub time_s: f64,
    pub altitude_m: f64,
    pub velocity_ms: f64,
    pub acceleration_ms2: f64,
    pub mass_kg: f64,
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
    pub total_impulse_ns: f64,
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
    Geometry(&'static str),
}

impl fmt::Display for SimError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            SimError::TimeStep => "dt_s must be between 0.001 and 10 s",
            SimError::Duration => "max_time_s must be > 0 and <= 3600 s",
            SimError::TooManySteps => "too many integration steps (max 500000); increase dt_s",
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

    // Constant mass flow over the burn; the burn time is the last time with
    // positive thrust, but never less than 1 s.
    let mut mass_flow_rate = 0.0;
    if !geometry.thrust_curve.is_empty() {
        let burn_time_total = geometry
            .thrust_curve
            .iter()
            .filter(|(_, f)| *f > 0.0)
            .map(|(t, _)| *t)
            .fold(1.0_f64, f64::max);
        if burn_time_total > 0.0 {
            mass_flow_rate = geometry.propellant_mass_kg / burn_time_total;
        }
    }

    while time <= opts.max_time_s && altitude >= 0.0 {
        let thrust = thrust_at_time(&geometry.thrust_curve, time);
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

        if thrust > 0.0 && mass_flow_rate > 0.0 {
            mass = geometry.dry_mass_kg.max(mass - mass_flow_rate * dt);
        }

        time += dt;

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
        if i > 0 {
            let prev = &trajectory[i - 1];
            total_impulse += (prev.thrust_n + p.thrust_n) / 2.0 * (p.time_s - prev.time_s);
        }
    }

    // Burnout is the sample right after the last one with meaningful thrust.
    let burnout_idx = match last_thrust_idx {
        Some(i) => (i + 1).min(n - 1),
        None => n - 1,
    };
    let burnout = &trajectory[burnout_idx];

    let propellant_used = first.mass_kg - burnout.mass_kg;
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
        // 1800 N for ~6 s plus the 0.1 s ramp-down sampled at 0.1 s.
        assert!((perf.total_impulse_ns - 1800.0 * 6.05).abs() < 1800.0 * 0.11);
        // Mass never drops below dry mass.
        assert!(traj.iter().all(|p| p.mass_kg >= 15.0));
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
    }
}
