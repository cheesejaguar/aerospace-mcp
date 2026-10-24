//! Lightweight optimizers and the rocket trajectory optimizations built on them.
//!
//! [`golden_section_search`] and [`projected_gradient_descent`] are generic
//! over a fallible objective so the same code serves both Python callables
//! (errors are Python exceptions) and native objectives. The rocket
//! optimizers ([`optimize_launch_angle`], [`optimize_thrust_profile`]) run
//! the simulation entirely in Rust, which is what makes them fast: the
//! Python implementation spent ~15 us per Euler step and ran hundreds of
//! simulations per optimization.
//!
//! Algorithms, constants and objective definitions are unchanged from the
//! original Python `aerospace_mcp.integrations.trajopt`.

use std::convert::Infallible;
use std::fmt;

use crate::rocket::{self, AtmosphereTable, G0, Geometry, SimOptions, TrajectoryPoint};

/// Golden-section search for the minimum of a unimodal function on `[lower, upper]`.
///
/// Returns `(x, f(x))` for the better of the two final probe points.
pub fn golden_section_search<E>(
    mut f: impl FnMut(f64) -> Result<f64, E>,
    mut lower: f64,
    mut upper: f64,
    tolerance: f64,
    max_iterations: usize,
) -> Result<(f64, f64), E> {
    let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
    let resphi = 2.0 - phi;

    let mut x1 = lower + resphi * (upper - lower);
    let mut x2 = upper - resphi * (upper - lower);
    let mut f1 = f(x1)?;
    let mut f2 = f(x2)?;

    for _ in 0..max_iterations {
        if (upper - lower).abs() < tolerance {
            break;
        }
        if f1 < f2 {
            upper = x2;
            x2 = x1;
            f2 = f1;
            x1 = lower + resphi * (upper - lower);
            f1 = f(x1)?;
        } else {
            lower = x1;
            x1 = x2;
            f1 = f2;
            x2 = upper - resphi * (upper - lower);
            f2 = f(x2)?;
        }
    }

    Ok(if f1 < f2 { (x1, f1) } else { (x2, f2) })
}

/// Result of [`projected_gradient_descent`].
#[derive(Debug, Clone, PartialEq)]
pub struct DescentResult {
    pub params: Vec<f64>,
    pub value: f64,
    pub iterations: usize,
    pub converged: bool,
}

/// Box-constrained gradient descent with a central-difference gradient.
///
/// The gradient step is `h = 1e-6` (perturbations are clamped to the bounds);
/// after each update the parameters are projected back onto the box.
/// Converges when both the largest parameter change and the objective change
/// fall below `tolerance`. `bounds` must have one entry per parameter.
pub fn projected_gradient_descent<E>(
    mut f: impl FnMut(&[f64]) -> Result<f64, E>,
    initial: &[f64],
    bounds: &[(f64, f64)],
    learning_rate: f64,
    tolerance: f64,
    max_iterations: usize,
) -> Result<DescentResult, E> {
    const H: f64 = 1e-6;
    let n = initial.len().min(bounds.len());
    let mut current = initial[..n].to_vec();
    let mut current_value = f(&current)?;
    let mut probe = current.clone();
    let mut gradient = vec![0.0; n];

    for iteration in 0..max_iterations {
        for i in 0..n {
            let param = current[i];
            probe.copy_from_slice(&current);
            probe[i] = bounds[i].1.min(param + H);
            let plus = f(&probe)?;
            probe[i] = bounds[i].0.max(param - H);
            let minus = f(&probe)?;
            gradient[i] = (plus - minus) / (2.0 * H);
        }

        let mut max_change = 0.0_f64;
        let next: Vec<f64> = (0..n)
            .map(|i| {
                let stepped = current[i] - learning_rate * gradient[i];
                let projected = bounds[i].0.max(bounds[i].1.min(stepped));
                max_change = max_change.max((projected - current[i]).abs());
                projected
            })
            .collect();
        let next_value = f(&next)?;

        if max_change < tolerance && (next_value - current_value).abs() < tolerance {
            return Ok(DescentResult {
                params: next,
                value: next_value,
                iterations: iteration + 1,
                converged: true,
            });
        }
        current = next;
        current_value = next_value;
    }

    Ok(DescentResult {
        params: current,
        value: current_value,
        iterations: max_iterations,
        converged: false,
    })
}

/// Invalid optimizer inputs.
#[derive(Debug, Clone, PartialEq)]
pub enum OptimizeError {
    InvalidInput(String),
}

impl fmt::Display for OptimizeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OptimizeError::InvalidInput(msg) => f.write_str(msg),
        }
    }
}

impl std::error::Error for OptimizeError {}

/// Find the launch angle (degrees) minimizing the objective; returns
/// `(angle_deg, objective_value)` where the value is the *minimized* one
/// (negative altitude for `"max_altitude"`).
///
/// Objectives: `"max_altitude"`; `"max_range"` is accepted but downrange is
/// not modelled, so it is constant zero (as in the Python original). Any
/// other objective evaluates to `+inf`.
pub fn optimize_launch_angle(
    geometry: &Geometry,
    atmosphere: &AtmosphereTable,
    objective: &str,
    lower_deg: f64,
    upper_deg: f64,
) -> (f64, f64) {
    let mut buffer = Vec::new();
    let eval = |angle: f64| -> Result<f64, Infallible> {
        let opts = SimOptions {
            launch_angle_deg: angle,
            ..SimOptions::default()
        };
        if rocket::simulate_into(geometry, atmosphere, &opts, &mut buffer).is_err()
            || buffer.is_empty()
        {
            return Ok(f64::INFINITY);
        }
        Ok(match objective {
            "max_altitude" => -rocket::analyze(&buffer).max_altitude_m,
            "max_range" => -0.0,
            _ => f64::INFINITY,
        })
    };
    match golden_section_search(eval, lower_deg, upper_deg, 0.1, 50) {
        Ok(result) => result,
        Err(never) => match never {},
    }
}

/// Piecewise-constant-start thrust curve for `multipliers`, rescaled so the
/// segment impulses sum to `total_impulse`.
///
/// Segment `i` starts at `i * burn_time / n` with thrust
/// `multiplier_i * total_impulse / burn_time`; a final zero-thrust sample
/// closes the burn. Returns `None` when the unscaled impulse is not positive.
pub fn thrust_profile_curve(
    multipliers: &[f64],
    burn_time_s: f64,
    total_impulse: f64,
) -> Option<Vec<(f64, f64)>> {
    let n = multipliers.len();
    let avg_thrust = total_impulse / burn_time_s;
    let segment_time = burn_time_s / n as f64;

    let current_impulse: f64 = multipliers
        .iter()
        .map(|m| avg_thrust * m * segment_time)
        .sum();
    if !(current_impulse > 0.0) {
        return None;
    }
    let scale = total_impulse / current_impulse;

    let mut curve = Vec::with_capacity(n + 1);
    for (i, m) in multipliers.iter().enumerate() {
        curve.push((i as f64 * segment_time, avg_thrust * m * scale));
    }
    curve.push((n as f64 * segment_time, 0.0 * scale));
    Some(curve)
}

/// Result of [`optimize_thrust_profile`].
#[derive(Debug, Clone, PartialEq)]
pub struct ThrustProfileResult {
    pub multipliers: Vec<f64>,
    /// Minimized objective value (negative altitude for `"max_altitude"`).
    pub value: f64,
    pub iterations: usize,
    pub converged: bool,
    pub thrust_curve: Vec<(f64, f64)>,
}

/// Optimize per-segment thrust multipliers (bounded to `[0.1, 3.0]`) for a
/// fixed burn time and total impulse.
///
/// Objectives: `"max_altitude"`, `"min_max_q"`, `"min_gravity_loss"`
/// (approximated as `g0 * burnout_time`). Unknown objectives evaluate to a
/// constant penalty of `1e6`.
pub fn optimize_thrust_profile(
    geometry: &Geometry,
    atmosphere: &AtmosphereTable,
    burn_time_s: f64,
    total_impulse: f64,
    n_segments: usize,
    objective: &str,
) -> Result<ThrustProfileResult, OptimizeError> {
    if n_segments == 0 {
        return Err(OptimizeError::InvalidInput(
            "n_segments must be >= 1".into(),
        ));
    }
    if !(burn_time_s > 0.0 && burn_time_s.is_finite()) {
        return Err(OptimizeError::InvalidInput(
            "burn_time_s must be > 0".into(),
        ));
    }
    if !(total_impulse > 0.0 && total_impulse.is_finite()) {
        return Err(OptimizeError::InvalidInput(
            "total_impulse_target must be > 0".into(),
        ));
    }

    let opts = SimOptions {
        dt_s: 0.2,
        max_time_s: 150.0,
        ..SimOptions::default()
    };
    let mut candidate = geometry.clone();
    let mut buffer: Vec<TrajectoryPoint> = Vec::new();
    let eval = |multipliers: &[f64]| -> Result<f64, Infallible> {
        let Some(curve) = thrust_profile_curve(multipliers, burn_time_s, total_impulse) else {
            return Ok(f64::INFINITY);
        };
        candidate.thrust_curve = curve;
        if rocket::simulate_into(&candidate, atmosphere, &opts, &mut buffer).is_err()
            || buffer.is_empty()
        {
            return Ok(1e6);
        }
        let perf = rocket::analyze(&buffer);
        Ok(match objective {
            "max_altitude" => -perf.max_altitude_m.max(1.0),
            "min_max_q" => perf.max_q_pa,
            "min_gravity_loss" => G0 * perf.burnout_time_s,
            _ => 1e6,
        })
    };

    let result = match projected_gradient_descent(
        eval,
        &vec![1.0; n_segments],
        &vec![(0.1, 3.0); n_segments],
        0.05,
        1e-3,
        100,
    ) {
        Ok(r) => r,
        Err(never) => match never {},
    };

    // Multipliers stay within [0.1, 3.0], so the impulse is always positive.
    let thrust_curve = thrust_profile_curve(&result.params, burn_time_s, total_impulse)
        .ok_or_else(|| OptimizeError::InvalidInput("degenerate thrust profile".into()))?;

    Ok(ThrustProfileResult {
        multipliers: result.params,
        value: result.value,
        iterations: result.iterations,
        converged: result.converged,
        thrust_curve,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rocket::tests::{test_atmosphere, test_rocket};

    fn ok<T>(v: T) -> Result<T, Infallible> {
        Ok(v)
    }

    #[test]
    fn golden_section_finds_quadratic_minimum() {
        let (x, fx) =
            golden_section_search(|x| ok((x - 3.0).powi(2) + 2.0), 0.0, 6.0, 1e-4, 100).unwrap();
        assert!((x - 3.0).abs() < 1e-3);
        assert!((fx - 2.0).abs() < 1e-6);
    }

    #[test]
    fn golden_section_propagates_errors() {
        let r = golden_section_search(|_| Err::<f64, _>("boom"), 0.0, 1.0, 1e-3, 10);
        assert_eq!(r, Err("boom"));
    }

    #[test]
    fn gradient_descent_respects_bounds() {
        let r =
            projected_gradient_descent(|p| ok(p[0] * p[0]), &[3.0], &[(1.0, 5.0)], 0.1, 0.01, 50)
                .unwrap();
        assert!(r.params[0] >= 1.0 && (r.params[0] - 1.0).abs() < 0.1);
    }

    #[test]
    fn gradient_descent_converges_in_2d() {
        let r = projected_gradient_descent(
            |p| ok((p[0] - 1.0).powi(2) + (p[1] - 2.0).powi(2)),
            &[0.0, 0.0],
            &[(-5.0, 5.0), (-5.0, 5.0)],
            0.1,
            0.01,
            100,
        )
        .unwrap();
        assert!(r.converged);
        assert!((r.params[0] - 1.0).abs() < 0.2 && (r.params[1] - 2.0).abs() < 0.2);
    }

    #[test]
    fn thrust_profile_curve_conserves_impulse() {
        let curve = thrust_profile_curve(&[0.5, 1.0, 2.0], 6.0, 12000.0).unwrap();
        assert_eq!(curve.len(), 4);
        let impulse: f64 = curve.windows(2).map(|w| w[0].1 * (w[1].0 - w[0].0)).sum();
        assert!((impulse - 12000.0).abs() < 1e-6);
        assert!(thrust_profile_curve(&[0.0], 6.0, 12000.0).is_none());
    }

    #[test]
    fn launch_angle_prefers_vertical_for_altitude() {
        let (angle, value) = optimize_launch_angle(
            &test_rocket(),
            &test_atmosphere(),
            "max_altitude",
            80.0,
            90.0,
        );
        assert!(angle > 89.0 && angle <= 90.0);
        assert!(value < -1000.0);
        let (_, unknown) =
            optimize_launch_angle(&test_rocket(), &test_atmosphere(), "nope", 80.0, 90.0);
        assert!(unknown.is_infinite());
    }

    #[test]
    fn thrust_profile_rejects_bad_inputs_and_runs() {
        let (g, atm) = (test_rocket(), test_atmosphere());
        assert!(optimize_thrust_profile(&g, &atm, 6.0, 10800.0, 0, "max_altitude").is_err());
        assert!(optimize_thrust_profile(&g, &atm, 0.0, 10800.0, 3, "max_altitude").is_err());
        assert!(optimize_thrust_profile(&g, &atm, 6.0, -1.0, 3, "max_altitude").is_err());
        let r = optimize_thrust_profile(&g, &atm, 6.0, 10800.0, 3, "min_max_q").unwrap();
        assert_eq!(r.multipliers.len(), 3);
        assert!(r.multipliers.iter().all(|m| (0.1..=3.0).contains(m)));
        assert_eq!(r.thrust_curve.len(), 4);
    }
}
