//! Lightweight optimizers and the rocket trajectory optimizations built on them.
//!
//! [`golden_section_search`], [`projected_gradient_descent`] and
//! [`projected_descent_line_search`] are generic over a fallible objective so
//! the same code serves both Python callables (errors are Python exceptions)
//! and native objectives. The rocket optimizers ([`optimize_launch_angle`],
//! [`optimize_thrust_profile`]) run the simulation entirely in Rust, which is
//! what makes them fast: the Python implementation spent ~15 us per Euler
//! step and ran hundreds of simulations per optimization.
//!
//! The golden-section search and the launch-angle optimizer are unchanged
//! from the original Python `aerospace_mcp.integrations.trajopt`. The
//! thrust-profile optimizer was reworked after the port because the original
//! degenerated (see [`optimize_thrust_profile`]).

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

/// Result of [`projected_gradient_descent`] and [`projected_descent_line_search`].
#[derive(Debug, Clone, PartialEq)]
pub struct DescentResult {
    pub params: Vec<f64>,
    pub value: f64,
    pub iterations: usize,
    pub converged: bool,
}

/// Central-difference gradient of `f` at `x` with step `h`, into `gradient`.
///
/// Probes are clamped to `bounds`, and each difference is divided by the
/// actual distance between its two probes, so a one-sided difference at a
/// bound is not halved. A parameter whose bounds coincide gets 0.
fn central_gradient<E>(
    f: &mut impl FnMut(&[f64]) -> Result<f64, E>,
    x: &[f64],
    bounds: &[(f64, f64)],
    h: f64,
    probe: &mut Vec<f64>,
    gradient: &mut [f64],
) -> Result<(), E> {
    for i in 0..x.len() {
        probe.clear();
        probe.extend_from_slice(x);
        let hi = bounds[i].1.min(x[i] + h);
        let lo = bounds[i].0.max(x[i] - h);
        probe[i] = hi;
        let plus = f(probe)?;
        probe[i] = lo;
        let minus = f(probe)?;
        gradient[i] = if hi > lo {
            (plus - minus) / (hi - lo)
        } else {
            0.0
        };
    }
    Ok(())
}

/// `x - step * gradient`, projected onto `bounds`.
fn projected_step(x: &[f64], gradient: &[f64], step: f64, bounds: &[(f64, f64)]) -> Vec<f64> {
    x.iter()
        .zip(gradient)
        .zip(bounds)
        .map(|((xi, gi), (lo, hi))| lo.max(hi.min(xi - step * gi)))
        .collect()
}

fn max_abs_change(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

/// Box-constrained gradient descent with a central-difference gradient and a
/// fixed learning rate.
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
    let bounds = &bounds[..n];
    let mut current = initial[..n].to_vec();
    let mut current_value = f(&current)?;
    let mut probe = Vec::with_capacity(n);
    let mut gradient = vec![0.0; n];

    for iteration in 0..max_iterations {
        central_gradient(&mut f, &current, bounds, H, &mut probe, &mut gradient)?;
        let next = projected_step(&current, &gradient, learning_rate, bounds);
        let max_change = max_abs_change(&next, &current);
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

/// Settings for [`projected_descent_line_search`].
#[derive(Debug, Clone, Copy)]
pub struct LineSearchOptions {
    /// First trial step length (in parameter units per unit gradient).
    pub initial_step: f64,
    /// Central-difference step for the gradient.
    pub fd_step: f64,
    /// Stop when a step moves no parameter by more than this, or improves
    /// the objective by less than this.
    pub tolerance: f64,
    pub max_iterations: usize,
}

/// Box-constrained steepest descent with a backtracking (Armijo) line search.
///
/// Each iteration takes the projected step `P(x - t * grad)` and halves `t`
/// until the objective decreases by at least `1e-4 * grad . (x - x_new)`;
/// a successful step lets the next iteration try `2 t`. Only decreasing
/// steps are accepted, so the result is never worse than `initial`.
/// Converged means the line search could not find a decrease with a step
/// longer than `tolerance` (a projected stationary point at that
/// resolution), or the last step improved the objective by less than
/// `tolerance`.
pub fn projected_descent_line_search<E>(
    mut f: impl FnMut(&[f64]) -> Result<f64, E>,
    initial: &[f64],
    bounds: &[(f64, f64)],
    opts: LineSearchOptions,
) -> Result<DescentResult, E> {
    const ARMIJO: f64 = 1e-4;
    const MAX_HALVINGS: usize = 40;
    let n = initial.len().min(bounds.len());
    let bounds = &bounds[..n];
    let mut current = initial[..n].to_vec();
    let mut current_value = f(&current)?;
    let mut probe = Vec::with_capacity(n);
    let mut gradient = vec![0.0; n];
    let mut step = opts.initial_step;

    for iteration in 0..opts.max_iterations {
        central_gradient(
            &mut f,
            &current,
            bounds,
            opts.fd_step,
            &mut probe,
            &mut gradient,
        )?;

        let mut accepted = None;
        for _ in 0..MAX_HALVINGS {
            let next = projected_step(&current, &gradient, step, bounds);
            if max_abs_change(&next, &current) <= opts.tolerance {
                break;
            }
            let next_value = f(&next)?;
            let predicted: f64 = gradient
                .iter()
                .zip(&current)
                .zip(&next)
                .map(|((g, x), y)| g * (x - y))
                .sum();
            if next_value <= current_value - ARMIJO * predicted {
                accepted = Some((next, next_value));
                break;
            }
            step *= 0.5;
        }

        let Some((next, next_value)) = accepted else {
            return Ok(DescentResult {
                params: current,
                value: current_value,
                iterations: iteration + 1,
                converged: true,
            });
        };
        let improvement = current_value - next_value;
        current = next;
        current_value = next_value;
        if improvement < opts.tolerance {
            return Ok(DescentResult {
                params: current,
                value: current_value,
                iterations: iteration + 1,
                converged: true,
            });
        }
        step *= 2.0;
    }

    Ok(DescentResult {
        params: current,
        value: current_value,
        iterations: opts.max_iterations,
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

impl From<rocket::SimError> for OptimizeError {
    fn from(e: rocket::SimError) -> Self {
        OptimizeError::InvalidInput(e.to_string())
    }
}

/// Find the launch angle (degrees) minimizing the objective; returns
/// `(angle_deg, objective_value)` where the value is the *minimized* one
/// (negative altitude for `"max_altitude"`).
///
/// Objectives: `"max_altitude"`; `"max_range"` is accepted but downrange is
/// not modelled, so it is constant zero (as in the Python original). Other
/// objectives, invalid geometry and non-finite bounds are errors.
pub fn optimize_launch_angle(
    geometry: &Geometry,
    atmosphere: &AtmosphereTable,
    objective: &str,
    lower_deg: f64,
    upper_deg: f64,
) -> Result<(f64, f64), OptimizeError> {
    if !matches!(objective, "max_altitude" | "max_range") {
        return Err(OptimizeError::InvalidInput(format!(
            "unknown objective {objective:?} (expected \"max_altitude\" or \"max_range\")"
        )));
    }
    if !(lower_deg.is_finite() && upper_deg.is_finite()) {
        return Err(OptimizeError::InvalidInput(
            "angle bounds must be finite".into(),
        ));
    }
    geometry.validate()?;

    let mut buffer = Vec::new();
    let eval = |angle: f64| -> Result<f64, rocket::SimError> {
        let opts = SimOptions {
            launch_angle_deg: angle,
            ..SimOptions::default()
        };
        rocket::simulate_into(geometry, atmosphere, &opts, &mut buffer)?;
        Ok(match objective {
            "max_altitude" => -rocket::analyze(&buffer).max_altitude_m,
            _ => -0.0,
        })
    };
    Ok(golden_section_search(eval, lower_deg, upper_deg, 0.1, 50)?)
}

/// Step thrust curve for `multipliers`, scaled to deliver `total_impulse`.
///
/// Segment `i` covers `[i, i + 1) * burn_time / n` at thrust
/// `multiplier_i * total_impulse / (segment_time * sum(multipliers))`. Each
/// segment is a pair of samples at its start and end; consecutive segments
/// share a time (a step, see [`rocket::thrust_at_time`]) and thrust is zero
/// from the last sample on, so the interpolated curve's impulse is exactly
/// `total_impulse`. Returns `None` unless the multipliers sum to a positive,
/// finite value.
pub fn thrust_profile_curve(
    multipliers: &[f64],
    burn_time_s: f64,
    total_impulse: f64,
) -> Option<Vec<(f64, f64)>> {
    let n = multipliers.len();
    let sum: f64 = multipliers.iter().sum();
    if !(sum > 0.0 && sum.is_finite()) {
        return None;
    }
    let segment_time = burn_time_s / n as f64;
    let thrust_per_unit = total_impulse / (segment_time * sum);

    let mut curve = Vec::with_capacity(2 * n);
    for (i, m) in multipliers.iter().enumerate() {
        let thrust = m * thrust_per_unit;
        curve.push((i as f64 * segment_time, thrust));
        curve.push(((i + 1) as f64 * segment_time, thrust));
    }
    Some(curve)
}

/// Upper bound on [`optimize_thrust_profile`]'s `n_segments`.
pub const MAX_THRUST_SEGMENTS: usize = 50;

/// Bounds on each thrust multiplier (relative to the average thrust).
pub const THRUST_MULTIPLIER_BOUNDS: (f64, f64) = (0.1, 3.0);

/// Result of [`optimize_thrust_profile`].
#[derive(Debug, Clone, PartialEq)]
pub struct ThrustProfileResult {
    pub multipliers: Vec<f64>,
    /// Minimized objective value (negative altitude for `"max_altitude"`).
    pub value: f64,
    /// Objective value of the constant-thrust starting profile.
    pub initial_value: f64,
    pub iterations: usize,
    pub converged: bool,
    pub thrust_curve: Vec<(f64, f64)>,
}

/// Optimize per-segment thrust multipliers (bounded to
/// [`THRUST_MULTIPLIER_BOUNDS`]) for a fixed burn time and total impulse.
///
/// Objectives: `"max_altitude"` (minimizes `-max_altitude_m`),
/// `"min_max_q"`, and `"min_gravity_loss"` (approximated as
/// `g0 * burnout_time`; with a fixed burn time this is nearly constant, so
/// the optimizer usually returns the starting profile). Candidates are
/// simulated with the default settings (`dt_s = 0.1`, `max_time_s = 300`),
/// the same ones the caller uses to report the final trajectory, so the
/// minimized value is that trajectory's metric.
///
/// The search starts from constant thrust and runs
/// [`projected_descent_line_search`] on the objective divided by its value
/// at the start, so the step length does not depend on the objective's
/// units (the original fixed 0.05 step on raw meters slammed every
/// multiplier to a bound on the first iteration). Only improving steps are
/// accepted, so the result is never worse than constant thrust.
pub fn optimize_thrust_profile(
    geometry: &Geometry,
    atmosphere: &AtmosphereTable,
    burn_time_s: f64,
    total_impulse: f64,
    n_segments: usize,
    objective: &str,
) -> Result<ThrustProfileResult, OptimizeError> {
    // Each iteration runs 2 * n_segments + 1 or more simulations and
    // allocates per-segment buffers, so an unbounded count can hang or
    // exhaust memory (an allocation failure aborts the whole server process).
    if !(1..=MAX_THRUST_SEGMENTS).contains(&n_segments) {
        return Err(OptimizeError::InvalidInput(format!(
            "n_segments must be between 1 and {MAX_THRUST_SEGMENTS}"
        )));
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
    let metric: fn(&rocket::Performance) -> f64 = match objective {
        "max_altitude" => |p| -p.max_altitude_m,
        "min_max_q" => |p| p.max_q_pa,
        "min_gravity_loss" => |p| G0 * p.burnout_time_s,
        _ => {
            return Err(OptimizeError::InvalidInput(format!(
                "unknown objective {objective:?} (expected \"max_altitude\", \
                 \"min_max_q\" or \"min_gravity_loss\")"
            )));
        }
    };

    let opts = SimOptions::default();
    let mut candidate = geometry.clone();
    let mut buffer: Vec<TrajectoryPoint> = Vec::new();
    let mut eval = |multipliers: &[f64]| -> Result<f64, OptimizeError> {
        candidate.thrust_curve = thrust_profile_curve(multipliers, burn_time_s, total_impulse)
            .ok_or_else(|| OptimizeError::InvalidInput("degenerate thrust profile".into()))?;
        rocket::simulate_into(&candidate, atmosphere, &opts, &mut buffer)?;
        Ok(metric(&rocket::analyze(&buffer)))
    };

    let initial = vec![1.0; n_segments];
    let initial_value = eval(&initial)?;
    let scale = if initial_value.abs() > 0.0 {
        initial_value.abs()
    } else {
        1.0
    };

    let result = projected_descent_line_search(
        |m: &[f64]| Ok::<f64, OptimizeError>(eval(m)? / scale),
        &initial,
        &vec![THRUST_MULTIPLIER_BOUNDS; n_segments],
        LineSearchOptions {
            initial_step: 1.0,
            // Multipliers are O(1); this is well above the noise of the
            // 0.1 s Euler simulation and small against the bounds.
            fd_step: 1e-3,
            tolerance: 1e-4,
            max_iterations: 100,
        },
    )?;

    // Multipliers stay within the bounds, so the impulse is always positive.
    let thrust_curve = thrust_profile_curve(&result.params, burn_time_s, total_impulse)
        .ok_or_else(|| OptimizeError::InvalidInput("degenerate thrust profile".into()))?;
    // Unscaled, so it is bit-for-bit the returned curve's metric.
    let value = eval(&result.params)?;

    Ok(ThrustProfileResult {
        multipliers: result.params,
        value,
        initial_value,
        iterations: result.iterations,
        converged: result.converged,
        thrust_curve,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rocket::ThrustProfile;
    use crate::rocket::tests::{test_atmosphere, test_rocket};
    use std::convert::Infallible;

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

    /// The probe is clamped at the upper bound, so the difference spans only
    /// h; dividing by 2h used to halve the gradient (1.0 -> 0.95, not 0.9).
    #[test]
    fn gradient_is_not_halved_at_a_bound() {
        for (start, bounds) in [(1.0, (0.0, 1.0)), (0.0, (0.0, 1.0)), (0.5, (0.0, 1.0))] {
            let r =
                projected_gradient_descent(|p| ok(p[0]), &[start], &[bounds], 0.1, 0.0, 1).unwrap();
            let expected = (start - 0.1_f64).max(0.0);
            assert!((r.params[0] - expected).abs() < 1e-9, "{r:?}");
        }
        // Degenerate (fixed) parameter: zero gradient, no NaN.
        let r =
            projected_gradient_descent(|p| ok(p[0]), &[1.0], &[(1.0, 1.0)], 0.1, 0.0, 1).unwrap();
        assert_eq!(r.params, vec![1.0]);
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
    fn line_search_handles_badly_scaled_objectives() {
        let opts = LineSearchOptions {
            initial_step: 1.0,
            fd_step: 1e-6,
            tolerance: 1e-10,
            max_iterations: 200,
        };
        // Gradient ~1e4 per unit: a fixed unit step would overshoot wildly.
        let r = projected_descent_line_search(
            |p| ok(1e4 * ((p[0] - 0.3).powi(2) + (p[1] + 0.5).powi(2))),
            &[2.0, 2.0],
            &[(-1.0, 3.0), (-1.0, 3.0)],
            opts,
        )
        .unwrap();
        assert!(r.converged);
        assert!((r.params[0] - 0.3).abs() < 1e-3 && (r.params[1] + 0.5).abs() < 1e-3);
        // Constrained optimum on the bound.
        let r = projected_descent_line_search(|p| ok(-p[0]), &[0.0], &[(-1.0, 2.0)], opts).unwrap();
        assert_eq!(r.params, vec![2.0]);
        // Never worse than the start, even when nothing improves it.
        let r = projected_descent_line_search(|p| ok(p[0].abs()), &[0.0], &[(-1.0, 1.0)], opts)
            .unwrap();
        assert_eq!((r.params[0], r.value, r.converged), (0.0, 0.0, true));
    }

    #[test]
    fn thrust_profile_curve_delivers_the_impulse() {
        for multipliers in [
            vec![1.0],
            vec![0.5, 1.0, 2.0],
            vec![3.0, 0.1, 0.1, 2.0, 0.7],
        ] {
            let curve = thrust_profile_curve(&multipliers, 6.0, 12000.0).unwrap();
            assert_eq!(curve.len(), 2 * multipliers.len());
            let impulse = ThrustProfile::new(&curve).total_impulse();
            assert!((impulse - 12000.0).abs() < 1e-9, "{impulse}");
            // Piecewise constant: mid-segment thrust is proportional to m_i.
            for (i, m) in multipliers.iter().enumerate() {
                let t = (i as f64 + 0.5) * 6.0 / multipliers.len() as f64;
                let thrust = rocket::thrust_at_time(&curve, t);
                assert!((thrust / curve[0].1 - m / multipliers[0]).abs() < 1e-12);
            }
        }
        assert!(thrust_profile_curve(&[0.0], 6.0, 12000.0).is_none());
        assert!(thrust_profile_curve(&[f64::NAN], 6.0, 12000.0).is_none());
    }

    #[test]
    fn simulated_impulse_matches_the_target() {
        let (mut g, atm) = (test_rocket(), test_atmosphere());
        for n in [1, 2, 5, 7] {
            let multipliers: Vec<f64> = (0..n).map(|i| 2.0 - i as f64 * 0.2).collect();
            g.thrust_curve = thrust_profile_curve(&multipliers, 6.0, 10800.0).unwrap();
            for dt_s in [0.1, 0.2, 0.07] {
                let opts = SimOptions {
                    dt_s,
                    ..SimOptions::default()
                };
                let perf = rocket::analyze(&rocket::simulate(&g, &atm, &opts).unwrap());
                assert!(
                    (perf.total_impulse_ns - 10800.0).abs() < 1e-6,
                    "{n} {dt_s} {perf:?}"
                );
            }
        }
    }

    #[test]
    fn launch_angle_prefers_vertical_for_altitude() {
        let (angle, value) = optimize_launch_angle(
            &test_rocket(),
            &test_atmosphere(),
            "max_altitude",
            80.0,
            90.0,
        )
        .unwrap();
        assert!(angle > 89.0 && angle <= 90.0);
        assert!(value < -1000.0);
        let (g, atm) = (test_rocket(), test_atmosphere());
        assert!(optimize_launch_angle(&g, &atm, "nope", 80.0, 90.0).is_err());
        assert!(optimize_launch_angle(&g, &atm, "max_altitude", f64::NAN, 90.0).is_err());
        let mut bad = test_rocket();
        bad.dry_mass_kg = 0.0;
        assert!(optimize_launch_angle(&bad, &atm, "max_altitude", 80.0, 90.0).is_err());
    }

    #[test]
    fn thrust_profile_rejects_bad_inputs_and_runs() {
        let (g, atm) = (test_rocket(), test_atmosphere());
        assert!(optimize_thrust_profile(&g, &atm, 6.0, 10800.0, 0, "max_altitude").is_err());
        assert!(optimize_thrust_profile(&g, &atm, 6.0, 10800.0, 51, "max_altitude").is_err());
        assert!(
            optimize_thrust_profile(&g, &atm, 6.0, 10800.0, usize::MAX, "max_altitude").is_err()
        );
        assert!(optimize_thrust_profile(&g, &atm, 0.0, 10800.0, 3, "max_altitude").is_err());
        assert!(optimize_thrust_profile(&g, &atm, 6.0, -1.0, 3, "max_altitude").is_err());
        assert!(optimize_thrust_profile(&g, &atm, 6.0, 10800.0, 3, "nope").is_err());
        let r = optimize_thrust_profile(&g, &atm, 6.0, 10800.0, 3, "min_max_q").unwrap();
        assert_eq!(r.multipliers.len(), 3);
        assert!(r.multipliers.iter().all(|m| (0.1..=3.0).contains(m)));
        assert_eq!(r.thrust_curve.len(), 6);
    }

    /// The original fixed-step descent on raw meters ended at 0 m for this
    /// rocket (every multiplier on a bound). The optimum must never be worse
    /// than the constant-thrust start, for any objective.
    #[test]
    fn thrust_profile_never_worse_than_constant_thrust() {
        let atm = test_atmosphere();
        let mut g = test_rocket();
        g.thrust_curve.clear();
        for objective in ["max_altitude", "min_max_q", "min_gravity_loss"] {
            for n in [1, 2, 5, 12] {
                let r = optimize_thrust_profile(&g, &atm, 6.0, 10800.0, n, objective).unwrap();
                assert!(r.value <= r.initial_value, "{objective} {n}: {r:?}");
                // The reported value is the metric of the returned curve.
                let mut check = g.clone();
                check.thrust_curve = r.thrust_curve.clone();
                let perf = rocket::analyze(
                    &rocket::simulate(&check, &atm, &SimOptions::default()).unwrap(),
                );
                let metric = match objective {
                    "max_altitude" => -perf.max_altitude_m,
                    "min_max_q" => perf.max_q_pa,
                    _ => G0 * perf.burnout_time_s,
                };
                assert!((metric - r.value).abs() <= 1e-9 * r.value.abs());
                assert!((perf.total_impulse_ns - 10800.0).abs() < 1e-6);
            }
        }
        let alt = optimize_thrust_profile(&g, &atm, 6.0, 10800.0, 5, "max_altitude").unwrap();
        assert!(-alt.value > 1900.0, "{alt:?}");
    }
}
