//! PyO3 bindings, exposed to Python as `aerospace_mcp._native`.
//!
//! The bindings are deliberately thin: they convert arguments, release the
//! GIL around the heavy loops, and map Rust errors to `ValueError` /
//! `RuntimeError`. Python-facing data models (dataclasses, JSON shaping)
//! stay in `aerospace_mcp.integrations`.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use crate::lambert::{self, LambertError, LambertOptions};
use crate::optimize;
use crate::orbit::{self, Elements, Gravity};
use crate::rocket::{self, AtmosphereTable, Geometry, Performance, SimOptions, TrajectoryPoint};
use crate::vec3::Vec3;

fn value_error(e: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// Atmosphere lookup table (altitude-sorted) used by the rocket simulation.
#[pyclass(name = "AtmosphereTable", module = "aerospace_mcp._native", frozen)]
struct PyAtmosphereTable(AtmosphereTable);

#[pymethods]
impl PyAtmosphereTable {
    #[new]
    fn new(
        altitudes_m: Vec<f64>,
        densities_kg_m3: Vec<f64>,
        speeds_of_sound_ms: Vec<f64>,
    ) -> PyResult<Self> {
        AtmosphereTable::new(altitudes_m, densities_kg_m3, speeds_of_sound_ms)
            .map(Self)
            .map_err(PyValueError::new_err)
    }
}

/// Any object with `RocketGeometry`'s attributes (the dataclass itself).
#[derive(FromPyObject)]
struct GeometryArg {
    dry_mass_kg: f64,
    propellant_mass_kg: f64,
    diameter_m: f64,
    cd: f64,
    thrust_curve: Option<Vec<Vec<f64>>>,
}

fn thrust_curve(rows: Option<Vec<Vec<f64>>>) -> PyResult<Vec<(f64, f64)>> {
    rows.unwrap_or_default()
        .into_iter()
        .map(|row| match row[..] {
            [t, f, ..] => Ok((t, f)),
            _ => Err(PyValueError::new_err(
                "thrust_curve entries must be [time_s, thrust_n] pairs",
            )),
        })
        .collect()
}

impl TryFrom<GeometryArg> for Geometry {
    type Error = PyErr;

    fn try_from(g: GeometryArg) -> PyResult<Self> {
        Ok(Geometry {
            dry_mass_kg: g.dry_mass_kg,
            propellant_mass_kg: g.propellant_mass_kg,
            diameter_m: g.diameter_m,
            cd: g.cd,
            thrust_curve: thrust_curve(g.thrust_curve)?,
        })
    }
}

/// Any object with `RocketTrajectoryPoint`'s attributes.
#[derive(FromPyObject)]
struct PointArg {
    time_s: f64,
    altitude_m: f64,
    velocity_ms: f64,
    acceleration_ms2: f64,
    mass_kg: f64,
    thrust_n: f64,
    drag_n: f64,
    mach: f64,
    dynamic_pressure_pa: f64,
}

type PointTuple = (f64, f64, f64, f64, f64, f64, f64, f64, f64);
type PerformanceTuple = (f64, f64, f64, f64, f64, f64, f64, f64, f64, f64);
type ThrustProfileTuple = (Vec<f64>, f64, usize, bool, Vec<(f64, f64)>);

fn point_tuple(p: &TrajectoryPoint) -> PointTuple {
    (
        p.time_s,
        p.altitude_m,
        p.velocity_ms,
        p.acceleration_ms2,
        p.mass_kg,
        p.thrust_n,
        p.drag_n,
        p.mach,
        p.dynamic_pressure_pa,
    )
}

fn performance_tuple(p: &Performance) -> PerformanceTuple {
    (
        p.max_altitude_m,
        p.apogee_time_s,
        p.max_velocity_ms,
        p.max_mach,
        p.max_q_pa,
        p.burnout_altitude_m,
        p.burnout_velocity_ms,
        p.burnout_time_s,
        p.total_impulse_ns,
        p.specific_impulse_s,
    )
}

/// Thrust (N) at `time_s` from `[[t, F], ...]` samples.
#[pyfunction]
fn thrust_at_time(thrust_curve: Option<Vec<Vec<f64>>>, time_s: f64) -> PyResult<f64> {
    Ok(rocket::thrust_at_time(
        &self::thrust_curve(thrust_curve)?,
        time_s,
    ))
}

/// Simulate a 3-DOF ascent; returns one 9-tuple per sample, in
/// `RocketTrajectoryPoint` field order.
#[pyfunction]
#[pyo3(signature = (geometry, atmosphere, dt_s=0.1, max_time_s=300.0, launch_angle_deg=90.0))]
fn rocket_trajectory(
    py: Python<'_>,
    geometry: GeometryArg,
    atmosphere: &PyAtmosphereTable,
    dt_s: f64,
    max_time_s: f64,
    launch_angle_deg: f64,
) -> PyResult<Vec<PointTuple>> {
    let geometry = Geometry::try_from(geometry)?;
    let atm = &atmosphere.0;
    let opts = SimOptions {
        dt_s,
        max_time_s,
        launch_angle_deg,
    };
    let points = py
        .detach(|| rocket::simulate(&geometry, atm, &opts))
        .map_err(value_error)?;
    Ok(points.iter().map(point_tuple).collect())
}

/// Performance metrics (in `RocketPerformance` field order) for a sequence
/// of trajectory points.
#[pyfunction]
fn analyze_performance(trajectory: Vec<PointArg>) -> PerformanceTuple {
    let points: Vec<TrajectoryPoint> = trajectory
        .into_iter()
        .map(|p| TrajectoryPoint {
            time_s: p.time_s,
            altitude_m: p.altitude_m,
            velocity_ms: p.velocity_ms,
            acceleration_ms2: p.acceleration_ms2,
            mass_kg: p.mass_kg,
            thrust_n: p.thrust_n,
            drag_n: p.drag_n,
            mach: p.mach,
            dynamic_pressure_pa: p.dynamic_pressure_pa,
        })
        .collect();
    performance_tuple(&rocket::analyze(&points))
}

/// Golden-section minimization of a Python callable; returns `(x, f(x))`.
#[pyfunction]
#[pyo3(signature = (objective, lower_bound, upper_bound, tolerance=1e-3, max_iterations=100))]
fn golden_section_search(
    objective: &Bound<'_, PyAny>,
    lower_bound: f64,
    upper_bound: f64,
    tolerance: f64,
    max_iterations: usize,
) -> PyResult<(f64, f64)> {
    optimize::golden_section_search(
        |x| objective.call1((x,))?.extract::<f64>(),
        lower_bound,
        upper_bound,
        tolerance,
        max_iterations,
    )
}

/// Projected gradient descent on a Python callable taking a list of floats;
/// returns `(params, value, iterations, converged)`.
#[pyfunction]
#[pyo3(signature = (objective, initial_params, param_bounds, learning_rate=0.01, tolerance=1e-3, max_iterations=100))]
fn gradient_descent(
    objective: &Bound<'_, PyAny>,
    initial_params: Vec<f64>,
    param_bounds: Vec<(f64, f64)>,
    learning_rate: f64,
    tolerance: f64,
    max_iterations: usize,
) -> PyResult<(Vec<f64>, f64, usize, bool)> {
    if initial_params.len() != param_bounds.len() {
        return Err(PyValueError::new_err(
            "param_bounds must have one (lower, upper) pair per parameter",
        ));
    }
    let r = optimize::projected_gradient_descent(
        |p| objective.call1((p.to_vec(),))?.extract::<f64>(),
        &initial_params,
        &param_bounds,
        learning_rate,
        tolerance,
        max_iterations,
    )?;
    Ok((r.params, r.value, r.iterations, r.converged))
}

/// Golden-section launch-angle optimization run entirely natively;
/// returns `(angle_deg, minimized_objective)`.
#[pyfunction]
fn optimize_launch_angle(
    py: Python<'_>,
    geometry: GeometryArg,
    atmosphere: &PyAtmosphereTable,
    objective: &str,
    lower_deg: f64,
    upper_deg: f64,
) -> PyResult<(f64, f64)> {
    let geometry = Geometry::try_from(geometry)?;
    let atm = &atmosphere.0;
    Ok(py.detach(|| {
        optimize::optimize_launch_angle(&geometry, atm, objective, lower_deg, upper_deg)
    }))
}

/// Thrust-profile optimization run entirely natively; returns
/// `(multipliers, minimized_objective, iterations, converged, thrust_curve)`.
#[pyfunction]
fn optimize_thrust_profile(
    py: Python<'_>,
    geometry: GeometryArg,
    atmosphere: &PyAtmosphereTable,
    burn_time_s: f64,
    total_impulse_target: f64,
    n_segments: usize,
    objective: &str,
) -> PyResult<ThrustProfileTuple> {
    let geometry = Geometry::try_from(geometry)?;
    let atm = &atmosphere.0;
    let r = py
        .detach(|| {
            optimize::optimize_thrust_profile(
                &geometry,
                atm,
                burn_time_s,
                total_impulse_target,
                n_segments,
                objective,
            )
        })
        .map_err(value_error)?;
    Ok((
        r.multipliers,
        r.value,
        r.iterations,
        r.converged,
        r.thrust_curve,
    ))
}

/// Eccentric anomaly (rad) from mean anomaly via Newton-Raphson.
#[pyfunction]
#[pyo3(signature = (mean_anomaly_rad, eccentricity, tolerance=1e-8, max_iterations=50))]
fn kepler_solve(
    mean_anomaly_rad: f64,
    eccentricity: f64,
    tolerance: f64,
    max_iterations: usize,
) -> f64 {
    orbit::kepler_solve(mean_anomaly_rad, eccentricity, tolerance, max_iterations)
}

/// Classical elements (degrees) -> `(position_m, velocity_ms)`.
///
/// Raises `ValueError` for parabolic, inconsistent or non-finite elements.
#[pyfunction]
#[pyo3(signature = (semi_major_axis_m, eccentricity, inclination_deg, raan_deg, arg_periapsis_deg, true_anomaly_deg, mu=orbit::MU_EARTH))]
fn elements_to_state(
    semi_major_axis_m: f64,
    eccentricity: f64,
    inclination_deg: f64,
    raan_deg: f64,
    arg_periapsis_deg: f64,
    true_anomaly_deg: f64,
    mu: f64,
) -> PyResult<(Vec3, Vec3)> {
    let el = Elements {
        semi_major_axis_m,
        eccentricity,
        inclination_deg,
        raan_deg,
        arg_periapsis_deg,
        true_anomaly_deg,
    };
    orbit::elements_to_state(&el, mu).map_err(value_error)
}

/// `(position_m, velocity_ms)` -> `(a, e, i, raan, arg_periapsis, true_anomaly)`
/// with angles in degrees.
#[pyfunction]
#[pyo3(signature = (position_m, velocity_ms, mu=orbit::MU_EARTH))]
fn state_to_elements(
    position_m: Vec3,
    velocity_ms: Vec3,
    mu: f64,
) -> PyResult<(f64, f64, f64, f64, f64, f64)> {
    let el = orbit::state_to_elements(position_m, velocity_ms, mu).map_err(value_error)?;
    Ok((
        el.semi_major_axis_m,
        el.eccentricity,
        el.inclination_deg,
        el.raan_deg,
        el.arg_periapsis_deg,
        el.true_anomaly_deg,
    ))
}

/// `(period_s, apoapsis_alt_m, periapsis_alt_m, energy_j_kg, angular_momentum_m2s)`
/// of an elliptical orbit; raises `ValueError` otherwise.
#[pyfunction]
#[pyo3(signature = (semi_major_axis_m, eccentricity, mu=orbit::MU_EARTH, body_radius_m=orbit::R_EARTH))]
fn orbit_properties(
    semi_major_axis_m: f64,
    eccentricity: f64,
    mu: f64,
    body_radius_m: f64,
) -> PyResult<(f64, f64, f64, f64, f64)> {
    let p = orbit::orbit_properties(semi_major_axis_m, eccentricity, mu, body_radius_m)
        .map_err(value_error)?;
    Ok((
        p.period_s,
        p.apoapsis_m,
        p.periapsis_m,
        p.energy_j_kg,
        p.angular_momentum_m2s,
    ))
}

/// RK4 propagation with J2; returns `[(t_s, position_m, velocity_ms), ...]`
/// including the initial state.
#[pyfunction]
#[pyo3(signature = (position_m, velocity_ms, time_span_s, time_step_s=60.0, mu=orbit::MU_EARTH, j2=orbit::J2_EARTH, equatorial_radius_m=orbit::R_EARTH))]
#[allow(clippy::too_many_arguments)]
fn propagate_j2(
    py: Python<'_>,
    position_m: Vec3,
    velocity_ms: Vec3,
    time_span_s: f64,
    time_step_s: f64,
    mu: f64,
    j2: f64,
    equatorial_radius_m: f64,
) -> PyResult<Vec<orbit::StateSample>> {
    let gravity = Gravity {
        mu,
        j2,
        equatorial_radius_m,
    };
    py.detach(|| orbit::propagate(position_m, velocity_ms, time_span_s, time_step_s, gravity))
        .map_err(value_error)
}

/// `[(latitude_deg, longitude_deg, altitude_m), ...]` for inertial positions
/// at `elapsed_s` seconds after the first sample.
#[pyfunction]
fn ground_track(positions_m: Vec<Vec3>, elapsed_s: Vec<f64>) -> PyResult<Vec<(f64, f64, f64)>> {
    orbit::ground_track(&positions_m, &elapsed_s).map_err(value_error)
}

/// `(delta_v1, delta_v2, total_delta_v, transfer_time_s, transfer_semi_major_axis_m)`.
#[pyfunction]
#[pyo3(signature = (r1_m, r2_m, mu=orbit::MU_EARTH))]
fn hohmann_transfer(r1_m: f64, r2_m: f64, mu: f64) -> PyResult<(f64, f64, f64, f64, f64)> {
    let h = orbit::hohmann(r1_m, r2_m, mu).map_err(value_error)?;
    Ok((
        h.delta_v1_ms,
        h.delta_v2_ms,
        h.total_delta_v_ms,
        h.transfer_time_s,
        h.transfer_semi_major_axis_m,
    ))
}

/// Izzo's Lambert solver; returns `(v1, v2, transfer_angle_rad)`, velocities
/// in units consistent with the inputs and the transfer angle actually
/// traversed (in the direction of motion, excluding full revolutions).
///
/// Raises `ValueError` for invalid or infeasible geometry and
/// `RuntimeError` if the iteration fails to converge.
#[pyfunction]
#[pyo3(signature = (mu, r1, r2, tof, revolutions=0, prograde=true, low_path=true, max_iterations=35, tolerance=1e-8))]
#[allow(clippy::too_many_arguments)]
fn lambert_izzo(
    mu: f64,
    r1: Vec3,
    r2: Vec3,
    tof: f64,
    revolutions: u32,
    prograde: bool,
    low_path: bool,
    max_iterations: usize,
    tolerance: f64,
) -> PyResult<(Vec3, Vec3, f64)> {
    let opts = LambertOptions {
        revolutions,
        prograde,
        low_path,
        max_iterations,
        tolerance,
    };
    lambert::izzo(mu, r1, r2, tof, opts)
        .map(|s| (s.v1, s.v2, s.transfer_angle_rad))
        .map_err(|e| match e {
            LambertError::NoConvergence => PyRuntimeError::new_err(e.to_string()),
            _ => value_error(e),
        })
}

/// Native trajectory and orbital-mechanics core for aerospace-mcp.
#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyAtmosphereTable>()?;
    m.add_function(wrap_pyfunction!(thrust_at_time, m)?)?;
    m.add_function(wrap_pyfunction!(rocket_trajectory, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_performance, m)?)?;
    m.add_function(wrap_pyfunction!(golden_section_search, m)?)?;
    m.add_function(wrap_pyfunction!(gradient_descent, m)?)?;
    m.add_function(wrap_pyfunction!(optimize_launch_angle, m)?)?;
    m.add_function(wrap_pyfunction!(optimize_thrust_profile, m)?)?;
    m.add_function(wrap_pyfunction!(kepler_solve, m)?)?;
    m.add_function(wrap_pyfunction!(elements_to_state, m)?)?;
    m.add_function(wrap_pyfunction!(state_to_elements, m)?)?;
    m.add_function(wrap_pyfunction!(orbit_properties, m)?)?;
    m.add_function(wrap_pyfunction!(propagate_j2, m)?)?;
    m.add_function(wrap_pyfunction!(ground_track, m)?)?;
    m.add_function(wrap_pyfunction!(hohmann_transfer, m)?)?;
    m.add_function(wrap_pyfunction!(lambert_izzo, m)?)?;
    m.add("MU_EARTH", orbit::MU_EARTH)?;
    m.add("R_EARTH", orbit::R_EARTH)?;
    m.add("J2_EARTH", orbit::J2_EARTH)?;
    m.add("OMEGA_EARTH", orbit::OMEGA_EARTH)?;
    m.add("G0", rocket::G0)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
