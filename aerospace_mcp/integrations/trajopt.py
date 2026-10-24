"""Trajectory optimization tools for aerospace MCP.

Provides rocket ascent trajectory optimization using lightweight numerical
methods:
    - **Golden section search**: 1-D optimization for single-parameter
      problems (e.g., launch angle).
    - **Gradient descent**: multi-dimensional optimization for thrust
      profile design (numerical central-difference gradient).
    - **Trajectory comparison**: side-by-side evaluation of multiple rocket
      configurations.
    - **Sensitivity analysis**: parameter sweeps with finite-difference
      sensitivity coefficients.

The optimization problem is formulated as::

    minimize  J(x)
    subject to  x_lower <= x <= x_upper

where *J* is the objective (e.g., negative max altitude for maximization)
and *x* is the design vector (launch angle, thrust multipliers, etc.).

The optimizers and the rocket simulation they drive run in the native Rust
core (``aerospace_mcp._native``, sources in ``rust/src/optimize.rs``).
``optimize_launch_angle`` and ``optimize_thrust_profile`` evaluate every
candidate trajectory natively, without a Python round-trip per evaluation.

References:
    - Betts, J.T., "Practical Methods for Optimal Control and Estimation
      Using Nonlinear Programming" (2nd ed., 2010)
    - Press et al., "Numerical Recipes" (3rd ed., 2007), Chapter 10

WARNING: This module is for educational and research purposes only.
Do NOT use for real flight planning, navigation, or aircraft operations.
"""

from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from typing import Any

from .. import _native
from .rockets import (
    RocketGeometry,
    _native_atmosphere,
    analyze_rocket_performance,
    rocket_3dof_trajectory,
    simulate_rocket,
)

# ===========================================================================
# Data Classes
# ===========================================================================


@dataclass
class OptimizationParameters:
    """General optimization algorithm parameters.

    Attributes:
        max_iterations: Maximum number of iterations.
        tolerance: Convergence tolerance.
        step_size: Step size for parameter updates.
        parameter_bounds: Dictionary mapping parameter names to (lower, upper).
    """

    max_iterations: int = 100
    tolerance: float = 1e-3
    step_size: float = 0.1
    parameter_bounds: dict[str, tuple[float, float]] = None


@dataclass
class TrajectoryOptimizationResult:
    """Result from a trajectory optimization run.

    Attributes:
        optimal_parameters: Best parameter values found.
        optimal_objective: Objective function value at the optimum.
        iterations: Number of iterations executed.
        converged: Whether the optimizer converged within tolerance.
        trajectory_points: Trajectory at the optimal parameters.
        performance: Performance summary at the optimal parameters.
        initial_objective: Objective of the starting design, if reported.
    """

    optimal_parameters: dict[str, float]
    optimal_objective: float
    iterations: int
    converged: bool
    trajectory_points: list[Any]  # RocketTrajectoryPoint
    performance: Any  # RocketPerformance
    # Objective of the starting design (same sign convention as
    # optimal_objective), when the optimizer reports it.
    initial_objective: float | None = None


# ===========================================================================
# 1-D Optimization: Golden Section Search
# ===========================================================================


def simple_golden_section_search(
    objective_func: Callable[[float], float],
    lower_bound: float,
    upper_bound: float,
    tolerance: float = 1e-3,
    max_iterations: int = 100,
) -> tuple[float, float]:
    """Golden section search for unimodal 1-D minimization.

    Iteratively narrows a bracketing interval by evaluating the objective
    at two interior points spaced by the golden ratio::

        phi = (1 + sqrt(5)) / 2  ~  1.618
        resphi = 2 - phi          ~  0.382

    Each iteration reduces the interval by a factor of phi, requiring
    only one new function evaluation.  Convergence rate is linear with
    ratio 1/phi ~ 0.618.

    Args:
        objective_func: Scalar objective function to minimize.
        lower_bound: Lower bound of the search interval.
        upper_bound: Upper bound of the search interval.
        tolerance: Convergence tolerance on interval width.
        max_iterations: Maximum number of iterations.

    Returns:
        Tuple of ``(optimal_x, optimal_value)``.
    """
    return _native.golden_section_search(
        objective_func, lower_bound, upper_bound, tolerance, max_iterations
    )


# ===========================================================================
# Multi-Dimensional Optimization: Gradient Descent
# ===========================================================================


def simple_gradient_descent(
    objective_func: Callable[[list[float]], float],
    initial_params: list[float],
    param_bounds: list[tuple[float, float]],
    learning_rate: float = 0.01,
    tolerance: float = 1e-3,
    max_iterations: int = 100,
) -> tuple[list[float], float, int, bool]:
    """Projected gradient descent with numerical central-difference gradient.

    The gradient is estimated via central finite differences::

        dJ/dx_i ~ (J(x + h*e_i) - J(x - h*e_i)) / (2*h)

    where h = 1e-6 (probes are clamped to the bounds, and the difference is
    divided by the actual probe spread).  Parameters are projected back onto the box
    constraints after each update step::

        x_{k+1} = clip(x_k - alpha * grad, x_lower, x_upper)

    Convergence is declared when both the maximum parameter change and
    the objective change are below ``tolerance``.

    Args:
        objective_func: Scalar objective function to minimize.
        initial_params: Initial parameter vector.
        param_bounds: List of ``(lower, upper)`` bound tuples per dimension.
        learning_rate: Step size multiplier (alpha).
        tolerance: Convergence tolerance.
        max_iterations: Maximum number of gradient steps.

    Returns:
        Tuple of ``(optimal_params, optimal_value, iterations, converged)``.

    Raises:
        ValueError: If ``param_bounds`` does not have one pair per parameter.
    """
    params, value, iterations, converged = _native.gradient_descent(
        objective_func,
        initial_params,
        param_bounds,
        learning_rate,
        tolerance,
        max_iterations,
    )
    return params, value, iterations, converged


# ===========================================================================
# Rocket Trajectory Optimization Functions
# ===========================================================================


def optimize_launch_angle(
    geometry: RocketGeometry,
    objective: str = "max_altitude",
    angle_bounds: tuple[float, float] = (80.0, 90.0),
) -> TrajectoryOptimizationResult:
    """Optimize the launch angle by golden-section search (0.1 deg tolerance).

    Each candidate is simulated with the default trajectory settings
    (``dt_s=0.1``, ``max_time_s=300``).

    Args:
        geometry: Rocket geometry and properties.
        objective: ``"max_altitude"``, or ``"max_range"`` (accepted, but
            downrange is not modelled, so every angle scores zero).
        angle_bounds: ``(min_angle_deg, max_angle_deg)``.

    Returns:
        Optimization result with the trajectory at the optimal angle.

    Raises:
        ValueError: For an unknown objective, non-finite bounds or invalid
            geometry.
    """
    optimal_angle, optimal_value = _native.optimize_launch_angle(
        geometry, _native_atmosphere(), objective, angle_bounds[0], angle_bounds[1]
    )

    final_trajectory = rocket_3dof_trajectory(geometry, launch_angle_deg=optimal_angle)
    final_performance = analyze_rocket_performance(final_trajectory)

    return TrajectoryOptimizationResult(
        optimal_parameters={"launch_angle_deg": optimal_angle},
        optimal_objective=-optimal_value,  # Convert back to positive
        iterations=50,  # Approximation
        converged=True,
        trajectory_points=final_trajectory,
        performance=final_performance,
    )


def optimize_thrust_profile(
    geometry: RocketGeometry,
    burn_time_s: float,
    total_impulse_target: float,
    n_segments: int = 5,
    objective: str = "max_altitude",
) -> TrajectoryOptimizationResult:
    """Optimize a segmented thrust profile at fixed burn time and total impulse.

    Each of ``n_segments`` equal-length segments gets a constant thrust with
    a multiplier in ``[0.1, 3.0]`` (relative to the average thrust); the
    step curve is scaled so it delivers exactly ``total_impulse_target``.
    The search starts from constant thrust and takes projected steepest
    descent steps with a backtracking line search on the objective
    normalized by its starting value, accepting only improvements, so the
    result is never worse than constant thrust (``initial_objective``).
    Candidates and the final trajectory use the default simulation settings
    (``dt_s=0.1``, ``max_time_s=300``), so ``optimal_objective`` is the
    returned trajectory's metric.

    Args:
        geometry: Base rocket geometry (its thrust curve is replaced).
        burn_time_s: Total burn time (s), > 0.
        total_impulse_target: Target total impulse (N*s), > 0.
        n_segments: Number of thrust profile segments, 1-50.
        objective: ``"max_altitude"`` (meters), ``"min_max_q"`` (Pa), or
            ``"min_gravity_loss"`` (approximated as ``g0 * burnout_time``,
            m/s; nearly constant at a fixed burn time, so usually the
            constant-thrust start is returned).

    Returns:
        Optimization result; ``optimal_parameters`` holds one
        ``thrust_mult_seg_<i>`` per segment plus the final ``thrust_curve``
        (``[t, thrust]`` pairs; consecutive segments share a time, a step).

    Raises:
        ValueError: If the burn time or impulse target is not positive, the
            segment count is outside 1-50, the objective is unknown, or the
            geometry is invalid.
    """
    if not isinstance(n_segments, int) or not 1 <= n_segments <= 50:
        raise ValueError("n_segments must be an integer between 1 and 50")
    multipliers, optimal_value, initial_value, iterations, converged, curve = (
        _native.optimize_thrust_profile(
            geometry,
            _native_atmosphere(),
            burn_time_s,
            total_impulse_target,
            n_segments,
            objective,
        )
    )
    final_thrust_curve = [[t, thrust] for t, thrust in curve]

    final_geometry = RocketGeometry(
        dry_mass_kg=geometry.dry_mass_kg,
        propellant_mass_kg=geometry.propellant_mass_kg,
        diameter_m=geometry.diameter_m,
        length_m=geometry.length_m,
        cd=geometry.cd,
        thrust_curve=final_thrust_curve,
    )
    final_trajectory = rocket_3dof_trajectory(final_geometry)
    final_performance = analyze_rocket_performance(final_trajectory)

    optimal_params: dict[str, Any] = {
        f"thrust_mult_seg_{i + 1}": mult for i, mult in enumerate(multipliers)
    }
    optimal_params["thrust_curve"] = final_thrust_curve

    sign = -1.0 if objective == "max_altitude" else 1.0
    return TrajectoryOptimizationResult(
        optimal_parameters=optimal_params,
        optimal_objective=sign * optimal_value,
        iterations=iterations,
        converged=converged,
        trajectory_points=final_trajectory,
        performance=final_performance,
        initial_objective=sign * initial_value,
    )


# ===========================================================================
# Trajectory Comparison and Sensitivity Analysis
# ===========================================================================


def compare_trajectories(
    geometries: list[RocketGeometry], names: list[str], launch_angle_deg: float = 90.0
) -> dict[str, Any]:
    """
    Compare multiple rocket trajectories.

    Args:
        geometries: List of rocket geometries to compare
        names: Names for each geometry
        launch_angle_deg: Launch angle for all rockets

    Returns:
        Comparison results dictionary
    """
    results = {}

    for _i, (geometry, name) in enumerate(zip(geometries, names, strict=False)):
        try:
            performance, trajectory = simulate_rocket(
                geometry, launch_angle_deg=launch_angle_deg
            )

            results[name] = {
                "geometry": asdict(geometry),
                "performance": asdict(performance),
                "trajectory_length": len(trajectory),
                "success": True,
            }
        except Exception as e:
            results[name] = {
                "geometry": asdict(geometry),
                "error": str(e),
                "success": False,
            }

    # Add comparison metrics
    if len([r for r in results.values() if r.get("success", False)]) > 1:
        successful_results = [r for r in results.values() if r.get("success", False)]

        # Find best performance in each category
        best_altitude = max(
            r["performance"]["max_altitude_m"] for r in successful_results
        )
        best_velocity = max(
            r["performance"]["max_velocity_ms"] for r in successful_results
        )
        best_efficiency = max(
            r["performance"]["specific_impulse_s"] for r in successful_results
        )

        results["comparison"] = {
            "best_altitude_m": best_altitude,
            "best_velocity_ms": best_velocity,
            "best_efficiency_isp_s": best_efficiency,
            "num_successful": len(successful_results),
            "total_compared": len(geometries),
        }

    return results


# Metrics trajectory_sensitivity_analysis can study, and the geometry
# parameters it can vary.
_SENSITIVITY_OBJECTIVES = {
    "max_altitude": "max_altitude_m",
    "max_velocity": "max_velocity_ms",
    "specific_impulse": "specific_impulse_s",
}
_SENSITIVITY_PARAMETERS = ("dry_mass_kg", "propellant_mass_kg", "diameter_m", "cd")


def _percent_change(value: float, reference: float) -> float | None:
    """100 * (value - reference) / reference, or None if reference is 0."""
    if reference == 0:
        return None
    return (value - reference) / reference * 100


def trajectory_sensitivity_analysis(
    base_geometry: RocketGeometry,
    parameter_variations: dict[str, list[float]],
    objective: str = "max_altitude",
) -> dict[str, Any]:
    """
    Perform sensitivity analysis on trajectory parameters.

    For each variation, ``sensitivity`` is the elasticity
    ``percent_change_objective / percent_change_param``. Percent changes
    relative to a zero baseline are undefined and reported as ``None`` (as
    is the sensitivity then), so the result never contains NaN/Infinity.

    Args:
        base_geometry: Baseline rocket geometry
        parameter_variations: Parameter names (``dry_mass_kg``,
            ``propellant_mass_kg``, ``diameter_m``, ``cd``) mapped to lists
            of values to try
        objective: ``"max_altitude"``, ``"max_velocity"`` or
            ``"specific_impulse"``

    Returns:
        Sensitivity analysis results

    Raises:
        ValueError: For an unknown objective or parameter name, or if the
            baseline geometry cannot be simulated.
    """
    if objective not in _SENSITIVITY_OBJECTIVES:
        raise ValueError(
            f"Unknown objective {objective!r}; use one of "
            + ", ".join(_SENSITIVITY_OBJECTIVES)
        )
    unknown = [p for p in parameter_variations if p not in _SENSITIVITY_PARAMETERS]
    if unknown:
        raise ValueError(
            f"Unknown parameter(s) {', '.join(map(repr, unknown))}; use "
            + ", ".join(_SENSITIVITY_PARAMETERS)
        )
    metric = _SENSITIVITY_OBJECTIVES[objective]

    baseline_performance, _ = simulate_rocket(base_geometry)
    baseline_value = getattr(baseline_performance, metric)

    sensitivity_results = {}

    for param_name, variations in parameter_variations.items():
        param_results = []
        base_param = getattr(base_geometry, param_name)

        for variation in variations:
            modified_geometry = replace(base_geometry, **{param_name: variation})

            try:
                performance, _ = simulate_rocket(modified_geometry)
            except Exception as e:
                param_results.append(
                    {
                        "parameter_value": variation,
                        "error": f"Simulation failed: {e}",
                        "sensitivity": None,
                    }
                )
                continue

            current_value = getattr(performance, metric)
            percent_change_param = _percent_change(variation, base_param)
            percent_change_objective = _percent_change(current_value, baseline_value)
            if percent_change_param == 0:
                sensitivity = 0
            elif percent_change_param is None or percent_change_objective is None:
                sensitivity = None
            else:
                sensitivity = percent_change_objective / percent_change_param

            param_results.append(
                {
                    "parameter_value": variation,
                    "objective_value": current_value,
                    "percent_change_param": percent_change_param,
                    "percent_change_objective": percent_change_objective,
                    "sensitivity": sensitivity,
                }
            )

        sensitivity_results[param_name] = param_results

    return {
        "baseline_value": baseline_value,
        "objective": objective,
        "parameter_sensitivities": sensitivity_results,
        "baseline_geometry": asdict(base_geometry),
    }


# Update availability
try:
    from . import update_availability

    update_availability("trajopt", True, {})
except ImportError:
    pass
