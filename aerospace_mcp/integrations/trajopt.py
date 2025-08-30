"""
Trajectory optimization tools for aerospace MCP.

Provides basic pitch optimization and trajectory analysis for rocket ascent.
Uses lightweight numerical methods with optional advanced optimization libraries.
"""

import math
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

from .rockets import RocketGeometry, analyze_rocket_performance, rocket_3dof_trajectory


@dataclass
class OptimizationParameters:
    """Parameters for trajectory optimization."""
    max_iterations: int = 100
    tolerance: float = 1e-3
    step_size: float = 0.1
    parameter_bounds: dict[str, tuple[float, float]] = None


@dataclass
class TrajectoryOptimizationResult:
    """Result from trajectory optimization."""
    optimal_parameters: dict[str, float]
    optimal_objective: float
    iterations: int
    converged: bool
    trajectory_points: list[Any]  # RocketTrajectoryPoint
    performance: Any  # RocketPerformance


def simple_golden_section_search(
    objective_func: Callable[[float], float],
    lower_bound: float,
    upper_bound: float,
    tolerance: float = 1e-3,
    max_iterations: int = 100
) -> tuple[float, float]:
    """
    Golden section search for 1D optimization.
    
    Returns:
        Tuple of (optimal_x, optimal_value)
    """
    # Golden ratio
    phi = (1 + math.sqrt(5)) / 2
    resphi = 2 - phi

    # Initial points
    x1 = lower_bound + resphi * (upper_bound - lower_bound)
    x2 = upper_bound - resphi * (upper_bound - lower_bound)
    f1 = objective_func(x1)
    f2 = objective_func(x2)

    for i in range(max_iterations):
        if abs(upper_bound - lower_bound) < tolerance:
            break

        if f1 < f2:  # f1 is better (assuming minimization)
            upper_bound = x2
            x2 = x1
            f2 = f1
            x1 = lower_bound + resphi * (upper_bound - lower_bound)
            f1 = objective_func(x1)
        else:  # f2 is better
            lower_bound = x1
            x1 = x2
            f1 = f2
            x2 = upper_bound - resphi * (upper_bound - lower_bound)
            f2 = objective_func(x2)

    # Return best point
    if f1 < f2:
        return x1, f1
    else:
        return x2, f2


def simple_gradient_descent(
    objective_func: Callable[[list[float]], float],
    initial_params: list[float],
    param_bounds: list[tuple[float, float]],
    learning_rate: float = 0.01,
    tolerance: float = 1e-3,
    max_iterations: int = 100
) -> tuple[list[float], float, int, bool]:
    """
    Simple gradient descent optimization for multi-dimensional problems.
    
    Returns:
        Tuple of (optimal_params, optimal_value, iterations, converged)
    """
    current_params = initial_params.copy()
    current_value = objective_func(current_params)

    for iteration in range(max_iterations):
        # Numerical gradient estimation
        gradient = []
        h = 1e-6  # Small step for numerical differentiation

        for i, param in enumerate(current_params):
            # Forward difference
            params_plus = current_params.copy()
            params_plus[i] = min(param_bounds[i][1], param + h)
            value_plus = objective_func(params_plus)

            # Backward difference
            params_minus = current_params.copy()
            params_minus[i] = max(param_bounds[i][0], param - h)
            value_minus = objective_func(params_minus)

            # Central difference
            grad = (value_plus - value_minus) / (2 * h)
            gradient.append(grad)

        # Update parameters
        new_params = []
        max_change = 0.0

        for i, (param, grad) in enumerate(zip(current_params, gradient, strict=False)):
            new_param = param - learning_rate * grad
            # Apply bounds
            new_param = max(param_bounds[i][0], min(param_bounds[i][1], new_param))
            new_params.append(new_param)
            max_change = max(max_change, abs(new_param - param))

        # Check convergence
        new_value = objective_func(new_params)

        if max_change < tolerance and abs(new_value - current_value) < tolerance:
            return new_params, new_value, iteration + 1, True

        current_params = new_params
        current_value = new_value

    return current_params, current_value, max_iterations, False


def optimize_launch_angle(
    geometry: RocketGeometry,
    objective: str = "max_altitude",
    angle_bounds: tuple[float, float] = (80.0, 90.0)
) -> TrajectoryOptimizationResult:
    """
    Optimize launch angle for maximum altitude or range.
    
    Args:
        geometry: Rocket geometry and properties
        objective: "max_altitude" or "max_range"
        angle_bounds: (min_angle_deg, max_angle_deg)
    
    Returns:
        Optimization result
    """
    def objective_function(angle_deg: float) -> float:
        """Objective function to minimize (negative for maximization)."""
        try:
            trajectory = rocket_3dof_trajectory(geometry, launch_angle_deg=angle_deg)
            if not trajectory:
                return float('inf')  # Invalid trajectory

            performance = analyze_rocket_performance(trajectory)

            if objective == "max_altitude":
                return -performance.max_altitude_m  # Negative for maximization
            elif objective == "max_range":
                # Estimate range from final horizontal position (simplified)
                final_horizontal_distance = 0.0  # Placeholder - would need 2D integration
                return -final_horizontal_distance
            else:
                return float('inf')
        except Exception:
            return float('inf')

    # Optimize using golden section search
    optimal_angle, optimal_value = simple_golden_section_search(
        objective_function,
        angle_bounds[0],
        angle_bounds[1],
        tolerance=0.1,  # 0.1 degree tolerance
        max_iterations=50
    )

    # Generate final trajectory with optimal parameters
    final_trajectory = rocket_3dof_trajectory(geometry, launch_angle_deg=optimal_angle)
    final_performance = analyze_rocket_performance(final_trajectory)

    return TrajectoryOptimizationResult(
        optimal_parameters={"launch_angle_deg": optimal_angle},
        optimal_objective=-optimal_value,  # Convert back to positive
        iterations=50,  # Approximation
        converged=True,
        trajectory_points=final_trajectory,
        performance=final_performance
    )


def optimize_thrust_profile(
    geometry: RocketGeometry,
    burn_time_s: float,
    total_impulse_target: float,
    n_segments: int = 5,
    objective: str = "max_altitude"
) -> TrajectoryOptimizationResult:
    """
    Optimize thrust profile for better performance.
    
    Args:
        geometry: Base rocket geometry
        burn_time_s: Total burn time
        total_impulse_target: Target total impulse
        n_segments: Number of thrust profile segments
        objective: "max_altitude", "min_max_q", or "min_gravity_loss"
    
    Returns:
        Optimization result
    """
    # Create initial thrust profile (constant thrust)
    avg_thrust = total_impulse_target / burn_time_s
    segment_time = burn_time_s / n_segments

    # Initial parameters: thrust multipliers for each segment
    initial_multipliers = [1.0] * n_segments
    multiplier_bounds = [(0.1, 3.0)] * n_segments  # 10% to 300% of average

    def objective_function(multipliers: list[float]) -> float:
        """Objective function for thrust profile optimization."""
        try:
            # Create thrust curve from multipliers
            thrust_curve = []
            for i, mult in enumerate(multipliers):
                time_start = i * segment_time
                time_end = (i + 1) * segment_time
                thrust_level = avg_thrust * mult
                thrust_curve.append([time_start, thrust_level])
                if i == len(multipliers) - 1:  # Last segment
                    thrust_curve.append([time_end, 0.0])  # End thrust

            # Normalize to maintain total impulse
            current_impulse = sum(avg_thrust * mult * segment_time for mult in multipliers)
            if current_impulse <= 0:
                return float('inf')

            scale_factor = total_impulse_target / current_impulse
            thrust_curve = [[t, thrust * scale_factor] for t, thrust in thrust_curve]

            # Update geometry with new thrust curve
            opt_geometry = RocketGeometry(
                dry_mass_kg=geometry.dry_mass_kg,
                propellant_mass_kg=geometry.propellant_mass_kg,
                diameter_m=geometry.diameter_m,
                length_m=geometry.length_m,
                cd=geometry.cd,
                thrust_curve=thrust_curve
            )

            # Run trajectory simulation with reasonable parameters
            trajectory = rocket_3dof_trajectory(opt_geometry, dt_s=0.2, max_time_s=150.0)
            if not trajectory:
                return 1e6  # Large penalty instead of inf

            performance = analyze_rocket_performance(trajectory)

            if objective == "max_altitude":
                # Return negative altitude for minimization, but ensure it's not zero
                result = -max(1.0, performance.max_altitude_m)  # At least -1 to avoid zero
                return result
            elif objective == "min_max_q":
                return performance.max_q_pa
            elif objective == "min_gravity_loss":
                # Estimate gravity loss (simplified)
                gravity_loss = 9.80665 * performance.burnout_time_s
                return gravity_loss
            else:
                return 1e6

        except Exception:
            return 1e6  # Large penalty for failed cases

    # Optimize using gradient descent
    optimal_multipliers, optimal_value, iterations, converged = simple_gradient_descent(
        objective_function,
        initial_multipliers,
        multiplier_bounds,
        learning_rate=0.05,
        tolerance=1e-3,
        max_iterations=100
    )

    # Generate final thrust curve and trajectory
    final_thrust_curve = []
    for i, mult in enumerate(optimal_multipliers):
        time_start = i * segment_time
        time_end = (i + 1) * segment_time
        thrust_level = avg_thrust * mult
        final_thrust_curve.append([time_start, thrust_level])
        if i == len(optimal_multipliers) - 1:
            final_thrust_curve.append([time_end, 0.0])

    # Normalize final curve
    current_impulse = sum(avg_thrust * mult * segment_time for mult in optimal_multipliers)
    scale_factor = total_impulse_target / current_impulse
    final_thrust_curve = [[t, thrust * scale_factor] for t, thrust in final_thrust_curve]

    # Generate final trajectory
    final_geometry = RocketGeometry(
        dry_mass_kg=geometry.dry_mass_kg,
        propellant_mass_kg=geometry.propellant_mass_kg,
        diameter_m=geometry.diameter_m,
        length_m=geometry.length_m,
        cd=geometry.cd,
        thrust_curve=final_thrust_curve
    )

    final_trajectory = rocket_3dof_trajectory(final_geometry)
    final_performance = analyze_rocket_performance(final_trajectory)

    # Prepare optimal parameters
    optimal_params = {f"thrust_mult_seg_{i+1}": mult for i, mult in enumerate(optimal_multipliers)}
    optimal_params["thrust_curve"] = final_thrust_curve

    return TrajectoryOptimizationResult(
        optimal_parameters=optimal_params,
        optimal_objective=-optimal_value if objective == "max_altitude" else optimal_value,
        iterations=iterations,
        converged=converged,
        trajectory_points=final_trajectory,
        performance=final_performance
    )


def compare_trajectories(
    geometries: list[RocketGeometry],
    names: list[str],
    launch_angle_deg: float = 90.0
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

    for i, (geometry, name) in enumerate(zip(geometries, names, strict=False)):
        try:
            trajectory = rocket_3dof_trajectory(geometry, launch_angle_deg=launch_angle_deg)
            performance = analyze_rocket_performance(trajectory)

            results[name] = {
                "geometry": asdict(geometry),
                "performance": asdict(performance),
                "trajectory_length": len(trajectory),
                "success": True
            }
        except Exception as e:
            results[name] = {
                "geometry": asdict(geometry),
                "error": str(e),
                "success": False
            }

    # Add comparison metrics
    if len([r for r in results.values() if r.get("success", False)]) > 1:
        successful_results = [r for r in results.values() if r.get("success", False)]

        # Find best performance in each category
        best_altitude = max(r["performance"]["max_altitude_m"] for r in successful_results)
        best_velocity = max(r["performance"]["max_velocity_ms"] for r in successful_results)
        best_efficiency = max(r["performance"]["specific_impulse_s"] for r in successful_results)

        results["comparison"] = {
            "best_altitude_m": best_altitude,
            "best_velocity_ms": best_velocity,
            "best_efficiency_isp_s": best_efficiency,
            "num_successful": len(successful_results),
            "total_compared": len(geometries)
        }

    return results


def trajectory_sensitivity_analysis(
    base_geometry: RocketGeometry,
    parameter_variations: dict[str, list[float]],
    objective: str = "max_altitude"
) -> dict[str, Any]:
    """
    Perform sensitivity analysis on trajectory parameters.
    
    Args:
        base_geometry: Baseline rocket geometry
        parameter_variations: Dict with parameter names and variation values
        objective: Objective to analyze sensitivity for
    
    Returns:
        Sensitivity analysis results
    """
    baseline_trajectory = rocket_3dof_trajectory(base_geometry)
    baseline_performance = analyze_rocket_performance(baseline_trajectory)

    if objective == "max_altitude":
        baseline_value = baseline_performance.max_altitude_m
    elif objective == "max_velocity":
        baseline_value = baseline_performance.max_velocity_ms
    elif objective == "specific_impulse":
        baseline_value = baseline_performance.specific_impulse_s
    else:
        baseline_value = baseline_performance.max_altitude_m

    sensitivity_results = {}

    for param_name, variations in parameter_variations.items():
        param_results = []

        for variation in variations:
            # Create modified geometry
            modified_geometry = RocketGeometry(
                dry_mass_kg=base_geometry.dry_mass_kg,
                propellant_mass_kg=base_geometry.propellant_mass_kg,
                diameter_m=base_geometry.diameter_m,
                length_m=base_geometry.length_m,
                cd=base_geometry.cd,
                thrust_curve=base_geometry.thrust_curve
            )

            # Apply variation
            if param_name == "dry_mass_kg":
                modified_geometry.dry_mass_kg = variation
            elif param_name == "propellant_mass_kg":
                modified_geometry.propellant_mass_kg = variation
            elif param_name == "diameter_m":
                modified_geometry.diameter_m = variation
            elif param_name == "cd":
                modified_geometry.cd = variation

            try:
                trajectory = rocket_3dof_trajectory(modified_geometry)
                performance = analyze_rocket_performance(trajectory)

                if objective == "max_altitude":
                    current_value = performance.max_altitude_m
                elif objective == "max_velocity":
                    current_value = performance.max_velocity_ms
                elif objective == "specific_impulse":
                    current_value = performance.specific_impulse_s
                else:
                    current_value = performance.max_altitude_m

                # Calculate sensitivity
                percent_change_param = (variation - getattr(base_geometry, param_name)) / getattr(base_geometry, param_name) * 100
                percent_change_objective = (current_value - baseline_value) / baseline_value * 100
                sensitivity = percent_change_objective / percent_change_param if percent_change_param != 0 else 0

                param_results.append({
                    "parameter_value": variation,
                    "objective_value": current_value,
                    "percent_change_param": percent_change_param,
                    "percent_change_objective": percent_change_objective,
                    "sensitivity": sensitivity
                })
            except Exception:
                param_results.append({
                    "parameter_value": variation,
                    "error": "Simulation failed",
                    "sensitivity": None
                })

        sensitivity_results[param_name] = param_results

    return {
        "baseline_value": baseline_value,
        "objective": objective,
        "parameter_sensitivities": sensitivity_results,
        "baseline_geometry": asdict(base_geometry)
    }


# Update availability
try:
    from . import update_availability
    update_availability("trajopt", True, {})
except ImportError:
    pass
