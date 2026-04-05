"""Trajectory optimization tools for the Aerospace MCP server.

Provides tools for trajectory and mission optimization using multiple methods:
- Direct thrust profile optimization for rocket trajectories.
- Sensitivity analysis (parameter variation studies).
- Genetic Algorithm (GA): evolutionary optimization with crossover/mutation.
- Particle Swarm Optimization (PSO): swarm-intelligence global search.
- Porkchop plot generation for interplanetary transfer window analysis.
- Monte Carlo uncertainty quantification via random sampling.

WARNING: This module is for educational and research purposes only.
Do NOT use for real flight planning, navigation, or aircraft operations.
"""

import json
import logging
from typing import Literal

logger = logging.getLogger(__name__)


def optimize_thrust_profile(
    rocket_geometry: dict,
    burn_time_s: float,
    total_impulse_target: float,
    n_segments: int = 5,
    objective: Literal[
        "max_altitude", "min_max_q", "min_gravity_loss"
    ] = "max_altitude",
) -> str:
    """Optimize rocket thrust profile for better performance using trajectory optimization.

    Args:
        rocket_geometry: Rocket geometry parameters
        burn_time_s: Burn time in seconds
        total_impulse_target: Target total impulse in N⋅s
        n_segments: Number of thrust segments
        objective: Optimization objective

    Returns:
        JSON string with optimized thrust profile including segment-wise thrust
        levels and resulting trajectory performance.

    Raises:
        No exceptions are raised directly; errors are returned as formatted strings.
    """
    try:
        from ..integrations.trajopt import (
            RocketGeometry,
        )
        from ..integrations.trajopt import (
            optimize_thrust_profile as _optimize,
        )

        geometry = RocketGeometry(**rocket_geometry)

        result = _optimize(
            geometry, burn_time_s, total_impulse_target, n_segments, objective
        )

        return json.dumps(
            {
                "optimization_result": result.optimization_result,
                "optimal_objective": result.optimal_objective,
                "optimal_parameters": result.optimal_parameters,
                "performance": {
                    "max_altitude_m": result.performance.max_altitude_m,
                    "max_velocity_ms": result.performance.max_velocity_ms,
                    "total_impulse_ns": result.performance.total_impulse_ns,
                    "apogee_time_s": result.performance.apogee_time_s,
                },
            },
            indent=2,
        )

    except ImportError:
        return "Trajectory optimization not available - install optimization packages"
    except Exception as e:
        logger.error(f"Thrust profile optimization error: {str(e)}", exc_info=True)
        return f"Thrust profile optimization error: {str(e)}"


def trajectory_sensitivity_analysis(
    rocket_geometry: dict,
    parameter_variations: dict,
    analysis_options: dict | None = None,
) -> str:
    """Perform sensitivity analysis on rocket trajectory parameters.

    Args:
        rocket_geometry: Baseline rocket geometry
        parameter_variations: Parameters to vary and their ranges
        analysis_options: Optional analysis settings

    Returns:
        JSON string with sensitivity analysis results showing how each parameter
        variation affects the trajectory outcome.

    Raises:
        No exceptions are raised directly; errors are returned as formatted strings.
    """
    try:
        from ..integrations.trajopt import (
            RocketGeometry,
        )
        from ..integrations.trajopt import (
            trajectory_sensitivity_analysis as _sensitivity,
        )

        geometry = RocketGeometry(**rocket_geometry)

        result = _sensitivity(geometry, parameter_variations, analysis_options or {})

        return json.dumps(result, indent=2)

    except ImportError:
        return "Sensitivity analysis not available - install optimization packages"
    except Exception as e:
        logger.error(f"Sensitivity analysis error: {str(e)}", exc_info=True)
        return f"Sensitivity analysis error: {str(e)}"


def genetic_algorithm_optimization(
    optimization_problem: dict, ga_parameters: dict | None = None
) -> str:
    """Optimize spacecraft trajectory using genetic algorithm.

    Args:
        optimization_problem: Problem definition (objective, constraints, variables)
        ga_parameters: Optional GA parameters (population_size, generations, etc.)

    Returns:
        JSON string with optimization results including best solution found,
        convergence history, and final objective value.

    Raises:
        No exceptions are raised directly; errors are returned as formatted strings.

    Note:
        The GA operates on a population of candidate solutions through:
        1. **Selection**: Tournament or roulette-wheel selection of parents.
        2. **Crossover**: Combining parent chromosomes (e.g., single-point or
           uniform crossover) to produce offspring that inherit traits from both.
        3. **Mutation**: Random perturbation of offspring genes with probability
           p_mutation to maintain diversity and avoid premature convergence.
        Each generation evaluates fitness, selects the best, and breeds the
        next generation until convergence or max generations reached.
    """
    try:
        from ..integrations.trajopt import (
            genetic_algorithm_optimization as _ga_optimize,
        )

        result = _ga_optimize(optimization_problem, ga_parameters or {})

        return json.dumps(result, indent=2)

    except ImportError:
        return "Genetic algorithm optimization not available - install optimization packages"
    except Exception as e:
        logger.error(f"GA optimization error: {str(e)}", exc_info=True)
        return f"GA optimization error: {str(e)}"


def particle_swarm_optimization(
    optimization_problem: dict, pso_parameters: dict | None = None
) -> str:
    """Optimize spacecraft trajectory using particle swarm optimization.

    Args:
        optimization_problem: Problem definition (objective, constraints, variables)
        pso_parameters: Optional PSO parameters (n_particles, iterations, etc.)

    Returns:
        JSON string with optimization results including best position found
        and convergence metrics.

    Raises:
        No exceptions are raised directly; errors are returned as formatted strings.

    Note:
        PSO updates each particle's velocity and position at every iteration:
            v_i(t+1) = w*v_i(t) + c1*r1*(p_best_i - x_i) + c2*r2*(g_best - x_i)
            x_i(t+1) = x_i(t) + v_i(t+1)
        where w is the inertia weight (balances exploration vs exploitation),
        c1/c2 are cognitive/social acceleration coefficients, r1/r2 are random
        numbers in [0,1], p_best_i is the particle's personal best, and g_best
        is the global best found by any particle in the swarm.
    """
    try:
        from ..integrations.trajopt import particle_swarm_optimization as _pso_optimize

        result = _pso_optimize(optimization_problem, pso_parameters or {})

        return json.dumps(result, indent=2)

    except ImportError:
        return (
            "Particle swarm optimization not available - install optimization packages"
        )
    except Exception as e:
        logger.error(f"PSO optimization error: {str(e)}", exc_info=True)
        return f"PSO optimization error: {str(e)}"


def porkchop_plot_analysis(
    departure_body: str,
    arrival_body: str,
    departure_date_range: list[str],
    arrival_date_range: list[str],
    analysis_options: dict | None = None,
) -> str:
    """Generate porkchop plot for interplanetary transfer opportunities.

    Args:
        departure_body: Departure celestial body name
        arrival_body: Arrival celestial body name
        departure_date_range: Range of departure dates (ISO format)
        arrival_date_range: Range of arrival dates (ISO format)
        analysis_options: Optional analysis settings

    Returns:
        JSON string with porkchop plot data (departure date vs arrival date
        grid with delta-V contours for identifying optimal launch windows).

    Raises:
        No exceptions are raised directly; errors are returned as formatted strings.
    """
    try:
        from ..integrations.trajopt import porkchop_plot_analysis as _porkchop

        result = _porkchop(
            departure_body,
            arrival_body,
            departure_date_range,
            arrival_date_range,
            analysis_options or {},
        )

        return json.dumps(result, indent=2)

    except ImportError:
        return (
            "Porkchop plot analysis not available - install space trajectory packages"
        )
    except Exception as e:
        logger.error(f"Porkchop plot error: {str(e)}", exc_info=True)
        return f"Porkchop plot error: {str(e)}"


def monte_carlo_uncertainty_analysis(
    nominal_trajectory: dict,
    uncertainty_parameters: dict,
    n_samples: int = 1000,
    analysis_options: dict | None = None,
) -> str:
    """Perform Monte Carlo uncertainty analysis on spacecraft trajectory.

    Args:
        nominal_trajectory: Nominal trajectory parameters
        uncertainty_parameters: Parameters with uncertainty distributions
        n_samples: Number of Monte Carlo samples
        analysis_options: Optional analysis settings

    Returns:
        JSON string with uncertainty analysis results including statistical
        summaries (mean, std, percentiles) of trajectory metrics.

    Raises:
        No exceptions are raised directly; errors are returned as formatted strings.

    Note:
        Monte Carlo analysis samples uncertain parameters from their specified
        distributions (e.g., Gaussian, uniform) and runs n_samples trajectory
        simulations. Statistical analysis of the results provides:
        - Mean and standard deviation of key performance metrics.
        - Confidence intervals (e.g., 95th percentile bounds).
        - Dispersion ellipses for correlated output parameters.
        Latin Hypercube Sampling (LHS) may be used for efficient coverage of
        the parameter space with fewer samples than pure random sampling.
    """
    try:
        from ..integrations.trajopt import (
            monte_carlo_uncertainty_analysis as _monte_carlo,
        )

        result = _monte_carlo(
            nominal_trajectory,
            uncertainty_parameters,
            n_samples,
            analysis_options or {},
        )

        return json.dumps(result, indent=2)

    except ImportError:
        return "Monte Carlo analysis not available - install optimization packages"
    except Exception as e:
        logger.error(f"Monte Carlo analysis error: {str(e)}", exc_info=True)
        return f"Monte Carlo analysis error: {str(e)}"
