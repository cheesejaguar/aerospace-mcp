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
                "objective": objective,
                "iterations": result.iterations,
                "converged": result.converged,
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
        parameter_variations: Parameters to vary (dry_mass_kg,
            propellant_mass_kg, diameter_m, cd) mapped to lists of values
        analysis_options: Optional analysis settings: ``objective`` is one of
            "max_altitude" (default), "max_velocity", "specific_impulse"

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

        objective = (analysis_options or {}).get("objective", "max_altitude")
        result = _sensitivity(geometry, parameter_variations, objective=objective)

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
        optimization_problem: Problem definition::

            {
              "trajectory": [waypoint, ...],          # required, see below
              "objective": "minimize_fuel",           # or minimize_delta_v,
                                                      # minimize_time,
                                                      # maximize_payload
              "constraints": {                        # all optional
                "max_thrust_n": 10000, "min_altitude_m": 200000,
                "max_delta_v_ms": 5000, "max_acceleration_ms2": 50,
                "time_bounds_s": [0, 86400]
              }
            }

            Each waypoint is ``{"time_s": float, "position_m": [x, y, z],
            "velocity_ms": [vx, vy, vz], "thrust_n": [Fx, Fy, Fz]
            (optional), "mass_kg": float (optional, default 1000)}`` in SI
            units (s, m, m/s, N, kg) with Earth-centred positions (the
            altitude constraint uses |r| - 6378.137 km); 2-100 waypoints
            with strictly increasing times.
        ga_parameters: Optional GA parameters: population_size (3-200,
            default 50), generations (1-500, default 100), mutation_rate
            (0-1, default 0.1), crossover_rate (0-1, default 0.8),
            elite_size (default 5), seed (int, for reproducible runs).
            population_size x generations x waypoints is capped at 500,000.

    Returns:
        JSON string with optimization results: optimal_cost (and the
        initial trajectory's cost for comparison), delta-V, fuel mass,
        iterations (generations actually run), converged (True when the
        best cost stagnated for 20 generations) and the optimal trajectory.

    Raises:
        No exceptions are raised directly; errors are returned as formatted strings.

    Note:
        Only the thrust vectors are optimised; times, positions, velocities
        and masses come from the input trajectory, so ``minimize_time`` only
        reacts to constraint penalties.  The GA operates on a population of
        candidate thrust profiles through:
        1. **Selection**: tournament selection (size 3) of parents.
        2. **Crossover**: single-point crossover at a random waypoint.
        3. **Mutation**: random scaling (0.8-1.2x) and re-pointing of thrust
           with probability mutation_rate.
        4. **Elitism**: the best elite_size individuals survive unchanged.
    """
    try:
        from ..integrations.gnc import run_genetic_algorithm as _ga_optimize

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
        optimization_problem: Problem definition, same schema as
            ``genetic_algorithm_optimization``: ``{"trajectory": [waypoint,
            ...], "objective": str, "constraints": {...}}``.
            Each waypoint is ``{"time_s": float, "position_m": [x, y, z],
            "velocity_ms": [vx, vy, vz], "thrust_n": [Fx, Fy, Fz]
            (optional), "mass_kg": float (optional, default 1000)}`` in SI
            units (s, m, m/s, N, kg) with Earth-centred positions (the
            altitude constraint uses |r| - 6378.137 km); 2-100 waypoints
            with strictly increasing times.
        pso_parameters: Optional PSO parameters: num_particles (alias
            n_particles; 1-200, default 30), max_iterations (alias
            iterations; 1-500, default 100), w (inertia, default 0.7), c1 /
            c2 (default 1.5), convergence_tolerance (default 1e-6), seed
            (int, for reproducible runs).  num_particles x max_iterations x
            waypoints is capped at 500,000.

    Returns:
        JSON string with optimization results: optimal_cost (and the
        initial trajectory's cost), delta-V, fuel mass, iterations actually
        run, converged (True when the global best improved by less than
        convergence_tolerance over 10 iterations) and the optimal trajectory.

    Raises:
        No exceptions are raised directly; errors are returned as formatted strings.

    Note:
        Each particle is a thrust profile [Fx0, Fy0, Fz0, Fx1, ...] on the
        input trajectory's times/positions/velocities/masses.  PSO updates
        each particle's velocity and position at every iteration:
            v_i(t+1) = w*v_i(t) + c1*r1*(p_best_i - x_i) + c2*r2*(g_best - x_i)
            x_i(t+1) = x_i(t) + v_i(t+1)
        where w is the inertia weight (balances exploration vs exploitation),
        c1/c2 are cognitive/social acceleration coefficients, r1/r2 are random
        numbers in [0,1], p_best_i is the particle's personal best, and g_best
        is the global best found by any particle in the swarm.
    """
    try:
        from ..integrations.gnc import run_particle_swarm as _pso_optimize

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
        departure_date_range: Departure dates to evaluate (ISO format)
        arrival_date_range: Arrival dates to evaluate (ISO format)
        analysis_options: Optional analysis settings: ``min_tof_days``
            (default 100) and ``max_tof_days`` (default 400)

    Returns:
        JSON string with porkchop plot data (departure date vs arrival date
        grid with delta-V contours for identifying optimal launch windows).

    Raises:
        No exceptions are raised directly; errors are returned as formatted strings.
    """
    try:
        from ..integrations.orbits import porkchop_plot_analysis as _porkchop

        options = analysis_options or {}
        result = _porkchop(
            departure_body,
            arrival_body,
            departure_dates=departure_date_range or None,
            arrival_dates=arrival_date_range or None,
            min_tof_days=options.get("min_tof_days", 100),
            max_tof_days=options.get("max_tof_days", 400),
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
        nominal_trajectory: ``{"trajectory": [waypoint, ...]}``.  Each
            waypoint is ``{"time_s": float, "position_m": [x, y, z],
            "velocity_ms": [vx, vy, vz], "thrust_n": [Fx, Fy, Fz]
            (optional), "mass_kg": float (optional, default 1000)}`` in SI
            units; 1-100 waypoints with strictly increasing times.
        uncertainty_parameters: 1-sigma Gaussian errors per vector
            component, e.g. ``{"position_m": {"std": 100.0},
            "velocity_ms": {"std": 1.0}, "thrust_n": {"std": 50.0}}`` (a
            bare number may replace ``{"std": ...}``).  Missing keys mean no
            uncertainty in that quantity.
        n_samples: Number of Monte Carlo samples (2-50,000; samples x
            waypoints is capped at 250,000)
        analysis_options: Optional settings: ``seed`` (int) for
            reproducible sampling

    Returns:
        JSON string with uncertainty analysis results: mean, sample std,
        min/max and 2.5/97.5 percentiles of total delta-V, flight time and
        the final-waypoint position and velocity error magnitudes.

    Raises:
        No exceptions are raised directly; errors are returned as formatted strings.

    Note:
        Errors are drawn independently for every waypoint and component
        (plain random sampling) and no dynamics are propagated: the delta-V
        dispersion comes from the thrust errors (dv = |F| / m * dt per
        segment), and the position/velocity statistics describe the errors
        injected at the final waypoint.  The 95% intervals are empirical
        percentiles, not mean +/- 1.96 sigma.
    """
    try:
        from ..integrations.gnc import run_monte_carlo as _monte_carlo

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
