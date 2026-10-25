"""Guidance, Navigation, and Control (GNC) tools for aerospace MCP.

Provides trajectory optimization using metaheuristic algorithms (Genetic
Algorithm, Particle Swarm Optimization), trajectory cost evaluation, and
Monte Carlo uncertainty analysis for spacecraft mission planning.

Key capabilities:
    - Trajectory cost evaluation with constraint penalty formulation
    - Genetic Algorithm (GA) optimizer with tournament selection, single-point
      crossover, and elitism
    - Particle Swarm Optimization (PSO) with inertia-weight velocity update
    - Monte Carlo uncertainty analysis for dispersion characterization
    - A dictionary API (``run_genetic_algorithm``, ``run_particle_swarm``,
      ``run_monte_carlo``) with input validation and size limits, used by
      the MCP optimization tools

References:
    - Goldberg, D.E., "Genetic Algorithms in Search, Optimization, and
      Machine Learning" (1989)
    - Kennedy, J. & Eberhart, R., "Particle Swarm Optimization" (1995)
    - Battin, R.H., "An Introduction to the Mathematics and Methods of
      Astrodynamics" (1999) -- for trajectory cost formulations

WARNING: This module is for educational and research purposes only.
Do NOT use for real flight planning, navigation, or spacecraft operations.
"""

import dataclasses
import math
import random
import statistics
import time
from dataclasses import dataclass
from typing import Any

# ===========================================================================
# Data Classes -- Trajectory and Optimization Configuration
# ===========================================================================


@dataclass
class TrajectoryWaypoint:
    """Single waypoint along a spacecraft trajectory.

    Attributes:
        time_s: Time since epoch in seconds.
        position_m: Position vector [x, y, z] in meters (inertial frame).
        velocity_ms: Velocity vector [vx, vy, vz] in m/s.
        acceleration_ms2: Acceleration vector [ax, ay, az] in m/s^2.
        thrust_n: Thrust vector [Fx, Fy, Fz] in Newtons.
        mass_kg: Spacecraft wet mass at this waypoint in kg.
    """

    time_s: float
    position_m: list[float]  # [x, y, z] position
    velocity_ms: list[float]  # [vx, vy, vz] velocity
    acceleration_ms2: list[float] = None  # [ax, ay, az] acceleration
    thrust_n: list[float] = None  # [Fx, Fy, Fz] thrust vector
    mass_kg: float = 1000.0  # Spacecraft mass


@dataclass
class OptimizationConstraints:
    """Inequality constraints for trajectory optimization.

    These define the feasible region of the optimization problem.
    Violations are penalized quadratically in the cost function.

    Attributes:
        max_thrust_n: Maximum allowable thrust magnitude (N).
        max_acceleration_ms2: Maximum allowable acceleration (m/s^2).
        min_altitude_m: Minimum altitude above Earth surface (m).
        max_delta_v_ms: Maximum total delta-V budget (m/s).
        time_bounds_s: Tuple of (start_time, end_time) in seconds.
    """

    max_thrust_n: float = 10000.0  # Maximum thrust magnitude
    max_acceleration_ms2: float = 50.0  # Maximum acceleration
    min_altitude_m: float = 200000.0  # Minimum altitude above Earth
    max_delta_v_ms: float = 5000.0  # Maximum total delta-V
    time_bounds_s: tuple[float, float] = (0.0, 86400.0)  # Time bounds


@dataclass
class OptimizationObjective:
    """Objective function specification for trajectory optimization.

    Attributes:
        type: Objective type string. One of ``"minimize_fuel"``,
            ``"minimize_time"``, ``"minimize_delta_v"``, or
            ``"maximize_payload"``.
        weights: Optional dictionary of objective weights for multi-objective.
        target_state: Optional target state vector for rendezvous problems.
    """

    type: str = "minimize_fuel"  # minimize_fuel, minimize_time, maximize_payload
    weights: dict[str, float] = None  # Objective weights
    target_state: list[float] = None  # Target state if applicable


@dataclass
class GeneticAlgorithmParams:
    """Configuration parameters for Genetic Algorithm (GA) optimization.

    The GA uses tournament selection (size 3), single-point crossover,
    and elitism to evolve a population of candidate trajectories.

    Attributes:
        population_size: Number of individuals per generation.
        generations: Maximum number of generations.
        mutation_rate: Probability of mutating each waypoint (0 to 1).
        crossover_rate: Probability of performing crossover vs. cloning.
        elite_size: Number of best individuals preserved unchanged.
        convergence_tolerance: Improvement threshold for early stopping.
    """

    population_size: int = 50
    generations: int = 100
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    elite_size: int = 5
    convergence_tolerance: float = 1e-6


@dataclass
class ParticleSwarmParams:
    """Configuration parameters for Particle Swarm Optimization (PSO).

    PSO velocity update equation::

        v_i(t+1) = w * v_i(t)
                  + c1 * r1 * (p_best_i - x_i(t))    # cognitive term
                  + c2 * r2 * (g_best - x_i(t))       # social term

    where r1, r2 are uniform random numbers in [0, 1].

    Attributes:
        num_particles: Number of particles in the swarm.
        max_iterations: Maximum number of PSO iterations.
        w: Inertia weight -- controls momentum of previous velocity.
        c1: Cognitive acceleration coefficient -- attraction to personal best.
        c2: Social acceleration coefficient -- attraction to global best.
        convergence_tolerance: Tolerance for early-stopping based on score spread.
    """

    num_particles: int = 30
    max_iterations: int = 100
    w: float = 0.7  # Inertia weight
    c1: float = 1.5  # Cognitive component
    c2: float = 1.5  # Social component
    convergence_tolerance: float = 1e-6


@dataclass
class OptimizationResult:
    """Output from a trajectory optimization run.

    Attributes:
        optimal_trajectory: Best trajectory found.
        optimal_cost: Cost function value of the best trajectory.
        delta_v_total_ms: Total delta-V of the optimal trajectory (m/s).
        fuel_mass_kg: Estimated fuel consumed (kg).
        converged: Whether the optimizer converged.
        iterations: Number of iterations/generations executed.
        computation_time_s: Wall-clock computation time in seconds.
        algorithm: Name of the optimization algorithm used.
    """

    optimal_trajectory: list[TrajectoryWaypoint]
    optimal_cost: float
    delta_v_total_ms: float
    fuel_mass_kg: float
    converged: bool
    iterations: int
    computation_time_s: float
    algorithm: str


# ===========================================================================
# Vector and Numeric Utility Functions
# ===========================================================================


def vector_magnitude(vec: list[float]) -> float:
    """Calculate the Euclidean (L2) norm of a vector.

    Args:
        vec: Input vector of arbitrary dimension.

    Returns:
        Scalar magnitude ||vec||.
    """
    return math.sqrt(sum(x**2 for x in vec))


def vector_distance(a: list[float], b: list[float]) -> float:
    """Calculate the Euclidean distance between two vectors.

    Args:
        a: First vector.
        b: Second vector (same length as *a*).

    Returns:
        Distance ||a - b||.
    """
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(len(a))))


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp a scalar value to the range [min_val, max_val].

    Args:
        value: Input value.
        min_val: Lower bound.
        max_val: Upper bound.

    Returns:
        Clamped value.
    """
    return max(min_val, min(max_val, value))


# ===========================================================================
# Trajectory Cost Evaluation
# ===========================================================================


def evaluate_trajectory_cost(
    trajectory: list[TrajectoryWaypoint],
    objective: OptimizationObjective,
    constraints: OptimizationConstraints,
) -> float:
    """Evaluate the augmented cost function for a candidate trajectory.

    The total cost is the sum of the objective function value and a
    quadratic penalty for constraint violations::

        J = J_objective + lambda * sum(max(0, g_i(x))^2)

    where *g_i* are the inequality constraint functions and *lambda* is
    a large penalty weight (1000).

    Fuel consumption is approximated using the Tsiolkovsky rocket
    equation linearized for small burns::

        dm = F * dt / (Isp * g0)

    where F is thrust magnitude, dt is the time step, Isp is the
    assumed specific impulse (300 s), and g0 = 9.80665 m/s^2.

    Args:
        trajectory: List of trajectory waypoints (at least 2 required).
        objective: Optimization objective specification.
        constraints: Feasibility constraints.

    Returns:
        Scalar cost value (lower is better). Returns ``inf`` for invalid
        trajectories.
    """
    if not trajectory or len(trajectory) < 2:
        return float("inf")

    cost = 0.0
    penalty = 0.0

    # Accumulate delta-V and fuel consumption along trajectory segments
    total_delta_v = 0.0
    total_fuel = 0.0

    for i in range(1, len(trajectory)):
        dt = trajectory[i].time_s - trajectory[i - 1].time_s
        if dt <= 0:
            return float("inf")  # Non-causal time ordering

        # Thrust-based delta-V: dv = (F / m) * dt
        if trajectory[i].thrust_n:
            thrust_mag = vector_magnitude(trajectory[i].thrust_n)
            if trajectory[i].mass_kg > 0:
                accel = thrust_mag / trajectory[i].mass_kg
                dv = accel * dt
                total_delta_v += dv

                # Fuel consumption from Tsiolkovsky approximation:
                # dm = F * dt / (Isp * g0)
                if thrust_mag > 0:
                    isp = 300.0  # Assumed specific impulse (s)
                    dm = thrust_mag * dt / (isp * 9.80665)
                    total_fuel += dm

        # Quadratic penalty for thrust magnitude constraint violation
        if trajectory[i].thrust_n:
            thrust_mag = vector_magnitude(trajectory[i].thrust_n)
            if thrust_mag > constraints.max_thrust_n:
                penalty += (thrust_mag - constraints.max_thrust_n) ** 2 * 1e-6

        # Quadratic penalty for minimum altitude constraint violation
        pos_mag = vector_magnitude(trajectory[i].position_m)
        altitude = pos_mag - 6.378137e6  # Subtract Earth equatorial radius (m)
        if altitude < constraints.min_altitude_m:
            penalty += (constraints.min_altitude_m - altitude) ** 2 * 1e-12

    # Quadratic penalty for total delta-V budget exceedance
    if total_delta_v > constraints.max_delta_v_ms:
        penalty += (total_delta_v - constraints.max_delta_v_ms) ** 2 * 1e-6

    # Compute objective function value based on selected type
    if objective.type == "minimize_fuel":
        cost = total_fuel
    elif objective.type == "minimize_time":
        cost = trajectory[-1].time_s - trajectory[0].time_s
    elif objective.type == "minimize_delta_v":
        cost = total_delta_v
    elif objective.type == "maximize_payload":
        # Negate for minimization: maximize (final_mass - fuel_burned)
        final_mass = trajectory[-1].mass_kg
        cost = -(final_mass - total_fuel)
    else:
        cost = total_delta_v  # Default: minimize delta-V

    # Augmented Lagrangian-style penalty: high weight to enforce feasibility
    cost += penalty * 1000

    return cost


def trajectory_delta_v_and_fuel(
    trajectory: list[TrajectoryWaypoint], isp_s: float = 300.0
) -> tuple[float, float]:
    """Total thrust delta-V and fuel mass of a trajectory.

    Uses the same per-segment model as :func:`evaluate_trajectory_cost`:
    the thrust at waypoint *i* acts over the segment (i-1, i), giving
    ``dv = |F| / m * dt`` and ``dm = |F| * dt / (Isp * g0)``.

    Args:
        trajectory: Trajectory waypoints.
        isp_s: Assumed specific impulse in seconds.

    Returns:
        Tuple ``(delta_v_ms, fuel_kg)``.
    """
    delta_v_total = 0.0
    fuel_mass = 0.0
    for i in range(1, len(trajectory)):
        if trajectory[i].thrust_n:
            thrust_mag = vector_magnitude(trajectory[i].thrust_n)
            dt = trajectory[i].time_s - trajectory[i - 1].time_s
            if trajectory[i].mass_kg > 0:
                delta_v_total += thrust_mag / trajectory[i].mass_kg * dt
                if thrust_mag > 0:
                    fuel_mass += thrust_mag * dt / (isp_s * 9.80665)
    return delta_v_total, fuel_mass


# ===========================================================================
# Genetic Algorithm (GA) Optimizer
# ===========================================================================


class GeneticAlgorithm:
    """Genetic Algorithm for trajectory optimization.

    Implements a standard GA with:
        - **Tournament selection** (size 3): pick 3 random individuals,
          select the fittest.
        - **Single-point crossover**: splice two parent trajectories at
          a random waypoint index.
        - **Mutation**: randomly perturb thrust vectors with probability
          ``mutation_rate``.
        - **Elitism**: preserve the top ``elite_size`` individuals
          unchanged each generation.
        - **Early stopping**: halt when no improvement is seen for 20
          consecutive generations (reported as ``converged=True``).

    Attributes:
        params: GA configuration parameters.
        population: Current generation of candidate trajectories.
        fitness_scores: Cost function values for each individual.
        rng: Random number source (the ``random`` module unless a seeded
            ``random.Random`` is passed in).
    """

    #: Generations without improvement that trigger early stopping.
    STAGNATION_LIMIT = 20

    def __init__(self, params: GeneticAlgorithmParams, rng: Any = None):
        self.params = params
        self.population = []
        self.fitness_scores = []
        self.rng = rng if rng is not None else random

    def _random_thrust(self, max_thrust_n: float) -> list[float]:
        """Random thrust vector; in-plane magnitude uniform in [0, max_thrust_n]."""
        thrust_mag = self.rng.uniform(0, max_thrust_n)
        thrust_dir = self.rng.uniform(0, 2 * math.pi)
        return [
            thrust_mag * math.cos(thrust_dir),
            thrust_mag * math.sin(thrust_dir),
            self.rng.uniform(-thrust_mag / 2, thrust_mag / 2),
        ]

    def random_trajectory(
        self,
        n_waypoints: int,
        constraints: OptimizationConstraints,
        template: list[TrajectoryWaypoint] | None = None,
    ) -> list[TrajectoryWaypoint]:
        """Generate a random trajectory for population initialization.

        With a ``template`` (the user's trajectory) only the thrust vectors
        are randomised; times, positions, velocities and masses are kept, so
        every individual describes the same mission.  Without a template,
        times are evenly spaced over ``constraints.time_bounds_s``, positions
        are sampled in spherical coordinates around low-Earth orbit, and
        velocities approximate circular orbital speed with perturbations.

        Args:
            n_waypoints: Number of waypoints in the trajectory.
            constraints: Optimization constraints for bound enforcement.
            template: Optional trajectory whose structure is reused.

        Returns:
            Randomly generated trajectory.
        """
        if template is not None:
            return [
                dataclasses.replace(
                    wp, thrust_n=self._random_thrust(constraints.max_thrust_n)
                )
                for wp in template
            ]

        trajectory = []
        t0, t1 = constraints.time_bounds_s

        for i in range(n_waypoints):
            # Evenly spaced, strictly increasing times over the time bounds
            t = t0 + (t1 - t0) * i / (n_waypoints - 1) if n_waypoints > 1 else t0

            # Random position (around Earth orbit)
            r = self.rng.uniform(
                constraints.min_altitude_m + 6.378137e6,
                constraints.min_altitude_m + 6.378137e6 + 1e6,
            )
            theta = self.rng.uniform(0, 2 * math.pi)
            phi = self.rng.uniform(0, math.pi)

            pos = [
                r * math.sin(phi) * math.cos(theta),
                r * math.sin(phi) * math.sin(theta),
                r * math.cos(phi),
            ]

            # Random velocity near circular orbital speed: v_circ = sqrt(mu/r)
            # Perturbed by +/- 20% to explore non-circular trajectories
            v_mag = math.sqrt(3.986004418e14 / r) * self.rng.uniform(0.8, 1.2)
            v_theta = self.rng.uniform(0, 2 * math.pi)
            vel = [
                v_mag * math.cos(v_theta),
                v_mag * math.sin(v_theta),
                self.rng.uniform(-1000, 1000),
            ]

            trajectory.append(
                TrajectoryWaypoint(
                    time_s=t,
                    position_m=pos,
                    velocity_ms=vel,
                    thrust_n=self._random_thrust(constraints.max_thrust_n),
                    mass_kg=self.rng.uniform(500, 2000),
                )
            )

        return trajectory

    def crossover(
        self, parent1: list[TrajectoryWaypoint], parent2: list[TrajectoryWaypoint]
    ) -> list[TrajectoryWaypoint]:
        """Single-point crossover between two parent trajectories.

        A random crossover point is selected; the child inherits waypoints
        0..point-1 from parent1 and point..end from parent2.

        Args:
            parent1: First parent trajectory.
            parent2: Second parent trajectory.

        Returns:
            Child trajectory combining segments of both parents.
        """
        if len(parent1) != len(parent2) or len(parent1) < 2:
            return parent1  # Nothing to splice (length mismatch or 1 waypoint)

        child = []
        crossover_point = self.rng.randint(1, len(parent1) - 1)

        for i in range(len(parent1)):
            if i < crossover_point:
                child.append(parent1[i])
            else:
                child.append(parent2[i])

        return child

    def mutate(
        self, trajectory: list[TrajectoryWaypoint], constraints: OptimizationConstraints
    ) -> list[TrajectoryWaypoint]:
        """Apply random mutation to thrust vectors in a trajectory.

        Each waypoint is mutated with probability ``self.params.mutation_rate``.
        Mutation randomly scales the thrust magnitude by 0.8-1.2x and
        randomizes the thrust direction.  A waypoint without thrust
        (``thrust_n=None``) is treated as zero thrust.

        Args:
            trajectory: Input trajectory to mutate.
            constraints: Used to clamp thrust within allowed bounds.

        Returns:
            Mutated copy of the trajectory.
        """
        mutated = []

        for waypoint in trajectory:
            if self.rng.random() < self.params.mutation_rate:
                # Mutate thrust
                thrust_mag = (
                    vector_magnitude(waypoint.thrust_n) if waypoint.thrust_n else 0.0
                )
                thrust_mag *= self.rng.uniform(0.8, 1.2)
                thrust_mag = clamp(thrust_mag, 0, constraints.max_thrust_n)

                thrust_dir = self.rng.uniform(0, 2 * math.pi)
                new_thrust = [
                    thrust_mag * math.cos(thrust_dir),
                    thrust_mag * math.sin(thrust_dir),
                    self.rng.uniform(-thrust_mag / 2, thrust_mag / 2),
                ]

                # Keep every other field (including acceleration_ms2)
                mutated.append(dataclasses.replace(waypoint, thrust_n=new_thrust))
            else:
                mutated.append(waypoint)

        return mutated

    def optimize(
        self,
        initial_trajectory: list[TrajectoryWaypoint],
        objective: OptimizationObjective,
        constraints: OptimizationConstraints,
    ) -> OptimizationResult:
        """Execute the full GA optimization loop.

        Algorithm:
            1. Initialize the population with random thrust profiles on the
               initial trajectory's time/position/velocity/mass structure
               (the initial guess itself is individual 0).
            2. For each generation: evaluate fitness, apply elitism,
               tournament selection, crossover, and mutation.
            3. Stop on convergence (20 generations without improvement)
               or max generations reached.

        Args:
            initial_trajectory: Initial guess trajectory (included in
                the initial population).
            objective: Optimization objective specification.
            constraints: Feasibility constraints.

        Returns:
            Optimization result with best trajectory and metrics.
            ``converged`` is True when the run stopped because the best
            cost stagnated, and ``iterations`` is the number of generations
            actually evaluated.

        Raises:
            ValueError: If the population settings are inconsistent.
        """
        if self.params.population_size < 1:
            raise ValueError("population_size must be at least 1")
        if not 0 <= self.params.elite_size <= self.params.population_size:
            raise ValueError("elite_size must be between 0 and population_size")
        if not initial_trajectory:
            raise ValueError("initial_trajectory must contain at least one waypoint")

        start_time = time.time()

        # Initialize population
        n_waypoints = len(initial_trajectory)
        self.population = [
            self.random_trajectory(
                n_waypoints, constraints, template=initial_trajectory
            )
            for _ in range(self.params.population_size)
        ]
        self.population[0] = initial_trajectory  # Include initial guess
        tournament = min(3, self.params.population_size)

        best_cost = float("inf")
        best_trajectory = initial_trajectory
        generations_without_improvement = 0
        generations_run = 0
        converged = False

        for _generation in range(self.params.generations):
            generations_run += 1
            # Evaluate fitness
            self.fitness_scores = [
                evaluate_trajectory_cost(traj, objective, constraints)
                for traj in self.population
            ]

            # Find best
            min_idx = min(
                range(len(self.fitness_scores)), key=lambda i: self.fitness_scores[i]
            )
            current_best_cost = self.fitness_scores[min_idx]

            if current_best_cost < best_cost:
                best_cost = current_best_cost
                best_trajectory = self.population[min_idx]
                generations_without_improvement = 0
            else:
                generations_without_improvement += 1

            # Check convergence
            if generations_without_improvement > self.STAGNATION_LIMIT:
                converged = True
                break

            # Selection and reproduction
            new_population = []

            # Elitism - keep best individuals
            sorted_indices = sorted(
                range(len(self.fitness_scores)), key=lambda i: self.fitness_scores[i]
            )
            for i in range(self.params.elite_size):
                new_population.append(self.population[sorted_indices[i]])

            # Generate offspring
            while len(new_population) < self.params.population_size:
                # Tournament selection
                parent1_idx = min(
                    self.rng.sample(range(len(self.population)), tournament),
                    key=lambda i: self.fitness_scores[i],
                )
                parent2_idx = min(
                    self.rng.sample(range(len(self.population)), tournament),
                    key=lambda i: self.fitness_scores[i],
                )

                parent1 = self.population[parent1_idx]
                parent2 = self.population[parent2_idx]

                # Crossover
                if self.rng.random() < self.params.crossover_rate:
                    child = self.crossover(parent1, parent2)
                else:
                    child = parent1 if self.rng.random() < 0.5 else parent2

                # Mutation
                child = self.mutate(child, constraints)

                new_population.append(child)

            self.population = new_population

        computation_time = time.time() - start_time
        delta_v_total, fuel_mass = trajectory_delta_v_and_fuel(best_trajectory)

        return OptimizationResult(
            optimal_trajectory=best_trajectory,
            optimal_cost=best_cost,
            delta_v_total_ms=delta_v_total,
            fuel_mass_kg=fuel_mass,
            converged=converged,
            iterations=generations_run,
            computation_time_s=computation_time,
            algorithm="Genetic Algorithm",
        )


# ===========================================================================
# Particle Swarm Optimization (PSO)
# ===========================================================================


class ParticleSwarmOptimizer:
    """Particle Swarm Optimization for trajectory thrust profile design.

    The decision variables are the thrust vector components at each
    waypoint, flattened into a single particle vector of dimension
    ``3 * n_waypoints``.

    PSO velocity update (applied per-dimension)::

        v_i(t+1) = w * v_i(t)
                  + c1 * r1 * (pbest_i - x_i(t))   # cognitive (personal best)
                  + c2 * r2 * (gbest   - x_i(t))   # social   (global best)

    Position update::

        x_i(t+1) = x_i(t) + v_i(t+1)

    Attributes:
        params: PSO configuration parameters.
        particles: Current positions of all particles.
        velocities: Current velocities of all particles.
        personal_best: Best position each particle has visited.
        personal_best_scores: Best cost for each particle.
        global_best: Best position found by any particle.
        global_best_score: Best cost found by any particle.
        best_score_history: Global best cost after each iteration.
        rng: Random number source (the ``random`` module unless a seeded
            ``random.Random`` is passed in).
    """

    #: Number of iterations over which the global best must improve by more
    #: than ``convergence_tolerance`` to keep going.
    CONVERGENCE_WINDOW = 10

    def __init__(self, params: ParticleSwarmParams, rng: Any = None):
        self.params = params
        self.rng = rng if rng is not None else random
        self.particles = []
        self.velocities = []
        self.personal_best = []
        self.personal_best_scores = []
        self.global_best = None
        self.global_best_score = float("inf")
        self.best_score_history: list[float] = []

    def optimize(
        self,
        initial_trajectory: list[TrajectoryWaypoint],
        objective: OptimizationObjective,
        constraints: OptimizationConstraints,
    ) -> OptimizationResult:
        """Execute the PSO optimization loop.

        Each particle encodes a complete thrust profile as a flat vector
        of thrust components [Fx0, Fy0, Fz0, Fx1, ...].  The initial
        trajectory structure (times, positions, velocities, masses) is
        preserved; only the thrust vectors are optimized.  Particle 0
        starts at the initial trajectory's thrust profile; particle
        positions are kept within +/- ``max_thrust_n`` per component.

        The run stops early (``converged=True``) when the global best cost
        improved by less than ``convergence_tolerance`` over the last
        ``CONVERGENCE_WINDOW`` iterations.

        Args:
            initial_trajectory: Reference trajectory defining the
                time/position/velocity/mass structure.
            objective: Optimization objective specification.
            constraints: Feasibility constraints.

        Returns:
            Optimization result with the best trajectory found.

        Raises:
            ValueError: If the swarm settings are inconsistent.
        """
        if self.params.num_particles < 1:
            raise ValueError("num_particles must be at least 1")
        if self.params.max_iterations < 1:
            raise ValueError("max_iterations must be at least 1")
        if not initial_trajectory:
            raise ValueError("initial_trajectory must contain at least one waypoint")

        start_time = time.time()

        n_waypoints = len(initial_trajectory)
        n_dimensions = n_waypoints * 3  # 3 thrust components per waypoint
        bound = constraints.max_thrust_n

        # Fresh state for every run (an optimizer instance may be reused)
        self.particles = []
        self.velocities = []
        self.global_best = None
        self.global_best_score = float("inf")
        self.best_score_history = []

        # Initialize particles as thrust profiles; particle 0 is the
        # initial guess (missing thrust = zero).
        initial_particle = []
        for wp in initial_trajectory:
            initial_particle.extend(
                clamp(float(c), -bound, bound) for c in (wp.thrust_n or [0.0] * 3)
            )
        for k in range(self.params.num_particles):
            if k == 0:
                particle = list(initial_particle)
            else:
                particle = [
                    self.rng.uniform(-bound, bound) for _ in range(n_dimensions)
                ]
            self.particles.append(particle)
            self.velocities.append(
                [self.rng.uniform(-100, 100) for _ in range(n_dimensions)]
            )

        # Initialize personal bests
        self.personal_best = [p.copy() for p in self.particles]
        self.personal_best_scores = [float("inf")] * self.params.num_particles

        best_cost = float("inf")
        best_trajectory = initial_trajectory
        converged = False
        iterations_run = 0

        for _iteration in range(self.params.max_iterations):
            iterations_run += 1
            for i, particle in enumerate(self.particles):
                # Convert particle to trajectory
                trajectory = []

                for j in range(n_waypoints):
                    thrust_idx = j * 3
                    thrust = [
                        particle[thrust_idx],
                        particle[thrust_idx + 1],
                        particle[thrust_idx + 2],
                    ]

                    # Clamp thrust magnitude
                    thrust_mag = vector_magnitude(thrust)
                    if thrust_mag > constraints.max_thrust_n:
                        scale = constraints.max_thrust_n / thrust_mag
                        thrust = [t * scale for t in thrust]

                    # Use initial trajectory structure with new thrust
                    trajectory.append(
                        dataclasses.replace(initial_trajectory[j], thrust_n=thrust)
                    )

                # Evaluate fitness
                cost = evaluate_trajectory_cost(trajectory, objective, constraints)

                # Update personal best
                if cost < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = cost
                    self.personal_best[i] = particle.copy()

                # Update global best
                if cost < self.global_best_score or self.global_best is None:
                    self.global_best_score = cost
                    self.global_best = particle.copy()
                    best_cost = cost
                    best_trajectory = trajectory

            self.best_score_history.append(self.global_best_score)

            # Check convergence: improvement of the global best over the
            # last CONVERGENCE_WINDOW iterations.
            window = self.CONVERGENCE_WINDOW
            if len(self.best_score_history) > window:
                previous = self.best_score_history[-window - 1]
                if math.isinf(self.global_best_score):
                    # Cost is inf only for structurally invalid trajectories
                    # (non-increasing times), which thrust cannot fix.
                    break
                if (
                    previous - self.global_best_score
                    < self.params.convergence_tolerance
                ):
                    converged = True
                    break

            # Update particle velocities and positions using the PSO equations
            for i in range(len(self.particles)):
                for j in range(n_dimensions):
                    # Random coefficients for stochastic exploration
                    r1, r2 = self.rng.random(), self.rng.random()

                    # Cognitive term: attraction toward personal best position
                    cognitive = (
                        self.params.c1
                        * r1
                        * (self.personal_best[i][j] - self.particles[i][j])
                    )
                    # Social term: attraction toward swarm-wide global best
                    social = (
                        self.params.c2
                        * r2
                        * (self.global_best[j] - self.particles[i][j])
                    )

                    # Velocity update: v = w*v + cognitive + social
                    self.velocities[i][j] = (
                        self.params.w * self.velocities[i][j] + cognitive + social
                    )

                    # Position update: x = x + v, kept inside the search box
                    self.particles[i][j] = clamp(
                        self.particles[i][j] + self.velocities[i][j], -bound, bound
                    )

        computation_time = time.time() - start_time
        delta_v_total, fuel_mass = trajectory_delta_v_and_fuel(best_trajectory)

        return OptimizationResult(
            optimal_trajectory=best_trajectory,
            optimal_cost=best_cost,
            delta_v_total_ms=delta_v_total,
            fuel_mass_kg=fuel_mass,
            converged=converged,
            iterations=iterations_run,
            computation_time_s=computation_time,
            algorithm="Particle Swarm Optimization",
        )


# ===========================================================================
# Monte Carlo Uncertainty Analysis
# ===========================================================================


def _percentile(sorted_values: list[float], q: float) -> float:
    """Linear-interpolation percentile (NumPy's default method) of sorted data."""
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = (len(sorted_values) - 1) * q / 100.0
    lo = math.floor(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


def _summary(values: list[float], unit: str) -> dict[str, float]:
    """Mean, sample std, min, max and the 2.5/97.5 percentiles of a sample."""
    ordered = sorted(values)
    return {
        f"mean_{unit}": statistics.fmean(values),
        f"std_{unit}": statistics.stdev(values),
        f"min_{unit}": ordered[0],
        f"max_{unit}": ordered[-1],
        f"p2_5_{unit}": _percentile(ordered, 2.5),
        f"p97_5_{unit}": _percentile(ordered, 97.5),
    }


def monte_carlo_uncertainty_analysis(
    nominal_trajectory: list[TrajectoryWaypoint],
    uncertainty_params: dict[str, dict[str, float]],
    n_samples: int = 1000,
    rng: Any = None,
) -> dict[str, Any]:
    """Perform Monte Carlo uncertainty analysis on a trajectory.

    Each sample perturbs every waypoint of the nominal trajectory by
    independent zero-mean Gaussian errors on each component of position,
    velocity, and thrust, then recomputes the metrics.  No dynamics are
    propagated: the delta-V dispersion comes from the thrust errors, and the
    final position/velocity errors are the errors injected at the last
    waypoint.  Results include mean, sample standard deviation, extremes and
    empirical 95% intervals (2.5th-97.5th percentiles).

    Args:
        nominal_trajectory: Nominal (unperturbed) trajectory.
        uncertainty_params: Dictionary mapping parameter names to
            ``{"std": value}`` dictionaries specifying 1-sigma
            uncertainties per vector component.  Supported keys:
            ``"position_m"``, ``"velocity_ms"``, ``"thrust_n"``.  A missing
            key means no uncertainty in that quantity.
        n_samples: Number of Monte Carlo samples to draw (at least 2).
        rng: Optional random source (e.g. a seeded ``random.Random``);
            defaults to the ``random`` module.

    Returns:
        Dictionary with delta-V, flight time, final position and velocity
        error statistics, plus 95% intervals.
    """
    rng = rng if rng is not None else random
    if n_samples < 2:
        return {"error": "Insufficient data for statistics", "n_samples": n_samples}
    if not nominal_trajectory:
        return {"error": "nominal_trajectory is empty", "n_samples": n_samples}

    unknown = set(uncertainty_params) - {"position_m", "velocity_ms", "thrust_n"}
    if unknown:
        raise ValueError(
            f"Unsupported uncertainty parameters {sorted(unknown)}; expected "
            "position_m, velocity_ms, thrust_n"
        )
    sigma = {
        key: float(uncertainty_params.get(key, {"std": 0.0})["std"])
        for key in ("position_m", "velocity_ms", "thrust_n")
    }
    if any(not math.isfinite(s) or s < 0 for s in sigma.values()):
        raise ValueError("uncertainty std values must be finite and non-negative")

    delta_v_samples = []
    flight_time_samples = []
    final_pos_errors = []
    final_vel_errors = []
    nominal_dv, _ = trajectory_delta_v_and_fuel(nominal_trajectory)

    for _ in range(n_samples):
        perturbed_trajectory = []

        for waypoint in nominal_trajectory:
            pos_err = [rng.gauss(0, sigma["position_m"]) for _ in range(3)]
            vel_err = [rng.gauss(0, sigma["velocity_ms"]) for _ in range(3)]
            perturbed_thrust = (
                [c + rng.gauss(0, sigma["thrust_n"]) for c in waypoint.thrust_n]
                if waypoint.thrust_n
                else None
            )
            perturbed_trajectory.append(
                dataclasses.replace(
                    waypoint,
                    position_m=[
                        p + e for p, e in zip(waypoint.position_m, pos_err, strict=True)
                    ],
                    velocity_ms=[
                        v + e
                        for v, e in zip(waypoint.velocity_ms, vel_err, strict=True)
                    ],
                    thrust_n=perturbed_thrust,
                )
            )

        total_delta_v, _ = trajectory_delta_v_and_fuel(perturbed_trajectory)
        delta_v_samples.append(total_delta_v)
        flight_time_samples.append(
            perturbed_trajectory[-1].time_s - perturbed_trajectory[0].time_s
        )
        # pos_err / vel_err still hold the errors drawn for the last waypoint
        final_pos_errors.append(vector_magnitude(pos_err))
        final_vel_errors.append(vector_magnitude(vel_err))

    dv = _summary(delta_v_samples, "ms")
    pos = _summary(final_pos_errors, "m")
    return {
        "n_samples": n_samples,
        "uncertainty_std": sigma,
        "nominal_delta_v_ms": nominal_dv,
        "delta_v_statistics": dv,
        "flight_time_statistics": _summary(flight_time_samples, "s"),
        "position_error_statistics": pos,
        "velocity_error_statistics": _summary(final_vel_errors, "ms"),
        "confidence_intervals": {
            # Empirical 2.5th-97.5th percentiles
            "delta_v_95_ms": [dv["p2_5_ms"], dv["p97_5_ms"]],
            "position_95_m": [pos["p2_5_m"], pos["p97_5_m"]],
        },
        "notes": [
            "Errors are independent per waypoint and per component; no "
            "dynamics are propagated",
            "Final position/velocity errors are the magnitudes of the "
            "errors injected at the last waypoint",
        ],
    }


# ===========================================================================
# Dictionary API used by the MCP optimization tools
# ===========================================================================

#: Upper bounds that keep a single tool call bounded in time and memory.
MAX_WAYPOINTS = 100
MAX_POPULATION = 200
MAX_GENERATIONS = 500
MAX_PARTICLES = 200
MAX_PSO_ITERATIONS = 500
MAX_MC_SAMPLES = 50_000
#: Cap on (individuals x generations x waypoints) cost-function work.
MAX_OPTIMIZER_WORK = 500_000
#: Cap on (samples x waypoints) for Monte Carlo.
MAX_MC_WORK = 250_000

_OBJECTIVES = ("minimize_fuel", "minimize_time", "minimize_delta_v", "maximize_payload")


def _vec3(value: Any, name: str) -> list[float]:
    """Validate a 3-vector of finite numbers."""
    if not isinstance(value, list | tuple) or len(value) != 3:
        raise ValueError(f"{name} must be a list of 3 numbers")
    try:
        out = [float(v) for v in value]
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a list of 3 numbers") from None
    if not all(math.isfinite(v) for v in out):
        raise ValueError(f"{name} must be finite")
    return out


def trajectory_from_dicts(
    waypoints: Any, name: str = "trajectory"
) -> list[TrajectoryWaypoint]:
    """Build and validate a trajectory from a list of waypoint dicts.

    Each waypoint is ``{"time_s": float, "position_m": [x, y, z],
    "velocity_ms": [vx, vy, vz], "thrust_n": [Fx, Fy, Fz] (optional),
    "mass_kg": float (optional, default 1000), "acceleration_ms2": [..]
    (optional)}``.  Times must be strictly increasing.

    Raises:
        ValueError: On missing keys, bad values or too many waypoints.
    """
    if not isinstance(waypoints, list) or not waypoints:
        raise ValueError(f"{name} must be a non-empty list of waypoint dicts")
    if len(waypoints) > MAX_WAYPOINTS:
        raise ValueError(
            f"{name} has {len(waypoints)} waypoints; at most {MAX_WAYPOINTS}"
        )
    allowed = {
        "time_s",
        "position_m",
        "velocity_ms",
        "thrust_n",
        "mass_kg",
        "acceleration_ms2",
    }
    out = []
    for i, wp in enumerate(waypoints):
        if not isinstance(wp, dict):
            raise ValueError(f"{name}[{i}] must be a dict")
        missing = {"time_s", "position_m", "velocity_ms"} - set(wp)
        if missing:
            raise ValueError(f"{name}[{i}] is missing {sorted(missing)}")
        extra = set(wp) - allowed
        if extra:
            raise ValueError(f"{name}[{i}] has unknown keys {sorted(extra)}")
        try:
            t = float(wp["time_s"])
            mass = float(wp.get("mass_kg", 1000.0))
        except (TypeError, ValueError):
            raise ValueError(
                f"{name}[{i}].time_s and mass_kg must be numbers"
            ) from None
        if not math.isfinite(t):
            raise ValueError(f"{name}[{i}].time_s must be finite")
        if not (math.isfinite(mass) and mass > 0):
            raise ValueError(f"{name}[{i}].mass_kg must be positive")
        if out and t <= out[-1].time_s:
            raise ValueError(f"{name} times must be strictly increasing (waypoint {i})")
        out.append(
            TrajectoryWaypoint(
                time_s=t,
                position_m=_vec3(wp["position_m"], f"{name}[{i}].position_m"),
                velocity_ms=_vec3(wp["velocity_ms"], f"{name}[{i}].velocity_ms"),
                acceleration_ms2=(
                    _vec3(wp["acceleration_ms2"], f"{name}[{i}].acceleration_ms2")
                    if wp.get("acceleration_ms2") is not None
                    else None
                ),
                thrust_n=(
                    _vec3(wp["thrust_n"], f"{name}[{i}].thrust_n")
                    if wp.get("thrust_n") is not None
                    else None
                ),
                mass_kg=mass,
            )
        )
    return out


def _json_safe(obj: Any) -> Any:
    """Recursively replace non-finite floats (inf/NaN) with None."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_json_safe(v) for v in obj]
    return obj


def _parse_problem(
    problem: dict,
) -> tuple[list[TrajectoryWaypoint], OptimizationObjective, OptimizationConstraints]:
    """Parse the ``optimization_problem`` dict (see the tool docstrings)."""
    if not isinstance(problem, dict):
        raise ValueError("optimization_problem must be a dict")
    extra = set(problem) - {"trajectory", "objective", "constraints"}
    if extra:
        raise ValueError(f"optimization_problem has unknown keys {sorted(extra)}")
    if "trajectory" not in problem:
        raise ValueError(
            "optimization_problem needs a 'trajectory': a list of waypoints "
            "{time_s, position_m, velocity_ms, thrust_n?, mass_kg?}"
        )
    trajectory = trajectory_from_dicts(problem["trajectory"])
    if len(trajectory) < 2:
        raise ValueError("trajectory needs at least 2 waypoints")

    objective = problem.get("objective", "minimize_fuel")
    if isinstance(objective, dict):
        objective = objective.get("type", "minimize_fuel")
    if objective not in _OBJECTIVES:
        raise ValueError(
            f"objective must be one of {list(_OBJECTIVES)}, got {objective!r}"
        )

    c = dict(problem.get("constraints") or {})
    fields = {f.name for f in dataclasses.fields(OptimizationConstraints)}
    extra = set(c) - fields
    if extra:
        raise ValueError(
            f"constraints has unknown keys {sorted(extra)}; expected {sorted(fields)}"
        )
    if "time_bounds_s" in c:
        c["time_bounds_s"] = tuple(float(v) for v in c["time_bounds_s"])
        if len(c["time_bounds_s"]) != 2:
            raise ValueError("constraints.time_bounds_s must be [start, end]")
    for key in fields - {"time_bounds_s"}:
        if key in c:
            c[key] = float(c[key])
            if not math.isfinite(c[key]) or c[key] < 0:
                raise ValueError(f"constraints.{key} must be finite and non-negative")
    return (
        trajectory,
        OptimizationObjective(type=objective),
        OptimizationConstraints(**c),
    )


def _take(params: dict, key: str, aliases: tuple[str, ...], default: Any) -> Any:
    """Pop ``key`` (or one of its aliases) from ``params``."""
    for k in (key, *aliases):
        if k in params:
            return params.pop(k)
    return default


def _bounded_int(value: Any, name: str, lo: int, hi: int) -> int:
    """Validate an integer parameter within [lo, hi]."""
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or value != int(value)
    ):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not lo <= value <= hi:
        raise ValueError(f"{name} must be between {lo} and {hi}, got {value}")
    return value


def _bounded_float(value: Any, name: str, lo: float, hi: float) -> float:
    """Validate a float parameter within [lo, hi]."""
    value = float(value)
    if not (math.isfinite(value) and lo <= value <= hi):
        raise ValueError(f"{name} must be between {lo} and {hi}, got {value}")
    return value


def _make_rng(seed: Any) -> Any:
    """A seeded ``random.Random`` or the shared ``random`` module."""
    if seed is None:
        return random
    return random.Random(_bounded_int(seed, "seed", 0, 2**63 - 1))


def _result_to_dict(
    result: OptimizationResult,
    initial: list[TrajectoryWaypoint],
    objective: OptimizationObjective,
    constraints: OptimizationConstraints,
    parameters: dict,
) -> dict[str, Any]:
    """Serialise an OptimizationResult for the tool layer."""
    initial_cost = evaluate_trajectory_cost(initial, objective, constraints)
    initial_dv, initial_fuel = trajectory_delta_v_and_fuel(initial)
    return _json_safe(
        {
            "algorithm": result.algorithm,
            "objective": objective.type,
            "parameters": parameters,
            "converged": result.converged,
            "iterations": result.iterations,
            "optimal_cost": result.optimal_cost,
            "initial_cost": initial_cost,
            "delta_v_total_ms": result.delta_v_total_ms,
            "fuel_mass_kg": result.fuel_mass_kg,
            "initial_delta_v_ms": initial_dv,
            "initial_fuel_mass_kg": initial_fuel,
            "computation_time_s": result.computation_time_s,
            "optimal_trajectory": [
                dataclasses.asdict(wp) for wp in result.optimal_trajectory
            ],
            "notes": [
                "Only thrust vectors are optimised; waypoint times, positions, "
                "velocities and masses are kept from the input trajectory",
                "Cost = objective + 1000 x quadratic constraint penalties; "
                "fuel uses Isp = 300 s",
            ],
        }
    )


def _check_work(units: int, cap: int, what: str) -> None:
    """Reject runs whose total work exceeds ``cap``."""
    if units > cap:
        raise ValueError(
            f"{what} = {units} exceeds the limit of {cap}; reduce the run size"
        )


def run_genetic_algorithm(problem: dict, params: dict | None = None) -> dict[str, Any]:
    """Run the GA on an ``optimization_problem`` dict and return a JSON-safe dict.

    Args:
        problem: ``{"trajectory": [...], "objective": str, "constraints": {...}}``.
        params: Optional GA settings: population_size (3-200, default 50),
            generations (1-500, default 100), mutation_rate (0-1, default
            0.1), crossover_rate (0-1, default 0.8), elite_size (default 5),
            seed (int, optional).

    Raises:
        ValueError: On invalid input or a run larger than the limits.
    """
    trajectory, objective, constraints = _parse_problem(problem)
    p = dict(params or {})
    population = _bounded_int(
        _take(p, "population_size", ("population",), 50),
        "population_size",
        3,
        MAX_POPULATION,
    )
    generations = _bounded_int(
        _take(p, "generations", ("max_generations",), 100),
        "generations",
        1,
        MAX_GENERATIONS,
    )
    mutation = _bounded_float(
        _take(p, "mutation_rate", (), 0.1), "mutation_rate", 0.0, 1.0
    )
    crossover = _bounded_float(
        _take(p, "crossover_rate", (), 0.8), "crossover_rate", 0.0, 1.0
    )
    elite = _bounded_int(
        _take(p, "elite_size", (), min(5, population)), "elite_size", 0, population
    )
    seed = _take(p, "seed", (), None)
    if p:
        raise ValueError(f"Unknown GA parameters {sorted(p)}")
    _check_work(
        population * generations * len(trajectory),
        MAX_OPTIMIZER_WORK,
        "population_size x generations x waypoints",
    )

    ga_params = GeneticAlgorithmParams(
        population_size=population,
        generations=generations,
        mutation_rate=mutation,
        crossover_rate=crossover,
        elite_size=elite,
    )
    result = GeneticAlgorithm(ga_params, rng=_make_rng(seed)).optimize(
        trajectory, objective, constraints
    )
    return _result_to_dict(
        result,
        trajectory,
        objective,
        constraints,
        {**dataclasses.asdict(ga_params), "seed": seed},
    )


def run_particle_swarm(problem: dict, params: dict | None = None) -> dict[str, Any]:
    """Run PSO on an ``optimization_problem`` dict and return a JSON-safe dict.

    Args:
        problem: ``{"trajectory": [...], "objective": str, "constraints": {...}}``.
        params: Optional PSO settings: num_particles / n_particles (1-200,
            default 30), max_iterations / iterations (1-500, default 100),
            w (0-2, default 0.7), c1 / c2 (0-4, default 1.5),
            convergence_tolerance (default 1e-6), seed (int, optional).

    Raises:
        ValueError: On invalid input or a run larger than the limits.
    """
    trajectory, objective, constraints = _parse_problem(problem)
    p = dict(params or {})
    particles = _bounded_int(
        _take(p, "num_particles", ("n_particles", "swarm_size"), 30),
        "num_particles",
        1,
        MAX_PARTICLES,
    )
    iterations = _bounded_int(
        _take(p, "max_iterations", ("iterations",), 100),
        "max_iterations",
        1,
        MAX_PSO_ITERATIONS,
    )
    w = _bounded_float(_take(p, "w", ("inertia_weight",), 0.7), "w", 0.0, 2.0)
    c1 = _bounded_float(_take(p, "c1", (), 1.5), "c1", 0.0, 4.0)
    c2 = _bounded_float(_take(p, "c2", (), 1.5), "c2", 0.0, 4.0)
    tol = _bounded_float(
        _take(p, "convergence_tolerance", (), 1e-6), "convergence_tolerance", 0.0, 1e12
    )
    seed = _take(p, "seed", (), None)
    if p:
        raise ValueError(f"Unknown PSO parameters {sorted(p)}")
    _check_work(
        particles * iterations * len(trajectory),
        MAX_OPTIMIZER_WORK,
        "num_particles x max_iterations x waypoints",
    )

    pso_params = ParticleSwarmParams(
        num_particles=particles,
        max_iterations=iterations,
        w=w,
        c1=c1,
        c2=c2,
        convergence_tolerance=tol,
    )
    result = ParticleSwarmOptimizer(pso_params, rng=_make_rng(seed)).optimize(
        trajectory, objective, constraints
    )
    return _result_to_dict(
        result,
        trajectory,
        objective,
        constraints,
        {**dataclasses.asdict(pso_params), "seed": seed},
    )


def run_monte_carlo(
    nominal_trajectory: dict,
    uncertainty_parameters: dict,
    n_samples: int = 1000,
    options: dict | None = None,
) -> dict[str, Any]:
    """Monte Carlo dispersion of a trajectory given as dicts.

    Args:
        nominal_trajectory: ``{"trajectory": [waypoint dicts]}`` (see
            :func:`trajectory_from_dicts`).
        uncertainty_parameters: ``{"position_m": {"std": s}, "velocity_ms":
            {"std": s}, "thrust_n": {"std": s}}``; a bare number is accepted
            in place of ``{"std": s}``.  Missing keys mean no uncertainty.
        n_samples: Number of samples (2-50,000).
        options: Optional ``{"seed": int}``.

    Raises:
        ValueError: On invalid input or a run larger than the limits.
    """
    if (
        not isinstance(nominal_trajectory, dict)
        or "trajectory" not in nominal_trajectory
    ):
        raise ValueError(
            "nominal_trajectory must be {'trajectory': [waypoints]} with waypoints "
            "{time_s, position_m, velocity_ms, thrust_n?, mass_kg?}"
        )
    extra = set(nominal_trajectory) - {"trajectory"}
    if extra:
        raise ValueError(f"nominal_trajectory has unknown keys {sorted(extra)}")
    trajectory = trajectory_from_dicts(nominal_trajectory["trajectory"])
    n_samples = _bounded_int(n_samples, "n_samples", 2, MAX_MC_SAMPLES)
    _check_work(n_samples * len(trajectory), MAX_MC_WORK, "n_samples x waypoints")

    if not isinstance(uncertainty_parameters, dict):
        raise ValueError("uncertainty_parameters must be a dict")
    sigma = {}
    for key, value in uncertainty_parameters.items():
        if isinstance(value, dict):
            if set(value) != {"std"}:
                raise ValueError(
                    f"uncertainty_parameters.{key} must be {{'std': value}}"
                )
            sigma[key] = {"std": float(value["std"])}
        else:
            sigma[key] = {"std": float(value)}

    opts = dict(options or {})
    seed = opts.pop("seed", None)
    if opts:
        raise ValueError(f"Unknown analysis_options {sorted(opts)}; supported: seed")

    result = monte_carlo_uncertainty_analysis(
        trajectory, sigma, n_samples, rng=_make_rng(seed)
    )
    if "error" in result:
        raise ValueError(result["error"])
    result["seed"] = seed
    return _json_safe(result)


# Update availability
try:
    from . import update_availability

    update_availability("gnc", True, {})
except ImportError:
    pass
