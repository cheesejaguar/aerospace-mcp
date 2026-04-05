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

References:
    - Goldberg, D.E., "Genetic Algorithms in Search, Optimization, and
      Machine Learning" (1989)
    - Kennedy, J. & Eberhart, R., "Particle Swarm Optimization" (1995)
    - Battin, R.H., "An Introduction to the Mathematics and Methods of
      Astrodynamics" (1999) -- for trajectory cost formulations

WARNING: This module is for educational and research purposes only.
Do NOT use for real flight planning, navigation, or spacecraft operations.
"""

import math
import random
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
          consecutive generations.

    Attributes:
        params: GA configuration parameters.
        population: Current generation of candidate trajectories.
        fitness_scores: Cost function values for each individual.
    """

    def __init__(self, params: GeneticAlgorithmParams):
        self.params = params
        self.population = []
        self.fitness_scores = []

    def random_trajectory(
        self, n_waypoints: int, constraints: OptimizationConstraints
    ) -> list[TrajectoryWaypoint]:
        """Generate a random feasible trajectory for population initialization.

        Positions are sampled in spherical coordinates around low-Earth orbit,
        velocities approximate circular orbital speed with perturbations, and
        thrust vectors are random within the constraint bounds.

        Args:
            n_waypoints: Number of waypoints in the trajectory.
            constraints: Optimization constraints for bound enforcement.

        Returns:
            Randomly generated trajectory.
        """
        trajectory = []

        for i in range(n_waypoints):
            # Random time
            t = (
                random.uniform(
                    constraints.time_bounds_s[0], constraints.time_bounds_s[1]
                )
                * i
                / (n_waypoints - 1)
            )

            # Random position (around Earth orbit)
            r = random.uniform(
                constraints.min_altitude_m + 6.378137e6,
                constraints.min_altitude_m + 6.378137e6 + 1e6,
            )
            theta = random.uniform(0, 2 * math.pi)
            phi = random.uniform(0, math.pi)

            pos = [
                r * math.sin(phi) * math.cos(theta),
                r * math.sin(phi) * math.sin(theta),
                r * math.cos(phi),
            ]

            # Random velocity near circular orbital speed: v_circ = sqrt(mu/r)
            # Perturbed by +/- 20% to explore non-circular trajectories
            v_mag = math.sqrt(3.986004418e14 / r) * random.uniform(0.8, 1.2)
            v_theta = random.uniform(0, 2 * math.pi)
            vel = [
                v_mag * math.cos(v_theta),
                v_mag * math.sin(v_theta),
                random.uniform(-1000, 1000),
            ]

            # Random thrust
            thrust_mag = random.uniform(0, constraints.max_thrust_n)
            thrust_dir = random.uniform(0, 2 * math.pi)
            thrust = [
                thrust_mag * math.cos(thrust_dir),
                thrust_mag * math.sin(thrust_dir),
                random.uniform(-thrust_mag / 2, thrust_mag / 2),
            ]

            trajectory.append(
                TrajectoryWaypoint(
                    time_s=t,
                    position_m=pos,
                    velocity_ms=vel,
                    thrust_n=thrust,
                    mass_kg=random.uniform(500, 2000),
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
        if len(parent1) != len(parent2):
            return parent1  # Return first parent if lengths don't match

        child = []
        crossover_point = random.randint(1, len(parent1) - 1)

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
        randomizes the thrust direction.

        Args:
            trajectory: Input trajectory to mutate.
            constraints: Used to clamp thrust within allowed bounds.

        Returns:
            Mutated copy of the trajectory.
        """
        mutated = []

        for waypoint in trajectory:
            if random.random() < self.params.mutation_rate:
                # Mutate thrust
                thrust_mag = vector_magnitude(waypoint.thrust_n)
                thrust_mag *= random.uniform(0.8, 1.2)
                thrust_mag = clamp(thrust_mag, 0, constraints.max_thrust_n)

                thrust_dir = random.uniform(0, 2 * math.pi)
                new_thrust = [
                    thrust_mag * math.cos(thrust_dir),
                    thrust_mag * math.sin(thrust_dir),
                    random.uniform(-thrust_mag / 2, thrust_mag / 2),
                ]

                new_waypoint = TrajectoryWaypoint(
                    time_s=waypoint.time_s,
                    position_m=waypoint.position_m,
                    velocity_ms=waypoint.velocity_ms,
                    thrust_n=new_thrust,
                    mass_kg=waypoint.mass_kg,
                )
                mutated.append(new_waypoint)
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
            1. Initialize population with random trajectories (seed with
               initial guess at index 0).
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
        """
        import time

        start_time = time.time()

        # Initialize population
        n_waypoints = len(initial_trajectory)
        self.population = [
            self.random_trajectory(n_waypoints, constraints)
            for _ in range(self.params.population_size)
        ]
        self.population[0] = initial_trajectory  # Include initial guess

        best_cost = float("inf")
        best_trajectory = initial_trajectory
        generations_without_improvement = 0

        for _generation in range(self.params.generations):
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
            if generations_without_improvement > 20:
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
                    random.sample(range(len(self.population)), 3),
                    key=lambda i: self.fitness_scores[i],
                )
                parent2_idx = min(
                    random.sample(range(len(self.population)), 3),
                    key=lambda i: self.fitness_scores[i],
                )

                parent1 = self.population[parent1_idx]
                parent2 = self.population[parent2_idx]

                # Crossover
                if random.random() < self.params.crossover_rate:
                    child = self.crossover(parent1, parent2)
                else:
                    child = parent1 if random.random() < 0.5 else parent2

                # Mutation
                child = self.mutate(child, constraints)

                new_population.append(child)

            self.population = new_population

        computation_time = time.time() - start_time

        # Calculate final metrics
        delta_v_total = 0.0
        fuel_mass = 0.0

        for i in range(1, len(best_trajectory)):
            if best_trajectory[i].thrust_n:
                thrust_mag = vector_magnitude(best_trajectory[i].thrust_n)
                dt = best_trajectory[i].time_s - best_trajectory[i - 1].time_s
                if best_trajectory[i].mass_kg > 0:
                    dv = thrust_mag / best_trajectory[i].mass_kg * dt
                    delta_v_total += dv

                    # Fuel consumption estimate
                    if thrust_mag > 0:
                        isp = 300.0
                        dm = thrust_mag * dt / (isp * 9.80665)
                        fuel_mass += dm

        return OptimizationResult(
            optimal_trajectory=best_trajectory,
            optimal_cost=best_cost,
            delta_v_total_ms=delta_v_total,
            fuel_mass_kg=fuel_mass,
            converged=generations_without_improvement <= 20,
            iterations=self.params.generations,
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
    """

    def __init__(self, params: ParticleSwarmParams):
        self.params = params
        self.particles = []
        self.velocities = []
        self.personal_best = []
        self.personal_best_scores = []
        self.global_best = None
        self.global_best_score = float("inf")

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
        preserved; only the thrust vectors are optimized.

        Args:
            initial_trajectory: Reference trajectory defining the
                time/position/velocity/mass structure.
            objective: Optimization objective specification.
            constraints: Feasibility constraints.

        Returns:
            Optimization result with the best trajectory found.
        """
        import time

        start_time = time.time()

        # Initialize particles (simplified - optimize thrust magnitudes)
        n_waypoints = len(initial_trajectory)
        n_dimensions = n_waypoints * 3  # 3 thrust components per waypoint

        # Initialize particles as thrust profiles
        self.particles = []
        self.velocities = []

        for _ in range(self.params.num_particles):
            # Random thrust profile
            particle = []
            velocity = []

            for _ in range(n_dimensions):
                particle.append(
                    random.uniform(-constraints.max_thrust_n, constraints.max_thrust_n)
                )
                velocity.append(random.uniform(-100, 100))

            self.particles.append(particle)
            self.velocities.append(velocity)

        # Initialize personal bests
        self.personal_best = [p.copy() for p in self.particles]
        self.personal_best_scores = [float("inf")] * self.params.num_particles

        best_cost = float("inf")
        best_trajectory = initial_trajectory

        for iteration in range(self.params.max_iterations):
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
                    waypoint = TrajectoryWaypoint(
                        time_s=initial_trajectory[j].time_s,
                        position_m=initial_trajectory[j].position_m,
                        velocity_ms=initial_trajectory[j].velocity_ms,
                        thrust_n=thrust,
                        mass_kg=initial_trajectory[j].mass_kg,
                    )
                    trajectory.append(waypoint)

                # Evaluate fitness
                cost = evaluate_trajectory_cost(trajectory, objective, constraints)

                # Update personal best
                if cost < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = cost
                    self.personal_best[i] = particle.copy()

                # Update global best
                if cost < self.global_best_score:
                    self.global_best_score = cost
                    self.global_best = particle.copy()
                    best_cost = cost
                    best_trajectory = trajectory

            # Update particle velocities and positions using the PSO equations
            for i in range(len(self.particles)):
                for j in range(n_dimensions):
                    # Random coefficients for stochastic exploration
                    r1, r2 = random.random(), random.random()

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

                    # Position update: x = x + v
                    self.particles[i][j] += self.velocities[i][j]

            # Check convergence
            if iteration > 10:
                recent_scores = [self.global_best_score] * 10
                if (
                    max(recent_scores) - min(recent_scores)
                    < self.params.convergence_tolerance
                ):
                    break

        computation_time = time.time() - start_time

        # Calculate final metrics
        delta_v_total = 0.0
        fuel_mass = 0.0

        for i in range(1, len(best_trajectory)):
            if best_trajectory[i].thrust_n:
                thrust_mag = vector_magnitude(best_trajectory[i].thrust_n)
                dt = best_trajectory[i].time_s - best_trajectory[i - 1].time_s
                if best_trajectory[i].mass_kg > 0:
                    dv = thrust_mag / best_trajectory[i].mass_kg * dt
                    delta_v_total += dv

                    if thrust_mag > 0:
                        isp = 300.0
                        dm = thrust_mag * dt / (isp * 9.80665)
                        fuel_mass += dm

        return OptimizationResult(
            optimal_trajectory=best_trajectory,
            optimal_cost=best_cost,
            delta_v_total_ms=delta_v_total,
            fuel_mass_kg=fuel_mass,
            converged=True,
            iterations=iteration + 1,
            computation_time_s=computation_time,
            algorithm="Particle Swarm Optimization",
        )


# ===========================================================================
# Monte Carlo Uncertainty Analysis
# ===========================================================================


def monte_carlo_uncertainty_analysis(
    nominal_trajectory: list[TrajectoryWaypoint],
    uncertainty_params: dict[str, dict[str, float]],
    n_samples: int = 1000,
) -> dict[str, Any]:
    """Perform Monte Carlo uncertainty analysis on a trajectory.

    Each sample perturbs the nominal trajectory by adding Gaussian noise
    to position, velocity, and thrust vectors, then recomputes the
    performance metrics.  Results include mean, standard deviation,
    and 95% confidence intervals (mean +/- 1.96*sigma).

    Args:
        nominal_trajectory: Nominal (unperturbed) trajectory.
        uncertainty_params: Dictionary mapping parameter names to
            ``{"std": value}`` dictionaries specifying 1-sigma
            uncertainties.  Supported keys: ``"position_m"``,
            ``"velocity_ms"``, ``"thrust_n"``.
        n_samples: Number of Monte Carlo samples to draw.

    Returns:
        Dictionary with delta-V, flight time, and position error
        statistics, plus 95% confidence intervals.
    """
    results = {
        "delta_v_samples": [],
        "final_position_samples": [],
        "final_velocity_samples": [],
        "flight_time_samples": [],
    }

    for _ in range(n_samples):
        # Perturb parameters
        perturbed_trajectory = []

        for waypoint in nominal_trajectory:
            # Apply uncertainties
            pos_uncertainty = uncertainty_params.get("position_m", {"std": 100.0})[
                "std"
            ]
            vel_uncertainty = uncertainty_params.get("velocity_ms", {"std": 10.0})[
                "std"
            ]
            thrust_uncertainty = uncertainty_params.get("thrust_n", {"std": 50.0})[
                "std"
            ]

            # Generate random perturbations
            pos_perturbation = [random.gauss(0, pos_uncertainty) for _ in range(3)]
            vel_perturbation = [random.gauss(0, vel_uncertainty) for _ in range(3)]
            thrust_perturbation = [
                random.gauss(0, thrust_uncertainty) for _ in range(3)
            ]

            perturbed_pos = [
                waypoint.position_m[i] + pos_perturbation[i] for i in range(3)
            ]
            perturbed_vel = [
                waypoint.velocity_ms[i] + vel_perturbation[i] for i in range(3)
            ]
            perturbed_thrust = (
                [waypoint.thrust_n[i] + thrust_perturbation[i] for i in range(3)]
                if waypoint.thrust_n
                else None
            )

            perturbed_waypoint = TrajectoryWaypoint(
                time_s=waypoint.time_s,
                position_m=perturbed_pos,
                velocity_ms=perturbed_vel,
                thrust_n=perturbed_thrust,
                mass_kg=waypoint.mass_kg,
            )
            perturbed_trajectory.append(perturbed_waypoint)

        # Calculate metrics for this sample
        total_delta_v = 0.0
        for i in range(1, len(perturbed_trajectory)):
            if perturbed_trajectory[i].thrust_n:
                thrust_mag = vector_magnitude(perturbed_trajectory[i].thrust_n)
                dt = perturbed_trajectory[i].time_s - perturbed_trajectory[i - 1].time_s
                if perturbed_trajectory[i].mass_kg > 0:
                    dv = thrust_mag / perturbed_trajectory[i].mass_kg * dt
                    total_delta_v += dv

        results["delta_v_samples"].append(total_delta_v)
        results["final_position_samples"].append(perturbed_trajectory[-1].position_m)
        results["final_velocity_samples"].append(perturbed_trajectory[-1].velocity_ms)
        results["flight_time_samples"].append(
            perturbed_trajectory[-1].time_s - perturbed_trajectory[0].time_s
        )

    # Calculate statistics
    import statistics

    try:
        delta_v_mean = statistics.mean(results["delta_v_samples"])
        delta_v_std = statistics.stdev(results["delta_v_samples"])

        flight_time_mean = statistics.mean(results["flight_time_samples"])
        flight_time_std = statistics.stdev(results["flight_time_samples"])

        # Final position statistics
        final_pos_errors = [
            vector_magnitude(
                [
                    results["final_position_samples"][i][j]
                    - nominal_trajectory[-1].position_m[j]
                    for j in range(3)
                ]
            )
            for i in range(n_samples)
        ]

        pos_error_mean = statistics.mean(final_pos_errors)
        pos_error_std = statistics.stdev(final_pos_errors)

        return {
            "n_samples": n_samples,
            "delta_v_statistics": {
                "mean_ms": delta_v_mean,
                "std_ms": delta_v_std,
                "min_ms": min(results["delta_v_samples"]),
                "max_ms": max(results["delta_v_samples"]),
            },
            "flight_time_statistics": {
                "mean_s": flight_time_mean,
                "std_s": flight_time_std,
                "min_s": min(results["flight_time_samples"]),
                "max_s": max(results["flight_time_samples"]),
            },
            "position_error_statistics": {
                "mean_m": pos_error_mean,
                "std_m": pos_error_std,
                "max_m": max(final_pos_errors),
            },
            "confidence_intervals": {
                "delta_v_95_ms": [
                    delta_v_mean - 1.96 * delta_v_std,
                    delta_v_mean + 1.96 * delta_v_std,
                ],
                "position_95_m": [
                    pos_error_mean - 1.96 * pos_error_std,
                    pos_error_mean + 1.96 * pos_error_std,
                ],
            },
        }
    except statistics.StatisticsError:
        return {"error": "Insufficient data for statistics", "n_samples": n_samples}


# Update availability
try:
    from . import update_availability

    update_availability("gnc", True, {})
except ImportError:
    pass
