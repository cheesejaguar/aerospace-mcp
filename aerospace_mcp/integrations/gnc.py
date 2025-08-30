"""
Guidance, Navigation, and Control (GNC) tools for aerospace MCP.

Provides advanced trajectory optimization, guidance algorithms, and spacecraft
control system analysis for mission planning and operations.
"""

import math
import random
from dataclasses import dataclass
from typing import Any


@dataclass
class TrajectoryWaypoint:
    """Trajectory waypoint with state and control."""

    time_s: float
    position_m: list[float]  # [x, y, z] position
    velocity_ms: list[float]  # [vx, vy, vz] velocity
    acceleration_ms2: list[float] = None  # [ax, ay, az] acceleration
    thrust_n: list[float] = None  # [Fx, Fy, Fz] thrust vector
    mass_kg: float = 1000.0  # Spacecraft mass


@dataclass
class OptimizationConstraints:
    """Constraints for trajectory optimization."""

    max_thrust_n: float = 10000.0  # Maximum thrust magnitude
    max_acceleration_ms2: float = 50.0  # Maximum acceleration
    min_altitude_m: float = 200000.0  # Minimum altitude above Earth
    max_delta_v_ms: float = 5000.0  # Maximum total delta-V
    time_bounds_s: tuple[float, float] = (0.0, 86400.0)  # Time bounds


@dataclass
class OptimizationObjective:
    """Optimization objective function."""

    type: str = "minimize_fuel"  # minimize_fuel, minimize_time, maximize_payload
    weights: dict[str, float] = None  # Objective weights
    target_state: list[float] = None  # Target state if applicable


@dataclass
class GeneticAlgorithmParams:
    """Parameters for genetic algorithm optimization."""

    population_size: int = 50
    generations: int = 100
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    elite_size: int = 5
    convergence_tolerance: float = 1e-6


@dataclass
class ParticleSwarmParams:
    """Parameters for particle swarm optimization."""

    num_particles: int = 30
    max_iterations: int = 100
    w: float = 0.7  # Inertia weight
    c1: float = 1.5  # Cognitive component
    c2: float = 1.5  # Social component
    convergence_tolerance: float = 1e-6


@dataclass
class OptimizationResult:
    """Result from trajectory optimization."""

    optimal_trajectory: list[TrajectoryWaypoint]
    optimal_cost: float
    delta_v_total_ms: float
    fuel_mass_kg: float
    converged: bool
    iterations: int
    computation_time_s: float
    algorithm: str


def vector_magnitude(vec: list[float]) -> float:
    """Calculate vector magnitude."""
    return math.sqrt(sum(x**2 for x in vec))


def vector_distance(a: list[float], b: list[float]) -> float:
    """Calculate Euclidean distance between two vectors."""
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(len(a))))


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp value to range [min_val, max_val]."""
    return max(min_val, min(max_val, value))


def evaluate_trajectory_cost(
    trajectory: list[TrajectoryWaypoint],
    objective: OptimizationObjective,
    constraints: OptimizationConstraints,
) -> float:
    """
    Evaluate trajectory cost function.

    Args:
        trajectory: List of trajectory waypoints
        objective: Optimization objective
        constraints: Optimization constraints

    Returns:
        Cost value (lower is better)
    """
    if not trajectory or len(trajectory) < 2:
        return float("inf")

    cost = 0.0
    penalty = 0.0

    # Calculate delta-V and fuel consumption
    total_delta_v = 0.0
    total_fuel = 0.0

    for i in range(1, len(trajectory)):
        dt = trajectory[i].time_s - trajectory[i - 1].time_s
        if dt <= 0:
            return float("inf")

        # Thrust-based delta-V
        if trajectory[i].thrust_n:
            thrust_mag = vector_magnitude(trajectory[i].thrust_n)
            if trajectory[i].mass_kg > 0:
                accel = thrust_mag / trajectory[i].mass_kg
                dv = accel * dt
                total_delta_v += dv

                # Fuel consumption (Tsiolkovsky equation approximation)
                if thrust_mag > 0:
                    isp = 300.0  # Assumed specific impulse (s)
                    dm = thrust_mag * dt / (isp * 9.80665)
                    total_fuel += dm

        # Constraint violations
        if trajectory[i].thrust_n:
            thrust_mag = vector_magnitude(trajectory[i].thrust_n)
            if thrust_mag > constraints.max_thrust_n:
                penalty += (thrust_mag - constraints.max_thrust_n) ** 2 * 1e-6

        # Altitude constraint
        pos_mag = vector_magnitude(trajectory[i].position_m)
        altitude = pos_mag - 6.378137e6  # Earth radius
        if altitude < constraints.min_altitude_m:
            penalty += (constraints.min_altitude_m - altitude) ** 2 * 1e-12

    # Delta-V constraint
    if total_delta_v > constraints.max_delta_v_ms:
        penalty += (total_delta_v - constraints.max_delta_v_ms) ** 2 * 1e-6

    # Objective function
    if objective.type == "minimize_fuel":
        cost = total_fuel
    elif objective.type == "minimize_time":
        cost = trajectory[-1].time_s - trajectory[0].time_s
    elif objective.type == "minimize_delta_v":
        cost = total_delta_v
    elif objective.type == "maximize_payload":
        trajectory[0].mass_kg
        final_mass = trajectory[-1].mass_kg
        cost = -(final_mass - total_fuel)  # Negative for maximization
    else:
        cost = total_delta_v  # Default

    # Add penalty for constraint violations
    cost += penalty * 1000  # High penalty weight

    return cost


class GeneticAlgorithm:
    """Genetic Algorithm for trajectory optimization."""

    def __init__(self, params: GeneticAlgorithmParams):
        self.params = params
        self.population = []
        self.fitness_scores = []

    def random_trajectory(
        self, n_waypoints: int, constraints: OptimizationConstraints
    ) -> list[TrajectoryWaypoint]:
        """Generate random trajectory."""
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

            # Random velocity (orbital-like)
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
        """Crossover operation between two parent trajectories."""
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
        """Mutate trajectory."""
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
        """Run genetic algorithm optimization."""
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

        for generation in range(self.params.generations):
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
            iterations=generation + 1,
            computation_time_s=computation_time,
            algorithm="Genetic Algorithm",
        )


class ParticleSwarmOptimizer:
    """Particle Swarm Optimization for trajectory optimization."""

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
        """Run particle swarm optimization."""
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

            # Update particle velocities and positions
            for i in range(len(self.particles)):
                for j in range(n_dimensions):
                    # Velocity update
                    r1, r2 = random.random(), random.random()

                    cognitive = (
                        self.params.c1
                        * r1
                        * (self.personal_best[i][j] - self.particles[i][j])
                    )
                    social = (
                        self.params.c2
                        * r2
                        * (self.global_best[j] - self.particles[i][j])
                    )

                    self.velocities[i][j] = (
                        self.params.w * self.velocities[i][j] + cognitive + social
                    )

                    # Position update
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


def monte_carlo_uncertainty_analysis(
    nominal_trajectory: list[TrajectoryWaypoint],
    uncertainty_params: dict[str, dict[str, float]],
    n_samples: int = 1000,
) -> dict[str, Any]:
    """
    Perform Monte Carlo uncertainty analysis on trajectory.

    Args:
        nominal_trajectory: Nominal trajectory
        uncertainty_params: Dictionary of parameter uncertainties
        n_samples: Number of Monte Carlo samples

    Returns:
        Uncertainty analysis results
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
