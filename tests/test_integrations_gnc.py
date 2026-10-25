"""Tests for guidance, navigation, and control module."""

import random

import pytest

from aerospace_mcp.integrations.gnc import (
    GeneticAlgorithm,
    GeneticAlgorithmParams,
    OptimizationConstraints,
    OptimizationObjective,
    OptimizationResult,
    ParticleSwarmOptimizer,
    ParticleSwarmParams,
    TrajectoryWaypoint,
    evaluate_trajectory_cost,
    monte_carlo_uncertainty_analysis,
    trajectory_delta_v_and_fuel,
)


class TestTrajectoryWaypoint:
    """Test TrajectoryWaypoint data class."""

    def test_valid_waypoint(self):
        """Test creating valid trajectory waypoint."""
        waypoint = TrajectoryWaypoint(
            time_s=0.0, position_m=[7000000.0, 0.0, 0.0], velocity_ms=[0.0, 7546.0, 0.0]
        )

        assert waypoint.time_s == 0.0
        assert waypoint.position_m[0] == 7000000.0
        assert waypoint.velocity_ms[1] == 7546.0

    def test_waypoint_with_optional_fields(self):
        """Test waypoint with optional fields."""
        waypoint = TrajectoryWaypoint(
            time_s=10.0,
            position_m=[1000.0, 2000.0, 3000.0],
            velocity_ms=[100.0, 200.0, 300.0],
            acceleration_ms2=[1.0, 2.0, 3.0],
            thrust_n=[500.0, 1000.0, 1500.0],
            mass_kg=1500.0,
        )

        assert waypoint.time_s == 10.0
        assert waypoint.mass_kg == 1500.0
        assert waypoint.acceleration_ms2[0] == 1.0
        assert waypoint.thrust_n[0] == 500.0


class TestGeneticAlgorithm:
    """Test Genetic Algorithm implementation."""

    def test_ga_initialization(self):
        """Test GA initialization."""
        ga = GeneticAlgorithm(GeneticAlgorithmParams())

        # Check that GA instance was created
        assert ga is not None
        assert hasattr(ga, "optimize")


class TestParticleSwarmOptimizer:
    """Test Particle Swarm Optimization implementation."""

    def test_pso_initialization(self):
        """Test PSO initialization."""
        pso = ParticleSwarmOptimizer(ParticleSwarmParams())

        # Check that PSO instance was created
        assert pso is not None
        assert hasattr(pso, "optimize")


class TestOptimizationResult:
    """Test OptimizationResult data class."""

    def test_optimization_result_creation(self):
        """Test creating optimization result."""
        waypoints = [
            TrajectoryWaypoint(
                time_s=0.0, position_m=[0.0, 0.0, 0.0], velocity_ms=[0.0, 0.0, 0.0]
            ),
            TrajectoryWaypoint(
                time_s=10.0,
                position_m=[1000.0, 0.0, 0.0],
                velocity_ms=[100.0, 0.0, 0.0],
            ),
        ]

        result = OptimizationResult(
            optimal_trajectory=waypoints,
            optimal_cost=1000.0,
            delta_v_total_ms=150.0,
            fuel_mass_kg=10.0,
            converged=True,
            iterations=50,
            computation_time_s=1.5,
            algorithm="test",
        )

        assert len(result.optimal_trajectory) == 2
        assert result.delta_v_total_ms == 150.0
        assert result.converged


class TestModuleImports:
    """Test that all expected functions and classes are available."""

    def test_imports(self):
        """Test that key classes can be imported."""
        # Test that we can import and create instances
        waypoint = TrajectoryWaypoint(
            time_s=0.0, position_m=[0.0, 0.0, 0.0], velocity_ms=[0.0, 0.0, 0.0]
        )

        ga = GeneticAlgorithm(GeneticAlgorithmParams())
        pso = ParticleSwarmOptimizer(ParticleSwarmParams())

        assert waypoint is not None
        assert ga is not None
        assert pso is not None


def _waypoints(thrust_n=(100.0, 0.0, 0.0), n=5):
    return [
        TrajectoryWaypoint(
            time_s=100.0 * i,
            position_m=[6.9e6, 0.0, 0.0],
            velocity_ms=[0.0, 7500.0, 0.0],
            thrust_n=list(thrust_n) if thrust_n is not None else None,
            mass_kg=1000.0,
        )
        for i in range(n)
    ]


class TestOptimizerRegressions:
    """Regression tests for the audit findings G9-G11."""

    objective = OptimizationObjective(type="minimize_fuel")
    constraints = OptimizationConstraints()

    def test_random_trajectory_times_are_monotonic(self):
        """G11a: times were uniform(t0, t1) * i / (n - 1), non-causal 76% of
        the time.  Now t_i = t0 + (t1 - t0) i / (n - 1)."""
        ga = GeneticAlgorithm(GeneticAlgorithmParams(), rng=random.Random(0))
        c = OptimizationConstraints(time_bounds_s=(10.0, 50.0))
        for _ in range(50):
            traj = ga.random_trajectory(5, c)
            assert [wp.time_s for wp in traj] == [10.0, 20.0, 30.0, 40.0, 50.0]
            assert evaluate_trajectory_cost(traj, self.objective, c) < float("inf")
        assert [wp.time_s for wp in ga.random_trajectory(1, c)] == [10.0]

    def test_ga_handles_waypoints_without_thrust(self):
        """G11b: mutate() raised TypeError for thrust_n=None."""
        params = GeneticAlgorithmParams(
            population_size=10, generations=5, mutation_rate=1.0
        )
        ga = GeneticAlgorithm(params, rng=random.Random(1))
        traj = _waypoints(thrust_n=None)
        mutated = ga.mutate(traj, self.constraints)
        assert all(wp.thrust_n == [0.0, 0.0, 0.0] for wp in mutated)
        result = ga.optimize(traj, self.objective, self.constraints)
        assert result.optimal_cost == 0.0

    def test_ga_converged_flag_and_iterations(self):
        """G10: converged was True only when the loop did NOT stagnate, and
        iterations always equalled params.generations."""
        params = GeneticAlgorithmParams(population_size=10, generations=200)
        # Zero thrust is optimal: no improvement after generation 1, so the
        # run stops after 1 + 21 generations.
        r = GeneticAlgorithm(params, rng=random.Random(2)).optimize(
            _waypoints(thrust_n=(0, 0, 0)), self.objective, self.constraints
        )
        assert (r.converged, r.iterations) == (True, 22)

        short = GeneticAlgorithmParams(population_size=10, generations=3)
        r = GeneticAlgorithm(short, rng=random.Random(2)).optimize(
            _waypoints(), self.objective, self.constraints
        )
        assert (r.converged, r.iterations) == (False, 3)

    def test_ga_keeps_mission_structure(self):
        """Population members share the input times, positions and masses."""
        ga = GeneticAlgorithm(
            GeneticAlgorithmParams(population_size=8, generations=10),
            rng=random.Random(3),
        )
        traj = _waypoints()
        r = ga.optimize(traj, self.objective, self.constraints)
        for got, ref in zip(r.optimal_trajectory, traj, strict=True):
            assert (got.time_s, got.position_m, got.mass_kg) == (
                ref.time_s,
                ref.position_m,
                ref.mass_kg,
            )
        assert r.optimal_cost <= evaluate_trajectory_cost(
            traj, self.objective, self.constraints
        )

    def test_pso_uses_real_score_history(self):
        """G9: recent_scores was [global_best_score] * 10, so max - min was
        always 0 and PSO stopped after 12 iterations."""
        params = ParticleSwarmParams(num_particles=10, max_iterations=100)
        pso = ParticleSwarmOptimizer(params, rng=random.Random(4))
        r = pso.optimize(_waypoints(), self.objective, self.constraints)
        assert r.iterations > 12
        assert len(pso.best_score_history) == r.iterations
        # The history is non-increasing and the result is its last entry
        hist = pso.best_score_history
        assert all(b <= a for a, b in zip(hist, hist[1:], strict=False))
        assert r.optimal_cost == hist[-1]
        if r.converged:
            window = ParticleSwarmOptimizer.CONVERGENCE_WINDOW
            assert hist[-window - 1] - hist[-1] < params.convergence_tolerance

    def test_pso_stops_when_best_cannot_improve(self):
        pso = ParticleSwarmOptimizer(
            ParticleSwarmParams(num_particles=5, max_iterations=100),
            rng=random.Random(5),
        )
        r = pso.optimize(
            _waypoints(thrust_n=(0, 0, 0)), self.objective, self.constraints
        )
        assert (r.converged, r.iterations, r.optimal_cost) == (True, 11, 0.0)
        # Re-using the instance starts from a clean state
        r2 = pso.optimize(_waypoints(), self.objective, self.constraints)
        assert r2.optimal_cost > 0.0

    def test_delta_v_and_fuel_helper(self):
        """dv = |F|/m dt and dm = |F| dt / (Isp g0) per segment."""
        dv, fuel = trajectory_delta_v_and_fuel(_waypoints())
        assert dv == pytest.approx(4 * 100.0 / 1000.0 * 100.0)
        assert fuel == pytest.approx(4 * 100.0 * 100.0 / (300.0 * 9.80665))

    def test_monte_carlo_missing_keys_mean_no_uncertainty(self):
        """Missing uncertainty keys used to inject default errors silently
        (100 m, 10 m/s, 50 N)."""
        out = monte_carlo_uncertainty_analysis(
            _waypoints(), {}, n_samples=20, rng=random.Random(6)
        )
        assert out["position_error_statistics"]["max_m"] == 0.0
        assert out["delta_v_statistics"]["std_ms"] == 0.0


if __name__ == "__main__":
    pytest.main([__file__])
