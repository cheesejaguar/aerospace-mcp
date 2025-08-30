"""Tests for guidance, navigation, and control module."""

import pytest

from aerospace_mcp.integrations.gnc import (
    GeneticAlgorithm,
    OptimizationResult,
    ParticleSwarmOptimizer,
    TrajectoryWaypoint,
)


class TestTrajectoryWaypoint:
    """Test TrajectoryWaypoint data class."""

    def test_valid_waypoint(self):
        """Test creating valid trajectory waypoint."""
        waypoint = TrajectoryWaypoint(
            time_s=0.0,
            position_m=[7000000.0, 0.0, 0.0],
            velocity_ms=[0.0, 7546.0, 0.0]
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
            mass_kg=1500.0
        )

        assert waypoint.time_s == 10.0
        assert waypoint.mass_kg == 1500.0
        assert waypoint.acceleration_ms2[0] == 1.0
        assert waypoint.thrust_n[0] == 500.0


class TestGeneticAlgorithm:
    """Test Genetic Algorithm implementation."""

    def test_ga_initialization(self):
        """Test GA initialization."""
        ga = GeneticAlgorithm()

        # Check that GA instance was created
        assert ga is not None
        assert hasattr(ga, 'optimize')


class TestParticleSwarmOptimizer:
    """Test Particle Swarm Optimization implementation."""

    def test_pso_initialization(self):
        """Test PSO initialization."""
        pso = ParticleSwarmOptimizer()

        # Check that PSO instance was created
        assert pso is not None
        assert hasattr(pso, 'optimize')


class TestOptimizationResult:
    """Test OptimizationResult data class."""

    def test_optimization_result_creation(self):
        """Test creating optimization result."""
        waypoints = [
            TrajectoryWaypoint(
                time_s=0.0,
                position_m=[0.0, 0.0, 0.0],
                velocity_ms=[0.0, 0.0, 0.0]
            ),
            TrajectoryWaypoint(
                time_s=10.0,
                position_m=[1000.0, 0.0, 0.0],
                velocity_ms=[100.0, 0.0, 0.0]
            )
        ]

        result = OptimizationResult(
            optimal_trajectory=waypoints,
            total_delta_v_ms=150.0,
            flight_time_s=3600.0,
            objective_value=1000.0,
            convergence_iterations=50,
            success=True
        )

        assert len(result.optimal_trajectory) == 2
        assert result.total_delta_v_ms == 150.0
        assert result.success == True


class TestModuleImports:
    """Test that all expected functions and classes are available."""

    def test_imports(self):
        """Test that key classes can be imported."""
        # Test that we can import and create instances
        waypoint = TrajectoryWaypoint(
            time_s=0.0,
            position_m=[0.0, 0.0, 0.0],
            velocity_ms=[0.0, 0.0, 0.0]
        )

        ga = GeneticAlgorithm()
        pso = ParticleSwarmOptimizer()

        assert waypoint is not None
        assert ga is not None
        assert pso is not None


if __name__ == "__main__":
    pytest.main([__file__])
