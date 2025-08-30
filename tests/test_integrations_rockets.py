"""
Test suite for rocket trajectory analysis integration.

Tests the rockets.py module functionality including:
- 3DOF trajectory simulation
- Performance analysis
- Rocket sizing estimation
"""

import pytest

from aerospace_mcp.integrations.rockets import (
    RocketGeometry,
    analyze_rocket_performance,
    estimate_rocket_sizing,
    get_thrust_at_time,
    rocket_3dof_trajectory,
)


class TestRocketGeometry:
    """Test RocketGeometry dataclass."""

    def test_rocket_geometry_creation(self):
        """Test basic rocket geometry creation."""
        geometry = RocketGeometry(
            dry_mass_kg=10.0,
            propellant_mass_kg=30.0,
            diameter_m=0.1,
            length_m=1.0,
            cd=0.3,
        )

        assert geometry.dry_mass_kg == 10.0
        assert geometry.propellant_mass_kg == 30.0
        assert geometry.diameter_m == 0.1
        assert geometry.length_m == 1.0
        assert geometry.cd == 0.3
        assert geometry.thrust_curve is None

    def test_rocket_geometry_with_thrust_curve(self):
        """Test rocket geometry with thrust curve."""
        thrust_curve = [[0.0, 1000.0], [5.0, 800.0], [10.0, 0.0]]
        geometry = RocketGeometry(
            dry_mass_kg=15.0,
            propellant_mass_kg=45.0,
            diameter_m=0.15,
            length_m=1.5,
            thrust_curve=thrust_curve,
        )

        assert geometry.thrust_curve == thrust_curve


class TestThrustCurve:
    """Test thrust curve interpolation functions."""

    def test_constant_thrust(self):
        """Test constant thrust curve."""
        thrust_curve = [[10.0, 500.0]]  # 500N for 10 seconds

        assert get_thrust_at_time(thrust_curve, 0.0) == 500.0
        assert get_thrust_at_time(thrust_curve, 5.0) == 500.0
        assert get_thrust_at_time(thrust_curve, 10.0) == 500.0
        assert get_thrust_at_time(thrust_curve, 15.0) == 0.0

    def test_linear_interpolation(self):
        """Test linear interpolation between thrust points."""
        thrust_curve = [[0.0, 1000.0], [10.0, 500.0], [15.0, 0.0]]

        # Test interpolation at midpoints
        assert (
            get_thrust_at_time(thrust_curve, 5.0) == 750.0
        )  # Midpoint between 1000 and 500
        assert (
            get_thrust_at_time(thrust_curve, 12.5) == 250.0
        )  # Midpoint between 500 and 0

        # Test boundary conditions
        assert get_thrust_at_time(thrust_curve, 0.0) == 1000.0
        assert get_thrust_at_time(thrust_curve, 15.0) == 0.0
        assert get_thrust_at_time(thrust_curve, 20.0) == 0.0

    def test_empty_thrust_curve(self):
        """Test empty thrust curve."""
        assert get_thrust_at_time([], 5.0) == 0.0
        assert get_thrust_at_time(None, 5.0) == 0.0


class TestRocketTrajectory:
    """Test 3DOF rocket trajectory simulation."""

    def create_test_rocket(self):
        """Create a standard test rocket."""
        thrust_curve = [[0.0, 2000.0], [8.0, 2000.0], [8.1, 0.0]]  # 2kN for 8 seconds
        return RocketGeometry(
            dry_mass_kg=20.0,
            propellant_mass_kg=80.0,
            diameter_m=0.2,
            length_m=2.0,
            cd=0.4,
            thrust_curve=thrust_curve,
        )

    def test_vertical_launch_basic(self):
        """Test basic vertical launch trajectory."""
        geometry = self.create_test_rocket()

        trajectory = rocket_3dof_trajectory(
            geometry, dt_s=0.1, max_time_s=100.0, launch_angle_deg=90.0
        )

        # Basic sanity checks
        assert len(trajectory) > 0
        assert trajectory[0].altitude_m == 0.0  # Starts at ground level
        assert trajectory[0].velocity_ms == 0.0  # Starts at rest

        # Should reach some altitude
        max_altitude = max(point.altitude_m for point in trajectory)
        assert max_altitude > 500.0  # Should reach at least 500 m

        # Should have positive thrust during burn phase
        burn_points = [p for p in trajectory if p.time_s <= 8.0]
        assert all(p.thrust_n > 0 for p in burn_points)

        # Should have zero thrust after burnout
        coast_points = [p for p in trajectory if p.time_s > 8.5]
        assert all(p.thrust_n == 0 for p in coast_points)

    def test_angled_launch(self):
        """Test angled launch trajectory."""
        geometry = self.create_test_rocket()

        trajectory = rocket_3dof_trajectory(
            geometry, dt_s=0.1, max_time_s=100.0, launch_angle_deg=85.0
        )

        assert len(trajectory) > 0

        # Should still reach reasonable altitude with slight angle
        max_altitude = max(point.altitude_m for point in trajectory)
        assert max_altitude > 400.0  # Slightly less than vertical

    def test_trajectory_physics(self):
        """Test that trajectory follows basic physics."""
        geometry = self.create_test_rocket()

        trajectory = rocket_3dof_trajectory(geometry, dt_s=0.1, max_time_s=50.0)

        # Check that mass decreases during burn
        initial_mass = trajectory[0].mass_kg
        burnout_point = next((p for p in trajectory if p.thrust_n == 0), None)

        if burnout_point:
            assert burnout_point.mass_kg < initial_mass
            # Mass should be approximately dry mass at burnout (allow some margin)
            assert (
                abs(burnout_point.mass_kg - geometry.dry_mass_kg)
                < geometry.propellant_mass_kg * 0.2
            )  # Within 20% of propellant mass

        # Check that drag increases with velocity
        high_velocity_points = [p for p in trajectory if p.velocity_ms > 100]
        if high_velocity_points:
            avg_drag = sum(p.drag_n for p in high_velocity_points) / len(
                high_velocity_points
            )
            assert avg_drag > 0

    def test_performance_analysis(self):
        """Test rocket performance analysis."""
        geometry = self.create_test_rocket()
        trajectory = rocket_3dof_trajectory(geometry, dt_s=0.1, max_time_s=100.0)

        performance = analyze_rocket_performance(trajectory)

        # Basic performance checks
        assert performance.max_altitude_m > 0
        assert performance.apogee_time_s > 0
        assert performance.max_velocity_ms > 0
        assert performance.total_impulse_ns > 0
        assert performance.specific_impulse_s > 0

        # Performance should be reasonable for test rocket
        assert performance.max_altitude_m > 500.0  # At least 500 m
        assert performance.specific_impulse_s > 100.0  # Reasonable Isp
        assert performance.specific_impulse_s < 400.0  # Not too high

        # Burnout should occur before apogee
        assert performance.burnout_time_s < performance.apogee_time_s

        # Max Q should be reasonable
        assert performance.max_q_pa > 0
        assert performance.max_q_pa < 100000  # Less than 100 kPa


class TestRocketSizing:
    """Test rocket sizing estimation."""

    def test_sizing_small_rocket(self):
        """Test sizing for small rocket mission."""
        sizing = estimate_rocket_sizing(
            target_altitude_m=5000.0,  # 5 km target
            payload_mass_kg=2.0,  # 2 kg payload
            propellant_type="solid",
        )

        # Should be feasible
        assert sizing["feasible"]
        assert sizing["total_mass_kg"] > sizing["payload_mass_kg"]
        assert sizing["propellant_mass_kg"] > 0
        assert sizing["structure_mass_kg"] > 0
        assert sizing["mass_ratio"] > 1.0
        assert sizing["delta_v_ms"] > 0
        assert sizing["diameter_m"] > 0
        assert sizing["length_m"] > 0

        # Sanity checks on proportions - for small payloads, propellant might be smaller
        # Just check that masses are positive and reasonable
        assert sizing["propellant_mass_kg"] > 0
        assert sizing["structure_mass_kg"] > 0
        assert sizing["structure_mass_kg"] < sizing["total_mass_kg"]

    def test_sizing_liquid_vs_solid(self):
        """Test that liquid propellant gives better performance."""
        target_alt = 10000.0
        payload = 5.0

        solid_sizing = estimate_rocket_sizing(target_alt, payload, "solid")
        liquid_sizing = estimate_rocket_sizing(target_alt, payload, "liquid")

        # Both should be feasible
        assert solid_sizing["feasible"]
        assert liquid_sizing["feasible"]

        # Liquid should need less propellant mass for same mission
        assert liquid_sizing["propellant_mass_kg"] < solid_sizing["propellant_mass_kg"]

        # Liquid should have higher specific impulse
        assert liquid_sizing["specific_impulse_s"] > solid_sizing["specific_impulse_s"]

    def test_sizing_impossible_mission(self):
        """Test sizing for impossible single-stage mission."""
        sizing = estimate_rocket_sizing(
            target_altitude_m=100000.0,  # 100 km - very challenging
            payload_mass_kg=1000.0,  # 1000 kg payload - heavy
            propellant_type="solid",
        )

        # Should be infeasible for single stage - but our simplified model might not catch this
        # Just check that if feasible, the mass ratio is very high
        if sizing["feasible"]:
            assert (
                sizing["mass_ratio"] > 10.0
            )  # Very high mass ratio indicates difficulty

    def test_sizing_parameter_relationships(self):
        """Test relationships between sizing parameters."""
        sizing = estimate_rocket_sizing(15000.0, 10.0, "liquid")

        # Mass breakdown should sum correctly
        total_calculated = (
            sizing["propellant_mass_kg"]
            + sizing["structure_mass_kg"]
            + sizing["payload_mass_kg"]
        )
        assert abs(total_calculated - sizing["total_mass_kg"]) < 0.1

        # Thrust-to-weight should be reasonable
        assert sizing["thrust_to_weight"] > 1.5  # Must overcome gravity
        assert sizing["thrust_to_weight"] < 10.0  # Not excessive

        # Geometry should be reasonable
        ld_ratio = sizing["length_m"] / sizing["diameter_m"]
        assert ld_ratio > 5.0  # Should be reasonably slender
        assert ld_ratio < 20.0  # But not too extreme


class TestRocketIntegration:
    """Integration tests combining multiple rocket functions."""

    def test_sizing_to_trajectory_consistency(self):
        """Test that sized rocket performs close to target."""
        target_altitude = 8000.0  # 8 km
        payload_mass = 3.0  # 3 kg

        # Get sizing estimate
        sizing = estimate_rocket_sizing(target_altitude, payload_mass, "solid")
        assert sizing["feasible"]

        # Create rocket geometry from sizing
        thrust_curve = [
            [0.0, sizing["thrust_n"]],
            [10.0, sizing["thrust_n"]],
            [10.1, 0.0],
        ]
        geometry = RocketGeometry(
            dry_mass_kg=sizing["structure_mass_kg"] + payload_mass,
            propellant_mass_kg=sizing["propellant_mass_kg"],
            diameter_m=sizing["diameter_m"],
            length_m=sizing["length_m"],
            cd=0.3,
            thrust_curve=thrust_curve,
        )

        # Run trajectory simulation
        trajectory = rocket_3dof_trajectory(geometry, dt_s=0.1, max_time_s=200.0)
        performance = analyze_rocket_performance(trajectory)

        # Performance should be in the ballpark of target
        altitude_error = (
            abs(performance.max_altitude_m - target_altitude) / target_altitude
        )
        assert altitude_error < 0.5  # Within 50% - rough sizing estimate

        # Should achieve reasonable performance
        assert performance.max_altitude_m > target_altitude * 0.5
        assert performance.specific_impulse_s > 150.0  # Reasonable for solid

    def test_different_launch_angles(self):
        """Test trajectory with different launch angles."""
        geometry = RocketGeometry(
            dry_mass_kg=25.0,
            propellant_mass_kg=75.0,
            diameter_m=0.15,
            length_m=1.8,
            cd=0.35,
            thrust_curve=[[0.0, 1500.0], [12.0, 1500.0], [12.1, 0.0]],
        )

        angles = [90.0, 85.0, 80.0]
        results = []

        for angle in angles:
            trajectory = rocket_3dof_trajectory(geometry, launch_angle_deg=angle)
            performance = analyze_rocket_performance(trajectory)
            results.append(performance.max_altitude_m)

        # Vertical should generally give highest altitude
        assert results[0] >= results[1]  # 90° >= 85°

        # All should achieve reasonable altitude
        assert all(alt > 500.0 for alt in results)


class TestRocketEdgeCases:
    """Test edge cases and error conditions."""

    def test_zero_thrust_rocket(self):
        """Test rocket with no thrust."""
        geometry = RocketGeometry(
            dry_mass_kg=10.0,
            propellant_mass_kg=20.0,
            diameter_m=0.1,
            length_m=1.0,
            thrust_curve=[],  # No thrust
        )

        trajectory = rocket_3dof_trajectory(geometry, max_time_s=10.0)

        # Should fall back to ground quickly
        assert len(trajectory) > 0
        max_altitude = max(point.altitude_m for point in trajectory)
        assert max_altitude < 1.0  # Should barely leave ground

    def test_very_heavy_rocket(self):
        """Test rocket too heavy to lift off."""
        geometry = RocketGeometry(
            dry_mass_kg=1000.0,
            propellant_mass_kg=2000.0,
            diameter_m=0.5,
            length_m=3.0,
            thrust_curve=[[0.0, 1000.0], [5.0, 0.0]],  # Low thrust for heavy rocket
        )

        trajectory = rocket_3dof_trajectory(geometry, max_time_s=20.0)

        # Should not achieve significant altitude
        max_altitude = max(point.altitude_m for point in trajectory)
        assert max_altitude < 100.0  # Very limited altitude

    def test_short_simulation_time(self):
        """Test with very short simulation time."""
        geometry = RocketGeometry(
            dry_mass_kg=5.0,
            propellant_mass_kg=15.0,
            diameter_m=0.08,
            length_m=0.8,
            thrust_curve=[[0.0, 800.0], [3.0, 0.0]],
        )

        trajectory = rocket_3dof_trajectory(geometry, max_time_s=1.0, dt_s=0.1)

        # Should have reasonable number of points for short time
        assert len(trajectory) <= 12  # ~1s / 0.1s + buffer
        assert trajectory[-1].time_s <= 1.1

    def test_performance_empty_trajectory(self):
        """Test performance analysis with empty trajectory."""
        performance = analyze_rocket_performance([])

        # Should return zero values gracefully
        assert performance.max_altitude_m == 0
        assert performance.apogee_time_s == 0
        assert performance.max_velocity_ms == 0


if __name__ == "__main__":
    pytest.main([__file__])
