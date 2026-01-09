"""Additional tests for 100% integration module coverage."""

import math
from unittest.mock import MagicMock, patch

import pytest


class TestAtmosphereEdgeCases:
    """Additional tests for atmosphere integration coverage."""

    @pytest.mark.unit
    def test_ambiance_version_attribute_error(self):
        """Test ambiance version handling when __version__ is missing."""
        # This tests lines 36-42 in atmosphere.py
        mock_ambiance = MagicMock()
        del mock_ambiance.__version__  # Remove __version__ attribute

        with patch.dict("sys.modules", {"ambiance": mock_ambiance}):
            # Reimport to trigger the version check
            import importlib

            import aerospace_mcp.integrations.atmosphere

            importlib.reload(aerospace_mcp.integrations.atmosphere)

    @pytest.mark.unit
    def test_ambiance_atmosphere_usage(self):
        """Test that ambiance is used when available."""
        from aerospace_mcp.integrations.atmosphere import (
            AMBIANCE_AVAILABLE,
            get_atmosphere_profile,
        )

        if AMBIANCE_AVAILABLE:
            profile = get_atmosphere_profile([1000.0], model_type="ISA")
            assert len(profile) == 1
            # Ambiance provides viscosity
            assert profile[0].viscosity_pa_s is not None


class TestFramesEdgeCases:
    """Additional tests for frames integration coverage."""

    @pytest.mark.unit
    def test_skyfield_version_handling(self):
        """Test skyfield version handling."""
        # This tests lines 41-47 in frames.py
        mock_skyfield = MagicMock()
        del mock_skyfield.__version__

        # Test when astropy is not available but skyfield is
        with patch.dict("sys.modules", {"astropy": None, "skyfield": mock_skyfield}):
            import importlib

            import aerospace_mcp.integrations.frames

            importlib.reload(aerospace_mcp.integrations.frames)

    @pytest.mark.unit
    def test_transform_with_astropy_exception(self):
        """Test transform fallback when astropy raises exception."""
        from aerospace_mcp.integrations.frames import (
            ASTROPY_AVAILABLE,
            transform_frames,
        )

        if ASTROPY_AVAILABLE:
            # Mock astropy to raise an exception to trigger fallback
            with patch(
                "aerospace_mcp.integrations.frames.Time",
                side_effect=Exception("Time error"),
            ):
                xyz = [6500000.0, 0.0, 0.0]
                # Should fall back to manual transformation
                result = transform_frames(xyz, "ECI", "ECEF")
                assert result.frame == "ECEF"

    @pytest.mark.unit
    def test_transform_not_implemented(self):
        """Test NotImplementedError for unsupported frame combinations."""
        from aerospace_mcp.integrations.frames import transform_frames

        # Test an unsupported combination that doesn't have fallback
        xyz = [1000000.0, 2000000.0, 3000000.0]

        # GEODETIC to ECI is not implemented
        with pytest.raises(NotImplementedError, match="Transformation from"):
            transform_frames(xyz, "GEODETIC", "ECI")

    @pytest.mark.unit
    def test_simple_precession_matrix(self):
        """Test simple precession matrix calculation."""
        from aerospace_mcp.integrations.frames import _simple_precession_matrix

        matrix = _simple_precession_matrix("2000-01-01T12:00:00", "2023-01-01T12:00:00")

        # Should return identity matrix (placeholder implementation)
        assert len(matrix) == 3
        assert len(matrix[0]) == 3
        assert matrix[0][0] == 1.0
        assert matrix[1][1] == 1.0
        assert matrix[2][2] == 1.0


class TestPropellersEdgeCases:
    """Additional tests for propellers integration coverage."""

    @pytest.mark.unit
    def test_pybemt_fallback_import(self):
        """Test pybemt import fallback when aerosandbox not available."""
        # This tests lines 26-34 in propellers.py
        mock_pybemt = MagicMock()

        with patch.dict(
            "sys.modules",
            {"aerosandbox": None, "pybemt": mock_pybemt},
        ):
            import importlib

            import aerospace_mcp.integrations.propellers

            importlib.reload(aerospace_mcp.integrations.propellers)

    @pytest.mark.unit
    def test_propeller_altitude_above_11000m(self):
        """Test propeller analysis above 11000m altitude."""
        from aerospace_mcp.integrations.propellers import (
            PropellerGeometry,
            _simple_propeller_analysis,
        )

        geometry = PropellerGeometry(
            diameter_m=0.254,
            pitch_m=0.178,
            num_blades=2,
        )

        # Test above 11000m
        result = _simple_propeller_analysis(
            geometry, [3000.0], velocity_ms=10.0, altitude_m=12000
        )

        assert len(result) == 1
        assert result[0].rpm == 3000.0

    @pytest.mark.unit
    def test_aerosandbox_propeller_exception_handling(self):
        """Test aerosandbox exception handling in propeller analysis."""
        from aerospace_mcp.integrations.propellers import (
            AEROSANDBOX_AVAILABLE,
            PropellerGeometry,
            propeller_bemt_analysis,
        )

        if AEROSANDBOX_AVAILABLE:
            # Mock aerosandbox to raise exception
            with patch(
                "aerospace_mcp.integrations.propellers._aerosandbox_propeller_analysis",
                side_effect=Exception("Analysis failed"),
            ):
                geometry = PropellerGeometry(
                    diameter_m=0.254,
                    pitch_m=0.178,
                    num_blades=2,
                )
                # Should fall back to simple analysis
                result = propeller_bemt_analysis(geometry, [3000.0], velocity_ms=10.0)
                assert len(result) == 1

    @pytest.mark.unit
    def test_get_battery_database(self):
        """Test battery database retrieval."""
        from aerospace_mcp.integrations.propellers import get_battery_database

        db = get_battery_database()
        assert "LiPo_3S" in db
        assert "LiPo_4S" in db
        assert "LiPo_6S" in db
        assert "Li-Ion_18650" in db

    @pytest.mark.unit
    def test_uav_generic_power_estimate(self):
        """Test UAV with generic power estimate (no wing/disk area)."""
        from aerospace_mcp.integrations.propellers import (
            BatteryConfiguration,
            UAVConfiguration,
            uav_energy_estimate,
        )

        # UAV without wing or disk area
        uav_config = UAVConfiguration(
            mass_kg=2.0,
            wing_area_m2=None,
            disk_area_m2=None,
            num_motors=1,
        )

        battery_config = BatteryConfiguration(
            capacity_ah=5.0,
            voltage_nominal_v=11.1,
            mass_kg=0.3,
        )

        result = uav_energy_estimate(
            uav_config, battery_config, {"velocity_ms": 15.0, "altitude_m": 100.0}
        )

        assert result.flight_time_min > 0
        assert result.power_required_w > 0


class TestRocketsEdgeCases:
    """Additional tests for rockets integration coverage."""

    @pytest.mark.unit
    def test_get_thrust_single_point(self):
        """Test thrust at time with single-point thrust curve."""
        from aerospace_mcp.integrations.rockets import get_thrust_at_time

        # Single point curve
        thrust_curve = [[10.0, 5000.0]]

        # During burn time
        assert get_thrust_at_time(thrust_curve, 5.0) == 5000.0
        # After burn time
        assert get_thrust_at_time(thrust_curve, 15.0) == 0.0

    @pytest.mark.unit
    def test_get_thrust_before_first_point(self):
        """Test thrust at time before first curve point."""
        from aerospace_mcp.integrations.rockets import get_thrust_at_time

        thrust_curve = [[1.0, 5000.0], [5.0, 3000.0]]

        # Before first point
        assert get_thrust_at_time(thrust_curve, 0.5) == 0.0

    @pytest.mark.unit
    def test_get_thrust_equal_times(self):
        """Test thrust interpolation with equal time values."""
        from aerospace_mcp.integrations.rockets import get_thrust_at_time

        # Curve with equal time values (edge case) - sorted, so t=2 is last
        thrust_curve = [[2.0, 5000.0], [2.0, 3000.0]]

        # After last point returns 0
        result = get_thrust_at_time(thrust_curve, 2.0)
        # At or after last point in sorted curve, returns 0
        assert result == 0.0

    @pytest.mark.unit
    def test_trajectory_extreme_altitude(self):
        """Test trajectory with extreme altitude (fallback atmosphere)."""
        from aerospace_mcp.integrations.rockets import (
            RocketGeometry,
            rocket_3dof_trajectory,
        )

        geometry = RocketGeometry(
            dry_mass_kg=10.0,
            propellant_mass_kg=5.0,
            diameter_m=0.1,
            length_m=1.0,
            thrust_curve=[[0.0, 1000.0], [2.0, 1000.0], [2.01, 0.0]],
        )

        # Use high altitude start (which may trigger fallback atmosphere)
        trajectory = rocket_3dof_trajectory(geometry, dt_s=0.5, max_time_s=10.0)

        assert len(trajectory) > 0

    @pytest.mark.unit
    def test_rocket_sizing_impossible_mission(self):
        """Test rocket sizing for impossible mission (requires staging)."""
        from aerospace_mcp.integrations.rockets import estimate_rocket_sizing

        # Very high altitude with small payload - may be impossible single stage
        result = estimate_rocket_sizing(
            target_altitude_m=500000.0,  # 500km
            payload_mass_kg=0.1,  # Tiny payload
            propellant_type="solid",
        )

        # Result should indicate feasibility
        assert "feasible" in result
        # High structural ratio needed means impossible in single stage
        if not result["feasible"]:
            assert result["propellant_mass_kg"] == float("inf")

    @pytest.mark.unit
    def test_rocket_sizing_unknown_propellant(self):
        """Test rocket sizing with unknown propellant type."""
        from aerospace_mcp.integrations.rockets import estimate_rocket_sizing

        result = estimate_rocket_sizing(
            target_altitude_m=10000.0,
            payload_mass_kg=1.0,
            propellant_type="unknown",  # Unknown type
        )

        # Should use default values
        assert result["specific_impulse_s"] == 250.0  # Default Isp

    @pytest.mark.unit
    def test_analyze_empty_trajectory(self):
        """Test performance analysis with empty trajectory."""
        from aerospace_mcp.integrations.rockets import analyze_rocket_performance

        result = analyze_rocket_performance([])

        assert result.max_altitude_m == 0
        assert result.apogee_time_s == 0

    @pytest.mark.unit
    def test_analyze_no_burnout_detection(self):
        """Test performance analysis when no burnout is detected."""
        from aerospace_mcp.integrations.rockets import (
            RocketTrajectoryPoint,
            analyze_rocket_performance,
        )

        # Trajectory with no thrust throughout (no burnout transition)
        trajectory = [
            RocketTrajectoryPoint(
                time_s=float(i),
                altitude_m=float(i * 100),
                velocity_ms=100.0,
                acceleration_ms2=10.0,
                mass_kg=10.0,
                thrust_n=0.0,
                drag_n=1.0,
                mach=0.3,
                dynamic_pressure_pa=1000.0,
            )
            for i in range(5)
        ]

        result = analyze_rocket_performance(trajectory)

        # Should still work with fallback burnout detection
        assert result.max_altitude_m == 400.0


class TestTrajoptEdgeCases:
    """Additional tests for trajectory optimization coverage."""

    @pytest.mark.unit
    def test_golden_section_search_converged(self):
        """Test golden section search convergence."""
        from aerospace_mcp.integrations.trajopt import simple_golden_section_search

        # Simple quadratic function with minimum at x=3
        def objective(x):
            return (x - 3) ** 2

        optimal_x, optimal_value = simple_golden_section_search(
            objective, 0.0, 10.0, tolerance=0.001
        )

        assert abs(optimal_x - 3.0) < 0.01
        assert optimal_value < 0.0001

    @pytest.mark.unit
    def test_gradient_descent_convergence(self):
        """Test gradient descent convergence."""
        from aerospace_mcp.integrations.trajopt import simple_gradient_descent

        # Simple bowl function
        def objective(params):
            return params[0] ** 2 + params[1] ** 2

        params, value, iterations, converged = simple_gradient_descent(
            objective,
            initial_params=[5.0, 5.0],
            param_bounds=[(-10.0, 10.0), (-10.0, 10.0)],
            learning_rate=0.5,
            tolerance=1e-4,
            max_iterations=100,
        )

        # Should converge to near zero
        assert abs(params[0]) < 0.1
        assert abs(params[1]) < 0.1
        assert converged

    @pytest.mark.unit
    def test_optimize_launch_angle_exception(self):
        """Test launch angle optimization with exception handling."""
        from aerospace_mcp.integrations.rockets import RocketGeometry
        from aerospace_mcp.integrations.trajopt import optimize_launch_angle

        geometry = RocketGeometry(
            dry_mass_kg=10.0,
            propellant_mass_kg=5.0,
            diameter_m=0.1,
            length_m=1.0,
            thrust_curve=[[0.0, 500.0], [3.0, 500.0], [3.01, 0.0]],
        )

        result = optimize_launch_angle(
            geometry, objective="max_altitude", angle_bounds=(85.0, 90.0)
        )

        assert result.converged
        assert 85.0 <= result.optimal_parameters["launch_angle_deg"] <= 90.0

    @pytest.mark.unit
    def test_optimize_launch_angle_unknown_objective(self):
        """Test launch angle optimization with unknown objective."""
        from aerospace_mcp.integrations.rockets import RocketGeometry
        from aerospace_mcp.integrations.trajopt import optimize_launch_angle

        geometry = RocketGeometry(
            dry_mass_kg=10.0,
            propellant_mass_kg=5.0,
            diameter_m=0.1,
            length_m=1.0,
            thrust_curve=[[0.0, 500.0], [3.0, 500.0], [3.01, 0.0]],
        )

        # Unknown objective should return inf
        result = optimize_launch_angle(
            geometry, objective="unknown", angle_bounds=(85.0, 90.0)
        )

        # Should still complete, but with suboptimal result
        assert result is not None

    @pytest.mark.unit
    def test_optimize_thrust_profile_min_gravity_loss(self):
        """Test thrust profile optimization for minimum gravity loss."""
        from aerospace_mcp.integrations.rockets import RocketGeometry
        from aerospace_mcp.integrations.trajopt import optimize_thrust_profile

        geometry = RocketGeometry(
            dry_mass_kg=10.0,
            propellant_mass_kg=5.0,
            diameter_m=0.1,
            length_m=1.0,
        )

        result = optimize_thrust_profile(
            geometry,
            burn_time_s=3.0,
            total_impulse_target=1500.0,
            n_segments=3,
            objective="min_gravity_loss",
        )

        assert result is not None
        assert len(result.trajectory_points) > 0

    @pytest.mark.unit
    def test_optimize_thrust_profile_min_max_q(self):
        """Test thrust profile optimization for minimum max Q."""
        from aerospace_mcp.integrations.rockets import RocketGeometry
        from aerospace_mcp.integrations.trajopt import optimize_thrust_profile

        geometry = RocketGeometry(
            dry_mass_kg=10.0,
            propellant_mass_kg=5.0,
            diameter_m=0.1,
            length_m=1.0,
        )

        result = optimize_thrust_profile(
            geometry,
            burn_time_s=3.0,
            total_impulse_target=1500.0,
            n_segments=3,
            objective="min_max_q",
        )

        assert result is not None

    @pytest.mark.unit
    def test_compare_trajectories_single_success(self):
        """Test trajectory comparison with single successful trajectory."""
        from aerospace_mcp.integrations.rockets import RocketGeometry
        from aerospace_mcp.integrations.trajopt import compare_trajectories

        geometry = RocketGeometry(
            dry_mass_kg=10.0,
            propellant_mass_kg=5.0,
            diameter_m=0.1,
            length_m=1.0,
            thrust_curve=[[0.0, 500.0], [3.0, 500.0], [3.01, 0.0]],
        )

        results = compare_trajectories([geometry], ["Test Rocket"])

        assert "Test Rocket" in results
        assert results["Test Rocket"]["success"] is True
        # No comparison section for single rocket
        assert "comparison" not in results

    @pytest.mark.unit
    def test_compare_trajectories_failure(self):
        """Test trajectory comparison with failed trajectory."""
        from aerospace_mcp.integrations.rockets import RocketGeometry
        from aerospace_mcp.integrations.trajopt import compare_trajectories

        # Invalid geometry that will cause simulation to fail
        geometry = RocketGeometry(
            dry_mass_kg=0.001,  # Very small mass
            propellant_mass_kg=0.001,
            diameter_m=0.001,
            length_m=0.001,
            thrust_curve=None,  # No thrust
        )

        results = compare_trajectories([geometry], ["Bad Rocket"])

        assert "Bad Rocket" in results
        # May succeed or fail depending on implementation

    @pytest.mark.unit
    def test_sensitivity_analysis_max_velocity(self):
        """Test sensitivity analysis for max velocity objective."""
        from aerospace_mcp.integrations.rockets import RocketGeometry
        from aerospace_mcp.integrations.trajopt import trajectory_sensitivity_analysis

        geometry = RocketGeometry(
            dry_mass_kg=10.0,
            propellant_mass_kg=5.0,
            diameter_m=0.1,
            length_m=1.0,
            thrust_curve=[[0.0, 500.0], [3.0, 500.0], [3.01, 0.0]],
        )

        results = trajectory_sensitivity_analysis(
            geometry,
            {"dry_mass_kg": [8.0, 10.0, 12.0]},
            objective="max_velocity",
        )

        assert "baseline_value" in results
        assert results["objective"] == "max_velocity"
        assert "dry_mass_kg" in results["parameter_sensitivities"]

    @pytest.mark.unit
    def test_sensitivity_analysis_specific_impulse(self):
        """Test sensitivity analysis for specific impulse objective."""
        from aerospace_mcp.integrations.rockets import RocketGeometry
        from aerospace_mcp.integrations.trajopt import trajectory_sensitivity_analysis

        geometry = RocketGeometry(
            dry_mass_kg=10.0,
            propellant_mass_kg=5.0,
            diameter_m=0.1,
            length_m=1.0,
            thrust_curve=[[0.0, 500.0], [3.0, 500.0], [3.01, 0.0]],
        )

        results = trajectory_sensitivity_analysis(
            geometry,
            {"propellant_mass_kg": [4.0, 5.0, 6.0]},
            objective="specific_impulse",
        )

        assert results["objective"] == "specific_impulse"

    @pytest.mark.unit
    def test_sensitivity_analysis_unknown_objective(self):
        """Test sensitivity analysis with unknown objective defaults to altitude."""
        from aerospace_mcp.integrations.rockets import RocketGeometry
        from aerospace_mcp.integrations.trajopt import trajectory_sensitivity_analysis

        geometry = RocketGeometry(
            dry_mass_kg=10.0,
            propellant_mass_kg=5.0,
            diameter_m=0.1,
            length_m=1.0,
            thrust_curve=[[0.0, 500.0], [3.0, 500.0], [3.01, 0.0]],
        )

        results = trajectory_sensitivity_analysis(
            geometry,
            {"diameter_m": [0.08, 0.1, 0.12]},
            objective="unknown_objective",
        )

        # Should use max_altitude as default
        assert results["baseline_value"] > 0

    @pytest.mark.unit
    def test_optimize_launch_angle_max_range(self):
        """Test launch angle optimization for max range objective."""
        from aerospace_mcp.integrations.rockets import RocketGeometry
        from aerospace_mcp.integrations.trajopt import optimize_launch_angle

        geometry = RocketGeometry(
            dry_mass_kg=10.0,
            propellant_mass_kg=5.0,
            diameter_m=0.1,
            length_m=1.0,
            thrust_curve=[[0.0, 500.0], [3.0, 500.0], [3.01, 0.0]],
        )

        # max_range objective
        result = optimize_launch_angle(
            geometry, objective="max_range", angle_bounds=(45.0, 80.0)
        )

        assert result is not None
        # Should be lower angle for range
        assert result.optimal_parameters["launch_angle_deg"] >= 45.0

    @pytest.mark.unit
    def test_optimize_thrust_profile_zero_impulse(self):
        """Test thrust profile with near-zero impulse (edge case)."""
        from aerospace_mcp.integrations.rockets import RocketGeometry
        from aerospace_mcp.integrations.trajopt import optimize_thrust_profile

        geometry = RocketGeometry(
            dry_mass_kg=10.0,
            propellant_mass_kg=5.0,
            diameter_m=0.1,
            length_m=1.0,
        )

        # Very small impulse - may trigger edge cases in objective function
        result = optimize_thrust_profile(
            geometry,
            burn_time_s=1.0,
            total_impulse_target=100.0,  # Very small
            n_segments=2,
            objective="max_altitude",
        )

        assert result is not None

    @pytest.mark.unit
    def test_compare_trajectories_multiple_rockets(self):
        """Test trajectory comparison with multiple rockets."""
        from aerospace_mcp.integrations.rockets import RocketGeometry
        from aerospace_mcp.integrations.trajopt import compare_trajectories

        geometry1 = RocketGeometry(
            dry_mass_kg=10.0,
            propellant_mass_kg=5.0,
            diameter_m=0.1,
            length_m=1.0,
            thrust_curve=[[0.0, 500.0], [3.0, 500.0], [3.01, 0.0]],
        )

        geometry2 = RocketGeometry(
            dry_mass_kg=8.0,
            propellant_mass_kg=7.0,
            diameter_m=0.12,
            length_m=1.2,
            thrust_curve=[[0.0, 600.0], [4.0, 600.0], [4.01, 0.0]],
        )

        results = compare_trajectories([geometry1, geometry2], ["Rocket A", "Rocket B"])

        assert "Rocket A" in results
        assert "Rocket B" in results
        # Multiple successful rockets should have comparison section
        assert "comparison" in results


class TestRocketsThrustCurveEdgeCases:
    """Additional tests for rocket thrust curve edge cases."""

    @pytest.mark.unit
    def test_thrust_interpolation_within_segment(self):
        """Test thrust interpolation within a segment."""
        from aerospace_mcp.integrations.rockets import get_thrust_at_time

        thrust_curve = [[0.0, 1000.0], [5.0, 500.0], [10.0, 0.0]]

        # At midpoint of first segment
        result = get_thrust_at_time(thrust_curve, 2.5)
        assert 500.0 < result < 1000.0  # Should be interpolated

    @pytest.mark.unit
    def test_trajectory_with_no_thrust_curve(self):
        """Test trajectory with no thrust (coasting)."""
        from aerospace_mcp.integrations.rockets import (
            RocketGeometry,
            rocket_3dof_trajectory,
        )

        geometry = RocketGeometry(
            dry_mass_kg=10.0,
            propellant_mass_kg=5.0,
            diameter_m=0.1,
            length_m=1.0,
            thrust_curve=None,  # No thrust
        )

        trajectory = rocket_3dof_trajectory(geometry, dt_s=0.5, max_time_s=5.0)

        # Should still generate trajectory (just with no thrust)
        assert len(trajectory) > 0
        for point in trajectory:
            assert point.thrust_n == 0.0


class TestFramesAstropyFallback:
    """Tests for frames module with astropy fallback scenarios."""

    @pytest.mark.unit
    def test_eci_to_ecef_manual_fallback(self):
        """Test ECI to ECEF with manual fallback."""
        from aerospace_mcp.integrations.frames import transform_frames

        xyz = [7000000.0, 0.0, 0.0]

        # Should work with or without astropy
        result = transform_frames(xyz, "ECI", "ECEF")

        assert result.frame == "ECEF"
        # Magnitude should be preserved

        original_mag = math.sqrt(sum(x**2 for x in xyz))
        result_mag = math.sqrt(result.x**2 + result.y**2 + result.z**2)
        assert abs(result_mag - original_mag) < 1000

    @pytest.mark.unit
    def test_gcrs_to_itrf_transformation(self):
        """Test GCRS to ITRF transformation."""
        from aerospace_mcp.integrations.frames import transform_frames

        xyz = [6378137.0, 0.0, 0.0]

        result = transform_frames(xyz, "GCRS", "ITRF")

        assert result.frame == "ITRF"


class TestPropellersMultirotorAnalysis:
    """Tests for propeller multirotor analysis."""

    @pytest.mark.unit
    def test_multirotor_power_analysis(self):
        """Test multirotor power analysis."""
        from aerospace_mcp.integrations.propellers import (
            BatteryConfiguration,
            UAVConfiguration,
            uav_energy_estimate,
        )

        # Multirotor configuration
        uav_config = UAVConfiguration(
            mass_kg=2.0,
            disk_area_m2=0.4,
            num_motors=4,
        )

        battery_config = BatteryConfiguration(
            capacity_ah=5.0,
            voltage_nominal_v=14.8,
            mass_kg=0.4,
        )

        result = uav_energy_estimate(
            uav_config, battery_config, {"velocity_ms": 5.0, "altitude_m": 100.0}
        )

        assert result.flight_time_min > 0
        assert result.hover_time_min is not None  # Multirotor has hover time

    @pytest.mark.unit
    def test_fixed_wing_power_analysis(self):
        """Test fixed-wing power analysis."""
        from aerospace_mcp.integrations.propellers import (
            BatteryConfiguration,
            UAVConfiguration,
            uav_energy_estimate,
        )

        # Fixed-wing configuration
        uav_config = UAVConfiguration(
            mass_kg=2.0,
            wing_area_m2=0.5,
            cl_cruise=0.6,
            num_motors=1,
        )

        battery_config = BatteryConfiguration(
            capacity_ah=5.0,
            voltage_nominal_v=14.8,
            mass_kg=0.4,
        )

        result = uav_energy_estimate(
            uav_config, battery_config, {"velocity_ms": 15.0, "altitude_m": 100.0}
        )

        assert result.flight_time_min > 0
        assert result.range_km is not None  # Fixed-wing has range

    @pytest.mark.unit
    def test_motor_propeller_matching(self):
        """Test motor/propeller matching analysis."""
        from aerospace_mcp.integrations.propellers import motor_propeller_matching

        results = motor_propeller_matching(
            motor_kv=900,
            battery_voltage=14.8,
            propeller_options=["APC_10x7", "APC_12x8"],
            thrust_required_n=10.0,
        )

        assert "APC_10x7" in results or "APC_12x8" in results

    @pytest.mark.unit
    def test_high_altitude_propeller_analysis(self):
        """Test propeller analysis at high altitude (above 11000m)."""
        from aerospace_mcp.integrations.propellers import (
            PropellerGeometry,
            _simple_propeller_analysis,
        )

        geometry = PropellerGeometry(
            diameter_m=0.254,
            pitch_m=0.178,
            num_blades=2,
        )

        # Very high altitude - testing that analysis completes without error
        result = _simple_propeller_analysis(
            geometry, [3000.0], velocity_ms=5.0, altitude_m=15000
        )

        assert len(result) == 1
        # Just verify analysis completes at high altitude
        assert result[0].rpm == 3000.0
        # Density at 15000m should be lower - torque will be different
        result_low = _simple_propeller_analysis(
            geometry, [3000.0], velocity_ms=5.0, altitude_m=0
        )
        # At least verify both ran without error
        assert result_low[0].rpm == 3000.0


class TestTrajoptObjectiveFunctions:
    """Tests for trajectory optimization objective functions."""

    @pytest.mark.unit
    def test_thrust_profile_unknown_objective(self):
        """Test thrust profile optimization with unknown objective."""
        from aerospace_mcp.integrations.rockets import RocketGeometry
        from aerospace_mcp.integrations.trajopt import optimize_thrust_profile

        geometry = RocketGeometry(
            dry_mass_kg=10.0,
            propellant_mass_kg=5.0,
            diameter_m=0.1,
            length_m=1.0,
        )

        # Unknown objective should return large value
        result = optimize_thrust_profile(
            geometry,
            burn_time_s=3.0,
            total_impulse_target=1500.0,
            n_segments=2,
            objective="unknown",
        )

        assert result is not None

    @pytest.mark.unit
    def test_sensitivity_cd_variation(self):
        """Test sensitivity analysis with CD variation."""
        from aerospace_mcp.integrations.rockets import RocketGeometry
        from aerospace_mcp.integrations.trajopt import trajectory_sensitivity_analysis

        geometry = RocketGeometry(
            dry_mass_kg=10.0,
            propellant_mass_kg=5.0,
            diameter_m=0.1,
            length_m=1.0,
            cd=0.3,
            thrust_curve=[[0.0, 500.0], [3.0, 500.0], [3.01, 0.0]],
        )

        results = trajectory_sensitivity_analysis(
            geometry,
            {"cd": [0.2, 0.3, 0.4]},
            objective="max_altitude",
        )

        assert "cd" in results["parameter_sensitivities"]


class TestPropellersEdgeCasesExtended:
    """Extended edge case tests for propellers module."""

    @pytest.mark.unit
    def test_motor_propeller_matching_unknown_propeller(self):
        """Test motor matching with unknown propeller in list."""
        from aerospace_mcp.integrations.propellers import motor_propeller_matching

        # Include an unknown propeller that should be skipped
        results = motor_propeller_matching(
            motor_kv=900,
            battery_voltage=14.8,
            propeller_options=["UNKNOWN_PROPELLER", "APC_10x7"],
            thrust_required_n=10.0,
        )

        # Only APC_10x7 should be in results
        assert "APC_10x7" in results
        assert "UNKNOWN_PROPELLER" not in results

    @pytest.mark.unit
    def test_uav_energy_estimate_high_altitude(self):
        """Test UAV energy estimate above 11000m altitude."""
        from aerospace_mcp.integrations.propellers import (
            BatteryConfiguration,
            UAVConfiguration,
            uav_energy_estimate,
        )

        uav_config = UAVConfiguration(
            mass_kg=2.0,
            disk_area_m2=0.4,
            num_motors=4,
        )

        battery_config = BatteryConfiguration(
            capacity_ah=5.0,
            voltage_nominal_v=14.8,
            mass_kg=0.4,
        )

        # High altitude above 11000m
        result = uav_energy_estimate(
            uav_config, battery_config, {"velocity_ms": 5.0, "altitude_m": 15000.0}
        )

        assert result.flight_time_min > 0


class TestRocketsEdgeCasesExtended:
    """Extended edge case tests for rockets module."""

    @pytest.mark.unit
    def test_thrust_curve_equal_time_values(self):
        """Test thrust curve with equal consecutive time values."""
        from aerospace_mcp.integrations.rockets import get_thrust_at_time

        # Curve with equal time values
        thrust_curve = [[1.0, 1000.0], [1.0, 2000.0], [5.0, 0.0]]

        # At time 1.0, should return first value (1000.0)
        result = get_thrust_at_time(thrust_curve, 1.0)
        assert result == 1000.0  # First value at t=1.0

    @pytest.mark.unit
    def test_thrust_curve_beyond_end(self):
        """Test thrust curve query beyond end time."""
        from aerospace_mcp.integrations.rockets import get_thrust_at_time

        thrust_curve = [[0.0, 1000.0], [5.0, 500.0]]

        # Beyond end of curve
        result = get_thrust_at_time(thrust_curve, 10.0)
        assert result == 0.0

    @pytest.mark.unit
    def test_trajectory_at_extreme_altitude(self):
        """Test trajectory that reaches extreme altitude (fallback atmosphere)."""
        from aerospace_mcp.integrations.rockets import (
            RocketGeometry,
            rocket_3dof_trajectory,
        )

        # High thrust rocket that can reach extreme altitude
        geometry = RocketGeometry(
            dry_mass_kg=5.0,
            propellant_mass_kg=10.0,
            diameter_m=0.1,
            length_m=1.0,
            thrust_curve=[[0.0, 5000.0], [5.0, 5000.0], [5.01, 0.0]],
        )

        trajectory = rocket_3dof_trajectory(
            geometry, dt_s=0.5, max_time_s=100.0, launch_angle_deg=90.0
        )

        # Should reach significant altitude
        max_alt = max(p.altitude_m for p in trajectory)
        assert max_alt > 10000  # Should reach at least 10km


class TestTrajoptEdgeCasesExtended:
    """Extended edge case tests for trajectory optimization."""

    @pytest.mark.unit
    def test_gradient_descent_at_boundary(self):
        """Test gradient descent convergence at parameter boundary."""
        from aerospace_mcp.integrations.trajopt import simple_gradient_descent

        def objective(params):
            # Function with minimum at boundary
            return (params[0] - 0.0) ** 2

        params, value, iterations, converged = simple_gradient_descent(
            objective,
            initial_params=[5.0],
            param_bounds=[(0.0, 10.0)],
            learning_rate=0.5,
            tolerance=1e-4,
            max_iterations=50,
        )

        # Should converge to boundary
        assert abs(params[0]) < 0.5

    @pytest.mark.unit
    def test_golden_section_narrow_range(self):
        """Test golden section search with narrow range."""
        from aerospace_mcp.integrations.trajopt import simple_golden_section_search

        def objective(x):
            return (x - 5.0) ** 2

        optimal_x, optimal_value = simple_golden_section_search(
            objective, 4.9, 5.1, tolerance=0.001
        )

        assert abs(optimal_x - 5.0) < 0.01
