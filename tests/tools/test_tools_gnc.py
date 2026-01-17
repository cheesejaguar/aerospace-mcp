"""Tests for aerospace_mcp.tools.gnc module."""

import json

from aerospace_mcp.tools.gnc import (
    kalman_filter_state_estimation,
    lqr_controller_design,
)


def extract_json(result: str) -> dict:
    """Extract JSON data from tool output that contains both text and JSON."""
    brace_count = 0
    json_start = None
    last_json_end = None

    for i, char in enumerate(result):
        if char == "{":
            if brace_count == 0:
                json_start = i
            brace_count += 1
        elif char == "}":
            brace_count -= 1
            if brace_count == 0 and json_start is not None:
                last_json_end = i + 1

    if json_start is not None and last_json_end is not None:
        try:
            return json.loads(result[json_start:last_json_end])
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not extract JSON from result: {result[:200]}...")


class TestKalmanFilterStateEstimation:
    """Tests for kalman_filter_state_estimation function."""

    def test_basic_2d_tracking(self):
        """Test basic 2D position tracking."""
        initial_state = [0, 0]
        initial_covariance = [[1, 0], [0, 1]]
        process_noise = [[0.1, 0], [0, 0.1]]
        measurement_noise = [[0.5, 0], [0, 0.5]]

        measurements = [
            {"z": [1.0, 0.5]},
            {"z": [2.0, 1.0]},
            {"z": [3.0, 1.5]},
            {"z": [4.0, 2.0]},
            {"z": [5.0, 2.5]},
        ]

        result = kalman_filter_state_estimation(
            initial_state=initial_state,
            initial_covariance=initial_covariance,
            process_noise=process_noise,
            measurement_noise=measurement_noise,
            measurements=measurements,
            dynamics_model="constant_velocity",
        )

        assert "KALMAN FILTER" in result
        data = extract_json(result)
        assert data["output"]["num_states_estimated"] == 5

    def test_constant_velocity_model(self):
        """Test constant velocity dynamics model."""
        initial_state = [0, 0, 1, 1]
        initial_covariance = [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0.1, 0],
            [0, 0, 0, 0.1],
        ]
        process_noise = [
            [0.1, 0, 0, 0],
            [0, 0.1, 0, 0],
            [0, 0, 0.01, 0],
            [0, 0, 0, 0.01],
        ]
        measurement_noise = [
            [0.5, 0, 0, 0],
            [0, 0.5, 0, 0],
            [0, 0, 0.1, 0],
            [0, 0, 0, 0.1],
        ]

        measurements = [
            {"z": [1, 1, 1, 1]},
            {"z": [2, 2, 1, 1]},
            {"z": [3, 3, 1, 1]},
        ]

        result = kalman_filter_state_estimation(
            initial_state=initial_state,
            initial_covariance=initial_covariance,
            process_noise=process_noise,
            measurement_noise=measurement_noise,
            measurements=measurements,
            dynamics_model="constant_velocity",
            dt=1.0,
        )

        data = extract_json(result)
        assert data["input"]["dynamics_model"] == "constant_velocity"

    def test_empty_measurements(self):
        """Test with no measurements."""
        result = kalman_filter_state_estimation(
            initial_state=[0, 0],
            initial_covariance=[[1, 0], [0, 1]],
            process_noise=[[0.1, 0], [0, 0.1]],
            measurement_noise=[[0.5, 0], [0, 0.5]],
            measurements=[],
        )
        # Should handle gracefully
        assert "KALMAN FILTER" in result or "final_state" in result


class TestLQRControllerDesign:
    """Tests for lqr_controller_design function."""

    def test_simple_double_integrator(self):
        """Test LQR for simple double integrator system."""
        A = [[0, 1], [0, 0]]
        B = [[0], [1]]
        Q = [[1, 0], [0, 1]]
        R = [[1]]

        result = lqr_controller_design(
            A_matrix=A,
            B_matrix=B,
            Q_matrix=Q,
            R_matrix=R,
            state_names=["position", "velocity"],
            input_names=["acceleration"],
        )

        assert "LQR CONTROLLER" in result
        data = extract_json(result)

        # System should be controllable
        assert data["controllability"]["is_controllable"] is True

        # Should have a gain matrix
        assert "gain_matrix_K" in data["lqr_solution"]
        assert len(data["lqr_solution"]["gain_matrix_K"]) == 1
        assert len(data["lqr_solution"]["gain_matrix_K"][0]) == 2

    def test_stable_closed_loop(self):
        """Test that closed-loop system is stable."""
        A = [[1, 1], [0, 1]]  # Unstable (eigenvalue at 1)
        B = [[0], [1]]
        Q = [[10, 0], [0, 1]]
        R = [[1]]

        result = lqr_controller_design(
            A_matrix=A,
            B_matrix=B,
            Q_matrix=Q,
            R_matrix=R,
        )

        data = extract_json(result)

        # Closed-loop should be stable (all eigenvalues in LHP)
        assert data["stability_analysis"]["closed_loop_stable"] is True
        for eig in data["stability_analysis"]["closed_loop_eigenvalues"]:
            assert eig < 0

    def test_uncontrollable_system(self):
        """Test handling of uncontrollable system."""
        A = [[1, 0], [0, 2]]
        B = [[0], [0]]
        Q = [[1, 0], [0, 1]]
        R = [[1]]

        result = lqr_controller_design(
            A_matrix=A,
            B_matrix=B,
            Q_matrix=Q,
            R_matrix=R,
        )

        # Should report controllability issue
        assert "controllable" in result.lower() or "error" in result.lower()

    def test_state_names_in_output(self):
        """Test that state names appear in output."""
        A = [[0, 1], [0, 0]]
        B = [[0], [1]]
        Q = [[1, 0], [0, 1]]
        R = [[1]]

        result = lqr_controller_design(
            A_matrix=A,
            B_matrix=B,
            Q_matrix=Q,
            R_matrix=R,
            state_names=["altitude", "climb_rate"],
            input_names=["thrust"],
        )

        assert "altitude" in result
        assert "climb_rate" in result
