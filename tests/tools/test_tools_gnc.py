"""Tests for aerospace_mcp.tools.gnc module."""

import json
import math

import numpy as np
import pytest
import scipy.linalg

from aerospace_mcp.tools.gnc import (
    kalman_filter_state_estimation,
    lqr_controller_design,
)
from tests.tools.test_tools_common import import_raising

CV_F = [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]]  # dt = 1
POS_H = [[1, 0, 0, 0], [0, 1, 0, 0]]


def reference_kf(x0, P0, F, Q, R, H, zs):
    """Textbook discrete KF (Simon 2006, Eq. 5.19) with an explicit inverse."""
    x, P = np.array(x0, float), np.array(P0, float)
    F, Q, R, H = (np.array(m, float) for m in (F, Q, R, H))
    for z in zs:
        x, P = F @ x, F @ P @ F.T + Q
        K = P @ H.T @ np.linalg.inv(H @ P @ H.T + R)
        x = x + K @ (np.array(z, float) - H @ x)
        P = (np.eye(len(x)) - K @ H) @ P
    return x, np.sqrt(np.diag(P))


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

        # Closed-loop should be stable (all eigenvalues in LHP).  Eigenvalues
        # are [real, imag] pairs (previously the imaginary part was dropped).
        assert data["stability_analysis"]["closed_loop_stable"] is True
        for re, _im in data["stability_analysis"]["closed_loop_eigenvalues"]:
            assert re < 0

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


class TestKalmanKnownAnswers:
    """Unstubbed known-answer checks against a textbook reference KF."""

    def test_full_state_matches_exact_kf(self):
        # Correlated P0 makes the old diagonal-gain approximation visibly wrong
        # (it gave [4.829, 10.078, 0.961, 1.962]).
        P0 = [[10, 5, 3, 0], [5, 10, 0, 3], [3, 0, 10, 0], [0, 3, 0, 10]]
        Q = (np.eye(4) * 0.01).tolist()
        R = np.eye(4).tolist()
        zs = [
            [t + 0.3 * (-1) ** t, 2.0 * t - 0.2 * (-1) ** t, 1 + 0.1 * (-1) ** t, 2.0]
            for t in range(1, 6)
        ]
        out = extract_json(
            kalman_filter_state_estimation(
                [0, 0, 0, 0],
                P0,
                Q,
                R,
                [{"time": t, "z": z} for t, z in enumerate(zs, 1)],
            )
        )
        x_ref, s_ref = reference_kf([0, 0, 0, 0], P0, CV_F, Q, R, np.eye(4), zs)
        np.testing.assert_allclose(out["output"]["final_state"], x_ref, atol=1e-6)
        np.testing.assert_allclose(out["output"]["final_state_std"], s_ref, atol=1e-6)
        np.testing.assert_allclose(
            out["output"]["final_state"], [4.9096, 10.0060, 0.9819, 1.9803], atol=1e-4
        )

    def test_position_only_measurements(self):
        """2 measured components, 4 states (previously 'list index out of range')."""
        zs = [[1.0 * t, 2.0 * t] for t in range(1, 21)]  # truth vx=1, vy=2
        P0 = (np.eye(4) * 10).tolist()
        Q = (np.eye(4) * 0.01).tolist()
        meas = [{"time": t, "z": z, "H": POS_H} for t, z in enumerate(zs, 1)]
        out = extract_json(
            kalman_filter_state_estimation(
                [0, 0, 0, 0], P0, Q, np.eye(2).tolist(), meas
            )
        )
        x_ref, s_ref = reference_kf([0] * 4, P0, CV_F, Q, np.eye(2), POS_H, zs)
        np.testing.assert_allclose(out["output"]["final_state"], x_ref, atol=1e-6)
        np.testing.assert_allclose(out["output"]["final_state_std"], s_ref, atol=1e-6)
        np.testing.assert_allclose(
            out["output"]["final_state"][2:], [1.0001, 2.0002], atol=1e-4
        )

    def test_missing_h_for_partial_measurement_is_explained(self):
        out = kalman_filter_state_estimation(
            [0, 0, 0, 0],
            np.eye(4).tolist(),
            np.eye(4).tolist(),
            np.eye(2).tolist(),
            [{"z": [1.0, 2.0]}],
        )
        assert out.startswith("Error in Kalman filter")
        assert "'H' of shape 2x4" in out

    def test_mismatched_r_is_rejected(self):
        out = kalman_filter_state_estimation(
            [0, 0, 0, 0],
            np.eye(4).tolist(),
            np.eye(4).tolist(),
            np.eye(4).tolist(),  # R is 4x4 but z has 2 components
            [{"z": [1.0, 2.0], "H": POS_H}],
        )
        assert "measurement_noise is 4x4" in out

    def test_timestamps_set_the_step_length(self):
        """Measurements every 0.5 s of x = t, vx = 1 (noise-free).

        Previously the filter predicted with dt = 1 regardless of the
        timestamps and filterpy ended at vx = 0.5.
        """
        meas = [
            {"time": 0.5 * k, "z": [0.5 * k, 1.0 * k], "H": POS_H} for k in range(1, 41)
        ]
        out = extract_json(
            kalman_filter_state_estimation(
                [0, 0, 1, 2],
                np.eye(4).tolist(),
                (np.eye(4) * 1e-6).tolist(),
                (np.eye(2) * 1e-4).tolist(),
                meas,
                initial_time=0.0,
            )
        )
        np.testing.assert_allclose(
            out["output"]["final_state"], [20.0, 40.0, 1.0, 2.0], atol=1e-4
        )
        assert max(out["statistics"]["innovation_rms"]) < 1e-6
        assert out["output"]["final_time"] == 20.0
        # Rows are labelled with measurement times, not indices
        assert out["filtered_times"] == [0.5 * k for k in range(31, 41)]

    def test_out_of_order_times_rejected(self):
        meas = [{"time": 2, "z": [1, 1]}, {"time": 1, "z": [1, 1]}]
        out = kalman_filter_state_estimation(
            [0, 0], np.eye(2).tolist(), np.eye(2).tolist(), np.eye(2).tolist(), meas
        )
        assert "time order" in out

    @pytest.mark.parametrize(
        "model,n",
        [("constant_velocity", 5), ("constant_acceleration", 4), ("orbital", 4)],
    )
    def test_dimension_mismatch_rejected(self, model, n):
        eye = np.eye(n).tolist()
        out = kalman_filter_state_estimation(
            [1.0] * n, eye, eye, eye, [{"z": [1.0] * n}], dynamics_model=model
        )
        assert out.startswith("Error in Kalman filter")

    def test_unknown_model_rejected(self):
        eye = np.eye(2).tolist()
        out = kalman_filter_state_estimation(
            [0, 0], eye, eye, eye, [{"z": [0, 0]}], dynamics_model="warp_drive"
        )
        assert "Unknown dynamics_model" in out

    def test_constant_acceleration_1d(self):
        """x = t + 0.05 t^2 sampled at 1 Hz -> v(29) = 3.9, a = 0.1."""
        meas = [
            {"time": t, "z": [t + 0.05 * t * t], "H": [[1, 0, 0]]} for t in range(1, 30)
        ]
        out = extract_json(
            kalman_filter_state_estimation(
                [0, 1, 0.1],
                np.eye(3).tolist(),
                (np.eye(3) * 1e-8).tolist(),
                [[1e-4]],
                meas,
                dynamics_model="constant_acceleration",
                initial_time=0,
            )
        )
        np.testing.assert_allclose(
            out["output"]["final_state"], [71.05, 3.9, 0.1], atol=1e-3
        )

    def test_orbital_model_is_two_body(self):
        """Orbital EKF prediction follows two-body motion, not identity.

        Circular LEO: after a quarter period (Kepler's third law,
        T = 2 pi sqrt(a^3 / mu)) the satellite moves from +x to +y.
        Measurements are given huge noise so the estimate is the prediction.
        """
        mu, r = 3.986004418e14, 7.0e6
        v = math.sqrt(mu / r)
        quarter = 0.5 * math.pi * math.sqrt(r**3 / mu)
        big = (np.eye(6) * 1e20).tolist()
        out = extract_json(
            kalman_filter_state_estimation(
                [r, 0, 0, 0, v, 0],
                np.eye(6).tolist(),
                np.zeros((6, 6)).tolist(),
                big,
                [{"time": quarter, "z": [0, 0, 0, 0, 0, 0]}],
                dynamics_model="orbital",
                initial_time=0.0,
            )
        )
        x = out["output"]["final_state"]
        np.testing.assert_allclose(x, [0, r, 0, -v, 0, 0], atol=0.5)


class TestLQRKnownAnswers:
    """LQR against scipy.linalg.solve_continuous_are and closed-form CARE."""

    @staticmethod
    def _design(A, B, Q, R, force_scipy):
        if force_scipy:
            with import_raising("control"):
                return lqr_controller_design(A, B, Q, R)
        return lqr_controller_design(A, B, Q, R)

    @pytest.mark.parametrize("force_scipy", [True, False])
    def test_double_integrator(self, force_scipy):
        """K = [1, sqrt(3)], poles -sqrt(3)/2 +/- 0.5j (closed-form CARE)."""
        data = extract_json(
            self._design(
                [[0, 1], [0, 0]], [[0], [1]], np.eye(2).tolist(), [[1]], force_scipy
            )
        )
        np.testing.assert_allclose(
            data["lqr_solution"]["gain_matrix_K"], [[1.0, math.sqrt(3)]], atol=1e-6
        )
        np.testing.assert_allclose(
            data["stability_analysis"]["closed_loop_eigenvalues"],
            [[-math.sqrt(3) / 2, -0.5], [-math.sqrt(3) / 2, 0.5]],
            atol=1e-6,
        )
        if force_scipy:
            assert data["implementation"] == "scipy"

    def test_matches_scipy_care(self):
        A, B, Q, R = [[0, 1], [-4, -0.4]], [[0], [1]], np.eye(2), np.eye(1)
        S = scipy.linalg.solve_continuous_are(
            np.array(A, float), np.array(B, float), Q, R
        )
        K_ref = np.linalg.solve(R, np.array(B, float).T @ S)
        data = extract_json(self._design(A, B, Q.tolist(), R.tolist(), True))
        np.testing.assert_allclose(
            data["lqr_solution"]["gain_matrix_K"], K_ref, atol=1e-6
        )
        np.testing.assert_allclose(K_ref, [[0.1231, 0.7858]], atol=1e-4)

    def test_negative_input_gain_is_stabilized(self):
        """B = [0, -1]^T: K = [-1, -sqrt(3)]; the old heuristic reported a
        positive K and 'Stable' while eig(A - BK) had a RHP pole."""
        A, B = [[0, 1], [0, 0]], [[0], [-1]]
        data = extract_json(self._design(A, B, np.eye(2).tolist(), [[1]], True))
        K = np.array(data["lqr_solution"]["gain_matrix_K"])
        np.testing.assert_allclose(K, [[-1.0, -math.sqrt(3)]], atol=1e-6)
        assert np.all(np.linalg.eigvals(np.array(A) - np.array(B) @ K).real < 0)
        assert data["stability_analysis"]["closed_loop_stable"] is True

    def test_scalar_system(self):
        """A = B = Q = R = 1: CARE 2S - S^2 + 1 = 0 -> K = S = 1 + sqrt(2)."""
        data = extract_json(self._design([[1.0]], [[1.0]], [[1.0]], [[1.0]], True))
        assert (
            abs(data["lqr_solution"]["gain_matrix_K"][0][0] - (1 + math.sqrt(2))) < 1e-6
        )
        assert data["stability_analysis"]["closed_loop_eigenvalues"] == [
            [round(-math.sqrt(2), 6), 0.0]
        ]

    def test_three_state_chain(self):
        """Triple integrator works without python-control (was rejected)."""
        A = [[0, 1, 0], [0, 0, 1], [0, 0, 0]]
        B = [[0], [0], [1]]
        data = extract_json(self._design(A, B, np.eye(3).tolist(), [[1]], True))
        S = scipy.linalg.solve_continuous_are(
            np.array(A, float), np.array(B, float), np.eye(3), np.eye(1)
        )
        np.testing.assert_allclose(
            data["lqr_solution"]["gain_matrix_K"], np.array(B, float).T @ S, atol=1e-6
        )

    @pytest.mark.parametrize("force_scipy", [True, False])
    def test_unstabilizable_reports_error_json(self, force_scipy):
        """Uncontrollable unstable mode at s = 2 (python-control path used to
        crash with 'int64 is not JSON serializable')."""
        out = self._design(
            [[1, 0], [0, 2]], [[1], [0]], np.eye(2).tolist(), [[1]], force_scipy
        )
        data = json.loads(out)
        assert "not stabilizable" in data["error"]
        assert data["controllability_rank"] == 1
        assert data["uncontrollable_unstable_eigenvalues"] == [[2.0, 0.0]]

    def test_uncontrollable_but_stabilizable(self):
        """Mode at s = -1 is uncontrollable but stable: LQR still exists."""
        data = extract_json(
            self._design([[-1, 0], [0, 2]], [[0], [1]], np.eye(2).tolist(), [[1]], True)
        )
        assert data["controllability"]["is_controllable"] is False
        assert data["controllability"]["is_stabilizable"] is True
        assert data["stability_analysis"]["closed_loop_stable"] is True
        # Scalar CARE for the controllable mode: 4S - S^2 + 1 = 0 -> S = 2 + sqrt(5)
        assert (
            abs(data["lqr_solution"]["gain_matrix_K"][0][1] - (2 + math.sqrt(5))) < 1e-6
        )
        assert data["warnings"]

    def test_invalid_weights_rejected(self):
        out = self._design([[0]], [[1]], [[1]], [[-1]], True)
        assert "positive definite" in out
        out = self._design([[0, 1], [0, 0]], [[0, 1]], np.eye(2).tolist(), [[1]], True)
        assert "B_matrix must have 2 rows" in out
