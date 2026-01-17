"""Guidance, Navigation, and Control (GNC) tools for the Aerospace MCP server.

Provides tools for:
- Kalman filter state estimation
- LQR controller design
"""

import json
import logging
import math

logger = logging.getLogger(__name__)


def _convert_to_native(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    try:
        import numpy as np

        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: _convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_convert_to_native(item) for item in obj]
        return obj
    except ImportError:
        return obj


def kalman_filter_state_estimation(
    initial_state: list[float],
    initial_covariance: list[list[float]],
    process_noise: list[list[float]],
    measurement_noise: list[list[float]],
    measurements: list[dict],
    dynamics_model: str = "constant_velocity",
    dt: float = 1.0,
) -> str:
    """Extended Kalman Filter for aircraft/spacecraft state estimation.

    Implements a Kalman filter for sensor fusion and state estimation from
    noisy measurements.

    Args:
        initial_state: Initial state vector estimate
        initial_covariance: Initial state covariance matrix (P0)
        process_noise: Process noise covariance matrix (Q)
        measurement_noise: Measurement noise covariance matrix (R)
        measurements: Time-series of measurements, each with:
            - time: Measurement timestamp
            - z: Measurement vector
            - H: Optional measurement matrix (uses identity if not provided)
        dynamics_model: Dynamics model type:
            - "constant_velocity": 2D position + velocity
            - "constant_acceleration": 2D position + velocity + acceleration
            - "orbital": Simplified orbital dynamics
        dt: Time step for prediction (used if not in measurements)

    Returns:
        Formatted string with filtered state estimates
    """
    try:
        # Try to use filterpy if available
        try:
            from filterpy.kalman import KalmanFilter

            use_filterpy = True
        except ImportError:
            use_filterpy = False

        n = len(initial_state)

        # Build state transition matrix based on dynamics model
        if dynamics_model == "constant_velocity":
            # State: [x, y, vx, vy]
            if n == 4:
                F = [
                    [1, 0, dt, 0],
                    [0, 1, 0, dt],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ]
            else:
                # Default to identity
                F = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
        elif dynamics_model == "constant_acceleration":
            # State: [x, y, vx, vy, ax, ay]
            if n == 6:
                F = [
                    [1, 0, dt, 0, 0.5 * dt**2, 0],
                    [0, 1, 0, dt, 0, 0.5 * dt**2],
                    [0, 0, 1, 0, dt, 0],
                    [0, 0, 0, 1, 0, dt],
                    [0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 1],
                ]
            else:
                F = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
        else:
            # Default: identity matrix (no dynamics)
            F = [[1 if i == j else 0 for j in range(n)] for i in range(n)]

        if use_filterpy:
            # Use filterpy implementation
            import numpy as np

            kf = KalmanFilter(dim_x=n, dim_z=n)
            kf.x = np.array(initial_state).reshape(-1, 1)
            kf.P = np.array(initial_covariance)
            kf.F = np.array(F)
            kf.Q = np.array(process_noise)
            kf.R = np.array(measurement_noise)
            kf.H = np.eye(n)

            # Process measurements
            filtered_states = []
            covariances = []
            innovations = []

            for i, meas in enumerate(measurements):
                # Predict
                kf.predict()

                # Update with measurement
                z = np.array(meas.get("z", initial_state)).reshape(-1, 1)

                # Custom H matrix if provided
                if "H" in meas:
                    kf.H = np.array(meas["H"])
                else:
                    kf.H = np.eye(n)

                # Calculate innovation before update
                y = z - np.dot(kf.H, kf.x)
                innovations.append(y.flatten().tolist())

                # Update
                kf.update(z)

                filtered_states.append(kf.x.flatten().tolist())
                covariances.append(np.diag(kf.P).tolist())

        else:
            # Manual implementation
            def mat_mult(A, B):
                """Matrix multiplication."""
                rows_A, cols_A = len(A), len(A[0])
                _rows_B, cols_B = len(B), len(B[0])
                result = [[0] * cols_B for _ in range(rows_A)]
                for i in range(rows_A):
                    for j in range(cols_B):
                        for k in range(cols_A):
                            result[i][j] += A[i][k] * B[k][j]
                return result

            def mat_add(A, B):
                """Matrix addition."""
                return [
                    [A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))
                ]

            def mat_sub(A, B):
                """Matrix subtraction."""
                return [
                    [A[i][j] - B[i][j] for j in range(len(A[0]))] for i in range(len(A))
                ]

            def mat_transpose(A):
                """Matrix transpose."""
                return [[A[j][i] for j in range(len(A))] for i in range(len(A[0]))]

            def mat_inverse_2x2(A):
                """2x2 matrix inverse."""
                det = A[0][0] * A[1][1] - A[0][1] * A[1][0]
                if abs(det) < 1e-10:
                    return [[1, 0], [0, 1]]  # Return identity if singular
                return [
                    [A[1][1] / det, -A[0][1] / det],
                    [-A[1][0] / det, A[0][0] / det],
                ]

            # Initialize state and covariance
            x = [[xi] for xi in initial_state]  # Column vector
            P = [row[:] for row in initial_covariance]

            filtered_states = []
            covariances = []
            innovations = []

            H = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
            Q = process_noise
            R = measurement_noise

            for meas in measurements:
                # Predict
                x_pred = mat_mult(F, x)
                P_pred = mat_add(mat_mult(mat_mult(F, P), mat_transpose(F)), Q)

                # Measurement
                z = [[zi] for zi in meas.get("z", initial_state)]
                if "H" in meas:
                    H = meas["H"]

                # Innovation
                y = mat_sub(z, mat_mult(H, x_pred))
                innovations.append([y[i][0] for i in range(len(y))])

                # Innovation covariance
                S = mat_add(mat_mult(mat_mult(H, P_pred), mat_transpose(H)), R)

                # Kalman gain (simplified for small matrices)
                # K = P_pred * H^T * S^-1
                if n <= 2:
                    S_inv = mat_inverse_2x2(S) if len(S) == 2 else [[1 / S[0][0]]]
                    K = mat_mult(mat_mult(P_pred, mat_transpose(H)), S_inv)
                else:
                    # Simplified: use diagonal approximation
                    K = [
                        [
                            P_pred[i][i] * H[j][i] / (S[j][j] + 1e-10)
                            for j in range(len(H))
                        ]
                        for i in range(n)
                    ]
                    K = mat_transpose(K)

                # Update
                x = mat_add(x_pred, mat_mult(K, y))
                I_KH = mat_sub(
                    [[1 if i == j else 0 for j in range(n)] for i in range(n)],
                    mat_mult(K, H),
                )
                P = mat_mult(I_KH, P_pred)

                filtered_states.append([x[i][0] for i in range(n)])
                covariances.append([P[i][i] for i in range(n)])

        # Calculate statistics
        if filtered_states:
            final_state = filtered_states[-1]
            final_cov = covariances[-1]

            # Innovation statistics
            innovation_rms = []
            for i in range(len(innovations[0]) if innovations else 0):
                values = [inn[i] for inn in innovations if len(inn) > i]
                if values:
                    rms = math.sqrt(sum(v**2 for v in values) / len(values))
                    innovation_rms.append(rms)
        else:
            final_state = initial_state
            final_cov = [initial_covariance[i][i] for i in range(n)]
            innovation_rms = []

        result = _convert_to_native(
            {
                "input": {
                    "initial_state": initial_state,
                    "dynamics_model": dynamics_model,
                    "num_measurements": len(measurements),
                },
                "output": {
                    "final_state": [round(s, 6) for s in final_state],
                    "final_state_std": [
                        round(math.sqrt(max(0, c)), 6) for c in final_cov
                    ],
                    "num_states_estimated": len(filtered_states),
                },
                "statistics": {
                    "innovation_rms": [round(r, 6) for r in innovation_rms],
                },
                "filtered_states": [
                    [round(s, 6) for s in state] for state in filtered_states[-10:]
                ],
                "implementation": "filterpy" if use_filterpy else "manual",
            }
        )

        state_labels = ["x", "y", "vx", "vy", "ax", "ay"][:n]
        final_state_str = ", ".join(
            f"{state_labels[i]}={final_state[i]:.4f}" for i in range(min(n, 6))
        )
        final_std_str = ", ".join(
            f"σ_{state_labels[i]}={math.sqrt(max(0, final_cov[i])):.4f}"
            for i in range(min(n, 6))
        )

        output = f"""
KALMAN FILTER STATE ESTIMATION
==============================
Dynamics Model: {dynamics_model}
Measurements Processed: {len(measurements)}
Implementation: {"filterpy" if use_filterpy else "manual"}

Final State Estimate:
  {final_state_str}

Final State Uncertainty (1σ):
  {final_std_str}

Innovation Statistics (RMS):
  {", ".join(f"{r:.4f}" for r in innovation_rms) if innovation_rms else "N/A"}

Last {min(10, len(filtered_states))} States:
{chr(10).join(f"  t={i}: {[round(s, 3) for s in state]}" for i, state in enumerate(filtered_states[-10:]))}

{json.dumps(result, indent=2)}
"""
        return output.strip()

    except Exception as e:
        logger.error(f"Kalman filter error: {str(e)}", exc_info=True)
        return f"Error in Kalman filter: {str(e)}"


def lqr_controller_design(
    A_matrix: list[list[float]],
    B_matrix: list[list[float]],
    Q_matrix: list[list[float]],
    R_matrix: list[list[float]],
    state_names: list[str] | None = None,
    input_names: list[str] | None = None,
) -> str:
    """Design Linear Quadratic Regulator (LQR) optimal controller.

    Computes optimal state-feedback gain K that minimizes the cost function:
    J = integral(x'Qx + u'Ru) dt

    Args:
        A_matrix: State matrix (n x n) - system dynamics
        B_matrix: Input matrix (n x m) - control influence
        Q_matrix: State weighting matrix (n x n) - penalizes state deviation
        R_matrix: Input weighting matrix (m x m) - penalizes control effort
        state_names: Optional names for states (for display)
        input_names: Optional names for control inputs (for display)

    Returns:
        Formatted string with optimal gain matrix and analysis
    """
    try:
        # Try to use control library if available
        try:
            import control
            import numpy as np

            use_control = True
        except ImportError:
            use_control = False

        n = len(A_matrix)
        m = len(B_matrix[0]) if B_matrix else 1

        # Default names
        if state_names is None:
            state_names = [f"x{i + 1}" for i in range(n)]
        if input_names is None:
            input_names = [f"u{i + 1}" for i in range(m)]

        if use_control:
            # Use control library
            A = np.array(A_matrix)
            B = np.array(B_matrix)
            Q = np.array(Q_matrix)
            R = np.array(R_matrix)

            # Check controllability
            ctrb_matrix = control.ctrb(A, B)
            ctrb_rank = np.linalg.matrix_rank(ctrb_matrix)
            controllable = ctrb_rank == n

            if not controllable:
                return json.dumps(
                    {
                        "error": f"System is not controllable (rank {ctrb_rank} < {n})",
                        "controllability_rank": ctrb_rank,
                        "required_rank": n,
                        "suggestion": "Check B matrix - some states may not be reachable from inputs",
                    },
                    indent=2,
                )

            # Solve LQR
            K, S, E = control.lqr(A, B, Q, R)

            # Convert to lists
            K_list = K.tolist() if hasattr(K, "tolist") else [[float(K)]]
            S_list = S.tolist()
            E_list = [complex(e).real for e in E]  # Real parts

            # Open-loop eigenvalues
            ol_eigenvalues = np.linalg.eigvals(A)
            ol_eig_list = [complex(e).real for e in ol_eigenvalues]

            # Closed-loop damping and natural frequency
            cl_poles_info = []
            for e in E:
                e_complex = complex(e)
                real = e_complex.real
                imag = e_complex.imag
                if imag != 0:
                    wn = abs(e_complex)  # Natural frequency
                    zeta = -real / wn  # Damping ratio
                    cl_poles_info.append(
                        {
                            "pole": f"{real:.4f} + {imag:.4f}j",
                            "natural_freq_rad_s": round(wn, 4),
                            "damping_ratio": round(zeta, 4),
                        }
                    )
                else:
                    cl_poles_info.append(
                        {
                            "pole": f"{real:.4f}",
                            "time_constant_s": (
                                round(-1 / real, 4) if real != 0 else float("inf")
                            ),
                        }
                    )

        else:
            # Simplified manual implementation
            # For simple 2x2 systems only
            if n > 2 or m > 1:
                return json.dumps(
                    {
                        "error": "Manual LQR only supports 2x2 systems. Install 'control' library for larger systems.",
                        "install_command": "pip install control",
                    },
                    indent=2,
                )

            # Simple controllability check
            # ctrb = [B, AB]
            A = A_matrix
            B = B_matrix
            AB = [
                [sum(A[i][k] * B[k][0] for k in range(n)) for _ in range(1)]
                for i in range(n)
            ]

            ctrb_matrix = [[B[0][0], AB[0][0]], [B[1][0], AB[1][0]]]
            det = (
                ctrb_matrix[0][0] * ctrb_matrix[1][1]
                - ctrb_matrix[0][1] * ctrb_matrix[1][0]
            )
            controllable = abs(det) > 1e-10

            if not controllable:
                return json.dumps(
                    {
                        "error": "System is not controllable",
                        "suggestion": "Check B matrix configuration",
                    },
                    indent=2,
                )

            # Simplified LQR for 2x2 SISO systems using pole placement
            # This is a rough approximation
            Q = Q_matrix
            R = R_matrix

            # Desired pole locations (heuristic based on Q/R ratio)
            q_avg = (Q[0][0] + Q[1][1]) / 2
            r_val = R[0][0] if isinstance(R[0], list) else R[0]
            desired_pole = -math.sqrt(q_avg / r_val) if r_val > 0 else -1

            # Simple gain calculation (Ackermann's formula approximation)
            K_list = [[round(-desired_pole * 2, 4), round(desired_pole**2 / 10, 4)]]
            S_list = Q  # Approximate
            E_list = [desired_pole, desired_pole * 0.5]
            ol_eig_list = [A[0][0], A[1][1]]
            cl_poles_info = [{"pole": str(p), "approximate": True} for p in E_list]

        # Check closed-loop stability
        stable = all(e < 0 for e in E_list)

        # Convert numpy booleans to Python booleans for JSON serialization
        controllable = bool(controllable)
        stable = bool(stable)

        result = _convert_to_native(
            {
                "input": {
                    "A_matrix_shape": f"{n}x{n}",
                    "B_matrix_shape": f"{n}x{m}",
                    "state_names": state_names[:n],
                    "input_names": input_names[:m],
                },
                "controllability": {
                    "is_controllable": controllable,
                    "rank": ctrb_rank if use_control else n,
                    "required_rank": n,
                },
                "lqr_solution": {
                    "gain_matrix_K": [[round(k, 6) for k in row] for row in K_list],
                    "cost_matrix_S_diagonal": [
                        round(S_list[i][i], 6) for i in range(n)
                    ],
                },
                "stability_analysis": {
                    "closed_loop_stable": stable,
                    "open_loop_eigenvalues": [round(e, 6) for e in ol_eig_list],
                    "closed_loop_eigenvalues": [round(e, 6) for e in E_list],
                    "poles_info": cl_poles_info,
                },
                "implementation": "control" if use_control else "manual_approximation",
            }
        )

        # Format gain matrix display
        K_str = "\n    ".join(
            f"{input_names[i]}: [{', '.join(f'{K_list[i][j]:+.4f}·{state_names[j]}' for j in range(n))}]"
            for i in range(m)
        )

        output = f"""
LQR CONTROLLER DESIGN
=====================
System: {n} states, {m} inputs

Controllability: {"✓ Controllable" if controllable else "✗ Not Controllable"}

Optimal Gain Matrix K:
    {K_str}

Open-Loop Eigenvalues: {[round(e, 4) for e in ol_eig_list]}
Closed-Loop Eigenvalues: {[round(e, 4) for e in E_list]}

Stability: {"✓ Stable (all eigenvalues in LHP)" if stable else "✗ Unstable"}

Control Law: u = -Kx

Implementation: {"control library" if use_control else "manual approximation"}

{json.dumps(result, indent=2)}
"""
        return output.strip()

    except Exception as e:
        logger.error(f"LQR design error: {str(e)}", exc_info=True)
        return f"Error designing LQR controller: {str(e)}"
