"""Guidance, Navigation, and Control (GNC) tools for the Aerospace MCP server.

Provides tools for:
- Kalman filter state estimation (prediction/update cycle for sensor fusion).
- LQR (Linear Quadratic Regulator) optimal controller design.

The Kalman filter implements the standard predict-update cycle:
    Predict:  x_pred = F * x,  P_pred = F * P * F^T + Q
    Update:   S = H * P_pred * H^T + R
              K = P_pred * H^T * S^{-1}
              x = x_pred + K * (z - H * x_pred)
              P = (I - K H) P_pred (I - K H)^T + K R K^T   (Joseph form)
For the "orbital" model F is the state transition matrix of the two-body
problem linearised about the propagated state (an extended Kalman filter).

The LQR controller minimizes the infinite-horizon cost function:
    J = integral(x^T Q x + u^T R u) dt
by solving the continuous algebraic Riccati equation (CARE):
    A^T S + S A - S B R^{-1} B^T S + Q = 0
to obtain the optimal gain K = R^{-1} B^T S.

References:
    - Simon, D., "Optimal State Estimation" (2006), Ch. 5 (discrete Kalman
      filter, Joseph-form covariance update) and Ch. 13 (extended KF)
    - Bar-Shalom, Y., Li, X.R. & Kirubarajan, T., "Estimation with
      Applications to Tracking and Navigation" (2001), Sec. 6.2-6.3
      (constant-velocity / constant-acceleration kinematic models)
    - Vallado, D.A., "Fundamentals of Astrodynamics and Applications"
      (4th ed., 2013), Sec. 10.7 (two-body state transition matrix)
    - Anderson, B.D.O. & Moore, J.B., "Optimal Control: Linear Quadratic
      Methods" (1990), Ch. 3 (CARE, stabilizability/detectability)

WARNING: This module is for educational and research purposes only.
Do NOT use for real flight planning, navigation, or aircraft operations.
"""

import json
import logging
import math

logger = logging.getLogger(__name__)

# Earth gravitational parameter (m^3/s^2), WGS-84 / EGM-96 value.
_MU_EARTH = 3.986004418e14

# Size limits so a single tool call cannot run unbounded.
_MAX_STATE_DIM = 50
_MAX_MEASUREMENTS = 10_000
# Two-body propagation: RK4 sub-step of at most 10 s, at most 2000
# sub-steps per measurement interval (longer gaps use longer sub-steps).
_ORBIT_MAX_SUBSTEP_S = 10.0
_ORBIT_MAX_SUBSTEPS = 2000
_MAX_LQR_DIM = 100


def _convert_to_native(obj):
    """Convert numpy types to native Python types for JSON serialization.

    Args:
        obj: Any Python object, potentially containing numpy types.

    Returns:
        The same object with all numpy scalars, arrays, and booleans
        replaced by their native Python equivalents.
    """
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
        elif isinstance(obj, list | tuple):
            return [_convert_to_native(item) for item in obj]
        return obj
    except ImportError:
        return obj


def _as_matrix(name, value, rows=None, cols=None):
    """Convert a nested list to a finite 2-D float array, checking its shape.

    Args:
        name: Parameter name used in error messages.
        value: Nested list (or scalar for a 1x1 matrix).
        rows: Required number of rows, or None.
        cols: Required number of columns, or None.

    Raises:
        ValueError: On ragged, non-numeric, non-finite or mis-shaped input.
    """
    import numpy as np

    try:
        arr = np.array(value, dtype=float)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a numeric matrix (list of rows)") from None
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2-D matrix, got {arr.ndim}-D")
    if (rows is not None and arr.shape[0] != rows) or (
        cols is not None and arr.shape[1] != cols
    ):
        if cols is None:
            want = f"have {rows} rows"
        elif rows is None:
            want = f"have {cols} columns"
        else:
            want = f"be {rows}x{cols}"
        raise ValueError(f"{name} must {want}, got {arr.shape[0]}x{arr.shape[1]}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    return arr


# ===========================================================================
# Kalman filter helpers
# ===========================================================================


def _kinematic_transition(dynamics_model: str, n: int, dt: float):
    """State transition matrix for the linear kinematic models.

    The state is laid out block-wise for ``k`` spatial axes:
        - constant_velocity:     [p_1..p_k, v_1..v_k]            (n = 2k)
        - constant_acceleration: [p_1..p_k, v_1..v_k, a_1..a_k]  (n = 3k)
        - static:                any n, F = I (random walk)
    with ``k`` in {1, 2, 3}.  For example the 2-D constant-velocity state is
    [x, y, vx, vy] and the 2-D constant-acceleration state is
    [x, y, vx, vy, ax, ay].

    Raises:
        ValueError: If the model is unknown or ``n`` does not fit it.
    """
    import numpy as np

    if dynamics_model == "static":
        return np.eye(n)
    if dynamics_model == "constant_velocity":
        if n not in (2, 4, 6):
            raise ValueError(
                "constant_velocity needs a state [positions, velocities] of "
                f"length 2, 4 or 6 (1-D, 2-D or 3-D); got {n}"
            )
        k = n // 2
        eye = np.eye(k)
        # p_new = p + v*dt ; v_new = v
        return np.block([[eye, dt * eye], [np.zeros((k, k)), eye]])
    if dynamics_model == "constant_acceleration":
        if n not in (3, 6, 9):
            raise ValueError(
                "constant_acceleration needs a state [positions, velocities, "
                f"accelerations] of length 3, 6 or 9; got {n}"
            )
        k = n // 3
        eye, zero = np.eye(k), np.zeros((k, k))
        # p_new = p + v*dt + a*dt^2/2 ; v_new = v + a*dt ; a_new = a
        return np.block(
            [
                [eye, dt * eye, 0.5 * dt**2 * eye],
                [zero, eye, dt * eye],
                [zero, zero, eye],
            ]
        )
    raise ValueError(
        f"Unknown dynamics_model {dynamics_model!r}; expected one of "
        "'constant_velocity', 'constant_acceleration', 'orbital', 'static'"
    )


def _two_body_propagate(x, dt: float):
    """Propagate a two-body state and its state transition matrix.

    Integrates the equations of motion r'' = -mu r / |r|^3 together with
    the variational equations Phi' = A(t) Phi, where::

        A = [[0, I], [G, 0]],   G = mu / |r|^5 * (3 r r^T - |r|^2 I)

    using fixed-step RK4 (sub-step <= 10 s).

    Args:
        x: State [x, y, z, vx, vy, vz] in m and m/s (Earth-centred inertial).
        dt: Propagation interval in seconds (>= 0).

    Returns:
        Tuple ``(x_new, Phi)`` with the propagated state and the 6x6 state
        transition matrix d x_new / d x.
    """
    import numpy as np

    if float(np.linalg.norm(x[:3])) < 1.0e5:
        raise ValueError(
            "orbital model needs an Earth-centred position in meters "
            "(|r| < 100 km is inside the Earth)"
        )

    def deriv(y):
        r = y[:3]
        v = y[3:6]
        phi = y[6:].reshape(6, 6)
        r2 = float(r @ r)
        rn = math.sqrt(r2)
        acc = -_MU_EARTH * r / rn**3
        G = _MU_EARTH / rn**5 * (3.0 * np.outer(r, r) - r2 * np.eye(3))
        A = np.zeros((6, 6))
        A[:3, 3:] = np.eye(3)
        A[3:, :3] = G
        return np.concatenate([v, acc, (A @ phi).ravel()])

    y = np.concatenate([np.asarray(x, dtype=float), np.eye(6).ravel()])
    if dt > 0:
        n_sub = min(_ORBIT_MAX_SUBSTEPS, max(1, math.ceil(dt / _ORBIT_MAX_SUBSTEP_S)))
        h = dt / n_sub
        for _ in range(n_sub):
            k1 = deriv(y)
            k2 = deriv(y + 0.5 * h * k1)
            k3 = deriv(y + 0.5 * h * k2)
            k4 = deriv(y + h * k3)
            y = y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return y[:6], y[6:].reshape(6, 6)


def kalman_filter_state_estimation(
    initial_state: list[float],
    initial_covariance: list[list[float]],
    process_noise: list[list[float]],
    measurement_noise: list[list[float]],
    measurements: list[dict],
    dynamics_model: str = "constant_velocity",
    dt: float = 1.0,
    initial_time: float | None = None,
) -> str:
    """Kalman filter (EKF for the orbital model) for aircraft/spacecraft state estimation.

    Implements a Kalman filter for sensor fusion and state estimation from
    noisy measurements, with the exact gain K = P H^T S^-1 and a Joseph-form
    covariance update.

    Args:
        initial_state: Initial state vector estimate (length n)
        initial_covariance: Initial state covariance matrix P0 (n x n)
        process_noise: Process noise covariance Q (n x n), added once per
            prediction step (it is not rescaled with the step length)
        measurement_noise: Measurement noise covariance R (m x m, where m is
            the length of each measurement vector z)
        measurements: Time-series of measurements, each with:
            - z: Measurement vector (length m)
            - time: Optional timestamp in seconds.  The prediction step
              before each measurement uses the elapsed time since the
              previous one; measurements without a time use ``dt``.
            - H: Measurement matrix (m x n).  Optional when m == n (identity
              is used); required otherwise, e.g. [[1,0,0,0],[0,1,0,0]] to
              measure position only with the 2-D constant-velocity model.
        dynamics_model: Dynamics model type:
            - "constant_velocity": state [positions, velocities], n = 2, 4
              or 6 (1-D, 2-D [x, y, vx, vy] or 3-D)
            - "constant_acceleration": state [positions, velocities,
              accelerations], n = 3, 6 or 9 (2-D: [x, y, vx, vy, ax, ay])
            - "orbital": Earth two-body dynamics, state [x, y, z, vx, vy, vz]
              in m and m/s (ECI); extended Kalman filter
            - "static": F = I (random-walk state), any n
        dt: Default time step in seconds, used for measurements without a
            ``time`` and for the first prediction when ``initial_time`` is
            not given
        initial_time: Optional time of the initial state.  When omitted the
            initial state is taken to be ``dt`` seconds before the first
            measurement.

    Returns:
        Formatted string with filtered state estimates, covariance diagonals,
        and innovation statistics.

    Raises:
        No exceptions are raised directly; errors are returned as formatted strings.
    """
    try:
        import numpy as np

        n = len(initial_state)
        if not 1 <= n <= _MAX_STATE_DIM:
            raise ValueError(f"initial_state must have 1..{_MAX_STATE_DIM} elements")
        if len(measurements) > _MAX_MEASUREMENTS:
            raise ValueError(f"at most {_MAX_MEASUREMENTS} measurements are supported")
        if not (math.isfinite(dt) and dt >= 0):
            raise ValueError("dt must be a finite, non-negative number")

        x = _as_matrix("initial_state", [initial_state], 1, n).ravel()
        P = _as_matrix("initial_covariance", initial_covariance, n, n)
        Q = _as_matrix("process_noise", process_noise, n, n)
        R_default = _as_matrix("measurement_noise", measurement_noise)
        if R_default.shape[0] != R_default.shape[1]:
            raise ValueError("measurement_noise must be square")
        if dynamics_model == "orbital":
            if n != 6:
                raise ValueError(
                    f"orbital model needs the state [x, y, z, vx, vy, vz]; got n={n}"
                )
        else:
            _kinematic_transition(dynamics_model, n, dt)  # validates model and n

        eye_n = np.eye(n)
        filtered_states = []
        covariances = []
        innovations = []
        nis_values = []
        times = []

        def _time_of(k, meas):
            """Timestamp of measurement k as a float, or None if absent."""
            t = meas.get("time")
            if t is None:
                return None
            try:
                t = float(t)
            except (TypeError, ValueError):
                raise ValueError(f"measurement {k} time must be a number") from None
            if not math.isfinite(t):
                raise ValueError(f"measurement {k} time must be finite")
            return t

        # Time of the current estimate.  Without initial_time the initial
        # state is taken to be dt before the first measurement (t = 0 when
        # the measurements carry no timestamps).
        t_prev = None if initial_time is None else float(initial_time)
        for k, meas in enumerate(measurements):
            if not isinstance(meas, dict) or "z" not in meas:
                raise ValueError(f"measurement {k} must be a dict with a 'z' vector")

            # -- Step length from timestamps (falls back to dt) --
            t_meas = _time_of(k, meas)
            if t_prev is None:
                t_prev = 0.0 if t_meas is None else t_meas - dt
            t_k = t_prev + dt if t_meas is None else t_meas
            step = t_k - t_prev
            if step < 0:
                raise ValueError(
                    f"measurement {k} time {t_k:g} is before the previous "
                    f"time {t_prev:g}; measurements must be in time order"
                )

            # -- PREDICT STEP (skipped when no time elapses) --
            # x_pred = F x (or the nonlinear two-body propagation)
            # P_pred = F P F^T + Q, with F rebuilt for this step length
            if step > 0:
                if dynamics_model == "orbital":
                    x, F = _two_body_propagate(x, step)
                else:
                    F = _kinematic_transition(dynamics_model, n, step)
                    x = F @ x
                P = F @ P @ F.T + Q

            # -- UPDATE STEP --
            z = _as_matrix(f"measurements[{k}].z", [meas["z"]]).ravel()
            m = z.size
            if "H" in meas and meas["H"] is not None:
                H = _as_matrix(f"measurements[{k}].H", meas["H"], m, n)
            elif m == n:
                H = eye_n
            else:
                raise ValueError(
                    f"measurement {k} has {m} components but the state has {n}; "
                    f"provide a measurement matrix 'H' of shape {m}x{n} "
                    "(e.g. [[1,0,0,0],[0,1,0,0]] for position-only)"
                )
            if "R" in meas and meas["R"] is not None:
                R = _as_matrix(f"measurements[{k}].R", meas["R"], m, m)
            elif R_default.shape == (m, m):
                R = R_default
            else:
                raise ValueError(
                    f"measurement_noise is {R_default.shape[0]}x{R_default.shape[1]} "
                    f"but measurement {k} has {m} components"
                )

            # Innovation (measurement residual): y = z - H x_pred
            y = z - H @ x
            # Innovation covariance: S = H P_pred H^T + R
            S = H @ P @ H.T + R
            # Exact gain K = P H^T S^-1, computed as solve(S, H P)^T
            # (S and P are symmetric).
            try:
                K = np.linalg.solve(S, H @ P).T
                nis = float(y @ np.linalg.solve(S, y))
            except np.linalg.LinAlgError:
                raise ValueError(
                    f"innovation covariance S is singular at measurement {k}; "
                    "check R and P"
                ) from None

            x = x + K @ y
            # Joseph form: valid for any gain and keeps P symmetric PSD
            I_KH = eye_n - K @ H
            P = I_KH @ P @ I_KH.T + K @ R @ K.T
            P = 0.5 * (P + P.T)

            innovations.append(y.tolist())
            nis_values.append(nis)
            filtered_states.append(x.tolist())
            covariances.append(np.diag(P).tolist())
            times.append(t_k)
            t_prev = t_k

        # Calculate statistics
        if filtered_states:
            final_state = filtered_states[-1]
            final_cov = covariances[-1]

            # Innovation statistics (per component, over the measurements
            # that have that component)
            innovation_rms = []
            for i in range(max(len(inn) for inn in innovations)):
                values = [inn[i] for inn in innovations if len(inn) > i]
                innovation_rms.append(
                    math.sqrt(sum(v**2 for v in values) / len(values))
                )
            mean_nis = sum(nis_values) / len(nis_values)
        else:
            final_state = x.tolist()
            final_cov = np.diag(P).tolist()
            innovation_rms = []
            mean_nis = None

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
                    "final_time": times[-1] if times else initial_time,
                    "num_states_estimated": len(filtered_states),
                },
                "statistics": {
                    "innovation_rms": [round(r, 6) for r in innovation_rms],
                    # Normalised innovation squared y^T S^-1 y; its mean is
                    # ~m (measurement dimension) for a consistent filter.
                    "mean_nis": round(mean_nis, 6) if mean_nis is not None else None,
                },
                "filtered_times": times[-10:],
                "filtered_states": [
                    [round(s, 6) for s in state] for state in filtered_states[-10:]
                ],
                "implementation": "numpy",
            }
        )

        state_labels = _state_labels(dynamics_model, n)
        final_state_str = ", ".join(
            f"{state_labels[i]}={final_state[i]:.4f}" for i in range(n)
        )
        final_std_str = ", ".join(
            f"σ_{state_labels[i]}={math.sqrt(max(0, final_cov[i])):.4f}"
            for i in range(n)
        )
        last_rows = "\n".join(
            f"  t={t:g}: {[round(s, 3) for s in state]}"
            for t, state in zip(times[-10:], filtered_states[-10:], strict=True)
        )

        output = f"""
KALMAN FILTER STATE ESTIMATION
==============================
Dynamics Model: {dynamics_model}
Measurements Processed: {len(measurements)}
Implementation: numpy (exact gain, Joseph-form covariance update)

Final State Estimate:
  {final_state_str}

Final State Uncertainty (1σ):
  {final_std_str}

Innovation Statistics (RMS):
  {", ".join(f"{r:.4f}" for r in innovation_rms) if innovation_rms else "N/A"}

Last {min(10, len(filtered_states))} States:
{last_rows}

{json.dumps(result, indent=2)}
"""
        return output.strip()

    except Exception as e:
        logger.error(f"Kalman filter error: {str(e)}", exc_info=True)
        return f"Error in Kalman filter: {str(e)}"


def _state_labels(dynamics_model: str, n: int) -> list[str]:
    """Human-readable state names for the text summary."""
    axes = ["x", "y", "z"]
    if dynamics_model == "orbital" and n == 6:
        return ["x", "y", "z", "vx", "vy", "vz"]
    if dynamics_model == "constant_velocity" and n in (2, 4, 6):
        k = n // 2
        return axes[:k] + [f"v{a}" for a in axes[:k]]
    if dynamics_model == "constant_acceleration" and n in (3, 6, 9):
        k = n // 3
        return axes[:k] + [f"v{a}" for a in axes[:k]] + [f"a{a}" for a in axes[:k]]
    return [f"s{i + 1}" for i in range(n)]


# ===========================================================================
# LQR
# ===========================================================================


def _eig_pairs(values) -> list[list[float]]:
    """Eigenvalues as [real, imag] pairs, sorted by real then imaginary part."""
    pairs = [[float(complex(e).real), float(complex(e).imag)] for e in values]
    pairs.sort(key=lambda p: (p[0], p[1]))
    # "+ 0.0" turns -0.0 into 0.0
    return [[round(re, 6) + 0.0, round(im, 6) + 0.0] for re, im in pairs]


def _fmt_complex(pair: list[float]) -> str:
    """Format a [real, imag] pair for the text summary."""
    re, im = pair
    if abs(im) < 1e-12:
        return f"{re:.4f}"
    return f"{re:.4f}{im:+.4f}j"


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
        Q_matrix: State weighting matrix (n x n, symmetric PSD) - penalizes
            state deviation
        R_matrix: Input weighting matrix (m x m, symmetric positive
            definite) - penalizes control effort
        state_names: Optional names for states (for display)
        input_names: Optional names for control inputs (for display)

    Returns:
        Formatted string with optimal gain matrix K, closed-loop eigenvalues,
        stability analysis, and controllability assessment.  Eigenvalues are
        reported as [real, imag] pairs.

    Raises:
        No exceptions are raised directly; errors are returned as formatted strings
        or JSON error objects (e.g., when the system is not stabilizable).
    """
    try:
        import numpy as np
        import scipy.linalg

        # Use python-control's lqr when available (it wraps the same CARE
        # solvers); otherwise solve the CARE with SciPy directly.
        try:
            import control

            use_control = True
        except ImportError:
            use_control = False

        A = _as_matrix("A_matrix", A_matrix)
        n = A.shape[0]
        if A.shape != (n, n):
            raise ValueError(f"A_matrix must be square, got {A.shape[0]}x{A.shape[1]}")
        if n > _MAX_LQR_DIM:
            raise ValueError(f"at most {_MAX_LQR_DIM} states are supported")
        B = _as_matrix("B_matrix", B_matrix, rows=n)
        m = B.shape[1]
        Q = _as_matrix("Q_matrix", Q_matrix, n, n)
        R = _as_matrix("R_matrix", R_matrix, m, m)

        if not np.allclose(Q, Q.T, atol=1e-9 * max(1.0, np.abs(Q).max())):
            raise ValueError("Q_matrix must be symmetric")
        if not np.allclose(R, R.T, atol=1e-9 * max(1.0, np.abs(R).max())):
            raise ValueError("R_matrix must be symmetric")
        if np.linalg.eigvalsh(Q).min() < -1e-9 * max(1.0, np.abs(Q).max()):
            raise ValueError("Q_matrix must be positive semi-definite")
        if np.linalg.eigvalsh(R).min() <= 0:
            raise ValueError("R_matrix must be positive definite")

        # Default names (pad short lists so every state/input has a label)
        state_names = list(state_names or [])
        state_names += [f"x{i + 1}" for i in range(len(state_names), n)]
        input_names = list(input_names or [])
        input_names += [f"u{i + 1}" for i in range(len(input_names), m)]

        # Controllability: rank of C = [B, AB, A^2B, ..., A^{n-1}B] must be n.
        blocks = [B]
        for _ in range(n - 1):
            blocks.append(A @ blocks[-1])
        ctrb_rank = int(np.linalg.matrix_rank(np.hstack(blocks)))
        controllable = ctrb_rank == n

        # Stabilizability (PBH test): every eigenvalue with Re >= 0 must be
        # controllable, i.e. rank([A - lambda I, B]) = n.
        ol_eigs = np.linalg.eigvals(A)
        uncontrollable_unstable = [
            lam
            for lam in ol_eigs
            if lam.real >= -1e-9
            and np.linalg.matrix_rank(np.hstack([A - lam * np.eye(n), B])) < n
        ]
        stabilizable = not uncontrollable_unstable

        if not stabilizable:
            return json.dumps(
                {
                    "error": (
                        f"System is not stabilizable: controllability rank "
                        f"{ctrb_rank} < {n} and the uncontrollable modes include "
                        "eigenvalues with non-negative real part, so no "
                        "state feedback can stabilize it"
                    ),
                    "is_controllable": False,
                    "controllability_rank": ctrb_rank,
                    "required_rank": n,
                    "uncontrollable_unstable_eigenvalues": _eig_pairs(
                        uncontrollable_unstable
                    ),
                    "suggestion": "Check B matrix - some states may not be reachable from inputs",
                },
                indent=2,
            )

        # Solve the CARE A^T S + S A - S B R^-1 B^T S + Q = 0, K = R^-1 B^T S.
        try:
            if use_control:
                K, S, _ = control.lqr(A, B, Q, R)
                K = np.atleast_2d(np.asarray(K, dtype=float))
                S = np.atleast_2d(np.asarray(S, dtype=float))
            else:
                S = scipy.linalg.solve_continuous_are(A, B, Q, R)
                K = np.linalg.solve(R, B.T @ S)
        except Exception as exc:
            return json.dumps(
                {
                    "error": f"Riccati equation has no stabilizing solution: {exc}",
                    "is_controllable": controllable,
                    "controllability_rank": ctrb_rank,
                    "required_rank": n,
                    "suggestion": (
                        "(Q, A) must be detectable: every unstable or "
                        "marginally stable mode must be penalised in Q"
                    ),
                },
                indent=2,
            )

        cl_eigs = np.linalg.eigvals(A - B @ K)
        # Closed-loop stability: all eigenvalues strictly in the LHP
        stable = bool(np.all(cl_eigs.real < 0))
        ol_pairs = _eig_pairs(ol_eigs)
        cl_pairs = _eig_pairs(cl_eigs)

        # Analyze closed-loop poles: for s = sigma +/- j*omega,
        #   wn = |s|,  zeta = -sigma / wn
        cl_poles_info = []
        for re, im in cl_pairs:
            if abs(im) > 1e-12:
                wn = math.hypot(re, im)  # Natural frequency
                cl_poles_info.append(
                    {
                        "pole": [re, im],
                        "natural_freq_rad_s": round(wn, 4),
                        "damping_ratio": round(-re / wn, 4),
                    }
                )
            else:
                cl_poles_info.append(
                    {
                        "pole": [re, im],
                        # Undefined for a pole at the origin
                        "time_constant_s": round(-1 / re, 4) if re != 0 else None,
                    }
                )

        warnings = []
        if not controllable:
            warnings.append(
                f"System is not controllable (rank {ctrb_rank} < {n}) but is "
                "stabilizable: the uncontrollable modes are already stable and "
                "keep their open-loop eigenvalues"
            )
        if not stable:
            warnings.append(
                "Closed loop is not asymptotically stable; (Q, A) is probably "
                "not detectable"
            )

        K_list = K.tolist()
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
                    "is_stabilizable": stabilizable,
                    "rank": ctrb_rank,
                    "required_rank": n,
                },
                "lqr_solution": {
                    "gain_matrix_K": [
                        [round(k, 6) + 0.0 for k in row] for row in K_list
                    ],
                    "cost_matrix_S_diagonal": [round(S[i][i], 6) for i in range(n)],
                },
                "stability_analysis": {
                    "closed_loop_stable": stable,
                    "open_loop_eigenvalues": ol_pairs,
                    "closed_loop_eigenvalues": cl_pairs,
                    "eigenvalue_format": "[real, imag]",
                    "poles_info": cl_poles_info,
                },
                "warnings": warnings,
                "implementation": "control" if use_control else "scipy",
            }
        )

        # Format gain matrix display
        K_str = "\n    ".join(
            f"{input_names[i]}: [{', '.join(f'{K_list[i][j]:+.4f}·{state_names[j]}' for j in range(n))}]"
            for i in range(m)
        )
        warn_str = "".join(f"\nWarning: {w}" for w in warnings)

        output = f"""
LQR CONTROLLER DESIGN
=====================
System: {n} states, {m} inputs

Controllability: {"✓ Controllable" if controllable else f"✗ Not Controllable (rank {ctrb_rank} < {n}), stabilizable"}

Optimal Gain Matrix K:
    {K_str}

Open-Loop Eigenvalues: [{", ".join(_fmt_complex(p) for p in ol_pairs)}]
Closed-Loop Eigenvalues: [{", ".join(_fmt_complex(p) for p in cl_pairs)}]

Stability: {"✓ Stable (all eigenvalues in LHP)" if stable else "✗ Unstable"}
{warn_str}
Control Law: u = -Kx

Implementation: {"python-control" if use_control else "SciPy solve_continuous_are"}

{json.dumps(result, indent=2)}
"""
        return output.strip()

    except Exception as e:
        logger.error(f"LQR design error: {str(e)}", exc_info=True)
        return f"Error designing LQR controller: {str(e)}"
