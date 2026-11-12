"""Lab 3 EKF: state = [x, y, theta, omega, v, acceleration].

Nonlinear functions propagate the mean; their Jacobians propagate uncertainty.
See docs/lab3_ekf.md for the derivation and sensor-frame assumptions.
"""

import numpy as np


class kalman_filter:
    def __init__(self, P, Q, R, x, dt):
        self.x = self._vector(x, 6, "state")
        # P: estimate uncertainty; Q: process noise per prediction;
        # R: measurement noise ordered [odom_v, odom_w, imu_ax, imu_ay].
        self.P = self._covariance(P, 6, "P")
        self.Q = self._covariance(Q, 6, "Q")
        self.R = self._covariance(R, 4, "R", positive_definite=True)
        self.dt = self._timestep(dt)

    @staticmethod
    def _vector(values, size, name):
        values = np.array(values, dtype=float, copy=True)
        if values.shape != (size,) or not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must be a finite vector of length {size}")
        return values

    @staticmethod
    def _covariance(values, size, name, positive_definite=False):
        values = np.array(values, dtype=float, copy=True)
        if values.shape != (size, size) or not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must be a finite {size} by {size} matrix")
        if not np.allclose(values, values.T, rtol=0, atol=1e-12):
            raise ValueError(f"{name} must be symmetric")
        minimum = np.linalg.eigvalsh(values).min()
        if minimum < -1e-12 or (positive_definite and minimum <= 0):
            raise ValueError(f"{name} must be positive {'definite' if positive_definite else 'semidefinite'}")
        return values

    @staticmethod
    def _timestep(dt):
        dt = float(dt)
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        return dt

    def _wrap_heading(self):
        self.x[2] = (self.x[2] + np.pi) % (2 * np.pi) - np.pi

    def predict(self, dt=None):
        """Predict mean and covariance over one sensor interval."""
        if dt is not None:
            self.dt = self._timestep(dt)
        # Linearize f at the previous corrected state, before moving it.
        self.A = self.jacobian_A()
        self.motion_model()
        self.P = self.A @ self.P @ self.A.T + self.Q
        self.P = (self.P + self.P.T) / 2
        # h is linearized at the predicted state, not the previous state.
        self.C = self.jacobian_H()

    def update(self, z):
        """Correct the prediction using synchronized odometry and IMU data."""
        z = self._vector(z, 4, "measurement")
        self.C = self.jacobian_H()
        self.innovation = z - self.measurement_model()
        S = self.C @ self.P @ self.C.T + self.R
        # Solve S K.T = C P rather than explicitly forming inverse(S).
        self.K = np.linalg.solve(S, self.C @ self.P).T
        self.x = self.x + self.K @ self.innovation
        self._wrap_heading()
        # Joseph form equals (I-KC)P in exact arithmetic and better preserves
        # covariance symmetry/positive semidefiniteness in floating point.
        residual_map = np.eye(6) - self.K @ self.C
        self.P = residual_map @ self.P @ residual_map.T + self.K @ self.R @ self.K.T
        self.P = (self.P + self.P.T) / 2

    def measurement_model(self):
        """Expected body-frame [v, omega, ax, ay]; ay = v * omega."""
        _, _, _, w, v, vdot = self.x
        return np.array([v, w, vdot, v * w])

    def motion_model(self):
        """Forward Euler integration; omega and forward acceleration persist."""
        x, y, th, w, v, vdot = self.x
        dt = self.dt
        self.x = np.array([
            x + v * np.cos(th) * dt,
            y + v * np.sin(th) * dt,
            th + w * dt,
            w,
            v + vdot * dt,
            vdot,
        ])
        self._wrap_heading()

    def jacobian_A(self):
        """df/dstate: differentiate v*cos(theta)*dt and v*sin(theta)*dt."""
        _, _, th, _, v, _ = self.x
        dt = self.dt
        return np.array([
            # x, y,           theta, omega,              v, acceleration
            [1, 0, -v * np.sin(th) * dt, 0, np.cos(th) * dt, 0],
            [0, 1,  v * np.cos(th) * dt, 0, np.sin(th) * dt, 0],
            [0, 0,                   1, dt,              0, 0],
            [0, 0,                   0,  1,              0, 0],
            [0, 0,                   0,  0,              1, dt],
            [0, 0,                   0,  0,              0, 1],
        ])

    def jacobian_H(self):
        """dh/dstate: d(v*omega)/domega = v; d(v*omega)/dv = omega."""
        _, _, _, w, v, _ = self.x
        return np.array([
            # x, y, theta, omega, v, acceleration
            [0, 0, 0, 0, 1, 0],  # v
            [0, 0, 0, 1, 0, 0],  # omega
            [0, 0, 0, 0, 0, 1],  # ax
            [0, 0, 0, v, w, 0],  # ay
        ])

    def get_states(self):
        """Return a copy so callers cannot accidentally alter the filter."""
        return self.x.copy()
