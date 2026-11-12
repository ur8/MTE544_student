"""Independent math and synthetic-sensor checks; no robot or ROS graph."""

import numpy as np
import pytest

from kalman_filter import kalman_filter


def make_filter(state=None, dt=0.1, q=0.005, r=0.05, P=None):
    return kalman_filter(
        np.eye(6) * 0.1 if P is None else P,
        np.eye(6) * q, np.eye(4) * r,
        np.zeros(6) if state is None else state, dt,
    )


def numerical_jacobian(function, state):
    epsilon = 1e-6
    columns = []
    for axis in np.eye(6):
        columns.append((function(state + epsilon * axis) - function(state - epsilon * axis)) / (2 * epsilon))
    return np.column_stack(columns)


@pytest.mark.parametrize("state,dt", [
    ([1, -2, 0.7, 0.3, 0.4, 0.2], 0.1),
    ([-1, 2, -1.2, -0.5, -0.2, -0.1], 0.037),
    ([0, 0, 0, 0, 0, 0], 0.2),
])
def test_jacobians_match_numerical_derivatives(state, dt):
    state = np.array(state, dtype=float)
    kf = make_filter(state, dt=dt)

    def motion(x):
        model = make_filter(x, dt=dt)
        model.motion_model()
        return model.get_states()

    def measurement(x):
        return make_filter(x).measurement_model()

    np.testing.assert_allclose(kf.jacobian_A(), numerical_jacobian(motion, state), atol=2e-9)
    np.testing.assert_allclose(kf.jacobian_H(), numerical_jacobian(measurement, state), atol=2e-9)


def test_update_uses_predicted_state_and_matches_gaussian_conditioning():
    # Nonzero acceleration changes v, and hence the lateral-acceleration
    # derivative. A stale measurement Jacobian produces a different result.
    rng = np.random.default_rng(123)
    root = rng.normal(size=(6, 6))
    kf = make_filter([0, 0, 0.4, 0.5, 0.3, 0.4], P=root @ root.T * 0.01)
    previous_v = kf.x[4]
    kf.predict(dt=0.3)
    assert kf.C[3, 3] != previous_v
    assert kf.C[3, 3] == kf.x[4]
    mean, covariance = kf.get_states(), kf.P.copy()
    H = numerical_jacobian(lambda x: make_filter(x).measurement_model(), mean)
    z = np.array([0.4, 0.45, 0.3, 0.18])
    innovation_cov = H @ covariance @ H.T + kf.R
    cross_cov = covariance @ H.T
    expected_mean = mean + cross_cov @ np.linalg.solve(innovation_cov, z - kf.measurement_model())
    expected_cov = covariance - cross_cov @ np.linalg.solve(innovation_cov, cross_cov.T)
    kf.update(z)
    np.testing.assert_allclose(kf.x, expected_mean, atol=1e-10)
    np.testing.assert_allclose(kf.P, expected_cov, atol=1e-10)


def test_repeated_noisy_turn_reduces_velocity_noise_and_keeps_valid_covariance():
    rng = np.random.default_rng(7)
    dt, v, w = 0.02, 0.2, 0.3
    kf = make_filter([0, 0, 0, w, v, 0], dt=dt, q=1e-5)
    sigma = np.array([0.04, 0.03, 0.08, 0.04])
    kf.R = np.diag(sigma ** 2)
    raw, estimated = [], []
    for _ in range(1000):
        z = np.array([v, w, 0, v * w]) + rng.normal(size=4) * sigma
        kf.predict()
        kf.update(z)
        assert np.all(np.isfinite(kf.x))
        np.testing.assert_allclose(kf.P, kf.P.T, atol=1e-12)
        assert np.linalg.eigvalsh(kf.P).min() >= -1e-12
        raw.append(z[:2])
        estimated.append(kf.x[[4, 3]])
    raw_rmse = np.sqrt(np.mean((np.array(raw)[100:] - [v, w]) ** 2, axis=0))
    estimated_rmse = np.sqrt(np.mean((np.array(estimated)[100:] - [v, w]) ** 2, axis=0))
    assert np.all(estimated_rmse < raw_rmse)
    # Independent continuous-time unicycle reference; allow integration and
    # accumulated noisy velocity/heading errors over the 20-second trace.
    time = 1000 * dt
    reference = [v / w * np.sin(w * time), v / w * (1 - np.cos(w * time))]
    assert np.linalg.norm(kf.x[:2] - reference) < 0.15


def test_covariance_scales_change_measurement_trust():
    def corrected_velocity(q, r):
        kf = make_filter(q=q, r=r)
        kf.predict()
        kf.update([1, 0, 0, 0])
        return kf.x[4]

    assert corrected_velocity(1, 0.5) > corrected_velocity(0.1, 0.5)
    assert corrected_velocity(0.5, 0.1) > corrected_velocity(0.5, 1)


def test_heading_wrap_and_invalid_inputs():
    kf = make_filter([0, 0, np.pi - 0.01, 1, 0, 0])
    kf.predict()
    assert -np.pi <= kf.x[2] < np.pi
    for dt in [0, -0.1, np.nan]:
        with pytest.raises(ValueError):
            kf.predict(dt=dt)
    with pytest.raises(ValueError):
        kf.update([1, 2, np.nan, 0])
    with pytest.raises(ValueError):
        make_filter(r=0)
    with pytest.raises(ValueError):
        make_filter(q=-1)
