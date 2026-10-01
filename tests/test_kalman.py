"""Tests for the Kalman filter: the properties the rest of the system relies on."""

from __future__ import annotations

import numpy as np
import pytest

from pool_guard.kalman import BoxTrackerFilter, KalmanFilter


# The filter's defaults are tuned for video, so the tests run at a video frame
# rate. Driving it at dt=1.0 would mean a full second of unmodelled acceleration
# per step, which is a regime the defaults were never meant for -- and gives
# failures that look like filter bugs but are really unit-mismatched tests.
FPS = 25.0
DT = 1.0 / FPS


def moving_box(step: int, vx: float = 6.0, vy: float = 0.0) -> np.ndarray:
    """A 40x100 box travelling at a constant velocity in pixels per *frame*."""
    x, y = 100.0 + vx * step, 200.0 + vy * step
    return np.array([x, y, x + 40.0, y + 100.0])


def test_estimate_converges_on_a_clean_track() -> None:
    """With noiseless measurements the estimate should sit on the true box."""
    kf = BoxTrackerFilter(moving_box(0), dt=DT)
    for step in range(1, 40):
        kf.predict()
        kf.update(moving_box(step))

    assert kf.box == pytest.approx(moving_box(39), abs=1.5)


def test_velocity_is_recovered_although_it_is_never_measured() -> None:
    """Velocity is not part of the measurement; the filter has to infer it."""
    vx, vy = 6.0, -2.5
    kf = BoxTrackerFilter(moving_box(0, vx, vy), dt=DT)
    for step in range(1, 60):
        kf.predict()
        kf.update(moving_box(step, vx, vy))

    # The state carries pixels per second; the scene moves per frame.
    expected = np.array([vx, vy]) * FPS
    assert kf.velocity == pytest.approx(expected, rel=0.12)


def test_prediction_coasts_through_a_gap_in_detections() -> None:
    """This is what keeps a track alive behind a bush, and it must not drift far."""
    kf = BoxTrackerFilter(moving_box(0), dt=DT)
    for step in range(1, 25):
        kf.predict()
        kf.update(moving_box(step))

    for _ in range(8):  # eight frames with nothing to correct against
        kf.predict()

    assert kf.box == pytest.approx(moving_box(32), abs=12.0)


def test_smoothing_beats_the_raw_measurements() -> None:
    """The filtered centre must be closer to the truth than the noisy input."""
    rng = np.random.default_rng(0)
    kf = BoxTrackerFilter(moving_box(0), dt=DT)

    filtered_error, raw_error = [], []
    for step in range(1, 60):
        truth = moving_box(step)
        noisy = truth + rng.normal(0, 6.0, 4)
        kf.predict()
        estimate = kf.update(noisy)
        filtered_error.append(abs(estimate[0] - truth[0]))
        raw_error.append(abs(noisy[0] - truth[0]))

    assert np.mean(filtered_error) < np.mean(raw_error)


def test_uncertainty_grows_on_predict_and_shrinks_on_update() -> None:
    kf = BoxTrackerFilter(moving_box(0), dt=DT)
    for step in range(1, 10):  # let the initial transient settle
        kf.predict()
        kf.update(moving_box(step))

    settled = np.trace(kf.kf.P)
    kf.predict()
    after_predict = np.trace(kf.kf.P)
    kf.update(moving_box(10))
    after_update = np.trace(kf.kf.P)

    assert after_predict > settled
    assert after_update < after_predict


def test_covariance_stays_symmetric_over_a_long_run() -> None:
    """The Joseph-form update exists for this; the short form drifts asymmetric."""
    rng = np.random.default_rng(1)
    kf = BoxTrackerFilter(moving_box(0), dt=DT)
    for step in range(1, 3000):
        kf.predict()
        kf.update(moving_box(step) + rng.normal(0, 4.0, 4))

    P = kf.kf.P
    assert np.allclose(P, P.T, atol=1e-9)
    assert np.all(np.linalg.eigvalsh(P) > -1e-9)  # still positive semi-definite


def test_mahalanobis_separates_a_plausible_box_from_an_absurd_one() -> None:
    kf = BoxTrackerFilter(moving_box(0), dt=DT)
    for step in range(1, 20):
        kf.predict()
        kf.update(moving_box(step))
    kf.predict()

    near = kf.gate_distance(moving_box(20))
    far = kf.gate_distance(moving_box(20) + np.array([600, 400, 600, 400]))
    assert near < far / 10


def test_dimension_mismatch_is_rejected() -> None:
    kf = KalmanFilter(
        transition=np.eye(2), observation=np.eye(2), process_noise=np.eye(2),
        measurement_noise=np.eye(2), initial_state=np.zeros(2), initial_covariance=np.eye(2),
    )
    with pytest.raises(ValueError):
        kf.update(np.zeros(3))
