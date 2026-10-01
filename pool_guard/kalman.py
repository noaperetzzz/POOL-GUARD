"""A linear Kalman filter, written from scratch on top of NumPy.

The filter is deliberately generic: it knows nothing about video, boxes or
pools.  `BoxTrackerFilter` below wires it up for the concrete job of smoothing
a noisy bounding box over time.

Why bother with a Kalman filter at all when the detector already reports a box
on every frame?  Three reasons, all of which the monitor depends on:

1. The detector's box jitters by several pixels per frame even for a subject
   standing perfectly still.  The filter fuses the noisy measurement with a
   motion model and returns a stable estimate.
2. Detectors miss frames.  A person half-hidden behind a chair may vanish for
   4-5 frames.  `predict()` keeps producing a plausible position while there is
   no measurement, so the track survives instead of being torn in two.
3. The state carries velocity, which is never measured directly.  It falls out
   of the filter for free and tells us whether an object is actually moving.

State vector (6x1):   [cx, cy, w, h, vx, vy]
Measurement (4x1):    [cx, cy, w, h]

Box width and height are modelled as constant-plus-noise rather than getting
their own velocity terms.  A person walking towards the camera does grow, but
slowly compared to the measurement noise, and the extra states cost more in
tuning trouble than they buy back in accuracy.
"""

from __future__ import annotations

import numpy as np

STATE_DIM = 6
MEAS_DIM = 4


class KalmanFilter:
    """Textbook linear Kalman filter: x_k = F x_{k-1} + noise, z_k = H x_k + noise."""

    def __init__(
        self,
        transition: np.ndarray,
        observation: np.ndarray,
        process_noise: np.ndarray,
        measurement_noise: np.ndarray,
        initial_state: np.ndarray,
        initial_covariance: np.ndarray,
    ) -> None:
        self.F = np.asarray(transition, dtype=float)
        self.H = np.asarray(observation, dtype=float)
        self.Q = np.asarray(process_noise, dtype=float)
        self.R = np.asarray(measurement_noise, dtype=float)
        self.x = np.asarray(initial_state, dtype=float).reshape(-1)
        self.P = np.asarray(initial_covariance, dtype=float)

        n = self.F.shape[0]
        if self.F.shape != (n, n):
            raise ValueError("transition matrix must be square")
        if self.H.shape[1] != n:
            raise ValueError("observation matrix has the wrong state dimension")
        if self.x.shape != (n,):
            raise ValueError("initial state has the wrong dimension")

    def predict(
        self, transition: np.ndarray | None = None, process_noise: np.ndarray | None = None
    ) -> np.ndarray:
        """Advance the state one time step. Uncertainty always grows here.

        `transition` and `process_noise` override the stored matrices for this
        step, which is what makes an uneven time step possible: when detection
        runs as fast as the hardware manages rather than once per frame, the gap
        between updates varies and both matrices depend on it.
        """
        F = self.F if transition is None else np.asarray(transition, dtype=float)
        Q = self.Q if process_noise is None else np.asarray(process_noise, dtype=float)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        return self.x.copy()

    def update(self, measurement: np.ndarray) -> np.ndarray:
        """Fold one measurement into the state. Uncertainty always shrinks here."""
        z = np.asarray(measurement, dtype=float).reshape(-1)
        if z.shape != (self.H.shape[0],):
            raise ValueError("measurement has the wrong dimension")

        residual = z - self.H @ self.x               # what the model got wrong
        S = self.H @ self.P @ self.H.T + self.R      # covariance of that error
        # Solving is both faster and better conditioned than inverting S.
        gain = np.linalg.solve(S.T, (self.P @ self.H.T).T).T

        self.x = self.x + gain @ residual
        identity = np.eye(self.P.shape[0])
        # Joseph form: algebraically equal to (I - KH)P but stays symmetric and
        # positive-definite over thousands of updates, where the short form can
        # drift into a non-symmetric P and quietly diverge.
        a = identity - gain @ self.H
        self.P = a @ self.P @ a.T + gain @ self.R @ gain.T
        return self.x.copy()

    def mahalanobis(self, measurement: np.ndarray) -> float:
        """Distance from a measurement to the prediction, in units of sigma.

        Used as a sanity gate: a candidate detection that sits 30 sigma away
        from where the track should be is almost certainly a different object.
        """
        z = np.asarray(measurement, dtype=float).reshape(-1)
        residual = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        return float(np.sqrt(residual @ np.linalg.solve(S, residual)))


def _box_to_measurement(box: np.ndarray) -> np.ndarray:
    """(x1, y1, x2, y2) -> (cx, cy, w, h)."""
    x1, y1, x2, y2 = (float(v) for v in box)
    return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1])


def _state_to_box(state: np.ndarray) -> np.ndarray:
    """(cx, cy, w, h, ...) -> (x1, y1, x2, y2), width and height clamped positive."""
    cx, cy, w, h = state[:4]
    w = max(float(w), 1.0)
    h = max(float(h), 1.0)
    return np.array([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0])


class BoxTrackerFilter:
    """Kalman filter specialised for a bounding box moving at roughly constant velocity.

    Every quantity here is in pixels and seconds. An earlier version of this
    class mixed the two -- it documented the process noise in pixels per *frame*
    squared while multiplying it by `dt` in seconds -- so the numbers you tuned
    were not the numbers the filter used, and `accel_noise=8` meant nothing
    physical at all. Mixed units in a noise model are a slow-burning bug: the
    filter still works, so nothing draws your attention to it, and every later
    attempt to tune it fights the scale error instead of the problem.

    Args:
        box: first observation, ``(x1, y1, x2, y2)`` in pixels.
        dt: seconds between frames. Feeding the real value keeps the velocity
            estimate in pixels/second, comparable across frame rates.
        accel_noise: std-dev of the unmodelled acceleration, in px/s^2. This is
            the "how abruptly can this thing change direction" knob: raise it for
            a running dog, lower it for smoother, more heavily filtered tracks.
        size_noise: std-dev of how fast the box may legitimately grow or shrink,
            in px/s -- a subject walking towards the camera.
        measurement_noise: std-dev of the detector's box jitter, in pixels.
    """

    def __init__(
        self,
        box: np.ndarray,
        dt: float = 1.0 / 25.0,
        accel_noise: float = 900.0,
        size_noise: float = 60.0,
        measurement_noise: float = 6.0,
    ) -> None:
        self.dt = float(dt)
        self.accel_noise = float(accel_noise)
        self.size_noise = float(size_noise)

        F, Q = self._matrices(self.dt)

        H = np.zeros((MEAS_DIM, STATE_DIM))
        H[0, 0] = H[1, 1] = H[2, 2] = H[3, 3] = 1.0

        r = float(measurement_noise) ** 2
        R = np.diag([r, r, r * 4.0, r * 4.0])  # box edges are noisier than its centre

        z = _box_to_measurement(box)
        x0 = np.array([z[0], z[1], z[2], z[3], 0.0, 0.0])
        # Position starts as trusted as the measurement; velocity is unknown, so
        # its variance starts large and the first few frames pull it into shape.
        P0 = np.diag([r, r, r, r, 1000.0, 1000.0])

        self.kf = KalmanFilter(F, H, Q, R, x0, P0)

    def _matrices(self, dt: float) -> tuple[np.ndarray, np.ndarray]:
        """Transition and process-noise matrices for a step of `dt` seconds."""
        F = np.eye(STATE_DIM)
        F[0, 4] = dt
        F[1, 5] = dt

        # Standard discrete-time constant-velocity process noise for a random
        # acceleration of variance a: position gains a*dt^4/4, velocity a*dt^2,
        # and the two are correlated by a*dt^3/2 -- the same acceleration moved
        # both, so the off-diagonal terms are not optional decoration.
        a = self.accel_noise ** 2
        Q = np.zeros((STATE_DIM, STATE_DIM))
        Q[0, 0] = Q[1, 1] = a * dt ** 4 / 4.0
        Q[4, 4] = Q[5, 5] = a * dt ** 2
        Q[0, 4] = Q[4, 0] = Q[1, 5] = Q[5, 1] = a * dt ** 3 / 2.0
        Q[2, 2] = Q[3, 3] = (self.size_noise * dt) ** 2
        return F, Q

    def predict(self, dt: float | None = None) -> np.ndarray:
        """Advance the state and return the predicted box.

        Pass the real elapsed seconds when the interval varies. Detection runs as
        fast as the machine manages rather than once per frame, so assuming a
        fixed step would make the filter expect motion that did not have time to
        happen -- or miss motion that did.
        """
        if dt is None or abs(dt - self.dt) < 1e-9:
            return _state_to_box(self.kf.predict())
        F, Q = self._matrices(float(dt))
        return _state_to_box(self.kf.predict(transition=F, process_noise=Q))

    def update(self, box: np.ndarray) -> np.ndarray:
        """Correct with a detected box and return the filtered box."""
        return _state_to_box(self.kf.update(_box_to_measurement(box)))

    @property
    def box(self) -> np.ndarray:
        """Current best estimate of the box."""
        return _state_to_box(self.kf.x)

    @property
    def velocity(self) -> np.ndarray:
        """Estimated (vx, vy) in pixels per second."""
        return self.kf.x[4:6].copy()

    @property
    def speed(self) -> float:
        """Estimated speed in pixels per second."""
        return float(np.hypot(*self.kf.x[4:6]))

    def gate_distance(self, box: np.ndarray) -> float:
        """Mahalanobis distance from a candidate box to this track's prediction."""
        return self.kf.mahalanobis(_box_to_measurement(box))
