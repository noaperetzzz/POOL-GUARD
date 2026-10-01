"""The decision layer: when does something inside the zone deserve an alarm?

Everything upstream is measurement -- boxes, tracks, labels. This module holds
the policy, and it is where the project's real engineering lives, because the
hard part of a safety alarm is not detecting people. It is *not* going off for
a wind-blown shadow or the inflatable ring that has been
sitting in the water since Tuesday.

Two ideas do most of that work:

**Graded certainty instead of a yes/no.** A weakly supported detection raises a
NOTICE (a soft chime and a saved still, "come and look"); only a well supported
one raises an ALARM (the siren). Since perfect discrimination is not achievable
by any vision system, the design does not pretend otherwise -- it keeps a human
in the loop for the ambiguous middle instead of guessing.

**Asymmetric failure costs.** A false alarm is an annoyance; a miss can be a
drowned child. So every threshold here is tuned towards sensitivity, and the
noise that decision lets in is filtered by *agreement over time* -- dwell
frames, label voting, minimum track age -- rather than by raising the bar on any
single frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from .calibration import Calibration
from .tracker import Track, iou_matrix
from .zone import Zone


class Mode(str, Enum):
    """Whether the pool is open for use or closed and being watched."""

    OPEN = "open"      # activity allowed, nothing is reported
    CLOSED = "closed"  # armed: anything entering the zone is reported


class Severity(str, Enum):
    IGNORE = "ignore"
    NOTICE = "notice"  # soft chime + saved still, human decides
    ALARM = "alarm"    # full siren


@dataclass
class Event:
    """One thing worth telling the user about.

    `speed_mps` is filled in only when the zone has been calibrated with real
    measurements; without a calibration it stays None and nothing downstream
    depends on it.

    `video_seconds` is how far into the recording this happened -- the number you
    need in order to go and look at it. The engine fills it in for a video file;
    on a live camera there is no such position and it stays None.
    """

    timestamp: float
    track_id: int
    label: str
    severity: Severity
    label_confidence: float
    box: tuple[float, float, float, float]
    reason: str
    speed_mps: float | None = None
    video_seconds: float | None = None


@dataclass
class Policy:
    """Tunables for the decision layer.

    Args:
        actions: COCO label -> highest severity it may raise. A dog in the pool
            can drown too, so it is worth a notice; a cat gets ignored because it
            can get itself out and would otherwise cry wolf every night.
        strong_confidence: share of frames that must agree on the label before a
            NOTICE is upgraded to an ALARM.
        min_label_confidence: below this the track is too ambiguous to report.
        dwell_seconds: how long the subject must stay inside the zone without
            interruption. The single most effective anti-false-alarm control: a
            flickering detection on the water's edge never survives half a second
            of it.

            In seconds rather than frames on purpose. Counted in frames, this
            threshold silently grew six times longer when the engine started
            analysing every 6th frame instead of every frame -- the kind of bug
            that does not announce itself, because the system goes on working and
            merely does everything at the wrong speed.
        cooldown_seconds: minimum gap between two events for the same track, so
            one intrusion does not produce two hundred log entries.
        static_travel_px: how far a track present at arming time must move before
            it counts as an intruder rather than as furniture.
        baseline_iou: overlap with a recorded scenery box above which a track is
            treated as that same piece of scenery.
        grace_seconds: quiet period after closing the pool, so the people who
            were still in the water when the button was pressed can climb out
            without setting the siren off. Every alarm panel has this exit
            delay for the same reason. Note the deliberate limit: someone who
            is *still* inside when the grace period ends does raise the alarm,
            because at that point they are a person in a closed pool.
    """

    actions: dict[str, Severity] = field(
        default_factory=lambda: {
            "person": Severity.ALARM,
            "dog": Severity.NOTICE,
            "cat": Severity.IGNORE,
        }
    )
    strong_confidence: float = 0.70
    min_label_confidence: float = 0.45
    dwell_seconds: float = 0.5
    cooldown_seconds: float = 20.0
    static_travel_px: float = 60.0
    grace_seconds: float = 8.0
    baseline_iou: float = 0.40


class PoolMonitor:
    """Applies `Policy` to the live tracks and emits events.

    The monitor is pure decision logic: it does no file or sound I/O and takes
    the current time as an argument, so its behaviour is fully reproducible in
    tests.
    """

    def __init__(
        self,
        zone: Zone,
        policy: Policy | None = None,
        calibration: Calibration | None = None,
    ) -> None:
        self.zone = zone
        self.policy = policy or Policy()
        self.calibration = calibration
        self.mode = Mode.OPEN
        # Seconds accumulated inside the zone, per track
        self._dwell: dict[int, float] = {}
        self._last_event_at: dict[int, float] = {}
        self._last_update_at: float | None = None
        self._armed_at: float = 0.0
        # Boxes that were already in view when the zone was closed: the forgotten
        # float, the sun lounger, the pool robot. Recorded as *positions* rather
        # than track ids on purpose -- a flickering object loses its track and is
        # reborn under a fresh id, which would let it slip out of an id-based
        # whitelist and set the siren off. Its location does not change.
        self._baseline_boxes: list[np.ndarray] = []

    def close_pool(self, tracks: list[Track], now: float = 0.0) -> None:
        """Arm the monitor. Whatever is already in view is recorded as scenery.

        This is the clean answer to "what about the inflatable someone left in
        the water?". At the moment the user declares the pool closed, the scene
        is assumed to be acceptable. Only new arrivals -- or baseline objects
        that start moving on their own -- can raise an event afterwards.

        Args:
            tracks: the tracks currently in view, recorded as known scenery.
            now: current time on the same clock passed to `update`; starts the
                grace period.
        """
        self.mode = Mode.CLOSED
        self._baseline_boxes = [np.asarray(t.box, dtype=float).copy() for t in tracks]
        self._armed_at = float(now)
        self._dwell.clear()

    def open_pool(self) -> None:
        """Disarm the monitor and forget all accumulated state."""
        self.mode = Mode.OPEN
        self._dwell.clear()
        self._baseline_boxes.clear()

    def _is_scenery(self, track: Track) -> bool:
        """Is this track a known object that was already here, and still hasn't moved?"""
        if not self._baseline_boxes or track.travelled >= self.policy.static_travel_px:
            return False
        overlaps = iou_matrix(np.asarray([track.box]), np.asarray(self._baseline_boxes))
        return bool(overlaps.max() >= self.policy.baseline_iou)

    def update(self, tracks: list[Track], frame_shape: tuple[int, ...], now: float) -> list[Event]:
        """Process one step's confirmed tracks. Returns any new events.

        `now` is the clock for everything here: the elapsed time between calls is
        what dwell accumulates, so the analysis rate does not affect any threshold.
        """
        elapsed = 0.0 if self._last_update_at is None else max(0.0, now - self._last_update_at)
        self._last_update_at = now

        if self.mode is Mode.OPEN:
            self._dwell.clear()
            return []

        if now - self._armed_at < self.policy.grace_seconds:
            # Exit delay. Dwell counters are cleared rather than frozen, so a
            # swimmer climbing out during the grace period does not arrive at
            # the far side of it with a nearly-full counter and alarm instantly.
            self._dwell.clear()
            return []

        live_ids = {t.track_id for t in tracks}
        for stale_id in set(self._dwell) - live_ids:
            del self._dwell[stale_id]

        events: list[Event] = []
        for track in tracks:
            severity, reason = self._evaluate(track, frame_shape, now, elapsed)
            if severity is Severity.IGNORE:
                continue
            self._last_event_at[track.track_id] = now
            events.append(
                Event(
                    timestamp=now,
                    track_id=track.track_id,
                    label=track.label,
                    severity=severity,
                    label_confidence=track.label_confidence,
                    box=tuple(float(v) for v in track.box),
                    reason=reason,
                    speed_mps=self._speed(track, frame_shape),
                )
            )
        return events

    def _speed(self, track: Track, frame_shape: tuple[int, ...]) -> float | None:
        """Ground speed in m/s, when the zone has been calibrated."""
        if self.calibration is None:
            return None
        calibration = self.calibration.scaled_to(frame_shape)
        return calibration.speed_mps(track.anchor(), track.filter.velocity)

    def _evaluate(
        self, track: Track, frame_shape: tuple[int, ...], now: float, elapsed: float = 0.0
    ) -> tuple[Severity, str]:
        """Decide one track's severity this step, and say why."""
        ceiling = self.policy.actions.get(track.label, Severity.IGNORE)
        if ceiling is Severity.IGNORE:
            return Severity.IGNORE, f"class '{track.label}' is not watched"

        if track.label_confidence < self.policy.min_label_confidence:
            return Severity.IGNORE, "label too unstable across frames"

        # An alarm needs a current sighting, not a prediction. A track coasting
        # through a gap keeps its identity, and its dwell timer is frozen rather
        # than reset so an occluded person picks up where they left off -- but
        # while nothing is actually being detected it cannot raise anything.
        # Without this, a track that had lost its subject could accumulate the
        # full dwell time on predicted positions alone and fire on nobody.
        if not track.detected_now:
            return Severity.IGNORE, "no detection this step; coasting on prediction"

        anchor_x, anchor_y = track.anchor()
        if not self.zone.contains_pixel(anchor_x, anchor_y, frame_shape):
            self._dwell.pop(track.track_id, None)
            return Severity.IGNORE, "outside the zone"

        # Scenery that has not gone anywhere stays scenery.
        if self._is_scenery(track):
            return Severity.IGNORE, "present when the pool was closed and has not moved"

        dwell = self._dwell.get(track.track_id, 0.0) + elapsed
        self._dwell[track.track_id] = dwell
        if dwell < self.policy.dwell_seconds:
            return (
                Severity.IGNORE,
                f"inside for {dwell:.1f}s of {self.policy.dwell_seconds:.1f}s",
            )

        last = self._last_event_at.get(track.track_id)
        if last is not None and now - last < self.policy.cooldown_seconds:
            return Severity.IGNORE, "still within the cooldown window"

        # Confident about what it is -> siren. Unsure -> ask the human to look.
        if ceiling is Severity.ALARM and track.label_confidence < self.policy.strong_confidence:
            return (
                Severity.NOTICE,
                f"probably a {track.label} ({track.label_confidence:.0%} of frames agree)",
            )
        return (
            ceiling,
            f"{track.label} inside the zone ({track.label_confidence:.0%} of frames agree)",
        )
