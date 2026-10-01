"""Tests for the decision layer -- the rules that decide whether to wake the house.

These are the tests that matter most. A bug in the Kalman filter makes boxes
wobble; a bug in here either wakes the family at 3 a.m. for a cat, or stays
silent while a child is in the water.
"""

from __future__ import annotations

import numpy as np
import pytest

from pool_guard.monitor import Mode, PoolMonitor, Policy, Severity
from pool_guard.tracker import Detection, Tracker
from pool_guard.zone import Zone

FRAME = (540, 960, 3)
# A trapezium roughly where a pool would sit in the lower half of the view.
ZONE = Zone(points=np.array([[0.30, 0.50], [0.75, 0.50], [0.80, 0.90], [0.25, 0.90]]))

INSIDE = np.array([400.0, 300.0, 460.0, 450.0])   # anchor (430, 450) -> inside
OUTSIDE = np.array([50.0, 300.0, 110.0, 450.0])   # anchor (80, 450)  -> outside


def make_track(box: np.ndarray, label: str = "person", frames: int = 20, every: int = 1):
    """Build one confirmed track by feeding a tracker a static detection.

    Args:
        every: detect on every Nth frame. `every=2` halves the detection rate,
            which is how a deliberately half-certain track is produced.
    """
    tracker = Tracker(dt=1 / 25, min_hits=4)
    for step in range(frames):
        detections = [Detection(box, 0.9, label)] if step % every == 0 else []
        tracker.update(detections)
    return tracker


def run(monitor: PoolMonitor, tracker: Tracker, frames: int, start: float = 100.0, box=INSIDE, label="person"):
    """Advance the monitor `frames` times at 25 fps and collect every event."""
    events = []
    for step in range(frames):
        tracks = tracker.update([Detection(box, 0.9, label)])
        events.extend(monitor.update(tracks, FRAME, now=start + step / 25))
    return events


def fresh(**kwargs) -> PoolMonitor:
    defaults = {"grace_seconds": 0.0}
    defaults.update(kwargs)
    return PoolMonitor(ZONE, Policy(**defaults))


def test_open_pool_stays_silent_with_someone_in_the_water() -> None:
    """Activity is allowed; this is the normal state and must never alarm."""
    monitor = fresh()
    assert monitor.mode is Mode.OPEN
    assert run(monitor, make_track(INSIDE), frames=40) == []


def test_closed_pool_alarms_on_a_person_inside() -> None:
    monitor = fresh()
    monitor.close_pool([], now=0.0)
    events = run(monitor, make_track(INSIDE), frames=20)

    assert len(events) == 1
    assert events[0].severity is Severity.ALARM
    assert events[0].label == "person"


def test_closed_pool_ignores_a_person_outside_the_zone() -> None:
    """The neighbour walking past their own garden is not an intrusion."""
    monitor = fresh()
    monitor.close_pool([], now=0.0)
    assert run(monitor, make_track(OUTSIDE), frames=40, box=OUTSIDE) == []


def test_dwell_requirement_suppresses_a_brief_flicker() -> None:
    monitor = fresh(dwell_seconds=0.5)
    monitor.close_pool([], now=0.0)
    assert run(monitor, make_track(INSIDE), frames=11) == []


def test_grace_period_lets_swimmers_climb_out() -> None:
    """Pressing 'close' while someone is still in the water must not alarm."""
    monitor = PoolMonitor(ZONE, Policy(grace_seconds=8.0))
    monitor.close_pool([], now=0.0)

    during = run(monitor, make_track(INSIDE), frames=100, start=0.0)
    assert during == []


def test_someone_still_inside_when_the_grace_period_ends_does_alarm() -> None:
    """The grace period is an exit delay, not an amnesty."""
    monitor = PoolMonitor(ZONE, Policy(grace_seconds=2.0))
    monitor.close_pool([], now=0.0)

    events = run(monitor, make_track(INSIDE), frames=100, start=0.0)
    assert len(events) == 1
    assert events[0].severity is Severity.ALARM
    assert events[0].timestamp >= 2.0


def test_an_object_present_at_closing_time_is_treated_as_scenery() -> None:
    """The forgotten inflatable: in the water, mislabelled, and must stay silent."""
    tracker = make_track(INSIDE, label="person")
    monitor = fresh()
    monitor.close_pool(tracker.confirmed_tracks(), now=0.0)

    assert run(monitor, tracker, frames=60) == []


def test_scenery_is_matched_by_position_so_a_reborn_track_stays_exempt() -> None:
    """A flickering object loses its track and comes back with a new id.

    Keying the whitelist on track ids let exactly this case through and fired a
    false alarm on a static float, so the rule is anchored to position instead.
    """
    tracker = make_track(INSIDE, label="person")
    monitor = fresh()
    monitor.close_pool(tracker.confirmed_tracks(), now=0.0)

    reborn = make_track(INSIDE, label="person")  # same place, brand new ids
    assert {t.track_id for t in reborn.confirmed_tracks()} == {1}
    assert run(monitor, reborn, frames=60) == []


def test_scenery_that_gets_up_and_walks_does_alarm() -> None:
    """Whitelisting a position must not whitelist whatever later moves through it."""
    tracker = Tracker(dt=1 / 25, min_hits=4)
    for _ in range(20):
        tracker.update([Detection(INSIDE, 0.9, "person")])
    monitor = fresh(static_travel_px=60.0)
    monitor.close_pool(tracker.confirmed_tracks(), now=0.0)

    events = []
    for step in range(40):  # same subject, now moving across the zone
        box = INSIDE + np.array([step * 6.0, 0.0, step * 6.0, 0.0])
        tracks = tracker.update([Detection(box, 0.9, "person")])
        events.extend(monitor.update(tracks, FRAME, now=step / 25))

    assert [e.severity for e in events] == [Severity.ALARM]


def test_cooldown_collapses_one_intrusion_into_one_event() -> None:
    monitor = fresh(cooldown_seconds=20.0)
    monitor.close_pool([], now=0.0)
    events = run(monitor, make_track(INSIDE), frames=300, start=0.0)

    assert len(events) == 1


def test_cooldown_expiry_allows_a_reminder() -> None:
    monitor = fresh(cooldown_seconds=2.0)
    monitor.close_pool([], now=0.0)
    events = run(monitor, make_track(INSIDE), frames=200, start=0.0)

    assert len(events) > 1
    assert events[1].timestamp - events[0].timestamp >= 2.0


@pytest.mark.parametrize("label,expected", [
    ("person", Severity.ALARM),
    ("dog", Severity.NOTICE),   # dogs drown too, but a chime is enough
    ("cat", None),              # would cry wolf every night
])
def test_severity_follows_the_class_policy(label: str, expected: Severity | None) -> None:
    monitor = fresh()
    monitor.close_pool([], now=0.0)
    events = run(monitor, make_track(INSIDE, label=label), frames=30, label=label)

    assert [e.severity for e in events] == ([expected] if expected else [])


def test_an_unstable_label_is_not_reported_at_all() -> None:
    """A shimmer detected on a third of the frames is not evidence of a person."""
    monitor = fresh(min_label_confidence=0.45)
    monitor.close_pool([], now=0.0)

    tracker = make_track(INSIDE, frames=60, every=3)
    track = tracker.confirmed_tracks()[0]
    assert track.detection_rate < 0.45

    events = []
    for step in range(60):
        tracks = tracker.update([Detection(INSIDE, 0.9, "person")] if step % 3 == 0 else [])
        events.extend(monitor.update(tracks, FRAME, now=step / 25))
    assert events == []


def test_a_half_seen_person_raises_a_notice_rather_than_a_siren() -> None:
    """The middle ground: tell the human, let them look at the snapshot."""
    monitor = fresh(min_label_confidence=0.40, strong_confidence=0.70)
    monitor.close_pool([], now=0.0)

    tracker = make_track(INSIDE, frames=40, every=2)
    events = []
    for step in range(60):
        tracks = tracker.update([Detection(INSIDE, 0.9, "person")] if step % 2 == 0 else [])
        events.extend(monitor.update(tracks, FRAME, now=step / 25))

    assert events, "a half-seen person in a closed pool must not be silently dropped"
    assert events[0].severity is Severity.NOTICE


def _calibration():
    """A plausible ground calibration for the test zone."""
    from pool_guard.calibration import Calibration

    world = np.array([[0.0, 0.0], [400.0, 0.0], [400.0, 400.0], [0.0, 400.0]])
    image = np.array([[100.0, 500.0], [500.0, 500.0], [450.0, 200.0], [150.0, 200.0]])
    return Calibration(image, world, frame_size=(FRAME[1], FRAME[0]))


def test_calibrated_events_carry_a_speed() -> None:
    monitor = PoolMonitor(ZONE, Policy(grace_seconds=0.0), calibration=_calibration())
    monitor.close_pool([], now=0.0)
    events = run(monitor, make_track(INSIDE), frames=20)

    assert events[0].speed_mps is not None
    assert events[0].speed_mps >= 0.0


def test_uncalibrated_events_leave_the_speed_empty() -> None:
    monitor = fresh()
    monitor.close_pool([], now=0.0)
    events = run(monitor, make_track(INSIDE), frames=20)

    assert events[0].speed_mps is None


def test_severity_never_depends_on_a_size_estimate() -> None:
    """Height was removed from the decision, not merely hidden.

    A ground-plane homography cannot measure a standing person, and the version
    that tried called a 160 cm teenager "adult" and a 60 cm dog "child-sized".
    A calibrated zone and an uncalibrated one must reach the same verdict.
    """
    with_calibration = PoolMonitor(ZONE, Policy(grace_seconds=0.0), calibration=_calibration())
    without = fresh()
    for monitor in (with_calibration, without):
        monitor.close_pool([], now=0.0)

    first = run(with_calibration, make_track(INSIDE), frames=20)
    second = run(without, make_track(INSIDE), frames=20)

    assert [e.severity for e in first] == [e.severity for e in second]
    assert [e.reason for e in first] == [e.reason for e in second]


def test_a_coasting_track_cannot_raise_anything() -> None:
    """A predicted box is not a sighting, however long it sits inside the zone."""
    monitor = fresh(dwell_seconds=0.5)
    monitor.close_pool([], now=0.0)

    tracker = make_track(INSIDE)          # confirmed, then never detected again
    events = []
    for step in range(40):                # 1.6 s of pure prediction
        tracks = tracker.update([], dt=1 / 25)
        events.extend(monitor.update(tracks, FRAME, now=step / 25))

    assert events == []


def test_dwell_is_frozen_by_an_occlusion_rather_than_reset() -> None:
    """An occluded person should resume, not start counting from zero again."""
    monitor = fresh(dwell_seconds=0.5)
    monitor.close_pool([], now=0.0)
    tracker = make_track(INSIDE)

    now = 0.0
    for _ in range(6):  # part-way through the dwell requirement
        tracker.update([Detection(INSIDE, 0.9, "person")], dt=1 / 25)
        monitor.update(tracker.confirmed_tracks(), FRAME, now=now)
        now += 1 / 25

    for _ in range(3):  # briefly hidden
        tracker.update([], dt=1 / 25)
        assert monitor.update(tracker.confirmed_tracks(), FRAME, now=now) == []
        now += 1 / 25

    events = []
    for _ in range(8):  # back in view
        tracker.update([Detection(INSIDE, 0.9, "person")], dt=1 / 25)
        events.extend(monitor.update(tracker.confirmed_tracks(), FRAME, now=now))
        now += 1 / 25

    assert [e.severity for e in events] == [Severity.ALARM]


def test_reopening_the_pool_clears_everything() -> None:
    monitor = fresh()
    tracker = make_track(INSIDE)
    monitor.close_pool(tracker.confirmed_tracks(), now=0.0)
    monitor.open_pool()

    assert monitor.mode is Mode.OPEN
    assert monitor._baseline_boxes == []
    assert run(monitor, tracker, frames=40) == []


def test_every_event_carries_a_human_readable_reason() -> None:
    """The log has to explain itself, or tuning the thresholds is guesswork."""
    monitor = fresh()
    monitor.close_pool([], now=0.0)
    events = run(monitor, make_track(INSIDE), frames=20)

    assert "person" in events[0].reason
    assert "%" in events[0].reason
