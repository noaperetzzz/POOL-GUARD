"""Tests for track identity, occlusion survival, and the label-voting logic."""

from __future__ import annotations

import numpy as np

from pool_guard.tracker import Detection, Tracker, iou_matrix


def box_at(step: int, vx: float = 8.0) -> np.ndarray:
    x = 100.0 + vx * step
    return np.array([x, 200.0, x + 50.0, 340.0])


def test_iou_matrix_basics() -> None:
    a = np.array([[0.0, 0.0, 10.0, 10.0]])
    identical = iou_matrix(a, a)
    disjoint = iou_matrix(a, np.array([[50.0, 50.0, 60.0, 60.0]]))
    half = iou_matrix(a, np.array([[5.0, 0.0, 15.0, 10.0]]))

    assert identical[0, 0] == 1.0
    assert disjoint[0, 0] == 0.0
    assert half[0, 0] == 1 / 3  # 50 overlap / 150 union


def test_empty_inputs_do_not_blow_up() -> None:
    assert iou_matrix(np.empty((0, 4)), np.array([[0.0, 0.0, 1.0, 1.0]])).shape == (0, 1)
    assert Tracker().update([]) == []


def test_identity_is_kept_across_frames() -> None:
    tracker = Tracker(dt=1 / 25, min_hits=3)
    ids = set()
    for step in range(30):
        tracks = tracker.update([Detection(box_at(step), 0.9, "person")])
        ids.update(t.track_id for t in tracks)

    assert ids == {1}


def test_two_subjects_keep_separate_tracks() -> None:
    tracker = Tracker(dt=1 / 25, min_hits=3)
    for step in range(20):
        tracker.update([
            Detection(box_at(step), 0.9, "person"),
            Detection(box_at(step) + np.array([0, 400, 0, 400]), 0.9, "dog"),
        ])

    tracks = tracker.confirmed_tracks()
    assert len(tracks) == 2
    assert {t.label for t in tracks} == {"person", "dog"}


def test_track_survives_a_short_occlusion_and_keeps_its_id() -> None:
    """The whole reason for the Kalman filter: no id change, no restarted dwell."""
    tracker = Tracker(dt=1 / 25, min_hits=3, max_coast_seconds=0.5)
    for step in range(15):
        tracker.update([Detection(box_at(step), 0.9, "person")])
    first_id = tracker.confirmed_tracks()[0].track_id

    for _ in range(8):  # occluded
        tracker.update([])
    assert len(tracker.confirmed_tracks()) == 1

    tracks = tracker.update([Detection(box_at(23), 0.9, "person")])
    assert [t.track_id for t in tracks] == [first_id]


def test_track_is_dropped_after_a_long_absence() -> None:
    tracker = Tracker(dt=1 / 25, min_hits=3, max_coast_seconds=0.25)
    for step in range(12):
        tracker.update([Detection(box_at(step), 0.9, "person")])

    for _ in range(9):
        tracker.update([])
    assert tracker.confirmed_tracks() == []


def test_a_single_frame_fluke_never_becomes_confirmed() -> None:
    tracker = Tracker(dt=1 / 25, min_hits=4)
    assert tracker.update([Detection(box_at(0), 0.9, "person")]) == []


def test_majority_vote_outweighs_a_wrong_label_on_a_few_frames() -> None:
    tracker = Tracker(dt=1 / 25, min_hits=3)
    for step in range(30):
        label = "dog" if step in (4, 11, 19) else "person"
        tracker.update([Detection(box_at(step), 0.8, label)])

    track = tracker.confirmed_tracks()[0]
    assert track.label == "person"
    assert 0.85 < track.label_agreement < 1.0


def test_flicker_and_occlusion_score_differently() -> None:
    """Both lose the same number of frames; only flicker should be punished."""
    flickery = Tracker(dt=1 / 25, min_hits=3, max_coast_seconds=0.5, occlusion_seconds=0.25)
    for step in range(60):
        detections = [Detection(box_at(step), 0.8, "person")] if step % 3 == 0 else []
        flickery.update(detections)

    occluded = Tracker(dt=1 / 25, min_hits=3, max_coast_seconds=0.5, occlusion_seconds=0.25)
    for step in range(60):
        detections = [] if 30 <= step <= 38 else [Detection(box_at(step), 0.8, "person")]
        occluded.update(detections)

    flickery_track = flickery.confirmed_tracks()[0]
    occluded_track = occluded.confirmed_tracks()[0]

    assert flickery_track.detection_rate < 0.5
    assert occluded_track.detection_rate > 0.95
    assert occluded_track.occluded_steps >= 8


def test_a_lost_track_is_dropped_before_it_can_drift_into_a_phantom() -> None:
    """The bug this was written for: a box around nobody, drifting across the lawn.

    Every threshold used to be counted in frames. When the engine began analysing
    every 6th frame instead of every frame, "survives 12 frames without a
    detection" quietly became 2.4 seconds instead of 0.4, so a track that had lost
    its subject went on being predicted and slid away from where anyone was.
    Expressed in seconds, the analysis rate cannot change how long that lasts.
    """
    for stride, fps in ((1, 25.0), (6, 25.0 / 6)):
        dt = 1.0 / fps
        tracker = Tracker(dt=dt, min_hits=3, max_coast_seconds=0.6)
        for step in range(12):
            tracker.update([Detection(box_at(step), 0.9, "person")], dt=dt)
        assert tracker.confirmed_tracks(), f"stride {stride}: should be tracking"

        coasted = 0.0
        while tracker.confirmed_tracks():
            tracker.update([], dt=dt)
            coasted += dt
            assert coasted < 2.0, f"stride {stride}: phantom survived {coasted:.1f}s"

        assert coasted <= 0.6 + dt, f"stride {stride}: coasted {coasted:.2f}s"


def test_a_subject_that_outruns_its_own_box_keeps_one_track() -> None:
    """At five analysed frames a second a walking person has no frame-to-frame
    overlap at all, so matching on overlap alone split them into a string of
    short tracks -- and every abandoned one became a phantom box."""
    dt = 6 / 25.0
    tracker = Tracker(dt=dt, min_hits=2)
    for step in range(12):
        # 70 px per step against a 50 px wide box: no overlap whatsoever.
        tracker.update([Detection(box_at(step, vx=70.0), 0.9, "person")], dt=dt)

    assert len(tracker.tracks) == 1
    assert tracker.tracks[0].track_id == 1


def test_travelled_separates_a_walker_from_scenery() -> None:
    moving = Tracker(dt=1 / 25, min_hits=3)
    for step in range(30):
        moving.update([Detection(box_at(step), 0.9, "person")])

    still = Tracker(dt=1 / 25, min_hits=3)
    rng = np.random.default_rng(3)
    for _ in range(30):
        still.update([Detection(box_at(0) + rng.normal(0, 2.0, 4), 0.9, "person")])

    assert moving.confirmed_tracks()[0].travelled > 200
    assert still.confirmed_tracks()[0].travelled < 15


def test_anchor_is_the_bottom_centre_of_the_box() -> None:
    tracker = Tracker(dt=1 / 25, min_hits=1)
    tracker.update([Detection(np.array([100.0, 200.0, 200.0, 400.0]), 0.9, "person")])
    x, y = tracker.confirmed_tracks()[0].anchor()

    assert x == 150.0
    assert y == 400.0


def test_a_distant_detection_does_not_hijack_an_existing_track() -> None:
    """Low IoU must be vetoed, or the assignment solver will pair anything."""
    tracker = Tracker(dt=1 / 25, min_hits=2, iou_threshold=0.25)
    for step in range(6):
        tracker.update([Detection(box_at(step), 0.9, "person")])
    original = tracker.confirmed_tracks()[0].track_id

    tracker.update([Detection(np.array([800.0, 600.0, 850.0, 740.0]), 0.9, "person")])
    ids = {t.track_id for t in tracker.tracks}

    assert original in ids
    assert len(ids) == 2  # a new track, not a teleporting old one
