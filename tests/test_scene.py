"""End-to-end tests over the scripted scene.

The unit tests check each rule in isolation; these check that the rules compose
into the right behaviour on a scene with five subjects at once, including the
three cases the system was built for. They pin the outcome, so a well-meant
tweak to a threshold that reintroduces a false alarm fails CI instead of being
discovered by a siren at midnight.
"""

from __future__ import annotations

import numpy as np

from pool_guard.monitor import Policy, Severity
from pool_guard.synthetic import ScriptedDetector, default_scene, replay
from pool_guard.tracker import Tracker, iou_matrix
from pool_guard.zone import Zone

CLOSE_AT = 80
POLICY = Policy(grace_seconds=5.0)


def collect(scene=None, close_at: int = CLOSE_AT, policy: Policy | None = None):
    """Run the scene and return every (frame_index, event) pair it produced."""
    scene = scene or default_scene()
    out = []
    for index, _tracks, events, _monitor in replay(scene, close_at, policy or POLICY):
        out.extend((index, event) for event in events)
    return out


def test_the_scene_produces_exactly_one_alarm_and_one_notice() -> None:
    events = collect()
    severities = [event.severity for _, event in events]

    assert severities.count(Severity.ALARM) == 1
    assert severities.count(Severity.NOTICE) == 1


def test_nothing_is_reported_while_the_pool_is_open() -> None:
    """An adult wades in the zone for the first 80 frames. That is the point of OPEN."""
    assert [index for index, _ in collect() if index <= CLOSE_AT] == []


def test_the_swimmer_leaving_during_the_grace_period_is_not_reported() -> None:
    """They are inside the zone when the button is pressed, and walk out by frame 127."""
    grace_frames = int(POLICY.grace_seconds * default_scene().fps)
    early = [index for index, _ in collect() if index <= CLOSE_AT + grace_frames]
    assert early == []


def test_the_alarm_is_the_person_who_walks_in_while_closed() -> None:
    alarms = [(index, event) for index, event in collect() if event.severity is Severity.ALARM]
    index, event = alarms[0]

    assert event.label == "person"
    assert 300 < index < 380, "should fire shortly after the intruder crosses the boundary"
    assert event.label_confidence > 0.85


def test_the_alarm_survives_the_occlusion_at_the_boundary() -> None:
    """The intruder is invisible for frames 332-340, right as they cross the line.

    Without a filter to coast the track through the gap the track would break,
    the dwell counter would restart, and the alarm would be late or absent.
    """
    scene = default_scene()
    intruder = next(obj for obj in scene.objects if obj.name == "intruder")
    assert intruder.dropout == (332, 340)

    ids_at = {}
    for index, tracks, _events, _monitor in replay(scene, CLOSE_AT, POLICY):
        if index in (325, 345):
            inside = [t for t in tracks if t.box[0] > 400 and t.label == "person"]
            ids_at[index] = {t.track_id for t in inside}

    assert ids_at[325], "the intruder should already be tracked before the occlusion"
    assert ids_at[325] & ids_at[345], "the same track id must come out the other side"


def test_the_tracker_is_more_accurate_than_the_raw_detections() -> None:
    """On a full scene with five subjects, filtering must beat the raw boxes.

    `test_kalman.py` proves this for one clean track in isolation. Here the
    tracker also has to associate the right detection with the right track while
    other subjects move nearby -- if association goes wrong, the filter's output
    is worse than the input it was given, however good the maths is.
    """
    scene = default_scene()
    intruder = next(obj for obj in scene.objects if obj.name == "intruder")
    detector = ScriptedDetector(scene)
    tracker = Tracker(dt=1.0 / scene.fps)

    def nearest_centre(items, target: float) -> float | None:
        near = [i for i in items if abs((i.box[0] + i.box[2]) / 2 - target) < 60]
        if not near:
            return None
        best = min(near, key=lambda i: abs((i.box[0] + i.box[2]) / 2 - target))
        return float((best.box[0] + best.box[2]) / 2)

    raw_errors, filtered_errors = [], []
    for index in range(scene.n_frames):
        detections = detector.detect(index=index)
        tracks = tracker.update(detections)
        true_box = intruder.box_at(index)
        if true_box is None:
            continue
        true_x = float((true_box[0] + true_box[2]) / 2)

        detected = nearest_centre(detections, true_x)
        estimated = nearest_centre(tracks, true_x)
        if detected is None or estimated is None:
            continue
        raw_errors.append(abs(detected - true_x))
        filtered_errors.append(abs(estimated - true_x))

    assert len(filtered_errors) > 100, "the intruder should be tracked for most of its life"
    assert np.mean(filtered_errors) < np.mean(raw_errors)


def test_the_intruder_is_one_track_from_start_to_finish() -> None:
    """Fragmentation would reset the dwell counter and delay or lose the alarm."""
    scene = default_scene()
    intruder = next(obj for obj in scene.objects if obj.name == "intruder")

    ids = set()
    for index, tracks, _events, _monitor in replay(scene, CLOSE_AT, POLICY):
        true_box = intruder.box_at(index)
        if true_box is None or not tracks:
            continue
        # Pair by box overlap, not by horizontal distance. The intruder walks
        # straight past the float, so a centre-distance test picks up the
        # float's track as well and reports a fragmentation that never happened.
        overlaps = iou_matrix(np.asarray([true_box]), np.asarray([t.box for t in tracks]))[0]
        best = int(overlaps.argmax())
        if overlaps[best] > 0.4:
            ids.add(tracks[best].track_id)

    assert len(ids) == 1, f"the intruder was split across tracks {sorted(ids)}"


# Frames during which the float is the only thing inside the zone: the swimmer
# has left and the half-seen figure has not arrived. Anything reported here can
# only be the float.
FLOAT_ALONE = range(140, 226)


def test_the_forgotten_float_never_raises_an_alarm() -> None:
    """It sits in the water the whole clip and is mislabelled `person` at times.

    The test window matters more than it looks. Identifying the float by how close
    an event is to it -- by centre distance or even by box overlap -- does not
    work, because the intruder walks straight over the float's position on the way
    across the zone, and gets mistaken for it. Picking a stretch of the clip where
    nothing else is inside the zone removes the ambiguity completely.
    """
    scene = default_scene()
    float_object = next(obj for obj in scene.objects if obj.name == "forgotten-float")
    assert float_object.label == "person"
    assert float_object.detect_prob < 0.3

    during = [(index, event) for index, event in collect(scene) if index in FLOAT_ALONE]
    assert during == [], f"the float raised {[e.reason for _, e in during]}"


def test_the_floats_own_track_stays_below_the_reporting_threshold() -> None:
    """The direct statement of the same thing, without going via events.

    The float is picked out by being *static* -- the one property the intruder
    walking over the same spot does not share. Detected on about a fifth of the
    frames, its track should score far too low on `detection_rate` to be reported,
    and it should be dismissed for its own unreliability rather than by luck.
    """
    scene = default_scene()
    float_object = next(obj for obj in scene.objects if obj.name == "forgotten-float")

    worst = 0.0
    for index, tracks, _events, _monitor in replay(scene, CLOSE_AT, POLICY):
        true_box = float_object.box_at(index)
        if true_box is None:
            continue
        for track in tracks:
            overlap = iou_matrix(np.asarray([track.box]), np.asarray([true_box]))[0, 0]
            if overlap > 0.4 and track.travelled < 40:
                worst = max(worst, track.label_confidence)

    assert 0 < worst < POLICY.min_label_confidence, f"float reached {worst:.2f}"


def test_the_passing_dog_outside_the_zone_is_ignored() -> None:
    assert [event for _, event in collect() if event.label == "dog"] == []


def test_a_dog_inside_the_zone_would_be_noticed() -> None:
    """Proof the previous test passes because of the zone, not because dogs are muted."""
    scene = default_scene()
    dog = next(obj for obj in scene.objects if obj.name == "passing-dog")
    dog.path = [(170, 400, 440), (300, 480, 445)]  # same dog, now in the water

    dog_events = [event for _, event in collect(scene) if event.label == "dog"]
    assert [event.severity for event in dog_events] == [Severity.NOTICE]


def test_an_empty_scene_is_completely_silent() -> None:
    scene = default_scene()
    scene.objects = []
    assert collect(scene) == []


def test_the_run_is_deterministic() -> None:
    """Same seed, same scene, same events -- otherwise CI would flake."""
    first = [(index, event.severity, round(event.label_confidence, 6)) for index, event in collect()]
    second = [(index, event.severity, round(event.label_confidence, 6)) for index, event in collect()]
    assert first == second


def test_the_marker_quadrilateral_becomes_a_usable_zone() -> None:
    """The four bottles on the ground are what defines the protected area."""
    scene = default_scene()
    zone = Zone(points=scene.zone_points)
    shape = (scene.height, scene.width, 3)

    assert zone.contains_pixel(500, 400, shape) is True    # in the water
    assert zone.contains_pixel(500, 150, shape) is False   # up the lawn
    assert np.all((scene.zone_points >= 0) & (scene.zone_points <= 1))


def test_the_scripted_detector_reports_nothing_outside_an_object_lifetime() -> None:
    scene = default_scene()
    detector = ScriptedDetector(scene)
    assert detector.detect(index=scene.n_frames + 50) == []
