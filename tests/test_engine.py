"""Tests for frame skipping and real-time pacing.

These use a generated video file rather than the scripted scene, because what is
under test is the capture-and-skip behaviour itself.
"""

from __future__ import annotations

import time

import cv2
import numpy as np
import pytest

from pool_guard.engine import MonitorEngine
from pool_guard.monitor import Policy
from pool_guard.zone import Zone

FPS = 30.0
FRAMES = 180  # six seconds
ZONE = Zone(points=np.array([[0.3, 0.5], [0.75, 0.5], [0.8, 0.9], [0.25, 0.9]]))


@pytest.fixture
def video(tmp_path):
    """A short video whose every frame is numbered, so skipping is observable."""
    path = tmp_path / "counted.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (320, 240))
    for index in range(FRAMES):
        frame = np.full((240, 320, 3), 40, dtype=np.uint8)
        cv2.putText(frame, str(index), (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        writer.write(frame)
    writer.release()
    return str(path)


class StubDetector:
    """Stands in for YOLOX: no model file, no inference, controllable cost."""

    def __init__(self, cost: float = 0.0) -> None:
        self.cost = cost
        self.calls = 0

    def detect(self, frame):
        self.calls += 1
        if self.cost:
            time.sleep(self.cost)
        return []


@pytest.fixture
def engine(video, monkeypatch):
    """An engine with the detector stubbed out, so no model file is required."""
    from pool_guard import engine as engine_module

    stub = StubDetector()
    monkeypatch.setattr(engine_module, "YoloxDetector", lambda *a, **k: stub)
    built = MonitorEngine(
        source=video, zone=ZONE, policy=Policy(grace_seconds=0.0),
        width=0, stride=6, sound=False,
    )
    built.stub = stub  # type: ignore[attr-defined]
    yield built
    built.close()


def test_it_analyses_one_frame_in_every_stride(engine) -> None:
    """The whole point: most frames are thrown away without being decoded."""
    steps = 0
    while engine.step() is not None:
        steps += 1

    expected = FRAMES // engine.stride
    assert steps == pytest.approx(expected, abs=2)
    assert engine.stub.calls == steps


def test_the_position_advances_by_the_stride(engine) -> None:
    first = engine.step()
    second = engine.step()

    assert second.position - first.position == engine.stride


def test_a_stride_of_one_analyses_everything(video, monkeypatch) -> None:
    from pool_guard import engine as engine_module

    stub = StubDetector()
    monkeypatch.setattr(engine_module, "YoloxDetector", lambda *a, **k: stub)
    engine = MonitorEngine(
        source=video, zone=ZONE, policy=Policy(grace_seconds=0.0),
        width=0, stride=1, sound=False,
    )
    steps = 0
    while engine.step() is not None:
        steps += 1
    engine.close()

    assert steps == pytest.approx(FRAMES, abs=2)


def test_the_end_of_the_video_is_reported_once(engine) -> None:
    while engine.step() is not None:
        pass

    assert engine.finished is True
    assert engine.step() is None


def test_pacing_leaves_time_when_detection_is_quick(engine) -> None:
    """A cheap detector should be told to wait, so playback is not fast-forward."""
    engine.step()
    budget_ms = engine.stride / engine.source_fps * 1000

    assert 1 < engine.next_delay_ms() <= budget_ms


def test_pacing_gives_up_when_detection_overruns(video, monkeypatch) -> None:
    """Slower than real time is the one thing skipping cannot fix, so it says so
    by asking to be called again immediately rather than sleeping anyway."""
    from pool_guard import engine as engine_module

    # 0.4 s per frame against a 0.2 s budget: twice too slow.
    stub = StubDetector(cost=0.4)
    monkeypatch.setattr(engine_module, "YoloxDetector", lambda *a, **k: stub)
    engine = MonitorEngine(
        source=video, zone=ZONE, policy=Policy(grace_seconds=0.0),
        width=0, stride=6, sound=False,
    )
    engine.step()
    delay = engine.next_delay_ms()
    engine.close()

    assert delay == 1


def test_pausing_stops_the_video_advancing(engine) -> None:
    engine.step()
    position = engine.position

    engine.set_paused(True)
    assert engine.step() is None
    assert engine.position == position

    engine.set_paused(False)
    assert engine.step() is not None
    assert engine.position > position


def test_seeking_jumps_and_clears_the_tracks(engine) -> None:
    """Old tracks describe a moment no longer on screen after a jump."""
    engine.step()
    engine.tracker.tracks.append(object())  # a stand-in for a live track

    engine.seek(120)
    assert engine.tracker.tracks == []

    step = engine.step()
    assert step.position >= 120


def test_skipping_backwards_is_clamped_at_the_start(engine) -> None:
    engine.step()
    engine.skip_seconds(-999)
    assert engine.position == 0


def test_skipping_forward_is_clamped_at_the_end(engine) -> None:
    engine.skip_seconds(999)
    assert engine.position <= FRAMES


def test_a_narrower_width_downscales_the_frame(video, monkeypatch) -> None:
    from pool_guard import engine as engine_module

    monkeypatch.setattr(engine_module, "YoloxDetector", lambda *a, **k: StubDetector())
    engine = MonitorEngine(
        source=video, zone=ZONE, policy=Policy(grace_seconds=0.0),
        width=160, stride=6, sound=False,
    )
    step = engine.step()
    engine.close()

    assert step.frame.shape[1] == 160


def test_opening_a_missing_file_fails_clearly(monkeypatch) -> None:
    from pool_guard import engine as engine_module

    monkeypatch.setattr(engine_module, "YoloxDetector", lambda *a, **k: StubDetector())
    with pytest.raises(RuntimeError, match="could not open"):
        MonitorEngine(source="no_such_video.mp4", zone=ZONE)
