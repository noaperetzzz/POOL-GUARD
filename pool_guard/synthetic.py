"""A scripted synthetic scene, so the whole pipeline can be demonstrated and
tested without a camera, without footage, and without the 35 MB model file.

This is not a substitute for the real detector -- a drawn stick figure is not
something YOLOX would recognise. It substitutes for the *footage*: the scene
comes with ground truth, so a `ScriptedDetector` can hand the tracker exactly
the kind of imperfect input a real detector produces (jittery boxes, missed
frames, an occasional wrong label) and the decision layer can be exercised
end to end, deterministically, in CI.

The built-in scene is written around the three cases that actually matter:

1. An adult in the zone while the pool is **open** -> must stay silent.
2. A float left in the water when the pool is **closed**, which the detector
   mislabels ``person`` in scattered frames -> must stay silent.
3. A person entering while **closed**, briefly occluded halfway -> must alarm,
   and the track must survive the occlusion rather than splitting in two.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

import cv2
import numpy as np

from .monitor import Event, PoolMonitor, Policy
from .tracker import Detection, Track, Tracker
from .zone import Zone

GRASS = (76, 132, 84)  # BGR


@dataclass
class SceneObject:
    """One thing in the scene, moving along a linearly interpolated path.

    Args:
        name: label used in printouts and tests.
        kind: how to draw it -- "person", "dog" or "float".
        label: what the detector reports for it. A float that is sometimes
            called ``person`` is the interesting case, so this is separate from
            `kind`.
        path: keyframes of ``(frame_index, x, y_bottom)`` in pixels. The object
            does not exist before the first or after the last keyframe.
        height: drawn height in pixels; width is derived from `kind`.
        detect_prob: probability the detector reports it on any given frame.
        stray_label: label occasionally reported instead of `label`.
        stray_prob: probability of `stray_label` on a frame where it is detected.
        dropout: inclusive ``(start, end)`` frame range with no detections at
            all -- an occlusion behind a bush or a garden chair.
    """

    name: str
    kind: str
    label: str
    path: list[tuple[int, float, float]]
    height: float = 150.0
    detect_prob: float = 0.95
    stray_label: str | None = None
    stray_prob: float = 0.0
    dropout: tuple[int, int] | None = None

    @property
    def width(self) -> float:
        ratios = {"person": 0.38, "dog": 0.95, "float": 1.0}
        return self.height * ratios.get(self.kind, 0.5)

    def position_at(self, frame: int) -> tuple[float, float] | None:
        """Interpolated (x, y_bottom) at this frame, or None if not present."""
        if not self.path or frame < self.path[0][0] or frame > self.path[-1][0]:
            return None
        for (f0, x0, y0), (f1, x1, y1) in zip(self.path, self.path[1:]):
            if f0 <= frame <= f1:
                span = f1 - f0
                t = 0.0 if span == 0 else (frame - f0) / span
                return (x0 + t * (x1 - x0), y0 + t * (y1 - y0))
        return (self.path[-1][1], self.path[-1][2])

    def box_at(self, frame: int) -> np.ndarray | None:
        """True bounding box (x1, y1, x2, y2) at this frame, or None."""
        position = self.position_at(frame)
        if position is None:
            return None
        x, y_bottom = position
        half = self.width / 2.0
        return np.array([x - half, y_bottom - self.height, x + half, y_bottom])


@dataclass
class SyntheticScene:
    """A drawable, ground-truth-carrying scene.

    Args:
        width, height: frame size in pixels.
        n_frames: length of the clip.
        fps: nominal frame rate, used for the output video and for `dt`.
        objects: the moving contents.
        markers: pixel positions of the physical objects that mark the zone
            corners -- the bottles you would put on the grass.
    """

    width: int = 960
    height: int = 540
    n_frames: int = 460
    fps: float = 25.0
    objects: list[SceneObject] = field(default_factory=list)
    markers: list[tuple[int, int]] = field(default_factory=list)

    @property
    def zone_points(self) -> np.ndarray:
        """The marker quadrilateral, normalised -- ready for `Zone(points=...)`."""
        return np.array(self.markers, dtype=float) / np.array([self.width, self.height])

    def _background(self) -> np.ndarray:
        """Grass with enough texture that a static frame does not look like a bug."""
        rng = np.random.default_rng(7)
        frame = np.full((self.height, self.width, 3), GRASS, dtype=np.uint8)
        noise = rng.normal(0, 7, (self.height, self.width, 1))
        frame = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        for _ in range(28):  # darker tufts
            cx, cy = rng.integers(0, self.width), rng.integers(0, self.height)
            cv2.ellipse(frame, (int(cx), int(cy)), (rng.integers(12, 40), rng.integers(5, 14)),
                        float(rng.integers(0, 180)), 0, 360, (62, 112, 70), -1)
        return frame

    def render(self, frame_index: int, base: np.ndarray | None = None) -> np.ndarray:
        """Draw the scene at this frame."""
        frame = (self._background() if base is None else base).copy()

        for marker in self.markers:  # the bottles standing in for pool corners
            cv2.rectangle(frame, (marker[0] - 7, marker[1] - 26), (marker[0] + 7, marker[1]),
                          (210, 225, 235), -1)
            cv2.rectangle(frame, (marker[0] - 7, marker[1] - 26), (marker[0] + 7, marker[1]),
                          (120, 130, 140), 1)

        for obj in self.objects:
            box = obj.box_at(frame_index)
            if box is None:
                continue
            self._draw_object(frame, obj, box)
        return frame

    @staticmethod
    def _draw_object(frame: np.ndarray, obj: SceneObject, box: np.ndarray) -> None:
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        cx = (x1 + x2) // 2
        if obj.kind == "person":
            head_r = max(6, (x2 - x1) // 3)
            cv2.circle(frame, (cx, y1 + head_r), head_r, (60, 90, 160), -1)
            cv2.rectangle(frame, (x1, y1 + 2 * head_r), (x2, y2), (150, 70, 60), -1)
        elif obj.kind == "dog":
            cv2.ellipse(frame, ((x1 + x2) // 2, (y1 + y2) // 2),
                        ((x2 - x1) // 2, (y2 - y1) // 2), 0, 0, 360, (70, 110, 155), -1)
            cv2.circle(frame, (x2 - (x2 - x1) // 8, y1 + (y2 - y1) // 3),
                       max(5, (y2 - y1) // 3), (70, 110, 155), -1)
        else:  # float: bright ring on the water
            cv2.circle(frame, (cx, (y1 + y2) // 2), max(8, (x2 - x1) // 2), (60, 190, 240), -1)
            cv2.circle(frame, (cx, (y1 + y2) // 2), max(3, (x2 - x1) // 5), (76, 132, 84), -1)


class ScriptedDetector:
    """Feeds the tracker ground truth degraded to look like a real detector.

    Args:
        scene: the scene to read ground truth from.
        noise_px: std-dev of the jitter added to each box edge, in pixels.
        seed: fixed so a demo or test run is bit-for-bit repeatable.
    """

    def __init__(self, scene: SyntheticScene, noise_px: float = 4.0, seed: int = 11) -> None:
        self.scene = scene
        self.noise_px = float(noise_px)
        self.rng = np.random.default_rng(seed)
        self.index = -1

    def detect(self, frame: np.ndarray | None = None, index: int | None = None) -> list[Detection]:
        """Detections for the next frame (or for `index`, if given).

        `frame` is accepted and ignored, so this is drop-in interchangeable with
        `YoloxDetector.detect`; the answers come from the scene's ground truth
        rather than from the pixels.
        """
        del frame
        self.index = self.index + 1 if index is None else index
        detections: list[Detection] = []

        for obj in self.scene.objects:
            box = obj.box_at(self.index)
            if box is None:
                continue
            if obj.dropout and obj.dropout[0] <= self.index <= obj.dropout[1]:
                continue  # occluded: the detector sees nothing at all
            if self.rng.random() > obj.detect_prob:
                continue  # ordinary missed frame

            label = obj.label
            if obj.stray_label and self.rng.random() < obj.stray_prob:
                label = obj.stray_label

            noisy = box + self.rng.normal(0, self.noise_px, 4)
            score = float(np.clip(self.rng.normal(0.72, 0.08), 0.36, 0.99))
            detections.append(Detection(box=noisy, score=score, label=label))
        return detections


def default_scene() -> SyntheticScene:
    """The three-act scene described in this module's docstring.

    Timeline at 25 fps (frame numbers in brackets):
      [0-80]    pool OPEN, an adult wades in the zone             -> silent
      [80]      the user closes the pool; the float becomes scenery
      [90-150]  the adult leaves, inside the grace period         -> silent
      [0-460]   the float sits in the water, mislabelled at times -> silent
      [170-260] a dog crosses the grass outside the zone          -> silent
      [200-300] a half-seen figure enters the zone                -> NOTICE
      [300-460] someone enters the zone, occluded at [332-340]    -> ALARM
    """
    markers = [(300, 300), (720, 300), (790, 470), (250, 470)]

    adult = SceneObject(
        name="adult-swimmer", kind="person", label="person", height=150,
        path=[(0, 200, 470), (40, 430, 430), (80, 520, 420), (150, 900, 430)],
    )
    # The forgotten inflatable: never moves, and the detector calls it a person
    # in a minority of frames. Exactly the case that must not raise an alarm.
    forgotten_float = SceneObject(
        name="forgotten-float", kind="float", label="person", height=54,
        path=[(0, 610, 400), (459, 610, 400)],
        detect_prob=0.22,
    )
    dog = SceneObject(
        name="passing-dog", kind="dog", label="dog", height=70,
        path=[(170, -60, 250), (260, 1020, 250)],  # well above the zone: outside
    )
    # Far from the camera and in poor light, so the detector only manages it on
    # roughly half the frames. Consistently a person, but not consistently seen:
    # this is what the NOTICE tier exists for -- a chime and a still image rather
    # than a siren, and a human settles it.
    distant_figure = SceneObject(
        name="distant-figure", kind="person", label="person", height=88,
        path=[(200, 170, 360), (240, 330, 358), (300, 560, 356)],
        detect_prob=0.55,
    )
    # The dropout lands exactly on the frames where this person crosses the
    # boundary -- the hardest possible moment to lose sight of them. The Kalman
    # filter coasts the track through it, so the dwell counter keeps running and
    # the alarm still fires. Without the filter the track would break in two and
    # the counter would restart from zero.
    intruder = SceneObject(
        name="intruder", kind="person", label="person", height=120,
        path=[(300, 1000, 440), (340, 700, 430), (380, 520, 420), (460, 430, 430)],
        detect_prob=0.94, dropout=(332, 340),
    )

    return SyntheticScene(
        objects=[adult, forgotten_float, dog, distant_figure, intruder],
        markers=markers,
    )


def replay(
    scene: SyntheticScene,
    close_at_frame: int,
    policy: Policy | None = None,
) -> Iterator[tuple[int, list[Track], list[Event], PoolMonitor]]:
    """Run detector -> tracker -> monitor over a scene, yielding per-frame state.

    Rendering is left to the caller, so the demo video and the integration tests
    drive exactly the same pipeline instead of two loops that could drift apart.

    The clock is the frame index divided by fps -- video time, not wall-clock
    time -- which makes cooldowns and the grace period reproducible run to run.
    """
    zone = Zone(points=scene.zone_points, name="pool")
    tracker = Tracker(dt=1.0 / scene.fps)
    monitor = PoolMonitor(zone, policy or Policy(grace_seconds=5.0))
    detector = ScriptedDetector(scene)
    frame_shape = (scene.height, scene.width, 3)

    for index in range(scene.n_frames):
        now = index / scene.fps
        if index == close_at_frame:
            monitor.close_pool(tracker.confirmed_tracks(), now=now)
        tracks = tracker.update(detector.detect(index=index))
        yield index, tracks, monitor.update(tracks, frame_shape, now=now), monitor
 