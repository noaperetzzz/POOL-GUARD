"""One step of the pipeline: skip frames, detect, track, decide, draw.

The naive loop -- read every frame, detect on it, draw, repeat -- plays a 30 fps
video at whatever speed the detector manages, about 5 or 6 frames a second on a
CPU. The result is slow motion. On a live camera it is worse than slow: undecoded
frames pile up in the capture buffer and the delay grows without limit, so the
picture drifts further and further into the past. For a camera meant to catch
someone entering a pool, that defeats the point.

The fix is to skip. Only every `stride`-th frame is decoded and analysed; the rest
are thrown away without being looked at. With a stride of 6 on a 30 fps source,
five analysed frames per second consume thirty source frames per second -- real
time. The picture updates five times a second rather than thirty, so motion looks
a little steppy, but it is *current*, and every frame on screen is precisely the
frame the boxes were computed from.

That last part is why this beats the obvious alternative of running detection on a
background thread. Threads would give a smooth 30 fps picture, but the boxes would
always belong to a frame from one detection ago, and it would cost locks, a queue
and a class of bug where a worker thread reaches into a widget. A single loop with
frames skipped keeps the video and the detector in step by construction.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .alarm import AlarmSink
from .calibration import Calibration
from .detector import YoloxDetector
from .imaging import open_video
from .monitor import Event, Mode, PoolMonitor, Policy, Severity
from .paths import model_path
from .tracker import Track, Tracker
from .zone import Zone

COLOURS = {
    "person": (80, 220, 120),
    "dog": (200, 180, 90),
    "cat": (200, 180, 90),
}
MODE_COLOUR = {Mode.OPEN: (120, 220, 120), Mode.CLOSED: (80, 120, 250)}


def draw_overlay(
    frame: np.ndarray,
    zone: Zone,
    tracks: list[Track],
    mode: Mode,
    fps: float = 0.0,
    banner: tuple[str, tuple[int, int, int]] | None = None,
    calibration: Calibration | None = None,
    status: str | None = None,
) -> np.ndarray:
    """Draw the zone, the tracks and the status bar onto a frame, in place.

    A free function rather than a method, because the demo and the figure scripts
    need the same picture without owning a camera or a detector.
    """
    zone.draw(frame)
    if calibration is not None:
        calibration = calibration.scaled_to(frame.shape)

    for track in tracks:
        # A coasting track is a prediction, not a sighting. Drawing it puts a box
        # around nobody, which is alarming to look at and tells the viewer
        # something false. The track stays alive internally to keep its identity
        # across the gap; it simply is not shown until a detection backs it again.
        if not track.detected_now:
            continue

        x1, y1, x2, y2 = (int(round(v)) for v in track.box)
        colour = COLOURS.get(track.label, (180, 180, 180))
        if mode is Mode.CLOSED and zone.contains_pixel(*track.anchor(), frame.shape):
            colour = (60, 60, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

        caption = f"#{track.track_id} {track.label} {track.label_confidence:.0%}"
        if calibration is not None:
            speed = calibration.speed_mps(track.anchor(), track.filter.velocity)
            caption += f"  {speed:.1f} m/s"
        cv2.putText(
            frame, caption, (x1, max(18, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2, cv2.LINE_AA,
        )
        # The anchor is what the zone test actually uses; drawing it makes a
        # near-boundary decision visible instead of mysterious.
        cv2.circle(frame, tuple(int(round(v)) for v in track.anchor()), 4, colour, -1)

    if status is None:
        status = "CLOSED - watching" if mode is Mode.CLOSED else "OPEN - activity allowed"
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 30), (25, 25, 25), -1)
    cv2.putText(
        frame, f"{status}   |   {fps:4.1f} analysed/s",
        (12, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, MODE_COLOUR[mode], 2, cv2.LINE_AA,
    )

    if banner:
        text, colour = banner
        height, width = frame.shape[:2]
        cv2.rectangle(frame, (0, height - 42), (width, height), colour, -1)
        cv2.putText(
            frame, text, (12, height - 13),
            cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA,
        )
    return frame


@dataclass
class Snapshot:
    """One analysed moment, ready to display."""

    frame: np.ndarray
    tracks: list[Track] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)
    analysed_fps: float = 0.0
    position: int = 0
    total: int = 0


class MonitorEngine:
    """Owns the capture, the models and the decision state for one session.

    Args:
        source: video file path, or a camera index such as "0".
        zone: the protected polygon.
        policy: decision thresholds.
        calibration: optional; enables speeds in metres per second.
        model: detector weights; defaults to the bundled model.
        conf: detector confidence threshold.
        width: downscale frames to this width, 0 to keep the original.
        stride: analyse every Nth frame and discard the rest. 6 against a 30 fps
            source means five analysed frames a second, which is roughly what a
            CPU manages -- so the video plays at real speed. Raise it on a slower
            machine, lower it for a finer-grained look at fast movement.
        sound: play the siren.
    """

    def __init__(
        self,
        source: str,
        zone: Zone,
        policy: Policy | None = None,
        calibration: Calibration | None = None,
        model: str | Path | None = None,
        conf: float = 0.35,
        width: int = 960,
        stride: int = 6,
        sound: bool = True,
        telegram_token: str | None = None,
        telegram_chat_id: str | None = None,
    ) -> None:
        self.capture = open_video(source)
        if not self.capture.isOpened():
            raise RuntimeError(f"could not open source: {source}")

        self.is_camera = str(source).isdigit() or str(source).startswith(("rtsp://", "http://", "https://"))
        source_fps = self.capture.get(cv2.CAP_PROP_FPS) or 0.0
        if not 1.0 < source_fps < 240.0:  # webcams often report 0 or nonsense
            source_fps = 25.0
        self.source_fps = source_fps
        self.total_frames = 0 if self.is_camera else int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(width)
        self.stride = max(1, int(stride))

        self.zone = zone
        self.detector = YoloxDetector(model or model_path(), conf_threshold=conf)
        self.tracker = Tracker(dt=self.stride / source_fps)
        self.monitor = PoolMonitor(zone, policy or Policy(), calibration=calibration)
        self.alarm = AlarmSink(
            sound=sound,
            telegram_token=telegram_token,
            telegram_chat_id=telegram_chat_id,
        )

        self.banner: tuple[str, tuple[int, int, int]] | None = None
        self._banner_until = 0.0
        self.analysed_fps = 0.0
        self.finished = False
        self.paused = False
        self.position = 0
        self._last_step_at = time.perf_counter()
        self._last_cost = 0.0

    # -------------------------------------------------------------------- control

    def start(self) -> None:
        self._last_step_at = time.perf_counter()

    def close(self) -> None:
        self.capture.release()

    @property
    def mode(self) -> Mode:
        return self.monitor.mode

    def close_pool(self) -> int:
        """Arm the monitor. Returns how many objects were recorded as scenery."""
        scenery = self.tracker.confirmed_tracks()
        self.monitor.close_pool(scenery, now=time.monotonic())
        self.flash(
            f"POOL CLOSED - {self.monitor.policy.grace_seconds:.0f}s grace, then watching",
            (40, 40, 160),
        )
        return len(scenery)

    def open_pool(self) -> None:
        self.monitor.open_pool()
        self.flash("POOL OPEN - monitoring off", (40, 120, 40))

    def toggle(self) -> Mode:
        """Flip between open and closed. This is what the GUI button calls."""
        if self.mode is Mode.CLOSED:
            self.open_pool()
        else:
            self.close_pool()
        return self.mode

    def grace_remaining(self) -> float:
        """Seconds left of the exit delay; 0 once it has passed."""
        if self.mode is not Mode.CLOSED:
            return 0.0
        return max(0.0, self.monitor.policy.grace_seconds - (time.monotonic() - self.monitor._armed_at))

    def seek(self, frame_index: int) -> None:
        """Jump to a frame in a file.

        Tracks are cleared: after a jump the old ones describe a moment that is no
        longer on screen, and keeping them would have the tracker try to match a
        subject to where somebody else stood ten seconds ago.
        """
        if self.is_camera:
            return
        target = max(0, min(int(frame_index), max(0, self.total_frames - 1)))
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, target)
        self.position = target
        self.finished = False
        self.tracker.reset()

    def skip_seconds(self, seconds: float) -> None:
        """Jump forward or back, in seconds."""
        self.seek(self.position + int(seconds * self.source_fps))

    def set_paused(self, paused: bool) -> None:
        self.paused = bool(paused)

    def flash(self, text: str, colour: tuple[int, int, int], seconds: float = 3.0) -> None:
        self.banner = (text, colour)
        self._banner_until = time.monotonic() + seconds

    def next_delay_ms(self) -> int:
        """How long the caller should wait before the next `step`, to play in real time.

        One analysed frame stands for ``stride / fps`` seconds of video, so that is
        the budget. Whatever the detector already spent comes off it; if it
        overran there is nothing to give back and the answer is "immediately",
        which is the point at which the video does start falling behind real time.
        """
        if self.paused:
            return 60
        if self.is_camera:
            return 1  # no pace of its own: read as fast as possible to stay current
        budget = self.stride / self.source_fps
        return max(1, int((budget - self._last_cost) * 1000))

    # ----------------------------------------------------------------------- work

    def step(self) -> Snapshot | None:
        """Skip to the next frame of interest, analyse it, and return it drawn.

        Returns None at the end of a file, or while paused.
        """
        if self.paused:
            return None

        # Discard the frames in between with grab(): it advances the stream without
        # decoding a full image, which is the whole saving. For files, this speeds 
        # up playback. For live cameras, the background thread already drops old frames.
        if not self.is_camera:
            for _ in range(self.stride - 1):
                if not self.capture.grab():
                    self.finished = True
                    return None

        ok, frame = self.capture.read()
        if not ok:
            self.finished = True
            return None
        self.position = int(self.capture.get(cv2.CAP_PROP_POS_FRAMES))

        if self.width and frame.shape[1] > self.width:
            scale = self.width / frame.shape[1]
            frame = cv2.resize(frame, (self.width, int(round(frame.shape[0] * scale))))

        started = time.perf_counter()
        # A file advances by exactly stride/fps of video time per step. A camera
        # advances by however long the wall clock says, since it does not wait.
        if self.is_camera:
            dt = max(started - self._last_step_at, 1e-3)
        else:
            dt = self.stride / self.source_fps
        self._last_step_at = started

        detections = self.detector.detect(frame)
        tracks = self.tracker.update(detections, dt=dt)
        events = self.monitor.update(tracks, frame.shape, time.monotonic())

        for event in events:
            # Where in the recording this happened. The monitor deals in elapsed
            # time and knows nothing about the source, so this is filled in here.
            if not self.is_camera and self.source_fps > 0:
                event.video_seconds = self.position / self.source_fps
            self.alarm.handle(event, frame)
            self.flash(
                f"{event.severity.value.upper()}: {event.reason}",
                (0, 0, 200) if event.severity is Severity.ALARM else (0, 140, 200),
            )

        self._last_cost = time.perf_counter() - started
        instant = 1.0 / self._last_cost if self._last_cost > 0 else 0.0
        self.analysed_fps = instant if self.analysed_fps == 0 else 0.8 * self.analysed_fps + 0.2 * instant

        if self.banner and time.monotonic() > self._banner_until:
            self.banner = None

        return Snapshot(
            frame=self._draw(frame, tracks),
            tracks=tracks,
            events=events,
            analysed_fps=self.analysed_fps,
            position=self.position,
            total=self.total_frames,
        )

    def _draw(self, frame: np.ndarray, tracks: list[Track]) -> np.ndarray:
        grace = self.grace_remaining()
        if self.mode is Mode.CLOSED:
            status = f"CLOSED - grace {grace:.0f}s" if grace > 0 else "CLOSED - watching"
        else:
            status = "OPEN - activity allowed"

        return draw_overlay(
            frame, self.zone, tracks, mode=self.mode, fps=self.analysed_fps,
            banner=self.banner, calibration=self.monitor.calibration, status=status,
        )
