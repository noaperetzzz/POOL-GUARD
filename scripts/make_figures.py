"""Generate the figures used in the README, straight from the real pipeline.

Two outputs:

* ``docs/kalman_occlusion.png`` -- the intruder's true position, the noisy
  detections, and the filter's estimate, with the occlusion shaded. This is the
  evidence that the tracker coasts through a gap instead of losing the subject.
* ``docs/stills.png`` -- three annotated frames: silent while the pool is open,
  a NOTICE on a half-seen figure, an ALARM on someone who walked in.

Usage:
    python scripts/make_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pool_guard.monitor import Policy, Severity  # noqa: E402
from pool_guard.engine import draw_overlay  # noqa: E402
from pool_guard.synthetic import ScriptedDetector, default_scene, replay  # noqa: E402
from pool_guard.tracker import Tracker  # noqa: E402
from pool_guard.zone import Zone  # noqa: E402

DOCS = Path(__file__).resolve().parent.parent / "docs"
CLOSE_AT = 80
POLICY = Policy(grace_seconds=5.0)


def centre_x(box: np.ndarray) -> float:
    return float((box[0] + box[2]) / 2)


def kalman_figure() -> Path:
    """Follow the intruder frame by frame and plot truth vs detection vs estimate."""
    scene = default_scene()
    intruder = next(obj for obj in scene.objects if obj.name == "intruder")
    detector = ScriptedDetector(scene)
    tracker = Tracker(dt=1.0 / scene.fps)

    frames, truth, measured, estimated = [], [], [], []
    for index in range(scene.n_frames):
        detections = detector.detect(index=index)
        tracks = tracker.update(detections)

        true_box = intruder.box_at(index)
        if true_box is None:
            continue
        true_x = centre_x(true_box)

        # Pair the detection and the track belonging to the intruder by proximity
        # to its known position -- the plot needs ground-truth pairing, which only
        # the synthetic scene can provide.
        #
        # Take the *nearest* candidate, not the first one inside the window. An
        # earlier version took `[0]`, and once the float's track drifted within
        # the window it was silently plotted as the intruder, which made the
        # filter look 3x worse than the raw detections it was cleaning up.
        def nearest(items):
            candidates = [i for i in items if abs(centre_x(i.box) - true_x) < 60]
            if not candidates:
                return np.nan
            return centre_x(min(candidates, key=lambda i: abs(centre_x(i.box) - true_x)).box)

        frames.append(index)
        truth.append(true_x)
        measured.append(nearest(detections))
        estimated.append(nearest(tracks))

    frames = np.array(frames)
    truth, measured, estimated = np.array(truth), np.array(measured), np.array(estimated)

    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )

    start, end = intruder.dropout
    for axis in (top, bottom):
        axis.axvspan(start, end, color="#d94f4f", alpha=0.12, lw=0)
        axis.grid(alpha=0.25, lw=0.6)

    top.plot(frames, truth, color="#444", lw=2.4, label="true position")
    top.plot(frames, measured, "o", color="#e08a3c", ms=4, alpha=0.75, label="detection (noisy)")
    top.plot(frames, estimated, color="#2a7fbf", lw=2.0, label="Kalman estimate")
    top.set_ylabel("horizontal position (px)")
    top.set_title("Tracking a subject through a 9-frame occlusion", fontsize=13)
    top.legend(loc="upper right", framealpha=0.95)
    top.text(
        (start + end) / 2, top.get_ylim()[1] * 0.97, "occluded\n(no detections)",
        ha="center", va="top", fontsize=9, color="#a33",
    )

    detection_error = np.abs(measured - truth)
    estimate_error = np.abs(estimated - truth)
    bottom.plot(frames, detection_error, "o", color="#e08a3c", ms=3.5, alpha=0.75, label="detection error")
    bottom.plot(frames, estimate_error, color="#2a7fbf", lw=1.8, label="estimate error")
    bottom.set_xlabel("frame")
    bottom.set_ylabel("|error| (px)")
    bottom.legend(loc="upper right", framealpha=0.95)

    with np.errstate(invalid="ignore"):
        raw_mean = float(np.nanmean(detection_error))
        visible = ~np.isnan(measured)
        filtered_mean = float(np.nanmean(estimate_error[visible]))
    bottom.set_title(
        f"mean error while visible: detections {raw_mean:.1f} px, filtered {filtered_mean:.1f} px",
        fontsize=10,
    )

    DOCS.mkdir(parents=True, exist_ok=True)
    out = DOCS / "kalman_occlusion.png"
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}  (detections {raw_mean:.1f} px -> filtered {filtered_mean:.1f} px)")
    return out


def stills_figure() -> Path:
    """A three-frame contact sheet: silent, notice, alarm."""
    scene = default_scene()
    zone = Zone(points=scene.zone_points, name="pool")
    background = scene._background()

    wanted = {"OPEN - swimmer in the pool, no alarm": 55}
    captured: dict[str, np.ndarray] = {}
    pending: list[tuple[str, int]] = []

    for index, tracks, events, monitor in replay(scene, CLOSE_AT, POLICY):
        frame = scene.render(index, base=background)
        banner = None
        for event in events:
            label = "NOTICE - half-seen figure, chime only" if event.severity is Severity.NOTICE \
                else "ALARM - person entered a closed pool"
            banner = (
                f"{event.severity.value.upper()}: {event.reason}",
                (0, 0, 200) if event.severity is Severity.ALARM else (0, 140, 200),
            )
            pending.append((label, index))

        display = draw_overlay(frame, zone, tracks, monitor.mode, scene.fps, banner)
        for label, at in list(pending):
            if index == at:  # the frame the event fired on
                captured[label] = display.copy()
                pending.remove((label, at))
        for label, at in wanted.items():
            if index == at:
                captured[label] = display.copy()

    order = [
        "OPEN - swimmer in the pool, no alarm",
        "NOTICE - half-seen figure, chime only",
        "ALARM - person entered a closed pool",
    ]
    panels = [captured[label] for label in order if label in captured]
    if len(panels) != 3:
        raise SystemExit(f"expected 3 panels, captured {list(captured)}")

    # Stack vertically with a caption strip above each panel.
    strip_height = 34
    width = panels[0].shape[1]
    pieces = []
    for label, panel in zip(order, panels):
        strip = np.full((strip_height, width, 3), 32, dtype=np.uint8)
        cv2.putText(strip, label, (12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (235, 235, 235), 1, cv2.LINE_AA)
        pieces.extend([strip, panel])

    sheet = np.vstack(pieces)
    DOCS.mkdir(parents=True, exist_ok=True)
    # JPEG, not PNG: the grass texture is noise, which PNG cannot compress, and
    # a 2.6 MB header image in a repository this size is rude.
    out = DOCS / "stills.jpg"
    cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"wrote {out}  ({sheet.shape[1]}x{sheet.shape[0]})")
    return out


if __name__ == "__main__":
    kalman_figure()
    stills_figure()
