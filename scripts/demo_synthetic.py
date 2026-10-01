"""Run the full pipeline over the synthetic scene and write an annotated video.

No camera, no footage, no model file needed -- the detector is scripted from the
scene's ground truth. This is what produces the animation in the README, and it
is the fastest way to see the decision logic behave.

Usage:
    python scripts/demo_synthetic.py [--out docs/demo.mp4] [--gif docs/demo.gif]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pool_guard.alarm import AlarmSink  # noqa: E402
from pool_guard.monitor import Policy, Severity  # noqa: E402
from pool_guard.engine import draw_overlay  # noqa: E402
from pool_guard.synthetic import default_scene, replay  # noqa: E402
from pool_guard.zone import Zone  # noqa: E402

CLOSE_AT_FRAME = 80  # the moment the user presses 'a'


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="docs/demo.mp4", help="annotated video output")
    parser.add_argument("--gif", default=None, help="also write a GIF here (needs imageio)")
    parser.add_argument("--snapshots", action="store_true", help="write event snapshots to events/")
    args = parser.parse_args(argv)

    scene = default_scene()
    zone = Zone(points=scene.zone_points, name="pool")
    alarm = AlarmSink(sound=False) if args.snapshots else None
    # Shorter than the 8 s default so the whole story fits in a 460-frame clip.
    policy = Policy(grace_seconds=5.0)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), scene.fps, (scene.width, scene.height)
    )
    gif_frames = []

    background = scene._background()  # rendered once; the grass does not change
    banner: tuple[str, tuple[int, int, int]] | None = None
    banner_until = -1
    counts = {Severity.ALARM: 0, Severity.NOTICE: 0}

    for index, tracks, events, monitor in replay(scene, CLOSE_AT_FRAME, policy):
        frame = scene.render(index, base=background)

        if index == CLOSE_AT_FRAME:
            print(f"frame {index:3d}  POOL CLOSED - {len(tracks)} object(s) recorded as scenery")
            banner = (f"POOL CLOSED - {policy.grace_seconds:.0f}s grace, then watching", (40, 40, 160))
            banner_until = index + 40

        for event in events:
            counts[event.severity] = counts.get(event.severity, 0) + 1
            print(
                f"frame {index:3d}  {event.severity.value.upper():6s} "
                f"track #{event.track_id} {event.label} :: {event.reason}"
            )
            if alarm:
                alarm.handle(event, frame)
            banner = (
                f"{event.severity.value.upper()}: {event.reason}",
                (0, 0, 200) if event.severity is Severity.ALARM else (0, 140, 200),
            )
            banner_until = index + 60

        if banner and index > banner_until:
            banner = None

        display = draw_overlay(frame, zone, tracks, monitor.mode, scene.fps, banner)
        writer.write(display)
        if args.gif and index % 2 == 0:  # halve the frame count to keep the GIF small
            gif_frames.append(cv2.cvtColor(display, cv2.COLOR_BGR2RGB))

    writer.release()
    print(f"\nwrote {out_path}")

    if args.gif:
        try:
            import imageio.v2 as imageio

            Path(args.gif).parent.mkdir(parents=True, exist_ok=True)
            imageio.mimsave(args.gif, gif_frames, fps=scene.fps / 2, loop=0)
            print(f"wrote {args.gif}")
        except ImportError:
            print("imageio not installed; skipped the GIF (pip install imageio)")

    print(
        f"summary: {counts[Severity.ALARM]} alarm(s), {counts[Severity.NOTICE]} notice(s) "
        f"over {scene.n_frames} frames"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
