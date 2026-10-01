"""Sanity-check the real detector on a still image.

Worth running once after downloading the model, and the fastest way to see
whether a camera angle or lighting is giving the detector trouble before
blaming the alarm logic for a miss.

Usage:
    python scripts/check_detector.py photo.jpg [--all-classes] [--out annotated.jpg]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pool_guard.detector import YoloxDetector  # noqa: E402
from pool_guard.paths import model_path  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", help="path to a photo")
    parser.add_argument("--model", default=str(model_path()))
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--all-classes", action="store_true", help="do not filter to person/dog/cat")
    parser.add_argument("--out", default=None, help="write an annotated copy here")
    args = parser.parse_args(argv)

    frame = cv2.imread(args.image)
    if frame is None:
        raise SystemExit(f"could not read image: {args.image}")

    detector = YoloxDetector(
        args.model, conf_threshold=args.conf,
        keep_classes=None if args.all_classes else ("person", "dog", "cat"),
    )
    detections = detector.detect(frame)

    print(f"{args.image}  {frame.shape[1]}x{frame.shape[0]}  ->  {len(detections)} detection(s)")
    for detection in sorted(detections, key=lambda d: -d.score):
        x1, y1, x2, y2 = (round(float(v)) for v in detection.box)
        print(f"  {detection.label:<10s} {detection.score:.2f}  box=({x1},{y1})-({x2},{y2})")

    if args.out:
        for detection in detections:
            x1, y1, x2, y2 = (int(round(float(v))) for v in detection.box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 220, 120), 2)
            cv2.putText(
                frame, f"{detection.label} {detection.score:.2f}", (x1, max(18, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 220, 120), 2, cv2.LINE_AA,
            )
        cv2.imwrite(args.out, frame)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
