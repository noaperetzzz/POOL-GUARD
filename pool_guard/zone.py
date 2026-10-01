"""The protected zone: an arbitrary polygon drawn once over the camera view.

The zone is stored in normalised coordinates (0..1 of frame width and height)
rather than pixels, so the same saved zone works whether the video is played
back at 1080p or downscaled to 640 wide for speed.

Point-in-polygon is implemented here rather than pulled in from Shapely. It is
twelve lines of ray casting, it removes a dependency, and it is exactly the kind
of thing worth being able to write rather than import.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


def point_in_polygon(x: float, y: float, polygon: np.ndarray) -> bool:
    """Ray-casting test: is (x, y) inside the polygon?

    Cast a ray straight to the right from the point and count how many edges it
    crosses. Odd means inside, even means outside -- true for any simple polygon,
    convex or not, which matters because a pool deck is rarely a neat rectangle.

    The ``(y1 > y) != (y2 > y)`` form counts each edge from exactly one side, so
    a vertex lying exactly on the ray is not counted twice.
    """
    polygon = np.asarray(polygon, dtype=float).reshape(-1, 2)
    if len(polygon) < 3:
        return False

    inside = False
    for i in range(len(polygon)):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % len(polygon)]
        if (y1 > y) != (y2 > y):
            # x coordinate where this edge crosses the horizontal line through y
            crossing_x = x1 + (y - y1) / (y2 - y1) * (x2 - x1)
            if x < crossing_x:
                inside = not inside
    return inside


@dataclass
class Zone:
    """A protected region, as normalised polygon vertices in view order.

    Args:
        points: at least 3 (x, y) pairs, each component in 0..1.
        name: shown in the overlay and written into the event log.
    """

    points: np.ndarray
    name: str = "pool"

    def __post_init__(self) -> None:
        self.points = np.asarray(self.points, dtype=float).reshape(-1, 2)
        if len(self.points) < 3:
            raise ValueError("a zone needs at least 3 points")

    def contains_pixel(self, x: float, y: float, frame_shape: tuple[int, ...]) -> bool:
        """Is this pixel coordinate inside the zone?"""
        height, width = frame_shape[:2]
        return point_in_polygon(x / width, y / height, self.points)

    def pixel_points(self, frame_shape: tuple[int, ...]) -> np.ndarray:
        """The polygon in pixel coordinates, ready for OpenCV drawing."""
        height, width = frame_shape[:2]
        return np.round(self.points * np.array([width, height])).astype(np.int32)

    def draw(
        self,
        frame: np.ndarray,
        colour: tuple[int, int, int] = (0, 200, 255),
        alpha: float = 0.18,
    ) -> np.ndarray:
        """Draw the zone onto the frame in place: filled tint plus outline."""
        points = self.pixel_points(frame.shape)
        overlay = frame.copy()
        cv2.fillPoly(overlay, [points], colour)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, dst=frame)
        cv2.polylines(frame, [points], isClosed=True, color=colour, thickness=2)
        return frame

    def to_dict(self) -> dict:
        return {"name": self.name, "points": self.points.tolist()}

    @classmethod
    def from_dict(cls, data: dict) -> "Zone":
        return cls(points=np.asarray(data["points"], dtype=float), name=data.get("name", "pool"))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Zone":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"no zone file at {path}\nUse the GUI to mark the zone: python -m pool_guard.setup_gui"
            )
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
