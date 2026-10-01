"""Map pixels to centimetres on the ground, using four marked points.

Once the four boundary points are known both in the image (clicked) and on the
ground (from `layout.solve_quad`), the two are related by a **homography**: the
projective transform that a pinhole camera applies to any flat surface. Four
point pairs determine it exactly, which is why a quadrilateral is the natural
thing to measure.

What this measures -- and what it deliberately does not
------------------------------------------------------
Exact, for anything lying **on the ground plane**:

* where a subject is standing, in centimetres
* how far apart two subjects are
* how fast someone is moving, in metres per second rather than pixels per frame

**Height is not offered, and that is a decision rather than an omission.** A
homography describes one flat surface and nothing above it, so a standing
person's head is not in its domain at all. An earlier version estimated height
from the local ground scale at the subject's feet anyway. Tested against a real
garden camera -- mounted high and angled down, as these cameras always are -- it
reported a 186 cm adult as anywhere from 154 to 204 cm, and the figure changed as
he walked towards the camera. Worse, it labelled a 160 cm teenager "adult" and a
60 cm dog "child-sized".

Doing it properly needs the vertical vanishing point or full camera pose. Until
then the honest answer is no answer: a plausible-looking number that is quietly
wrong is more dangerous in a safety system than an absent one, because it invites
decisions the measurement cannot support.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .layout import QuadPlan


@dataclass
class Calibration:
    """A homography between image pixels and ground centimetres.

    Args:
        image_points: ``(4, 2)`` pixel coordinates of A, B, C, D, as clicked.
        world_points_cm: ``(4, 2)`` ground coordinates of the same points.
        frame_size: ``(width, height)`` of the image the points were clicked on,
            so the calibration can be rescaled if the video is a different size.
    """

    image_points: np.ndarray
    world_points_cm: np.ndarray
    frame_size: tuple[int, int]

    def __post_init__(self) -> None:
        self.image_points = np.asarray(self.image_points, dtype=np.float32).reshape(4, 2)
        self.world_points_cm = np.asarray(self.world_points_cm, dtype=np.float32).reshape(4, 2)
        self.frame_size = (int(self.frame_size[0]), int(self.frame_size[1]))
        self._rebuild()

    def _rebuild(self) -> None:
        self.to_ground = cv2.getPerspectiveTransform(self.image_points, self.world_points_cm)
        self.to_image = np.linalg.inv(self.to_ground)

    def scaled_to(self, frame_shape: tuple[int, ...]) -> "Calibration":
        """A copy valid for a frame of a different resolution.

        The monitor downscales frames for speed, so the pixels it reports are not
        the pixels the points were clicked on. Scaling the image points keeps the
        two in step; forgetting to would silently misplace every measurement.
        """
        height, width = frame_shape[:2]
        if (width, height) == self.frame_size:
            return self
        scale = np.array([width / self.frame_size[0], height / self.frame_size[1]])
        return Calibration(
            image_points=self.image_points * scale,
            world_points_cm=self.world_points_cm,
            frame_size=(width, height),
        )

    def to_world(self, x: float, y: float) -> tuple[float, float]:
        """Ground position in centimetres of an image point on the ground plane."""
        point = np.array([[[float(x), float(y)]]], dtype=np.float32)
        mapped = cv2.perspectiveTransform(point, self.to_ground)[0, 0]
        return (float(mapped[0]), float(mapped[1]))

    def ground_distance_cm(
        self, first: tuple[float, float], second: tuple[float, float]
    ) -> float:
        """Real distance in centimetres between two points on the ground."""
        a = np.array(self.to_world(*first))
        b = np.array(self.to_world(*second))
        return float(np.linalg.norm(a - b))

    def cm_per_pixel_at(self, x: float, y: float) -> float:
        """Local vertical scale: ground centimetres spanned by one pixel upwards.

        Not a constant -- that is the whole point of a perspective view. A pixel
        near the top of the frame can be worth many times one near the bottom.
        """
        return self.ground_distance_cm((x, y), (x, y - 1.0))

    def speed_mps(self, anchor: tuple[float, float], velocity_px_per_s: np.ndarray) -> float:
        """Convert a track's pixel velocity into metres per second at its position."""
        vx, vy = (float(v) for v in velocity_px_per_s[:2])
        x, y = anchor
        moved = self.ground_distance_cm((x, y), (x + vx, y + vy))
        return moved / 100.0

    def to_dict(self) -> dict:
        return {
            "image_points": self.image_points.tolist(),
            "world_points_cm": self.world_points_cm.tolist(),
            "frame_size": list(self.frame_size),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Calibration":
        return cls(
            image_points=np.asarray(data["image_points"], dtype=np.float32),
            world_points_cm=np.asarray(data["world_points_cm"], dtype=np.float32),
            frame_size=tuple(data["frame_size"]),
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Calibration":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def from_plan(
        cls, image_points: np.ndarray, plan: QuadPlan, frame_size: tuple[int, int]
    ) -> "Calibration":
        """Build from clicked pixels plus a solved ground plan."""
        return cls(
            image_points=image_points,
            world_points_cm=plan.points_cm,
            frame_size=frame_size,
        )

    def check_cm(self) -> float:
        """Round-trip error in centimetres, as a guard against a broken matrix.

        With exactly four correspondences the homography fits them perfectly, so
        this is near zero by construction and proves nothing about the survey --
        it only catches a degenerate click pattern such as three points in a line.
        The real validation of the measurements is
        `QuadPlan.diagonal_error_cm`.
        """
        worst = 0.0
        for pixel, world in zip(self.image_points, self.world_points_cm):
            mapped = np.array(self.to_world(*pixel))
            worst = max(worst, float(np.linalg.norm(mapped - world)))
        return worst


def from_settings(
    image_points: np.ndarray, measurements: dict[str, float], frame_size: tuple[int, int]
) -> "Calibration | None":
    """Build a calibration from the INI's ``[measurements]`` section, or None.

    Returns None rather than raising when the numbers are missing or inconsistent:
    the calibration only adds real-world units, so the pool must go on being
    watched regardless of what is in the settings file.
    """
    from .layout import solve_quad

    try:
        plan = solve_quad(
            ab=measurements.get("side_1_2"),
            bc=measurements.get("side_2_3"),
            cd=measurements.get("side_3_4"),
            da=measurements.get("side_4_1"),
            ac=measurements.get("diagonal_1_3"),
            bd=measurements.get("diagonal_2_4"),
        )
    except (ValueError, TypeError):
        return None
    return Calibration.from_plan(image_points, plan, frame_size)
