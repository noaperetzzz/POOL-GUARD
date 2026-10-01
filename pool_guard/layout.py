"""Turn tape-measure readings of a four-sided zone into a flat ground plan.

The zone is a quadrilateral A-B-C-D, in boundary order. It does **not** have to
be a rectangle -- any four-sided shape works, which is what a real garden or pool
deck usually is.

Why a diagonal is required
--------------------------
A triangle is rigid: three sides determine it completely. A quadrilateral is not.
Given only AB, BC, CD and DA it can flex like a hinged frame, so those four
numbers describe infinitely many different shapes. Measuring one diagonal splits
the quadrilateral into two triangles, and two rigid triangles glued along a shared
edge give exactly one shape.

Measuring *both* diagonals is better still. Five numbers determine the plan, so
the sixth is redundant -- and a redundant measurement is a free check on all the
others. `QuadPlan.diagonal_error_cm` reports how far the unused diagonal came out
from what was measured, which catches a mistyped number or a tape snagged on the
grass before it silently corrupts the calibration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

LABELS = ("A", "B", "C", "D")


@dataclass(frozen=True)
class QuadPlan:
    """A reconstructed four-sided ground plan, in centimetres.

    Args:
        points_cm: ``(4, 2)`` coordinates for A, B, C, D. A sits at the origin
            and B on the +X axis, so the plan is in a fixed, reproducible frame.
        diagonal_error_cm: how far the redundant diagonal missed, or None if only
            one diagonal was measured.
        used_diagonal: which diagonal defined the shape, "AC" or "BD".
    """

    points_cm: np.ndarray
    diagonal_error_cm: float | None
    used_diagonal: str

    @property
    def area_m2(self) -> float:
        """Zone area by the shoelace formula, in square metres."""
        x, y = self.points_cm[:, 0], self.points_cm[:, 1]
        area_cm2 = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
        return float(area_cm2) / 10_000.0

    def side_lengths_cm(self) -> dict[str, float]:
        """The four sides as reconstructed, for comparison with what was measured."""
        sides = {}
        for index in range(4):
            a, b = index, (index + 1) % 4
            name = f"{LABELS[a]}{LABELS[b]}"
            sides[name] = float(np.linalg.norm(self.points_cm[a] - self.points_cm[b]))
        return sides

    def report(self) -> str:
        """A short human-readable summary for the GUI and the console."""
        lines = [f"Ground plan from diagonal {self.used_diagonal}, area {self.area_m2:.1f} m2"]
        for label, point in zip(LABELS, self.points_cm):
            lines.append(f"  {label}: X={point[0]:8.1f}  Y={point[1]:8.1f} cm")
        if self.diagonal_error_cm is not None:
            other = "BD" if self.used_diagonal == "AC" else "AC"
            lines.append(f"  cross-check on {other}: off by {self.diagonal_error_cm:.1f} cm")
        return "\n".join(lines)


def _third_vertex(
    base_length: float, from_first: float, from_second: float, above: bool
) -> np.ndarray:
    """Locate a point given its distance to (0, 0) and to (base_length, 0).

    Intersecting two circles. `above` selects which of the two mirror-image
    solutions to take, which is how B and D are placed on opposite sides of the
    shared diagonal and the quadrilateral comes out in boundary order rather than
    folded over itself.
    """
    x = (from_first ** 2 + base_length ** 2 - from_second ** 2) / (2.0 * base_length)
    height_squared = from_first ** 2 - x ** 2
    if height_squared < 0:
        # The circles do not meet: the triangle inequality is violated, so these
        # three lengths cannot coexist in a plane.
        raise ValueError(
            f"impossible triangle: sides {from_first:.0f} and {from_second:.0f} cm "
            f"cannot span a {base_length:.0f} cm base"
        )
    y = math.sqrt(height_squared)
    return np.array([x, y if above else -y])


def _canonicalise(points: np.ndarray) -> np.ndarray:
    """Put A at the origin, B on the +X axis, and the polygon counter-clockwise.

    Distances alone cannot tell a shape from its mirror image, so the reflection
    has to be pinned down explicitly. Without this the same measurements could
    yield a flipped plan, and a homography fitted from it would map the scene
    inside out while looking perfectly plausible.
    """
    points = np.asarray(points, dtype=float) - points[0]

    angle = math.atan2(points[1, 1], points[1, 0])
    cos, sin = math.cos(-angle), math.sin(-angle)
    rotation = np.array([[cos, -sin], [sin, cos]])
    points = points @ rotation.T

    x, y = points[:, 0], points[:, 1]
    signed_area = 0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    if signed_area < 0:
        points[:, 1] *= -1
    return points


def solve_quad(
    ab: float,
    bc: float,
    cd: float,
    da: float,
    ac: float | None = None,
    bd: float | None = None,
) -> QuadPlan:
    """Reconstruct the ground plan of a four-sided zone.

    Args:
        ab, bc, cd, da: side lengths in centimetres, in boundary order.
        ac: diagonal from A to C, in centimetres.
        bd: diagonal from B to D, in centimetres.

    At least one diagonal is required; give both and the unused one becomes a
    consistency check.

    Raises:
        ValueError: if no diagonal is given, a length is not positive, or the
            lengths describe a shape that cannot exist in a plane.
    """
    sides = {"AB": ab, "BC": bc, "CD": cd, "DA": da}
    for name, value in sides.items():
        if value is None or value <= 0:
            raise ValueError(f"side {name} must be a positive length")
    if ac is None and bd is None:
        raise ValueError(
            "at least one diagonal (AC or BD) is required -- four sides alone do "
            "not determine a quadrilateral"
        )

    if ac is not None:
        # Split along AC into triangles ABC and ACD. B and D go on opposite sides.
        a = np.array([0.0, 0.0])
        c = np.array([float(ac), 0.0])
        b = _third_vertex(ac, ab, bc, above=True)
        d = _third_vertex(ac, da, cd, above=False)
        used = "AC"
    else:
        # Split along BD into triangles ABD and BCD instead.
        b = np.array([0.0, 0.0])
        d = np.array([float(bd), 0.0])
        a = _third_vertex(bd, ab, da, above=True)
        c = _third_vertex(bd, bc, cd, above=False)
        used = "BD"

    points = _canonicalise(np.stack([a, b, c, d]))

    error = None
    if ac is not None and bd is not None:
        computed_bd = float(np.linalg.norm(points[1] - points[3]))
        error = abs(computed_bd - float(bd))

    return QuadPlan(points_cm=points, diagonal_error_cm=error, used_diagonal=used)
