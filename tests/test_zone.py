"""Tests for the zone polygon and the hand-rolled point-in-polygon test."""

from __future__ import annotations

import numpy as np
import pytest

from pool_guard.zone import Zone, point_in_polygon

SQUARE = np.array([[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]])

# An L shape. Its bounding box contains points the polygon itself does not, so a
# naive rectangle check would pass this while ray casting correctly rejects it --
# and a real pool deck is far more often L-shaped than rectangular.
L_SHAPE = np.array([[0.1, 0.1], [0.5, 0.1], [0.5, 0.5], [0.9, 0.5], [0.9, 0.9], [0.1, 0.9]])


@pytest.mark.parametrize("x,y,expected", [
    (0.5, 0.5, True),    # middle
    (0.25, 0.25, True),  # just inside a corner
    (0.1, 0.5, False),   # left of the polygon
    (0.9, 0.5, False),   # right of it
    (0.5, 0.1, False),   # above it
    (0.5, 0.9, False),   # below it
])
def test_square(x: float, y: float, expected: bool) -> None:
    assert point_in_polygon(x, y, SQUARE) is expected


@pytest.mark.parametrize("x,y,expected", [
    (0.3, 0.3, True),    # in the narrow upper arm
    (0.7, 0.7, True),    # in the wide lower part
    (0.7, 0.3, False),   # the notch: inside the bounding box, outside the shape
    (0.95, 0.7, False),
])
def test_concave_shape(x: float, y: float, expected: bool) -> None:
    assert point_in_polygon(x, y, L_SHAPE) is expected


def test_horizontal_ray_through_a_vertex_is_not_counted_twice() -> None:
    """A vertex exactly on the ray is the classic way ray casting goes wrong."""
    diamond = np.array([[0.5, 0.2], [0.8, 0.5], [0.5, 0.8], [0.2, 0.5]])
    assert point_in_polygon(0.5, 0.5, diamond) is True   # ray hits both side vertices
    assert point_in_polygon(0.05, 0.5, diamond) is False  # same ray, from outside
    assert point_in_polygon(0.95, 0.5, diamond) is False


def test_degenerate_polygon_is_never_inside() -> None:
    assert point_in_polygon(0.5, 0.5, np.array([[0.0, 0.0], [1.0, 1.0]])) is False


def test_zone_needs_three_points() -> None:
    with pytest.raises(ValueError):
        Zone(points=np.array([[0.0, 0.0], [1.0, 1.0]]))


def test_zone_is_resolution_independent() -> None:
    """Normalised storage is the point: the same zone must hold at any size."""
    zone = Zone(points=SQUARE)
    for shape in [(540, 960, 3), (1080, 1920, 3), (360, 640, 3)]:
        height, width = shape[:2]
        assert zone.contains_pixel(width * 0.5, height * 0.5, shape) is True
        assert zone.contains_pixel(width * 0.05, height * 0.5, shape) is False


def test_pixel_points_scale_with_the_frame() -> None:
    zone = Zone(points=SQUARE)
    points = zone.pixel_points((500, 1000, 3))
    assert points.tolist() == [[200, 100], [800, 100], [800, 400], [200, 400]]


def test_save_and_load_round_trip(tmp_path) -> None:
    path = tmp_path / "zone.json"
    Zone(points=L_SHAPE, name="lawn").save(path)
    loaded = Zone.load(path)

    assert loaded.name == "lawn"
    assert np.allclose(loaded.points, L_SHAPE)


def test_missing_zone_file_says_what_to_do(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="setup_gui"):
        Zone.load(tmp_path / "absent.json")


def test_draw_marks_inside_pixels_without_changing_the_frame_size() -> None:
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    Zone(points=SQUARE).draw(frame)

    assert frame.shape == (100, 200, 3)
    assert frame[50, 100].sum() > 0  # centre tinted
    assert frame[50, 5].sum() == 0   # far outside untouched
