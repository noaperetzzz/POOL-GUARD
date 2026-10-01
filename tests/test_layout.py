"""Tests for reconstructing a four-sided ground plan from tape measurements."""

from __future__ import annotations

import math

import numpy as np
import pytest

from pool_guard.layout import solve_quad


def distances(points: np.ndarray) -> dict[str, float]:
    """All six pairwise distances of a quadrilateral given as four points."""
    a, b, c, d = points
    return {
        "ab": float(np.linalg.norm(a - b)),
        "bc": float(np.linalg.norm(b - c)),
        "cd": float(np.linalg.norm(c - d)),
        "da": float(np.linalg.norm(d - a)),
        "ac": float(np.linalg.norm(a - c)),
        "bd": float(np.linalg.norm(b - d)),
    }


def shape_signature(points: np.ndarray) -> list[float]:
    """All six distances, sorted -- invariant under rotation, translation, reflection."""
    return sorted(distances(points).values())


def test_a_rectangle_comes_back_as_a_rectangle() -> None:
    plan = solve_quad(ab=300, bc=400, cd=300, da=400, ac=500)
    sides = plan.side_lengths_cm()

    assert sides["AB"] == pytest.approx(300, abs=0.1)
    assert sides["BC"] == pytest.approx(400, abs=0.1)
    assert plan.area_m2 == pytest.approx(12.0, abs=0.05)

    # Opposite corners of a 300x400 rectangle are 500 apart both ways.
    computed_bd = float(np.linalg.norm(plan.points_cm[1] - plan.points_cm[3]))
    assert computed_bd == pytest.approx(500, abs=0.5)


def test_an_irregular_quadrilateral_round_trips() -> None:
    """Take a known shape, measure it, reconstruct it, compare."""
    truth = np.array([[0.0, 0.0], [420.0, 30.0], [500.0, 380.0], [-60.0, 300.0]])
    measured = distances(truth)

    plan = solve_quad(**measured)

    assert shape_signature(plan.points_cm) == pytest.approx(shape_signature(truth), abs=0.5)
    assert plan.diagonal_error_cm == pytest.approx(0.0, abs=0.5)


def test_the_real_garden_measurements_are_self_consistent() -> None:
    """Four paving slabs measured with a tape: 309, 452, 340, 457, diagonals 540 and 579."""
    plan = solve_quad(ab=309, bc=452, cd=340, da=457, ac=540, bd=579)

    assert plan.used_diagonal == "AC"
    # Under 1% of the diagonal is what careful tape work on grass looks like.
    assert plan.diagonal_error_cm < 0.01 * 579
    assert 13.0 < plan.area_m2 < 17.0


def test_either_diagonal_gives_the_same_shape() -> None:
    """AC and BD are different routes to one answer; consistent data must agree."""
    truth = np.array([[0.0, 0.0], [380.0, -20.0], [430.0, 350.0], [-40.0, 320.0]])
    measured = distances(truth)

    from_ac = solve_quad(**{k: v for k, v in measured.items() if k != "bd"})
    from_bd = solve_quad(**{k: v for k, v in measured.items() if k != "ac"})

    assert from_ac.used_diagonal == "AC"
    assert from_bd.used_diagonal == "BD"
    assert shape_signature(from_ac.points_cm) == pytest.approx(
        shape_signature(from_bd.points_cm), abs=0.5
    )


def test_a_bad_diagonal_shows_up_as_a_large_error() -> None:
    """The redundant measurement exists to catch exactly this."""
    honest = solve_quad(ab=309, bc=452, cd=340, da=457, ac=540, bd=579)
    typo = solve_quad(ab=309, bc=452, cd=340, da=457, ac=540, bd=479)  # 5 typed as 4

    assert honest.diagonal_error_cm < 5
    assert typo.diagonal_error_cm > 90


def test_four_sides_without_a_diagonal_are_refused() -> None:
    """A quadrilateral is not rigid; refusing beats inventing a shape."""
    with pytest.raises(ValueError, match="diagonal"):
        solve_quad(ab=309, bc=452, cd=340, da=457)


def test_impossible_lengths_are_refused() -> None:
    """Two short sides cannot span a long diagonal."""
    with pytest.raises(ValueError, match="impossible"):
        solve_quad(ab=100, bc=100, cd=340, da=457, ac=540)


@pytest.mark.parametrize("bad", [0, -10, None])
def test_non_positive_sides_are_refused(bad) -> None:
    with pytest.raises(ValueError, match="positive"):
        solve_quad(ab=bad, bc=452, cd=340, da=457, ac=540)


def test_the_plan_is_in_a_canonical_frame() -> None:
    """A at the origin, B on the +X axis, and wound counter-clockwise."""
    plan = solve_quad(ab=309, bc=452, cd=340, da=457, ac=540, bd=579)
    points = plan.points_cm

    assert points[0] == pytest.approx([0.0, 0.0], abs=1e-6)
    assert points[1, 1] == pytest.approx(0.0, abs=1e-6)
    assert points[1, 0] > 0

    x, y = points[:, 0], points[:, 1]
    signed_area = 0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    assert signed_area > 0


def test_reconstruction_is_independent_of_how_the_zone_was_oriented() -> None:
    """Rotating the real zone must not change the reconstructed plan."""
    truth = np.array([[0.0, 0.0], [400.0, 0.0], [450.0, 330.0], [-30.0, 310.0]])
    angle = math.radians(37.0)
    rotation = np.array([
        [math.cos(angle), -math.sin(angle)],
        [math.sin(angle), math.cos(angle)],
    ])
    rotated = truth @ rotation.T + np.array([1234.0, -567.0])

    first = solve_quad(**distances(truth))
    second = solve_quad(**distances(rotated))

    assert first.points_cm == pytest.approx(second.points_cm, abs=0.5)
