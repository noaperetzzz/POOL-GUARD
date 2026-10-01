"""Tests for the pixel-to-centimetre homography."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from pool_guard.calibration import Calibration, from_settings
from pool_guard.layout import solve_quad

# A ground plan, and where a tilted camera would see its four corners.
WORLD = np.array([[0.0, 0.0], [400.0, 0.0], [430.0, 350.0], [-30.0, 320.0]], dtype=np.float32)
IMAGE = np.array([[180.0, 700.0], [520.0, 690.0], [470.0, 330.0], [150.0, 340.0]], dtype=np.float32)
FRAME = (720, 960)  # width, height


def make() -> Calibration:
    return Calibration(image_points=IMAGE, world_points_cm=WORLD, frame_size=FRAME)


def test_the_marked_corners_map_to_their_measured_positions() -> None:
    calibration = make()
    for pixel, world in zip(IMAGE, WORLD):
        assert calibration.to_world(*pixel) == pytest.approx(tuple(world), abs=0.5)
    assert calibration.check_cm() < 0.5


def test_an_interior_point_round_trips_through_both_directions() -> None:
    """Anything on the ground plane must survive image -> world -> image."""
    calibration = make()
    to_image = cv2.getPerspectiveTransform(WORLD, IMAGE)

    for world_point in ([200.0, 175.0], [50.0, 300.0], [380.0, 60.0]):
        source = np.array([[world_point]], dtype=np.float32)
        pixel = cv2.perspectiveTransform(source, to_image)[0, 0]
        assert calibration.to_world(*pixel) == pytest.approx(world_point, abs=0.5)


def test_ground_distances_are_exact_on_the_plane() -> None:
    """This is the claim the project actually relies on."""
    calibration = make()
    to_image = cv2.getPerspectiveTransform(WORLD, IMAGE)

    first, second = [0.0, 0.0], [300.0, 400.0]  # exactly 500 cm apart
    pixels = cv2.perspectiveTransform(
        np.array([[first, second]], dtype=np.float32), to_image
    )[0]

    assert calibration.ground_distance_cm(pixels[0], pixels[1]) == pytest.approx(500.0, abs=1.0)


def test_scale_varies_across_the_frame() -> None:
    """Perspective means one pixel is not one fixed number of centimetres."""
    calibration = make()
    near = calibration.cm_per_pixel_at(350.0, 690.0)   # bottom: close to the camera
    far = calibration.cm_per_pixel_at(310.0, 340.0)    # top: further away

    assert far > near * 1.2


def test_scaling_to_a_downscaled_frame_keeps_measurements_consistent() -> None:
    """The monitor shrinks frames for speed; the calibration must follow."""
    calibration = make()
    half = calibration.scaled_to((FRAME[1] // 2, FRAME[0] // 2, 3))

    for pixel, world in zip(IMAGE, WORLD):
        halved = (pixel[0] / 2.0, pixel[1] / 2.0)
        assert half.to_world(*halved) == pytest.approx(tuple(world), abs=1.0)


def test_scaling_to_the_same_size_is_a_no_op() -> None:
    calibration = make()
    assert calibration.scaled_to((FRAME[1], FRAME[0], 3)) is calibration


def test_height_estimation_is_not_offered() -> None:
    """Deliberately absent, not forgotten.

    A ground-plane homography cannot measure anything standing up out of that
    plane. The version that tried reported a 186 cm adult as 154-204 cm depending
    on where he stood, and called a 60 cm dog "child-sized". In a safety system a
    confident wrong number is worse than no number, because it invites decisions
    the measurement cannot support.
    """
    calibration = make()
    assert not hasattr(calibration, "estimate_height_cm")
    assert not hasattr(calibration, "calibrate_height_factor")


def test_speed_is_reported_in_metres_per_second() -> None:
    """Build the velocity from a real 1 m ground step, not from a pixel scale.

    Perspective scale is anisotropic: centimetres per pixel sideways is not the
    same as upwards at the same spot, so deriving a horizontal velocity from the
    vertical scale would test the wrong thing.
    """
    calibration = make()
    to_image = cv2.getPerspectiveTransform(WORLD, IMAGE)

    start_world, end_world = [150.0, 150.0], [250.0, 150.0]  # exactly 1 m apart
    pixels = cv2.perspectiveTransform(
        np.array([[start_world, end_world]], dtype=np.float32), to_image
    )[0]
    velocity = pixels[1] - pixels[0]  # pixels travelled in one second

    assert calibration.speed_mps(tuple(pixels[0]), velocity) == pytest.approx(1.0, abs=0.02)


def test_the_same_pixel_velocity_means_more_speed_further_away() -> None:
    """A subject at the far side of the zone covers more ground per pixel."""
    calibration = make()
    velocity = np.array([20.0, 0.0])

    near = calibration.speed_mps((350.0, 685.0), velocity)
    far = calibration.speed_mps((310.0, 340.0), velocity)
    assert far > near


MEASUREMENTS = {
    "side_1_2": 309.0, "side_2_3": 452.0, "side_3_4": 340.0, "side_4_1": 457.0,
    "diagonal_1_3": 540.0, "diagonal_2_4": 579.0,
}


def test_from_settings_builds_a_calibration_from_the_ini_values() -> None:
    calibration = from_settings(IMAGE, MEASUREMENTS, FRAME)
    assert calibration is not None
    assert calibration.check_cm() < 0.5


def test_from_settings_returns_none_when_the_numbers_are_unusable() -> None:
    """Real units are a bonus; the pool must go on being watched regardless."""
    no_diagonal = {k: v for k, v in MEASUREMENTS.items() if not k.startswith("diagonal")}
    assert from_settings(IMAGE, no_diagonal, FRAME) is None
    assert from_settings(IMAGE, {}, FRAME) is None
    assert from_settings(IMAGE, {**MEASUREMENTS, "side_1_2": 10.0}, FRAME) is None


def test_build_from_a_solved_plan() -> None:
    plan = solve_quad(ab=309, bc=452, cd=340, da=457, ac=540, bd=579)
    calibration = Calibration.from_plan(IMAGE, plan, frame_size=FRAME)

    assert calibration.check_cm() < 0.5
    # The measured side must come back out of the calibration.
    ab = calibration.ground_distance_cm(IMAGE[0], IMAGE[1])
    assert ab == pytest.approx(309.0, abs=2.0)


def test_save_and_load_round_trip(tmp_path) -> None:
    calibration = make()
    path = tmp_path / "calibration.json"
    calibration.save(path)

    loaded = Calibration.load(path)
    assert loaded.to_world(*IMAGE[2]) == pytest.approx(calibration.to_world(*IMAGE[2]), abs=1e-6)
