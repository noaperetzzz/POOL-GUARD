"""Tests for image I/O on awkward paths, and for tkinter display conversion."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from pool_guard.imaging import fit_scale, read_image, write_image

# A Hebrew filename, because that is what broke it in the field: OpenCV hands the
# path to its C++ layer in the system codepage, so on Windows this file is
# invisible to cv2.imread -- and it returns None rather than raising, so a valid
# photo looks like a corrupt one.
AWKWARD_NAMES = ["בדיקה.png", "ру́сский.png", "with space.png", "café.jpg"]


@pytest.fixture
def picture() -> np.ndarray:
    image = np.zeros((40, 60, 3), dtype=np.uint8)
    image[:, :30] = (200, 100, 50)
    return image


@pytest.mark.parametrize("name", AWKWARD_NAMES)
def test_round_trip_through_a_non_ascii_path(tmp_path, picture, name: str) -> None:
    path = tmp_path / name
    assert write_image(path, picture) is True

    loaded = read_image(path)
    assert loaded is not None
    assert loaded.shape == picture.shape


def test_a_hebrew_path_defeats_cv2_imread_but_not_ours(tmp_path, picture) -> None:
    """The regression test for the actual reported bug."""
    path = tmp_path / "גיהוץ.png"
    assert write_image(path, picture) is True

    # Not asserted as a hard failure: some builds and locales cope. The point is
    # that read_image works either way, so the app never depends on which.
    assert read_image(path) is not None


def test_reading_a_missing_file_returns_none(tmp_path) -> None:
    assert read_image(tmp_path / "nope.png") is None


def test_reading_a_file_that_is_not_an_image_returns_none(tmp_path) -> None:
    path = tmp_path / "notanimage.png"
    path.write_text("this is not a picture", encoding="utf-8")
    assert read_image(path) is None


def test_reading_an_empty_file_returns_none(tmp_path) -> None:
    path = tmp_path / "empty.png"
    path.write_bytes(b"")
    assert read_image(path) is None


def test_jpeg_quality_is_applied(tmp_path, picture) -> None:
    low = tmp_path / "low.jpg"
    high = tmp_path / "high.jpg"
    noisy = np.random.default_rng(0).integers(0, 255, (200, 200, 3), dtype=np.uint8)

    write_image(low, noisy, quality=20)
    write_image(high, noisy, quality=98)
    assert low.stat().st_size < high.stat().st_size


@pytest.mark.parametrize("shape,box,expected", [
    ((100, 200, 3), (200, 100), 1.0),      # already fits exactly
    ((200, 400, 3), (200, 100), 0.5),      # twice too big
    ((50, 50, 3), (500, 500), 1.0),        # smaller: never enlarged
    ((1024, 576, 3), (660, 470), 470 / 1024),  # a portrait phone video
])
def test_fit_scale(shape, box, expected) -> None:
    frame = np.zeros(shape, dtype=np.uint8)
    assert fit_scale(frame, *box) == pytest.approx(expected)


def test_a_scaled_frame_fits_the_box() -> None:
    frame = np.zeros((1024, 576, 3), dtype=np.uint8)
    scale = fit_scale(frame, 660, 470)
    resized = cv2.resize(frame, None, fx=scale, fy=scale)

    assert resized.shape[1] <= 660
    assert resized.shape[0] <= 470
