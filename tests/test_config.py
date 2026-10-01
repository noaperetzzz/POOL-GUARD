"""Tests for the INI settings file next to the program."""

from __future__ import annotations

import pytest

from pool_guard import config
from pool_guard.monitor import Severity


def test_the_file_is_created_with_defaults_on_first_run(tmp_path) -> None:
    path = tmp_path / "poolguard.ini"
    assert not path.exists()

    settings = config.load(path)

    assert path.exists()
    assert settings.dwell_seconds == 0.5
    assert settings.frame_stride == 6
    assert settings.class_actions["person"] is Severity.ALARM


def test_the_template_explains_itself(tmp_path) -> None:
    """Somebody is going to open this file with no documentation to hand."""
    path = config.ensure_file(tmp_path / "poolguard.ini")
    text = path.read_text(encoding="utf-8")

    assert text.count(";") > 8, "every setting should carry a comment"
    for section in ("[alarm]", "[detector]", "[classes]", "[measurements]"):
        assert section in text


def test_editing_a_value_changes_the_policy(tmp_path) -> None:
    path = tmp_path / "poolguard.ini"
    config.ensure_file(path)
    path.write_text(
        "[alarm]\ndwell_seconds = 3.0\ngrace_seconds = 2.5\nsound = no\n"
        "[detector]\nframe_stride = 3\n",
        encoding="utf-8",
    )

    settings = config.load(path)
    policy = settings.policy()

    assert policy.dwell_seconds == pytest.approx(3.0)
    assert policy.grace_seconds == pytest.approx(2.5)
    assert settings.sound is False
    assert settings.frame_stride == 3


def test_class_actions_can_be_retuned(tmp_path) -> None:
    path = tmp_path / "poolguard.ini"
    path.write_text("[classes]\ndog = alarm\ncat = notice\n", encoding="utf-8")

    actions = config.load(path).class_actions
    assert actions["dog"] is Severity.ALARM
    assert actions["cat"] is Severity.NOTICE
    assert actions["person"] is Severity.ALARM  # untouched, so still the default


def test_nonsense_values_fall_back_instead_of_crashing(tmp_path) -> None:
    """A typo in a settings file must never stop the pool being watched."""
    path = tmp_path / "poolguard.ini"
    path.write_text(
        "[alarm]\ndwell_seconds = soon\ngrace_seconds = \n"
        "[detector]\nframe_stride = -4\n"
        "[classes]\ndog = maybe\n",
        encoding="utf-8",
    )

    settings = config.load(path)

    assert settings.dwell_seconds == 0.5
    assert settings.grace_seconds == 8.0
    assert settings.frame_stride == 1  # clamped, never zero or negative
    assert settings.class_actions["dog"] is Severity.NOTICE


def test_a_completely_corrupt_file_still_yields_defaults(tmp_path) -> None:
    path = tmp_path / "poolguard.ini"
    path.write_text("this is not an ini file at all\n===\n", encoding="utf-8")

    assert config.load(path).dwell_seconds == 0.5


CORNERS = [(0.10, 0.62), (0.95, 0.61), (0.75, 0.32), (0.21, 0.35)]


def test_a_saved_zone_comes_back_on_the_next_start(tmp_path) -> None:
    """The whole point: the same pool is never marked twice."""
    path = tmp_path / "poolguard.ini"
    config.ensure_file(path)

    config.save_zone(CORNERS, source=r"c:\tmp\pool.mp4", path=path)
    settings = config.load(path)

    assert settings.has_zone
    assert settings.source == r"c:\tmp\pool.mp4"
    for saved, loaded in zip(CORNERS, settings.zone_points):
        assert loaded[0] == pytest.approx(saved[0], abs=1e-4)
        assert loaded[1] == pytest.approx(saved[1], abs=1e-4)


def test_a_fresh_file_has_no_zone(tmp_path) -> None:
    settings = config.load(tmp_path / "poolguard.ini")
    assert settings.has_zone is False
    assert settings.source is None


def test_saving_a_zone_keeps_the_other_settings(tmp_path) -> None:
    path = tmp_path / "poolguard.ini"
    config.ensure_file(path)
    config.save_zone(CORNERS, source="0", path=path)

    settings = config.load(path)
    assert settings.dwell_seconds == 0.5
    assert settings.frame_stride == 6
    assert settings.source == "0"


def test_a_partly_written_zone_is_ignored(tmp_path) -> None:
    """Three corners cannot define the area, so it must not be half-loaded."""
    path = tmp_path / "poolguard.ini"
    path.write_text(
        "[zone]\npoint_1 = 0.1,0.6\npoint_2 = 0.9,0.6\npoint_3 = 0.8,0.3\npoint_4 =\n",
        encoding="utf-8",
    )
    assert config.load(path).has_zone is False


def test_a_corrupt_corner_is_ignored(tmp_path) -> None:
    path = tmp_path / "poolguard.ini"
    path.write_text(
        "[zone]\npoint_1 = 0.1,0.6\npoint_2 = somewhere\n"
        "point_3 = 0.8,0.3\npoint_4 = 0.2,0.35\n",
        encoding="utf-8",
    )
    assert config.load(path).has_zone is False


def test_a_path_containing_a_percent_sign_survives(tmp_path) -> None:
    """configparser reads % as a reference to another setting unless told not to,
    and a filename is perfectly entitled to contain one."""
    path = tmp_path / "poolguard.ini"
    config.ensure_file(path)
    awkward = r"c:\videos\100%_pool.mp4"

    config.save_zone(CORNERS, source=awkward, path=path)
    assert config.load(path).source == awkward


FULL_MEASUREMENTS = {
    "side_1_2": 309.0, "side_2_3": 452.0, "side_3_4": 340.0, "side_4_1": 457.0,
    "diagonal_1_3": 540.0, "diagonal_2_4": 579.0,
}


def test_moving_the_zone_discards_the_measurements(tmp_path) -> None:
    """They described the old corners. Applying them to a new area would report
    speeds that are wrong, with nothing on screen to suggest it."""
    path = tmp_path / "poolguard.ini"
    config.save_zone(CORNERS, source="a.mp4", path=path)
    config.save_measurements(FULL_MEASUREMENTS, path)
    assert config.load(path).has_measurements

    elsewhere = [(0.2, 0.7), (0.8, 0.7), (0.7, 0.4), (0.3, 0.4)]
    discarded = config.save_zone(elsewhere, source="b.mp4", path=path)

    assert discarded is True
    settings = config.load(path)
    assert settings.has_measurements is False
    assert settings.measurements == {}
    assert settings.source == "b.mp4"


def test_re_saving_the_same_zone_keeps_the_measurements(tmp_path) -> None:
    """Pressing Start again on an unchanged pool must not throw away the survey."""
    path = tmp_path / "poolguard.ini"
    config.save_zone(CORNERS, source="a.mp4", path=path)
    config.save_measurements(FULL_MEASUREMENTS, path)

    discarded = config.save_zone(CORNERS, source="a.mp4", path=path)

    assert discarded is False
    assert config.load(path).has_measurements is True


def test_moving_the_zone_with_no_measurements_reports_nothing_discarded(tmp_path) -> None:
    path = tmp_path / "poolguard.ini"
    config.save_zone(CORNERS, source="a.mp4", path=path)

    elsewhere = [(0.2, 0.7), (0.8, 0.7), (0.7, 0.4), (0.3, 0.4)]
    assert config.save_zone(elsewhere, source="b.mp4", path=path) is False


def test_switching_pools_replaces_the_whole_zone(tmp_path) -> None:
    path = tmp_path / "poolguard.ini"
    config.save_zone(CORNERS, source=r"c:\pool_a.mp4", path=path)

    pool_b = [(0.05, 0.90), (0.90, 0.88), (0.70, 0.20), (0.15, 0.25)]
    config.save_zone(pool_b, source=r"c:\pool_b.mp4", path=path)

    settings = config.load(path)
    assert settings.source == r"c:\pool_b.mp4"
    for saved, loaded in zip(pool_b, settings.zone_points):
        assert loaded[0] == pytest.approx(saved[0], abs=1e-4)


def test_measurements_are_read_and_reported_as_complete(tmp_path) -> None:
    path = tmp_path / "poolguard.ini"
    path.write_text(
        "[measurements]\nside_1_2 = 309\nside_2_3 = 452\nside_3_4 = 340\n"
        "side_4_1 = 457\ndiagonal_1_3 = 540\ndiagonal_2_4 = 579\n",
        encoding="utf-8",
    )

    settings = config.load(path)
    assert settings.has_measurements
    assert settings.measurements["side_2_3"] == pytest.approx(452.0)


def test_measurements_without_a_diagonal_are_not_complete(tmp_path) -> None:
    """Four sides cannot define a quadrilateral, so this must not be accepted."""
    path = tmp_path / "poolguard.ini"
    path.write_text(
        "[measurements]\nside_1_2 = 309\nside_2_3 = 452\nside_3_4 = 340\nside_4_1 = 457\n",
        encoding="utf-8",
    )

    assert config.load(path).has_measurements is False


def test_blank_measurements_are_simply_absent(tmp_path) -> None:
    """The template ships with them blank, and that has to mean 'work in pixels'."""
    path = config.ensure_file(tmp_path / "poolguard.ini")
    settings = config.load(path)

    assert settings.measurements == {}
    assert settings.has_measurements is False


def test_a_comma_decimal_separator_is_accepted(tmp_path) -> None:
    path = tmp_path / "poolguard.ini"
    path.write_text("[measurements]\nside_1_2 = 309,5\n", encoding="utf-8")

    assert config.load(path).measurements["side_1_2"] == pytest.approx(309.5)


def test_saving_measurements_keeps_the_rest_of_the_file(tmp_path) -> None:
    path = tmp_path / "poolguard.ini"
    config.ensure_file(path)

    config.save_measurements({"side_1_2": 309, "diagonal_1_3": 540}, path)
    settings = config.load(path)

    assert settings.measurements["side_1_2"] == pytest.approx(309.0)
    assert settings.dwell_seconds == 0.5  # the alarm section survived the rewrite
