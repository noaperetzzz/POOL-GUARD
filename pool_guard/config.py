"""Settings in an INI file next to the program, created on first run.

Everything worth adjusting without opening the code lives here: how long someone
must be inside the zone before it counts, the exit delay, what to do about a dog,
whether the siren sounds. The file is written with its defaults the first time the
program starts, so there is always something to look at and edit.

The optional ``[measurements]`` section holds the real distances between the four
corners. Fill it in and speeds are reported in metres per second; leave it empty
and everything works exactly as before, in pixels. Note what is *not* offered:
a subject's height. See `calibration.py` for why a ground-plane calibration
cannot measure it and why guessing was worse than declining.
"""

from __future__ import annotations

import configparser
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .monitor import Policy, Severity
from .paths import data_dir

FILENAME = "poolguard.ini"
CORNER_KEYS = ("point_1", "point_2", "point_3", "point_4")
MEASUREMENT_KEYS = (
    "side_1_2", "side_2_3", "side_3_4", "side_4_1", "diagonal_1_3", "diagonal_2_4",
)


def _parser() -> configparser.ConfigParser:
    """A parser that tolerates real-world values.

    Interpolation is off: it treats ``%`` as a reference to another setting.
    """
    return configparser.ConfigParser(inline_comment_prefixes=(";", "#"), interpolation=None)

TEMPLATE = """\
; Pool Guard settings. Edit and restart. Delete this file to get the defaults back.

[alarm]
; How long a subject must stay inside the zone, in SECONDS, before it is
; reported. The single most effective control on false alarms. In seconds and
; not frames on purpose: counted in frames it would silently grow six times
; longer whenever frame_stride below changes.
dwell_seconds = 0.5
; Seconds of quiet after closing the pool, so swimmers can climb out.
grace_seconds = 8
; Minimum gap in seconds between two reports of the same subject.
cooldown_seconds = 20
; Play a sound. Snapshots and the log are written either way.
sound = yes
; Optional: Telegram bot token from @BotFather. Leave blank to disable.
telegram_token =
; Optional: Telegram chat id to send alerts to. Leave blank to disable.
telegram_chat_id =

[detector]
; Lower finds more and reports more rubbish; the dwell time and the track
; voting clean that up, and a miss matters far more than a false alarm.
confidence = 0.35
; Downscale frames to this width before detection. Smaller is faster.
width = 960
; Analyse every Nth frame and throw the rest away. This is what keeps the video
; playing at real speed: on a 30 fps source, 6 means five analysed frames a
; second, which is about what a CPU manages. Raise it on a slower machine (the
; picture updates less often but stays in real time); lower it to look more
; closely at fast movement, at the cost of the video lagging behind.
frame_stride = 6

[classes]
; What each kind of subject may raise: alarm, notice, or ignore.
; A dog can drown too, so it is worth a chime. A cat would cry wolf nightly.
person = alarm
dog = notice
cat = ignore

[zone]
; The four corners of the protected area, written here when you mark them, so
; the same pool never has to be marked twice. Each is a fraction of the frame
; width and height, in the order they were clicked -- fractions rather than
; pixels so the same zone holds at any resolution.
; Delete these four lines to start marking from scratch.
point_1 =
point_2 =
point_3 =
point_4 =
; The video file or camera index last watched, reopened on the next start.
source =

[measurements]
; Optional. Real distances between the four marked corners, in centimetres,
; in the order you clicked them. Sides plus at least one diagonal -- four sides
; alone cannot define a quadrilateral, it flexes. Fill these in for speeds in
; metres per second. Leave them blank to work in pixels.
side_1_2 =
side_2_3 =
side_3_4 =
side_4_1 =
diagonal_1_3 =
diagonal_2_4 =
"""


@dataclass
class Settings:
    """Everything read from the INI."""

    dwell_seconds: float = 0.5
    grace_seconds: float = 8.0
    cooldown_seconds: float = 20.0
    sound: bool = True
    telegram_token: str | None = None
    telegram_chat_id: str | None = None
    confidence: float = 0.35
    width: int = 960
    frame_stride: int = 6
    class_actions: dict[str, Severity] = None  # type: ignore[assignment]
    measurements: dict[str, float] = None  # type: ignore[assignment]
    zone_points: np.ndarray | None = None   # (4, 2) normalised, or None if unmarked
    source: str | None = None               # the video or camera last watched

    def __post_init__(self) -> None:
        if self.class_actions is None:
            self.class_actions = {
                "person": Severity.ALARM,
                "dog": Severity.NOTICE,
                "cat": Severity.IGNORE,
            }
        if self.measurements is None:
            self.measurements = {}

    def policy(self) -> Policy:
        return Policy(
            actions=dict(self.class_actions),
            dwell_seconds=self.dwell_seconds,
            grace_seconds=self.grace_seconds,
            cooldown_seconds=self.cooldown_seconds,
        )

    @property
    def has_zone(self) -> bool:
        """True when four usable corners were read from the file."""
        return self.zone_points is not None and len(self.zone_points) == 4

    @property
    def has_measurements(self) -> bool:
        """True when enough was filled in to fix the shape: 4 sides and a diagonal."""
        sides = ("side_1_2", "side_2_3", "side_3_4", "side_4_1")
        diagonals = ("diagonal_1_3", "diagonal_2_4")
        return all(name in self.measurements for name in sides) and any(
            name in self.measurements for name in diagonals
        )


def config_path() -> Path:
    return data_dir() / FILENAME


def ensure_file(path: Path | None = None) -> Path:
    """Write the annotated template if there is no settings file yet."""
    path = path or config_path()
    if not path.exists():
        path.write_text(TEMPLATE, encoding="utf-8")
    return path


def load(path: Path | None = None) -> Settings:
    """Read the settings, falling back to defaults for anything missing or broken.

    A malformed value must never stop the program from watching the pool, so
    every read is individually defensive rather than validated up front.
    """
    path = ensure_file(path)
    parser = _parser()
    try:
        parser.read(path, encoding="utf-8")
    except configparser.Error:
        return Settings()

    settings = Settings()

    def number(section: str, option: str, current: float) -> float:
        try:
            return parser.getfloat(section, option)
        except (configparser.Error, ValueError):
            return current

    settings.dwell_seconds = number("alarm", "dwell_seconds", settings.dwell_seconds)
    settings.grace_seconds = number("alarm", "grace_seconds", settings.grace_seconds)
    settings.cooldown_seconds = number("alarm", "cooldown_seconds", settings.cooldown_seconds)
    try:
        settings.sound = parser.getboolean("alarm", "sound")
    except (configparser.Error, ValueError):
        pass
    token = parser.get("alarm", "telegram_token", fallback="").strip()
    settings.telegram_token = token or None
    chat_id = parser.get("alarm", "telegram_chat_id", fallback="").strip()
    settings.telegram_chat_id = chat_id or None
    settings.confidence = number("detector", "confidence", settings.confidence)
    settings.width = int(number("detector", "width", settings.width))
    settings.frame_stride = max(1, int(number("detector", "frame_stride", settings.frame_stride)))

    if parser.has_section("classes"):
        for label, value in parser.items("classes"):
            try:
                settings.class_actions[label] = Severity(value.strip().lower())
            except ValueError:
                pass  # an unknown word: keep whatever the default was

    if parser.has_section("zone"):
        corners = []
        for key in CORNER_KEYS:
            text = parser.get("zone", key, fallback="").strip()
            if not text:
                break
            try:
                x, y = (float(part) for part in text.replace(";", ",").split(",")[:2])
            except ValueError:
                break
            corners.append((x, y))
        if len(corners) == len(CORNER_KEYS):
            settings.zone_points = np.array(corners, dtype=float)
        source = parser.get("zone", "source", fallback="").strip()
        settings.source = source or None

    if parser.has_section("measurements"):
        for name, value in parser.items("measurements"):
            text = value.strip().replace(",", ".")
            if not text:
                continue
            try:
                distance = float(text)
            except ValueError:
                continue
            if distance > 0:
                settings.measurements[name] = distance

    return settings


def save_measurements(values: dict[str, float], path: Path | None = None) -> None:
    """Write measurements back, preserving the rest of the file."""
    _update(path, "measurements", {name: f"{value:g}" for name, value in values.items()})


def save_zone(points, source: str | None = None, path: Path | None = None) -> bool:
    """Remember the four corners, and the source they were marked on.

    This is why the same pool never has to be marked twice: the corners live in
    the settings file, so the next start finds them already there. They are stored
    as fractions of the frame, which is what lets them hold at any resolution.

    Returns True if measurements were discarded because the area moved. Tape
    measurements describe *these four corners* and mean nothing once the corners
    are somewhere else. Leaving them in place when the zone changes would quietly
    apply the old pool's dimensions to the new one and report speeds that are
    simply wrong -- with nothing on screen to suggest it.
    """
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    previous = load(path)
    moved = not (
        previous.has_zone
        and previous.zone_points.shape == points.shape
        and np.allclose(previous.zone_points, points, atol=1e-4)
    )

    values = {
        key: f"{float(x):.5f},{float(y):.5f}"
        for key, (x, y) in zip(CORNER_KEYS, points)
    }
    if source is not None:
        values["source"] = str(source)
    _update(path, "zone", values)

    discarded = moved and previous.has_measurements
    if moved:
        _update(path, "measurements", {name: "" for name in MEASUREMENT_KEYS})
    return discarded


def _update(path: Path | None, section: str, values: dict[str, str]) -> None:
    """Set some keys in one section, leaving the rest of the file alone."""
    path = ensure_file(path)
    parser = _parser()
    parser.read(path, encoding="utf-8")
    if not parser.has_section(section):
        parser.add_section(section)
    for name, value in values.items():
        parser.set(section, name, value)
    with path.open("w", encoding="utf-8") as handle:
        parser.write(handle)
