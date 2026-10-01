"""Resolve files relative to the program itself, never to the caller's cwd.

Two different directories are needed, and conflating them breaks the packaged
build in a way that is easy to miss:

* **Resources** (the detector weights) are read-only and travel inside the
  executable. PyInstaller unpacks them to a temporary folder named in
  ``sys._MEIPASS``.
* **Data** (your zone, event snapshots, the log) must be written next to the
  executable and survive. Writing it to ``sys._MEIPASS`` would appear to work
  and then vanish when the program exits, taking every saved alarm with it.

Running from source, both are simply the project root.
"""

from __future__ import annotations

import sys
from pathlib import Path


def is_frozen() -> bool:
    """True when running from a PyInstaller-built executable."""
    return getattr(sys, "frozen", False)


def resource_dir() -> Path:
    """Where files bundled with the program live (read-only when frozen)."""
    bundled = getattr(sys, "_MEIPASS", None)
    if bundled:
        return Path(bundled)
    return Path(__file__).resolve().parent.parent


def data_dir() -> Path:
    """Where the program writes: alongside the executable, or the project root."""
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def resource(*parts: str) -> Path:
    """Path to a file shipped with the program, e.g. resource("models", "x.onnx")."""
    return resource_dir().joinpath(*parts)


def model_path() -> Path:
    """The detector weights.

    Bundled inside the executable, but a copy placed next to it wins, so the
    model can be swapped for a bigger one without rebuilding.
    """
    beside_exe = data_dir() / "models" / "yolox_s.onnx"
    if beside_exe.exists():
        return beside_exe
    return resource_dir() / "models" / "yolox_s.onnx"


def zone_path() -> Path:
    """The saved zone, kept next to the program."""
    return data_dir() / "zone.json"


def calibration_path() -> Path:
    """The saved real-world calibration, kept next to the program."""
    return data_dir() / "calibration.json"


def events_dir() -> Path:
    """Where snapshots and the event log are written. Created on demand."""
    path = data_dir() / "events"
    path.mkdir(parents=True, exist_ok=True)
    return path
