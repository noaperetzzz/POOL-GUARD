"""Build PoolGuard.exe with PyInstaller.

Usage:
    python build_exe.py            # one self-contained file in dist/
    python build_exe.py --onedir   # a folder, which starts noticeably faster

The detector weights are bundled inside the executable, so the result is a single
file you can copy anywhere. Note the split enforced in `pool_guard/paths.py`:
bundled resources are read from the temporary unpack directory, while `zone.json`
and the `events/` folder are written next to the executable, where they survive.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "models" / "yolox_s.onnx"

# SciPy is used for optimal detection-to-track assignment, but `tracker.py` has a
# greedy fallback that is close enough at these thresholds -- and leaving SciPy
# out saves upwards of 60 MB. matplotlib and pytest are development-only.
EXCLUDES = ["scipy", "matplotlib", "pytest", "imageio", "PIL", "pandas"]


PRESERVE = ("poolguard.ini", "zone.json", "calibration.json", "events")


def _preserve(dist: Path) -> dict[str, Path]:
    """Move the user's own files out of dist/ before it is wiped."""
    if not dist.exists():
        return {}
    stash = ROOT / ".build-keep"
    if stash.exists():
        shutil.rmtree(stash)
    stash.mkdir()

    kept: dict[str, Path] = {}
    for name in PRESERVE:
        source = dist / name
        if not source.exists():
            continue
        target = stash / name
        shutil.move(str(source), str(target))
        kept[name] = target
    return kept


def _restore(kept: dict[str, Path], dist: Path) -> list[str]:
    """Put them back once the new executable is in place."""
    restored = []
    for name, stashed in kept.items():
        dist.mkdir(parents=True, exist_ok=True)
        destination = dist / name
        if destination.exists():
            continue
        shutil.move(str(stashed), str(destination))
        restored.append(name)
    stash = ROOT / ".build-keep"
    if stash.exists():
        shutil.rmtree(stash)
    return restored


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onedir", action="store_true", help="build a folder instead of one file")
    parser.add_argument("--name", default="PoolGuard")
    parser.add_argument(
        "--console", action="store_true",
        help="keep a console window: ugly, but the only way to see a startup failure",
    )
    args = parser.parse_args(argv)

    if not MODEL.exists():
        print(f"model not found at {MODEL}\nRun: python scripts/download_model.py", file=sys.stderr)
        return 1

    running = _running_instances(args.name)
    if running:
        print(
            f"{args.name}.exe is running (pid {', '.join(running)}). Close it and "
            "try again -- Windows will not let the file be replaced while it is open.",
            file=sys.stderr,
        )
        return 1

    # The user's settings and saved events live in dist/, next to the executable.
    # Deleting dist/ wholesale on every rebuild threw away their marked zone and
    # every alarm snapshot, so each new build silently asked them to mark the pool
    # again. Keep those aside and put them back.
    #
    # Everything from here on sits inside try/finally: an earlier version restored
    # them only on the two paths it thought about, so when the wipe itself failed
    # the files were left stranded in the stash -- the very outcome this exists to
    # prevent.
    keep = _preserve(ROOT / "dist")
    try:
        return _build(args, keep)
    finally:
        restored = _restore(keep, ROOT / "dist")
        if restored:
            print(f"kept your existing {', '.join(restored)}")


def _running_instances(name: str) -> list[str]:
    """Process ids of any running build of this executable."""
    if sys.platform != "win32":
        return []
    try:
        output = subprocess.run(
            ["tasklist", "/FI", f"IMAGENAME eq {name}.exe", "/FO", "CSV", "/NH"],
            capture_output=True, text=True, timeout=20,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return [
        line.split('","')[1]
        for line in output.splitlines()
        if line.lower().startswith(f'"{name.lower()}.exe"')
    ]


def _build(args, keep: dict[str, Path]) -> int:
    for stale in (ROOT / "build", ROOT / "dist", ROOT / f"{args.name}.spec"):
        if stale.is_dir():
            shutil.rmtree(stale)
        elif stale.exists():
            stale.unlink()

    command = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm", "--clean",
        "--onedir" if args.onedir else "--onefile",
        "--name", args.name,
        # No console by default. The window shows the event log, so a black
        # terminal behind it is pure noise -- and OpenCV writes warnings there
        # that look alarming and mean nothing to whoever is using the thing.
        "--console" if args.console else "--windowed",
        "--add-data", f"{MODEL}{';' if sys.platform == 'win32' else ':'}models",
        "--collect-submodules", "pool_guard",
    ]
    for module in EXCLUDES:
        command += ["--exclude-module", module]
    # The entry point must be the launcher outside the package, not
    # pool_guard/main.py -- see the docstring in poolguard_app.py.
    command.append(str(ROOT / "poolguard_app.py"))

    print(" ".join(command), "\n")
    result = subprocess.run(command, cwd=ROOT)
    if result.returncode != 0:
        _restore(keep, ROOT / "dist")  # a failed build must not cost them their zone
        return result.returncode

    restored = _restore(keep, ROOT / "dist")
    if restored:
        print(f"kept your existing {', '.join(restored)}")

    built = ROOT / "dist" / (args.name if args.onedir else f"{args.name}.exe")
    if args.onedir:
        built = built / f"{args.name}.exe"
    size = built.stat().st_size / 1e6 if built.exists() else 0
    print(f"\nbuilt {built}  ({size:.0f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
