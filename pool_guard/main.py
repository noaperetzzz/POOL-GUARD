"""Entry point for the packaged executable.

Double-clicked, it opens the GUI window.
"""

from __future__ import annotations

import sys


def _report_crash(error: BaseException) -> None:
    """Write the traceback next to the executable and show it.

    A windowed build has no console, so an unhandled exception makes the whole
    program vanish with no explanation whatsoever -- the worst possible failure
    mode for someone who just double-clicked it. A log file and a dialog turn that
    into something reportable.
    """
    import traceback

    text = "".join(traceback.format_exception(error))
    try:
        from .paths import data_dir

        log = data_dir() / "poolguard-error.log"
        log.write_text(text, encoding="utf-8")
        location = f"\n\nWritten to:\n{log}"
    except Exception:
        location = ""

    print(text, file=sys.stderr)
    try:
        import tkinter as tk
        from tkinter import messagebox

        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Pool Guard - error", f"{text[-1500:]}{location}")
        root.destroy()
    except Exception:
        pass  # no display available; the log file is the record


def main() -> int:
    """Launch Pool Guard.

    This is the only public entry point.  It always opens the GUI window;
    there is no command-line / terminal mode.
    """
    try:
        from . import setup_gui

        return setup_gui.main()
    except SystemExit:
        raise
    except BaseException as error:  # noqa: BLE001 - last resort before vanishing
        _report_crash(error)
        return 1
