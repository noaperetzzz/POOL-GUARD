"""Launcher for the packaged executable.

This has to live outside the `pool_guard` package. PyInstaller runs its entry
script as `__main__` with no parent package, so using `pool_guard/main.py`
directly makes every relative import in it fail with "attempted relative import
with no known parent package". A one-line launcher that imports the package
absolutely sidesteps the whole problem.
"""

import sys

from pool_guard.main import main

if __name__ == "__main__":
    sys.exit(main())
