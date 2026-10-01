"""Make the package importable when pytest is run from anywhere in the tree."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
