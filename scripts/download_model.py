"""Fetch the YOLOX-S ONNX detector into models/.

The model is 35 MB, so it is not committed to the repository. It comes from the
OpenCV Model Zoo, is exported for `cv2.dnn`, and needs neither PyTorch nor a GPU.

Usage:
    python scripts/download_model.py
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pool_guard.paths import resource_dir  # noqa: E402

URL = (
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/"
    "models/object_detection_yolox/object_detection_yolox_2022nov.onnx"
)
EXPECTED_BYTES = 35_858_002


def main() -> int:
    target = resource_dir() / "models" / "yolox_s.onnx"
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.exists() and target.stat().st_size == EXPECTED_BYTES:
        print(f"already present: {target}")
        return 0

    print(f"downloading {EXPECTED_BYTES / 1e6:.0f} MB from the OpenCV Model Zoo ...")
    urllib.request.urlretrieve(URL, target)

    size = target.stat().st_size
    if size != EXPECTED_BYTES:
        # A short file usually means an LFS pointer or an HTML error page was
        # saved instead of the model, which would fail confusingly much later.
        target.unlink()
        print(f"unexpected size {size} bytes; download failed", file=sys.stderr)
        return 1

    print(f"saved {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
