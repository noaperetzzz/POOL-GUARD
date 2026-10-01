"""Image I/O that survives Hebrew paths, and frames displayed inside tkinter.

Two Windows-specific traps live here.

**`cv2.imread` and `cv2.imwrite` cannot handle non-ASCII paths on Windows.** They
hand the path to the C++ layer as bytes in the system codepage, so
``C:/tmp/גיהוץ.png`` simply fails to open -- and it fails *quietly*, returning
None rather than raising, which is how a perfectly valid photo ends up looking
like a corrupt file. Reading the bytes in Python and decoding them in memory
avoids the filesystem call entirely.

**Showing a NumPy frame in tkinter** normally means depending on Pillow. Tk 8.6
reads PNG natively from base64 data, so encoding the frame in memory does the job
with nothing extra installed.
"""

from __future__ import annotations

import base64
import tkinter as tk
from pathlib import Path

import cv2
import numpy as np


def read_image(path: str | Path) -> np.ndarray | None:
    """Read an image from any path, including one with non-ASCII characters."""
    path = Path(path)
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def write_image(path: str | Path, image: np.ndarray, quality: int = 92) -> bool:
    """Write an image to any path, including one with non-ASCII characters."""
    path = Path(path)
    extension = path.suffix or ".jpg"
    params = [cv2.IMWRITE_JPEG_QUALITY, int(quality)] if extension.lower() in {".jpg", ".jpeg"} else []
    ok, buffer = cv2.imencode(extension, image, params)
    if not ok:
        return False
    try:
        buffer.tofile(str(path))
    except OSError:
        return False
    return True


import threading

class ThreadedCamera:
    """Wraps cv2.VideoCapture in a background thread to prevent FFmpeg buffer lag."""
    def __init__(self, source: str | int):
        self.cap = cv2.VideoCapture(source)
        if isinstance(source, int) or str(source).startswith(("rtsp://", "http://", "https://")):
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        self.ok = False
        self.frame = None
        self.running = False
        self.new_frame_ready = False
        self._cond = threading.Condition()
        
        if self.cap.isOpened():
            self.ok, self.frame = self.cap.read()
            self.running = True
            self.thread = threading.Thread(target=self._update, daemon=True)
            self.thread.start()

    def _update(self):
        while self.running:
            ok, frame = self.cap.read()
            with self._cond:
                self.ok = ok
                self.frame = frame
                self.new_frame_ready = True
                self._cond.notify_all()
            if not ok:
                break

    def grab(self):
        with self._cond:
            while not self.new_frame_ready and self.running and self.ok:
                self._cond.wait()
            self.new_frame_ready = False
            return self.ok

    def read(self):
        with self._cond:
            while not self.new_frame_ready and self.running and self.ok:
                self._cond.wait()
            self.new_frame_ready = False
            return self.ok, (self.frame.copy() if self.frame is not None else None)

    def isOpened(self):
        return self.cap.isOpened()

    def release(self):
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=1.0)
        self.cap.release()

    def get(self, propId):
        return self.cap.get(propId)

    def set(self, propId, value):
        return self.cap.set(propId, value)


def open_video(path: str | Path):
    """Open a video file or a camera index, tolerating non-ASCII paths."""
    text = str(path)
    
    # Dramatically reduce delay for IP cameras (RTSP/HTTP streams)
    is_stream = text.startswith(("rtsp://", "http://", "https://"))
    if is_stream:
        import os
        # Add analyzeduration and probesize to stop FFmpeg from buffering 2 seconds initially
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|analyzeduration;0|probesize;32"
        
    if text.isdigit():
        return ThreadedCamera(int(text))
    elif is_stream:
        return ThreadedCamera(text)
    else:
        return cv2.VideoCapture(text)


def fit_scale(frame: np.ndarray, max_width: int, max_height: int) -> float:
    """Scale factor that fits a frame inside a box without enlarging it."""
    height, width = frame.shape[:2]
    return min(max_width / width, max_height / height, 1.0)


def to_photo_image(frame: np.ndarray) -> tk.PhotoImage:
    """Convert a BGR frame into something a tkinter widget can display.

    The caller must keep a reference to the returned image. A `PhotoImage` that
    goes out of scope is garbage collected while Tk is still pointing at it, and
    the widget silently goes blank -- the single most common tkinter surprise.
    """
    ok, buffer = cv2.imencode(".png", frame)
    if not ok:
        raise ValueError("could not encode the frame for display")
    return tk.PhotoImage(data=base64.b64encode(buffer.tobytes()))
