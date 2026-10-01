"""Raising the alarm: sound in the house, a still image on disk, a line in the log.

Sound runs on a daemon thread. A siren that blocks the main loop would freeze
the video for its whole duration, which is exactly the moment you most want to
keep watching -- and would also stall detection on the frames that matter.

Telegram delivery to a phone is included but off by default; it needs a bot
token and turns a self-contained local tool into one that talks to the internet,
which should be an explicit choice rather than a default.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .imaging import write_image
from .monitor import Event, Severity
from .paths import events_dir

try:
    import winsound  # Windows only, part of the standard library

    _HAS_WINSOUND = True
except ImportError:  # pragma: no cover - non-Windows
    _HAS_WINSOUND = False


# (frequency Hz, duration ms) pairs. The alarm pattern alternates two tones like
# a smoke detector -- deliberately unpleasant and audible from another room. The
# notice is two soft blips that say "look at the screen", not "run".
ALARM_PATTERN = [(1400, 300), (1000, 300)] * 6
NOTICE_PATTERN = [(880, 120), (0, 80), (880, 120)]


def _clock(seconds: float | None) -> str | None:
    """Seconds into a recording as m:ss, the form you can scrub to."""
    if seconds is None:
        return None
    return f"{int(seconds) // 60:d}:{int(seconds) % 60:02d}"


def _play(pattern: list[tuple[int, int]]) -> None:
    if not _HAS_WINSOUND:  # pragma: no cover - non-Windows fallback
        print("\a", end="", flush=True)
        return
    for frequency, duration in pattern:
        if frequency <= 0:
            winsound.Beep(37, duration)  # 37 Hz is the lowest Beep accepts: a pause
        else:
            winsound.Beep(frequency, duration)


@dataclass
class AlarmSink:
    """Turns `Event` objects into sound, snapshots and log lines.

    Args:
        sound: play audio at all. Turn off for headless test runs.
        telegram_token: bot token; when set together with `telegram_chat_id`,
            each event's snapshot is pushed to that chat.
        telegram_chat_id: destination chat for Telegram delivery.
    """

    sound: bool = True
    telegram_token: str | None = None
    telegram_chat_id: str | None = None

    def __post_init__(self) -> None:
        self.log_path = events_dir() / "log.jsonl"
        self._sound_thread: threading.Thread | None = None

    def handle(self, event: Event, frame: np.ndarray) -> Path | None:
        """React to one event. Returns the snapshot path, or None if it failed."""
        snapshot = self._save_snapshot(event, frame)
        self._append_log(event, snapshot)
        if self.sound:
            self._beep(event.severity)
        if self.telegram_token and self.telegram_chat_id and snapshot:
            self._send_telegram(event, snapshot)
        return snapshot

    def _beep(self, severity: Severity) -> None:
        """Start the sound pattern without blocking the video loop.

        If a siren is already sounding, let it finish rather than stacking
        threads that would fight over the sound device.
        """
        if self._sound_thread is not None and self._sound_thread.is_alive():
            return
        pattern = ALARM_PATTERN if severity is Severity.ALARM else NOTICE_PATTERN
        self._sound_thread = threading.Thread(target=_play, args=(pattern,), daemon=True)
        self._sound_thread.start()

    def _save_snapshot(self, event: Event, frame: np.ndarray) -> Path | None:
        """Write the frame that triggered the event, with the box drawn on it.

        The snapshot is the whole point of the notice tier: it lets the user
        settle an ambiguous case in one glance instead of running outside.
        """
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = events_dir() / f"{stamp}_{event.severity.value}_{event.label}_{event.track_id}.jpg"

        annotated = frame.copy()
        height, width = annotated.shape[:2]
        x1, y1, x2, y2 = (int(round(v)) for v in event.box)
        colour = (0, 0, 255) if event.severity is Severity.ALARM else (0, 200, 255)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, 3)

        caption = f"{event.severity.value.upper()}: {event.label} {event.label_confidence:.0%}"
        font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
        (text_w, text_h), _ = cv2.getTextSize(caption, font, scale, thickness)
        # Keep the caption inside the frame. Anchoring it to the box's left edge
        # pushes it off-screen for a subject near the right border -- which is
        # exactly where an intruder entering from the side tends to be.
        text_x = max(4, min(x1, width - text_w - 4))
        text_y = y1 - 10 if y1 - 10 > text_h else min(height - 4, y2 + text_h + 8)
        cv2.putText(annotated, caption, (text_x, text_y), font, scale, colour, thickness, cv2.LINE_AA)

        # write_image, not cv2.imwrite: the executable may well be sitting in a
        # folder whose name is not ASCII, and cv2.imwrite fails silently there.
        return path if write_image(path, annotated) else None

    def _append_log(self, event: Event, snapshot: Path | None) -> None:
        """Append one JSON line per event: greppable, and easy to replay later."""
        record = {
            "time": datetime.now().isoformat(timespec="seconds"),
            "video_time": _clock(event.video_seconds),
            "severity": event.severity.value,
            "label": event.label,
            "track_id": event.track_id,
            "label_confidence": round(event.label_confidence, 3),
            "reason": event.reason,
            "box": [round(v, 1) for v in event.box],
            "snapshot": snapshot.name if snapshot else None,
        }
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _send_telegram(self, event: Event, snapshot: Path) -> None:
        """Push the snapshot to a phone. Never allowed to break the video loop."""

        def worker() -> None:
            try:
                import urllib.request
                import uuid

                boundary = uuid.uuid4().hex
                caption = f"{event.severity.value.upper()}: {event.reason}"
                photo_data = snapshot.read_bytes()
                filename = snapshot.name

                lines: list[bytes] = []
                # chat_id field
                lines.append(f"--{boundary}".encode())
                lines.append(b'Content-Disposition: form-data; name="chat_id"')
                lines.append(b"")
                lines.append(str(self.telegram_chat_id).encode())
                # caption field
                lines.append(f"--{boundary}".encode())
                lines.append(b'Content-Disposition: form-data; name="caption"')
                lines.append(b"")
                lines.append(caption.encode("utf-8"))
                # photo file
                lines.append(f"--{boundary}".encode())
                lines.append(
                    f'Content-Disposition: form-data; name="photo"; filename="{filename}"'.encode()
                )
                lines.append(b"Content-Type: image/jpeg")
                lines.append(b"")
                lines.append(photo_data)
                # closing boundary
                lines.append(f"--{boundary}--".encode())
                lines.append(b"")

                body = b"\r\n".join(lines)
                req = urllib.request.Request(
                    f"https://api.telegram.org/bot{self.telegram_token}/sendPhoto",
                    data=body,
                    headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
                )
                urllib.request.urlopen(req, timeout=10)  # noqa: S310
            except Exception as error:  # network trouble must not stop monitoring
                print(f"[alarm] telegram delivery failed: {error}")

        threading.Thread(target=worker, daemon=True).start()

