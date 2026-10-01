"""The whole application in one window: mark the zone, then watch it.

Everything happens inside this window -- scrubbing to a frame, clicking the
corners, and the live video while monitoring. There are no separate OpenCV
windows and no console.

That is a deliberate correction. An earlier version put frame picking in OpenCV
windows and hid the main window while they were open, so the interface vanished, a
black console sat behind everything, and nothing on screen said what to do next.
Mixing two window toolkits was the cause.

The reference frame comes from the video or camera being watched, and only from
there. Marking corners on a separate photograph invites the one mistake that
silently ruins everything downstream: a picture taken from a slightly different
position than the camera now sits in, so the zone is a few degrees off and nobody
can tell by looking.

Settings live in `poolguard.ini` next to the program -- see `config.py`.

The display loop runs on tkinter's `after` timer. Detection runs on its own
thread inside the engine and hands results back over a queue, so the picture stays
smooth and current while the models take as long as they take. No thread here ever
touches a widget.
"""

from __future__ import annotations

import sys
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk

import cv2
import numpy as np

from . import config
from .calibration import from_settings
from .engine import MonitorEngine
from .imaging import fit_scale, open_video, to_photo_image
from .monitor import Mode, Severity
from .paths import data_dir, events_dir, model_path
from .zone import Zone

VIDEO_TYPES = [("Video files", "*.mp4 *.mov *.avi *.mkv *.m4v"), ("All files", "*.*")]

# The largest the video area may grow to. The canvas is then resized to the
# video's own aspect ratio inside this box, so a portrait phone clip does not sit
# in the middle of a wide black rectangle.
MAX_VIDEO_W, MAX_VIDEO_H = 760, 620
CANVAS_W, CANVAS_H = 680, 480
CORNER_COUNT = 4
POINT_COLOUR = "#ffb300"
ZONE_COLOUR = "#00c8ff"


class SetupApp:
    """Zone marking and monitoring, sharing one window."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Pool Guard")

        self.settings = config.load()

        self.source: str | None = None
        self.reference: np.ndarray | None = None
        self.points: list[tuple[float, float]] = []
        self.capture: cv2.VideoCapture | None = None
        self.total_frames = 0

        self.photo: tk.PhotoImage | None = None
        self.canvas_scale = 1.0

        self.hint = tk.StringVar()
        self.source_text = tk.StringVar(value="no video chosen")
        self.engine: MonitorEngine | None = None
        self._tick_job: str | None = None

        self._build_setup()
        self._restore_previous_session()
        self._update_hint()
        self._centre()

    def _restore_previous_session(self) -> None:
        """Reopen the last source with its corners already marked.

        The whole point of keeping the zone in the settings file: on the same pool
        there is nothing to do but press Start. If the file has moved or the camera
        is not there, fall back silently to the empty setup screen -- a missing
        video is not an error worth a dialog on startup.
        """
        if not self.settings.has_zone or not self.settings.source:
            return
        source = self.settings.source
        is_stream = source.startswith(("rtsp://", "http://", "https://"))
        if not source.isdigit() and not is_stream and not Path(source).exists():
            self.source_text.set(f"{Path(source).name} — not found, choose a video")
            return

        if source.isdigit():
            self.open_webcam(remembered=True)
        elif is_stream:
            self.open_ip_camera(remembered=True, url=source)
        else:
            self._load_video(source, remembered=True)

        if self.reference is not None:
            height, width = self.reference.shape[:2]
            self.points = [
                (float(x) * width, float(y) * height) for x, y in self.settings.zone_points
            ]
            self._redraw_points()

    # --------------------------------------------------------------- setup view

    def _build_setup(self) -> None:
        self.setup_frame = ttk.Frame(self.root, padding=12)
        self.setup_frame.grid(row=0, column=0, sticky="nsew")

        ttk.Label(
            self.setup_frame,
            text="Watch a zone and raise the alarm when it should be empty.",
            font=("Segoe UI", 11, "bold"),
        ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 8))

        buttons = ttk.Frame(self.setup_frame)
        buttons.grid(row=1, column=0, columnspan=3, sticky="w", pady=(0, 8))
        ttk.Button(buttons, text="Open video...", command=self.open_video_file).pack(side="left")
        ttk.Button(buttons, text="IP Camera...", command=self.open_ip_camera).pack(side="left", padx=6)
        ttk.Label(buttons, textvariable=self.source_text, foreground="#555").pack(side="left", padx=8)

        self.canvas = tk.Canvas(
            self.setup_frame, width=CANVAS_W, height=CANVAS_H, bg="#1c1c1c",
            highlightthickness=1, highlightbackground="#999", cursor="crosshair",
        )
        self.canvas.grid(row=2, column=0, columnspan=3)
        self.canvas.bind("<Button-1>", self.on_canvas_click)

        self.scrub_row = ttk.Frame(self.setup_frame)
        self.scrub_row.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(6, 0))
        self.frame_slider = ttk.Scale(self.scrub_row, from_=0, to=100, orient="horizontal")
        self.frame_slider.bind("<ButtonRelease-1>", self.on_slider_release)
        self.frame_slider.pack(side="left", fill="x", expand=True)
        self.frame_label = ttk.Label(self.scrub_row, text="", width=20)
        self.frame_label.pack(side="left", padx=(8, 0))
        self.scrub_row.grid_remove()

        ttk.Label(
            self.setup_frame, textvariable=self.hint, foreground="#0a58ca",
            wraplength=CANVAS_W, font=("Segoe UI", 10, "bold"), justify="left",
        ).grid(row=4, column=0, columnspan=3, sticky="w", pady=(10, 6))

        actions = ttk.Frame(self.setup_frame)
        actions.grid(row=5, column=0, columnspan=3, sticky="w")
        ttk.Button(actions, text="Undo corner", command=self.undo_point).pack(side="left")
        ttk.Button(actions, text="Clear", command=self.clear_points).pack(side="left", padx=6)
        # Disabled until the corners exist. A button that greys out and then
        # lights up *is* the confirmation step -- clearer than asking for a
        # keystroke, and it cannot be missed the way small print can.
        self.start_button = ttk.Button(actions, text="Start watching", command=self.start)
        self.start_button.pack(side="left", padx=(18, 0))
        # No "use saved zone" button: the saved zone is simply already loaded.
        ttk.Button(actions, text="Settings file", command=self._open_settings).pack(side="left", padx=6)
        ttk.Button(actions, text="Quit", command=self.stop).pack(side="left", padx=6)

    def _centre(self) -> None:
        self.root.update_idletasks()
        x = max(0, (self.root.winfo_screenwidth() - self.root.winfo_width()) // 2)
        y = max(0, (self.root.winfo_screenheight() - self.root.winfo_height()) // 10)
        self.root.geometry(f"+{x}+{y}")

    def _update_hint(self) -> None:
        """Always say what the next click does, and gate the buttons to match."""
        complete = len(self.points) == CORNER_COUNT
        self.start_button.configure(state="normal" if complete else "disabled")

        if self.reference is None:
            self.hint.set("Open a video or the webcam to begin.")
        elif not complete:
            remaining = CORNER_COUNT - len(self.points)
            self.hint.set(
                f"Click corner {len(self.points) + 1} of {CORNER_COUNT} on the picture "
                f"({remaining} more to go). Go round the boundary in order."
            )
        else:
            self.hint.set("✓ All four corners set. Nothing to confirm — press Start watching.")

    # ------------------------------------------------------------ loading media

    def _show_frame(self, frame: np.ndarray) -> None:
        self.canvas_scale = fit_scale(frame, MAX_VIDEO_W, MAX_VIDEO_H)
        preview = (
            cv2.resize(frame, None, fx=self.canvas_scale, fy=self.canvas_scale)
            if self.canvas_scale < 1.0 else frame
        )
        # Shrink the canvas onto the picture. Leaving it at a fixed size wraps a
        # portrait clip in a wide black margin and wastes most of the window.
        self.canvas.configure(width=preview.shape[1], height=preview.shape[0])
        self.photo = to_photo_image(preview)  # kept on self: Tk does not own it
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo)
        self._redraw_points()
        self._centre()

    def open_video_file(self) -> None:
        path = filedialog.askopenfilename(title="Open a video", filetypes=VIDEO_TYPES)
        if path:
            self._load_video(path)

    def _load_video(self, path: str, remembered: bool = False) -> None:
        capture = open_video(path)
        if not capture.isOpened():
            if not remembered:
                messagebox.showerror("Pool Guard", f"Could not open:\n{path}")
            return

        ok, frame = capture.read()
        if not ok:
            if not remembered:
                messagebox.showerror("Pool Guard", f"Could not read a frame from:\n{path}")
            capture.release()
            return

        self._release_capture()
        self.capture = capture
        self.total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.source = path
        self.source_text.set(Path(path).name)

        if self.total_frames > 1:
            self.frame_slider.configure(to=max(1, self.total_frames - 1))
            self.frame_slider.set(0)
            self.frame_label.configure(text=f"frame 0 / {self.total_frames - 1}")
            self.scrub_row.grid()
        else:
            self.scrub_row.grid_remove()

        if not remembered:
            # A different video means the old corners refer to a different scene.
            # Reopening the remembered one is the exception: they still apply.
            self.points = []
        self.reference = frame
        self._show_frame(frame)
        self._update_hint()

    def on_slider_release(self, _event: object) -> None:
        """Load the frame the slider landed on.

        On release rather than on every movement: seeking a large file once per
        pixel of drag would make the slider crawl.
        """
        if self.capture is None:
            return
        index = int(float(self.frame_slider.get()))
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = self.capture.read()
        if not ok:
            return
        self.frame_label.configure(text=f"frame {index} / {self.total_frames - 1}")
        self.reference = frame
        # Same camera, same scene: corners stay valid across frames of one video,
        # so scrubbing to a clearer moment does not cost the marking already done.
        self._show_frame(frame)
        self._update_hint()

    def open_webcam(self, remembered: bool = False) -> None:
        capture = open_video("0")
        frame = None
        for _ in range(5):  # let the sensor settle
            ok, candidate = capture.read()
            if ok:
                frame = candidate
        capture.release()
        if frame is None:
            if not remembered:
                messagebox.showerror("Pool Guard", "Could not read from the webcam.")
            return
        self._release_capture()
        self.scrub_row.grid_remove()
        self.source = "0"
        self.source_text.set("webcam")
        if not remembered:
            self.points = []
        self.reference = frame
        self._show_frame(frame)
        self._update_hint()

    def open_ip_camera(self, remembered: bool = False, url: str | None = None) -> None:
        if url is None:
            initial = (
                self.settings.source
                if (self.settings.source and self.settings.source.startswith(("rtsp://", "http://", "https://")))
                else "rtsp://admin:123456@192.168.1.100:554/live"
            )
            url = simpledialog.askstring(
                "IP Camera",
                "Enter IP Camera RTSP URL:\n(e.g. rtsp://admin:123456@192.168.1.100:554/live)",
                initialvalue=initial,
                parent=self.root,
            )
        if not url:
            return

        url = url.strip()
        capture = open_video(url)
        frame = None
        for _ in range(10):
            ok, candidate = capture.read()
            if ok:
                frame = candidate
        capture.release()
        if frame is None:
            if not remembered:
                messagebox.showerror("Pool Guard", f"Could not connect to IP camera at:\n{url}")
            return
        self._release_capture()
        self.scrub_row.grid_remove()
        self.source = url
        display_name = url.split("@")[-1] if "@" in url else url
        if len(display_name) > 35:
            display_name = display_name[:32] + "..."
        self.source_text.set(f"IP Camera ({display_name})")
        if not remembered:
            self.points = []
        self.reference = frame
        self._show_frame(frame)
        self._update_hint()

    def _release_capture(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def _open_settings(self) -> None:
        path = config.ensure_file()
        if sys.platform == "win32":
            import os

            os.startfile(path)  # noqa: S606 - our own settings file
        else:
            messagebox.showinfo("Pool Guard", str(path))

    # ------------------------------------------------------------- corner clicks

    def on_canvas_click(self, event: tk.Event) -> None:
        if self.reference is None:
            self.hint.set("Open a video or the webcam first.")
            return
        if len(self.points) >= CORNER_COUNT:
            self.hint.set("Four corners already marked. Use Undo or Clear to change them.")
            return
        # Canvas coordinates back to full-resolution image pixels: the zone must
        # refer to the video's own pixels, not the preview's.
        self.points.append((event.x / self.canvas_scale, event.y / self.canvas_scale))
        self._redraw_points()
        self._update_hint()

    def undo_point(self) -> None:
        if self.points:
            self.points.pop()
            self._redraw_points()
            self._update_hint()

    def clear_points(self) -> None:
        self.points = []
        self._redraw_points()
        self._update_hint()

    def _redraw_points(self) -> None:
        self.canvas.delete("marker")
        if not self.points:
            return
        scaled = [(x * self.canvas_scale, y * self.canvas_scale) for x, y in self.points]
        if len(scaled) >= 2:
            closing = scaled[0] if len(scaled) >= 3 else ()
            self.canvas.create_line(
                *[c for point in scaled for c in point], *closing,
                fill=ZONE_COLOUR, width=2, tags="marker",
            )
        for index, (x, y) in enumerate(scaled, start=1):
            self.canvas.create_oval(
                x - 5, y - 5, x + 5, y + 5, fill=POINT_COLOUR, outline="", tags="marker"
            )
            self.canvas.create_text(
                x + 12, y - 10, text=str(index), fill=POINT_COLOUR,
                font=("Segoe UI", 11, "bold"), tags="marker",
            )

    # -------------------------------------------------------------------- starting

    def start(self) -> None:
        if self.reference is None or len(self.points) != CORNER_COUNT or self.source is None:
            self._update_hint()
            return

        height, width = self.reference.shape[:2]
        normalised = np.array(self.points) / np.array([width, height])
        zone = Zone(points=normalised, name="zone")

        # Written into the settings file, which is why the same pool is never
        # marked twice: the next start reopens this source with these corners
        # already in place.
        discarded = config.save_zone(normalised, source=self.source)


        if discarded:
            # Never silently: measurements describe the corners they were taken
            # against, and applying an old pool's dimensions to a new area would
            # report speeds that are wrong with nothing on screen to hint at it.
            messagebox.showinfo(
                "Pool Guard",
                "The area has moved, so the measurements in the settings file have "
                "been cleared.\n\nThey described the previous corners. Measure the "
                "new area and type them in if you want speeds in metres per second; "
                "everything else works exactly the same without them.",
            )
            self.settings = config.load()

        self._launch(zone, np.array(self.points), (width, height))

    def _launch(
        self,
        zone: Zone,
        image_points: np.ndarray | None,
        frame_size: tuple[int, int] | None,
    ) -> None:
        if not model_path().exists():
            messagebox.showerror(
                "Pool Guard",
                f"The detector model is missing.\n\nExpected at:\n{model_path()}\n\n"
                "From source, run: python scripts/download_model.py",
            )
            return

        calibration = None
        if self.settings.has_measurements and image_points is not None and frame_size is not None:
            calibration = from_settings(image_points, self.settings.measurements, frame_size)

        try:
            self.engine = MonitorEngine(
                source=self.source, zone=zone, policy=self.settings.policy(),
                calibration=calibration, conf=self.settings.confidence,
                width=self.settings.width, stride=self.settings.frame_stride,
                sound=self.settings.sound,
                telegram_token=self.settings.telegram_token,
                telegram_chat_id=self.settings.telegram_chat_id,
            )
        except Exception as error:  # a bad path, a busy camera, a corrupt model
            messagebox.showerror("Pool Guard", f"Could not start:\n\n{error}")
            return

        self._release_capture()
        self.setup_frame.destroy()
        self._build_control(calibrated=calibration is not None)
        self.engine.start()
        self._tick()

    # ------------------------------------------------------------- control view

    def _build_control(self, calibrated: bool) -> None:
        frame = ttk.Frame(self.root, padding=12)
        frame.grid(row=0, column=0, sticky="nsew")
        self.root.title("Pool Guard - watching")

        self.mode_text = tk.StringVar(value="OPEN - activity allowed")
        self.detail_text = tk.StringVar(value="")

        self.video_canvas = tk.Canvas(
            frame, width=CANVAS_W, height=CANVAS_H, bg="#111", highlightthickness=0
        )
        self.video_canvas.grid(row=0, column=0, sticky="n")

        transport = ttk.Frame(frame)
        transport.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        self.transport = transport
        ttk.Button(transport, text="<< 30s", width=7, command=lambda: self._skip(-30)).pack(side="left")
        ttk.Button(transport, text="< 5s", width=6, command=lambda: self._skip(-5)).pack(side="left", padx=3)
        self.pause_button = ttk.Button(transport, text="Pause", width=8, command=self._toggle_pause)
        self.pause_button.pack(side="left", padx=3)
        ttk.Button(transport, text="5s >", width=6, command=lambda: self._skip(5)).pack(side="left", padx=3)
        ttk.Button(transport, text="30s >>", width=7, command=lambda: self._skip(30)).pack(side="left")
        self.position_slider = ttk.Scale(transport, from_=0, to=100, orient="horizontal")
        self.position_slider.bind("<ButtonRelease-1>", self._on_seek)
        self.position_slider.pack(side="left", fill="x", expand=True, padx=(10, 0))

        panel = ttk.Frame(frame)
        panel.grid(row=0, column=1, rowspan=2, sticky="n", padx=(14, 0))

        self.mode_banner = tk.Label(
            panel, textvariable=self.mode_text, font=("Segoe UI", 12, "bold"),
            bg="#2e7d32", fg="white", padx=10, pady=10, width=30, wraplength=250,
        )
        self.mode_banner.grid(row=0, column=0, sticky="ew")

        self.toggle_button = tk.Button(
            panel, text="Close the pool", font=("Segoe UI", 13, "bold"),
            bg="#c62828", fg="white", activebackground="#8e0000", activeforeground="white",
            height=2, command=self.toggle_mode, cursor="hand2",
        )
        self.toggle_button.grid(row=1, column=0, sticky="ew", pady=10)

        ttk.Label(panel, textvariable=self.detail_text, foreground="#555").grid(
            row=2, column=0, sticky="w"
        )
        if calibrated:
            ttk.Label(panel, text="Calibrated: speeds in m/s.", foreground="#0a58ca").grid(
                row=3, column=0, sticky="w", pady=(2, 0)
            )

        ttk.Label(panel, text="Events", font=("Segoe UI", 10, "bold")).grid(
            row=4, column=0, sticky="w", pady=(12, 4)
        )
        self.event_list = tk.Listbox(panel, width=44, height=16, activestyle="none")
        self.event_list.grid(row=5, column=0, sticky="ew")

        buttons = ttk.Frame(panel)
        buttons.grid(row=6, column=0, sticky="ew", pady=(10, 0))
        ttk.Button(buttons, text="Open events folder", command=self._open_events).pack(side="left")
        ttk.Button(buttons, text="Stop", command=self.stop).pack(side="right")

        if self.engine is not None and self.engine.is_camera:
            self.transport.grid_remove()  # nothing to seek on a live camera
        else:
            self.position_slider.configure(to=max(1, (self.engine.total_frames if self.engine else 1) - 1))

        self.root.bind("<space>", lambda _event: self.toggle_mode())
        self.root.protocol("WM_DELETE_WINDOW", self.stop)
        self._centre()

    def toggle_mode(self) -> None:
        if self.engine is None:
            return
        if self.engine.toggle() is Mode.CLOSED:
            scenery = len(self.engine.tracker.confirmed_tracks())
            self._log(f"pool CLOSED — {scenery} object(s) recorded as scenery")
        else:
            self._log("pool OPEN — monitoring off")

    def _skip(self, seconds: float) -> None:
        if self.engine is not None:
            self.engine.skip_seconds(seconds)
            self._log(f"jumped {seconds:+.0f}s")

    def _on_seek(self, _event: object) -> None:
        if self.engine is not None:
            self.engine.seek(int(float(self.position_slider.get())))

    def _toggle_pause(self) -> None:
        if self.engine is None:
            return
        self.engine.set_paused(not self.engine.paused)
        self.pause_button.configure(text="Resume" if self.engine.paused else "Pause")

    def _open_events(self) -> None:
        path = events_dir()
        if sys.platform == "win32":
            import os

            os.startfile(path)  # noqa: S606 - our own output folder
        else:
            messagebox.showinfo("Pool Guard", str(path))

    def _log(self, text: str) -> None:
        """Add a line to the event list, stamped with the time it happened.

        For a recording that means the position *in the recording* -- 4:32 is a
        place you can scrub to, whereas the wall-clock time it was analysed tells
        you nothing about where to look. A live camera has no such position, so
        there the wall clock is the only meaningful stamp.
        """
        stamp = None
        if self.engine is not None and not self.engine.is_camera and self.engine.source_fps > 0:
            seconds = self.engine.position / self.engine.source_fps
            stamp = f"{int(seconds) // 60:d}:{int(seconds) % 60:02d}"
        if stamp is None:
            stamp = datetime.now().strftime("%H:%M:%S")

        self.event_list.insert(0, f"{stamp:>8}  {text}")
        if self.event_list.size() > 300:
            self.event_list.delete(300, tk.END)

    def _tick(self) -> None:
        """Analyse one frame, draw it, and schedule the next at real-time pace."""
        engine = self.engine
        if engine is None:
            return

        snapshot = engine.step()
        if snapshot is not None:
            for event in snapshot.events:
                extra = f"  {event.speed_mps:.1f}m/s" if event.speed_mps is not None else ""
                self._log(
                    f"{event.severity.value.upper():6s} #{event.track_id} "
                    f"{event.label}{extra} :: {event.reason}"
                )
                if event.severity is Severity.ALARM:
                    self.mode_banner.configure(bg="#b71c1c")

            self._show_video(snapshot.frame)
            self._refresh_status(snapshot)
        elif engine.finished:
            self.mode_text.set("END OF VIDEO")
            self.mode_banner.configure(bg="#555")
            self.toggle_button.configure(state="disabled")
            return

        # The engine says when the next frame is due, so the video plays at the
        # speed it was shot at rather than the speed the detector manages.
        self._tick_job = self.root.after(engine.next_delay_ms(), self._tick)

    def _show_video(self, display: np.ndarray) -> None:
        scale = fit_scale(display, MAX_VIDEO_W, MAX_VIDEO_H)
        preview = cv2.resize(display, None, fx=scale, fy=scale) if scale < 1.0 else display
        if (int(self.video_canvas["width"]), int(self.video_canvas["height"])) != (
            preview.shape[1], preview.shape[0]
        ):
            # Once, on the first frame: fit the canvas to the video's shape.
            self.video_canvas.configure(width=preview.shape[1], height=preview.shape[0])
            self._centre()
        self.photo = to_photo_image(preview)  # kept on self or Tk shows a blank
        self.video_canvas.delete("all")
        self.video_canvas.create_image(0, 0, anchor="nw", image=self.photo)

    def _refresh_status(self, snapshot) -> None:
        engine = self.engine
        if engine is None:
            return

        if engine.mode is Mode.CLOSED:
            grace = engine.grace_remaining()
            if grace > 0:
                self.mode_text.set(f"CLOSED - exit delay {grace:.0f}s")
                self.mode_banner.configure(bg="#ef6c00")
            else:
                self.mode_text.set("CLOSED - watching")
                if self.mode_banner.cget("bg") != "#b71c1c":
                    self.mode_banner.configure(bg="#c62828")
            self.toggle_button.configure(text="Open the pool", bg="#2e7d32", activebackground="#1b5e20")
        else:
            self.mode_text.set("OPEN - activity allowed")
            self.mode_banner.configure(bg="#2e7d32")
            self.toggle_button.configure(text="Close the pool", bg="#c62828", activebackground="#8e0000")

        people = sum(1 for track in snapshot.tracks if track.label == "person")
        where = ""
        if snapshot.total:
            seconds = snapshot.position / max(engine.source_fps, 1)
            where = f"   {int(seconds) // 60:d}:{int(seconds) % 60:02d}"
            if not self.position_slider.instate(["pressed"]):
                self.position_slider.set(snapshot.position)
        self.detail_text.set(
            f"{snapshot.analysed_fps:.1f} analysed/s   every {engine.stride}th frame   "
            f"{len(snapshot.tracks)} tracked ({people} person){where}"
        )

    def stop(self) -> None:
        if self._tick_job is not None:
            self.root.after_cancel(self._tick_job)
            self._tick_job = None
        if self.engine is not None:
            self.engine.close()
            self.engine = None
        self._release_capture()
        self.root.destroy()


def _report_callback_exception(exc, value, tb) -> None:
    """Surface an exception raised inside a Tk callback instead of losing it.

    Tk's default is to print the traceback to stderr and carry on. In a windowed
    build there is no stderr, so a failing click handler does precisely nothing
    visible -- the button or the picture simply stops responding, with no clue
    anywhere as to why. That is a miserable thing to debug from a distance, so
    anything that escapes a callback is written down and shown.
    """
    import traceback

    text = "".join(traceback.format_exception(exc, value, tb))
    print(text, file=sys.stderr)
    try:
        log = data_dir() / "poolguard-error.log"
        log.write_text(text, encoding="utf-8")
        where = f"\n\nWritten to:\n{log}"
    except Exception:
        where = ""
    messagebox.showerror("Pool Guard - error", f"{text[-1200:]}{where}")


def main(argv: list[str] | None = None) -> int:
    config.ensure_file()  # so there is always a settings file to look at
    root = tk.Tk()
    root.report_callback_exception = _report_callback_exception
    SetupApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
