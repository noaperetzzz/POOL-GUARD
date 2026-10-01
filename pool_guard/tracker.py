"""Multi-object tracking: turns per-frame detections into persistent tracks.

The detector answers "what is in this frame?".  The tracker answers "is this
the same subject I saw a second ago?", which is the question the alarm logic
actually needs.  It is a compact variant of SORT: a Kalman filter per track,
IoU matching between predictions and new detections, and a small state machine
that promotes a track once it has been seen enough times.

One addition beyond plain SORT matters a lot here: each track keeps a tally of
every class label the detector ever assigned to it, and reports the winner of
that vote.  A single frame can plausibly label a yellow inflatable ring
``person 0.55``; forty frames in a row will not.  Deciding on the vote instead
of the latest frame is what keeps the siren quiet.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

try:  # optimal assignment when SciPy is around, greedy fallback when it is not
    from scipy.optimize import linear_sum_assignment

    _HAS_SCIPY = True
except ImportError:  # pragma: no cover - exercised only on minimal installs
    _HAS_SCIPY = False


@dataclass
class Detection:
    """One detector output in one frame."""

    # (x1,y1)- coordinates of the upper left-most corner of the object, (x2,y2)- coordinates of the lower right-most corner of the object
    box: np.ndarray  # (x1, y1, x2, y2) in pixels
    score: float
    label: str

    def __post_init__(self) -> None:
        self.box = np.asarray(self.box, dtype=float).reshape(4)


def iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Pairwise intersection-over-union between two sets of (x1, y1, x2, y2) boxes."""
    # Handle empty box cases
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)))
    
    a = np.asarray(boxes_a, dtype=float).reshape(-1, 1, 4)
    b = np.asarray(boxes_b, dtype=float).reshape(1, -1, 4)

    # 4 matrices of every side of the rectangle of the overlapping
    inter_x1 = np.maximum(a[..., 0], b[..., 0])
    inter_y1 = np.maximum(a[..., 1], b[..., 1])
    inter_x2 = np.minimum(a[..., 2], b[..., 2])
    inter_y2 = np.minimum(a[..., 3], b[..., 3])

    # matrix of the area of the overlapping of every 2 frames, with no values under 0
    inter = np.clip(inter_x2 - inter_x1, 0, None) * np.clip(inter_y2 - inter_y1, 0, None)
    # matrix of the area of every rectangle in a and b (x2-x1) * (y2-y1)
    area_a = np.clip(a[..., 2] - a[..., 0], 0, None) * np.clip(a[..., 3] - a[..., 1], 0, None)
    area_b = np.clip(b[..., 2] - b[..., 0], 0, None) * np.clip(b[..., 3] - b[..., 1], 0, None)

    union = area_a + area_b - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(union > 0, inter / union, 0.0)


def _match(cost: np.ndarray) -> list[tuple[int, int]]:
    """Maximise total IoU. Hungarian if SciPy is available, else greedy."""
    if _HAS_SCIPY:
        # linear_sum_assignment finds the optimal matching (minimum cost).
        # We pass '-cost' to invert values, effectively maximizing the total IoU overlapping score.
        # 'rows' contains row indices of matrix A, 'cols' contains column indices of matrix B.
        # At the end, it finds the optimal combination of unique index pairs (row, col) that maximizes the total overlap score.
        rows, cols = linear_sum_assignment(-cost)
        return list(zip(rows.tolist(), cols.tolist()))

    pairs: list[tuple[int, int]] = []
    used_rows: set[int] = set()
    used_cols: set[int] = set()
    # Convert and sort matrix coordinates (row, col) from highest to lowest IoU score
    order = np.dstack(np.unravel_index(np.argsort(-cost, axis=None), cost.shape))[0]
    for row, col in order:
        if row in used_rows or col in used_cols:
            continue
        used_rows.add(int(row))
        used_cols.add(int(col))
        pairs.append((int(row), int(col)))
    return pairs


@dataclass
class Track:
    """A single subject followed across frames."""

    # Unique ID number for this object (no other active subject can have the same ID)
    track_id: int

    # The math filter (Kalman filter) that predicts the box position and movement
    filter: object  # BoxTrackerFilter; typed loosely to avoid a circular import

    # A counter that counts all class label votes received over time about the subject in the image
    votes: Counter = field(default_factory=Counter)

    # How many frames this object was actually seen and detected by the model
    hits: int = 1

    # Total age of this track in frames since it was created
    age: int = 1

    # Number of consecutive frames missed since the last real detection
    steps_since_detection: int = 0

    # Seconds, not steps. Only elapsed time says how long a track has been
    # coasting on prediction alone, and that is what decides when to drop it.
    seconds_since_detection: float = 0.0

    # True if the track is confirmed as a real object rather than temporary camera noise
    confirmed: bool = False

    # Snapshot of the first box coordinates when the track was born
    first_box: np.ndarray | None = None

    # Missed steps are split in two, because they mean opposite things:
    sporadic_misses: int = 0   # short gaps: an unreliable, flickering detection
    occluded_steps: int = 0    # one long gap: a real subject that went behind something

    @property
    def box(self) -> np.ndarray:
        """Returns the current predicted box coordinates from the Kalman filter"""
        return self.filter.box

    @property
    def label(self) -> str:
        """Majority class over the track's whole life, not just this frame"""
        if not self.votes:
            return "unknown"
        return self.votes.most_common(1)[0][0]

    @property
    def detected_now(self) -> bool:
        """Was this track backed by a detection on the most recent step?

        When False the box is pure prediction. That is legitimate for holding on
        to a subject through a brief occlusion, but it is not evidence that
        anybody is there, so nothing user-facing should treat it as such: it is
        neither drawn nor allowed to raise an alarm.
        """
        return self.steps_since_detection == 0

    @property
    def label_agreement(self) -> float:
        """Returns the proportion of votes that agree with the winning class label (0.0 to 1.0)"""
        if not self.votes:
            return 0.0
        return self.votes.most_common(1)[0][1] / sum(self.votes.values())

    @property
    def detection_rate(self) -> float:
        """Calculates detection reliability (0.0 to 1.0).
        
        Ignores long continuous gaps (e.g., walking behind an obstacle) to avoid 
        lowering confidence during valid occlusions, but penalizes short, 
        flickering misses that indicate an unstable or noisy detection.
        """
        return self.hits / max(self.hits + self.sporadic_misses, 1)

    @property
    def label_confidence(self) -> float:
        """Overall confidence in the track's classification, between 0.0 and 1.0.

        It combines class agreement with detection stability:
        - Agreement alone is not enough, because a false detection that pops up
          only a few times might still have 100% agreement on its label.
        - By multiplying agreement with the detection rate, we ensure that noisy 
          or flickering objects get a low score, while real objects detected 
          consistently trigger the proper response.
        """
        return self.label_agreement * self.detection_rate

    @property
    def speed(self) -> float:
        """Estimated speed in pixels per second."""
        return self.filter.speed

    @property
    def travelled(self) -> float:
        """Straight-line distance in pixels from where the track began.

        Small travel over a long life is the signature of something that was
        never going anywhere: a float resting on the water, a garden chair, a
        towel on the grass. The monitor uses it to ignore static objects.
        """
        if self.first_box is None:
            return 0.0
        start = self.first_box
        now = self.box
        start_centre = np.array([(start[0] + start[2]) / 2, (start[1] + start[3]) / 2])
        now_centre = np.array([(now[0] + now[2]) / 2, (now[1] + now[3]) / 2])
        # distance = (x_now - x_start)^2 + (y_now - y_start)^2
        return float(np.linalg.norm(now_centre - start_centre))

    def anchor(self) -> tuple[float, float]:
        """The point used to decide whether the subject is inside the zone.

        Bottom-centre of the box, i.e. roughly where the feet touch the ground.
        Using the box centre instead would fire early, while the subject is
        still outside and only leaning over the boundary.
        """
        x1, _, x2, y2 = self.box
        # Calculate bottom-center coordinates (X = middle width, Y = bottom feet level)
        return ((x1 + x2) / 2.0, y2)


class Tracker:
    """Maintains the set of live tracks across frames.

    All time limits in this class are measured in seconds rather than frame counts. 
    Counting in frames breaks whenever processing speed changes or when frames are skipped—for 
    example, a limit like 'keep for 12 frames' suddenly lasts much longer in real-world time, 
    leaving empty "ghost" boxes on the screen. Using real-time seconds guarantees that 
    the system behaves consistently no matter how fast or slow the video is processed.

    Args:
        dt: nominal seconds between updates, for the Kalman filters. `update`
            takes the measured value, so this is only a starting point.
        iou_threshold: below this an overlap is not accepted as a match.
        max_coast_seconds: how long a track may go undetected before it is
            dropped. Long enough to ride out a brief occlusion, short enough
            that a lost track disappears before anyone notices it drifting.
        min_hits: detections needed before a track is trusted (``confirmed``).
            The first line of defence against one-off flukes.
        occlusion_seconds: a gap longer than this counts as an occlusion
            (excluded from `detection_rate`) rather than as flicker.
        distance_gate: second-chance matching. After the overlap pass, an
            unmatched detection may still join an unmatched track if their
            centres are within this multiple of the mean box size. At five
            analysed frames a second a walking person moves most of their own
            width between steps, so overlap alone splits one person into a
            string of short tracks -- and the abandoned ones are exactly the
            phantom boxes this is meant to prevent.
    """

    def __init__(
        self,
        dt: float = 1.0 / 25.0,
        iou_threshold: float = 0.2,
        max_coast_seconds: float = 0.6,
        min_hits: int = 4,
        occlusion_seconds: float = 0.3,
        distance_gate: float = 1.0,
    ) -> None:
        self.dt = float(dt)
        self.iou_threshold = float(iou_threshold)
        self.max_coast_seconds = float(max_coast_seconds)
        self.min_hits = int(min_hits)
        self.occlusion_seconds = float(occlusion_seconds)
        self.distance_gate = float(distance_gate)
        self.tracks: list[Track] = []
        self._next_id = 1

    def update(self, detections: list[Detection], dt: float | None = None) -> list[Track]:
        """Processes a single video frame to update all object tracks.

        Must be called on every frame. If no objects are detected in the current
        frame, pass an empty list.

        Args:
            detections: List of bounding boxes and detections found in this frame.
            dt: Exact time elapsed (in seconds) since the previous frame. 
                Leave as None if running at a fixed frame rate (FPS).

        Returns:
            list[Track]: A list containing only the active, confirmed tracks for display or alert logic.
        """
        from .kalman import BoxTrackerFilter  # local import breaks the cycle

        step = float(dt) if dt is not None else self.dt
        for track in self.tracks:
            track.filter.predict(dt)
            track.age += 1
            track.steps_since_detection += 1 # later will be 0 if the subject exists in the frame
            track.seconds_since_detection += step

        predictions = np.array([t.box for t in self.tracks]) if self.tracks else np.empty((0, 4))
        det_boxes = np.array([d.box for d in detections]) if detections else np.empty((0, 4))

        matched_tracks: set[int] = set()
        matched_dets: set[int] = set()

        cost = iou_matrix(predictions, det_boxes)
        if cost.size:
            for track_idx, det_idx in _match(cost):
                if cost[track_idx, det_idx] < self.iou_threshold:
                    continue  # the assignment solver will pair anything; we veto weak pairs
                matched_tracks.add(track_idx)
                matched_dets.add(det_idx)
                self._absorb(self.tracks[track_idx], detections[det_idx], step)

        # Second pass, on whatever the overlap test could not pair. A subject that
        # moved further than its own box between two analysed frames has no
        # overlap at all, and without this it would be abandoned and replaced by
        # a new track -- leaving the old one to coast as a box around nobody.
        for track_idx, det_idx in self._distance_matches(
            predictions, det_boxes, matched_tracks, matched_dets
        ):
            matched_tracks.add(track_idx)
            matched_dets.add(det_idx)
            self._absorb(self.tracks[track_idx], detections[det_idx], step)

        for det_idx, detection in enumerate(detections):
            if det_idx in matched_dets:
                continue
            track = Track(
                track_id=self._next_id,
                filter=BoxTrackerFilter(detection.box, dt=self.dt),
                first_box=detection.box.copy(),
            )
            track.votes[detection.label] += 1
            # A fresh track already has one hit, so check it here too rather than
            # only on the next match -- otherwise min_hits=1 confirms nothing.
            track.confirmed = track.hits >= self.min_hits
            self._next_id += 1
            self.tracks.append(track)

        self.tracks = [
            t for t in self.tracks if t.seconds_since_detection <= self.max_coast_seconds
        ]
        return self.confirmed_tracks()

    def _absorb(self, track: Track, detection: Detection, step: float) -> None:
        """Fold a matched detection into a track and update its bookkeeping."""
        track.filter.update(detection.box)
        track.votes[detection.label] += 1
        track.hits += 1

        # steps_since_detection was incremented at the top of update(), so it is
        # 1 for a track detected on consecutive steps and the gap is zero.
        gap_steps = track.steps_since_detection - 1
        if gap_steps > 0:
            if gap_steps * step <= self.occlusion_seconds:
                track.sporadic_misses += gap_steps
            else:
                track.occluded_steps += gap_steps
        track.steps_since_detection = 0
        track.seconds_since_detection = 0.0

        if track.hits >= self.min_hits:
            track.confirmed = True

    def _distance_matches(
        self,
        predictions: np.ndarray,
        det_boxes: np.ndarray,
        matched_tracks: set[int],
        matched_dets: set[int],
    ) -> list[tuple[int, int]]:
        """Pair leftovers by centre distance, gated by box size.

        Greedy nearest-first over the remaining candidates. The gate scales with
        the box so it means the same thing for a subject close to the camera as
        for one at the far end of the garden.
        """
        free_tracks = [i for i in range(len(predictions)) if i not in matched_tracks]
        free_dets = [j for j in range(len(det_boxes)) if j not in matched_dets]
        if not free_tracks or not free_dets:
            return []

        # the coordinates of the center of the box
        def centre(box: np.ndarray) -> np.ndarray:
            return np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])

        # average between height and width
        def size(box: np.ndarray) -> float:
            return (abs(box[2] - box[0]) + abs(box[3] - box[1])) / 2

        candidates = []
        for i in free_tracks:
            for j in free_dets:
                distance = float(np.linalg.norm(centre(predictions[i]) - centre(det_boxes[j])))
                allowed = self.distance_gate * (size(predictions[i]) + size(det_boxes[j])) / 2
                if distance <= allowed:
                    candidates.append((distance, i, j))

        pairs: list[tuple[int, int]] = []
        used_tracks: set[int] = set()
        used_dets: set[int] = set()
        for _distance, i, j in sorted(candidates):
            if i in used_tracks or j in used_dets:
                continue
            used_tracks.add(i)
            used_dets.add(j)
            pairs.append((i, j))
        return pairs

    def confirmed_tracks(self) -> list[Track]:
        """Tracks seen often enough to be acted on."""
        return [t for t in self.tracks if t.confirmed]

    def reset(self) -> None:
        self.tracks = []
