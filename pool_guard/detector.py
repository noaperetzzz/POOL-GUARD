"""Object detection with YOLOX-S, run through OpenCV's DNN module.

Deliberately no PyTorch and no GPU: the model is a 35 MB ONNX file and
`cv2.dnn` runs it on the CPU, so the whole project installs from a two-line
requirements file and works on any laptop.

The choice of a detector trained on COCO is what makes the "is it a person, a
child, a dog, or a forgotten inflatable?" question answerable at all. COCO has
separate `person`, `dog` and `cat` classes, and it has no class for pool floats
or garden furniture -- so an inflatable ring simply does not get reported.
That is a far stronger signal than any amount of motion analysis, which can
only ever say "some pixels changed" and is hopeless around water and moving
shadows.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .tracker import Detection

COCO_CLASSES: tuple[str, ...] = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
)

# Classes that can walk into a pool zone on their own. Everything else the
# detector reports is discarded before it ever reaches the tracker.
DEFAULT_KEEP = ("person", "dog", "cat")

def letterbox(
    image: np.ndarray, size: tuple[int, int] = (640, 640), pad_value: float = 114.0
) -> tuple[np.ndarray, float]:
    """Resizes the image to fit the target size while keeping its original shape.
    Adds gray padding to fill the rest of the square.
    Returns the padded image and the scale used, so we can draw the bounding boxes back on the original image later.
    """
    height, width = image.shape[:2]
    scale = min(size[1] / height, size[0] / width)
    new_w, new_h = int(round(width * scale)), int(round(height * scale))

    canvas = np.full((size[1], size[0], 3), pad_value, dtype=np.float32)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas[:new_h, :new_w] = resized.astype(np.float32)
    return canvas, scale


class YoloxDetector:

    """YOLOX-S wrapped so that `detect(frame)` returns a list of `Detection`.

    Args:
        model_path: path to the ONNX file (see `scripts/download_model.py`).
        conf_threshold: minimum objectness x class score to report. Kept low on
            purpose -- missing a child is far worse than reporting a shrub, and
            the tracker's voting stage cleans up the extra noise downstream.
        nms_threshold: IoU above which two boxes are treated as the same object.
        keep_classes: COCO class names to pass through; None keeps all 80.
        input_size: network input, must match the exported model.
    """

    def __init__(
        self,
        model_path: str | Path,
        conf_threshold: float = 0.35,
        nms_threshold: float = 0.5,
        keep_classes: tuple[str, ...] | None = DEFAULT_KEEP,
        input_size: tuple[int, int] = (640, 640),
        swap_rb: bool = True,
    ) -> None:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"model not found: {model_path}\n"
                "Run: python scripts/download_model.py"
            )

        self.net = cv2.dnn.readNet(str(model_path))
        self.conf_threshold = float(conf_threshold)
        self.nms_threshold = float(nms_threshold)
        self.keep_classes = set(keep_classes) if keep_classes else None
        self.input_size = input_size
        self.swap_rb = bool(swap_rb)
        self.strides = (8, 16, 32)
        self._grids, self._grid_strides = self._build_grids()

    def _build_grids(self) -> tuple[np.ndarray, np.ndarray]:
        """Precompute the grid coordinates and strides for YOLOX prediction decoding.
        WHY IS THIS NEEDED?
        -------------------
        YOLOX is an "anchor-free" detector. Instead of giving final pixel positions 
        directly (like X=312, Y=208), the model outputs relative OFFSETS from a specific 
        grid cell. 

        To convert the model's output into actual pixel locations on the image:
            Pixel Position = (Grid Cell Position + Model Offset) * Stride
        EXAMPLE:
        --------
        - Image Size: 640x640
        - Target Cell: (X=3, Y=2) in Stride 8 (cell size is 8x8 pixels)
        - Model Prediction Output: Offset = (0.5, 0.1)

        Calculation:
            Actual X = (3 + 0.5) * 8 = 28.0 pixels
            Actual Y = (2 + 0.1) * 8 = 16.8 pixels

        WHY DO THIS IN __init__?
        ------------------------
        Since the input image size is fixed (e.g., 640x640), the grid structure never 
        changes between video frames. Generating all 8,400 grid cells once during 
        initialization saves a massive amount of CPU computation on every single frame.
        """
        grids, strides = [], []
        for stride in self.strides:
            # Calculate grid dimensions (number of cells) for the current stride (8, 16 or 32)
            h_cells = self.input_size[1] // stride
            w_cells = self.input_size[0] // stride
            # Generate a 2D meshgrid of (x, y) cell indices
            xv, yv = np.meshgrid(np.arange(w_cells), np.arange(h_cells))
            # Reshape the 2D grid into an Nx2 matrix of (x, y) coordinates
            grid = np.stack((xv, yv), axis=2).reshape(-1, 2)
            grids.append(grid)
            # Create an Nx1 column storing the stride value for each grid cell
            strides.append(np.full((grid.shape[0], 1), stride, dtype=np.float32))
        return (
            # Concatenate into two main matrices:
            # 1. (8400, 2) grid matrix storing (x, y) cell indices for all strides
            # 2. (8400, 1) strides matrix storing the corresponding stride value for each cell
            np.concatenate(grids, axis=0).astype(np.float32),
            np.concatenate(strides, axis=0),
        )

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Detect objects in one BGR frame."""
        padded, scale = letterbox(frame, self.input_size)
        if self.swap_rb:
            padded = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)

        # Convert image from OpenCV format (H, W, C) to Deep Learning Blob format (B, C, H, W):
        # 1. transpose(2, 0, 1): Moves channels to the front -> (Channels, Height, Width)
        # 2. np.newaxis: Adds a batch dimension at index 0 -> (1, Channels, Height, Width)
        # Result shape: (1, 3, 640, 640) - ready to feed into network.setInput()
        blob = padded.transpose(2, 0, 1)[np.newaxis, ...]
        self.net.setInput(blob)
        raw = self.net.forward()[0]  # (num_cells, 85)

        # Decode network outputs to bounding boxes (x, y, w, h) while x,y are pixels, width and height of the object in the original image
        # confidence scores computed by Objectness Score×Max Class Probability.
        # Objectness score- the probability there's an object, Max Class Probability- what is the highest probability of all types of objects
        # class IDs- the types of objects that were found
        boxes_xywh, scores, class_ids = self._decode(raw, scale)
        if len(boxes_xywh) == 0:
            return []

        keep = cv2.dnn.NMSBoxes(
            boxes_xywh.tolist(), scores.tolist(), self.conf_threshold, self.nms_threshold
        )
        if len(keep) == 0:
            return []

        detections: list[Detection] = []
        for index in np.asarray(keep).reshape(-1):
            label = COCO_CLASSES[int(class_ids[index])]
            if self.keep_classes is not None and label not in self.keep_classes:
                continue
            x, y, w, h = boxes_xywh[index]
            detections.append(
                # (x, y) is the upper left-most corner, (x+w, y+h) is the lower right most corner of the rectangle in the image
                Detection(box=np.array([x, y, x + w, y + h]), score=float(scores[index]), label=label)
            )
        return detections

    def _decode(
        self, raw: np.ndarray, scale: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Turn raw network output into pixel-space boxes, scores and class ids."""
        centres = (raw[:, :2] + self._grids) * self._grid_strides
        sizes = np.exp(raw[:, 2:4]) * self._grid_strides
        objectness = raw[:, 4]
        class_scores = raw[:, 5:]

        class_ids = class_scores.argmax(axis=1)
        best_class = class_scores[np.arange(class_scores.shape[0]), class_ids]
        # YOLOX splits confidence in two: "is anything here?" x "what is it?".
        confidence = objectness * best_class

        mask = confidence >= self.conf_threshold
        if not np.any(mask):
            return np.empty((0, 4)), np.empty(0), np.empty(0, dtype=int)

        centres, sizes = centres[mask], sizes[mask]
        # Undo the letterbox scaling to get back to the source frame's pixels.
        top_left = (centres - sizes / 2.0) / scale
        sizes = sizes / scale
        return (
            # 1. Boxes: Matrix of [x, y, width, height] in original image pixels
            np.concatenate([top_left, sizes], axis=1),
            # 2. Scores: Array of confidence levels for each box (e.g., 0.95 = 95% sure)
            confidence[mask],
            # 3. Class IDs: Array of category numbers (e.g., 0 for person, 16 for dog)
            class_ids[mask],
        )
