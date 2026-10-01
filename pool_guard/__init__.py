"""Pool Guard -- a camera that watches a zone and shouts when it should be empty.

Pipeline:

    frame -> detector -> tracker (Kalman) -> monitor (policy) -> alarm
             what is it?  is it the same    should anyone      sound, snapshot,
                          thing as before?  be told?           log line
"""

from .kalman import BoxTrackerFilter, KalmanFilter
from .monitor import Event, Mode, PoolMonitor, Policy, Severity
from .tracker import Detection, Track, Tracker
from .zone import Zone, point_in_polygon

__version__ = "0.1.0"

__all__ = [
    "BoxTrackerFilter",
    "Detection",
    "Event",
    "KalmanFilter",
    "Mode",
    "PoolMonitor",
    "Policy",
    "Severity",
    "Track",
    "Tracker",
    "Zone",
    "point_in_polygon",
]
