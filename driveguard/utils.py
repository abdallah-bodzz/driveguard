"""
DriveGuard — Utilities
Pure helper functions with no UI or detection dependencies.
"""

import math
import os
import subprocess
import sys
import time


# ─────────────────────────────────────────────────────────────────────
#  Geometry
# ─────────────────────────────────────────────────────────────────────

def edist(p1, p2) -> float:
    """Euclidean distance between two MediaPipe landmark objects."""
    return math.hypot(p1.x - p2.x, p1.y - p2.y)


def rotation_vector_to_euler(rvec):
    """
    Convert an OpenCV rotation vector to Euler angles (degrees).
    Returns (pitch, yaw, roll).
    """
    import cv2
    R, _ = cv2.Rodrigues(rvec)
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        pitch = math.atan2(-R[2, 0], sy)
        yaw   = math.atan2( R[1, 0], R[0, 0])
        roll  = math.atan2( R[2, 1], R[2, 2])
    else:
        pitch = math.atan2(-R[2, 0], sy)
        yaw   = math.atan2(-R[1, 2], R[1, 1])
        roll  = 0.0
    return math.degrees(pitch), math.degrees(yaw), math.degrees(roll)


# ─────────────────────────────────────────────────────────────────────
#  System
# ─────────────────────────────────────────────────────────────────────

def open_folder(path: str) -> None:
    """Open a folder in the native system file explorer."""
    os.makedirs(path, exist_ok=True)
    try:
        if sys.platform == "win32":
            os.startfile(path)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])
    except Exception as exc:
        print(f"[DriveGuard] open_folder error: {exc}")


# ─────────────────────────────────────────────────────────────────────
#  Time / formatting
# ─────────────────────────────────────────────────────────────────────

def fmt_duration(seconds: int) -> str:
    """Return 'MM:SS' string from an integer number of seconds."""
    m, s = divmod(max(0, seconds), 60)
    return f"{m:02d}:{s:02d}"


def now_str(fmt: str = "%H:%M:%S") -> str:
    """Current local time as a formatted string."""
    return time.strftime(fmt)


def timestamp_filename() -> str:
    """Return a consistent compact timestamp string for filenames: YYYYMMDD_HHMMSS."""
    return time.strftime("%Y%m%d_%H%M%S")
