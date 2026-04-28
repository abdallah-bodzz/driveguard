"""
DriveGuard — Configuration
All constants, colour definitions, threshold defaults, and asset paths.
Edit this file to tune any parameter without touching detection or UI logic.
"""

import os
import numpy as np

# ─────────────────────────────────────────────────────────────────────
#  Base directory — all asset paths are absolute, relative to this file
# ─────────────────────────────────────────────────────────────────────
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def asset(relative_path: str) -> str:
    """Return absolute path to an asset relative to the project root."""
    return os.path.join(_BASE_DIR, relative_path)


# ─────────────────────────────────────────────────────────────────────
#  UI Colours  (hex — CustomTkinter / Tkinter)
# ─────────────────────────────────────────────────────────────────────
CLR_BG      = "#0D1117"
CLR_PANEL   = "#161B22"
CLR_BORDER  = "#30363D"
CLR_GREEN   = "#3FB950"
CLR_YELLOW  = "#D29922"
CLR_ORANGE  = "#E07B39"
CLR_RED     = "#F85149"
CLR_BLUE    = "#58A6FF"
CLR_TEXT    = "#E6EDF3"
CLR_SUBTEXT = "#8B949E"

# ─────────────────────────────────────────────────────────────────────
#  OpenCV BGR colours
# ─────────────────────────────────────────────────────────────────────
CV_GREEN  = (80,  185,  63)
CV_YELLOW = (34,  153, 210)
CV_ORANGE = (57,  123, 224)
CV_RED    = (73,   81, 248)
CV_BLUE   = (255, 166,  88)
CV_CYAN   = (255, 255,   0)   # hand landmarks
CV_WHITE  = (220, 220, 220)
CV_GRAY   = (80,   80,  80)

# ─────────────────────────────────────────────────────────────────────
#  MediaPipe Face Landmark Indices
# ─────────────────────────────────────────────────────────────────────
LEFT_EYE  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# ─────────────────────────────────────────────────────────────────────
#  3-D Face Model for solvePnP (6 canonical points)
# ─────────────────────────────────────────────────────────────────────
FACE_3D = np.array([
    ( 0.0,   0.0,   0.0),
    ( 0.0, -30.0, -30.0),
    (-30.0, -10.0, -30.0),
    ( 30.0, -10.0, -30.0),
    (-20.0,  -5.0, -30.0),
    ( 20.0,  -5.0, -30.0),
], dtype=np.float32)

FACE_IDX = [1, 152, 33, 362, 78, 308]

# ─────────────────────────────────────────────────────────────────────
#  Detection Defaults  (overridden by calibration at runtime)
# ─────────────────────────────────────────────────────────────────────
EAR_THRESHOLD_DEFAULT   = 0.215
MAR_THRESHOLD_DEFAULT   = 0.68
HEAD_PITCH_THRESH       = 15.0   # degrees
CONSEC_FRAMES           = 6      # consecutive closed frames before scoring
ALARM_ON_THRESH         = 48
ALARM_OFF_THRESH        = 35
ALARM_CONFIRM_FRAMES    = 60     # 2 s at 30 fps
ALARM_COOLDOWN_SEC      = 30     # minimum gap between consecutive alarms
POST_ALARM_RAISE_SEC    = 600    # raise threshold window after an alarm
NO_FACE_DECAY_FRAMES    = 15     # frames before score starts decaying
NO_FACE_ALARM_STOP      = 900    # frames (~30 s) before alarm auto-stops
CNN_BORDER_MARGIN       = 0.04   # ±EAR margin for CNN borderline check
CNN_DISPLAY_INTERVAL    = 5      # frames between display-only CNN calls
CNN_CLOSED_THRESH       = 0.75   # sleepy probability → closed
CNN_OPEN_THRESH         = 0.35   # sleepy probability → open
BREAK_SCORE_THRESH      = 35.0   # score above which break suggestion may show
BREAK_TIME_SEC          = 30.0   # sustained seconds above threshold before suggestion
BLINK_RATE_LOW          = 8      # bpm below which blink rate is flagged (drowsiness)
BLINK_RATE_HIGH         = 25     # bpm above which blink rate is flagged (stress)

# Auto-calibration
AUTO_CALIB_SETTLE       = 90     # frames to skip before collecting (camera warm-up)
AUTO_CALIB_FRAMES       = 900    # frames to collect (~30 s at 30 fps)

# Manual calibration
MANUAL_CALIB_FRAMES     = 600

# ─────────────────────────────────────────────────────────────────────
#  Incident Recording
# ─────────────────────────────────────────────────────────────────────
BUFFER_SIZE             = 150    # pre-event frame buffer length
BUFFER_WIDTH            = 640
BUFFER_HEIGHT           = 360
POST_ALARM_REC_FRAMES   = 90     # frames to keep recording after alarm stops

# ─────────────────────────────────────────────────────────────────────
#  Asset Paths
# ─────────────────────────────────────────────────────────────────────
MEDIAPIPE_FACE = asset(os.path.join("assets", "mediapipe", "face_landmarker.task"))
MEDIAPIPE_HAND = asset(os.path.join("assets", "mediapipe", "hand_landmarker.task"))

ALARM_SOUND_PATHS = [asset(os.path.join("assets", "sounds", "soundreality-alarm-a.wav"))]
DING_SOUND_PATH = asset(os.path.join("assets", "sounds", "ding-sfx.wav"))

CNN_MODEL_PATHS = [asset(os.path.join("assets", "models", "eye_classifier.h5")),
                   asset(os.path.join("assets", "models", "eye_classifier.keras"))]

RECORDINGS_DIR = asset("recordings")