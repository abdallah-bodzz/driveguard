# DriveGuard API Reference

This document describes the main modules and classes of the DriveGuard Python package.  
It is intended for developers who wish to extend or integrate the system.

---

## Package Structure

`
driveguard/
├── config.py          – Constants, paths, thresholds
├── detection.py       – DetectionEngine (core logic)
├── ui.py              – DriveGuardApp (main UI)
├── recording.py       – RecordingManager (video/CSV)
├── utils.py           – Geometry, file, time helpers
└── main.py            – Entry point (if __name__ == "__main__")
`

---

## Core Class: DetectionEngine (detection.py)

Handles all signal processing, scoring, and calibration.

### Initialisation
`python
engine = DetectionEngine()
`

### Methods

| Method | Description |
|--------|-------------|
| process(frame, face_landmarks, w, h, elapsed_min, session_start) | Processes one frame, returns DetectionResult. |
| eset_session() | Full reset for a new monitoring session. |
| eset_scores() | Resets fatigue scores and counters (keeps calibration). |
| dismiss() | Resets scores after alarm dismissal. |
| set_gesture_dismiss_callback(callable) | Registers a callback for hand‑gesture dismiss events. |

### Key Attributes (readable after process)

| Attribute | Type | Description |
|-----------|------|-------------|
| atigue_score | float | Weighted combination of short‑ and long‑term scores (0–120). |
| larm_active | bool | True when alarm condition is met. |
| ear_threshold | float | Current EAR threshold (auto‑calibrated). |
| mar_threshold | float | Current MAR threshold. |
| link_rate_bpm | float | Rolling blinks per minute (60‑second window). |

### DetectionResult Dataclass

Returned by process(). Contains all metrics for the current frame:
- ear, mar, pitch, yaw, oll
- cnn_prob, cnn_left, cnn_right
- atigue_score, short_term_score, long_term_score
- larm_active, larm_triggered
- perclos, link_rate_bpm, dominant_cause, ear_trend
- uto_calibrating, uto_calib_pct

---

## UI Class: DriveGuardApp (ui.py)

Main application window. Inherits from customtkinter.CTk.

### Initialisation
`python
app = DriveGuardApp()
app.run()
`

### Public Methods

| Method | Description |
|--------|-------------|
| start_camera() | Opens webcam and starts detection thread. |
| stop_camera() | Stops monitoring and releases resources. |
| un() | Starts the Tkinter main loop. |

### Keyboard Shortcuts (bound automatically)

| Key | Action |
|-----|--------|
| Space | Mute/unmute alarm |
| D | Dismiss active alarm |
| R | Reset fatigue score |
| Esc | Stop monitoring |

---

## Recording Manager: RecordingManager (ecording.py)

Handles incident video capture and session CSV logging.

### Methods

| Method | Description |
|--------|-------------|
| push_frame(frame) | Adds a downscaled frame (640×360) to the pre‑event buffer. |
| start_incident_recording(frame) | Begins MP4 recording, writes pre‑event buffer. |
| write_frame(frame) | Appends live frame to active recording. |
| stop_incident_recording() | Finalises and saves the video file. |
| open_session_csv(session_start) | Creates a new CSV file and writes the header. |
| log_csv_row(detection_result) | Writes one data row (rate‑limited to 1/s). |
| close_session_csv() | Writes summary footer and closes the file. |

### CSV Columns

| Column | Description |
|--------|-------------|
| 	ime_sec | Seconds since session start |
| EAR, MAR | Raw geometric ratios |
| atigue_score | Weighted score (0–120) |
| short_term, long_term | Individual score components |
| larm_active | 0/1 flag |
| dominant_cause | "eyes", "yawning", "nodding", or combination |
| perclos, link_rate_bpm | Additional metrics |

---

## Configuration (config.py)

All constants are defined here. Edit to tune thresholds or change asset paths.

| Constant | Default | Description |
|----------|---------|-------------|
| EAR_THRESHOLD_DEFAULT | 0.215 | Initial EAR threshold |
| MAR_THRESHOLD_DEFAULT | 0.68 | Initial MAR threshold |
| ALARM_ON_THRESH | 48 | Alarm activation score |
| ALARM_OFF_THRESH | 35 | Deactivation score (hysteresis) |
| ALARM_COOLDOWN_SEC | 30 | Minimum seconds between alarms |
| BUFFER_SIZE | 150 | Pre‑event frame buffer length |
| CNN_DISPLAY_INTERVAL | 5 | Frames between display‑only CNN calls |

---

## Utilities (utils.py)

| Function | Description |
|----------|-------------|
| edist(p1, p2) | Euclidean distance between two MediaPipe landmarks. |
| otation_vector_to_euler(rvec) | Converts OpenCV rotation vector to (pitch, yaw, roll) degrees. |
| open_folder(path) | Opens folder in native file explorer. |
| mt_duration(seconds) | Formats seconds as MM:SS. |
| 	imestamp_filename() | Returns YYYYMMDD_HHMMSS string. |

---

## Integration Example

`python
from driveguard.detection import DetectionEngine
from driveguard.ui import DriveGuardApp

# Use the engine standalone (without UI)
engine = DetectionEngine()
frame = cv2.imread("frame.jpg")
# ... obtain face_landmarks from MediaPipe ...
res = engine.process(frame, landmarks, w, h, elapsed_min, session_start)
print(f"Fatigue score: {res.fatigue_score}")

# Or launch the full application
app = DriveGuardApp()
app.run()
`

For detailed algorithm descriptions, see docs/user_guide.md.
