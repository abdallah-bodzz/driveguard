"""
DriveGuard — Recording Manager
Handles all file I/O:
  - 640×360 pre-event frame ring buffer
  - Incident MP4 video writer
  - Session CSV with per-row logging and a summary footer
  - Emergency CSV flush via atexit
"""

import atexit
import csv
import os
import time
from collections import deque
from datetime import datetime
from typing import Optional

import cv2

from config import (
    BUFFER_SIZE, BUFFER_WIDTH, BUFFER_HEIGHT,
    POST_ALARM_REC_FRAMES, RECORDINGS_DIR,
    ALARM_ON_THRESH,
)
from utils import timestamp_filename


class RecordingManager:
    """Manages incident video recording and session CSV export."""

    def __init__(self):
        self._frame_buffer: deque = deque(maxlen=BUFFER_SIZE)
        self._video_writer: Optional[cv2.VideoWriter] = None
        self.recording        = False
        self._rec_start_time  = 0.0
        self._post_rec_frames = 0

        # Blink state for overlay dot
        self._rec_blink   = False
        self._rec_blink_t = 0.0

        # CSV
        self._csv_file    = None
        self._csv_writer  = None
        self._session_start: Optional[float] = None
        self._last_csv_time = 0.0

        # Stats collected for CSV summary
        self._incidents = []          # list of (timestamp_str, score_int)
        self._score_cache = []        # scores seen this session

        atexit.register(self._emergency_close)

    # ── Buffer ────────────────────────────────────────────────────────
    def push_frame(self, frame) -> None:
        """
        Downscale frame to 640×360 and add to pre-event ring buffer.
        Always call this, even when not recording.
        """
        small = cv2.resize(frame, (BUFFER_WIDTH, BUFFER_HEIGHT))
        self._frame_buffer.append(small)

    # ── Recording ─────────────────────────────────────────────────────
    def start_incident_recording(self, frame) -> Optional[str]:
        """
        Begin incident recording.  Writes pre-event buffer to MP4 file.
        Returns the file path, or None if already recording.
        """
        if self.recording:
            return None
        self.recording        = True
        self._post_rec_frames = 0
        self._rec_start_time  = time.time()

        os.makedirs(RECORDINGS_DIR, exist_ok=True)
        ts  = timestamp_filename()
        fn  = os.path.join(RECORDINGS_DIR, f"incident_{ts}.mp4")
        h, w = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._video_writer = cv2.VideoWriter(fn, fourcc, 30.0, (w, h))

        # Write pre-event buffer (upscale from 640×360 to original)
        for bf in self._frame_buffer:
            upscaled = cv2.resize(bf, (w, h))
            self._video_writer.write(upscaled)

        return fn

    def write_frame(self, frame) -> None:
        """Append a live frame to the active video recording."""
        if self.recording and self._video_writer:
            self._video_writer.write(frame)

    def stop_incident_recording(self) -> None:
        """Finalise and close the incident video file."""
        if not self.recording:
            return
        if self._video_writer:
            self._video_writer.release()
            self._video_writer = None
        self.recording = False

    def tick_post_alarm(self, alarm_active: bool, no_face_frames: int) -> bool:
        """
        Call every frame while recording.
        Returns True when recording should stop (post-alarm tail expired,
        face missing, etc.).
        """
        if not self.recording:
            return False
        if alarm_active:
            self._post_rec_frames = POST_ALARM_REC_FRAMES
            return False
        self._post_rec_frames -= 1
        if self._post_rec_frames <= 0:
            return True
        if no_face_frames > 60:
            return True
        return False

    @property
    def rec_elapsed(self) -> int:
        """Seconds elapsed since recording started."""
        return int(time.time() - self._rec_start_time) if self.recording else 0

    def draw_rec_dot(self, frame) -> None:
        """Draw a blinking red circle on the frame when recording."""
        if not self.recording:
            return
        now = time.time()
        if now - self._rec_blink_t > 0.5:
            self._rec_blink   = not self._rec_blink
            self._rec_blink_t = now
        if self._rec_blink:
            h, w = frame.shape[:2]
            cv2.circle(frame, (w - 28, 28), 10, (73, 81, 248), -1)

    def clear_buffer(self) -> None:
        self._frame_buffer.clear()

    # ── CSV ───────────────────────────────────────────────────────────
    def open_session_csv(self, session_start: float) -> Optional[str]:
        """Open a new session CSV file. Returns path or None on failure."""
        os.makedirs(RECORDINGS_DIR, exist_ok=True)
        ts   = timestamp_filename()
        path = os.path.join(RECORDINGS_DIR, f"session_{ts}.csv")
        try:
            self._csv_file    = open(path, "w", newline="", encoding="utf-8")
            self._csv_writer  = csv.writer(self._csv_file)
            self._csv_writer.writerow([
                "time_sec", "EAR", "MAR", "fatigue_score",
                "short_term", "long_term", "alarm_active", "dominant_cause",
                "perclos", "blink_rate_bpm",
            ])
            self._session_start   = session_start
            self._last_csv_time   = 0.0
            self._incidents       = []
            self._score_cache     = []
            return path
        except Exception as exc:
            print(f"[DriveGuard] CSV open failed: {exc}")
            return None

    def log_csv_row(self, detection_result, force: bool = False) -> None:
        """
        Write one data row to the CSV.  Rate-limited to once per second
        unless force=True.
        """
        if not self._csv_writer or self._session_start is None:
            return
        now = time.time()
        if not force and now - self._last_csv_time < 1.0:
            return
        self._last_csv_time = now
        r = detection_result
        elapsed = now - self._session_start
        try:
            self._csv_writer.writerow([
                f"{elapsed:.1f}",
                f"{r.ear:.3f}",
                f"{r.mar:.3f}",
                f"{r.fatigue_score:.1f}",
                f"{r.short_term_score:.1f}",
                f"{r.long_term_score:.1f}",
                int(r.alarm_active),
                r.dominant_cause,
                r.perclos,
                f"{r.blink_rate_bpm:.1f}",
            ])
            self._csv_file.flush()
            self._score_cache.append(int(r.fatigue_score))
        except Exception:
            pass

    def log_incident(self, score: int) -> None:
        """Record an alarm event for CSV summary."""
        ts = datetime.now().strftime("%H:%M:%S")
        self._incidents.append((ts, score))

    def close_session_csv(self, session_start: Optional[float] = None) -> None:
        """Write summary footer and close the session CSV."""
        if not self._csv_file:
            return
        try:
            start = session_start or self._session_start
            dur   = int(time.time() - start) if start else 0
            scores = self._score_cache
            mx    = max(scores) if scores else 0
            avg   = sum(scores) / len(scores) if scores else 0.0
            critical_sec = sum(1 for s in scores if s >= ALARM_ON_THRESH) / 30.0
            self._csv_file.write(
                f"# Summary: duration={dur}s, "
                f"alerts={len(self._incidents)}, "
                f"max_score={mx}, "
                f"avg_score={avg:.1f}, "
                f"critical_seconds={critical_sec:.1f}\n"
            )
            self._csv_file.close()
        except Exception:
            pass
        finally:
            self._csv_file   = None
            self._csv_writer = None

    def _emergency_close(self) -> None:
        """atexit handler — flush and close CSV if the app crashes."""
        if self._csv_file:
            try:
                self._csv_file.flush()
                self._csv_file.close()
            except Exception:
                pass

    # ── Incident list (read by UI) ─────────────────────────────────────
    @property
    def incidents(self):
        return list(self._incidents)

    @property
    def score_history(self):
        return list(self._score_cache)

    # ── Session reset ─────────────────────────────────────────────────
    def reset_session(self) -> None:
        self.clear_buffer()
        if self.recording:
            self.stop_incident_recording()
        self._incidents     = []
        self._score_cache   = []
        self._last_csv_time = 0.0
