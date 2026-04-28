"""
DriveGuard — Detection Engine
Encapsulates all signal-processing logic:
  - EAR / MAR geometry
  - Head-pose estimation via solvePnP (throttled, cached)
  - Pitch-velocity nod detection
  - Hybrid CNN eye verification (per-eye, continuous display mode)
  - Dual-score fatigue model (short-term + long-term)
  - Duration-weighted event scoring with hard caps
  - Hysteresis alarm logic with session-context threshold adjustment
  - Auto-calibration and manual calibration
  - PERCLOS (CNN-adjusted)
  - Blink-rate counter
  - EAR trend arrow
  - Hand-gesture dismiss (wave + fist)
  - Dominant-cause tracking

No CustomTkinter, no OpenCV drawing, no file I/O in this module.
All state is readable as plain attributes for the UI to consume.
"""

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Tuple

import cv2
import numpy as np

from config import (
    LEFT_EYE, RIGHT_EYE, FACE_3D, FACE_IDX,
    EAR_THRESHOLD_DEFAULT, MAR_THRESHOLD_DEFAULT,
    HEAD_PITCH_THRESH, CONSEC_FRAMES,
    ALARM_ON_THRESH, ALARM_OFF_THRESH, ALARM_CONFIRM_FRAMES,
    ALARM_COOLDOWN_SEC, POST_ALARM_RAISE_SEC,
    NO_FACE_DECAY_FRAMES, NO_FACE_ALARM_STOP,
    CNN_BORDER_MARGIN, CNN_DISPLAY_INTERVAL,
    CNN_CLOSED_THRESH, CNN_OPEN_THRESH,
    BREAK_SCORE_THRESH, BREAK_TIME_SEC,
    BLINK_RATE_LOW, BLINK_RATE_HIGH,
    AUTO_CALIB_SETTLE, AUTO_CALIB_FRAMES,
    MANUAL_CALIB_FRAMES,
    CNN_MODEL_PATHS,
)
from utils import edist, rotation_vector_to_euler


# ─────────────────────────────────────────────────────────────────────
#  Result dataclass — snapshot the engine produces each frame
# ─────────────────────────────────────────────────────────────────────

@dataclass
class DetectionResult:
    # Raw geometry
    ear: float = 0.30
    mar: float = 0.30
    ear_for_score: float = 0.30   # CNN-adjusted EAR used for scoring
    pitch: float = 0.0
    yaw: float = 0.0
    roll: float = 0.0

    # Eye state
    eye_closed: bool = False

    # CNN
    cnn_override: bool = False
    cnn_prob: Optional[float] = None       # average sleepy probability
    cnn_left: Optional[float] = None       # left-eye sleepy probability
    cnn_right: Optional[float] = None      # right-eye sleepy probability

    # Nodding
    head_nodding: bool = False

    # Scores
    short_term_score: float = 0.0
    long_term_score: float = 0.0
    fatigue_score: float = 0.0

    # Alarm
    alarm_triggered: bool = False    # True only on the frame the alarm fires
    alarm_active: bool = False

    # PERCLOS / blink
    perclos: int = 0
    blink_rate_bpm: float = 0.0

    # Dominant cause
    dominant_cause: str = "mixed"

    # Calibration flags
    auto_calibrating: bool = True
    auto_calib_pct: float = 0.0

    # Break suggestion
    break_suggested: bool = False

    # EAR trend
    ear_trend: str = "→"

    # Face presence
    face_present: bool = False


# ─────────────────────────────────────────────────────────────────────
#  DetectionEngine
# ─────────────────────────────────────────────────────────────────────

class DetectionEngine:
    """
    Stateful detection engine.  Call process() once per frame.
    Read public attributes or use the returned DetectionResult snapshot.
    """

    # ── Init ──────────────────────────────────────────────────────────
    def __init__(self):
        # Thresholds (may be updated by calibration)
        self.ear_threshold   = EAR_THRESHOLD_DEFAULT
        self.mar_threshold   = MAR_THRESHOLD_DEFAULT
        self.pitch_threshold = HEAD_PITCH_THRESH

        # Alarm state
        self.alarm_active           = False
        self.high_score_counter     = 0
        self.last_strong_alarm_time = 0.0   # reset on session start

        # Alarm sensitivity (set by UI slider)
        self.alarm_on_thresh     = ALARM_ON_THRESH
        self.alarm_off_thresh    = ALARM_OFF_THRESH
        self.alarm_confirm_frames = ALARM_CONFIRM_FRAMES

        # Scores
        self.short_term_score = 0.0
        self.long_term_score  = 0.0
        self.fatigue_score    = 0.0

        # Frame counters
        self.closed_frame_count = 0
        self.yawn_frame_count   = 0
        self.head_nod_count     = 0
        self.total_frames       = 0

        # History
        self.ear_history        = deque(maxlen=180)
        self.mar_history        = deque(maxlen=180)
        self.perclos_history    = deque(maxlen=180)
        self.score_history      = deque(maxlen=180)
        self.eye_closed_history = deque(maxlen=180)

        # Dominant cause (900 ≈ 30 s at 30 fps)
        self.eye_contribution  = deque(maxlen=900)
        self.yawn_contribution = deque(maxlen=900)
        self.nod_contribution  = deque(maxlen=900)

        # Head pose
        self.pitch_history        = deque(maxlen=15)
        self.cached_pitch         = 0.0
        self.cached_yaw           = 0.0
        self.cached_roll          = 0.0

        # Last geometry (for UI display)
        self.last_ear      = 0.30
        self.last_mar      = 0.30
        self.last_pitch    = 0.0

        # CNN
        self.last_cnn_prob  = None
        self.last_cnn_left  = None
        self.last_cnn_right = None
        self._cnn_display_counter = 0
        self.cnn_model    = None
        self.cnn_available = False
        self._load_cnn()

        # Blink rate — no maxlen; rely purely on 60-s time window
        self._prev_eye_closed   = False
        self._blink_timestamps  = deque()
        self.blink_rate_bpm     = 0.0

        # Auto-calibration
        self.auto_calibrating         = True
        self._auto_calib_ears         = []
        self._auto_calib_mars         = []
        self._auto_calib_settle       = AUTO_CALIB_SETTLE

        # Manual calibration
        self.calibrating    = False
        self.calibrated     = False
        self._calib_frames  = []

        # Break suggestion
        self._high_score_since = 0.0
        self._break_suggested  = False
        self._break_fired      = False   # one-shot per high-score window

        # No-face tracking
        self.no_face_frames = 0

        # Options (set by UI switches)
        self.head_nod_enabled  = False
        self.cnn_enabled       = True
        self.consec_frames     = CONSEC_FRAMES

        # Hand gesture
        self._hand_centers = deque(maxlen=20)
        self._fist_counter = 0
        self._gesture_dismiss_cb = None   # callable set by UI

    # ── CNN loader ────────────────────────────────────────────────────
    def _load_cnn(self):
        try:
            from tensorflow.keras.models import load_model as _load
        except ImportError:
            print("[DriveGuard] TensorFlow not available — CNN disabled.")
            return
        for path in CNN_MODEL_PATHS:
            import os
            if os.path.exists(path):
                try:
                    self.cnn_model    = _load(path, compile=False)
                    self.cnn_available = True
                    print(f"[DriveGuard] CNN loaded: {path}")
                    return
                except Exception as exc:
                    print(f"[DriveGuard] CNN load failed ({path}): {exc}")
        print("[DriveGuard] CNN model not found — hybrid verification disabled.")

    # ── Session reset ─────────────────────────────────────────────────
    def reset_session(self):
        """Full reset for a new monitoring session."""
        self.alarm_active           = False
        self.high_score_counter     = 0
        self.last_strong_alarm_time = 0.0   # no inter-session cooldown bleed
        self.short_term_score       = 0.0
        self.long_term_score        = 0.0
        self.fatigue_score          = 0.0
        self.closed_frame_count     = 0
        self.yawn_frame_count       = 0
        self.head_nod_count         = 0
        self.total_frames           = 0
        self.no_face_frames         = 0

        self.ear_history.clear()
        self.mar_history.clear()
        self.perclos_history.clear()
        self.score_history.clear()
        self.eye_closed_history.clear()
        self.eye_contribution.clear()
        self.yawn_contribution.clear()
        self.nod_contribution.clear()
        self.pitch_history.clear()

        self.cached_pitch = 0.0
        self.cached_yaw   = 0.0
        self.cached_roll  = 0.0
        self.last_ear     = 0.30
        self.last_mar     = 0.30
        self.last_pitch   = 0.0
        self.last_cnn_prob  = None
        self.last_cnn_left  = None
        self.last_cnn_right = None
        self._cnn_display_counter = 0

        self._prev_eye_closed  = False
        self._blink_timestamps = deque()
        self.blink_rate_bpm    = 0.0

        self.auto_calibrating   = True
        self._auto_calib_ears   = []
        self._auto_calib_mars   = []
        self._auto_calib_settle = AUTO_CALIB_SETTLE

        self.calibrating   = False
        self.calibrated    = False
        self._calib_frames = []

        self._high_score_since = 0.0
        self._break_suggested  = False
        self._break_fired      = False

        self._hand_centers.clear()
        self._fist_counter = 0

    def reset_scores(self):
        """Partial reset — scores and counters only (keeps calibration)."""
        self.short_term_score   = 0.0
        self.long_term_score    = 0.0
        self.fatigue_score      = 0.0
        self.closed_frame_count = 0
        self.yawn_frame_count   = 0
        self.head_nod_count     = 0
        self.high_score_counter = 0
        self._high_score_since  = 0.0
        self._break_suggested   = False
        self._break_fired       = False
        self.score_history.clear()
        self.eye_closed_history.clear()

    # ── Geometry ──────────────────────────────────────────────────────
    def calculate_ear(self, lm) -> float:
        if len(lm) < 400:
            return 0.30
        lA = edist(lm[LEFT_EYE[1]], lm[LEFT_EYE[5]])
        lB = edist(lm[LEFT_EYE[2]], lm[LEFT_EYE[4]])
        lC = edist(lm[LEFT_EYE[0]], lm[LEFT_EYE[3]])
        rA = edist(lm[RIGHT_EYE[1]], lm[RIGHT_EYE[5]])
        rB = edist(lm[RIGHT_EYE[2]], lm[RIGHT_EYE[4]])
        rC = edist(lm[RIGHT_EYE[0]], lm[RIGHT_EYE[3]])
        left  = (lA + lB) / (2.0 * lC) if lC else 0.0
        right = (rA + rB) / (2.0 * rC) if rC else 0.0
        return (left + right) / 2.0

    def calculate_mar(self, lm) -> float:
        if len(lm) < 400:
            return 0.0
        v = edist(lm[13], lm[14])
        h = edist(lm[78], lm[308])
        return v / h if h else 0.0

    # ── Head pose ─────────────────────────────────────────────────────
    def get_head_pose(self, lm, w: int, h: int) -> Tuple[float, float, float]:
        pts2d = np.array(
            [(lm[i].x * w, lm[i].y * h) for i in FACE_IDX],
            dtype=np.float32)
        focal = float(w)
        cam = np.array([[focal, 0, w / 2],
                        [0, focal, h / 2],
                        [0,     0,     1]], dtype=np.float32)
        dist = np.zeros((4, 1), dtype=np.float32)
        ok, rvec, _ = cv2.solvePnP(FACE_3D, pts2d, cam, dist,
                                    flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return 0.0, 0.0, 0.0
        return rotation_vector_to_euler(rvec)

    # ── Nod detection ─────────────────────────────────────────────────
    def _detect_nod(self) -> bool:
        if len(self.pitch_history) < 5:
            return False
        pitches = list(self.pitch_history)
        if abs(pitches[-1]) > self.pitch_threshold:
            return True
        deltas   = [pitches[i] - pitches[i - 1] for i in range(1, len(pitches))]
        mean_vel = sum(deltas[-5:]) / min(len(deltas), 5)
        return abs(mean_vel) > 2.5

    # ── CNN eye check ─────────────────────────────────────────────────
    def _cnn_eye_check(self, frame, lm, w: int, h: int
                       ) -> Tuple[Optional[bool], Optional[float],
                                  Optional[float], Optional[float]]:
        """
        Returns (decision, avg_prob, left_prob, right_prob).
        decision: True=closed, False=open, None=uncertain
        Probabilities are the sleepy (closed) probability 0–1, or None.
        """
        if self.cnn_model is None:
            return None, None, None, None
        try:
            eye_probs = []
            for eye_idx in [LEFT_EYE, RIGHT_EYE]:
                xs = [lm[i].x for i in eye_idx]
                ys = [lm[i].y for i in eye_idx]
                x1 = max(0, int(min(xs) * w) - 8)
                y1 = max(0, int(min(ys) * h) - 8)
                x2 = min(w, int(max(xs) * w) + 8)
                y2 = min(h, int(max(ys) * h) + 8)
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    eye_probs.append(None)
                    continue
                g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                g = cv2.resize(g, (24, 24)).reshape(1, 24, 24, 1).astype("float32") / 255.0
                eye_probs.append(float(self.cnn_model.predict(g, verbose=0)[0][1]))

            left_prob  = eye_probs[0] if len(eye_probs) > 0 else None
            right_prob = eye_probs[1] if len(eye_probs) > 1 else None
            valid      = [p for p in eye_probs if p is not None]
            if not valid:
                return None, None, None, None
            avg = sum(valid) / len(valid)
            if avg > CNN_CLOSED_THRESH:
                decision = True
            elif avg < CNN_OPEN_THRESH:
                decision = False
            else:
                decision = None
            return decision, avg, left_prob, right_prob
        except Exception:
            return None, None, None, None

    # ── Blink rate ────────────────────────────────────────────────────
    def _update_blink_rate(self, eye_closed_now: bool) -> None:
        """Track closed→open transitions and compute rolling bpm over 60 s."""
        if self._prev_eye_closed and not eye_closed_now:
            self._blink_timestamps.append(time.time())
        self._prev_eye_closed = eye_closed_now
        now    = time.time()
        cutoff = now - 60.0
        # Prune old timestamps (deque has no maxlen — we trim manually)
        while self._blink_timestamps and self._blink_timestamps[0] < cutoff:
            self._blink_timestamps.popleft()
        self.blink_rate_bpm = float(len(self._blink_timestamps))

    # ── EAR trend ─────────────────────────────────────────────────────
    def ear_trend(self) -> str:
        """
        Returns '↓' (declining), '↑' (recovering), or '→' (stable)
        based on median EAR over the last 30 frames.
        Using median avoids false arrows caused by normal blink transients.
        """
        if len(self.ear_history) < 30:
            return "→"
        recent    = list(self.ear_history)[-30:]
        half      = len(recent) // 2
        avg_first  = float(np.median(recent[:half]))
        avg_second = float(np.median(recent[half:]))
        delta = avg_second - avg_first
        if delta < -0.008:
            return "↓"
        if delta > 0.008:
            return "↑"
        return "→"

    # ── PERCLOS ───────────────────────────────────────────────────────
    def calc_perclos(self) -> int:
        if not self.eye_closed_history:
            return 0
        return int(sum(self.eye_closed_history) / len(self.eye_closed_history) * 100)

    # ── Dominant cause ────────────────────────────────────────────────
    def get_dominant_cause(self) -> str:
        def ratio(d):
            return sum(d) / len(d) if d else 0.0

        e = ratio(self.eye_contribution)
        y = ratio(self.yawn_contribution)
        n = ratio(self.nod_contribution)

        if max(e, y, n) < 0.08:
            return "mixed"

        causes = sorted([("eyes", e), ("yawning", y), ("nodding", n)],
                        key=lambda x: x[1], reverse=True)
        top_name, top_val = causes[0]
        sec_name, sec_val = causes[1]
        if sec_val > 0.08 and sec_val > top_val * 0.65:
            pair = tuple(sorted([top_name, sec_name]))
            return f"{pair[0]}+{pair[1]}"
        return top_name

    # ── Calibration ───────────────────────────────────────────────────
    def start_manual_calibration(self) -> None:
        self.calibrating   = True
        self._calib_frames = []

    def finish_auto_calibration(self) -> None:
        ears = self._auto_calib_ears
        mars = self._auto_calib_mars
        if not ears:
            return
        mean_ear = float(np.mean(ears))
        std_ear  = float(np.std(ears))
        offset   = max(0.05, min(0.10, 1.5 * std_ear))
        self.ear_threshold = max(0.15, mean_ear - offset)
        p95_mar = float(np.percentile(mars, 95))
        self.mar_threshold = max(0.45, p95_mar * 1.5)
        self.auto_calibrating = False

    def finish_manual_calibration(self) -> bool:
        """Compute thresholds from calib_frames. Returns True on success."""
        frames = self._calib_frames
        if not frames:
            self.calibrating = False
            return False
        ears = [e for e, _ in frames]
        mars = [m for _, m in frames]
        open_ear   = float(np.percentile(ears, 90))
        closed_ear = float(np.percentile(ears, 10))
        yawn_mar   = float(np.percentile(mars, 95))
        self.ear_threshold = (open_ear + closed_ear) / 2 - 0.03
        self.mar_threshold = max(0.45, yawn_mar - 0.08)
        self.calibrated    = True
        self.calibrating   = False
        return True

    # ── Hand gesture ──────────────────────────────────────────────────
    def set_gesture_dismiss_callback(self, cb) -> None:
        """Register a callable(gesture_type) to invoke on gesture recognition."""
        self._gesture_dismiss_cb = cb

    def process_hand_gesture(self, hand_lm, w: int, h: int) -> None:
        """
        Evaluate hand landmarks for wave or fist gesture.
        Calls the registered dismiss callback if recognised.
        """
        cx = sum(lm.x for lm in hand_lm) / len(hand_lm) * w
        self._hand_centers.append(cx)

        # Wave: amplitude > 30 px with ≥ 3 direction reversals
        if len(self._hand_centers) >= 10:
            xs        = list(self._hand_centers)
            amplitude = max(xs) - min(xs)
            if amplitude > 30:
                diffs = np.diff(xs)
                zc = int(np.sum(np.abs(np.diff(np.sign(diffs)))) // 2)
                if zc >= 3:
                    if self._gesture_dismiss_cb:
                        self._gesture_dismiss_cb("wave")
                    self._hand_centers.clear()
                    return

        # Fist: avg tip-to-wrist distance < 0.12
        wrist = hand_lm[0]
        tips  = [4, 8, 12, 16, 20]
        dists = [math.hypot(wrist.x - hand_lm[i].x,
                            wrist.y - hand_lm[i].y) for i in tips]
        if sum(dists) / len(dists) < 0.12:
            self._fist_counter += 1
            if self._fist_counter >= 5:
                if self._gesture_dismiss_cb:
                    self._gesture_dismiss_cb("fist")
                self._fist_counter = 0
        else:
            self._fist_counter = 0

    def clear_hand_state(self) -> None:
        self._hand_centers.clear()
        self._fist_counter = 0

    # ── Alarm helpers ─────────────────────────────────────────────────
    def dismiss(self) -> None:
        """Fully reset scores after an alarm dismissal."""
        self.alarm_active           = False
        self.high_score_counter     = 0
        self.short_term_score       = 0.0
        self.long_term_score        = max(0.0, self.long_term_score - 20.0)
        self.last_strong_alarm_time = time.time()
        self._break_suggested       = False
        self._break_fired           = False

    def stop_alarm(self) -> None:
        """Deactivate alarm (score fell below off-threshold)."""
        self.alarm_active       = False
        self.high_score_counter = 0

    # ── Main process ──────────────────────────────────────────────────
    def process(
        self,
        frame,              # BGR ndarray
        face_landmarks,     # MediaPipe face landmark list or None
        w: int, h: int,
        elapsed_min: float,
        session_start: float,
    ) -> DetectionResult:
        """
        Process one frame.  Call once per camera frame.
        Returns a DetectionResult snapshot.
        """
        result = DetectionResult()
        self.total_frames += 1

        # ── Face present ─────────────────────────────────────────────
        if face_landmarks:
            self.no_face_frames = 0
            lm  = face_landmarks
            ear = self.calculate_ear(lm)
            mar = self.calculate_mar(lm)

            # Head pose (throttled every 3rd frame)
            if self.total_frames % 3 == 0:
                try:
                    p, y_, r = self.get_head_pose(lm, w, h)
                    self.cached_pitch = p
                    self.cached_yaw   = y_
                    self.cached_roll  = r
                    self.pitch_history.append(p)
                except Exception:
                    pass

            # CNN: borderline override + continuous display every N frames
            ear_for_score = ear
            cnn_override  = False
            self._cnn_display_counter += 1

            in_borderline  = (self.cnn_enabled and self.cnn_available and
                              self.ear_threshold - CNN_BORDER_MARGIN < ear
                              < self.ear_threshold + CNN_BORDER_MARGIN)
            for_display    = (self.cnn_available and
                              self._cnn_display_counter >= CNN_DISPLAY_INTERVAL)

            if in_borderline or for_display:
                # Always reset counter on any CNN call
                self._cnn_display_counter = 0
                decision, prob, lp, rp = self._cnn_eye_check(frame, lm, w, h)
                if prob is not None:
                    self.last_cnn_prob  = prob
                    self.last_cnn_left  = lp
                    self.last_cnn_right = rp
                if in_borderline and decision is not None:
                    if decision is True:
                        ear_for_score = self.ear_threshold - 0.025
                        cnn_override  = True
                    elif decision is False:
                        ear_for_score = self.ear_threshold + 0.025
                        cnn_override  = True

            # History
            self.ear_history.append(ear)
            self.mar_history.append(mar)
            eye_closed = ear_for_score < self.ear_threshold
            self.eye_closed_history.append(1 if eye_closed else 0)
            self._update_blink_rate(eye_closed)

            # Auto-calibration
            auto_calib_pct = 0.0
            if self.auto_calibrating:
                if self._auto_calib_settle > 0:
                    self._auto_calib_settle -= 1
                else:
                    self._auto_calib_ears.append(ear)
                    self._auto_calib_mars.append(mar)
                    collected = len(self._auto_calib_ears)
                    auto_calib_pct = min(1.0, collected / AUTO_CALIB_FRAMES)
                    if collected >= AUTO_CALIB_FRAMES:
                        self.finish_auto_calibration()

            # Manual calibration
            if self.calibrating:
                self._calib_frames.append((ear, mar))

            # Frame counters (hard-reset on eyes open/close — no gradual decay)
            if eye_closed:
                self.closed_frame_count += 1
            else:
                self.closed_frame_count = 0

            if mar > self.mar_threshold:
                self.yawn_frame_count += 1
            else:
                self.yawn_frame_count = 0

            # Head nod
            head_nodding = False
            if self.head_nod_enabled:
                head_nodding = self._detect_nod()
            if head_nodding:
                self.head_nod_count += 1
            else:
                self.head_nod_count = max(0, self.head_nod_count - 1)

            # Dominant cause tracking
            self.eye_contribution.append(1 if eye_closed else 0)
            self.yawn_contribution.append(1 if mar > self.mar_threshold else 0)
            self.nod_contribution.append(1 if head_nodding else 0)

            # ── Score update ─────────────────────────────────────────
            event = (eye_closed or mar > self.mar_threshold or head_nodding)

            base_delta = 0.0
            if self.closed_frame_count > self.consec_frames:
                base_delta += 1.5
            if self.yawn_frame_count > 4:
                base_delta += 1.0
            if self.head_nod_count > 8:
                base_delta += 1.0

            duration_bonus = 0.0
            if self.closed_frame_count > 25:
                duration_bonus += min(2.0 + (self.closed_frame_count - 25) * 0.10, 6.0)
            if self.yawn_frame_count > 30:
                duration_bonus += min(1.5 + (self.yawn_frame_count - 30) * 0.08, 5.0)
            if self.head_nod_count > 45:
                duration_bonus += min(2.0 + (self.head_nod_count - 45) * 0.15, 7.0)

            total_delta = base_delta + duration_bonus

            if event:
                self.short_term_score += total_delta if total_delta else 1.5
                self.long_term_score  += 0.4
            else:
                self.short_term_score *= 0.96
                self.long_term_score  *= 0.99

            self.short_term_score = min(120.0, max(0.0, self.short_term_score))
            self.long_term_score  = min(120.0, max(0.0, self.long_term_score))
            self.fatigue_score    = (0.65 * self.short_term_score
                                     + 0.35 * self.long_term_score)
            self.score_history.append(int(self.fatigue_score))

            # ── Break suggestion ─────────────────────────────────────
            now_t         = time.time()
            break_fire    = False
            if self.fatigue_score > BREAK_SCORE_THRESH and not self.alarm_active:
                if self._high_score_since == 0.0:
                    self._high_score_since = now_t
                elif (not self._break_fired and
                      now_t - self._high_score_since > BREAK_TIME_SEC):
                    self._break_fired     = True
                    self._break_suggested = True
                    break_fire            = True
            elif self.fatigue_score < BREAK_SCORE_THRESH:
                self._high_score_since = 0.0
                self._break_suggested  = False
                self._break_fired      = False

            # ── Effective alarm threshold (session context) ───────────
            base_thr = float(self.alarm_on_thresh)
            if elapsed_min < 8:
                effective_thr = base_thr * 1.15
            elif elapsed_min > 90:
                effective_thr = base_thr * 0.90
            else:
                effective_thr = base_thr

            if (self.last_strong_alarm_time and
                    (now_t - self.last_strong_alarm_time) < POST_ALARM_RAISE_SEC):
                effective_thr = max(effective_thr, base_thr * 1.15)

            alarm_off_thr = effective_thr - (self.alarm_on_thresh - self.alarm_off_thresh)

            # ── Hysteresis alarm ─────────────────────────────────────
            alarm_triggered = False
            if not self.alarm_active:
                if self.fatigue_score > effective_thr:
                    self.high_score_counter += 1
                    if self.high_score_counter >= self.alarm_confirm_frames:
                        # Check cooldown
                        if (not self.last_strong_alarm_time or
                                now_t - self.last_strong_alarm_time >= ALARM_COOLDOWN_SEC):
                            self.alarm_active           = True
                            self.last_strong_alarm_time = now_t
                            alarm_triggered             = True
                else:
                    self.high_score_counter = max(0, self.high_score_counter - 1)
            else:
                if self.fatigue_score < alarm_off_thr:
                    self.stop_alarm()

            # Cache for UI
            self.last_ear   = ear
            self.last_mar   = mar
            self.last_pitch = self.cached_pitch

            # ── Build result ─────────────────────────────────────────
            result.face_present    = True
            result.ear             = ear
            result.mar             = mar
            result.ear_for_score   = ear_for_score
            result.pitch           = self.cached_pitch
            result.yaw             = self.cached_yaw
            result.roll            = self.cached_roll
            result.eye_closed      = eye_closed
            result.cnn_override    = cnn_override
            result.cnn_prob        = self.last_cnn_prob
            result.cnn_left        = self.last_cnn_left
            result.cnn_right       = self.last_cnn_right
            result.head_nodding    = head_nodding
            result.auto_calibrating = self.auto_calibrating
            result.auto_calib_pct  = auto_calib_pct
            result.break_suggested = break_fire

        else:
            # ── No face ───────────────────────────────────────────────
            self.no_face_frames += 1
            result.face_present = False

            if self.no_face_frames > NO_FACE_DECAY_FRAMES:
                self.short_term_score *= 0.97
                self.long_term_score  *= 0.995
                self.fatigue_score     = (0.65 * self.short_term_score
                                          + 0.35 * self.long_term_score)

            if self.no_face_frames > NO_FACE_ALARM_STOP and self.alarm_active:
                self.stop_alarm()

        # ── Always-computed fields ────────────────────────────────────
        perclos = self.calc_perclos()
        self.perclos_history.append(perclos)

        result.short_term_score = self.short_term_score
        result.long_term_score  = self.long_term_score
        result.fatigue_score    = self.fatigue_score
        result.alarm_active     = self.alarm_active
        result.alarm_triggered  = alarm_triggered if face_landmarks else False
        result.perclos          = perclos
        result.blink_rate_bpm   = self.blink_rate_bpm
        result.dominant_cause   = self.get_dominant_cause()
        result.ear_trend        = self.ear_trend()

        return result

    # ── CNN state label helper ─────────────────────────────────────────
    @staticmethod
    def cnn_state_text(prob: Optional[float]) -> Tuple[str, str]:
        """
        Returns (label_text, colour_hex) for a sleepy probability.
        e.g. ("Sleepy  72.1%", CLR_RED)
        """
        from config import CLR_RED, CLR_GREEN, CLR_YELLOW, CLR_SUBTEXT
        if prob is None:
            return "—", CLR_SUBTEXT
        if prob > CNN_CLOSED_THRESH:
            return f"Sleepy  {prob * 100:.1f}%", CLR_RED
        if prob < CNN_OPEN_THRESH:
            return f"Awake  {(1.0 - prob) * 100:.1f}%", CLR_GREEN
        return f"Uncertain  {prob * 100:.1f}%", CLR_YELLOW

    # ── Alert message builder ─────────────────────────────────────────
    def get_alert_message(self, cause: str) -> Tuple[str, str, str]:
        """Return (title, body, colour_hex) for an alarm banner."""
        from config import CLR_YELLOW, CLR_ORANGE, CLR_RED
        detail = {
            "eyes":            "Eyes closing more often than usual.",
            "yawning":         "Repeated yawning — a clear sign of fatigue.",
            "nodding":         "Head nodding detected — possible micro-sleeps.",
            "eyes+yawning":    "Eyes closing frequently and repeated yawning.",
            "eyes+nodding":    "Eyes closing and head nodding — strong signals.",
            "yawning+nodding": "Yawning and head nodding together.",
            "mixed":           "Low-level fatigue signals across multiple indicators.",
        }.get(cause, "Fatigue signals detected.")

        s = self.fatigue_score
        if s < 40:
            return ("Mild Fatigue Detected",
                    f"{detail}  Consider a short break in the next 10-15 minutes.",
                    CLR_YELLOW)
        if s < self.alarm_on_thresh:
            return ("Fatigue Is Building Up",
                    f"{detail}  Find a place to stop and rest soon.",
                    CLR_ORANGE)
        return ("Drowsiness Detected — Please Act",
                f"{detail}  Pull over safely and rest for 15-20 minutes.  "
                f"Your reaction time may already be reduced.",
                CLR_RED)
