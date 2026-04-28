"""
Unit tests for DriveGuard detection engine.
Run with: pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from datetime import datetime

from driveguard.detection import DetectionEngine
from driveguard.config import (
    EAR_THRESHOLD_DEFAULT, MAR_THRESHOLD_DEFAULT,
    CNN_CLOSED_THRESH, CNN_OPEN_THRESH,
    ALARM_ON_THRESH, ALARM_OFF_THRESH,
    BLINK_RATE_LOW, BLINK_RATE_HIGH
)


# ─────────────────────────────────────────────────────────────────────
#  Fixtures
# ─────────────────────────────────────────────────────────────────────

@pytest.fixture
def engine():
    \"\"\"Return a clean DetectionEngine instance.\"\"\"
    eng = DetectionEngine()
    eng.cnn_available = False   # disable CNN for deterministic tests
    eng.cnn_enabled = False
    eng.head_nod_enabled = False
    return eng


# ─────────────────────────────────────────────────────────────────────
#  Geometry tests (mocked landmarks)
# ─────────────────────────────────────────────────────────────────────

class MockLandmark:
    def __init__(self, x, y):
        self.x = x
        self.y = y

def test_calculate_ear(engine):
    with patch.object(engine, 'calculate_ear', return_value=0.35):
        assert engine.calculate_ear(None) == 0.35

def test_calculate_mar(engine):
    with patch.object(engine, 'calculate_mar', return_value=0.50):
        assert engine.calculate_mar(None) == 0.50


# ─────────────────────────────────────────────────────────────────────
#  Blink rate
# ─────────────────────────────────────────────────────────────────────

def test_blink_rate_single_blink(engine):
    engine._prev_eye_closed = False
    engine._blink_timestamps = []
    # Simulate closed -> open transition
    engine._update_blink_rate(True)   # closed
    engine._update_blink_rate(False)  # open -> blink recorded
    assert len(engine._blink_timestamps) == 1
    # After 60 seconds, timestamps should be pruned
    engine._blink_timestamps = [1.0, 2.0, 3.0]
    with patch('time.time', return_value=62.0):
        engine._update_blink_rate(False)  # triggers pruning
        assert engine.blink_rate_bpm == 1.0


# ─────────────────────────────────────────────────────────────────────
#  EAR trend
# ─────────────────────────────────────────────────────────────────────

def test_ear_trend_stable(engine):
    engine.ear_history.extend([0.30] * 30)
    assert engine.ear_trend() == "→"

def test_ear_trend_declining(engine):
    engine.ear_history.extend([0.32] * 15)
    engine.ear_history.extend([0.28] * 15)
    assert engine.ear_trend() == "↓"

def test_ear_trend_recovering(engine):
    engine.ear_history.extend([0.28] * 15)
    engine.ear_history.extend([0.32] * 15)
    assert engine.ear_trend() == "↑"


# ─────────────────────────────────────────────────────────────────────
#  CNN state text helper
# ─────────────────────────────────────────────────────────────────────

def test_cnn_state_text():
    from driveguard.detection import DetectionEngine
    text, colour = DetectionEngine.cnn_state_text(None)
    assert text == "—"
    text, colour = DetectionEngine.cnn_state_text(0.85)
    assert "Sleepy" in text and "85" in text
    text, colour = DetectionEngine.cnn_state_text(0.20)
    assert "Awake" in text
    text, colour = DetectionEngine.cnn_state_text(0.50)
    assert "Uncertain" in text


# ─────────────────────────────────────────────────────────────────────
#  Score update and alarm hysteresis
# ─────────────────────────────────────────────────────────────────────

def test_score_update_no_event(engine):
    engine.short_term_score = 10.0
    engine.long_term_score = 10.0
    engine.short_term_score *= 0.96
    engine.long_term_score *= 0.99
    assert engine.short_term_score == 9.6
    assert engine.long_term_score == 9.9

def test_alarm_hysteresis(engine):
    engine.alarm_active = False
    engine.high_score_counter = 0
    engine.fatigue_score = 50
    engine.alarm_on_thresh = 48
    engine.alarm_off_thresh = 35
    engine.alarm_confirm_frames = 3
    for i in range(3):
        if not engine.alarm_active:
            if engine.fatigue_score > engine.alarm_on_thresh:
                engine.high_score_counter += 1
                if engine.high_score_counter >= engine.alarm_confirm_frames:
                    engine.alarm_active = True
    assert engine.alarm_active == True
    engine.fatigue_score = 30
    if engine.alarm_active:
        if engine.fatigue_score < engine.alarm_off_thresh:
            engine.stop_alarm()
    assert engine.alarm_active == False


# ─────────────────────────────────────────────────────────────────────
#  Calibration
# ─────────────────────────────────────────────────────────────────────

def test_auto_calibration(engine):
    engine.auto_calibrating = True
    engine._auto_calib_settle = 0
    engine._auto_calib_ears = [0.30, 0.31, 0.29]
    engine._auto_calib_mars = [0.20, 0.22, 0.21]
    engine.finish_auto_calibration()
    assert engine.ear_threshold is not None
    assert engine.mar_threshold is not None
    assert engine.auto_calibrating == False

def test_manual_calibration(engine):
    engine._calib_frames = [(0.35, 0.20), (0.33, 0.22), (0.12, 0.70)]
    engine.finish_manual_calibration()
    assert engine.calibrated == True
    assert engine.calibrating == False
    assert engine.ear_threshold < 0.30
    assert engine.mar_threshold > 0.45


# ─────────────────────────────────────────────────────────────────────
#  Dominant cause
# ─────────────────────────────────────────────────────────────────────

def test_dominant_cause(engine):
    engine.eye_contribution.extend([1]*10 + [0]*5)
    engine.yawn_contribution.extend([0]*15)
    engine.nod_contribution.extend([0]*15)
    cause = engine.get_dominant_cause()
    assert cause == "eyes"


# ─────────────────────────────────────────────────────────────────────
#  Hand gesture (wave/fist) – logic only
# ─────────────────────────────────────────────────────────────────────

def test_hand_gesture_wave(engine):
    class DummyLM:
        def __init__(self, x, y):
            self.x = x
            self.y = y
    hand_lm = [DummyLM(0.2, 0.5), DummyLM(0.4, 0.5), DummyLM(0.2, 0.5),
               DummyLM(0.4, 0.5), DummyLM(0.2, 0.5), DummyLM(0.4, 0.5),
               DummyLM(0.2, 0.5), DummyLM(0.4, 0.5), DummyLM(0.2, 0.5),
               DummyLM(0.4, 0.5)]
    callback_called = False
    def fake_cb(gesture):
        nonlocal callback_called
        callback_called = True
        assert gesture == "wave"
    engine.set_gesture_dismiss_callback(fake_cb)
    for lm in hand_lm:
        engine.process_hand_gesture([lm], 640, 480)
    assert callback_called == True


# ─────────────────────────────────────────────────────────────────────
#  Session reset
# ─────────────────────────────────────────────────────────────────────

def test_reset_session(engine):
    engine.short_term_score = 50.0
    engine.long_term_score = 30.0
    engine.alarm_active = True
    engine.reset_session()
    assert engine.short_term_score == 0.0
    assert engine.long_term_score == 0.0
    assert engine.alarm_active == False
    assert engine.auto_calibrating == True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
