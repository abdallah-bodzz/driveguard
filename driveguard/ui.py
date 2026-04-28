"""
DriveGuard — Main Application UI
Orchestrates the camera loop, MediaPipe models, detection engine,
recording manager, alarm sound, and all CustomTkinter widgets.
"""

import atexit
import os
import threading
import time
from datetime import datetime
from tkinter import filedialog, messagebox
from typing import Optional

import cv2
import customtkinter as ctk
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image, ImageTk
from pygame import mixer

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from config import (
    # colours
    CLR_BG, CLR_PANEL, CLR_BORDER,
    CLR_GREEN, CLR_YELLOW, CLR_ORANGE, CLR_RED, CLR_BLUE,
    CLR_TEXT, CLR_SUBTEXT,
    # CV colours
    CV_GREEN, CV_YELLOW, CV_ORANGE, CV_RED,
    CV_BLUE, CV_CYAN, CV_WHITE, CV_GRAY,
    # thresholds / timing
    ALARM_ON_THRESH, ALARM_OFF_THRESH, ALARM_COOLDOWN_SEC,
    CNN_DISPLAY_INTERVAL,
    BLINK_RATE_LOW, BLINK_RATE_HIGH,
    RECORDINGS_DIR,
    # paths
    MEDIAPIPE_FACE, MEDIAPIPE_HAND,
    ALARM_SOUND_PATHS, DING_SOUND_PATH,
    CNN_MODEL_PATHS,
    MANUAL_CALIB_FRAMES,
)
from detection import DetectionEngine
from recording import RecordingManager
from utils import fmt_duration, open_folder, now_str, timestamp_filename

# Optional psutil
try:
    import psutil as _psutil
    _PSUTIL_OK = True
except ImportError:
    _PSUTIL_OK = False

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class DriveGuardApp:
    """Main application window."""

    # ══════════════════════════════════════════════════════════════════
    #  Init
    # ══════════════════════════════════════════════════════════════════
    def __init__(self):
        self.root = ctk.CTk()
        self.root.title("DriveGuard  ·  Intelligent Driver Fatigue Assistant")
        self.root.geometry("1440x900")
        self.root.minsize(1200, 780)
        self.root.configure(fg_color=CLR_BG)

        # ── Sub-systems ────────────────────────────────────────────
        self.engine   = DetectionEngine()
        self.recorder = RecordingManager()

        # ── Runtime flags ──────────────────────────────────────────
        self.running       = False
        self.cap           = None
        self.alarm_muted   = False
        self.alarm_playing = False
        self.session_start: Optional[float] = None
        self.last_timestamp = 0

        # ── Alarm state (UI side) ──────────────────────────────────
        self.alarm_active  = False
        self._banner_shown = False
        self._pulse_id     = None

        # ── Temp on-screen message ─────────────────────────────────
        self._temp_msg_text  = ""
        self._temp_msg_until = 0.0

        # ── FPS / CPU ──────────────────────────────────────────────
        self.fps           = 0
        self._frame_count  = 0
        self._prev_fps_t   = time.time()
        self._cpu_percent  = 0.0

        # ── UI throttle ────────────────────────────────────────────
        self._last_ui_t        = 0.0
        self._last_summary_t   = 0.0
        self._last_bar_color   = CLR_GREEN   # prevent redundant configure calls

        # ── Calibration progress (shared with camera panel) ────────
        self._calib_pct = 0.0

        # ── Alarm count badge counter ──────────────────────────────
        self._alarm_badge_count = 0

        # ── CSV path (shown in log) ────────────────────────────────
        self._csv_path: Optional[str] = None

        # ── Audio ──────────────────────────────────────────────────
        mixer.init()
        self.alarm_sound = None
        for p in ALARM_SOUND_PATHS:
            if os.path.exists(p):
                try:
                    self.alarm_sound = mixer.Sound(p)
                    break
                except Exception:
                    pass
        if self.alarm_sound is None:
            print("[DriveGuard] No alarm sound found — visual alarm only.")

        self.ding_sound = None
        if os.path.exists(DING_SOUND_PATH):
            try:
                self.ding_sound = mixer.Sound(DING_SOUND_PATH)
            except Exception:
                pass

        # ── MediaPipe ──────────────────────────────────────────────
        self.face_landmarker = None
        self.hand_landmarker = None
        self._init_mediapipe()

        # ── Register gesture dismiss callback ──────────────────────
        self.engine.set_gesture_dismiss_callback(self._on_gesture_dismiss)

        # ── Log buffer ─────────────────────────────────────────────
        self._log_messages = []

        # ── Build UI ───────────────────────────────────────────────
        self._build_ui()

        # ── Cooldown label dedicated ticker ───────────────────────
        self._update_cooldown_label()

    # ── MediaPipe init ────────────────────────────────────────────────
    def _init_mediapipe(self):
        if not os.path.exists(MEDIAPIPE_FACE):
            messagebox.showerror(
                "Missing Model",
                f"face_landmarker.task not found.\n\nExpected:\n{MEDIAPIPE_FACE}\n\n"
                "Download from the MediaPipe Models page.")
            self.face_landmarker = None
        else:
            try:
                opts = vision.FaceLandmarkerOptions(
                    base_options=python.BaseOptions(model_asset_path=MEDIAPIPE_FACE),
                    running_mode=vision.RunningMode.VIDEO,
                    num_faces=1)
                self.face_landmarker = vision.FaceLandmarker.create_from_options(opts)
            except Exception as exc:
                messagebox.showerror("Startup Error", f"Cannot load face model\n\n{exc}")
                self.face_landmarker = None

        if not os.path.exists(MEDIAPIPE_HAND):
            self.hand_landmarker = None
        else:
            try:
                hand_opts = vision.HandLandmarkerOptions(
                    base_options=python.BaseOptions(model_asset_path=MEDIAPIPE_HAND),
                    running_mode=vision.RunningMode.VIDEO,
                    num_hands=1)
                self.hand_landmarker = vision.HandLandmarker.create_from_options(hand_opts)
            except Exception as exc:
                print(f"[DriveGuard] Hand landmarker not loaded: {exc}")
                self.hand_landmarker = None

    # ══════════════════════════════════════════════════════════════════
    #  UI Construction
    # ══════════════════════════════════════════════════════════════════
    def _build_ui(self):
        # ── Title bar ─────────────────────────────────────────────
        title_bar = ctk.CTkFrame(self.root, height=54, corner_radius=0,
                                 fg_color=CLR_PANEL)
        title_bar.pack(fill="x", side="top")
        title_bar.pack_propagate(False)

        ctk.CTkLabel(title_bar, text="⬡  DriveGuard",
                     font=ctk.CTkFont(family="Courier New", size=22, weight="bold"),
                     text_color=CLR_BLUE).pack(side="left", padx=24, pady=12)

        self.status_pill = ctk.CTkLabel(title_bar, text="● OFFLINE",
                                        font=ctk.CTkFont(size=13, weight="bold"),
                                        text_color=CLR_SUBTEXT)
        self.status_pill.pack(side="left", padx=10)

        # Alarm count badge (⚠ N) — shown after first alarm
        self.alarm_badge_lbl = ctk.CTkLabel(title_bar, text="",
                                            font=ctk.CTkFont(size=12, weight="bold"),
                                            text_color=CLR_ORANGE)
        # packed dynamically

        # REC timer in title bar
        self.rec_timer_lbl = ctk.CTkLabel(title_bar, text="",
                                          font=ctk.CTkFont(size=12, weight="bold"),
                                          text_color=CLR_RED)

        ctk.CTkLabel(title_bar, text="Intelligent Driver Fatigue Assistant",
                     font=ctk.CTkFont(size=12),
                     text_color=CLR_SUBTEXT).pack(side="right", padx=8)
        ctk.CTkLabel(title_bar, text="DriveGuard",
                     font=ctk.CTkFont(size=11, weight="bold"),
                     text_color=CLR_SUBTEXT).pack(side="right", padx=8)

        body = ctk.CTkFrame(self.root, fg_color=CLR_BG, corner_radius=0)
        body.pack(fill="both", expand=True)

        self._build_sidebar(body)
        self._build_main(body)

        # Keyboard shortcuts
        self.root.bind("<space>",  lambda e: self._toggle_mute() if self.alarm_sound else None)
        self.root.bind("r",        lambda e: self._confirm_reset() if self.running else None)
        self.root.bind("R",        lambda e: self._confirm_reset() if self.running else None)
        self.root.bind("<Escape>", lambda e: self.stop_camera() if self.running else None)
        self.root.bind("d",        lambda e: self._dismiss_alarm_manual() if self.alarm_active else None)
        self.root.bind("D",        lambda e: self._dismiss_alarm_manual() if self.alarm_active else None)

    # ──────────────────────────────────────────────────────────────────
    def _build_sidebar(self, parent):
        sb = ctk.CTkFrame(parent, width=295, fg_color=CLR_PANEL, corner_radius=0)
        sb.pack(side="left", fill="y")
        sb.pack_propagate(False)

        scroll = ctk.CTkScrollableFrame(sb, fg_color=CLR_PANEL, corner_radius=0)
        scroll.pack(fill="both", expand=True)

        def section(text):
            ctk.CTkLabel(scroll, text=text.upper(),
                         font=ctk.CTkFont(size=10, weight="bold"),
                         text_color=CLR_SUBTEXT).pack(anchor="w", padx=20, pady=(18, 4))
            ctk.CTkFrame(scroll, height=1, fg_color=CLR_BORDER
                         ).pack(fill="x", padx=20, pady=(0, 10))

        # ── 1. Session ────────────────────────────────────────────
        section("Session")
        self.start_btn = ctk.CTkButton(
            scroll, text="▶  Start Monitoring",
            command=self.start_camera, height=42,
            fg_color=CLR_GREEN, hover_color="#2EA043",
            font=ctk.CTkFont(size=14, weight="bold"), text_color="#0D1117")
        self.start_btn.pack(fill="x", padx=20, pady=4)

        self.stop_btn = ctk.CTkButton(
            scroll, text="⏹  Stop Monitoring",
            command=self.stop_camera, height=42,
            fg_color=CLR_RED, hover_color="#B62324",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="white", state="disabled")
        self.stop_btn.pack(fill="x", padx=20, pady=4)

        ctk.CTkButton(
            scroll, text="↺  Reset Score",
            command=self._confirm_reset, height=36,
            fg_color=CLR_BORDER, hover_color="#444C56",
            font=ctk.CTkFont(size=13)).pack(fill="x", padx=20, pady=4)

        ctk.CTkLabel(scroll,
                     text="Space=mute  ·  R=reset  ·  Esc=stop  ·  D=dismiss",
                     font=ctk.CTkFont(size=9), text_color=CLR_SUBTEXT
                     ).pack(anchor="w", padx=22, pady=(0, 2))

        # ── 2. Driver Status ──────────────────────────────────────
        section("Driver Status")
        self.status_pill_large = ctk.CTkLabel(
            scroll, text="Not Monitoring",
            font=ctk.CTkFont(size=15, weight="bold"),
            fg_color=CLR_BORDER, corner_radius=20, text_color=CLR_SUBTEXT)
        self.status_pill_large.pack(fill="x", padx=20, pady=(0, 6))

        self.fatigue_bar = ctk.CTkProgressBar(scroll, height=14, progress_color=CLR_GREEN)
        self.fatigue_bar.set(0)
        self.fatigue_bar.pack(fill="x", padx=20, pady=(0, 4))

        self.fatigue_risk_lbl = ctk.CTkLabel(
            scroll, text="Awaiting session start",
            font=ctk.CTkFont(size=12), text_color=CLR_SUBTEXT)
        self.fatigue_risk_lbl.pack(anchor="w", padx=20)

        self.reason_lbl = ctk.CTkLabel(
            scroll, text="",
            font=ctk.CTkFont(size=11), text_color=CLR_SUBTEXT,
            wraplength=250, justify="left")
        self.reason_lbl.pack(anchor="w", padx=20, pady=(4, 0))

        self.confidence_lbl = ctk.CTkLabel(
            scroll, text="",
            font=ctk.CTkFont(size=11, weight="bold"), text_color=CLR_SUBTEXT)
        self.confidence_lbl.pack(anchor="w", padx=20, pady=(2, 0))

        # ── 3. Alert Sensitivity ──────────────────────────────────
        section("Alert Sensitivity")
        self.threshold_val_lbl = ctk.CTkLabel(
            scroll, text=str(self.engine.alarm_on_thresh),
            font=ctk.CTkFont(size=28, weight="bold"), text_color=CLR_BLUE)
        self.threshold_val_lbl.pack(pady=(0, 4))

        ctk.CTkLabel(scroll, text="Lower = more sensitive  ·  Higher = less sensitive",
                     font=ctk.CTkFont(size=10), text_color=CLR_SUBTEXT).pack(padx=20)

        self.threshold_slider = ctk.CTkSlider(
            scroll, from_=25, to=85, number_of_steps=30,
            command=self._on_threshold, progress_color=CLR_BLUE)
        self.threshold_slider.set(self.engine.alarm_on_thresh)
        self.threshold_slider.pack(fill="x", padx=20, pady=6)

        # ── 4. Calibration ────────────────────────────────────────
        section("Calibration")
        self.calib_status_lbl = ctk.CTkLabel(
            scroll, text="Auto-calibration runs on session start",
            font=ctk.CTkFont(size=12), text_color=CLR_SUBTEXT, wraplength=250)
        self.calib_status_lbl.pack(anchor="w", padx=20, pady=(0, 6))

        ctk.CTkButton(
            scroll, text="Run Manual Calibration",
            command=self.calibrate_thresholds, height=38,
            fg_color="#1F6FEB", hover_color="#1158C7",
            font=ctk.CTkFont(size=13)).pack(fill="x", padx=20, pady=4)

        self.calib_bar = ctk.CTkProgressBar(scroll, height=8, progress_color=CLR_BLUE)
        self.calib_bar.set(0)
        self.calib_bar.pack(fill="x", padx=20, pady=4)

        self.calib_pct_lbl = ctk.CTkLabel(
            scroll, text="", font=ctk.CTkFont(size=11), text_color=CLR_SUBTEXT)
        self.calib_pct_lbl.pack(anchor="w", padx=20)

        # ── 5. Options ────────────────────────────────────────────
        section("Options")
        self.mute_sw = ctk.CTkSwitch(scroll, text="Mute Alarm",
                                     command=self._toggle_mute,
                                     progress_color=CLR_BLUE)
        self.mute_sw.pack(anchor="w", padx=20, pady=4)
        if self.alarm_sound is None:
            self.mute_sw.configure(state="disabled", text="Mute Alarm (no sound file)")

        self.landmark_sw = ctk.CTkSwitch(scroll, text="Show Face & Hand Landmarks",
                                         command=self._toggle_landmarks,
                                         progress_color=CLR_BLUE)
        self.landmark_sw.select()
        self.landmark_sw.pack(anchor="w", padx=20, pady=4)
        self._draw_landmarks = True

        self.head_sw = ctk.CTkSwitch(scroll, text="Head Nod Detection (experimental)",
                                     progress_color=CLR_BLUE)
        self.head_sw.pack(anchor="w", padx=20, pady=4)

        self.hybrid_sw = ctk.CTkSwitch(scroll, text="CNN Eye Verification",
                                       progress_color=CLR_BLUE)
        if self.engine.cnn_available:
            self.hybrid_sw.select()
        else:
            self.hybrid_sw.configure(state="disabled",
                                     text="CNN Eye Verification (model missing)")
        self.hybrid_sw.pack(anchor="w", padx=20, pady=4)

        self.gesture_sw = ctk.CTkSwitch(scroll, text="Hand Gesture Dismiss",
                                        progress_color=CLR_BLUE)
        if self.hand_landmarker:
            self.gesture_sw.select()
        else:
            self.gesture_sw.configure(state="disabled",
                                      text="Hand Gesture (model missing)")
        self.gesture_sw.pack(anchor="w", padx=20, pady=4)

        self.mirror_sw = ctk.CTkSwitch(scroll, text="Mirror Camera Feed",
                                       progress_color=CLR_BLUE)
        self.mirror_sw.select()   # ON by default — natural mirror view
        self.mirror_sw.pack(anchor="w", padx=20, pady=4)

        self.csv_sw = ctk.CTkSwitch(scroll, text="Save Session CSV",
                                    progress_color=CLR_BLUE)
        self.csv_sw.select()
        self.csv_sw.pack(anchor="w", padx=20, pady=4)

        # ── 6. Metrics ────────────────────────────────────────────
        section("Metrics")

        def simple_metric(label):
            f = ctk.CTkFrame(scroll, fg_color=CLR_BG, corner_radius=6)
            f.pack(fill="x", padx=20, pady=2)
            ctk.CTkLabel(f, text=label, font=ctk.CTkFont(size=11),
                         text_color=CLR_SUBTEXT).pack(side="left", padx=10, pady=6)
            lbl = ctk.CTkLabel(f, text="—",
                               font=ctk.CTkFont(size=13, weight="bold"),
                               text_color=CLR_TEXT)
            lbl.pack(side="right", padx=10)
            return lbl

        self.lbl_session_simple   = simple_metric("Drive Time")
        self.lbl_incidents_simple = simple_metric("Alerts This Session")

        self.advanced_sw = ctk.CTkSwitch(
            scroll, text="Show advanced metrics",
            command=self._toggle_advanced, progress_color=CLR_BLUE)
        self.advanced_sw.pack(anchor="w", padx=20, pady=(8, 4))

        self.advanced_frame = ctk.CTkFrame(scroll, fg_color=CLR_BG, corner_radius=8)

        def adv_metric(label):
            f = ctk.CTkFrame(self.advanced_frame, fg_color="#0D1117", corner_radius=6)
            f.pack(fill="x", padx=10, pady=2)
            ctk.CTkLabel(f, text=label, font=ctk.CTkFont(size=11),
                         text_color=CLR_SUBTEXT).pack(side="left", padx=10, pady=6)
            lbl = ctk.CTkLabel(f, text="—",
                               font=ctk.CTkFont(size=13, weight="bold"),
                               text_color=CLR_TEXT)
            lbl.pack(side="right", padx=10)
            return lbl

        self.lbl_ear        = adv_metric("EAR  (trend)")
        self.lbl_mar        = adv_metric("MAR")
        self.lbl_pitch      = adv_metric("Head Pitch  (°)")
        self.lbl_perclos    = adv_metric("PERCLOS  (%)")
        self.lbl_fps        = adv_metric("FPS")
        self.lbl_session    = adv_metric("Session Time")
        self.lbl_score      = adv_metric("Fatigue Score")
        self.lbl_st_score   = adv_metric("Short-term Score")
        self.lbl_lt_score   = adv_metric("Long-term Score")
        self.lbl_cnn_state  = adv_metric("CNN Eye State")
        self.lbl_cnn_left   = adv_metric("CNN Left Eye")
        self.lbl_cnn_right  = adv_metric("CNN Right Eye")
        self.lbl_blink_rate = adv_metric("Blink Rate (bpm)")

    # ──────────────────────────────────────────────────────────────────
    def _build_main(self, parent):
        main = ctk.CTkFrame(parent, fg_color=CLR_BG, corner_radius=0)
        main.pack(side="right", fill="both", expand=True, padx=12, pady=12)

        # Alert banner (hidden until alarm)
        self.alert_banner = ctk.CTkFrame(main, fg_color=CLR_RED, corner_radius=10)
        self.alert_banner_title = ctk.CTkLabel(
            self.alert_banner, text="",
            font=ctk.CTkFont(size=16, weight="bold"), text_color="white")
        self.alert_banner_title.pack(pady=(10, 2))
        self.alert_banner_body = ctk.CTkLabel(
            self.alert_banner, text="",
            font=ctk.CTkFont(size=12), text_color="white",
            wraplength=800, justify="center")
        self.alert_banner_body.pack(pady=(0, 4))

        btn_row = ctk.CTkFrame(self.alert_banner, fg_color="transparent")
        btn_row.pack(pady=(0, 10))
        ctk.CTkButton(
            btn_row, text="Dismiss  (D)",
            command=self._dismiss_alarm_manual,
            width=110, height=26,
            fg_color="white", hover_color="#DDDDDD",
            text_color=CLR_RED,
            font=ctk.CTkFont(size=11, weight="bold")).pack(side="left", padx=6)

        # Tabs
        self.tabs = ctk.CTkTabview(
            main, fg_color=CLR_PANEL,
            segmented_button_fg_color=CLR_BORDER,
            segmented_button_selected_color=CLR_BLUE)
        self.tabs.pack(fill="both", expand=True)

        for tab in ("Camera", "Live Graphs", "Incident Log", "How It Works"):
            self.tabs.add(tab)

        self._build_camera_tab()
        self._build_graphs_tab()
        self._build_log_tab()
        self._build_help_tab()

    # ──────────────────────────────────────────────────────────────────
    def _build_camera_tab(self):
        """Horizontal split: video feed (left ~70%) + live stats panel (right ~30%)."""
        tab = self.tabs.tab("Camera")
        outer = ctk.CTkFrame(tab, fg_color=CLR_BG, corner_radius=0)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(0, weight=7)
        outer.columnconfigure(1, weight=3)
        outer.rowconfigure(0, weight=1)

        # Left: video feed
        vf = ctk.CTkFrame(outer, fg_color=CLR_PANEL, corner_radius=10)
        vf.grid(row=0, column=0, sticky="nsew", padx=(0, 6))

        self.video_label = ctk.CTkLabel(
            vf,
            text="Camera feed ready\nPress  Start Monitoring  to begin",
            font=ctk.CTkFont(size=16), text_color=CLR_SUBTEXT,
            fg_color=CLR_PANEL, corner_radius=10)
        self.video_label.pack(fill="both", expand=True)

        # Right: live stats panel
        stats = ctk.CTkScrollableFrame(outer, fg_color=CLR_PANEL, corner_radius=10)
        stats.grid(row=0, column=1, sticky="nsew")

        def s_section(text):
            ctk.CTkLabel(stats, text=text.upper(),
                         font=ctk.CTkFont(size=9, weight="bold"),
                         text_color=CLR_SUBTEXT).pack(anchor="w", padx=14, pady=(14, 2))
            ctk.CTkFrame(stats, height=1, fg_color=CLR_BORDER
                         ).pack(fill="x", padx=14, pady=(0, 8))

        def s_row(label, default="—", bold=False):
            f = ctk.CTkFrame(stats, fg_color=CLR_BG, corner_radius=6)
            f.pack(fill="x", padx=14, pady=2)
            ctk.CTkLabel(f, text=label,
                         font=ctk.CTkFont(size=10), text_color=CLR_SUBTEXT
                         ).pack(side="left", padx=8, pady=5)
            lbl = ctk.CTkLabel(f, text=default,
                               font=ctk.CTkFont(size=12,
                                                weight="bold" if bold else "normal"),
                               text_color=CLR_TEXT)
            lbl.pack(side="right", padx=8)
            return lbl

        # Status
        s_section("Status")
        self.cam_status_pill = ctk.CTkLabel(
            stats, text="Not Monitoring",
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color=CLR_BORDER, corner_radius=16, text_color=CLR_SUBTEXT)
        self.cam_status_pill.pack(fill="x", padx=14, pady=(0, 6))

        self.cam_fatigue_bar = ctk.CTkProgressBar(stats, height=10,
                                                  progress_color=CLR_GREEN)
        self.cam_fatigue_bar.set(0)
        self.cam_fatigue_bar.pack(fill="x", padx=14, pady=(0, 4))

        self.cam_risk_lbl = ctk.CTkLabel(stats, text="Awaiting session",
                                         font=ctk.CTkFont(size=11),
                                         text_color=CLR_SUBTEXT)
        self.cam_risk_lbl.pack(anchor="w", padx=14)

        self.cam_cause_lbl = ctk.CTkLabel(
            stats, text="",
            font=ctk.CTkFont(size=10), text_color=CLR_SUBTEXT,
            wraplength=200, justify="left")
        self.cam_cause_lbl.pack(anchor="w", padx=14, pady=(3, 0))

        # Live readings
        s_section("Live Readings")
        self.cam_lbl_score   = s_row("Fatigue Score", bold=True)
        self.cam_lbl_ear     = s_row("EAR")
        self.cam_lbl_mar     = s_row("MAR")
        self.cam_lbl_perclos = s_row("PERCLOS")
        self.cam_lbl_blink   = s_row("Blink Rate")
        self.cam_lbl_cnn     = s_row("CNN Eye State")

        # Session info
        s_section("Session")
        self.cam_lbl_time    = s_row("Drive Time")
        self.cam_lbl_alerts  = s_row("Alerts")
        self.cam_lbl_fps     = s_row("FPS")
        self.cam_lbl_calib   = s_row("Calibration")

        # Quick actions
        s_section("Quick Actions")
        self._dismiss_cam_btn = ctk.CTkButton(
            stats, text="Dismiss Alarm  (D)",
            command=self._dismiss_alarm_manual,
            height=32, fg_color=CLR_BORDER, hover_color="#444C56",
            font=ctk.CTkFont(size=11), state="disabled")
        self._dismiss_cam_btn.pack(fill="x", padx=14, pady=3)

        ctk.CTkButton(stats, text="Reset Score  (R)",
                      command=self._confirm_reset,
                      height=32, fg_color=CLR_BORDER, hover_color="#444C56",
                      font=ctk.CTkFont(size=11)).pack(fill="x", padx=14, pady=3)

    # ──────────────────────────────────────────────────────────────────
    def _build_graphs_tab(self):
        tab = self.tabs.tab("Live Graphs")
        plt.style.use("dark_background")
        self.fig, self.axes = plt.subplots(4, 1, figsize=(8, 6.5), facecolor="#161B22")
        self.fig.tight_layout(pad=2.5)

        for ax in self.axes:
            ax.set_facecolor("#0D1117")
            ax.tick_params(colors=CLR_SUBTEXT, labelsize=8)
            for spine in ax.spines.values():
                spine.set_edgecolor(CLR_BORDER)

        graph_labels = ["EAR", "MAR", "PERCLOS %", "Fatigue Score"]
        for ax, lbl in zip(self.axes, graph_labels):
            ax.set_ylabel(lbl, color=CLR_SUBTEXT, fontsize=9)

        self.axes[3].set_xlabel("frames  (180 ≈ 6 s)", color=CLR_SUBTEXT, fontsize=8)

        self.line_ear,     = self.axes[0].plot([], [], color=CLR_BLUE,   lw=1.5)
        self.line_ear_thr, = self.axes[0].plot([], [], color=CLR_RED,    lw=1,
                                               linestyle="--", label="threshold")
        self.line_mar,     = self.axes[1].plot([], [], color=CLR_YELLOW, lw=1.5)
        self.line_mar_thr, = self.axes[1].plot([], [], color=CLR_RED,    lw=1,
                                               linestyle="--", label="threshold")
        self.line_perclos, = self.axes[2].plot([], [], color=CLR_GREEN,  lw=1.5)
        self.line_fatigue, = self.axes[3].plot([], [], color=CLR_ORANGE, lw=1.5)
        self.line_fat_thr, = self.axes[3].plot([], [], color=CLR_RED,    lw=1,
                                               linestyle="--", label="alarm threshold")

        for ax in self.axes:
            ax.legend(fontsize=7, loc="upper right",
                      facecolor="#161B22", edgecolor=CLR_BORDER, labelcolor=CLR_SUBTEXT)

        self.graph_canvas = FigureCanvasTkAgg(self.fig, master=tab)
        self.graph_canvas.draw()
        self.graph_canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=8)
        self._update_graph()

    # ──────────────────────────────────────────────────────────────────
    def _build_log_tab(self):
        tab = self.tabs.tab("Incident Log")

        # Summary header
        self.log_summary_frame = ctk.CTkFrame(tab, fg_color=CLR_PANEL, corner_radius=8)
        self.log_summary_frame.pack(fill="x", padx=10, pady=(10, 4))
        self.log_summary_lbl = ctk.CTkLabel(
            self.log_summary_frame, text="No session data yet",
            font=ctk.CTkFont(size=12), text_color=CLR_SUBTEXT, justify="left")
        self.log_summary_lbl.pack(side="left", padx=14, pady=8)

        # Sparkline (small score chart) — updated by _refresh_log_summary
        self.sparkline_fig, self.sparkline_ax = plt.subplots(figsize=(6, 0.9),
                                                              facecolor="#161B22")
        self.sparkline_ax.set_facecolor("#0D1117")
        self.sparkline_ax.tick_params(left=False, bottom=False,
                                      labelleft=False, labelbottom=False)
        for spine in self.sparkline_ax.spines.values():
            spine.set_visible(False)
        self.sparkline_line, = self.sparkline_ax.plot([], [], color=CLR_ORANGE, lw=1.5)
        self.sparkline_fig.tight_layout(pad=0.1)

        self.sparkline_canvas = FigureCanvasTkAgg(self.sparkline_fig, master=tab)
        self.sparkline_canvas.draw()
        self.sparkline_canvas.get_tk_widget().pack(fill="x", padx=10, pady=(0, 4))

        # Alarm history table
        self.alarm_table_frame = ctk.CTkFrame(tab, fg_color=CLR_BG, corner_radius=6)
        self.alarm_table_frame.pack(fill="x", padx=10, pady=(0, 4))
        ctk.CTkLabel(self.alarm_table_frame, text="Alarm History",
                     font=ctk.CTkFont(size=11, weight="bold"),
                     text_color=CLR_SUBTEXT).pack(anchor="w", padx=10, pady=(6, 2))

        self.alarm_table_inner = ctk.CTkScrollableFrame(self.alarm_table_frame,
                                                         fg_color=CLR_BG, height=80,
                                                         corner_radius=4)
        self.alarm_table_inner.pack(fill="x", padx=8, pady=(0, 6))
        self._alarm_table_rows = []   # list of CTkLabel row frames

        # Toolbar
        top = ctk.CTkFrame(tab, fg_color=CLR_BG, corner_radius=0)
        top.pack(fill="x", padx=10, pady=(0, 4))
        ctk.CTkLabel(top, text="Event Log",
                     font=ctk.CTkFont(size=14, weight="bold"),
                     text_color=CLR_TEXT).pack(side="left")
        ctk.CTkButton(top, text="Export Log", width=90, height=28,
                      fg_color="#1F6FEB", hover_color="#1158C7",
                      command=self._export_log,
                      font=ctk.CTkFont(size=12)).pack(side="right", padx=4)
        ctk.CTkButton(top, text="Clear Log", width=90, height=28,
                      fg_color=CLR_BORDER, hover_color="#444C56",
                      command=self._clear_log,
                      font=ctk.CTkFont(size=12)).pack(side="right", padx=4)

        self.log_text = ctk.CTkTextbox(
            tab, state="disabled",
            font=ctk.CTkFont(family="Courier New", size=12),
            fg_color=CLR_PANEL, text_color=CLR_TEXT, corner_radius=8)
        self.log_text.pack(fill="both", expand=True, padx=10, pady=(0, 10))

    # ──────────────────────────────────────────────────────────────────
    def _build_help_tab(self):
        tab = self.tabs.tab("How It Works")
        scroll = ctk.CTkScrollableFrame(tab, fg_color=CLR_BG, corner_radius=0)
        scroll.pack(fill="both", expand=True, padx=16, pady=10)

        def heading(text):
            ctk.CTkLabel(scroll, text=text,
                         font=ctk.CTkFont(size=15, weight="bold"),
                         text_color=CLR_BLUE, anchor="w").pack(anchor="w", pady=(14, 2))

        def para(text):
            ctk.CTkLabel(scroll, text=text,
                         font=ctk.CTkFont(size=13), text_color=CLR_TEXT,
                         wraplength=780, justify="left", anchor="w"
                         ).pack(anchor="w", pady=(0, 8))

        heading("What DriveGuard Monitors")
        para("Eye openness (EAR) — frequent or prolonged closing indicates fatigue.\n"
             "Yawning (MAR) — repeated yawns are a strong tiredness signal.\n"
             "Blink rate — normal is 15-20 blinks/min; below 8 bpm is a drowsiness indicator.\n"
             "Head nodding (pitch velocity) — forward head movement (experimental, off by default).\n"
             "Hand gestures — wave or fist to silence the alarm hands-free.")

        heading("How Alerts Work")
        para("DriveGuard uses two scores: a short-term score (reacts quickly) and a long-term "
             "score (builds over minutes). An alarm fires only after the combined score stays "
             "high for 2 full seconds. It turns off only when the score drops significantly — "
             "no flickering. There is a minimum 30-second cooldown between consecutive alarms.")

        heading("Severity Levels")
        para("Green  — Fully Attentive: normal driving, no signals detected.\n"
             "Yellow — Mild Fatigue: early signs, consider a break in 10-15 minutes.\n"
             "Orange — High Fatigue: clear fatigue building, rest recommended now.\n"
             "Red    — Critical: strong drowsiness detected, pull over safely.")

        heading("Auto-Calibration")
        para("During the first 30 seconds of monitoring, DriveGuard silently measures your "
             "normal eye openness and resting mouth position. No action needed. Thresholds "
             "are set specifically for your face. You can also run guided manual calibration "
             "from the sidebar.")

        heading("CNN Eye Verification (Hybrid Mode)")
        para("A lightweight neural network (98.8% validation accuracy, trained on 84,000 eye "
             "images) runs every 5 frames on both eyes independently. In borderline EAR cases "
             "it overrides the raw value. Advanced Metrics shows per-eye state (Left / Right) "
             "plus an overall CNN Eye State with confidence percentage. A CNN badge appears on "
             "the camera feed when an override is active.")

        heading("Hand Gesture Dismiss")
        para("Wave your hand left-right in front of the camera (3+ direction reversals), or "
             "hold a closed fist for about 5 frames. The alarm silences immediately and the "
             "fatigue score drops. When Show Landmarks is ON, yellow dots track the hand.")

        heading("Alert Sensitivity Slider")
        para("Lower = more sensitive (alarms trigger faster). Higher = requires stronger or "
             "longer fatigue signals before triggering. Default 48 is a good balance.")

        heading("Mirror Camera Feed")
        para("When enabled (default), the video feed is displayed as a mirror image — what "
             "you see matches how you expect to see yourself. Detection logic is unaffected.")

        heading("Incident Recording")
        para("When an alarm triggers, the system saves a 5-second pre-event video clip (MP4) "
             "at 640x360 to the recordings folder, plus a full session CSV for post-drive "
             "analysis. The REC timer is visible in the title bar and on the camera feed.")

        heading("Keyboard Shortcuts")
        para("Space — mute / unmute alarm\n"
             "D     — dismiss active alarm\n"
             "R     — reset fatigue score\n"
             "Esc   — stop monitoring")

        heading("Important Disclaimer")
        para("DriveGuard is a driver assistance tool — not a substitute for responsible "
             "driving. Always obey traffic laws. If you feel tired, stop and rest regardless "
             "of what the system shows.")

        ctk.CTkButton(scroll, text="Open Recordings Folder",
                      command=lambda: open_folder(RECORDINGS_DIR),
                      width=200, height=34,
                      fg_color=CLR_BORDER, hover_color="#444C56",
                      font=ctk.CTkFont(size=12)).pack(anchor="w", pady=(8, 4))

        footer = ctk.CTkFrame(scroll, fg_color=CLR_PANEL, corner_radius=8)
        footer.pack(fill="x", pady=(20, 4))
        ctk.CTkLabel(footer, text="DriveGuard  ·  Intelligent Driver Fatigue Assistant",
                     font=ctk.CTkFont(size=12, weight="bold"),
                     text_color=CLR_TEXT).pack(side="left", padx=14, pady=8)
        ctk.CTkLabel(footer, text="MediaPipe  ·  CNN  ·  Dual-score  ·  Gesture dismiss",
                     font=ctk.CTkFont(size=11), text_color=CLR_SUBTEXT
                     ).pack(side="right", padx=14)

    # ══════════════════════════════════════════════════════════════════
    #  Alarm control
    # ══════════════════════════════════════════════════════════════════
    def _trigger_alarm(self):
        """Called when the engine reports alarm_triggered=True."""
        self.alarm_active = True
        cause = self.engine.get_dominant_cause()
        title, body, col = self.engine.get_alert_message(cause)

        self._alarm_badge_count += 1
        score = int(self.engine.fatigue_score)
        self.recorder.log_incident(score)
        self._log(f"ALARM triggered — score {score} — cause: {cause}")
        self._update_alarm_badge()

        def _show():
            self.alert_banner.configure(fg_color=col)
            self.alert_banner_title.configure(text=title)
            self.alert_banner_body.configure(text=body)
            if not self._banner_shown:
                self.alert_banner.pack(fill="x", pady=(0, 8), before=self.tabs)
                self._banner_shown = True
            if self._pulse_id:
                self.root.after_cancel(self._pulse_id)
                self._pulse_id = None
            self._pulse_banner(col, 0)
            self._dismiss_cam_btn.configure(state="normal")

        self.root.after(0, _show)

        if not self.alarm_muted and self.alarm_sound and not self.alarm_playing:
            self.alarm_sound.play(-1)
            self.alarm_playing = True

        # Start recording if face was present recently
        self._maybe_start_recording()

    def _stop_alarm_ui(self):
        """Hide alarm UI when the engine reports the alarm has stopped."""
        self.alarm_active = False
        if self._pulse_id:
            self.root.after_cancel(self._pulse_id)
            self._pulse_id = None
        if self.alarm_playing and self.alarm_sound:
            self.alarm_sound.stop()
            self.alarm_playing = False

        def _hide():
            if self._banner_shown:
                self.alert_banner.pack_forget()
                self._banner_shown = False
            self._dismiss_cam_btn.configure(state="disabled")

        self.root.after(0, _hide)

    def _dismiss_alarm_manual(self):
        if not self.alarm_active:
            return
        self.engine.dismiss()
        self._stop_alarm_ui()
        self.alarm_active = False
        self._log("Alarm dismissed via button / keyboard")

    def _on_gesture_dismiss(self, gesture_type: str):
        if not self.alarm_active:
            return
        self.engine.dismiss()
        self._stop_alarm_ui()
        self.alarm_active = False
        self._log(f"Alarm dismissed by {gesture_type} gesture")
        self._temp_msg_text  = f"Gesture recognised — alarm silenced  ({gesture_type})"
        self._temp_msg_until = time.time() + 3.0
        if self.ding_sound:
            try:
                self.ding_sound.play()
            except Exception:
                pass

    def _pulse_banner(self, base_col, state):
        if not self.alarm_active or not self._banner_shown:
            self._pulse_id = None
            return
        _pairs = {
            CLR_RED:    ("#F85149", "#C0302A"),
            CLR_ORANGE: ("#E07B39", "#A85A28"),
            CLR_YELLOW: ("#D29922", "#9A700F"),
        }
        pair = _pairs.get(base_col, (base_col, base_col))
        self.alert_banner.configure(fg_color=pair[state % 2])
        self._pulse_id = self.root.after(600,
                                         lambda: self._pulse_banner(base_col, state + 1))

    def _update_alarm_badge(self):
        n = self._alarm_badge_count
        if n > 0:
            self.alarm_badge_lbl.configure(text=f"⚠ {n}")
            self.alarm_badge_lbl.pack(side="left", padx=4)
        else:
            try:
                self.alarm_badge_lbl.pack_forget()
            except Exception:
                pass

    # ══════════════════════════════════════════════════════════════════
    #  Recording helpers
    # ══════════════════════════════════════════════════════════════════
    def _maybe_start_recording(self):
        """Start recording if not already, and face is present."""
        pass   # called from process_frames where we have the frame

    def _start_recording(self, frame):
        if self.recorder.recording:
            return
        path = self.recorder.start_incident_recording(frame)
        if path:
            self._log(f"Recording started → {path}")
            self.root.after(0, lambda: self.rec_timer_lbl.pack(side="left", padx=4))
            self.root.after(0, self._update_rec_timer)

    def _stop_recording(self):
        if not self.recorder.recording:
            return
        self.recorder.stop_incident_recording()
        self._log("Recording saved")
        self.root.after(0, lambda: (
            self.rec_timer_lbl.pack_forget(),
            self.rec_timer_lbl.configure(text="")))

    def _update_rec_timer(self):
        if not self.recorder.recording:
            return
        self.rec_timer_lbl.configure(text=f"REC  {self.recorder.rec_elapsed:02d}s")
        self.root.after(1000, self._update_rec_timer)

    # ══════════════════════════════════════════════════════════════════
    #  Calibration (UI-facing)
    # ══════════════════════════════════════════════════════════════════
    def calibrate_thresholds(self):
        if not self.running:
            messagebox.showwarning("Camera Off", "Start monitoring before calibrating.")
            return
        if self.engine.auto_calibrating:
            messagebox.showwarning("Auto-Calibrating",
                                   "Please wait for auto-calibration to finish (~30 s).")
            return
        if self.engine.calibrating:
            return
        if not messagebox.askokcancel("Manual Calibration",
                                      "Instructions:\n"
                                      "  1. Keep eyes WIDE open for 8 s\n"
                                      "  2. Close your eyes gently for 8 s\n"
                                      "  3. Yawn naturally 2-3 times\n\n"
                                      "Press OK to start (~20 s)."):
            return
        self.engine.start_manual_calibration()
        self.calib_bar.set(0)
        self.calib_pct_lbl.configure(text="Calibrating… 0%")
        self.calib_status_lbl.configure(text="Calibrating…", text_color=CLR_BLUE)
        self._log("Manual calibration started")

    def _finish_manual_calibration(self):
        ok = self.engine.finish_manual_calibration()
        self.calib_bar.set(1.0)
        if ok:
            self.calib_pct_lbl.configure(text="Complete")
            self.calib_status_lbl.configure(
                text=f"Calibrated — EAR={self.engine.ear_threshold:.3f}  "
                     f"MAR={self.engine.mar_threshold:.3f}",
                text_color=CLR_GREEN)
            self._log(f"Manual calibration done → EAR={self.engine.ear_threshold:.3f}"
                      f"  MAR={self.engine.mar_threshold:.3f}")
            messagebox.showinfo("Calibration Complete",
                                f"EAR = {self.engine.ear_threshold:.3f}\n"
                                f"MAR = {self.engine.mar_threshold:.3f}")
        else:
            self.calib_status_lbl.configure(text="Calibration failed", text_color=CLR_RED)
            self._log("Calibration failed: no data")

    def _finish_auto_calibration(self):
        self.calib_bar.set(1.0)
        self.calib_pct_lbl.configure(text="Auto-calibrated")
        self.calib_status_lbl.configure(
            text=f"Auto-calibrated\n"
                 f"EAR={self.engine.ear_threshold:.3f}  "
                 f"MAR={self.engine.mar_threshold:.3f}",
            text_color=CLR_GREEN)
        self._temp_msg_text  = "Auto-calibration complete — thresholds tuned to your face"
        self._temp_msg_until = time.time() + 4.0
        self._log(f"Auto-calibration done → EAR={self.engine.ear_threshold:.3f}"
                  f"  MAR={self.engine.mar_threshold:.3f}")

    # ══════════════════════════════════════════════════════════════════
    #  Main processing loop  (runs in background thread)
    # ══════════════════════════════════════════════════════════════════
    def process_frames(self):
        while self.running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]

            # Mirror flip for natural webcam view
            if self.mirror_sw.get():
                frame = cv2.flip(frame, 1)

            # Pre-event buffer push
            self.recorder.push_frame(frame)

            # MediaPipe timestamp
            ts_ms = int(time.time() * 1000)
            if ts_ms <= self.last_timestamp:
                ts_ms = self.last_timestamp + 1
            self.last_timestamp = ts_ms

            # Face detection
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            face_lm = None
            try:
                face_result = self.face_landmarker.detect_for_video(mp_img, ts_ms)
                if face_result and face_result.face_landmarks:
                    face_lm = face_result.face_landmarks[0]
            except Exception:
                face_lm = None

            # Sync option switches to engine
            self.engine.cnn_enabled        = bool(self.hybrid_sw.get())
            self.engine.head_nod_enabled   = bool(self.head_sw.get())

            # Process frame in engine
            elapsed_min = ((time.time() - self.session_start) / 60.0
                           if self.session_start else 0.0)
            res = self.engine.process(frame, face_lm, w, h,
                                      elapsed_min, self.session_start or time.time())

            # ── Calibration completions ────────────────────────────
            if res.auto_calib_pct > 0:
                self._calib_pct = res.auto_calib_pct
                self.root.after(0, lambda p=res.auto_calib_pct: (
                    self.calib_bar.set(p),
                    self.calib_pct_lbl.configure(
                        text=f"Auto-calibrating… {int(p * 100)}%")))
            if not res.auto_calibrating and self.engine.auto_calibrating is False:
                # auto-calib just finished (transition)
                pass  # handled below via previous flag

            if (not res.auto_calibrating and
                    getattr(self, "_prev_auto_calibrating", True)):
                self.root.after(0, self._finish_auto_calibration)
            self._prev_auto_calibrating = res.auto_calibrating

            if (not self.engine.calibrating and
                    getattr(self, "_was_calibrating", False)):
                self.root.after(0, self._finish_manual_calibration)
            self._was_calibrating = self.engine.calibrating

            # Manual calibration progress
            if self.engine.calibrating:
                pct = len(self.engine._calib_frames) / MANUAL_CALIB_FRAMES
                self.root.after(0, lambda p=pct: (
                    self.calib_bar.set(p),
                    self.calib_pct_lbl.configure(
                        text=f"Calibrating… {int(p * 100)}%")))
                cv2.putText(frame, f"CALIBRATING {int(pct * 100)}%",
                            (w // 2 - 160, 70), cv2.FONT_HERSHEY_SIMPLEX,
                            1.1, CV_BLUE, 3)

            # ── Alarm state sync ──────────────────────────────────
            if res.alarm_triggered and not self.alarm_active:
                self.root.after(0, self._trigger_alarm)
            elif (not res.alarm_active and self.alarm_active and not res.alarm_triggered):
                self.root.after(0, self._stop_alarm_ui)

            # ── Recording ─────────────────────────────────────────
            if res.alarm_active and res.face_present:
                if not self.recorder.recording:
                    self._start_recording(frame)
            if self.recorder.recording:
                self.recorder.write_frame(frame)
                if self.recorder.tick_post_alarm(res.alarm_active,
                                                  self.engine.no_face_frames):
                    self.root.after(0, self._stop_recording)

            # REC blink dot (no timer text — title bar handles that)
            self.recorder.draw_rec_dot(frame)

            # ── Break suggestion ──────────────────────────────────
            if res.break_suggested:
                elapsed_drive = int((time.time() - self.session_start) / 60)
                self._temp_msg_text  = (f"Fatigue building — consider a break soon "
                                        f"({elapsed_drive} min into drive)")
                self._temp_msg_until = time.time() + 6.0
                self._log("Break suggestion shown")

            # ── Landmarks on frame ────────────────────────────────
            if self._draw_landmarks and face_lm:
                for pt in face_lm:
                    cv2.circle(frame, (int(pt.x * w), int(pt.y * h)),
                               1, (0, 200, 100), -1)

            # ── Overlay ───────────────────────────────────────────
            self._draw_overlay(frame, w, h, res)

            # ── Hand gesture ──────────────────────────────────────
            if self.gesture_sw.get() and self.hand_landmarker:
                try:
                    hres = self.hand_landmarker.detect_for_video(mp_img, ts_ms)
                    if hres.hand_landmarks:
                        hand_lm = hres.hand_landmarks[0]
                        self.engine.process_hand_gesture(hand_lm, w, h)
                        if self._draw_landmarks:
                            for pt in hand_lm:
                                cv2.circle(frame,
                                           (int(pt.x * w), int(pt.y * h)),
                                           2, CV_CYAN, -1)
                except Exception:
                    pass

            # ── CSV logging ───────────────────────────────────────
            if self.csv_sw.get() and self._csv_path:
                self.recorder.log_csv_row(res)

            # ── FPS ───────────────────────────────────────────────
            self._frame_count += 1
            now = time.time()
            if now - self._prev_fps_t >= 1.0:
                self.fps         = self._frame_count
                self._frame_count = 0
                self._prev_fps_t  = now
                if _PSUTIL_OK:
                    try:
                        self._cpu_percent = _psutil.cpu_percent(interval=None)
                    except Exception:
                        pass

            # FPS overlay
            fps_text = f"{self.fps} FPS"
            if _PSUTIL_OK and self._cpu_percent > 0:
                fps_text += f"  CPU {self._cpu_percent:.0f}%"
            cv2.putText(frame, fps_text, (10, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, CV_GRAY, 1)

            # ── Push to UI (throttled ~30 fps) ────────────────────
            if now - self._last_ui_t >= 0.033:
                self._last_ui_t = now
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img       = Image.fromarray(frame_rgb)
                imgtk     = ImageTk.PhotoImage(image=img)
                self.root.after(0, self._refresh_ui, imgtk, res)

        # Loop ended — camera disconnected or stopped
        if self.running:
            self.root.after(0, self._handle_camera_disconnect)

    # ──────────────────────────────────────────────────────────────────
    def _handle_camera_disconnect(self):
        """Called if cap.read() fails mid-session (cable pull, driver crash)."""
        self._log("Camera disconnected unexpectedly")
        self.stop_camera()
        messagebox.showwarning("Camera Disconnected",
                               "The camera feed was lost.\n"
                               "Please reconnect your webcam and press Start.")

    # ──────────────────────────────────────────────────────────────────
    def _draw_overlay(self, frame, w, h, res):
        """Draw status box, fatigue bar, alarm border, and temp messages."""
        from config import CV_RED, CV_ORANGE, CV_YELLOW, CV_GREEN, CV_GRAY, CV_WHITE, CV_CYAN

        # Status
        s = res.fatigue_score
        if res.alarm_active:
            status_text, cv_col = "DROWSY",       CV_RED
        elif s >= 40:
            status_text, cv_col = "HIGH FATIGUE", CV_ORANGE
        elif s >= 25:
            status_text, cv_col = "CAUTION",      CV_YELLOW
        else:
            status_text, cv_col = "ATTENTIVE",    CV_GREEN

        # Status box (bottom-right)
        bx1, by1 = w - 230, h - 110
        bx2, by2 = w - 10,  h - 10
        overlay = frame.copy()
        cv2.rectangle(overlay, (bx1, by1), (bx2, by2), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        cv2.putText(frame, status_text, (bx1 + 10, by1 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, cv_col, 2)

        bar_w  = bx2 - bx1 - 20
        filled = int((min(s, 120.0) / 120.0) * bar_w)
        cv2.rectangle(frame, (bx1 + 10, by1 + 40), (bx2 - 10, by1 + 52), CV_GRAY, 1)
        if filled > 0:
            cv2.rectangle(frame, (bx1 + 10, by1 + 41),
                          (bx1 + 10 + filled, by1 + 51), cv_col, -1)

        cv2.putText(frame, f"Score {int(s)}", (bx1 + 10, by1 + 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, CV_WHITE, 1)

        if self.session_start:
            elapsed_min = (time.time() - self.session_start) / 60.0
            cv2.putText(frame, f"Drive {elapsed_min:.0f} min",
                        (bx1 + 10, by1 + 84),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, CV_WHITE, 1)

        # Cooldown indicator (only when frame is tall enough to avoid overlap)
        if (self.engine.last_strong_alarm_time and not res.alarm_active and h > 500):
            cd = max(0.0, ALARM_COOLDOWN_SEC - (time.time() - self.engine.last_strong_alarm_time))
            if cd > 0:
                cv2.putText(frame, f"Cooldown {int(cd)}s",
                            (bx1 + 10, by1 + 98),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.34, CV_YELLOW, 1)

        # Alarm border + text
        if res.alarm_active:
            cv2.rectangle(frame, (0, 0), (w, h), CV_RED, 18)
            cv2.putText(frame, "  DROWSINESS DETECTED — STAY SAFE",
                        (w // 2 - 270, h - 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, CV_RED, 3)

        # Head nod
        if res.head_nodding:
            cv2.putText(frame, "HEAD NOD", (16, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, CV_ORANGE, 2)

        # CNN override badge
        if res.cnn_override:
            cv2.putText(frame, "CNN", (w - 48, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, CV_CYAN, 1)

        # Temp message (gesture ack, break suggestion, auto-calib done)
        if time.time() < self._temp_msg_until:
            cv2.putText(frame, self._temp_msg_text, (16, h - 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, CV_GREEN, 2)

    # ══════════════════════════════════════════════════════════════════
    #  UI refresh  (main thread)
    # ══════════════════════════════════════════════════════════════════
    def _refresh_ui(self, imgtk, res):
        self.video_label.configure(image=imgtk, text="")
        self.video_label.image = imgtk

        s = res.fatigue_score

        # ── State colours ─────────────────────────────────────────
        if not self.running:
            pill_text  = "Not Monitoring"
            pill_color = CLR_BORDER
            pill_tc    = CLR_SUBTEXT
            risk_text  = "Awaiting session start"
            bar_color  = CLR_GREEN
        elif self.alarm_active or s >= self.engine.alarm_on_thresh:
            pill_text  = "Critical — Take a Break"
            pill_color = CLR_RED
            pill_tc    = "white"
            risk_text  = "Immediate Action Needed"
            bar_color  = CLR_RED
        elif s >= 40:
            pill_text  = "High Fatigue"
            pill_color = CLR_ORANGE
            pill_tc    = "white"
            risk_text  = "High Risk — Rest Soon"
            bar_color  = CLR_ORANGE
        elif s >= 25:
            pill_text  = "Mild Fatigue"
            pill_color = CLR_YELLOW
            pill_tc    = "#0D1117"
            risk_text  = "Moderate Risk"
            bar_color  = CLR_YELLOW
        else:
            pill_text  = "Fully Attentive"
            pill_color = CLR_GREEN
            pill_tc    = "#0D1117"
            risk_text  = "Low Risk"
            bar_color  = CLR_GREEN

        # ── Sidebar status ─────────────────────────────────────────
        self.status_pill_large.configure(text=pill_text,
                                         fg_color=pill_color, text_color=pill_tc)
        self.fatigue_bar.set(min(s / 120.0, 1.0))
        # Only call configure when colour actually changes (Bug 5 fix)
        if bar_color != self._last_bar_color:
            self.fatigue_bar.configure(progress_color=bar_color)
            self._last_bar_color = bar_color
        self.fatigue_risk_lbl.configure(text=risk_text)

        if s > 20 and self.running:
            cause = res.dominant_cause
            cause_detail = self.engine.get_alert_message(cause)[1].split(".")[0] + "."
            self.reason_lbl.configure(text=cause_detail, text_color=CLR_SUBTEXT)
            conf     = "High" if s > 55 else "Medium" if s > 30 else "Low"
            conf_col = CLR_RED if conf == "High" else CLR_YELLOW if conf == "Medium" else CLR_GREEN
            self.confidence_lbl.configure(text=f"Detection confidence: {conf}",
                                          text_color=conf_col)
        else:
            self.confidence_lbl.configure(text="")
            self.reason_lbl.configure(text="")

        # ── Advanced metrics ───────────────────────────────────────
        ear_col = CLR_RED if res.ear < self.engine.ear_threshold else CLR_GREEN
        mar_col = CLR_RED if res.mar > self.engine.mar_threshold else CLR_GREEN
        pit_col = CLR_RED if abs(res.pitch) > self.engine.pitch_threshold else CLR_GREEN
        scr_col = CLR_RED if self.alarm_active else CLR_YELLOW if s > 25 else CLR_GREEN

        trend = res.ear_trend
        self.lbl_ear.configure(text=f"{res.ear:.3f}  {trend}", text_color=ear_col)
        self.lbl_mar.configure(text=f"{res.mar:.3f}", text_color=mar_col)
        self.lbl_pitch.configure(text=f"{res.pitch:.1f}°", text_color=pit_col)

        if len(self.engine.perclos_history) < 30:
            self.lbl_perclos.configure(text="—", text_color=CLR_SUBTEXT)
        else:
            self.lbl_perclos.configure(text=f"{res.perclos}%")

        self.lbl_fps.configure(text=str(self.fps))
        self.lbl_score.configure(text=f"{int(s)}", text_color=scr_col)
        self.lbl_st_score.configure(text=f"{res.short_term_score:.1f}", text_color=scr_col)
        self.lbl_lt_score.configure(text=f"{res.long_term_score:.1f}", text_color=scr_col)

        cnn_text, cnn_col = DetectionEngine.cnn_state_text(res.cnn_prob)
        self.lbl_cnn_state.configure(text=cnn_text, text_color=cnn_col)
        lt, lc = DetectionEngine.cnn_state_text(res.cnn_left)
        rt, rc = DetectionEngine.cnn_state_text(res.cnn_right)
        self.lbl_cnn_left.configure(text=lt, text_color=lc)
        self.lbl_cnn_right.configure(text=rt, text_color=rc)

        # Blink rate
        bpm = res.blink_rate_bpm
        session_elapsed = (time.time() - self.session_start) if self.session_start else 0
        if bpm < 1:
            blink_text = "Warming up…" if session_elapsed < 60 else "—"
            blink_col  = CLR_SUBTEXT
        elif bpm < BLINK_RATE_LOW:
            blink_text, blink_col = f"{bpm:.0f} bpm  (low)", CLR_RED
        elif bpm > BLINK_RATE_HIGH:
            blink_text, blink_col = f"{bpm:.0f} bpm  (high)", CLR_YELLOW
        else:
            blink_text, blink_col = f"{bpm:.0f} bpm", CLR_GREEN
        self.lbl_blink_rate.configure(text=blink_text, text_color=blink_col)

        if self.session_start:
            elapsed = int(time.time() - self.session_start)
            dur_str = fmt_duration(elapsed)
            self.lbl_session.configure(text=dur_str)
            self.lbl_session_simple.configure(text=dur_str)

        incidents = self.recorder.incidents
        self.lbl_incidents_simple.configure(text=str(len(incidents)))

        # ── Camera panel ───────────────────────────────────────────
        self.cam_status_pill.configure(text=pill_text,
                                       fg_color=pill_color, text_color=pill_tc)
        self.cam_fatigue_bar.set(min(s / 120.0, 1.0))
        if bar_color != getattr(self, "_last_cam_bar_color", None):
            self.cam_fatigue_bar.configure(progress_color=bar_color)
            self._last_cam_bar_color = bar_color
        self.cam_risk_lbl.configure(text=risk_text)

        if s > 20 and self.running:
            # Use the detail from alert message (already extracted above)
            pass
        self.cam_cause_lbl.configure(
            text="" if s <= 20 else res.dominant_cause.replace("+", " + "))

        self.cam_lbl_score.configure(text=str(int(s)), text_color=scr_col)
        self.cam_lbl_ear.configure(text=f"{res.ear:.3f}  {trend}", text_color=ear_col)
        self.cam_lbl_mar.configure(text=f"{res.mar:.3f}", text_color=mar_col)

        if len(self.engine.perclos_history) < 30:
            self.cam_lbl_perclos.configure(text="—", text_color=CLR_SUBTEXT)
        else:
            pc = CLR_RED if res.perclos > 40 else CLR_YELLOW if res.perclos > 20 else CLR_GREEN
            self.cam_lbl_perclos.configure(text=f"{res.perclos}%", text_color=pc)

        self.cam_lbl_blink.configure(text=blink_text, text_color=blink_col)
        self.cam_lbl_cnn.configure(text=cnn_text, text_color=cnn_col)

        if self.session_start:
            elapsed = int(time.time() - self.session_start)
            self.cam_lbl_time.configure(text=fmt_duration(elapsed))

        self.cam_lbl_alerts.configure(text=str(len(incidents)))
        self.cam_lbl_fps.configure(text=str(self.fps))

        if res.auto_calibrating:
            calib_pct_str = f"Auto-calibrating… {int(self._calib_pct * 100)}%"
            self.cam_lbl_calib.configure(text=calib_pct_str, text_color=CLR_YELLOW)
        else:
            self.cam_lbl_calib.configure(text="Calibrated", text_color=CLR_GREEN)

        # ── Title bar status pill ──────────────────────────────────
        if self.alarm_active:
            self.status_pill.configure(text="● DROWSY", text_color=CLR_RED)
        elif s >= 40:
            self.status_pill.configure(text="● CAUTION", text_color=CLR_ORANGE)
        elif s >= 25:
            self.status_pill.configure(text="● CAUTION", text_color=CLR_YELLOW)
        elif self.running:
            self.status_pill.configure(text="● LIVE", text_color=CLR_GREEN)

        # ── Log summary (throttled 1/s) ────────────────────────────
        now = time.time()
        if self.running and now - self._last_summary_t >= 1.0:
            self._last_summary_t = now
            self._refresh_log_summary()

    # ──────────────────────────────────────────────────────────────────
    def _refresh_log_summary(self):
        incidents = self.recorder.incidents
        scores    = self.recorder.score_history
        max_s     = max(scores) if scores else 0

        if len(self.engine.perclos_history) >= 30:
            pa = sum(self.engine.perclos_history) / len(self.engine.perclos_history)
            ps = f"{pa:.1f}%"
        else:
            ps = "—"

        txt = (f"Alerts: {len(incidents)}   |   "
               f"Max score: {max_s}   |   "
               f"Avg PERCLOS: {ps}")
        if incidents:
            last_t, last_s = incidents[-1]
            txt += f"   |   Last alert: {last_t} (score {last_s})"
        self.log_summary_lbl.configure(text=txt, text_color=CLR_TEXT)

        # Sparkline update
        try:
            sc = self.recorder.score_history
            if len(sc) > 2:
                self.sparkline_line.set_data(range(len(sc)), sc)
                self.sparkline_ax.set_xlim(0, len(sc))
                self.sparkline_ax.set_ylim(0, 120)
                self.sparkline_canvas.draw_idle()
        except Exception:
            pass

        # Alarm history table
        self._refresh_alarm_table(incidents)

    def _refresh_alarm_table(self, incidents):
        """Rebuild alarm history rows (only when incident list changes)."""
        if len(incidents) == len(self._alarm_table_rows):
            return
        # Add new rows only
        for i in range(len(self._alarm_table_rows), len(incidents)):
            ts, sc = incidents[i]
            row = ctk.CTkFrame(self.alarm_table_inner, fg_color="#0D1117", corner_radius=4)
            row.pack(fill="x", padx=4, pady=1)
            ctk.CTkLabel(row, text=f"{ts}  —  score {sc}",
                         font=ctk.CTkFont(family="Courier New", size=11),
                         text_color=CLR_TEXT).pack(side="left", padx=8, pady=3)
            self._alarm_table_rows.append(row)

    # ══════════════════════════════════════════════════════════════════
    #  Graph update
    # ══════════════════════════════════════════════════════════════════
    def _update_graph(self):
        try:
            ear_list     = list(self.engine.ear_history)
            mar_list     = list(self.engine.mar_history)
            perclos_list = list(self.engine.perclos_history)
            score_list   = list(self.engine.score_history)

            n = len(ear_list)
            if n > 2:
                xs = list(range(n))
                self.line_ear.set_data(xs, ear_list)
                self.line_ear_thr.set_data([0, n], [self.engine.ear_threshold] * 2)
                self.axes[0].set_xlim(0, n)
                self.axes[0].set_ylim(0, 0.5)

                mn = len(mar_list)
                self.line_mar.set_data(range(mn), mar_list)
                self.line_mar_thr.set_data([0, mn], [self.engine.mar_threshold] * 2)
                self.axes[1].set_xlim(0, mn)
                self.axes[1].set_ylim(0, 1.2)

                pn = len(perclos_list)
                self.line_perclos.set_data(range(pn), perclos_list)
                self.axes[2].set_xlim(0, pn)
                self.axes[2].set_ylim(0, 100)

                sn = len(score_list)
                if sn > 2:
                    self.line_fatigue.set_data(range(sn), score_list)
                    self.line_fat_thr.set_data([0, sn],
                                               [self.engine.alarm_on_thresh] * 2)
                    self.axes[3].set_xlim(0, sn)
                    self.axes[3].set_ylim(0, 120)

                self.graph_canvas.draw_idle()
        except Exception:
            pass
        self.root.after(1000, self._update_graph)

    # ══════════════════════════════════════════════════════════════════
    #  Cooldown label (dedicated 1-s ticker — lighter than 30-fps refresh)
    # ══════════════════════════════════════════════════════════════════
    def _update_cooldown_label(self):
        if (self.engine.last_strong_alarm_time and
                not self.alarm_active and self.running):
            cd = max(0.0, ALARM_COOLDOWN_SEC -
                     (time.time() - self.engine.last_strong_alarm_time))
            if cd > 0:
                self.reason_lbl.configure(text=f"Alarm cooldown: {int(cd)}s",
                                          text_color=CLR_SUBTEXT)
        self.root.after(1000, self._update_cooldown_label)

    # ══════════════════════════════════════════════════════════════════
    #  Logging
    # ══════════════════════════════════════════════════════════════════
    def _log(self, msg: str):
        entry = f"[{now_str()}]  {msg}\n"
        self._log_messages.append(entry)

        def _write():
            self.log_text.configure(state="normal")
            if len(self._log_messages) > 300:
                self._log_messages = self._log_messages[-300:]
                self.log_text.delete("1.0", "end")
                self.log_text.insert("end", "".join(self._log_messages))
            else:
                self.log_text.insert("end", entry)
            self.log_text.see("end")
            self.log_text.configure(state="disabled")

        self.root.after(0, _write)

    def _clear_log(self):
        self._log_messages.clear()
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")

    def _export_log(self):
        if not self._log_messages:
            messagebox.showinfo("Export Log", "No log entries to export.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            initialfile=f"driveguard_log_{timestamp_filename()}.txt",
            title="Export Event Log")
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write("DriveGuard — Event Log Export\n")
                f.write(f"Exported: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 60 + "\n\n")
                f.writelines(self._log_messages)
                incidents = self.recorder.incidents
                if incidents:
                    f.write("\n" + "=" * 60 + "\n")
                    f.write("INCIDENT SUMMARY\n")
                    for t, sc in incidents:
                        f.write(f"  {t}  —  score {sc}\n")
            messagebox.showinfo("Export Log", f"Log exported to:\n{path}")
        except Exception as exc:
            messagebox.showerror("Export Failed", str(exc))

    # ══════════════════════════════════════════════════════════════════
    #  UI controls
    # ══════════════════════════════════════════════════════════════════
    def _on_threshold(self, val):
        t = int(val)
        self.engine.alarm_on_thresh  = t
        self.engine.alarm_off_thresh = max(10, t - 13)
        self.threshold_val_lbl.configure(text=str(t))

    def _toggle_mute(self):
        self.alarm_muted = bool(self.mute_sw.get())
        if self.alarm_muted and self.alarm_playing and self.alarm_sound:
            self.alarm_sound.stop()
            self.alarm_playing = False

    def _toggle_landmarks(self):
        self._draw_landmarks = bool(self.landmark_sw.get())

    def _toggle_advanced(self):
        if bool(self.advanced_sw.get()):
            self.advanced_frame.pack(fill="x", padx=20, pady=4)
        else:
            self.advanced_frame.pack_forget()

    def _confirm_reset(self):
        if messagebox.askyesno("Reset Score", "Reset fatigue score and all counters?"):
            self.engine.reset_scores()
            self._stop_alarm_ui()
            self.alarm_active = False
            self._log("Score and counters reset by user")

    # ══════════════════════════════════════════════════════════════════
    #  Camera control
    # ══════════════════════════════════════════════════════════════════
    def start_camera(self):
        if self.running:
            return
        if self.face_landmarker is None:
            messagebox.showerror("Model Missing",
                                 "Face landmarker model not loaded.\n"
                                 "Place face_landmarker.task in the mediapipe-tasks/ folder.")
            return

        self.start_btn.configure(state="disabled")
        self.video_label.configure(text="Initialising camera…")
        self.root.update()

        # Try camera indices 0, 1, 2
        self.cap = None
        for idx in [0, 1, 2]:
            try:
                cap_attempt = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
                if cap_attempt.isOpened():
                    self.cap = cap_attempt
                    break
                cap_attempt.release()
            except Exception:
                pass

        if self.cap is None or not self.cap.isOpened():
            for idx in [0, 1, 2]:
                try:
                    cap_attempt = cv2.VideoCapture(idx)
                    if cap_attempt.isOpened():
                        self.cap = cap_attempt
                        break
                    cap_attempt.release()
                except Exception:
                    pass

        if self.cap is None or not self.cap.isOpened():
            messagebox.showerror("Camera Error", "Cannot open webcam. Tried indices 0, 1, 2.")
            self.start_btn.configure(state="normal")
            self.video_label.configure(
                text="Camera feed ready\nPress  Start Monitoring  to begin")
            return

        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Full reset
        self.running       = True
        self.session_start = time.time()
        self.last_timestamp = int(time.time() * 1000)
        self.alarm_active  = False
        self._alarm_badge_count = 0
        self._last_bar_color    = CLR_GREEN
        self._last_ui_t         = 0.0
        self._last_summary_t    = 0.0
        self._calib_pct         = 0.0
        self._prev_auto_calibrating = True
        self._was_calibrating       = False
        self.fps             = 0
        self._frame_count    = 0
        self._prev_fps_t     = time.time()

        self.engine.reset_session()
        self.recorder.reset_session()
        self._log_messages.clear()
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")
        self._alarm_table_rows.clear()
        for w in self.alarm_table_inner.winfo_children():
            w.destroy()

        # Open CSV
        if self.csv_sw.get():
            self._csv_path = self.recorder.open_session_csv(self.session_start)
            if self._csv_path:
                self._log(f"Session CSV → {self._csv_path}")

        self.calib_bar.set(0)
        self.calib_pct_lbl.configure(text="")
        self.calib_status_lbl.configure(text="Auto-calibrating (30 s)…",
                                         text_color=CLR_YELLOW)

        self.stop_btn.configure(state="normal")
        self.status_pill.configure(text="● LIVE", text_color=CLR_GREEN)
        self.video_label.configure(text="")
        self._update_alarm_badge()

        self._log("Monitoring started")

        def _warmup_then_run():
            for _ in range(10):
                self.cap.read()
            self.process_frames()

        threading.Thread(target=_warmup_then_run, daemon=True).start()

    def stop_camera(self):
        self.running      = False
        self.alarm_active = False
        if self.alarm_playing and self.alarm_sound:
            self.alarm_sound.stop()
            self.alarm_playing = False
        if self._pulse_id:
            self.root.after_cancel(self._pulse_id)
            self._pulse_id = None
        if self.recorder.recording:
            self._stop_recording()
        if self._banner_shown:
            self.alert_banner.pack_forget()
            self._banner_shown = False

        # Session summary
        if self.session_start:
            dur    = int(time.time() - self.session_start)
            m, sec = divmod(dur, 60)
            scores = self.recorder.score_history
            mx     = max(scores) if scores else 0
            avg    = sum(scores) / len(scores) if scores else 0.0
            self._log(f"Session ended — {m:02d}:{sec:02d} drive, "
                      f"{len(self.recorder.incidents)} alert(s), "
                      f"max score {mx}, avg score {avg:.1f}")

        self.recorder.close_session_csv(self.session_start)
        self._csv_path = None

        if self.cap:
            self.cap.release()
            self.cap = None

        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.status_pill.configure(text="● OFFLINE", text_color=CLR_SUBTEXT)
        self.status_pill_large.configure(text="Not Monitoring",
                                          fg_color=CLR_BORDER, text_color=CLR_SUBTEXT)

        # Reset progress bar colour (Bug 8 fix)
        self.fatigue_bar.set(0)
        self.fatigue_bar.configure(progress_color=CLR_GREEN)
        self._last_bar_color = CLR_GREEN

        self.fatigue_risk_lbl.configure(text="Session ended")
        self.reason_lbl.configure(text="")
        self.confidence_lbl.configure(text="")

        # Clear video label properly (Bug 9 fix)
        self.video_label.configure(
            image="",
            text="Camera stopped\nPress  Start Monitoring  to resume")
        self.video_label.image = None

        # Reset camera panel
        self.cam_status_pill.configure(text="Not Monitoring",
                                        fg_color=CLR_BORDER, text_color=CLR_SUBTEXT)
        self.cam_fatigue_bar.set(0)
        self.cam_fatigue_bar.configure(progress_color=CLR_GREEN)
        self.cam_risk_lbl.configure(text="Session ended")
        self.cam_cause_lbl.configure(text="")
        for lbl in (self.cam_lbl_score, self.cam_lbl_ear, self.cam_lbl_mar,
                    self.cam_lbl_perclos, self.cam_lbl_blink, self.cam_lbl_cnn,
                    self.cam_lbl_time, self.cam_lbl_alerts, self.cam_lbl_fps):
            lbl.configure(text="—", text_color=CLR_TEXT)
        self.cam_lbl_calib.configure(text="—", text_color=CLR_TEXT)
        self._dismiss_cam_btn.configure(state="disabled")

        self.log_summary_lbl.configure(text="Session ended — see log above",
                                        text_color=CLR_SUBTEXT)

    # ══════════════════════════════════════════════════════════════════
    #  Entry point
    # ══════════════════════════════════════════════════════════════════
    def on_closing(self):
        if self.running:
            if messagebox.askokcancel("Exit", "Monitoring is active. Stop and exit?"):
                self.stop_camera()
                self.root.destroy()
        else:
            self.root.destroy()

    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self._log("DriveGuard initialised")
        self.root.mainloop()
