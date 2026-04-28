# DriveGuard User Guide

Welcome to DriveGuard – your intelligent driver fatigue assistant.  
This guide will help you install, configure, and use the system effectively.

---

## Table of Contents

1. [Installation](#installation)
2. [First Launch & Auto‑Calibration](#first-launch--auto-calibration)
3. [Monitoring Session](#monitoring-session)
4. [Understanding Alerts](#understanding-alerts)
5. [Hand Gestures & Keyboard Shortcuts](#hand-gestures--keyboard-shortcuts)
6. [Incident Recording & CSV Export](#incident-recording--csv-export)
7. [Advanced Settings](#advanced-settings)
8. [Troubleshooting](#troubleshooting)
9. [FAQ](#faq)
10. [Safety Disclaimer](#safety-disclaimer)

---

## Installation

### Prerequisites
- Windows, macOS, or Linux
- Python 3.9 – 3.11
- A working webcam (built‑in or external)

### Steps

1. Download or clone the DriveGuard repository.
2. Open a terminal (Command Prompt, PowerShell, or bash) in the project folder.
3. Install dependencies:
   `ash
   pip install -r requirements.txt
   `
4. Run the application:
   `ash
   python run.py
   `

> **Note**: On first run, the app will look for MediaPipe task files and the CNN model in the ssets/ folder. These are included – no extra downloads needed.

---

## First Launch & Auto‑Calibration

When you click **Start Monitoring**, DriveGuard automatically calibrates itself:

- **Settle period (first 3 seconds)** – camera stabilises.
- **Data collection (next 27 seconds)** – measures your normal eye openness (EAR) and resting mouth position (MAR).
- **Threshold calculation** – personalised EAR and MAR thresholds are set.

> **What you should do during auto‑calibration:**  
> Sit naturally, look straight at the camera, and blink normally. Do not try to keep your eyes wide open – the system needs your typical values.

After 30 seconds, the calibration status will change to “Calibrated”. You are now ready for real monitoring.

### Manual Calibration (Optional)

If you want even more accuracy, you can run a guided manual calibration:
1. Click **Run Manual Calibration** in the sidebar.
2. Follow the on‑screen instructions:
   - Keep eyes **wide open** for 8 seconds
   - **Close your eyes gently** for 8 seconds
   - **Yawn naturally** 2‑3 times
3. The app will compute personalised thresholds and show a confirmation.

Manual calibration overrides auto‑calibration values.

---

## Monitoring Session

While monitoring, the main **Camera** tab shows:

- **Left side:** Live video feed (mirrored by default – natural view)
- **Right side:** Live stats panel – current fatigue score, EAR, MAR, PERCLOS, blink rate, CNN eye state, drive time, alerts count.

The sidebar shows similar information plus advanced metrics (when enabled).

### Fatigue Score & Severity Levels

| Score Range | Severity | Action |
|-------------|----------|--------|
| 0 – 24 | 🟢 Fully Attentive | Normal driving. |
| 25 – 39 | 🟡 Mild Fatigue | Consider a break in 10‑15 minutes. |
| 40 – 47 | 🟠 High Fatigue | Find a place to rest soon. |
| 48 – 120 | 🔴 Critical | Pull over safely and rest 15‑20 minutes. |

The score is updated in real time. It combines a **short‑term** component (reacts quickly to sustained events) and a **long‑term** component (builds slowly over minutes).

---

## Understanding Alerts

An alarm triggers only when the fatigue score stays **above 48 for at least 2 seconds**. This prevents false alarms from single blinks or yawns.

When an alarm fires:
- The screen shows a red border and a banner explaining the cause (e.g., “Eyes closing frequently”).
- An audible alarm plays (unless muted).
- Incident recording begins (saves the last 5 seconds before the alarm and continues until 3 seconds after the alarm stops).

The alarm automatically stops when the score drops **below 35** (hysteresis). You can also dismiss it manually.

---

## Hand Gestures & Keyboard Shortcuts

### Hand Gesture Dismiss (Hands‑Free)

When the alarm is active, you can silence it without touching the computer:

- **Wave left‑right** in front of the camera (3+ direction reversals).
- **Hold a closed fist** for about 5 frames (≈0.2 seconds).

When a gesture is recognised:
- The alarm stops immediately.
- A green message appears on the camera feed: “Gesture recognised – alarm silenced”.
- A short **ding** sound plays (if sound file present).

> **Tip:** Enable **Show Face & Hand Landmarks** to see yellow dots tracking your hand – this confirms the system sees you.

### Keyboard Shortcuts (Always Active)

| Key | Action |
|-----|--------|
| Space | Mute / unmute the audible alarm |
| D    | Dismiss active alarm (same as clicking the banner button) |
| R    | Reset fatigue score to zero |
| Esc  | Stop monitoring |

---

## Incident Recording & CSV Export

### Video Recordings

Whenever an alarm triggers, DriveGuard saves an MP4 file in the ecordings/ folder. The clip includes:
- The **5 seconds before** the alarm (pre‑event buffer)
- The entire alarm period
- **3 seconds after** the alarm stops

File name format: incident_YYYYMMDD_HHMMSS.mp4

### Session CSV

If **Save Session CSV** is enabled (default), a CSV file is created at the start of each monitoring session. It contains one row per second with:
- 	ime_sec, EAR, MAR, atigue_score, short_term, long_term, larm_active, dominant_cause, perclos, link_rate_bpm

At the end of the session, a summary line is appended with duration, number of alerts, max score, average score, and critical seconds.

To open the recordings folder, go to the **How It Works** tab and click **Open Recordings Folder**.

---

## Advanced Settings

Click **Show advanced metrics** in the sidebar to reveal technical details:
- EAR (with trend arrow), MAR, head pitch
- PERCLOS (percentage of closed‑eye frames)
- Short‑term and long‑term scores
- CNN eye state (Sleepy / Awake / Uncertain with percentage)
- Per‑eye CNN confidence
- Blink rate (bpm)

Other options:

| Option | Description |
|--------|-------------|
| **Mute Alarm** | Disables the audible alarm (visual alerts still appear). |
| **Show Face & Hand Landmarks** | Draws green dots on face and yellow dots on hand. |
| **Head Nod Detection (experimental)** | Enables pitch‑velocity nod detection (off by default – may be unreliable). |
| **CNN Eye Verification** | Enables the neural network for borderline EAR cases. |
| **Hand Gesture Dismiss** | Enables wave / fist detection (requires hand model). |
| **Mirror Camera Feed** | Flips the video horizontally (natural mirror view). |
| **Save Session CSV** | Creates a CSV file for each session. |

### Alert Sensitivity Slider

Adjusts the **activation threshold** (default 48).  
- **Lower** = more sensitive (alarms trigger faster / at lower scores).  
- **Higher** = less sensitive (requires stronger fatigue signals).

The deactivation threshold is always 13 points lower than the activation threshold (hysteresis).

---

## Troubleshooting

| Problem | Likely Cause | Solution |
|---------|--------------|----------|
| Camera not opening | Wrong camera index | The app tries indices 0, 1, 2 automatically. Ensure no other app is using the webcam. |
| “face_landmarker.task not found” | Missing MediaPipe model | Download the file from MediaPipe and place in ssets/mediapipe/. |
| No face detected | Poor lighting, face out of frame | Adjust lighting, sit facing the camera. |
| CNN confidence always “—” | CNN model not loaded or hybrid switch off | Check that ssets/models/eye_classifier.h5 exists and the “CNN Eye Verification” switch is ON. |
| Alarm does not trigger | Sensitivity too high, or you are not showing sustained fatigue | Lower the sensitivity slider, or deliberately close your eyes for 2‑3 seconds. |
| High CPU usage | CNN running every frame on borderline EAR | (Potential bug – can be fixed by code change. For now, ensure lighting is good so EAR is not constantly near threshold.) |

---

## FAQ

**Q: Does DriveGuard work at night?**  
A: Yes, but performance depends on webcam quality and infrared sensitivity. In very low light, the system may struggle to detect landmarks.

**Q: Can I use it with glasses?**  
A: Yes – the CNN hybrid mode is specifically designed to handle glasses and reflections.

**Q: Does it work with sunglasses?**  
A: No – sunglasses block eye detection. Remove them while driving.

**Q: How do I know the app is actually doing something?**  
A: The live stats panel shows EAR, MAR, and the fatigue score. The CNN confidence updates every few frames. You can also enable landmarks to see face dots.

**Q: What does “PERCLOS” mean?**  
A: Percentage of eyelid closure over time – a standard drowsiness metric. Values above 40% indicate high risk.

**Q: Can I use the app for fleet management?**  
A: Yes – session CSV files can be collected and analysed centrally.

---

## Safety Disclaimer

⚠️ **DriveGuard is an assistance tool, not a substitute for responsible driving.**

- Always obey traffic laws.
- If you feel tired, stop and rest – regardless of what the system shows.
- Do not rely solely on the app to keep you awake.
- The system may have false positives or false negatives.
- The authors assume no liability for any incidents arising from the use of this software.

Drive safe!

---

## Need Help?

- Check the **How It Works** tab inside the app for an interactive guide.
- Open an issue on the project repository (if available).
- Refer to the pi_reference.md for developer documentation.

*Thank you for using DriveGuard.*
