# DriveGuard - Real-Time Driver Fatigue Detection

[![Python Version](https://img.shields.io/badge/python-3.9--3.11-blue)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Code style](https://img.shields.io/badge/code%20style-black-black)](https://github.com/psf/black)
[![Tests](https://github.com/abdallah-bodzz/driveguard/actions/workflows/tests.yml/badge.svg)](https://github.com/abdallah-bodzz/driveguard/actions)

DriveGuard is a computer vision system that detects driver drowsiness using a standard webcam. It combines geometric facial metrics with a lightweight convolutional neural network to achieve reliable fatigue detection without specialized hardware. The system runs entirely locally with no cloud dependencies.

## Table of Contents

- [System Overview](#system-overview)
- [Technical Architecture](#technical-architecture)
- [Detection Algorithms](#detection-algorithms)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Performance Characteristics](#performance-characteristics)
- [Project Structure](#project-structure)
- [Testing](#testing)
- [License](#license)

## System Overview

DriveGuard monitors three primary fatigue indicators:

1. Eye closure intensity (EAR metric with CNN verification)
2. Yawning frequency (MAR metric)  
3. Head nodding motion (pitch velocity)

These signals feed into a dual-timescale scoring model that balances immediate reaction time against false positive resistance. The system triggers visual and audible alarms when sustained fatigue patterns are detected.

### Key Capabilities

- Real-time processing at 25-30 FPS on standard laptop hardware
- Auto-calibration to individual facial geometry within 30 seconds
- Hand gesture recognition for alarm dismissal without touch input
- Incident recording with 5-second pre-event buffer
- Session data export for post-drive analysis
- Fully local operation - no network transmission of video data

## Technical Architecture

### Core Components

| Component | Implementation | Purpose |
|-----------|---------------|---------|
| Face detection | MediaPipe Face Landmarker (478 points) | Facial feature localization |
| Eye state | EAR + CNN hybrid | Closed/open classification |
| Mouth state | MAR threshold | Yawn detection |
| Head pose | solvePnP with 6-point 3D model | Pitch/yaw/roll estimation |
| Fatigue scoring | Dual-timescale integrator | Alarm decision logic |
| UI layer | CustomTkinter | Real-time visualization |
| Recording | Ring buffer + MP4 writer | Incident capture |

### Data Flow

1. Camera capture → frame preprocessing (mirror, resize)
2. MediaPipe inference → 478 facial landmarks
3. Detection engine calculates EAR, MAR, head pose
4. CNN runs on borderline EAR cases (configurable interval)
5. Score integrator updates short-term and long-term scores
6. Hysteresis logic determines alarm state
7. UI updates and recording manager writes incident clips

## Detection Algorithms

### Eye Aspect Ratio (EAR)

```
EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
```

Where p1-p6 are six landmarks around each eye. The value decreases as the eye closes, reaching near-zero when fully closed. Individual thresholds are calibrated per user.

### CNN Hybrid Verification

The geometric EAR measurement faces two limitations:
- Susceptibility to partial occlusion (glasses, hair)
- Person-dependent baseline variation

DriveGuard addresses this with a 24x24 grayscale CNN trained on 84,000 eye images (MRL Eye Dataset). The model achieves 98.8% validation accuracy and runs inference in 3-5ms per eye on CPU. The hybrid logic:

- If EAR is outside borderline range (threshold +/- 4%), use EAR directly
- If EAR is borderline, query CNN for final decision
- CNN display runs every 5 frames for UI feedback regardless

### Dual-Score Fatigue Model

Short-term score (65% weight):
- Rate: +1.5 per 6 consecutive closed frames, decays at 0.96x per frame
- Duration bonuses: +2.0 at 25 frames, ramping to +6.0 at 85 frames
- Purpose: rapid response to acute drowsiness

Long-term score (35% weight):  
- Rate: +0.4 per fatigue event, decays at 0.99x per frame
- Purpose: persistent fatigue tracking across minutes

Combined score = 0.65*short + 0.35*long (capped at 120)

### Hysteresis Alarm Logic

| Parameter | Default | Description |
|-----------|---------|-------------|
| Alarm on threshold | 48 | Score required to trigger |
| Alarm off threshold | 35 | Score required to clear |
| Confirmation frames | 60 (2 seconds) | Sustained score requirement |
| Cooldown | 30 seconds | Minimum between alarms |
| Post-alarm raise | 600 seconds | Temporarily raises threshold |

Session context adjustments:
- First 8 minutes: threshold increased 15% (warm-up period)
- After 90 minutes: threshold decreased 10% (fatigue accumulation)

### Head Pose Estimation

Six canonical 3D points mapped to corresponding 2D landmarks:

| 3D Point (mm) | Landmark Index |
|---------------|----------------|
| (0, 0, 0) | 1 (nose tip) |
| (0, -30, -30) | 152 (chin) |
| (-30, -10, -30) | 33 (left eye corner) |
| (30, -10, -30) | 362 (right eye corner) |
| (-20, -5, -30) | 78 (left mouth corner) |
| (20, -5, -30) | 308 (right mouth corner) |

solvePnP with iterative method yields pitch/yaw/roll. Pitch velocity over 5 frames >2.5 deg/frame indicates nodding.

## Installation

### System Requirements

- Operating System: Windows 10/11, Linux (Ubuntu 20.04+), macOS 11+
- Python: 3.9, 3.10, or 3.11
- RAM: 4GB minimum, 8GB recommended
- CPU: Intel Core i5-8250U or equivalent (2017+)
- GPU: Optional for CNN inference (OpenCL or CUDA)
- Webcam: 720p minimum, 30 FPS recommended

### Dependencies

Core packages (requirements.txt):
```
opencv-python==4.8.1
mediapipe==0.10.7
tensorflow==2.13.0
customtkinter==5.2.1
Pillow==10.0.0
numpy==1.24.3
pygame==2.5.2
matplotlib==3.7.2
```

Development dependencies (requirements_dev.txt):
```
pytest==7.4.0
pytest-cov==4.1.0
black==23.7.0
flake8==6.1.0
```

### Setup Instructions

Windows (PowerShell):
```powershell
git clone https://github.com/abdallah-bodzz/driveguard.git
cd driveguard
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
python run.py
```

Linux / macOS:
```bash
git clone https://github.com/abdallah-bodzz/driveguard.git
cd driveguard
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python run.py
```

## Usage

### Basic Operation

1. Launch the application: `python run.py`
2. Click "Start Monitoring" (system auto-calibrates for 30 seconds)
3. Normal driving pose - maintain eye contact with camera
4. The system will trigger alarms when sustained fatigue is detected
5. Dismiss alarms via hand gesture or keyboard (D key)

### Configuration Parameters

All tunable parameters are in `driveguard/config.py`:

```python
# Detection thresholds
EAR_THRESHOLD_DEFAULT   = 0.215   # Person-specific after calibration
MAR_THRESHOLD_DEFAULT   = 0.68    # Yawn detection threshold
HEAD_PITCH_THRESH       = 15.0    # Degrees for nod detection

# Scoring parameters
CONSEC_FRAMES           = 6       # Frames before event scoring
ALARM_ON_THRESH         = 48      # Score threshold for alarm
ALARM_OFF_THRESH        = 35      # Score threshold to clear
ALARM_CONFIRM_FRAMES    = 60      # Sustained frames (2 seconds)
ALARM_COOLDOWN_SEC      = 30      # Minimum gap between alarms

# CNN settings
CNN_BORDER_MARGIN       = 0.04    # EAR margin for CNN override
CNN_DISPLAY_INTERVAL    = 5       # Frames between display-only inference
CNN_CLOSED_THRESH       = 0.75    # Probability threshold for closed
CNN_OPEN_THRESH         = 0.35    # Probability threshold for open

# Calibration
AUTO_CALIB_SETTLE       = 90      # Warm-up frames before collection
AUTO_CALIB_FRAMES       = 900     # Frames for calibration (30 seconds)

# Recording
BUFFER_SIZE             = 150     # Pre-event frames (5 seconds at 30 FPS)
BUFFER_WIDTH            = 640
BUFFER_HEIGHT           = 360
```

### Keyboard Controls

| Key | Action |
|-----|--------|
| Space | Mute / unmute alarm |
| D | Dismiss active alarm |
| R | Reset fatigue scores |
| Esc | Stop monitoring |

### Hand Gestures

| Gesture | Detection Method |
|---------|------------------|
| Wave | Hand center x-coordinate amplitude >30px with 3+ direction reversals |
| Fist | Average distance from wrist to fingertips <0.12 normalized |

### Output Files

Recordings directory structure:
```
recordings/
├── incident_YYYYMMDD_HHMMSS.mp4    # Alarm-triggered video clip
├── session_YYYYMMDD_HHMMSS.csv      # Per-second metrics
└── .gitkeep                        # Preserves empty directory
```

CSV schema:
| Column | Type | Description |
|--------|------|-------------|
| time_sec | float | Seconds since session start |
| EAR | float | Eye aspect ratio |
| MAR | float | Mouth aspect ratio |
| fatigue_score | float | Combined 0-120 score |
| short_term | float | Fast-timescale component |
| long_term | float | Slow-timescale component |
| alarm_active | int | 0/1 alarm state |
| dominant_cause | string | Primary fatigue indicator |
| perclos | int | Percentage of closed eyes (0-100) |
| blink_rate_bpm | float | Blinks per minute |

## Performance Characteristics

### Benchmarks (Intel i7-1165G7, 720p@30fps)

| Component | CPU (1 core) | Memory | Latency |
|-----------|--------------|--------|---------|
| MediaPipe face | 8-12% | 180 MB | 15-18ms |
| Detection engine | 3-5% | 12 MB | <1ms |
| CNN inference | 4-6% | 25 MB | 3-5ms per eye |
| UI + rendering | 5-8% | 50 MB | 2-3ms |
| Recording | 1-2% | Variable | 1ms |
| **Total** | **22-33%** | **270 MB** | **22-28ms** |

### Detection Accuracy

Tested on 5 subjects (3 male, 2 female), varied lighting conditions:

| Metric | Value |
|--------|-------|
| Eye closure detection (EAR-only) | 94.2% |
| Eye closure detection (EAR+CNN) | 98.8% |
| Yawn detection | 91.5% |
| Head nod detection | 84.3% (experimental) |
| False positive rate (normal driving) | <0.5 per hour |
| False negative rate (simulated drowsiness) | <2% |

## Project Structure

```
driveguard/
├── driveguard/                  # Main package
│   ├── __init__.py             # Package initializer
│   ├── config.py               # Constants, thresholds, paths
│   ├── detection.py            # DetectionEngine (core logic)
│   ├── ui.py                   # CustomTkinter application
│   ├── recording.py            # Video buffer + CSV writer
│   └── utils.py                # Geometry helpers, formatting
│
├── assets/                      # Static resources
│   ├── mediapipe/              # .task files (face, hand, pose)
│   ├── models/                 # CNN weights (HDF5, Keras, ONNX)
│   └── sounds/                 # Alarm and notification WAVs
│
├── recordings/                  # Runtime output (gitignored)
│   └── .gitkeep                # Preserves directory structure
│
├── tests/                       # Unit tests
│   ├── conftest.py             # pytest fixtures
│   └── test_detection.py       # Engine test suite
│
├── docs/                        # Documentation
│   ├── api_reference.md        # Function signatures
│   ├── user_guide.md           # End-user manual
│   └── competition_notes.md    # Submission context
│
├── training/                    # CNN training resources
│   ├── train_eye_classifier.py # Training script
│   ├── model_training.md       # Training methodology
│   └── requirements_train.txt  # Training dependencies
│
├── demo/                        # Competition materials
│   └── presentation.pptx       # Project overview
│
├── .github/workflows/           # CI/CD
│   └── tests.yml               # GitHub Actions test runner
│
├── run.py                       # Application entry point
├── setup.py                     # Package installation
├── requirements.txt             # Runtime dependencies
├── requirements_dev.txt         # Development dependencies
├── pytest.ini                   # Test configuration
├── .gitignore                   # Excluded patterns
├── LICENSE                      # MIT license
└── README.md                    # This file
```

## Testing

Run the test suite:

```bash
# Install development dependencies
pip install -r requirements_dev.txt

# Run all tests with coverage
pytest tests/ -v --cov=driveguard

# Run specific test module
pytest tests/test_detection.py -v
```

Test coverage targets:
- detection.py: 85%+ (core logic has complex conditional paths)
- utils.py: 95%+ (pure functions)
- config.py: 100% (constants only)

Continuous integration runs on GitHub Actions for Python 3.9, 3.10, 3.11 on Ubuntu, Windows, and macOS.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Camera not opening | Check device manager for webcam driver. Try changing camera index in ui.py (line 858-870). |
| Low FPS (<20) | Reduce CNN_DISPLAY_INTERVAL or disable CNN hybrid mode. Close other applications. |
| CNN model not loading | Verify assets/models/eye_classifier.h5 exists. TensorFlow 2.13+ required. |
| MediaPipe errors | Download .task files from Google MediaPipe models page. Place in assets/mediapipe/. |
| Import errors | Activate virtual environment. Run `pip install -r requirements.txt` again. |
| Alarm sound not playing | Check system volume. Pygame mixer may need different audio backend on Linux. |

## License

MIT License - see LICENSE file for details. You are free to use, modify, and distribute this software for commercial or non-commercial purposes with attribution.

## References

1. T. Soukupova and J. Cech, "Real-Time Eye Blink Detection using Facial Landmarks," 2016.
2. MediaPipe Solutions, Google Research, 2023.
3. MRL Eye Dataset, Machine Learning Research Laboratory, 2015.
4. W. Deng et al., "RetinaFace: Single-stage Dense Face Localisation in the Wild," 2019.