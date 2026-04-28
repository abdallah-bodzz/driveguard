# 🚗 DriveGuard – Intelligent Driver Fatigue Assistant

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-orange)](https://tensorflow.org)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10-ff69b4)](https://mediapipe.dev)

**Real‑time drowsiness detection using only a standard webcam.**  
No cloud, no special hardware – runs locally on your laptop.  
Designed for highway drivers, fleet operators, and safety‑conscious individuals.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 👁️ **Eye tracking** | Eye Aspect Ratio (EAR) + CNN verification (98.8% accuracy) |
| 🥱 **Yawning detection** | Mouth Aspect Ratio (MAR) with adaptive threshold |
| 😴 **Head nod detection** | solvePnP pose estimation (experimental, off by default) |
| 📊 **Dual‑score fatigue model** | Short‑term (fast) + long‑term (slow) with hysteresis |
| 🤚 **Hand gesture dismiss** | Wave or fist – silence the alarm hands‑free |
| 🔧 **Auto‑calibration** | Learns your face in the first 30 seconds |
| 📹 **Incident recording** | 5‑second pre‑event MP4 clips |
| 📈 **Session CSV export** | Full data for post‑drive analysis |
| 🖥️ **Modern UI** | Camera split view, live graphs, event log, dark theme |
| ⌨️ **Keyboard shortcuts** | Space (mute), D (dismiss), R (reset), Esc (stop) |

---

## 🎥 Demo

![Demo](demo/demo_video.mp4)  
*Placeholder – replace with actual screen recording link.*

---

## 🛠️ Installation

### Prerequisites
- Python 3.9 – 3.11
- A working webcam
- (Optional) NVIDIA GPU for faster CNN inference – but CPU works fine.

### Clone and install
`ash
git clone https://github.com/yourusername/DriveGuard.git
cd DriveGuard
pip install -r requirements.txt
`

> **Note**: On first run, the app will look for MediaPipe task files in ssets/mediapipe/ and the CNN model in ssets/models/. These are included in the repository.

---

## 🚀 Usage

1. Run the launcher:
   `ash
   python run.py
   `
2. Click **Start Monitoring**.
3. Look at the camera – the app auto‑calibrates for 30 seconds.
4. Simulate drowsiness by closing your eyes for 2–3 seconds – the alarm will trigger.
5. Wave your hand left‑right (or make a fist) to dismiss the alarm.

**Keyboard shortcuts** (active when monitoring):
- Space – mute / unmute alarm
- D – dismiss active alarm
- R – reset fatigue score
- Esc – stop monitoring

---

## 📁 Project Structure

`
DriveGuard/
├── driveguard/           # Python package
│   ├── config.py         # Constants & paths
│   ├── detection.py      # DetectionEngine
│   ├── ui.py             # Main UI
│   ├── recording.py      # Video & CSV
│   └── utils.py          # Helpers
├── assets/               # Models, MediaPipe tasks, sounds
├── recordings/           # Session output (auto‑created)
├── training/             # CNN training scripts & docs
├── docs/                 # User guide, competition notes
├── demo/                 # Demo video & presentation
├── run.py                # Launcher
├── requirements.txt      # Dependencies
├── setup.py              # Install script
└── LICENSE               # MIT license
`

---

## 🧠 How It Works

DriveGuard combines **geometric rules** (EAR, MAR) with a **lightweight CNN** (98.8% validation accuracy, 1.6 MB) to detect eye closure. The fatigue score is a weighted combination of short‑term and long‑term scores with duration bonuses, hysteresis, and session‑context awareness.

See [	raining/model_training.md](training/model_training.md) for the full CNN training journey.

---

## 📊 Performance

| Metric | Value |
|--------|-------|
| Frame rate | 30 FPS on Intel i7‑1165G7 |
| CPU usage | 25‑35% (CNN adds ~5%) |
| Alarm latency | 2‑3 seconds (sustained closure) |
| False alarm rate | <1 per hour in normal driving |
| Model size | 1.6 MB (eye_classifier.h5) |

---

## 📜 License

Distributed under the MIT License. See LICENSE for more information.

---

## 🙏 Acknowledgements

- [MediaPipe](https://mediapipe.dev) – face & hand landmarks
- [TensorFlow](https://tensorflow.org) – CNN training & inference
- [CustomTkinter](https://github.com/TomSchimansky/CustomTkinter) – modern UI
- [MRL Eye Dataset](http://mrl.cs.vsb.cz/eyedataset) – training data

---

## ⚠️ Disclaimer

**DriveGuard is a driver assistance tool – not a substitute for responsible driving.**  
Always obey traffic laws. If you feel tired, stop and rest regardless of what the system shows.  
The authors assume no liability for any incidents arising from the use of this software.
