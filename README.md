---
title: ISL Indian Sign Language Detection
emoji: "🤟"
colorFrom: yellow
colorTo: green
sdk: docker
pinned: false
license: mit
app_port: 7860
---

# 🤟 ISL Indian Sign Language Detection System

> **Live Demo → [https://huggingface.co/spaces/deepakroshan/isl-detection](https://huggingface.co/spaces/deepakroshan/isl-detection)**

Real-time Indian Sign Language detection with emotion recognition and speech-to-sign avatar — optimised for mobile and cloud deployment.

<div align="center">

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Latest-green.svg)
![Flask](https://img.shields.io/badge/Flask-3.0+-red.svg)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Spaces-yellow.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

</div>

---

## ✨ Features

### 🎯 Dual Application System
| App | Route | Description |
|-----|-------|-------------|
| **ISL Emotion Translator** | `/` | Real-time ISL A–Z letter recognition + facial emotion detection |
| **Speech-to-Sign Avatar** | `/avatar/` | Text / speech → animated 3D ISL fingerspelling |

---

### 🎭 App 1 — ISL Emotion Translator

**Core Processing (`isl_detection.py`)**
- Real-time hand tracking via MediaPipe (21 landmarks)
- ISL letter recognition via TensorFlow CNN (`model.h5`) — A–Z alphabet
- Smart word formation with stability checking and buffer logic
- Sentence builder with context-aware construction
- **Cloud architecture** — browser-streamed frames processed via in-memory `queue.Queue` (single-process threading, no shared memory IPC)
- Crash-safe startup — leftover `isl_frame_shm` is unlinked automatically on restart

**Emotion Detection**
- Dual-engine AI — FER library + custom face landmark analysis
- 5 emotion categories — Happy, Sad, Angry, Surprise, Neutral
- CLAHE-adaptive fusion — dynamic weighting based on lighting
- Temporal smoothing — 15-frame rolling average

**Dashboard (`isl_ui_dashboard.py` + `dashboard.html`)**
- Socket.IO WebSocket — real-time frame and state streaming
- Live stats — FPS, letter count, word count, emotion changes
- Smart word suggestions — frequency-based predictions
- Keyboard controls — speak (gTTS), backspace, reset, accept suggestion
- Emotion timeline — Chart.js visualisation
- **Hand skeleton overlay** — canvas rendering of annotated MJPEG frames with landmark points

---

### 🎤 App 2 — Speech-to-Sign Avatar

**Speech Recognition (`speech_to_sign_avatar.py`)**
- Google Speech Recognition API — real-time microphone input (local mode)
- Language: English (`en-IN`)
- Noise handling — adaptive ambient noise adjustment
- **Cloud mode** — microphone disabled; text input used instead

**3D Avatar Animation (`avatar_interface.html`)**
- Full A–Z ISL fingerspelling — keyframe-animated with Three.js
- Smooth interpolation at 60 FPS
- Letter-by-letter sequence rendering with configurable timing

**Translation Engine (`translation.py`)**
- 50+ sign-to-English mappings
- Fingerspell token handling (`SPELL_*`)
- Extensible dictionary

```python
from translation import translate_signs

signs = ["HELLO", "I", "WANT", "WATER"]
english = translate_signs(signs)
# → "Hello I want water"
```

---

## 📁 Project Structure

```
ISL-Mobile-Cloud-Version/
│
├── start_gesture.py           # ← HF Spaces entry point (cloud, single-process)
├── launcher.py                # Local multi-process launcher
├── unified_launcher.py        # Start both apps from one terminal (local)
│
├── isl_detection.py           # Core CV + ML engine (MediaPipe + TF + emotion)
├── isl_ui_dashboard.py        # Flask + Socket.IO dashboard server
├── speech_to_sign_avatar.py   # Speech-to-Sign backend
├── translation.py             # Sign ↔ English conversion
│
├── model.h5                   # Trained ISL CNN model (A–Z)
│
├── templates/
│   ├── dashboard.html         # Emotion Translator UI
│   └── avatar_interface.html  # Speech-to-Sign Avatar UI
│
├── Dockerfile                 # HuggingFace Spaces Docker config
├── supervisord.conf           # Process supervisor config (cloud)
├── nginx.conf                 # Reverse proxy config
├── requirements.txt
└── README.md
```

---

## 🏗️ Architecture

### Cloud (HuggingFace Spaces) — Single-Process Threading

```
Dockerfile  →  supervisord  →  start_gesture.py
                                      │
                                      ▼
                              SharedState (queue.Queue)
                                      │
                        ┌─────────────┴──────────────┐
                        ▼                            ▼
               dashboard_process()          core_processing_engine()
               Flask + Socket.IO            MediaPipe + TF model
               (main thread)                (daemon thread)
                        │
                        ▼
               Browser → canvas overlay
               (annotated MJPEG frames)
```

> **Why single-process?** HuggingFace Spaces restricts `multiprocessing.shared_memory` and fork-based IPC. The threading model passes `SharedState` as a single in-memory `queue.Queue` object — no IPC, no file descriptors, no crashes.

### Local — Multi-Process with Shared Memory

```
unified_launcher.py
        │
        ├──▶  launcher.py  (port 5000)
        │         ├─ CoreProcessor  (camera · MediaPipe · TF)
        │         └─ WebDashboard   (Flask · Socket.IO)
        │              SharedMemory "isl_frame_shm" (zero-copy 640×480 BGR)
        │
        └──▶  speech_to_sign_avatar.py  (port 5001)
```

---

## 🚀 Quick Start (Local)

### Prerequisites
- Python 3.8+
- Webcam (720p+ recommended)
- Microphone (optional, for Speech-to-Sign)
- 4 GB+ RAM

### Installation

```bash
git clone https://github.com/deepakroshan11/ISL-Mobile-Cloud-Version.git
cd ISL-Mobile-Cloud-Version

python -m venv venv
# Windows:   venv\Scripts\activate
# Linux/Mac: source venv/bin/activate

pip install -r requirements.txt
```

### Run

```bash
# Both apps simultaneously (recommended)
python unified_launcher.py

# Emotion Translator only (port 5000)
python launcher.py

# Speech-to-Sign Avatar only (port 5001)
python speech_to_sign_avatar.py
```

| URL | App |
|-----|-----|
| http://localhost:5000 | ISL Emotion Translator |
| http://localhost:5001 | Speech-to-Sign Avatar |

---

## 🔧 Configuration

### Emotion Translator — `isl_detection.py`
```python
CAM_INDEX           = 0      # Camera index
BUFFER_SIZE         = 12     # Detection buffer
CONF_THRESHOLD      = 0.8    # Letter confidence
STABILITY_THRESHOLD = 0.7
SPACE_THRESHOLD     = 15     # Frames before word space
SENTENCE_DELAY      = 2.0    # Seconds before new sentence
SMOOTHING_FRAMES    = 15     # Emotion rolling average
BASE_FER_WEIGHT     = 0.65
BASE_LANDMARK_WEIGHT= 0.35
ENABLE_AUTO_TTS     = True
TTS_LANGUAGE        = "en"
```

### Speech-to-Sign Avatar — `speech_to_sign_avatar.py`
```python
PORT              = 5001
ENERGY_THRESHOLD  = 3000
PAUSE_THRESHOLD   = 0.8
TIMEOUT           = 5
PHRASE_TIME_LIMIT = 10
```

---

## 🎮 Usage

### Emotion Translator Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl + R` | Reset session |
| `Ctrl + Backspace` | Delete last letter / word |
| `1` / `2` / `3` | Accept word suggestion |

**Workflow:** Position hand in frame → make ISL sign → hold steady → remove hand to insert space.

### Speech-to-Sign Avatar Input Methods
- **Text:** Type in the input box → click "Convert to Signs"
- **Speech (local only):** Click "🎤 Record Speech" → speak → avatar animates letter by letter

---

## 📊 Performance

| Metric | Emotion Translator | Speech-to-Sign Avatar |
|--------|-------------------|----------------------|
| FPS | 25–30 | 60 (animation) |
| Letter accuracy | ~95% | — |
| Frame latency | <50 ms | — |
| Speech latency | — | <2 s |
| Hand landmarks | 21 points | — |
| Gesture library | A–Z (26) | A–Z fingerspelling |

---

## 🛠️ Troubleshooting

**Port already in use (local):**
```bash
# Windows
netstat -ano | findstr :5000
taskkill /PID <PID> /F
```

**Camera not detected:**
```python
CAM_INDEX = 1  # try index 1 or 2 in isl_detection.py
```

**PyAudio install fails (Windows):**
```bash
# Download wheel: https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio
pip install PyAudio-0.2.13-cp38-cp38-win_amd64.whl
```

**Shared memory crash on restart (local):**
Already handled — `launcher.py` automatically unlinks `isl_frame_shm` at startup.

**Microphone not working on HuggingFace Spaces:**
Microphone access is disabled in the cloud deployment by design. Use the text input box instead.

---

## 🤝 Contributing

1. Fork the repository
2. Create a branch: `git checkout -b feature/YourFeature`
3. Commit: `git commit -m 'Add YourFeature'`
4. Push: `git push origin feature/YourFeature`
5. Open a Pull Request

**Good areas to contribute:**
- Additional ISL gestures and signs
- Improved emotion detection accuracy
- Offline speech recognition (Whisper local)
- Mobile app wrapper
- Edge device (Raspberry Pi) optimisation

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

MediaPipe · TensorFlow · FER · Flask + Socket.IO · Three.js · Chart.js · gTTS · SpeechRecognition

---

## 📧 Contact

**Deepak Roshan**
GitHub: [@deepakroshan11](https://github.com/deepakroshan11)
Repo: [ISL-Mobile-Cloud-Version](https://github.com/deepakroshan11/ISL-Mobile-Cloud-Version)
Live Demo: [HuggingFace Spaces](https://huggingface.co/spaces/deepakroshan/isl-detection)

---

<div align="center">

**Made with ❤️ for the deaf and hard-of-hearing community**

![ISL](https://img.shields.io/badge/ISL-Indian_Sign_Language-blue)
![Accessibility](https://img.shields.io/badge/Accessibility-First-green)
![Open Source](https://img.shields.io/badge/Open_Source-Yes-orange)
![HuggingFace](https://img.shields.io/badge/Live_on-HuggingFace_Spaces-yellow)

</div>
