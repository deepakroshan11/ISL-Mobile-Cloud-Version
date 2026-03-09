# 🤟 Indian Sign Language Detection System — Mobile Cloud Version

<div align="center">

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Latest-green.svg)
![Flask](https://img.shields.io/badge/Flask-2.3+-red.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

**Dual-app ISL system: Real-time Emotion Recognition + Speech-to-Sign Avatar Translation**  
*Optimized for mobile & cloud deployment*

[Features](#-features) • [Quick Start](#-quick-start) • [Applications](#-applications) • [Architecture](#️-architecture) • [Usage](#-usage) • [Troubleshooting](#-troubleshooting)

</div>

---

## 📁 Project Structure

```
ISL-Mobile-Cloud-Version/
│
├── 📄 unified_launcher.py         # Start both apps simultaneously
├── 📄 launcher.py                 # Emotion Translator launcher only
├── 📄 isl_detection.py            # Core CV & ML processing engine
├── 📄 isl_ui_dashboard.py         # Emotion Translator Flask dashboard
├── 📄 speech_to_sign_avatar.py    # Speech-to-Sign Avatar backend
├── 📄 translation.py              # Sign ↔ English conversion module
│
├── 🤖 model.h5                    # Trained ISL CNN model
│
├── 📁 templates/
│   ├── dashboard.html             # Emotion Translator UI
│   └── avatar_interface.html      # Speech-to-Sign Avatar UI
│
├── 📋 requirements.txt            # Python dependencies
├── 📋 .gitignore
└── 📋 README.md
```

---

## ✨ Features

### 🎯 Dual Application System
- **Emotion Translator** (Port 5000) — Real-time ISL letter recognition + facial emotion detection
- **Speech-to-Sign Avatar** (Port 5001) — Text/speech to animated 3D ISL gestures
- **Unified Launcher** — Start both apps with a single command
- **Integrated Navigation Bar** — Seamlessly switch between apps
- **Glassmorphism UI** — Consistent modern design across both interfaces

---

### 🎭 App 1: ISL Emotion Translator

#### Core Processing (`isl_detection.py`)
- **Real-time Hand Tracking** via MediaPipe (21 landmarks)
- **ISL Letter Recognition** via TensorFlow CNN model (`model.h5`) — A–Z alphabet
- **Smart Word Formation** with stability checking and buffer logic
- **Sentence Builder** with context-aware construction
- **Shared Memory Architecture** — Uses `multiprocessing.shared_memory` for inter-process frame sharing
- **Crash Recovery** — Cleans up leftover shared memory (`isl_frame_shm`) on restart

#### Emotion Detection
- **Dual-Engine AI** — FER library + Custom face landmark analysis
- **5 Emotion Categories** — Happy, Sad, Angry, Surprise, Neutral
- **CLAHE-Adaptive Fusion** — Dynamic weighting based on lighting conditions
- **Temporal Smoothing** — 15-frame rolling average

#### Dashboard (`isl_ui_dashboard.py` + `dashboard.html`)
- **Socket.IO WebSocket** — Real-time frame and data streaming
- **Live Stats** — FPS, letter count, word count, emotion changes
- **Smart Word Suggestions** — Frequency-based predictions
- **Controls** — Speak (gTTS), Backspace, Reset
- **Emotion Timeline** — Chart.js visualization

---

### 🎤 App 2: Speech-to-Sign Avatar (`speech_to_sign_avatar.py`)

#### Speech Recognition
- **Google Speech Recognition API** — Real-time microphone input
- **Languages** — English (`en-IN`), Hindi (`hi-IN`)
- **Noise Handling** — Adaptive ambient noise adjustment

#### 3D Avatar Animation (`avatar_interface.html`)
- **40+ ISL Gestures** — Keyframe-animated with Three.js
- **Categories** — Greetings, Politeness, Pronouns, Actions, Food, Places, Emotions, Time, Questions
- **Fingerspelling Fallback** — Letter-by-letter for unknown words
- **Smooth Interpolation** — Keyframe transitions at 60 FPS

#### Translation Engine (`translation.py`)
- **50+ sign-to-English mappings**
- **Fingerspell token handling** (`SPELL_*`)
- **Extensible dictionary**

```python
from translation import translate_signs

signs = ["HELLO", "I", "WANT", "WATER"]
english = translate_signs(signs)
# Output: "Hello I want water"
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Webcam (720p+ recommended) — for Emotion Translator
- Microphone — for Speech-to-Sign (optional)
- 4 GB+ RAM
- Windows / Linux / macOS
- Internet connection (Google Speech API)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/deepakroshan11/ISL-Mobile-Cloud-Version.git
cd ISL-Mobile-Cloud-Version

# 2. Create virtual environment (recommended)
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Required Dependencies (`requirements.txt`)
```
tensorflow>=2.10.0
opencv-python>=4.8.0
mediapipe>=0.10.0
numpy>=1.23.0
pandas>=2.0.0
fer>=22.5.0
gtts>=2.3.0
pygame>=2.5.0
flask>=2.3.0
flask-socketio>=5.3.0
python-socketio>=5.9.0
SpeechRecognition>=3.10.0
PyAudio>=0.2.13
```

---

## 🎮 Applications

### Start Both Apps (Recommended)
```bash
python unified_launcher.py
```
- Emotion Translator → http://localhost:5000
- Speech-to-Sign Avatar → http://localhost:5001

### Start Individually
```bash
# Emotion Translator only
python launcher.py

# Speech-to-Sign Avatar only
python speech_to_sign_avatar.py
```

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────┐
│          unified_launcher.py — manages both apps         │
└──────────────┬───────────────────────────┬───────────────┘
               │                           │
               ▼                           ▼
┌──────────────────────────┐  ┌────────────────────────────┐
│  EMOTION TRANSLATOR      │  │  SPEECH-TO-SIGN AVATAR     │
│  Port 5000               │  │  Port 5001                 │
│                          │  │                            │
│  isl_detection.py        │  │  speech_to_sign_avatar.py  │
│  ├─ MediaPipe tracking   │  │  ├─ Speech recognition     │
│  ├─ TensorFlow model.h5  │  │  ├─ Gesture database       │
│  ├─ FER emotion engine   │  │  └─ Socket.IO server       │
│  └─ Shared memory IPC    │  │                            │
│                          │  │  avatar_interface.html     │
│  isl_ui_dashboard.py     │  │  ├─ Three.js 3D avatar     │
│  └─ Flask + Socket.IO    │  │  └─ Keyframe animations    │
│                          │  │                            │
│  dashboard.html          │  │  translation.py            │
│  └─ Real-time Chart.js   │  │  └─ Sign ↔ English map     │
└──────────────────────────┘  └────────────────────────────┘
        📹 Webcam                      🎤 Microphone
```

---

## 🔧 Configuration

### Emotion Translator — `isl_detection.py`
```python
CAM_INDEX          = 0       # Camera index
BUFFER_SIZE        = 12      # Detection buffer
CONF_THRESHOLD     = 0.8     # Letter confidence
STABILITY_THRESHOLD= 0.7
SPACE_THRESHOLD    = 15      # Frames before word space
SENTENCE_DELAY     = 2.0     # Seconds before new sentence
SMOOTHING_FRAMES   = 15      # Emotion rolling average
BASE_FER_WEIGHT    = 0.65
BASE_LANDMARK_WEIGHT=0.35
ENABLE_AUTO_TTS    = True
TTS_LANGUAGE       = "en"
```

### Speech-to-Sign Avatar — `speech_to_sign_avatar.py`
```python
PORT               = 5001
HOST               = '0.0.0.0'
ENERGY_THRESHOLD   = 4000
PAUSE_THRESHOLD    = 0.8
TIMEOUT            = 5
PHRASE_TIME_LIMIT  = 10
LANGUAGES          = ['en-IN', 'hi-IN']
```

---

## 🎮 Usage

### Emotion Translator Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| Ctrl + R | Reset session |
| Ctrl + Backspace | Delete last letter/word |
| Ctrl + S | Speak current text |
| 1 / 2 / 3 | Accept word suggestion |

**Workflow:** Position hand → Make ISL sign → Hold steady → Remove hand to space

### Speech-to-Sign Avatar Input Methods
- **Text:** Type in input box → Click "Convert to Signs"
- **Speech:** Click "🎤 Record Speech" → Speak → Avatar animates

---

## 🛠️ Troubleshooting

**Port already in use:**
```bash
netstat -ano | findstr :5000
taskkill /PID <PID> /F
```

**Camera not detected:**
```python
CAM_INDEX = 1  # Try a different index in isl_detection.py
```

**PyAudio install fails (Windows):**
```bash
# Download wheel from: https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio
pip install PyAudio-0.2.13-cp38-cp38-win_amd64.whl
```

**Speech recognition not working:**
- Check internet connection (Google API required)
- Lower the energy threshold: `recognizer.energy_threshold = 2000`

**Shared memory crash on restart:**  
Already handled — `isl_detection.py` automatically cleans up `isl_frame_shm` at startup.

---

## 📊 Performance

| Metric | Emotion Translator | Speech-to-Sign Avatar |
|--------|-------------------|----------------------|
| FPS | 25–30 | 60 (animation) |
| Letter Accuracy | ~95% | — |
| Latency | <50ms/frame | <2s (speech) |
| Hand Landmarks | 21 points | — |
| Gesture Library | — | 40+ signs |

---

## 🤝 Contributing

1. Fork the repository
2. Create a branch: `git checkout -b feature/YourFeature`
3. Commit: `git commit -m 'Add YourFeature'`
4. Push: `git push origin feature/YourFeature`
5. Open a Pull Request

**Areas for contribution:**
- Add more ISL gestures
- Improve emotion detection accuracy
- Offline speech recognition support
- Mobile app version
- Edge device optimization

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

MediaPipe · TensorFlow · FER · Flask + Socket.IO · Three.js · Chart.js · gTTS · SpeechRecognition · Pygame

---

## 📧 Contact

**Deepak Roshan**  
GitHub: [@deepakroshan11](https://github.com/deepakroshan11)  
Repo: [ISL-Mobile-Cloud-Version](https://github.com/deepakroshan11/ISL-Mobile-Cloud-Version)

---

<div align="center">

**Made with ❤️ for the deaf and hard-of-hearing community**

![ISL](https://img.shields.io/badge/ISL-Indian_Sign_Language-blue)
![Accessibility](https://img.shields.io/badge/Accessibility-First-green)
![Open Source](https://img.shields.io/badge/Open_Source-Yes-orange)
![Version](https://img.shields.io/badge/Version-Mobile--Cloud-purple)

</div>