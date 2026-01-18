# 🤟 Indian Sign Language Detection System

<div align="center">

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Latest-green.svg)
![Flask](https://img.shields.io/badge/Flask-2.3+-red.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

**Comprehensive ISL system with dual applications: Real-time Emotion Recognition + Speech-to-Sign Avatar Translation**

[Features](#-features) • [Quick Start](#-quick-start) • [Applications](#-applications) • [Architecture](#️-architecture) • [Usage](#-usage) • [Documentation](#-documentation)

</div>

---

## ✨ Features

### 🎯 Dual Application System
- **Emotion Translator (Port 5000)** - Real-time ISL recognition with AI emotion detection
- **Speech-to-Sign Avatar (Port 5001)** - Text/speech to animated 3D ISL gestures
- **Seamless Navigation** - Switch between apps with integrated navigation bar
- **Unified Launcher** - Start both applications simultaneously
- **Consistent UI** - Modern glassmorphism design across both apps

### 🎭 **Application 1: ISL Emotion Translator**

#### Core Functionality
- **Real-time Hand Tracking** - MediaPipe-based hand landmark detection
- **ISL Letter Recognition** - TensorFlow CNN model for A-Z alphabet recognition
- **Letter Confidence Overlay** - Live display of detected letter and accuracy
- **Smart Word Formation** - Automatic word completion with stability checking
- **Sentence Building** - Context-aware sentence construction

#### Emotion Detection
- **Dual-Engine Emotion AI** - FER + Custom Landmark Analysis
- **5 Emotion Categories** - Happy, Sad, Angry, Surprise, Neutral
- **Adaptive Fusion** - Dynamic weighting based on lighting (CLAHE)
- **Temporal Smoothing** - 15-frame rolling average for stability
- **Real-time Timeline** - Live emotion tracking with Chart.js

#### Professional Dashboard
- **Modern Glassmorphism UI** - Beautiful gradients with blur effects
- **Real-time Updates** - Socket.IO WebSocket communication
- **Live Statistics** - FPS, letters, words, sentences, emotion changes
- **Smart Suggestions** - Frequency-based word predictions
- **Interactive Controls** - Speak (TTS), Reset, Backspace
- **Emotion Visualization** - Progress bars + timeline chart

### 🎤 **Application 2: Speech-to-Sign Avatar**

#### Speech Recognition
- **Real-time Speech Input** - Google Speech Recognition API
- **Multi-language Support** - English (en-IN) and Hindi (hi-IN)
- **Adaptive Noise Handling** - Automatic ambient noise adjustment
- **Phrase Detection** - 10-second phrase limit with timeout handling

#### 3D Avatar Animation
- **40+ ISL Gestures** - Complete gesture library with keyframe animations
- **Categories Covered:**
  - Greetings (Hello, Hi, Goodbye, Bye)
  - Politeness (Please, Thank You, Sorry, Welcome)
  - Pronouns (I, You, We, They)
  - Basic Responses (Yes, No, OK, Good, Bad)
  - Actions (Want, Need, Help, Go, Come, Stop, Sit, Stand)
  - Food & Drink (Water, Food, Eat, Drink)
  - Places (Home, School, Work)
  - Emotions (Happy, Sad, Angry)
  - Time (Morning, Afternoon, Evening, Night, Today, Tomorrow, Yesterday)
  - Questions (How, What, Where, When, Why)

#### Text-to-Sign Conversion
- **Fingerspelling** - Letter-by-letter spelling for unknown words
- **Smooth Animations** - Interpolated keyframe transitions
- **Performance Stats** - Track conversions and gesture usage
- **Translation Engine** - Sign-to-English conversion support

---

## 🚀 Quick Start

### Prerequisites
- **Python 3.8+**
- **Webcam** (for Emotion Translator - 720p+ recommended)
- **Microphone** (for Speech-to-Sign - optional)
- **4GB+ RAM**
- **Windows / Linux / macOS**

### Installation

1. **Clone the Repository**
```bash
git clone https://github.com/deepakroshan11/Indian-Sign-Language-Detection.git
cd Indian-Sign-Language-Detection
```

2. **Create Virtual Environment** (Recommended)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

### 📦 Required Dependencies
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

### **App 1: ISL Emotion Translator**

**Start the Application:**
```bash
# Method 1: Unified launcher (starts both apps)
python unified_launcher.py

# Method 2: Manual start (Emotion Translator only)
python launcher.py
```

**Access Dashboard:**
```
http://localhost:5000
```

**Features:**
- Real-time webcam-based ISL letter detection
- Live emotion tracking from facial expressions
- Word suggestions and sentence formation
- Text-to-speech output
- Interactive statistics dashboard

---

### **App 2: Speech-to-Sign Avatar**

**Start the Application:**
```bash
# Method 1: Unified launcher (recommended)
python unified_launcher.py

# Method 2: Manual start (Avatar only)
python speech_to_sign_avatar.py
```

**Access Interface:**
```
http://localhost:5001
```

**Features:**
- Type or speak text to convert to ISL signs
- 3D animated hand displays gestures
- 40+ pre-programmed ISL gestures
- Automatic fingerspelling for unknown words
- Real-time speech recognition

---

## 🔄 Navigation Between Apps

Both applications include an integrated navigation bar at the top:

```
┌─────────────────────────────────────────────────┐
│  😊 Emotion Translator  |  🤟 Speech-to-Sign   │
└─────────────────────────────────────────────────┘
```

**Navigation Features:**
- Click buttons to switch between applications
- Active app highlighted with gradient background
- Smooth page transitions with fade animations
- Consistent UI design across both apps
- Hover effects for better UX

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│           UNIFIED LAUNCHER (unified_launcher.py)            │
│         Manages both applications in parallel               │
└──────────────┬──────────────────────────┬───────────────────┘
               │                          │
               ▼                          ▼
┌──────────────────────────┐   ┌──────────────────────────────┐
│  EMOTION TRANSLATOR      │   │  SPEECH-TO-SIGN AVATAR       │
│  (Port 5000)             │   │  (Port 5001)                 │
│                          │   │                              │
│  ┌────────────────────┐  │   │  ┌─────────────────────────┐ │
│  │  Core Processor    │  │   │  │  Flask Backend          │ │
│  │  - MediaPipe       │  │   │  │  - Speech Recognition   │ │
│  │  - TensorFlow      │  │   │  │  - Gesture Database     │ │
│  │  - FER Emotions    │  │   │  │  - Socket.IO Server     │ │
│  └────────────────────┘  │   │  └─────────────────────────┘ │
│           ↕              │   │             ↕                │
│  ┌────────────────────┐  │   │  ┌─────────────────────────┐ │
│  │  Flask Dashboard   │  │   │  │  3D Avatar Frontend     │ │
│  │  - Socket.IO       │  │   │  │  - Three.js Animation   │ │
│  │  - Real-time UI    │  │   │  │  - Keyframe System      │ │
│  │  - Chart.js        │  │   │  │  - Audio Playback       │ │
│  └────────────────────┘  │   │  └─────────────────────────┘ │
└──────────────────────────┘   └──────────────────────────────┘
          ↕                                  ↕
   📹 Webcam Input                   🎤 Microphone Input
```

### Translation Module

The `translation.py` module provides bidirectional conversion:

```python
from translation import translate_signs

# Convert sign tokens to English
signs = ["HELLO", "I", "WANT", "WATER"]
english = translate_signs(signs)
# Output: "Hello I want water"
```

**Features:**
- 50+ sign-to-English mappings
- Fingerspell handling (SPELL_* tokens)
- Grammar improvements
- Extensible dictionary

---

## 🎮 Usage

### Emotion Translator Usage

**Keyboard Shortcuts:**

| Shortcut | Action |
|----------|--------|
| **Ctrl + R** | Reset session |
| **Ctrl + Backspace** | Delete last letter/word |
| **Ctrl + S** | Speak current text |
| **1 / 2 / 3** | Accept suggestions |

**Dashboard Controls:**
- **🔊 Speak** - Text-to-speech output
- **⌫ Back** - Remove last character
- **🔄 Reset** - Clear everything

**Workflow:**
1. Position hand in front of webcam
2. Make ISL letter signs
3. Hold steady for detection
4. Remove hand to complete word
5. Click suggestions or continue signing

---

### Speech-to-Sign Avatar Usage

**Input Methods:**

**Method 1: Text Input**
1. Type text in the input box
2. Click "Convert to Signs"
3. Watch avatar animate the signs

**Method 2: Speech Input**
1. Click "🎤 Record Speech"
2. Speak clearly when prompted
3. Avatar automatically displays signs

**Features:**
- Real-time gesture animation
- Automatic fingerspelling fallback
- Performance statistics tracking
- Sign playback controls

---

## 🔧 Configuration

### Emotion Translator Settings

Edit `isl_detection.py`:
```python
# Camera
CAM_INDEX = 0

# Detection
BUFFER_SIZE = 12
CONF_THRESHOLD = 0.8
STABILITY_THRESHOLD = 0.7
SPACE_THRESHOLD = 15
SENTENCE_DELAY = 2.0

# Emotion
SMOOTHING_FRAMES = 15
BASE_FER_WEIGHT = 0.65
BASE_LANDMARK_WEIGHT = 0.35

# TTS
ENABLE_AUTO_TTS = True
TTS_LANGUAGE = "en"
```

### Speech-to-Sign Avatar Settings

Edit `speech_to_sign_avatar.py`:
```python
# Server
PORT = 5001
HOST = '0.0.0.0'

# Speech Recognition
ENERGY_THRESHOLD = 4000
PAUSE_THRESHOLD = 0.8
TIMEOUT = 5
PHRASE_TIME_LIMIT = 10

# Languages
LANGUAGES = ['en-IN', 'hi-IN']
```

---

## 📁 Project Structure

```
Indian-Sign-Language-Detection/
│
├── 📄 unified_launcher.py            # Start both apps
├── 📄 launcher.py                    # Emotion translator launcher
├── 📄 isl_detection.py               # Core CV processing
├── 📄 isl_ui_dashboard.py            # Emotion dashboard
├── 📄 speech_to_sign_avatar.py       # Avatar backend
├── 📄 translation.py                 # Sign-English conversion
├── 📄 cleanup_audio_files.py         # Audio cleanup utility
├── 📄 dataset_keypoint_generation.py # Data collection
│
├── 🤖 model.h5                       # Trained ISL model (11.5 MB)
├── 📊 keypoint.csv                   # Training dataset
│
├── 📁 templates/
│   ├── dashboard.html                # Emotion translator UI
│   ├── avatar_interface.html         # Speech-to-sign UI
│   └── dashboard.html.backup         # Backup
│
├── 📁 dataset/                       # Training images
├── 📁 images/                        # Collected gestures
├── 📁 tmp_audio/                     # TTS audio cache
│
├── 📋 requirements.txt               # Dependencies
├── 📋 .gitignore                     # Git ignore rules
└── 📋 README.md                      # This file
```

---

## 🛠️ Troubleshooting

### Common Issues

**Issue: Port Already in Use**
```bash
# Find process using port
netstat -ano | findstr :5000
netstat -ano | findstr :5001

# Kill process (Windows)
taskkill /PID <PID> /F

# Or change port in code
# emotion translator: port=5000
# avatar: port=5001
```

**Issue: Camera Not Detected**
```python
# Try different camera index
CAM_INDEX = 1  # Change in isl_detection.py
```

**Issue: PyAudio Installation Failed (Windows)**
```bash
# Download wheel from: https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio
pip install PyAudio-0.2.13-cp38-cp38-win_amd64.whl
```

**Issue: Microphone Not Working**
```bash
# Grant microphone permissions
# Windows: Settings → Privacy → Microphone → Allow apps
# Mac: System Preferences → Security & Privacy → Microphone
```

**Issue: Speech Recognition Not Working**
```bash
# Check internet connection (Google API required)
# Or adjust energy threshold
recognizer.energy_threshold = 2000  # Lower value
```

---

## 📊 Performance Metrics

### Emotion Translator
- **Frame Rate**: 25-30 FPS
- **Letter Accuracy**: ~95%
- **Emotion Latency**: <50ms/frame
- **Hand Landmarks**: 21 points
- **Face Landmarks**: 468 points

### Speech-to-Sign Avatar
- **Gesture Library**: 40+ signs
- **Animation FPS**: 60 FPS
- **Speech Recognition**: <2s latency
- **Fingerspelling Speed**: 2 letters/second
- **Conversion Rate**: Real-time

---

## 🎓 Training Custom Models

### Collect Training Data
```bash
python dataset_keypoint_generation.py
```

### Add Custom Gestures

Edit `speech_to_sign_avatar.py`:
```python
ISL_GESTURES["your_word"] = {
    "name": "YOUR_WORD",
    "keyframes": [
        {"hand": "right", "pos": [x, y, z], "rot": [rx, ry, rz], "time": 0.0},
        {"hand": "right", "pos": [x2, y2, z2], "rot": [rx2, ry2, rz2], "time": 1.0}
    ],
    "duration": 1.5,
    "desc": "Description",
    "type": "sign"
}
```

### Add Translations

Edit `translation.py`:
```python
_TRANSLATION_MAP["YOUR_SIGN"] = "English translation"
```

---

## 🤝 Contributing

Contributions welcome! Areas for improvement:

- [ ] Add more ISL gestures and words
- [ ] Improve emotion detection accuracy
- [ ] Multi-language TTS support
- [ ] Mobile app version
- [ ] Offline speech recognition
- [ ] Real-time sign translation (both directions)
- [ ] Training data augmentation
- [ ] Edge device optimization

**How to Contribute:**
1. Fork the repository
2. Create feature branch: `git checkout -b feature/AmazingFeature`
3. Commit changes: `git commit -m 'Add AmazingFeature'`
4. Push to branch: `git push origin feature/AmazingFeature`
5. Open Pull Request

---

## 📜 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **MediaPipe** - Hand and face landmark detection
- **TensorFlow** - Deep learning framework
- **FER** - Facial emotion recognition
- **Flask + Socket.IO** - Real-time web framework
- **Three.js** - 3D avatar animation
- **Chart.js** - Data visualization
- **gTTS** - Google Text-to-Speech
- **SpeechRecognition** - Python speech library
- **Pygame** - Audio playback

---

## 📧 Contact

**Deepak Roshan**
- 🐙 GitHub: [@deepakroshan11](https://github.com/deepakroshan11)
- 📁 Repository: [Indian-Sign-Language-Detection](https://github.com/deepakroshan11/Indian-Sign-Language-Detection)

---

## 🌟 Show Your Support

If this project helped you:
- ⭐ **Star this repository**
- 🍴 **Fork and contribute**
- 📢 **Share with others**
- 💬 **Report issues**
- 📝 **Improve documentation**

---

## 📸 Screenshots

### Emotion Translator Dashboard
- Live webcam feed with ISL detection
- Real-time emotion tracking bars
- Smart word suggestions panel
- Emotion timeline graph
- Professional glassmorphism UI

### Speech-to-Sign Avatar Interface
- 3D animated hand display
- Text and speech input options
- Gesture library visualization
- Performance statistics
- Integrated navigation bar

---

<div align="center">

**Made with ❤️ for the deaf and hard-of-hearing community**

![ISL](https://img.shields.io/badge/ISL-Indian_Sign_Language-blue)
![Accessibility](https://img.shields.io/badge/Accessibility-First-green)
![Open Source](https://img.shields.io/badge/Open_Source-Yes-orange)
![Dual Apps](https://img.shields.io/badge/Apps-Emotion_+_Avatar-purple)

**Version 2.0 - Now with Speech-to-Sign Avatar System**

</div>