---
title: ISL Indian Sign Language Detection
emoji: 🤟
colorFrom: yellow
colorTo: green
sdk: docker
pinned: false
license: mit
app_port: 7860
---

# 🤟 ISL Indian Sign Language Detection System

Real-time Indian Sign Language detection with emotion recognition and speech-to-sign avatar.

## Features
- **Gesture Recognition** — Real-time ISL A-Z letter detection via MediaPipe + TensorFlow
- **Emotion Detection** — Happy, Sad, Angry, Surprise, Neutral via facial landmarks
- **Speech-to-Sign Avatar** — 3D animated hand avatar for fingerspelling

## Usage
- Main dashboard: `/`
- Speech-to-Sign avatar: `/avatar/`

## Built with
- Flask + Socket.IO
- TensorFlow CNN model
- MediaPipe Hand Tracking
- Three.js 3D Avatar
