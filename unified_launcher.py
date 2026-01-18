"""
🎭 SPEECH-TO-SIGN AVATAR SYSTEM
Complete backend with 40+ ISL gestures and speech recognition

Features:
- 40+ ISL gestures with keyframes
- Real-time speech recognition
- Text-to-sign conversion
- Fingerspelling for unknown words
- Socket.IO real-time communication
- Performance statistics
"""

import os
import json
import time
import threading
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import speech_recognition as sr

# ==================== ISL GESTURE DATABASE ====================
# Complete database with 40+ gestures and keyframe animations

ISL_GESTURES = {
    # ===== GREETINGS =====
    "hello": {
        "name": "HELLO",
        "keyframes": [
            {"hand": "right", "pos": [0.2, 1.6, 0.4], "rot": [0, 0, 0], "time": 0.0},
            {"hand": "right", "pos": [0.3, 1.6, 0.4], "rot": [0, 0, 20], "time": 0.3},
            {"hand": "right", "pos": [0.2, 1.6, 0.4], "rot": [0, 0, -20], "time": 0.6},
            {"hand": "right", "pos": [0.3, 1.6, 0.4], "rot": [0, 0, 20], "time": 0.9},
            {"hand": "right", "pos": [0.2, 1.6, 0.4], "rot": [0, 0, 0], "time": 1.2}
        ],
        "duration": 1.5,
        "desc": "Wave hand at head level",
        "type": "sign"
    },
    
    "hi": {
        "name": "HI",
        "keyframes": [
            {"hand": "right", "pos": [0.2, 1.5, 0.4], "rot": [0, 0, 0], "time": 0.0},
            {"hand": "right", "pos": [0.3, 1.5, 0.4], "rot": [0, 0, 15], "time": 0.4},
            {"hand": "right", "pos": [0.2, 1.5, 0.4], "rot": [0, 0, 0], "time": 0.8}
        ],
        "duration": 1.0,
        "desc": "Quick wave",
        "type": "sign"
    },
    
    "goodbye": {
        "name": "GOODBYE",
        "keyframes": [
            {"hand": "right", "pos": [0.2, 1.5, 0.4], "rot": [0, 0, 0], "time": 0.0},
            {"hand": "right", "pos": [0.3, 1.5, 0.5], "rot": [0, 0, 20], "time": 0.4},
            {"hand": "right", "pos": [0.1, 1.5, 0.5], "rot": [0, 0, -20], "time": 0.8},
            {"hand": "right", "pos": [0.3, 1.5, 0.5], "rot": [0, 0, 20], "time": 1.2},
            {"hand": "right", "pos": [0.2, 1.5, 0.4], "rot": [0, 0, 0], "time": 1.6}
        ],
        "duration": 2.0,
        "desc": "Wave goodbye",
        "type": "sign"
    },
    
    "bye": {
        "name": "BYE",
        "keyframes": [
            {"hand": "right", "pos": [0.2, 1.5, 0.4], "rot": [0, 0, 0], "time": 0.0},
            {"hand": "right", "pos": [0.3, 1.5, 0.5], "rot": [0, 0, 15], "time": 0.5},
            {"hand": "right", "pos": [0.2, 1.5, 0.4], "rot": [0, 0, 0], "time": 1.0}
        ],
        "duration": 1.2,
        "desc": "Quick goodbye wave",
        "type": "sign"
    },
    
    # ===== POLITENESS =====
    "thank": {
        "name": "THANK_YOU",
        "keyframes": [
            {"hand": "right", "pos": [0, 1.5, 0.2], "rot": [0, 0, 0], "time": 0.0},
            {"hand": "right", "pos": [0, 1.3, 0.5], "rot": [45, 0, 0], "time": 0.8},
            {"hand": "right", "pos": [0, 1.1, 0.7], "rot": [45, 0, 0], "time": 1.6}
        ],
        "duration": 2.0,
        "desc": "Hand from chin outward",
        "type": "sign"
    },
    
    "thanks": {
        "name": "THANK_YOU",
        "keyframes": [
            {"hand": "right", "pos": [0, 1.5, 0.2], "rot": [0, 0, 0], "time": 0.0},
            {"hand": "right", "pos": [0, 1.3, 0.5], "rot": [45, 0, 0], "time": 0.8},
            {"hand": "right", "pos": [0, 1.1, 0.7], "rot": [45, 0, 0], "time": 1.6}
        ],
        "duration": 2.0,
        "desc": "Hand from chin outward",
        "type": "sign"
    },
    
    "please": {
        "name": "PLEASE",
        "keyframes": [
            {"hand": "right", "pos": [0, 1.3, 0.3], "rot": [0, 0, 0], "time": 0.0},
            {"hand": "right", "pos": [0.1, 1.3, 0.3], "rot": [0, 0, 30], "time": 0.5},
            {"hand": "right", "pos": [-0.1, 1.3, 0.3], "rot": [0, 0, -30], "time": 1.0},
            {"hand": "right", "pos": [0, 1.3, 0.3], "rot": [0, 0, 0], "time": 1.5}
        ],
        "duration": 1.8,
        "desc": "Circular motion on chest",
        "type": "sign"
    },
    
    "sorry": {
        "name": "SORRY",
        "keyframes": [
            {"hand": "right", "pos": [0, 1.3, 0.3], "rot": [0, 0, 0], "time": 0.0},
            {"hand": "right", "pos": [0.08, 1.25, 0.3], "rot": [0, 0, 45], "time": 0.5},
            {"hand": "right", "pos": [-0.08, 1.35, 0.3], "rot": [0, 0, -45], "time": 1.0},
            {"hand": "right", "pos": [0, 1.3, 0.3], "rot": [0, 0, 0], "time": 1.5}
        ],
        "duration": 1.8,
        "desc": "Circular fist on chest",
        "type": "sign"
    },
    
    "welcome": {
        "name": "WELCOME",
        "keyframes": [
            {"hand": "both", "pos": [0.5, 1.4, 0.6], "rot": [0, 0, 0], "time": 0.0},
            {"hand": "both", "pos": [0.3, 1.3, 0.4], "rot": [0, 0, 0], "time": 0.8},
            {"hand": "both", "pos": [0, 1.2, 0.3], "rot": [0, 0, 0], "time": 1.6}
        ],
        "duration": 2.0,
        "desc": "Both hands pull inward",
        "type": "sign"
    },
    
    # ===== BASIC RESPONSES =====
    "yes": {
        "name": "YES",
        "keyframes": [
            {"hand": "right", "pos": [0, 1.4, 0.4], "rot": [0, 0, 0], "time": 0.0},
            {"hand": "right", "pos": [0, 1.4, 0.4], "rot": [25, 0, 0], "time": 0.3},
            {"hand": "right", "pos": [0, 1.4, 0.4], "rot": [0, 0, 0], "time": 0.6},
            {"hand": "right", "pos": [0, 1.4, 0.4], "rot": [25, 0, 0], "time": 0.9},
            {"hand": "right", "pos": [0, 1.4, 0.4], "rot": [0, 0, 0], "time": 1.2}
        ],
        "duration": 1.3,
        "desc": "Fist nod up and down",
        "type": "sign"
    },
    
    "no": {
        "name": "NO",
        "keyframes": [
            {"hand": "right", "pos": [0.2, 1.5, 0.4], "rot": [0, 0, 0], "time": 0.0},
            {"hand": "right", "pos": [-0.2, 1.5, 0.4], "rot": [0, 0, 0], "time": 0.4},
            {"hand": "right", "pos": [0.2, 1.5, 0.4], "rot": [0, 0, 0], "time": 0.8},
            {"hand": "right", "pos": [-0.2, 1.5, 0.4], "rot": [0, 0, 0], "time": 1.2}
        ],
        "duration": 1.5,
        "desc": "Shake finger side to side",
        "type": "sign"
    },
    
    "ok": {
        "name": "OK",
        "keyframes": [
            {"hand": "right", "pos": [0, 1.4, 0.4], "rot": [0, 0, 0], "time": 0.0},
            {"hand": "right", "pos": [0, 1.4, 0.4], "rot": [0, 0, 0], "time": 0.8}
        ],
        "duration": 1.0,
        "desc": "OK hand sign",
        "type": "sign"
    },
    
    "good": {
        "name": "GOOD",
        "keyframes": [
            {"hand": "right", "pos": [0, 1.4, 0.3], "rot": [0, 0, 0], "time": 0.0},
            {"hand": "right", "pos": [0, 1.5, 0.3], "rot": [-10, 0, 0], "time": 0.6}
        ],
        "duration": 1.0,
        "desc": "Thumbs up",
        "type": "sign"
    },
    
    "bad": {
        "name": "BAD",
        "keyframes": [
            {"hand": "right", "pos": [0, 1.4, 0.3], "rot": [0, 0, 180], "time": 0.0},
            {"hand": "right", "pos": [0, 1.3, 0.3], "rot": [0, 0, 180], "time": 0.6}
        ],
        "duration": 1.0,
        "desc": "Thumbs down",
        "type": "sign"
    },
    
    # ===== PRONOUNS =====
    "i": {
        "name": "I",
        "keyframes": [
            {"hand": "right", "pos": [0, 1.4, 0.3], "rot": [0, 0, 0], "time": 0.0},
            {"hand": "right", "pos": [0, 1.3, 0.25], "rot": [0, 0, 0], "time": 0.5}
        ],
        "duration": 0.8,
        "desc": "Point to chest",
        "type": "sign"
    },
    
    "me": {
        "name": "ME",
        "keyframes": [
            {"hand": "right", "pos": [0, 1.4, 0.3], "rot": [0, 0, 0], "time": 0.0},
            {"hand": "right", "pos": [0, 1.3, 0.25], "rot": [0, 0, 0], "time": 0.5}
        ],
        "duration": 0.8,
        "desc": "Point to chest",
        "type": "sign"
    },
    
    "you": {
        "name": "YOU",
        "keyframes": [
            {"hand": "right", "pos": [0, 1.5, 0.4], "rot": [0, 0, 0], "time": 0.0},
            {"hand": "right", "pos": [0, 1.5, 0.6], "rot": [0, 0, 0], "time": 0.5}
        ],
        "duration": 0.8,
        "desc": "Point forward",
        "type": "sign"
    },
    
    "we": {
        "name": "WE",
        "keyframes": [
            {"hand": "right", "pos": [-0.2, 1.4, 0.3], "rot": [0, 0, 0], "time": 0.0},
            {"hand": "right", "pos": [0.2, 1.4, 0.3], "rot": [0, 0, 0], "time": 0.6},
            {"hand": "right", "pos": [0, 1.4, 0.3], "rot": [0, 0, 0], "time": 1.2}
        ],
        "duration": 1.5,
        "desc": "Circle including everyone",
        "type": "sign"
    },
    
    # ===== ACTIONS =====
    "want": {
        "name": "WANT",
        "keyframes": [
            {"hand": "both", "pos": [0.3, 1.3, 0.6], "rot": [0, 0, 0], "time": 0.0},
            {"hand": "both", "pos": [0.2, 1.2, 0.4], "rot": [0, 0, 0], "time": 0.7},
            {"hand": "both", "pos": [0, 1.1, 0.3], "rot": [0, 0, 0], "time": 1.4}
        ],
        "duration": 1.6,
        "desc": "Pull hands toward body",
        "type": "sign"
    },
    
    "need": {
        "name": "NEED",
        "keyframes": [
            {"hand": "right", "pos": [0, 1.4, 0.4], "rot": [0, 0, 0], "time": 0.0},
            {"hand": "right", "pos": [0, 1.2, 0.4], "rot": [30, 0, 0], "time": 0.6},
            {"hand": "right", "pos": [0, 1.0, 0.4], "rot": [45, 0, 0], "time": 1.2}
        ],
        "duration": 1.4,
        "desc": "Bent hand pull down",
        "type": "sign"
    },
    
    "help": {
        "name": "HELP",
        "keyframes": [
            {"hand": "left", "pos": [-0.2, 1.2, 0.3], "rot": [0, 0, 0], "time": 0.0},
            {"hand": "right", "pos": [0, 1.2, 0.3], "rot": [0, 0, 0], "time": 0.0},
            {"hand": "right", "pos": [0, 1.5, 0.3], "rot": [0, 0, 0], "time": 0.9},
            {"hand": "right", "pos": [0, 1.6, 0.3], "rot": [0, 0, 0], "time": 1.8}
        ],
        "duration": 2.0,
        "desc": "Right fist on left palm, lift",
        "type": "sign"
    },
    
    "go": {
        "name": "GO",
        "keyframes": [
            {"hand": "both", "pos": [0, 1.3, 0.3], "rot": [0, 0, 0], "time": 0.0},
            {"hand": "both", "pos": [0, 1.3, 0.6], "rot": [0, 0, 0], "time": 0.7}
        ],
        "duration": 1.0,
        "desc": "Point forward",
        "type": "sign"
    },
    
    "come": {
        "name": "COME",
        "keyframes": [
            {"hand": "right", "pos": [0, 1.3, 0.6], "rot": [0, 0, 0], "time": 0.0},
            {"hand": "right", "pos": [0, 1.3, 0.3], "rot": [0, 0, 0], "time": 0.7}
        ],
        "duration": 1.0,
        "desc": "Beckon with hand",
        "type": "sign"
    },
    
    "stop": {
        "name": "STOP",
        "keyframes": [
            {"hand": "right", "pos": [0, 1.2, 0.4], "rot": [0, 0, 0], "time": 0.0},
            {"hand": "right", "pos": [0, 1.4, 0.5], "rot": [0, 0, 0], "time": 0.5}
        ],
        "duration": 0.8,
        "desc": "Hand up, palm forward",
        "type": "sign"
    },
    
    "sit": {
        "name": "SIT",
        "keyframes": [
            {"hand": "both", "pos": [0, 1.4, 0.4], "rot": [0, 0, 0], "time": 0.0},
            {"hand": "both", "pos": [0, 1.2, 0.4], "rot": [0, 0, 0], "time": 0.6}
        ],
        "duration": 1.0,
        "desc": "Two hands down",
        "type": "sign"
    },
    
    "stand": {
        "name": "STAND",
        "keyframes": [
            {"hand": "both", "pos": [0, 1.1, 0.4], "rot": [0, 0, 0], "time": 0.0},
            {"hand": "both", "pos": [0, 1.4, 0.4], "rot": [0, 0, 0], "time": 0.6}
        ],
        "duration": 1.0,
        "desc": "Two hands up",
        "type": "sign"
    },
    
    # ===== FOOD & DRINK =====
    "water": {
        "name": "WATER",
        "keyframes": [
            {"hand": "right", "pos": [0.1, 1.3, 0.4], "rot": [0, 0, 0], "time": 0.0},
            {"hand": "right", "pos": [0.1, 1.5, 0.3], "rot": [-25, 0, 0], "time": 0.7},
            {"hand": "right", "pos": [0.1, 1.6, 0.2], "rot": [-40, 0, 0], "time": 1.4}
        ],
        "duration": 1.6,
        "desc": "W-shape to mouth",
        "type": "sign"
    },
    
    "drink": {
        "name": "DRINK",
        "keyframes": [
            {"hand": "right", "pos": [0.1, 1.3, 0.4], "rot": [0, 0, 0], "time": 0.0},
            {"hand": "right", "pos": [0.1, 1.5, 0.3], "rot": [-30, 0, 0], "time": 0.6},
            {"hand": "right", "pos": [0.1, 1.5, 0.4], "rot": [0, 0, 0], "time": 1.2}
        ],
        "duration": 1.4,
        "desc": "Cup hand to mouth",
        "type": "sign"
    },
    
    "food": {
        "name": "FOOD",
        "keyframes": [
            {"hand": "right", "pos": [0, 1.5, 0.3], "rot": [0, 0, 0], "time": 0.0},
            {"hand": "right", "pos": [0, 1.6, 0.2], "rot": [0, 0, 0], "time": 0.4},
            {"hand": "right", "pos": [0, 1.5, 0.3], "rot": [0, 0, 0], "time": 0.8},
            {"hand": "right", "pos": [0, 1.6, 0.2], "rot": [0, 0, 0], "time": 1.2}
        ],
        "duration": 1.5,
        "desc": "Pinch fingers to mouth",
        "type": "sign"
    },
    
    "eat": {
        "name": "EAT",
        "keyframes": [
            {"hand": "right", "pos": [0, 1.5, 0.3], "rot": [0, 0, 0], "time": 0.0},
            {"hand": "right", "pos": [0, 1.6, 0.2], "rot": [0, 0, 0], "time": 0.4},
            {"hand": "right", "pos": [0, 1.5, 0.3], "rot": [0, 0, 0], "time": 0.8}
        ],
        "duration": 1.2,
        "desc": "Pinch to mouth",
        "type": "sign"
    },
    
    # ===== PLACES =====
    "home": {
        "name": "HOME",
        "keyframes": [
            {"hand": "right", "pos": [0.1, 1.5, 0.3], "rot": [0, 0, 0], "time": 0.0},
            {"hand": "right", "pos": [0, 1.4, 0.3], "rot": [0, 0, -30], "time": 0.6},
            {"hand": "right", "pos": [-0.1, 1.3, 0.3], "rot": [0, 0, -45], "time": 1.2}
        ],
        "duration": 1.5,
        "desc": "Outline roof shape",
        "type": "sign"
    },
    
    "school": {
        "name": "SCHOOL",
        "keyframes": [
            {"hand": "both", "pos": [0.3, 1.4, 0.3], "rot": [0, 0, 0], "time": 0.0},
            {"hand": "both", "pos": [0.3, 1.4, 0.3], "rot": [0, 0, 0], "time": 0.5},
            {"hand": "both", "pos": [0, 1.4, 0.3], "rot": [0, 0, 0], "time": 1.0}
        ],
        "duration": 1.5,
        "desc": "Clap hands twice",
        "type": "sign"
    },
    
    "work": {
        "name": "WORK",
        "keyframes": [
            {"hand": "right", "pos": [0, 1.3, 0.4], "rot": [0, 0, 0], "time": 0.0},
            {"hand": "right", "pos": [0.1, 1.3, 0.4], "rot": [0, 0, 0], "time": 0.3},
            {"hand": "right", "pos": [0, 1.3, 0.4], "rot": [0, 0, 0], "time": 0.6},
            {"hand": "right", "pos": [0.1, 1.3, 0.4], "rot": [0, 0, 0], "time": 0.9}
        ],
        "duration": 1.2,
        "desc": "Pound fist motion",
        "type": "sign"
    },
    
    # ===== EMOTIONS =====
    "happy": {
        "name": "HAPPY",
        "keyframes": [
            {"hand": "right", "pos": [0, 1.3, 0.3], "rot": [0, 0, 0], "time": 0.0},
            {"hand": "right", "pos": [0, 1.5, 0.3], "rot": [0, 0, 0], "time": 0.5},
            {"hand": "right", "pos": [0, 1.3, 0.3], "rot": [0, 0, 0], "time": 1.0},
            {"hand": "right", "pos": [0, 1.5, 0.3], "rot": [0, 0, 0], "time": 1.5}
        ],
        "duration": 1.8,
        "desc": "Brush up on chest twice",
        "type": "sign"
    },
    
    "sad": {
        "name": "SAD",
        "keyframes": [
            {"hand": "both", "pos": [0.15, 1.5, 0.3], "rot": [0, 0, 0], "time": 0.0},
            {"hand": "both", "pos": [0.15, 1.3, 0.3], "rot": [0, 0, 0], "time": 0.8},
            {"hand": "both", "pos": [0.15, 1.1, 0.3], "rot": [0, 0, 0], "time": 1.6}
        ],
        "duration": 2.0,
        "desc": "Both hands down face",
        "type": "sign"
    },
    
    "angry": {
        "name": "ANGRY",
        "keyframes": [
            {"hand": "both", "pos": [0.2, 1.5, 0.3], "rot": [0, 0, 0], "time": 0.0},
            {"hand": "both", "pos": [0.2, 1.4, 0.35], "rot": [15, 0, 0], "time": 0.4},
            {"hand": "both", "pos": [0.2, 1.3, 0.4], "rot": [30, 0, 0], "time": 0.8}
        ],
        "duration": 1.2,
        "desc": "Clawed hands from face",
        "type": "sign"
    },
    
    # ===== TIME =====
    "morning": {
        "name": "MORNING",
        "keyframes": [
            {"hand": "right", "pos": [-0.3, 1.0, 0.4], "rot": [0, 0, 0], "time": 0.0},
            {"hand": "right", "pos": [0, 1.3, 0.4], "rot": [0, 0, 30], "time": 0.8},
            {"hand": "right", "pos": [0.3, 1.5, 0.4], "rot": [0, 0, 45], "time": 1.6}
        ],
        "duration": 2.0,
        "desc": "Sunrise motion",
        "type": "sign"
    },
    
    "afternoon": {
        "name": "AFTERNOON",
        "keyframes": [
            {"hand": "right", "pos": [0, 1.6, 0.4], "rot": [0, 0, 0], "time": 0.0},
            {"hand": "right", "pos": [0.2, 1.4, 0.4], "rot": [0, 0, -30], "time": 1.0}
        ],
        "duration": 1.5,
        "desc": "Sun at peak",
        "type": "sign"
    },
    
    "evening": {
        "name": "EVENING",
        "keyframes": [
            {"hand": "right", "pos": [0.3, 1.5, 0.4], "rot": [0, 0, 45], "time": 0.0},
            {"hand": "right", "pos": [0, 1.3, 0.4], "rot": [0, 0, 0], "time": 0.8},
            {"hand": "right", "pos": [-0.2, 1.1, 0.4], "rot": [0, 0, -30], "time": 1.6}
        ],
        "duration": 2.0,
        "desc": "Sunset motion",
        "type": "sign"
    },
    
    "night": {
        "name": "NIGHT",
        "keyframes": [
            {"hand": "right", "pos": [0.3, 1.5, 0.4], "rot": [0, 0, 45], "time": 0.0},
            {"hand": "right", "pos": [0, 1.3, 0.4], "rot": [0, 0, 0], "time": 0.8},
            {"hand": "right", "pos": [-0.3, 1.0, 0.4], "rot": [0, 0, -45], "time": 1.6}
        ],
        "duration": 2.0,
        "desc": "Sunset motion",
        "type": "sign"
    },
    
    "today": {
        "name": "TODAY",
        "keyframes": [
            {"hand": "both", "pos": [0, 1.4, 0.4], "rot": [0, 0, 0], "time": 0.0},
            {"hand": "both", "pos": [0, 1.2, 0.4], "rot": [0, 0, 0], "time": 0.7}
        ],
        "duration": 1.0,
        "desc": "Both hands down",
        "type": "sign"
    },
    
    "tomorrow": {
        "name": "TOMORROW",
        "keyframes": [
            {"hand": "right", "pos": [0, 1.5, 0.3], "rot": [0, 0, 0], "time": 0.0},
            {"hand": "right", "pos": [0.2, 1.5, 0.5], "rot": [0, 0, 30], "time": 0.7}
        ],
        "duration": 1.0,
        "desc": "Move forward",
        "type": "sign"
    },
    
    "yesterday": {
        "name": "YESTERDAY",
        "keyframes": [
            {"hand": "right", "pos": [0, 1.5, 0.3], "rot": [0, 0, 0], "time": 0.0},
            {"hand": "right", "pos": [-0.2, 1.5, 0.2], "rot": [0, 0, -30], "time": 0.7}
        ],
        "duration": 1.0,
        "desc": "Move backward",
        "type": "sign"
    },
    
    "now": {
        "name": "NOW",
        "keyframes": [
            {"hand": "both", "pos": [0, 1.4, 0.4], "rot": [0, 0, 0], "time": 0.0},
            {"hand": "both", "pos": [0, 1.2, 0.4], "rot": [0, 0, 0], "time": 0.5}
        ],
        "duration": 0.8,
        "desc": "Sharp down motion",
        "type": "sign"
    },
    
    # ===== QUESTIONS =====
    "how": {
        "name": "HOW",
        "keyframes": [
            {"hand": "both", "pos": [0.2, 1.2, 0.4], "rot": [0, 0, 0], "time": 0.0},
            {"hand": "both", "pos": [0.2, 1.4, 0.4], "rot": [0, 0, 0], "time": 0.5}
        ],
        "duration": 0.8,
        "desc": "Both hands up",
        "type": "sign"
    },
    
    "what": {
        "name": "WHAT",
        "keyframes": [
            {"hand": "both", "pos": [0, 1.4, 0.4], "rot": [0, 0, 0], "time": 0.0},
            {"hand": "both", "pos": [0.2, 1.4, 0.4], "rot": [0, 0, 15], "time": 0.3},
            {"hand": "both", "pos": [-0.2, 1.4, 0.4], "rot": [0, 0, -15], "time": 0.6}
        ],
        "duration": 0.9,
        "desc": "Shake hands side to side",
        "type": "sign"
    },
    
    "where": {
        "name": "WHERE",
        "keyframes": [
            {"hand": "right", "pos": [0, 1.5, 0.4], "rot": [0, 0, 0], "time": 0.0},
            {"hand": "right", "pos": [0.2, 1.5, 0.4], "rot": [0, 0, 20], "time": 0.4},
            {"hand": "right", "pos": [-0.2, 1.5, 0.4], "rot": [0, 0, -20], "time": 0.8}
        ],
        "duration": 1.0,
        "desc": "Point and search",
        "type": "sign"
    },
    
    "when": {
        "name": "WHEN",
        "keyframes": [
            {"hand": "right", "pos": [0, 1.4, 0.4], "rot": [0, 0, 0], "time": 0.0},
            {"hand": "right", "pos": [0, 1.4, 0.4], "rot": [0, 0, 360], "time": 0.8}
        ],
        "duration": 1.0,
        "desc": "Circle with finger",
        "type": "sign"
    },
    
    "why": {
        "name": "WHY",
        "keyframes": [
            {"hand": "right", "pos": [0.1, 1.6, 0.3], "rot": [0, 0, 0], "time": 0.0},
            {"hand": "right", "pos": [0.1, 1.5, 0.4], "rot": [0, 0, 0], "time": 0.7}
        ],
        "duration": 1.0,
        "desc": "Touch forehead and out",
        "type": "sign"
    },
    
    # ===== OTHERS =====
    "are": {
        "name": "ARE",
        "keyframes": [
            {"hand": "right", "pos": [0, 1.4, 0.3], "rot": [0, 0, 0], "time": 0.0},
            {"hand": "right", "pos": [0, 1.4, 0.5], "rot": [0, 0, 0], "time": 0.5}
        ],
        "duration": 0.7,
        "desc": "Move forward",
        "type": "sign"
    },
    
    "love": {
        "name": "LOVE",
        "keyframes": [
            {"hand": "both", "pos": [0, 1.3, 0.2], "rot": [0, 0, 0], "time": 0.0},
            {"hand": "both", "pos": [0, 1.3, 0.25], "rot": [0, 0, 0], "time": 0.8}
        ],
        "duration": 1.2,
        "desc": "Cross arms over chest",
        "type": "sign"
    },
}

# ==================== FLASK SETUP ====================

app = Flask(__name__)
app.config['SECRET_KEY'] = 'isl-avatar-secret-2024'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Speech recognizer
recognizer = sr.Recognizer()
recognizer.energy_threshold = 4000
recognizer.dynamic_energy_threshold = True
recognizer.pause_threshold = 0.8

# System statistics
stats = {
    "total_conversions": 0,
    "total_gestures": 0,
    "speech_recognitions": 0,
    "errors": 0,
    "start_time": datetime.now()
}

# ==================== CONVERTER CLASS ====================

class SpeechToSignConverter:
    """Convert speech/text to ISL gesture sequence"""
    
    def __init__(self):
        self.gesture_db = ISL_GESTURES
        
    def text_to_signs(self, text):
        """Convert text to sign sequence"""
        # Clean and split text
        text = text.lower().strip()
        text = text.replace(",", "").replace(".", "").replace("!", "").replace("?", "")
        words = text.split()
        
        sign_sequence = []
        unknown_words = []
        
        for word in words:
            if word in self.gesture_db:
                gesture = self.gesture_db[word].copy()
                gesture["word"] = word
                sign_sequence.append(gesture)
            else:
                # Fingerspell unknown words
                unknown_words.append(word)
                fingerspell = self._create_fingerspell(word)
                sign_sequence.append(fingerspell)
        
        return {
            "signs": sign_sequence,
            "total_duration": sum(s["duration"] for s in sign_sequence),
            "word_count": len(words),
            "unknown_words": unknown_words
        }
    
    def _create_fingerspell(self, word):
        """Create fingerspelling animation for unknown words"""
        keyframes = []
        base_pos = [0.2, 1.4, 0.4]
        
        for i, letter in enumerate(word.upper()):
            keyframes.append({
                "hand": "right",
                "pos": [base_pos[0] + i * 0.05, base_pos[1], base_pos[2]],
                "rot": [0, 0, i * 3],
                "time": i * 0.5,
                "letter": letter
            })
        
        # Add final keyframe
        keyframes.append({
            "hand": "right",
            "pos": [base_pos[0], base_pos[1], base_pos[2]],
            "rot": [0, 0, 0],
            "time": len(word) * 0.5
        })
        
        return {
            "name": f"SPELL_{word.upper()}",
            "word": word,
            "keyframes": keyframes,
            "duration": len(word) * 0.5,
            "type": "fingerspell",
            "desc": f"Fingerspell: {word.upper()}"
        }

converter = SpeechToSignConverter()

# ==================== ROUTES ====================

@app.route('/')
def index():
    """Serve main interface"""
    return render_template('avatar_interface.html')
@app.route('/api/text-to-signs', methods=['POST'])
def text_to_signs_api():
    """Convert text to signs"""
    try:
        data = request.json
        text = data.get('text', '')
        
        if not text:
            return jsonify({"success": False, "error": "No text provided"}), 400
        
        result = converter.text_to_signs(text)
        
        stats["total_conversions"] += 1
        stats["total_gestures"] += len(result["signs"])
        
        return jsonify({
            "success": True,
            "text": text,
            "result": result,
            "stats": get_stats()
        })
        
    except Exception as e:
        stats["errors"] += 1
        print(f"Error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/stats')
def get_stats_api():
    """Get system statistics"""
    return jsonify(get_stats())

@app.route('/api/gestures')
def get_gestures():
    """Get list of available gestures"""
    gestures = []
    
    for word, data in ISL_GESTURES.items():
        gestures.append({
            "word": word,
            "name": data["name"],
            "duration": data["duration"],
            "desc": data["desc"]
        })
    
    return jsonify({"gestures": sorted(gestures, key=lambda x: x["word"])})

@app.route('/api/translate', methods=['POST'])
def translate_signs_api():
    """Translate sign tokens to English"""
    try:
        from translation import translate_signs
        
        data = request.json
        signs = data.get('signs', [])
        
        if not signs:
            return jsonify({"success": False, "error": "No signs provided"}), 400
        
        # Extract sign names from sign objects
        sign_tokens = []
        for sign in signs:
            if isinstance(sign, dict):
                sign_tokens.append(sign.get('name', sign.get('word', '')))
            else:
                sign_tokens.append(sign)
        
        # Translate
        translated = translate_signs(sign_tokens)
        
        return jsonify({
            "success": True,
            "signs": sign_tokens,
            "translation": translated
        })
        
    except Exception as e:
        print(f"Translation error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

def get_stats():
    """Calculate current statistics"""
    uptime = (datetime.now() - stats["start_time"]).total_seconds()
    
    return {
        "total_conversions": stats["total_conversions"],
        "total_gestures": stats["total_gestures"],
        "speech_recognitions": stats["speech_recognitions"],
        "errors": stats["errors"],
        "uptime_seconds": int(uptime),
        "uptime_str": format_uptime(uptime),
        "gestures_available": len(ISL_GESTURES)
    }

def format_uptime(seconds):
    """Format uptime string"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours}h {minutes}m {secs}s"

# ==================== SOCKET.IO EVENTS ====================

@socketio.on('connect')
def handle_connect():
    """Client connected"""
    print('🎭 Avatar client connected')
    emit('status', {'message': 'Connected', 'stats': get_stats()})

@socketio.on('disconnect')
def handle_disconnect():
    """Client disconnected"""
    print('👋 Avatar client disconnected')

@socketio.on('speech_to_sign')
def handle_speech_to_sign(data):
    """Handle text-to-sign conversion"""
    try:
        text = data.get('text', '')
        
        if not text:
            emit('error', {'message': 'No text provided'})
            return
        
        print(f'🗣️  Converting: "{text}"')
        
        result = converter.text_to_signs(text)
        
        stats["total_conversions"] += 1
        stats["total_gestures"] += len(result["signs"])
        
        emit('play_signs', {
            'text': text,
            'result': result,
            'stats': get_stats()
        })
        
    except Exception as e:
        print(f'❌ Error: {e}')
        stats["errors"] += 1
        emit('error', {'message': str(e)})

@socketio.on('record_speech')
def handle_record_speech():
    """Record and convert speech"""
    def record_and_convert():
        try:
            with sr.Microphone() as source:
                print('🎤 Listening...')
                socketio.emit('recording_status', {'status': 'listening'})
                
                # Adjust for ambient noise
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                
                # Listen for speech
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
                
                socketio.emit('recording_status', {'status': 'processing'})
                print('🔄 Processing...')
                
                try:
                    # Try English first
                    try:
                        text = recognizer.recognize_google(audio, language='en-IN')
                    except:
                        # Fallback to Hindi
                        text = recognizer.recognize_google(audio, language='hi-IN')
                    
                    print(f'✅ Recognized: "{text}"')
                    
                    stats["speech_recognitions"] += 1
                    
                    # Convert to signs
                    result = converter.text_to_signs(text)
                    
                    stats["total_conversions"] += 1
                    stats["total_gestures"] += len(result["signs"])
                    
                    socketio.emit('play_signs', {
                        'text': text,
                        'result': result,
                        'stats': get_stats()
                    })
                    
                except sr.UnknownValueError:
                    print('❌ Could not understand')
                    stats["errors"] += 1
                    socketio.emit('error', {
                        'message': 'Could not understand speech. Please try again.'
                    })
                    
                except sr.RequestError as e:
                    print(f'❌ API error: {e}')
                    stats["errors"] += 1
                    socketio.emit('error', {
                        'message': 'Speech recognition service error. Check internet.'
                    })
                
        except Exception as e:
            print(f'❌ Recording error: {e}')
            stats["errors"] += 1
            socketio.emit('error', {'message': f'Recording error: {str(e)}'})
    
    # Run in separate thread
    threading.Thread(target=record_and_convert, daemon=True).start()

# ==================== MAIN ====================

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🎭 SPEECH-TO-SIGN AVATAR SYSTEM v2.0")
    print("="*70)
    print()
    print("✨ Features:")
    print(f"   • {len(ISL_GESTURES)} ISL Gestures")
    print("   • Real-time Speech Recognition")
    print("   • Smooth 3D Animation")
    print("   • Fingerspelling Support")
    print("   • Performance Statistics")
    print()
    print("🌐 Avatar Interface: http://localhost:5001")
    print("📊 Statistics API:   http://localhost:5001/api/stats")
    print("📚 Gestures List:    http://localhost:5001/api/gestures")
    print()
    print("="*70)
    print()
    
    # Create templates folder
    os.makedirs('templates', exist_ok=True)
    
    # Run server
    socketio.run(app, host='0.0.0.0', port=5001, debug=False, allow_unsafe_werkzeug=True)