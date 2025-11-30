# isl_detection.py - NATURAL EMOTION DETECTION
# OPTIMIZED FOR: Everyday subtle expressions
# Enhanced sensitivity for happy, sad, angry, surprise detection

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import string
import time
import copy
import itertools
import os
import threading
from collections import deque, defaultdict, Counter
from multiprocessing import Queue, Value
import ctypes

from tensorflow.keras.models import load_model
from fer import FER
from gtts import gTTS
import pygame

# -------------------------------
# ---------- CONFIG -------------
# -------------------------------

ISL_MODEL_PATH = "model.h5"
CAM_INDEX = 0

# NATURAL EMOTION SENSITIVITY: More balanced fusion for everyday expressions
BASE_FER_WEIGHT = 0.45
BASE_LANDMARK_WEIGHT = 0.55
SMOOTHING_FRAMES = 4  # Faster response to emotion changes
MIN_CONF_TO_SHOW = 0.08  # Much more sensitive to subtle emotions

EMOTIONS = ["happy", "sad", "angry", "surprise", "neutral"]

BUFFER_SIZE = 12
CONF_THRESHOLD = 0.8
STABILITY_THRESHOLD = 0.7
SENTENCE_DELAY = 2.0
SPACE_THRESHOLD = 15

ENABLE_AUTO_TTS = True
TTS_LANGUAGE = "en"

SHOW_LETTER_OVERLAY = True
OVERLAY_COLOR_LETTER = (0, 255, 255)
OVERLAY_COLOR_CONF = (255, 255, 0)
OVERLAY_POSITION = (20, 50)
OVERLAY_FONT_SCALE = 1.5
OVERLAY_THICKNESS = 3

COMMON_WORDS = [
    "HELLO", "HELP", "PLEASE", "THANK", "YOU", "YES", "NO", "GOOD", "BAD",
    "MORNING", "AFTERNOON", "EVENING", "NIGHT", "TODAY", "TOMORROW", "WATER",
    "FOOD", "HOME", "SCHOOL", "WORK", "HAPPY", "SAD", "SORRY", "WELCOME"
]

# -------------------------------
# --------- SHARED STATE --------
# -------------------------------

class SharedState:
    def __init__(self):
        self.frame_width = Value(ctypes.c_int, 640)
        self.frame_height = Value(ctypes.c_int, 480)
        self.frame_ready = Value(ctypes.c_bool, False)
        self.ui_queue = Queue(maxsize=2)
        self.command_queue = Queue(maxsize=10)
        self.fps = Value(ctypes.c_double, 0.0)
        self.processing_fps = Value(ctypes.c_double, 0.0)

# -------------------------------
# --------- TTS ENGINE ----------
# -------------------------------

try:
    pygame.mixer.init()
    print("✅ TTS engine initialized")
except Exception as e:
    print(f"⚠️  TTS init warning: {e}")

speaking_word = ""
pause_audio = False

def speak_text(text, lang=TTS_LANGUAGE):
    global speaking_word
    if not text or not text.strip():
        return

    def _speak():
        global pause_audio, speaking_word
        filename = None
        try:
            filename = f"voice_{int(time.time()*1000)}_{threading.get_ident()}.mp3"
            tts = gTTS(text=text, lang=lang)
            tts.save(filename)
            speaking_word = text
            pygame.mixer.music.load(filename)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                if pause_audio:
                    pygame.mixer.music.pause()
                    while pause_audio:
                        time.sleep(0.1)
                    pygame.mixer.music.unpause()
                pygame.time.Clock().tick(10)
            speaking_word = ""
            
            pygame.mixer.music.stop()
            pygame.mixer.music.unload()
            time.sleep(0.1)
            
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    if os.path.exists(filename):
                        os.remove(filename)
                    break
                except PermissionError:
                    if attempt < max_retries - 1:
                        time.sleep(0.2)
        except Exception as e:
            print(f"TTS error: {e}")
            speaking_word = ""

    threading.Thread(target=_speak, daemon=True).start()

# -------------------------------
# --------- UTILITIES -----------
# -------------------------------

def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    return [[min(int(lm.x * image_width), image_width - 1),
             min(int(lm.y * image_height), image_height - 1)] for lm in landmarks.landmark]

def pre_process_landmark(landmark_list):
    temp = copy.deepcopy(landmark_list)
    base_x, base_y = temp[0]
    for i in range(len(temp)):
        temp[i][0] -= base_x
        temp[i][1] -= base_y
    temp = list(itertools.chain.from_iterable(temp))
    max_value = max(list(map(abs, temp))) if temp else 1
    return [n / max_value for n in temp]

def apply_clahe(frame):
    try:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
        l2 = clahe.apply(l)
        lab = cv2.merge((l2, a, b))
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    except Exception:
        return frame

def compute_brightness(frame):
    yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    return float(np.mean(yuv[:,:,0]))

def extract_face_features(face_landmarks, img_shape):
    """Extract 28 facial features for natural emotion detection"""
    try:
        pts = np.array(face_landmarks, dtype=np.float32)
        h, w = img_shape[:2]
        face_width = max(np.max(pts[:,0]) - np.min(pts[:,0]), 1.0)
        face_height = max(np.max(pts[:,1]) - np.min(pts[:,1]), 1.0)

        # === MOUTH REGION ===
        left_corner = pts[61]
        right_corner = pts[291]
        top_lip_center = pts[13]
        bottom_lip_center = pts[14]
        upper_lip_top = pts[0]
        lower_lip_bottom = pts[17]
        upper_lip_left = pts[78]
        upper_lip_right = pts[308]
        lower_lip_left = pts[88]
        lower_lip_right = pts[318]
        
        mouth_width = np.linalg.norm(right_corner - left_corner)
        mouth_height = np.linalg.norm(bottom_lip_center - top_lip_center)
        mouth_aspect_ratio = mouth_width / (mouth_height + 1e-6)
        mouth_openness = mouth_height / face_height
        
        # Mouth curvature (CRITICAL for happy/sad)
        mouth_center_y = (top_lip_center[1] + bottom_lip_center[1]) / 2.0
        left_corner_curve = (left_corner[1] - mouth_center_y) / face_height
        right_corner_curve = (right_corner[1] - mouth_center_y) / face_height
        avg_mouth_curve = (left_corner_curve + right_corner_curve) / 2.0
        
        upper_lip_thickness = np.linalg.norm(upper_lip_top - upper_lip_left) / face_width
        lower_lip_thickness = np.linalg.norm(lower_lip_bottom - lower_lip_left) / face_width
        lip_compression = upper_lip_thickness + lower_lip_thickness
        lip_gap = np.linalg.norm(upper_lip_top - lower_lip_bottom) / face_height
        mouth_width_ratio = mouth_width / face_width
        lower_lip_center_droop = (lower_lip_bottom[1] - bottom_lip_center[1]) / face_height
        mouth_asymmetry = abs(left_corner_curve - right_corner_curve)

        # === EYE REGION ===
        left_eye_top = pts[159]
        left_eye_bottom = pts[145]
        left_eye_left = pts[33]
        left_eye_right = pts[133]
        left_eye_inner = pts[133]
        
        right_eye_top = pts[386]
        right_eye_bottom = pts[374]
        right_eye_left = pts[362]
        right_eye_right = pts[263]
        right_eye_inner = pts[362]
        
        left_eye_height = np.linalg.norm(left_eye_top - left_eye_bottom)
        right_eye_height = np.linalg.norm(right_eye_top - right_eye_bottom)
        avg_eye_openness = ((left_eye_height + right_eye_height) / 2.0) / face_height
        
        left_eye_width = np.linalg.norm(left_eye_right - left_eye_left)
        right_eye_width = np.linalg.norm(right_eye_right - right_eye_left)
        avg_eye_width = ((left_eye_width + right_eye_width) / 2.0) / face_width
        
        left_ear = left_eye_height / (left_eye_width + 1e-6)
        right_ear = right_eye_height / (right_eye_width + 1e-6)
        eye_aspect_ratio = (left_ear + right_ear) / 2.0
        
        left_upper_lid_droop = (left_eye_top[1] - left_eye_inner[1]) / face_height
        right_upper_lid_droop = (right_eye_top[1] - right_eye_inner[1]) / face_height
        avg_upper_lid_droop = (left_upper_lid_droop + right_upper_lid_droop) / 2.0
        
        eye_tension = 1.0 - eye_aspect_ratio

        # === EYEBROW REGION ===
        left_inner_brow = pts[70]
        right_inner_brow = pts[300]
        left_outer_brow = pts[105]
        right_outer_brow = pts[334]
        left_mid_brow = pts[107]
        right_mid_brow = pts[336]
        
        left_brow_height = (left_inner_brow[1] - left_eye_top[1]) / face_height
        right_brow_height = (right_inner_brow[1] - right_eye_top[1]) / face_height
        avg_brow_height = (left_brow_height + right_brow_height) / 2.0
        
        left_brow_slant = (left_outer_brow[1] - left_inner_brow[1]) / face_width
        right_brow_slant = (right_outer_brow[1] - right_inner_brow[1]) / face_width
        avg_brow_slant = (left_brow_slant + right_brow_slant) / 2.0
        
        brow_distance = np.linalg.norm(right_inner_brow - left_inner_brow) / face_width
        
        left_brow_arch = (left_mid_brow[1] - left_inner_brow[1]) / face_height
        right_brow_arch = (right_mid_brow[1] - right_inner_brow[1]) / face_height
        avg_brow_arch = (left_brow_arch + right_brow_arch) / 2.0
        
        left_inner_raise = (left_eye_top[1] - left_inner_brow[1]) / face_height
        right_inner_raise = (right_eye_top[1] - right_inner_brow[1]) / face_height
        inner_brow_raise = (left_inner_raise + right_inner_raise) / 2.0
        
        left_outer_lower = (left_outer_brow[1] - left_eye_top[1]) / face_height
        right_outer_lower = (right_outer_brow[1] - right_eye_top[1]) / face_height
        outer_brow_lower = (left_outer_lower + right_outer_lower) / 2.0
        
        brow_tension = 1.0 / (brow_distance + 0.01)

        # === JAW/CHIN REGION ===
        chin = pts[152]
        left_jaw = pts[172]
        right_jaw = pts[397]
        left_jaw_angle = pts[234]
        right_jaw_angle = pts[454]
        
        jaw_drop = np.linalg.norm(chin - top_lip_center) / face_height
        jaw_width = np.linalg.norm(right_jaw - left_jaw) / face_width
        
        left_jaw_clench = np.linalg.norm(left_jaw_angle - left_corner) / face_width
        right_jaw_clench = np.linalg.norm(right_jaw_angle - right_corner) / face_width
        jaw_clench = (left_jaw_clench + right_jaw_clench) / 2.0
        
        chin_protrusion = (chin[1] - bottom_lip_center[1]) / face_height

        # === CHEEK REGION ===
        left_cheek = pts[50]
        right_cheek = pts[280]
        nose_bottom = pts[2]
        
        left_cheek_height = (nose_bottom[1] - left_cheek[1]) / face_height
        right_cheek_height = (nose_bottom[1] - right_cheek[1]) / face_height
        avg_cheek_raise = (left_cheek_height + right_cheek_height) / 2.0
        
        left_cheek_width = np.linalg.norm(left_cheek - left_corner) / face_width
        right_cheek_width = np.linalg.norm(right_cheek - right_corner) / face_width
        cheek_puff = (left_cheek_width + right_cheek_width) / 2.0

        # === NOSE REGION ===
        nose_tip = pts[1]
        nose_bridge = pts[6]
        left_nostril = pts[98]
        right_nostril = pts[327]
        
        nose_wrinkle = np.linalg.norm(nose_tip - nose_bridge) / face_height
        nostril_width = np.linalg.norm(right_nostril - left_nostril) / face_width

        return {
            # Mouth (8 features)
            "mouth_aspect_ratio": float(mouth_aspect_ratio),
            "mouth_openness": float(mouth_openness),
            "mouth_curve": float(avg_mouth_curve),
            "lip_gap": float(lip_gap),
            "mouth_width_ratio": float(mouth_width_ratio),
            "lip_compression": float(lip_compression),
            "lower_lip_droop": float(lower_lip_center_droop),
            "mouth_asymmetry": float(mouth_asymmetry),
            
            # Eyes (5 features)
            "eye_openness": float(avg_eye_openness),
            "eye_width": float(avg_eye_width),
            "eye_aspect_ratio": float(eye_aspect_ratio),
            "upper_lid_droop": float(avg_upper_lid_droop),
            "eye_tension": float(eye_tension),
            
            # Eyebrows (7 features)
            "brow_height": float(avg_brow_height),
            "brow_slant": float(avg_brow_slant),
            "brow_distance": float(brow_distance),
            "brow_arch": float(avg_brow_arch),
            "inner_brow_raise": float(inner_brow_raise),
            "outer_brow_lower": float(outer_brow_lower),
            "brow_tension": float(brow_tension),
            
            # Jaw/Chin (4 features)
            "jaw_drop": float(jaw_drop),
            "jaw_width": float(jaw_width),
            "jaw_clench": float(jaw_clench),
            "chin_protrusion": float(chin_protrusion),
            
            # Cheeks (2 features)
            "cheek_raise": float(avg_cheek_raise),
            "cheek_puff": float(cheek_puff),
            
            # Nose (2 features)
            "nose_wrinkle": float(nose_wrinkle),
            "nostril_flare": float(nostril_width)
        }
    except Exception as e:
        return {}

def landmark_scores_from_features(feat):
    """Natural emotion detection from facial features"""
    if not feat:
        return {e: 0.0 for e in EMOTIONS[:-1]} | {"neutral": 1.0}

    # Extract all features
    mouth_ratio = feat.get("mouth_aspect_ratio", 1.5)
    mouth_open = feat.get("mouth_openness", 0.0)
    mouth_curve = feat.get("mouth_curve", 0.0)
    lip_gap = feat.get("lip_gap", 0.03)
    mouth_width = feat.get("mouth_width_ratio", 0.4)
    lip_compression = feat.get("lip_compression", 0.1)
    lower_lip_droop = feat.get("lower_lip_droop", 0.0)
    mouth_asymmetry = feat.get("mouth_asymmetry", 0.0)
    
    eye_open = feat.get("eye_openness", 0.06)
    eye_width = feat.get("eye_width", 0.15)
    eye_ar = feat.get("eye_aspect_ratio", 0.3)
    upper_lid_droop = feat.get("upper_lid_droop", 0.0)
    eye_tension = feat.get("eye_tension", 0.0)
    
    brow_height = feat.get("brow_height", -0.08)
    brow_slant = feat.get("brow_slant", 0.0)
    brow_dist = feat.get("brow_distance", 0.15)
    brow_arch = feat.get("brow_arch", 0.0)
    inner_brow_raise = feat.get("inner_brow_raise", 0.0)
    outer_brow_lower = feat.get("outer_brow_lower", 0.0)
    brow_tension = feat.get("brow_tension", 0.0)
    
    jaw_drop = feat.get("jaw_drop", 0.2)
    jaw_width = feat.get("jaw_width", 0.35)
    jaw_clench = feat.get("jaw_clench", 0.15)
    chin_protrusion = feat.get("chin_protrusion", 0.0)
    
    cheek_raise = feat.get("cheek_raise", 0.0)
    cheek_puff = feat.get("cheek_puff", 0.0)
    
    nose_wrinkle = feat.get("nose_wrinkle", 0.0)
    nostril_flare = feat.get("nostril_flare", 0.0)

    # Initialize scores
    happy = 0.0
    sad = 0.0
    angry = 0.0
    surprise = 0.0

    # ========================================
    # HAPPY DETECTION - Natural everyday smiles
    # ========================================
    # Even slight smile activation
    if mouth_curve < -0.002:  # Much more sensitive to upward curve
        happy += min(0.40, abs(mouth_curve) / 0.025)
    
    # Natural smile width
    if mouth_ratio > 1.35:  # Lower threshold
        happy += min(0.25, (mouth_ratio - 1.35) / 0.45)
    
    # Cheek activation
    if cheek_raise > 0.008:  # More sensitive
        happy += min(0.22, cheek_raise / 0.022)
    
    # Cheek puffing
    if cheek_puff > 0.15:  # Lower threshold
        happy += 0.18
    
    # Natural eye squint from smile
    if eye_ar < 0.28 and eye_tension > 0.5:  # More relaxed
        happy += 0.18
    
    # SOFTER EXCLUSIONS - allow more natural happy
    if mouth_curve > 0.003:  # Only exclude strong downturns
        happy *= 0.3
    if brow_height < -0.095:  # Only exclude very angry brows
        happy *= 0.4
    if inner_brow_raise > 0.08:  # Only exclude strong sad brows
        happy *= 0.4
    if mouth_open > 0.07:  # Allow wider smiles
        happy *= 0.6
    
    happy = max(0.0, min(1.0, happy * 1.2))

    # ========================================
    # SAD DETECTION - Natural sadness/concern
    # ========================================
    sad_score = 0.0
    
    # Inner eyebrows pattern - more sensitive
    inner_brow_pull = inner_brow_raise * (1.0 / (brow_dist + 0.01))
    if inner_brow_raise > 0.055 and brow_dist < 0.165:  # More relaxed
        sad_score += min(0.45, inner_brow_pull * 6.0)
    elif inner_brow_raise > 0.055:
        sad_score += min(0.30, inner_brow_raise / 0.028)
    
    # Mouth corners down
    if mouth_curve > 0.001:  # Much more sensitive
        sad_score += min(0.35, mouth_curve / 0.025)
    
    # Lip compression
    if lip_compression < 0.095 and lip_gap < 0.035:  # More relaxed
        sad_score += 0.20
    
    # Narrow mouth
    if mouth_width < 0.395:  # More relaxed
        sad_score += min(0.16, (0.395 - mouth_width) / 0.05)
    
    # Eyebrow slant
    if outer_brow_lower > -0.085:  # More sensitive
        sad_score += min(0.16, abs(outer_brow_lower + 0.085) / 0.025)
    
    # Overall brow position
    if brow_height > -0.078:  # More relaxed
        sad_score += min(0.18, (brow_height + 0.078) / 0.04)
    
    # Natural eye state
    if 0.05 < eye_open < 0.08 and eye_tension < 0.62:  # More relaxed
        sad_score += 0.16
    
    # Droopy eyelids
    if upper_lid_droop < -0.015:  # More sensitive
        sad_score += min(0.14, abs(upper_lid_droop) / 0.03)
    
    # Jaw position
    if jaw_drop < 0.22:  # More relaxed
        sad_score += 0.10
    
    # SOFTER EXCLUSIONS
    if mouth_curve < -0.003:  # Only exclude clear smiles
        sad_score *= 0.2
    if cheek_raise > 0.015:  # Only exclude strong cheek raise
        sad_score *= 0.3
    if brow_height < -0.092:  # Only exclude very lowered brows
        sad_score *= 0.35
    if eye_open > 0.095:  # Only exclude very wide eyes
        sad_score *= 0.4
    if brow_dist < 0.125:  # Only exclude very furrowed
        sad_score *= 0.35
    if mouth_open > 0.055:  # Only exclude wide open
        sad_score *= 0.45
    
    sad = max(0.0, min(1.0, sad_score * 1.5))

    # ========================================
    # ANGRY DETECTION - Natural frustration/annoyance
    # ========================================
    angry_score = 0.0
    
    # Eyebrows furrowed
    if brow_dist < 0.145:  # More sensitive
        angry_score += min(0.42, (0.145 - brow_dist) / 0.042)
    
    # Eyebrows lowered
    if brow_height < -0.088:  # More sensitive
        angry_score += min(0.38, abs(brow_height + 0.088) / 0.032)
    
    # Brow tension
    if brow_tension > 6.8:  # More sensitive
        angry_score += min(0.25, (brow_tension - 6.8) / 5.2)
    
    # Eyes narrowed
    if eye_open < 0.062:  # More sensitive
        angry_score += min(0.30, (0.062 - eye_open) / 0.032)
    
    if eye_tension > 0.62:  # More sensitive
        angry_score += 0.20
    
    # Lips pressed
    if lip_gap < 0.03:  # More sensitive
        angry_score += 0.18
    
    if lip_compression < 0.09:  # More sensitive
        angry_score += 0.16
    
    # Jaw tension
    if jaw_width < 0.335:  # More sensitive
        angry_score += min(0.20, (0.335 - jaw_width) / 0.052)
    
    if jaw_clench < 0.148:  # More sensitive
        angry_score += min(0.16, (0.148 - jaw_clench) / 0.032)
    
    # Eyebrow angle
    if brow_slant > 0.005:  # More sensitive
        angry_score += min(0.18, brow_slant / 0.032)
    
    # Chin protrusion
    if chin_protrusion > 0.17:  # More sensitive
        angry_score += min(0.16, (chin_protrusion - 0.17) / 0.052)
    
    # Nose features
    if nose_wrinkle > 0.038:  # More sensitive
        angry_score += min(0.14, (nose_wrinkle - 0.038) / 0.042)
    
    if nostril_flare > 0.115:  # More sensitive
        angry_score += 0.12
    
    # SOFTER EXCLUSIONS
    if mouth_curve < -0.005:  # Only exclude clear smiles
        angry_score *= 0.2
    if brow_height > -0.065:  # Only exclude clearly raised brows
        angry_score *= 0.3
    if mouth_open > 0.06:  # Allow some mouth opening
        angry_score *= 0.45
    if cheek_raise > 0.013:  # Only exclude strong cheek raise
        angry_score *= 0.35
    if inner_brow_raise > 0.08:  # Only exclude strong inner raise
        angry_score *= 0.4
    if eye_open > 0.09:  # Only exclude very wide eyes
        angry_score *= 0.45
    
    angry = max(0.0, min(1.0, angry_score * 1.5))

    # ========================================
    # SURPRISE DETECTION - Natural surprise/interest
    # ========================================
    surprise_score = 0.0
    
    # Eyebrows raised
    if brow_height > -0.062:  # More sensitive
        surprise_score += min(0.45, (brow_height + 0.062) / 0.048)
    
    # Eyes wide
    if eye_open > 0.082:  # More sensitive
        surprise_score += min(0.40, (eye_open - 0.082) / 0.058)
    
    # Jaw dropped
    if mouth_open > 0.038:  # More sensitive
        surprise_score += min(0.32, (mouth_open - 0.038) / 0.052)
    
    # Eyebrows arched
    if brow_arch < -0.010:  # More sensitive
        surprise_score += min(0.20, abs(brow_arch) / 0.020)
    
    # Brows wide apart
    if brow_dist > 0.155:  # More sensitive
        surprise_score += min(0.16, (brow_dist - 0.155) / 0.045)
    
    # Mouth O-shape
    if 1.15 < mouth_ratio < 1.75 and mouth_open > 0.035:  # More relaxed
        surprise_score += 0.18
    
    # Relaxed eyes
    if eye_tension < 0.52 and eye_open > 0.078:  # More sensitive
        surprise_score += 0.16
    
    # Nostrils flared
    if nostril_flare > 0.125:  # More sensitive
        surprise_score += 0.13
    
    # Neutral mouth curve
    if abs(mouth_curve) < 0.007:  # More relaxed
        surprise_score += 0.10
    
    # SOFTER EXCLUSIONS
    if brow_dist < 0.138:  # Only exclude very furrowed
        surprise_score *= 0.3
    if brow_height < -0.082:  # Only exclude very lowered
        surprise_score *= 0.35
    if abs(mouth_curve) > 0.008:  # Allow some curve
        surprise_score *= 0.4
    if eye_tension > 0.65:  # Only exclude strong tension
        surprise_score *= 0.45
    if cheek_raise > 0.016:  # Only exclude strong raise
        surprise_score *= 0.5
    if lip_gap < 0.028:  # Only exclude very pressed lips
        surprise_score *= 0.55
    
    surprise = max(0.0, min(1.0, surprise_score * 1.4))

    # ========================================
    # NEUTRAL CALCULATION
    # More balanced - allow neutral when emotions are subtle
    # ========================================
    emotions_sum = happy + sad + angry + surprise
    neutral = max(0.0, 1.0 - (emotions_sum * 1.3))  # Reduced multiplier

    # ========================================
    # NORMALIZATION
    # ========================================
    total = happy + sad + angry + surprise + neutral
    if total > 0:
        happy /= total
        sad /= total
        angry /= total
        surprise /= total
        neutral /= total

    return {
        "happy": float(happy),
        "sad": float(sad),
        "angry": float(angry),
        "surprise": float(surprise),
        "neutral": float(neutral)
    }

def get_word_suggestions(partial_word, word_freq):
    if not partial_word:
        return []
    
    partial = partial_word.upper()
    matches = []
    
    for word, freq in word_freq.most_common(20):
        if word.startswith(partial) and word != partial:
            matches.append((word, freq))
    
    for word in COMMON_WORDS:
        if word.startswith(partial) and word not in [m[0] for m in matches]:
            matches.append((word, 0))
    
    matches.sort(key=lambda x: (-x[1], x[0]))
    return [m[0] for m in matches[:3]]

def clear_camera_buffer(cap, num_frames=5):
    for _ in range(num_frames):
        cap.read()

def draw_hand_landmarks(frame, hand_landmarks):
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    
    landmark_style = mp_drawing.DrawingSpec(
        color=(0, 255, 0),
        thickness=3,
        circle_radius=4
    )
    connection_style = mp_drawing.DrawingSpec(
        color=(0, 0, 255),
        thickness=2
    )
    
    mp_drawing.draw_landmarks(
        frame,
        hand_landmarks,
        mp_hands.HAND_CONNECTIONS,
        landmark_style,
        connection_style
    )

def draw_letter_overlay(frame, letter, confidence):
    if not SHOW_LETTER_OVERLAY:
        return frame
    
    if letter and confidence > 0:
        text_letter = f"{letter}"
        text_conf = f"({confidence:.2f})"
        
        x, y = OVERLAY_POSITION
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (x-10, y-40), (x+180, y+20), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        
        cv2.putText(frame, text_letter, (x, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, OVERLAY_FONT_SCALE, 
                   OVERLAY_COLOR_LETTER, OVERLAY_THICKNESS, cv2.LINE_AA)
        
        text_width = cv2.getTextSize(text_letter, cv2.FONT_HERSHEY_SIMPLEX, 
                                     OVERLAY_FONT_SCALE, OVERLAY_THICKNESS)[0][0]
        cv2.putText(frame, text_conf, (x + text_width + 10, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, OVERLAY_FONT_SCALE * 0.7, 
                   OVERLAY_COLOR_CONF, OVERLAY_THICKNESS - 1, cv2.LINE_AA)
    
    return frame

# -------------------------------
# ------- CORE PROCESSOR --------
# -------------------------------

def core_processing_engine(shared_state):
    print("🚀 Starting natural emotion detection (balanced sensitivity)...")
    print("✨ Tuned for everyday expressions:")
    print("   • HAPPY: Slight smiles, natural warmth")
    print("   • SAD: Concern, mild unhappiness, thoughtfulness")
    print("   • ANGRY: Frustration, annoyance, determination")
    print("   • SURPRISE: Interest, mild surprise, curiosity")
    print("   • NEUTRAL: Relaxed, neutral expressions")
    print("   • Balanced fusion for natural, subtle emotions!")
    
    try:
        model = load_model(ISL_MODEL_PATH)
        print("✅ ISL model loaded")
    except Exception as e:
        print(f"❌ Failed to load ISL model: {e}")
        model = None
    
    alphabet = list(string.ascii_uppercase)
    emotion_detector = FER(mtcnn=False)
    
    mp_hands = mp.solutions.hands
    mp_face = mp.solutions.face_mesh
    
    buffer = []
    current_word = []
    sentence = []
    no_hand_frames = 0
    last_letter = None
    letter_confidence_buffer = deque(maxlen=5)
    hand_position_buffer = deque(maxlen=5)
    word_frequency = Counter()
    
    current_detected_letter = ""
    current_confidence = 0.0
    
    emotion_history = deque(maxlen=SMOOTHING_FRAMES)
    current_emotion = "neutral"
    emotion_scores = {e: 0.0 for e in EMOTIONS}
    emotion_timeline = deque(maxlen=100)
    last_emotion = "neutral"
    
    frame_count = 0
    last_fps_update = time.time()
    
    stats = {
        "letters_detected": 0,
        "words_formed": 0,
        "sentences_formed": 0,
        "emotion_changes": 0,
        "session_start": time.time()
    }
    
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("❌ Camera not found")
        return
    
    print("✅ Clearing camera buffer...")
    clear_camera_buffer(cap, 10)
    print("✅ Camera ready - Natural emotion mode active!")
    
    with mp_hands.Hands(model_complexity=0, max_num_hands=1,
                        min_detection_confidence=0.6, 
                        min_tracking_confidence=0.6) as hands, \
         mp_face.FaceMesh(static_image_mode=False, max_num_faces=1,
                         min_detection_confidence=0.5, 
                         min_tracking_confidence=0.5) as face_mesh:
        
        while True:
            loop_start = time.time()
            
            # Command handling
            try:
                while not shared_state.command_queue.empty():
                    cmd = shared_state.command_queue.get_nowait()
                    
                    if cmd['action'] == 'reset':
                        buffer = []
                        current_word = []
                        sentence = []
                        no_hand_frames = 0
                        last_letter = None
                        letter_confidence_buffer.clear()
                        hand_position_buffer.clear()
                        current_detected_letter = ""
                        current_confidence = 0.0
                        print("🔄 Reset executed")
                    
                    elif cmd['action'] == 'backspace':
                        if current_word:
                            removed = current_word.pop()
                            print(f"⌫ Backspace: removed '{removed}'")
                        elif sentence:
                            removed_word = sentence.pop()
                            print(f"⌫ Backspace: removed word '{removed_word}'")
                    
                    elif cmd['action'] == 'accept_suggestion':
                        suggestion = cmd.get('word', '')
                        if suggestion:
                            current_word = list(suggestion)
                            print(f"✅ Accepted suggestion: {suggestion}")
                    
                    elif cmd['action'] == 'speak':
                        text_to_speak = cmd.get('text', '')
                        if not text_to_speak:
                            if sentence:
                                text_to_speak = " ".join(sentence)
                            elif current_word:
                                text_to_speak = "".join(current_word)
                        
                        if text_to_speak:
                            print(f"🔊 Speaking: {text_to_speak}")
                            speak_text(text_to_speak)
                    
                    elif cmd['action'] == 'stop':
                        print("🛑 Stop signal received")
                        cap.release()
                        return
            except:
                pass
            
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Hand detection
            hand_results = hands.process(frame_rgb)
            detected_letter = ""
            hand_detected = False
            
            if hand_results.multi_hand_landmarks:
                hand_detected = True
                no_hand_frames = 0
                
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    draw_hand_landmarks(frame, hand_landmarks)
                    
                    wrist = hand_landmarks.landmark[0]
                    pos = np.array([wrist.x * frame.shape[1], wrist.y * frame.shape[0]])
                    hand_position_buffer.append(pos)
                    
                    is_stable = False
                    if len(hand_position_buffer) >= 3:
                        positions = np.array(hand_position_buffer)
                        variance = np.var(positions, axis=0)
                        stability = 1.0 / (1.0 + np.mean(variance) / 100)
                        is_stable = stability > STABILITY_THRESHOLD
                    
                    try:
                        lm_list = calc_landmark_list(frame, hand_landmarks)
                        pre_list = pre_process_landmark(lm_list)
                        df = pd.DataFrame(pre_list).transpose()
                        
                        if model is not None and is_stable:
                            pred = model.predict(df, verbose=0)
                            prob = float(np.max(pred))
                            letter_confidence_buffer.append(prob)
                            
                            avg_conf = np.mean(letter_confidence_buffer) if letter_confidence_buffer else 0
                            
                            predicted_letter = alphabet[int(np.argmax(pred))]
                            current_detected_letter = predicted_letter
                            current_confidence = avg_conf
                            
                            if avg_conf >= CONF_THRESHOLD:
                                detected_letter = predicted_letter
                                buffer.append(detected_letter)
                                if len(buffer) > BUFFER_SIZE:
                                    buffer.pop(0)
                                if buffer.count(detected_letter) > BUFFER_SIZE // 2:
                                    if (last_letter is None) or (detected_letter != last_letter):
                                        current_word.append(detected_letter)
                                        last_letter = detected_letter
                                        stats["letters_detected"] += 1
                                        print(f"✍️  Letter confirmed: {detected_letter} ({avg_conf:.2f})")
                                        buffer.clear()
                                        letter_confidence_buffer.clear()
                    except Exception as e:
                        pass
            else:
                no_hand_frames += 1
                letter_confidence_buffer.clear()
                current_detected_letter = ""
                current_confidence = 0.0
                
                if no_hand_frames >= SPACE_THRESHOLD:
                    if current_word:
                        word = "".join(current_word)
                        sentence.append(word)
                        word_frequency[word] += 1
                        stats["words_formed"] += 1
                        print(f"📝 Word formed: {word}")
                        
                        if ENABLE_AUTO_TTS:
                            speak_text(word)
                        
                        current_word = []
                        last_letter = None
                    no_hand_frames = 0
            
            frame = draw_letter_overlay(frame, current_detected_letter, current_confidence)
            
            # NATURAL EMOTION DETECTION with balanced weights
            frame_clahe = apply_clahe(frame)
            brightness = compute_brightness(frame_clahe)
            bright_scale = np.clip((brightness / 255.0)*1.2, 0.5, 1.0)
            
            # Landmarks are natural expression detection, balance with FER
            fer_weight = BASE_FER_WEIGHT * bright_scale
            landmark_weight = max(0.0, 1.0 - fer_weight)
            
            fer_probs = {}
            try:
                fer_res = emotion_detector.detect_emotions(frame_clahe)
                if fer_res:
                    fer_probs = fer_res[0]["emotions"]
                else:
                    fer_probs = {"angry":0,"happy":0,"sad":0,"surprise":0,"neutral":1.0}
            except:
                fer_probs = {"angry":0,"happy":0,"sad":0,"surprise":0,"neutral":1.0}
            
            fer_subset = {
                "happy": fer_probs.get("happy",0.0),
                "sad": fer_probs.get("sad",0.0),
                "angry": fer_probs.get("angry",0.0),
                "surprise": fer_probs.get("surprise",0.0),
                "neutral": fer_probs.get("neutral",0.0)
            }
            
            mesh_res = face_mesh.process(frame_rgb)
            landmark_scores = {k:0.0 for k in EMOTIONS}
            if mesh_res.multi_face_landmarks:
                landmarks = mesh_res.multi_face_landmarks[0]
                face_landmarks = calc_landmark_list(frame, landmarks)
                feats = extract_face_features(face_landmarks, frame.shape)
                landmark_scores = landmark_scores_from_features(feats)
            
            # BALANCED FUSION for natural expressions
            fused = {}
            for emo in EMOTIONS:
                f = fer_subset.get(emo, 0.0)
                l = landmark_scores.get(emo, 0.0)
                
                # Balanced approach - both methods contribute
                if f > 0.12 and l > 0.12:
                    # Both agree - boost confidence
                    fused_score = fer_weight * f + landmark_weight * l * 1.15
                else:
                    # Standard fusion
                    fused_score = fer_weight * f + landmark_weight * l
                
                fused[emo] = float(np.clip(fused_score, 0.0, 1.0))
            
            # Re-normalize after fusion
            total_fused = sum(fused.values())
            if total_fused > 0:
                for emo in EMOTIONS:
                    fused[emo] /= total_fused
            
            emotion_history.append(fused)
            avg_emotions = defaultdict(float)
            for d in emotion_history:
                for k,v in d.items():
                    avg_emotions[k] += v
            if len(emotion_history) > 0:
                for k in avg_emotions:
                    avg_emotions[k] = avg_emotions[k] / len(emotion_history)
            
            if avg_emotions:
                max_emo = max(avg_emotions, key=avg_emotions.get)
                max_val = avg_emotions[max_emo]
                # Much lower threshold for natural expressions
                dominant = max_emo if max_val >= 0.08 else "neutral"
            else:
                dominant = "neutral"
                avg_emotions = {k:0.0 for k in EMOTIONS}
                avg_emotions["neutral"] = 1.0
            
            if current_emotion != dominant and dominant != last_emotion:
                stats["emotion_changes"] += 1
                
                # Natural emotion logging
                scores_str = " | ".join([f"{k.upper()}:{avg_emotions[k]:.3f}" for k in EMOTIONS])
                print(f"😊 Emotion: {current_emotion.upper()} → {dominant.upper()}")
                print(f"   Scores: {scores_str}")
                last_emotion = current_emotion
            
            current_emotion = dominant
            emotion_scores = dict(avg_emotions)
            
            emotion_timeline.append({
                "time": time.time(),
                "emotion": dominant,
                "scores": dict(avg_emotions)
            })
            
            # Calculate FPS
            frame_count += 1
            now = time.time()
            if now - last_fps_update >= 0.5:
                fps = frame_count / (now - last_fps_update)
                shared_state.processing_fps.value = fps
                frame_count = 0
                last_fps_update = now
            
            # Get suggestions
            current_display = "".join(current_word)
            suggestions = get_word_suggestions(current_display, word_frequency)
            
            # Prepare data packet for UI
            try:
                _, buffer_img = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                frame_bytes = buffer_img.tobytes()
                
                data_packet = {
                    "frame": frame_bytes,
                    "frame_shape": frame.shape,
                    "current_word": current_display,
                    "sentence": " ".join(sentence),
                    "suggestions": suggestions,
                    "emotion": current_emotion,
                    "emotion_scores": emotion_scores,
                    "emotion_timeline": list(emotion_timeline)[-20:],
                    "stats": stats.copy(),
                    "detected_letter": detected_letter,
                    "hand_detected": hand_detected,
                    "fps": shared_state.processing_fps.value,
                    "timestamp": time.time(),
                    "speaking": speaking_word,
                    "overlay_letter": current_detected_letter,
                    "overlay_confidence": current_confidence
                }
                
                if shared_state.ui_queue.full():
                    try:
                        shared_state.ui_queue.get_nowait()
                    except:
                        pass
                
                shared_state.ui_queue.put_nowait(data_packet)
            except:
                pass
            
            # Dynamic sleep
            loop_time = time.time() - loop_start
            target_frame_time = 0.033
            if loop_time < target_frame_time:
                time.sleep(target_frame_time - loop_time)
            
            if frame_count % 30 == 0:
                clear_camera_buffer(cap, 2)
    
    cap.release()
    print("✅ Emotion detector stopped - Natural expression mode complete!")

if __name__ == "__main__":
    shared_state = SharedState()
    core_processing_engine(shared_state)