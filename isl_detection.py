# isl_detection.py — CLOUD PRODUCTION VERSION
#
# ════════════════════════════════════════════════════════════════════════════
#  CLOUD FIXES APPLIED (3 changes only — everything else identical):
#
#  FIX 1 — pygame TTS: init only in LOCAL mode (crashes on headless server)
#  FIX 2 — cv2.CAP_DSHOW: Windows-only flag removed for Linux cloud server
#  FIX 3 — SharedMemory cleanup on startup (prevents crash on restart)
#
#  LOCAL CAMERA FIXES (built-in laptop webcam static/noise):
#  FIX 4 — MJPG codec + 30-frame flush to clear DirectShow buffer on Windows
#  FIX 5 — CAP_PROP_BUFFERSIZE set before flush (not after)
#
#  ALL original features preserved:
#  ✅ Emotion detection v3 (happy/sad/angry/surprise/neutral)
#  ✅ Letter hold timer, same-letter cooldown, confidence buffer
#  ✅ SharedMemory zero-copy frame transfer
#  ✅ CLOUD mode browser frame ingestion (_cloud_frame)
#  ✅ gTTS speak button (disabled silently in cloud — browser TTS handles it)
#  ✅ Word suggestions, auto-TTS, reset, backspace, accept_suggestion
#  ✅ Oracle Free Tier safe: TARGET_FPS=15, EMOTION_PROCESS_INTERVAL=5
# ════════════════════════════════════════════════════════════════════════════

import cv2
import mediapipe as mp
import numpy as np
import string
import time
import copy
import itertools
import os
import base64
import platform
import threading
from collections import deque, defaultdict, Counter
from multiprocessing import Manager, Value, shared_memory
import ctypes

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from tensorflow.keras.models import load_model
from gtts import gTTS
import pygame

try:
    import translation
    TRANSLATION_AVAILABLE = True
except ImportError:
    TRANSLATION_AVAILABLE = False

# ─────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────
RUN_MODE          = os.getenv("RUN_MODE", "LOCAL")
ISL_MODEL_PATH    = "model.h5"
CAM_INDEX         = 0

# ── Letter timing ──────────────────────────────────────────────────────────
LETTER_HOLD_SEC          = 0.9
SAME_LETTER_COOLDOWN_SEC = 1.2

# ── Emotion thresholds ────────────────────────────────────────────────────
SMOOTHING_FRAMES         = 2
MIN_CONF_TO_SHOW         = 0.06
EMOTIONS                 = ["happy", "sad", "angry", "surprise", "neutral"]

CONF_THRESHOLD           = 0.32
SPACE_THRESHOLD          = 18
ENABLE_AUTO_TTS          = True
TTS_LANGUAGE             = "en"

SHOW_LETTER_OVERLAY      = True
OVERLAY_COLOR_LETTER     = (0, 255, 255)
OVERLAY_COLOR_CONF       = (255, 255, 0)
OVERLAY_POSITION         = (20, 50)
OVERLAY_FONT_SCALE       = 1.5
OVERLAY_THICKNESS        = 3

COMMON_WORDS = [
    "HELLO", "HELP", "PLEASE", "THANK", "YOU", "YES", "NO", "GOOD", "BAD",
    "MORNING", "AFTERNOON", "EVENING", "NIGHT", "TODAY", "TOMORROW", "WATER",
    "FOOD", "HOME", "SCHOOL", "WORK", "HAPPY", "SAD", "SORRY", "WELCOME"
]

TARGET_FPS               = 15
CAMERA_RESOLUTION        = (640, 480)
EMOTION_PROCESS_INTERVAL = 5
FRAME_W, FRAME_H         = 640, 480


# ─────────────────────────────────────────────────────────────
#  SHARED STATE
# ─────────────────────────────────────────────────────────────
class SharedState:
    def __init__(self):
        manager = Manager()
        self.ui_queue       = manager.Queue(200)
        self.command_queue  = manager.Queue(20)
        self.processing_fps = Value(ctypes.c_double, 0.0)


# ─────────────────────────────────────────────────────────────
#  TTS ENGINE
#  FIX 1: pygame.mixer.init() only runs in LOCAL mode.
#          On cloud (headless Linux) there is no audio device —
#          init() would crash the entire process at import time.
#          In CLOUD mode TTS is silently disabled here; the browser
#          handles speech via the Web Speech API on the client side.
# ─────────────────────────────────────────────────────────────
TTS_ENABLED = False
if RUN_MODE == "LOCAL":
    try:
        pygame.mixer.init()
        TTS_ENABLED = True
        print("✅ TTS engine initialized")
    except Exception as e:
        print(f"⚠️  TTS init warning: {e}")
else:
    print("ℹ️  TTS disabled (CLOUD/headless mode) — browser handles speech")

speaking_word = ""
pause_audio   = False


def speak_text(text, lang=TTS_LANGUAGE):
    global speaking_word

    # FIX 1 continued: guard every call — silently skip on cloud
    if not TTS_ENABLED or not text or not text.strip():
        return

    def _speak():
        global pause_audio, speaking_word
        try:
            fname = f"voice_{int(time.time()*1000)}.mp3"
            gTTS(text=text, lang=lang).save(fname)
            speaking_word = text
            pygame.mixer.music.load(fname)
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
            time.sleep(0.15)
            for _ in range(3):
                try:
                    if os.path.exists(fname):
                        os.remove(fname)
                    break
                except PermissionError:
                    time.sleep(0.2)
        except Exception as e:
            print(f"TTS error: {e}")
            speaking_word = ""

    threading.Thread(target=_speak, daemon=True).start()


# ─────────────────────────────────────────────────────────────
#  LANDMARK UTILITIES
# ─────────────────────────────────────────────────────────────
def calc_landmark_list(image, landmarks):
    w, h = image.shape[1], image.shape[0]
    return [
        [min(int(lm.x * w), w - 1), min(int(lm.y * h), h - 1)]
        for lm in landmarks.landmark
    ]


def pre_process_landmark(landmark_list):
    temp = copy.deepcopy(landmark_list)
    bx, by = temp[0]
    for i in range(len(temp)):
        temp[i][0] -= bx
        temp[i][1] -= by
    flat = list(itertools.chain.from_iterable(temp))
    mv   = max(map(abs, flat)) if flat else 1
    return [n / mv for n in flat]


def apply_clahe(frame):
    """Enhance low-light / low-contrast frames for mobile cameras."""
    try:
        lab     = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe   = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        return cv2.cvtColor(cv2.merge((clahe.apply(l), a, b)), cv2.COLOR_LAB2BGR)
    except Exception:
        return frame


# ─────────────────────────────────────────────────────────────
#  EMOTION DETECTION — v3
#  Redesigned SAD / ANGRY / SURPRISE with external facial cues.
#  HAPPY unchanged.
# ─────────────────────────────────────────────────────────────

def _dist(a, b):
    """Euclidean distance between two 2-D landmark points."""
    return float(np.linalg.norm(
        np.array(a, dtype=np.float32) - np.array(b, dtype=np.float32)
    ))


def detect_emotion_from_landmarks(face_landmarks, img_shape):
    """
    Derive emotion scores from 468 MediaPipe Face Mesh landmarks.
    Returns {emotion: float} summing to 1.0.

    Key landmark reference:
      Mouth corners : 61 (L), 291 (R)
      Lip inner     : 13 (top), 14 (bottom)
      Lip outer     : 0 (top), 17 (bottom)
      L eye lids    : 159 (upper), 145 (lower)
      R eye lids    : 386 (upper), 374 (lower)
      L eye corners : 33 (outer), 133 (inner)
      R eye corners : 362 (outer), 263 (inner)
      L brow        : 70 (inner), 105 (outer), 52 (arch)
      R brow        : 300 (inner), 334 (outer), 282 (arch)
      Nose tip      : 4
      Nostrils      : 129 (L ala), 358 (R ala)
      Cheeks        : 50 (L), 280 (R)
    """
    try:
        pts    = np.array(face_landmarks, dtype=np.float32)
        face_w = max(float(np.max(pts[:, 0]) - np.min(pts[:, 0])), 1.0)
        face_h = max(float(np.max(pts[:, 1]) - np.min(pts[:, 1])), 1.0)

        def p(i):
            return pts[i]

        def nw(a, b):
            return _dist(p(a), p(b)) / face_w

        def nh(a, b):
            return _dist(p(a), p(b)) / face_h

        def clamp01(v):
            return max(0.0, min(1.0, float(v)))

        # ── Mouth ────────────────────────────────────────
        lc, rc = p(61),  p(291)
        tl, bl = p(13),  p(14)
        ut, lb = p(0),   p(17)

        mid_y       = (tl[1] + bl[1]) / 2.0
        mouth_curve = ((lc[1] - mid_y) + (rc[1] - mid_y)) / (2.0 * face_h)
        mouth_open  = _dist(ut, lb) / face_h
        inner_gap   = _dist(tl, bl) / face_h
        mouth_w     = _dist(lc, rc) / face_w
        o_ratio     = (mouth_open / (mouth_w + 1e-6)) if mouth_w > 0.05 else 0.0

        # ── Eyes ─────────────────────────────────────────
        leu, led = p(159), p(145)
        reu, red = p(386), p(374)
        lel, ler = p(33),  p(133)
        rel, rer = p(362), p(263)

        l_eye_h = _dist(leu, led)
        r_eye_h = _dist(reu, red)
        l_eye_w = _dist(lel, ler)
        r_eye_w = _dist(rel, rer)

        l_ear   = l_eye_h / (l_eye_w + 1e-6)
        r_ear   = r_eye_h / (r_eye_w + 1e-6)
        avg_ear = (l_ear + r_ear) / 2.0
        eye_open = ((l_eye_h + r_eye_h) / 2.0) / face_h

        # ── Eyebrows ──────────────────────────────────────
        lib, rib = p(70),  p(300)
        lob, rob = p(105), p(334)
        lab, rab = p(52),  p(282)

        l_brow_in_h  = (leu[1] - lib[1]) / face_h
        r_brow_in_h  = (reu[1] - rib[1]) / face_h
        avg_brow_in_h = (l_brow_in_h + r_brow_in_h) / 2.0

        l_brow_out_h  = (leu[1] - lob[1]) / face_h
        r_brow_out_h  = (reu[1] - rob[1]) / face_h
        avg_brow_out_h = (l_brow_out_h + r_brow_out_h) / 2.0

        brow_inner_gap = nw(70, 300)
        brow_oblique   = avg_brow_in_h - avg_brow_out_h

        l_arch_h   = (leu[1] - lab[1]) / face_h
        r_arch_h   = (reu[1] - rab[1]) / face_h
        avg_arch_h = (l_arch_h + r_arch_h) / 2.0

        # ── Nose ─────────────────────────────────────────
        nostril_w  = nw(129, 358)
        nose_flare = max(0.0, nostril_w - 0.26)

        # ── Cheeks ───────────────────────────────────────
        lch, rch, nb = p(50), p(280), p(4)
        cheek_raise  = ((nb[1] - lch[1]) + (nb[1] - rch[1])) / (2.0 * face_h)

        # ── HAPPY ────────────────────────────────────────
        happy = 0.0
        if mouth_curve < -0.003:
            happy += clamp01(abs(mouth_curve) / 0.018) * 0.55
        if mouth_w > 0.36:
            happy += clamp01((mouth_w - 0.36) / 0.07) * 0.28
        if cheek_raise > 0.007:
            happy += clamp01(cheek_raise / 0.018) * 0.26
        if avg_ear < 0.27 and eye_open > 0.04:
            happy += 0.12
        if mouth_curve > 0.004:         happy *= 0.15
        if avg_brow_in_h < -0.05:       happy *= 0.40
        if mouth_open > 0.08:           happy *= 0.55
        happy = clamp01(happy * 1.30)

        # ── SAD — v3 ──────────────────────────────────────
        sad = 0.0
        if brow_oblique > 0.02:
            sad += clamp01((brow_oblique - 0.02) / 0.055) * 0.45
        if 0.08 < avg_ear < 0.23:
            peak_dist = abs(avg_ear - 0.16)
            sad += clamp01(1.0 - peak_dist / 0.08) * 0.30
        if 0.02 < eye_open < 0.065:
            sad += clamp01((0.065 - eye_open) / 0.025) * 0.18
        if mouth_curve > 0.002:
            sad += clamp01(mouth_curve / 0.020) * 0.40
        if mouth_w < 0.36:
            sad += clamp01((0.36 - mouth_w) / 0.055) * 0.18
        if avg_brow_in_h > 0.04:
            sad += clamp01((avg_brow_in_h - 0.04) / 0.040) * 0.20
        if mouth_curve < -0.004:                            sad *= 0.08
        if cheek_raise > 0.014:                             sad *= 0.20
        if avg_ear > 0.30:                                  sad *= 0.25
        if avg_brow_in_h < -0.06 and brow_inner_gap < 0.14: sad *= 0.15
        sad = clamp01(sad * 1.50)

        # ── ANGRY — v3 ────────────────────────────────────
        angry = 0.0
        if nose_flare > 0.005:
            angry += clamp01(nose_flare / 0.055) * 0.35
        if avg_brow_in_h < -0.04:
            angry += clamp01(abs(avg_brow_in_h + 0.04) / 0.045) * 0.40
        if brow_inner_gap < 0.155:
            angry += clamp01((0.155 - brow_inner_gap) / 0.045) * 0.35
        if avg_brow_out_h < -0.03:
            angry += clamp01(abs(avg_brow_out_h + 0.03) / 0.040) * 0.20
        if avg_ear < 0.20 and eye_open < 0.055:
            angry += clamp01((0.055 - eye_open) / 0.025) * 0.25
        if inner_gap < 0.030:
            angry += clamp01((0.030 - inner_gap) / 0.025) * 0.22
        if brow_oblique < 0.015:
            angry += 0.08
        if mouth_curve < -0.005:        angry *= 0.15
        if cheek_raise > 0.014:         angry *= 0.20
        if avg_brow_in_h > -0.01:       angry *= 0.20
        if mouth_open > 0.07:           angry *= 0.40
        if brow_oblique > 0.04:         angry *= 0.35
        angry = clamp01(angry * 1.45)

        # ── SURPRISE — v3 ─────────────────────────────────
        surprise = 0.0
        if 0.40 < o_ratio < 1.60:
            peak_dist = abs(o_ratio - 0.90)
            surprise += clamp01(1.0 - peak_dist / 0.55) * 0.50
        if 0.03 < mouth_open < 0.11:
            surprise += clamp01((mouth_open - 0.03) / 0.06) * 0.25
        if mouth_w > 0.40:
            surprise *= max(0.10, 1.0 - (mouth_w - 0.40) / 0.10)
        if avg_ear > 0.27:
            surprise += clamp01((avg_ear - 0.27) / 0.10) * 0.40
        if eye_open > 0.07:
            surprise += clamp01((eye_open - 0.07) / 0.055) * 0.28
        if avg_arch_h > 0.05:
            surprise += clamp01((avg_arch_h - 0.05) / 0.055) * 0.35
        if avg_brow_in_h > 0.04 and avg_brow_out_h > 0.02:
            surprise += 0.15
        if abs(brow_oblique) < 0.025:
            surprise += 0.08
        if mouth_curve < -0.005:        surprise *= 0.30
        if avg_brow_in_h < -0.03:       surprise *= 0.20
        if avg_ear < 0.18:              surprise *= 0.25
        if brow_inner_gap < 0.13:       surprise *= 0.25
        if mouth_w > 0.44:              surprise *= 0.25
        if mouth_open > 0.12 and o_ratio < 0.40:
            surprise *= 0.30
        surprise = clamp01(surprise * 1.35)

        # ── NEUTRAL ───────────────────────────────────────
        neutral = max(0.0, 1.0 - (happy + sad + angry + surprise) * 1.05)

        total = happy + sad + angry + surprise + neutral
        if total > 1e-6:
            happy    /= total
            sad      /= total
            angry    /= total
            surprise /= total
            neutral  /= total

        return {k: round(float(v), 4)
                for k, v in zip(EMOTIONS, [happy, sad, angry, surprise, neutral])}

    except Exception as ex:
        print(f"⚠️  Emotion detection error: {ex}")
        return {e: (1.0 if e == "neutral" else 0.0) for e in EMOTIONS}


# ─────────────────────────────────────────────────────────────
#  DISPLAY HELPERS
# ─────────────────────────────────────────────────────────────
def get_word_suggestions(partial, word_freq):
    if not partial:
        return []
    p_up = partial.upper()
    seen = {}
    out  = []
    for word, freq in word_freq.most_common(20):
        if word.startswith(p_up) and word != p_up:
            out.append((word, freq)); seen[word] = True
    for word in COMMON_WORDS:
        if word.startswith(p_up) and word not in seen:
            out.append((word, 0))
    out.sort(key=lambda x: (-x[1], x[0]))
    return [m[0] for m in out[:3]]


def draw_hand_landmarks(frame, hand_landmarks):
    mp.solutions.drawing_utils.draw_landmarks(
        frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS,
        mp.solutions.drawing_utils.DrawingSpec(
            color=(0, 255, 0), thickness=2, circle_radius=3),
        mp.solutions.drawing_utils.DrawingSpec(
            color=(0, 255, 255), thickness=2),
    )


def draw_letter_overlay(frame, letter, confidence):
    if not SHOW_LETTER_OVERLAY or not letter or confidence <= 0:
        return frame
    x, y    = OVERLAY_POSITION
    overlay = frame.copy()
    cv2.rectangle(overlay, (x - 10, y - 40), (x + 220, y + 20), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
    cv2.putText(frame, letter, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, OVERLAY_FONT_SCALE,
                OVERLAY_COLOR_LETTER, OVERLAY_THICKNESS, cv2.LINE_AA)
    tw = cv2.getTextSize(letter, cv2.FONT_HERSHEY_SIMPLEX,
                         OVERLAY_FONT_SCALE, OVERLAY_THICKNESS)[0][0]
    cv2.putText(frame, f"({confidence:.2f})", (x + tw + 10, y),
                cv2.FONT_HERSHEY_SIMPLEX, OVERLAY_FONT_SCALE * 0.7,
                OVERLAY_COLOR_CONF, OVERLAY_THICKNESS - 1, cv2.LINE_AA)
    return frame


# ─────────────────────────────────────────────────────────────
#  CORE PROCESSING ENGINE
# ─────────────────────────────────────────────────────────────
def core_processing_engine(shared_state, shm_name=None, frame_lock_val=None, frame_seq=None):
    print("=" * 70)
    print("🚀 ISL Detection System Starting…  (emotion engine v3)")
    print(f"   RUN_MODE         = {RUN_MODE}")
    print(f"   CONF_THRESHOLD   = {CONF_THRESHOLD}")
    print(f"   LETTER_HOLD_SEC  = {LETTER_HOLD_SEC}s")
    print(f"   SharedMemory     = {'enabled' if shm_name else 'disabled (base64 fallback)'}")
    print("=" * 70)

    # ── SharedMemory init ──────────────────────────────────────────────────
    shm       = None
    shm_array = None

    if shm_name is not None:
        try:
            shm       = shared_memory.SharedMemory(name=shm_name)
            shm_array = np.ndarray((480, 640, 3), dtype=np.uint8, buffer=shm.buf)
            print("✅ SharedMemory connected (Zero-copy mode)")
        except Exception as e:
            print(f"❌ SharedMemory init failed: {e}")
            return

    # ── Load ISL model ─────────────────────────────────────────────────────
    try:
        model = load_model(ISL_MODEL_PATH)
        print("✅ ISL model loaded")
        print(f"   Input  shape : {model.input_shape}")
        print(f"   Output shape : {model.output_shape}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        if shm:
            shm.close()
        return

    try:
        model.predict(np.zeros((1, 42), np.float32), verbose=0)
        print("✅ Model warmed up")
    except Exception as e:
        print(f"⚠️  Warmup: {e}")

    alphabet = list(string.ascii_uppercase) + [str(i) for i in range(1, 10)]
    print(f"   Alphabet : A-Z letters only  (model outputs {model.output_shape[1]} classes, digits ignored)")

    # ── Detection state ────────────────────────────────────────────────────
    current_word             = []
    sentence                 = []
    no_hand_frames           = 0

    candidate_letter         = ""
    candidate_since          = 0.0
    last_accepted_letter     = ""
    last_accepted_time       = 0.0

    letter_confidence_buffer = deque(maxlen=4)
    word_frequency           = Counter()
    current_detected_letter  = ""
    current_confidence       = 0.0

    emotion_history          = deque(maxlen=SMOOTHING_FRAMES)
    current_emotion          = "neutral"
    emotion_scores           = {e: (1.0 if e == "neutral" else 0.0) for e in EMOTIONS}
    emotion_timeline         = deque(maxlen=100)
    emotion_frame_counter    = 0

    frame_count     = 0
    fps_frame_times = deque(maxlen=30)

    stats = {
        "letters_detected": 0,
        "words_formed":     0,
        "sentences_formed": 0,
        "emotion_changes":  0,
        "session_start":    time.time(),
    }

    cmd_queue = shared_state.command_queue
    ui_queue  = shared_state.ui_queue

    # ── Camera init ────────────────────────────────────────────────────────
    print("🎥 Initializing camera…")
    if RUN_MODE == "LOCAL":
        # FIX 2: cv2.CAP_DSHOW is Windows-only — use platform check
        if platform.system() == "Windows":
            cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(CAM_INDEX)   # Linux/Mac — no CAP_DSHOW

        if not cap.isOpened():
            cap = cv2.VideoCapture(CAM_INDEX)   # fallback without backend flag
        if not cap.isOpened():
            print("❌ Camera not found — is another app using it?")
            if shm:
                shm.close()
            return

        # FIX 4: Set MJPG codec BEFORE resolution — forces hardware MJPEG path
        # on built-in laptop webcams, avoiding the raw uncompressed buffer that
        # causes the horizontal static/noise stripes seen in the MJPEG stream.
        # Must be set before width/height or it has no effect on some drivers.
        if platform.system() == "Windows":
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAMERA_RESOLUTION[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_RESOLUTION[1])
        cap.set(cv2.CAP_PROP_FPS,          TARGET_FPS)

        # FIX 5: Buffer size must be 1 to prevent stale frame accumulation.
        # Set AFTER resolution so the driver doesn't reset it.
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # FIX 4 continued: Flush 30 frames (was 10) to drain the DirectShow
        # ring buffer. Built-in webcams pre-fill ~20 frames before stabilizing.
        print("🎥 Flushing camera buffer (30 frames)…")
        for _ in range(30):
            cap.read()

        # Verify the camera is actually producing valid frames
        ret, test_frame = cap.read()
        if not ret or test_frame is None:
            print("❌ Camera flush failed — no valid frame received")
            cap.release()
            if shm:
                shm.close()
            return

        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"✅ Camera: {actual_w}x{actual_h}")
        print("✅ Camera ready!")
    else:
        cap = None
        print("☁️  CLOUD mode — waiting for browser frames via socket")

    print("=" * 70)

    mp_hands = mp.solutions.hands
    mp_face  = mp.solutions.face_mesh

    with mp_hands.Hands(
            static_image_mode=False,
            model_complexity=0,
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        ) as hands, \
         mp_face.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        ) as face_mesh:

        print("✅ MediaPipe initialized — entering main loop")

        while True:
            t0  = time.time()
            now = t0

            # ── Commands ───────────────────────────────────────────────────
            try:
                while not cmd_queue.empty():
                    cmd    = cmd_queue.get_nowait()
                    action = cmd.get("action", "")

                    if action == "reset":
                        current_word         = []
                        sentence             = []
                        no_hand_frames       = 0
                        candidate_letter     = ""
                        candidate_since      = 0.0
                        last_accepted_letter = ""
                        last_accepted_time   = 0.0
                        letter_confidence_buffer.clear()
                        current_detected_letter = ""
                        current_confidence      = 0.0
                        print("🔄 Reset")

                    elif action == "backspace":
                        if current_word:
                            current_word.pop()
                        elif sentence:
                            sentence.pop()

                    elif action == "accept_suggestion":
                        w = cmd.get("word", "")
                        if w:
                            current_word = list(w)

                    elif action == "speak":
                        txt = (cmd.get("text", "")
                               or (" ".join(sentence) if sentence
                                   else "".join(current_word)))
                        if txt:
                            speak_text(txt)  # silently skipped in CLOUD mode

                    elif action == "stop":
                        if cap:
                            cap.release()
                        if shm:
                            shm.close()
                        return
            except Exception:
                pass

            # ── Capture ────────────────────────────────────────────────────
            if RUN_MODE == "LOCAL":
                ret, raw = cap.read()
                if not ret or raw is None:
                    print("⚠️  Camera read failed — retrying")
                    time.sleep(0.02)
                    continue
            else:
                fd = getattr(shared_state, "_cloud_frame", None)
                if fd is None:
                    time.sleep(0.03)
                    continue
                shared_state._cloud_frame = None
                try:
                    raw = cv2.imdecode(
                        np.frombuffer(base64.b64decode(fd), np.uint8),
                        cv2.IMREAD_COLOR)
                    if raw is None:
                        continue
                except Exception:
                    continue

            frame     = cv2.resize(raw, (FRAME_W, FRAME_H))
            frame     = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False

            # ── Hand detection ─────────────────────────────────────────────
            hand_results = hands.process(frame_rgb)
            frame_rgb.flags.writeable = True

            detected_letter = ""
            hand_detected   = False

            if hand_results and hand_results.multi_hand_landmarks:
                hand_detected  = True
                no_hand_frames = 0

                for hlm in hand_results.multi_hand_landmarks:
                    draw_hand_landmarks(frame, hlm)
                    try:
                        lm_list  = calc_landmark_list(frame, hlm)
                        pre_list = pre_process_landmark(lm_list)

                        if model is not None and len(pre_list) == 42:
                            features   = np.array(pre_list, dtype=np.float32).reshape(1, 42)
                            pred       = model.predict(features, verbose=0)
                            prob       = float(np.max(pred))
                            raw_letter = alphabet[int(np.argmax(pred))]

                            # Skip number predictions — only A-Z letters
                            if raw_letter.isdigit():
                                current_detected_letter = ""
                                current_confidence      = 0.0
                                continue

                            current_detected_letter = raw_letter
                            current_confidence      = prob

                            print(f"DEBUG → {raw_letter} {prob:.4f}")

                            if prob >= CONF_THRESHOLD:
                                if raw_letter == candidate_letter:
                                    letter_confidence_buffer.append(prob)
                                else:
                                    candidate_letter = raw_letter
                                    candidate_since  = now
                                    letter_confidence_buffer.clear()
                                    letter_confidence_buffer.append(prob)

                                avg_conf  = float(np.mean(letter_confidence_buffer))
                                hold_time = now - candidate_since

                                if hold_time >= LETTER_HOLD_SEC and avg_conf >= CONF_THRESHOLD:
                                    same_letter_ok = (
                                        raw_letter != last_accepted_letter or
                                        (now - last_accepted_time) >= SAME_LETTER_COOLDOWN_SEC
                                    )
                                    if same_letter_ok:
                                        detected_letter      = raw_letter
                                        last_accepted_letter = raw_letter
                                        last_accepted_time   = now
                                        current_word.append(raw_letter)
                                        stats["letters_detected"] += 1
                                        print(f"✍️  Letter: {raw_letter}  "
                                              f"(held {hold_time:.2f}s, conf {avg_conf:.3f})")
                                        candidate_letter = ""
                                        candidate_since  = 0.0
                                        letter_confidence_buffer.clear()
                            else:
                                candidate_letter = ""
                                candidate_since  = 0.0
                                letter_confidence_buffer.clear()

                    except Exception as e:
                        print(f"⚠️  Prediction error: {e}")

            else:
                no_hand_frames += 1
                letter_confidence_buffer.clear()
                candidate_letter        = ""
                candidate_since         = 0.0
                current_detected_letter = ""
                current_confidence      = 0.0

                if no_hand_frames >= SPACE_THRESHOLD and current_word:
                    word = "".join(current_word)
                    sentence.append(word)
                    word_frequency[word] += 1
                    stats["words_formed"] += 1
                    print(f"📝 Word committed: {word}")
                    if ENABLE_AUTO_TTS:
                        speak_text(word)   # silently skipped in CLOUD mode
                    current_word         = []
                    last_accepted_letter = ""
                    last_accepted_time   = 0.0
                    no_hand_frames       = 0

            frame = draw_letter_overlay(frame, current_detected_letter, current_confidence)

            # ── Emotion (every N frames) ───────────────────────────────────
            emotion_frame_counter += 1
            if emotion_frame_counter % EMOTION_PROCESS_INTERVAL == 0:
                ns = {e: (1.0 if e == "neutral" else 0.0) for e in EMOTIONS}
                try:
                    cr = cv2.cvtColor(apply_clahe(frame), cv2.COLOR_BGR2RGB)
                    cr.flags.writeable = False
                    mr = face_mesh.process(cr)
                    cr.flags.writeable = True
                    if mr and mr.multi_face_landmarks:
                        ns = detect_emotion_from_landmarks(
                            calc_landmark_list(frame, mr.multi_face_landmarks[0]),
                            frame.shape)
                except Exception:
                    pass

                emotion_history.append(ns)
                avg_e = defaultdict(float)
                for d in emotion_history:
                    for k, v in d.items():
                        avg_e[k] += v
                for k in avg_e:
                    avg_e[k] /= len(emotion_history)

                dominant = (max(avg_e, key=avg_e.get)
                            if avg_e and max(avg_e.values()) >= MIN_CONF_TO_SHOW
                            else "neutral")

                if current_emotion != dominant:
                    stats["emotion_changes"] += 1
                    scores_str = " | ".join(
                        [f"{k.upper()}:{avg_e[k]:.3f}" for k in EMOTIONS]
                    )
                    print(f"😊 Emotion: {current_emotion.upper()} → {dominant.upper()}")
                    print(f"   Scores  : {scores_str}")

                current_emotion = dominant
                emotion_scores  = dict(avg_e)
                emotion_timeline.append({
                    "time":    time.time(),
                    "emotion": dominant,
                    "scores":  dict(avg_e),
                })

            # ── FPS ────────────────────────────────────────────────────────
            frame_count += 1
            fps_frame_times.append(time.time())
            if len(fps_frame_times) > 1:
                td = fps_frame_times[-1] - fps_frame_times[0]
                if td > 0:
                    shared_state.processing_fps.value = (len(fps_frame_times) - 1) / td

            # ── Frame output ───────────────────────────────────────────────
            if shm_array is not None:
                shm_array[:] = frame
                if frame_lock_val is not None:
                    frame_lock_val.value = 1
                if frame_seq is not None:
                    frame_seq.value += 1
                frame_base64 = None
            else:
                _, buffer_img = cv2.imencode(
                    ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 65])
                frame_base64 = base64.b64encode(buffer_img.tobytes()).decode("utf-8")

            # ── State packet ───────────────────────────────────────────────
            state_packet = {
                "frame":              frame_base64 if frame_base64 else "",
                "current_word":       "".join(current_word),
                "sentence":           " ".join(sentence),
                "suggestions":        get_word_suggestions(
                                          "".join(current_word), word_frequency),
                "emotion":            current_emotion,
                "emotion_scores":     emotion_scores,
                "emotion_timeline":   list(emotion_timeline)[-20:],
                "stats":              stats.copy(),
                "detected_letter":    detected_letter,
                "hand_detected":      hand_detected,
                "fps":                shared_state.processing_fps.value,
                "timestamp":          time.time(),
                "speaking":           speaking_word,
                "overlay_letter":     current_detected_letter,
                "overlay_confidence": current_confidence,
                "hold_progress":      min(1.0,
                                          (now - candidate_since) / LETTER_HOLD_SEC)
                                      if candidate_letter else 0.0,
            }

            try:
                ui_queue.put_nowait(state_packet)
            except Exception:
                try:
                    ui_queue.get_nowait()
                    ui_queue.put_nowait(state_packet)
                except Exception:
                    pass

            if frame_count == 1:
                print("📤 First frame sent — dashboard can read now")
                if shm_array is not None:
                    print("📤 SharedMemory active — zero-copy mode")

            # ── Frame-rate limiter ─────────────────────────────────────────
            wait = (1.0 / TARGET_FPS) - (time.time() - t0)
            if wait > 0:
                time.sleep(wait)

    if cap:
        cap.release()
    if shm is not None:
        shm.close()

    print("✅ System stopped")


# ─────────────────────────────────────────────────────────────
#  STANDALONE ENTRY POINT
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    FRAME_BYTES = FRAME_W * FRAME_H * 3

    # FIX 3: Clean up leftover shared memory from a previous crash
    # Without this, restarting after a crash raises FileExistsError
    try:
        _old_shm = shared_memory.SharedMemory(name="isl_frame_shm")
        _old_shm.close()
        _old_shm.unlink()
        print("🧹 Cleaned up leftover shared memory from previous run")
    except FileNotFoundError:
        pass  # normal — no leftover

    shm  = shared_memory.SharedMemory(create=True, size=FRAME_BYTES, name="isl_frame_shm")
    seq  = Value(ctypes.c_uint64, 0)
    lock = Value(ctypes.c_bool, False)
    ss   = SharedState()
    try:
        core_processing_engine(ss, shm_name=shm.name, frame_lock_val=lock, frame_seq=seq)
    finally:
        shm.close()
        shm.unlink()