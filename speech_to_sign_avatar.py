"""
ISL Fingerspelling System - Backend Server
Full A-Z Letter-by-Letter Speech-to-Sign Conversion

Install dependencies:
pip install SpeechRecognition flask flask-socketio eventlet

Usage:
python speech_to_sign_avatar_fixed.py

Then open: http://localhost:5001
"""

import eventlet
eventlet.monkey_patch()

import os
import json
import time
import threading
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_socketio import SocketIO, emit
import speech_recognition as sr

# ======================================
# RUN MODE CONFIG (LOCAL or CLOUD)
# ======================================

RUN_MODE = os.getenv("RUN_MODE", "LOCAL")

# Disable microphone loading in cloud environment
if RUN_MODE == "CLOUD":
    sr.Microphone = None

os.environ["EVENTLET_NO_GREENDNS"] = "yes"

# ==================== ISL ALPHABET LIBRARY (A-Z) ====================

print("Loading ISL gesture database...")

ISL_LIBRARY = {
    'a': {'name': 'A', 'fingers': {'thumb': 0, 'index': 1.5, 'middle': 1.5, 'ring': 1.5, 'pinky': 1.5}, 'rot': [0, 0, 0], 'desc': 'Fist with thumb out'},
    'b': {'name': 'B', 'fingers': {'thumb': 1.2, 'index': 0, 'middle': 0, 'ring': 0, 'pinky': 0}, 'rot': [0, 0, 0], 'desc': 'All fingers straight up'},
    'c': {'name': 'C', 'fingers': {'thumb': 0.3, 'index': 0.5, 'middle': 0.5, 'ring': 0.5, 'pinky': 0.5}, 'rot': [0, 0, 0], 'desc': 'Curved C shape'},
    'd': {'name': 'D', 'fingers': {'thumb': 1.2, 'index': 0, 'middle': 1.5, 'ring': 1.5, 'pinky': 1.5}, 'rot': [0, 0, 0], 'desc': 'Index up, others closed'},
    'e': {'name': 'E', 'fingers': {'thumb': 0.8, 'index': 1.2, 'middle': 1.2, 'ring': 1.2, 'pinky': 1.2}, 'rot': [0, 0, 0], 'desc': 'All fingers curled'},
    'f': {'name': 'F', 'fingers': {'thumb': 0.8, 'index': 0.8, 'middle': 0, 'ring': 0, 'pinky': 0}, 'rot': [0, 0, 0], 'desc': 'OK sign variant'},
    'g': {'name': 'G', 'fingers': {'thumb': 0, 'index': 0, 'middle': 1.5, 'ring': 1.5, 'pinky': 1.5}, 'rot': [0, 0, 90], 'desc': 'Index and thumb point sideways'},
    'h': {'name': 'H', 'fingers': {'thumb': 1.5, 'index': 0, 'middle': 0, 'ring': 1.5, 'pinky': 1.5}, 'rot': [0, 0, 90], 'desc': 'Two fingers horizontal'},
    'i': {'name': 'I', 'fingers': {'thumb': 1.5, 'index': 1.5, 'middle': 1.5, 'ring': 1.5, 'pinky': 0}, 'rot': [0, 0, 0], 'desc': 'Pinky up'},
    'j': {'name': 'J', 'fingers': {'thumb': 1.5, 'index': 1.5, 'middle': 1.5, 'ring': 1.5, 'pinky': 0}, 'rot': [0, 0, 0], 'desc': 'Pinky draws J'},
    'k': {'name': 'K', 'fingers': {'thumb': 0.5, 'index': 0, 'middle': 0, 'ring': 1.5, 'pinky': 1.5}, 'rot': [0, 0, 45], 'desc': 'Index and middle V shape'},
    'l': {'name': 'L', 'fingers': {'thumb': 0, 'index': 0, 'middle': 1.5, 'ring': 1.5, 'pinky': 1.5}, 'rot': [0, 0, 0], 'desc': 'L shape'},
    'm': {'name': 'M', 'fingers': {'thumb': 0.8, 'index': 1.5, 'middle': 1.5, 'ring': 1.5, 'pinky': 1.5}, 'rot': [0, 0, 0], 'desc': 'Three fingers tucked under thumb'},
    'n': {'name': 'N', 'fingers': {'thumb': 0.8, 'index': 1.5, 'middle': 1.5, 'ring': 1.5, 'pinky': 1.5}, 'rot': [0, 0, 0], 'desc': 'Two fingers tucked'},
    'o': {'name': 'O', 'fingers': {'thumb': 0.5, 'index': 0.8, 'middle': 0.8, 'ring': 0.8, 'pinky': 0.8}, 'rot': [0, 0, 0], 'desc': 'All fingers form O'},
    'p': {'name': 'P', 'fingers': {'thumb': 0.5, 'index': 0, 'middle': 0, 'ring': 1.5, 'pinky': 1.5}, 'rot': [0, 0, -45], 'desc': 'K pointing down'},
    'q': {'name': 'Q', 'fingers': {'thumb': 0, 'index': 0, 'middle': 1.5, 'ring': 1.5, 'pinky': 1.5}, 'rot': [0, 0, -90], 'desc': 'G pointing down'},
    'r': {'name': 'R', 'fingers': {'thumb': 1.2, 'index': 0, 'middle': 0, 'ring': 1.5, 'pinky': 1.5}, 'rot': [0, 0, 0], 'desc': 'Index and middle crossed'},
    's': {'name': 'S', 'fingers': {'thumb': 0.5, 'index': 1.5, 'middle': 1.5, 'ring': 1.5, 'pinky': 1.5}, 'rot': [0, 0, 0], 'desc': 'Fist with thumb across'},
    't': {'name': 'T', 'fingers': {'thumb': 0.8, 'index': 1.5, 'middle': 1.5, 'ring': 1.5, 'pinky': 1.5}, 'rot': [0, 0, 0], 'desc': 'Thumb between index and middle'},
    'u': {'name': 'U', 'fingers': {'thumb': 1.2, 'index': 0, 'middle': 0, 'ring': 1.5, 'pinky': 1.5}, 'rot': [0, 0, 0], 'desc': 'Two fingers together'},
    'v': {'name': 'V', 'fingers': {'thumb': 1.2, 'index': 0, 'middle': 0, 'ring': 1.5, 'pinky': 1.5}, 'rot': [0, 0, 0], 'desc': 'Victory/Peace sign'},
    'w': {'name': 'W', 'fingers': {'thumb': 1.2, 'index': 0, 'middle': 0, 'ring': 0, 'pinky': 1.5}, 'rot': [0, 0, 0], 'desc': 'Three fingers up'},
    'x': {'name': 'X', 'fingers': {'thumb': 1.5, 'index': 0.8, 'middle': 1.5, 'ring': 1.5, 'pinky': 1.5}, 'rot': [0, 0, 0], 'desc': 'Index bent like hook'},
    'y': {'name': 'Y', 'fingers': {'thumb': 0, 'index': 1.5, 'middle': 1.5, 'ring': 1.5, 'pinky': 0}, 'rot': [0, 0, 0], 'desc': 'Thumb and pinky out (hang loose)'},
    'z': {'name': 'Z', 'fingers': {'thumb': 1.5, 'index': 0, 'middle': 1.5, 'ring': 1.5, 'pinky': 1.5}, 'rot': [0, 0, 0], 'desc': 'Index draws Z'}
}

print(f"Loaded {len(ISL_LIBRARY)} gestures")

# ==================== Flask Setup ====================
app = Flask(__name__)
app.config['SECRET_KEY'] = 'isl-fingerspelling-2024'
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode="eventlet",
    ping_timeout=60,
    ping_interval=25
)

print("=" * 60)
print("ISL SPEECH TO SIGN AVATAR SYSTEM")
print("RUN MODE:", RUN_MODE)
print("=" * 60)

# Speech recognizer
recognizer = sr.Recognizer()
recognizer.energy_threshold = 3000
recognizer.dynamic_energy_threshold = True
recognizer.pause_threshold = 0.8
recognizer.non_speaking_duration = 0.5

# System statistics
stats = {
    "total_conversions": 0,
    "total_gestures": 0,
    "speech_recognitions": 0,
    "errors": 0,
    "start_time": datetime.now()
}

# ==================== Text to Letter Sequence ====================
def text_to_letter_sequence(text):
    """Convert text to letter-by-letter fingerspelling sequence"""
    clean_text = text.lower().replace(",", "").replace(".", "").replace("!", "").replace("?", "").replace("-", " ")
    chars = list(clean_text)

    sequence = []

    for i, char in enumerate(chars):
        if char == ' ':
            sequence.append({
                'name': 'SPACE',
                'desc': 'Word space - pause between words',
                'duration': 0.6,
                'isSpace': True,
                'keyframes': [
                    {
                        'time': 0,
                        'hand': 'right',
                        'pos': [0, 0, 0],
                        'rot': [0, 0, 0],
                        'fingers': {'thumb': 0, 'index': 0, 'middle': 0, 'ring': 0, 'pinky': 0}
                    },
                    {
                        'time': 0.6,
                        'hand': 'right',
                        'pos': [0.2, 0, 0],
                        'rot': [0, 0, 0],
                        'fingers': {'thumb': 0, 'index': 0, 'middle': 0, 'ring': 0, 'pinky': 0}
                    }
                ]
            })
            continue

        if char in ISL_LIBRARY:
            letter_data = ISL_LIBRARY[char]
            rot_rad = [r * 3.14159 / 180 for r in letter_data['rot']]

            sequence.append({
                'name': letter_data['name'],
                'letter': char,
                'desc': letter_data['desc'],
                'duration': 0.7,
                'keyframes': [
                    {
                        'time': 0,
                        'hand': 'right',
                        'pos': [0, 0, 0],
                        'rot': rot_rad,
                        'fingers': letter_data['fingers']
                    },
                    {
                        'time': 0.3,
                        'hand': 'right',
                        'pos': [0, 0.15, 0],
                        'rot': rot_rad,
                        'fingers': letter_data['fingers']
                    },
                    {
                        'time': 0.7,
                        'hand': 'right',
                        'pos': [0, 0.15, 0],
                        'rot': rot_rad,
                        'fingers': letter_data['fingers']
                    }
                ]
            })

    return sequence

# ==================== Routes ====================
@app.route('/')
def index():
    """Serve main interface"""
    try:
        return send_from_directory(os.path.join(os.getcwd(), 'templates'), 'avatar_interface.html')
    except Exception:
        return "Avatar UI Loaded. Frontend missing."

@app.route('/health')
def health():
    """Health check endpoint for cloud platforms"""
    return jsonify({"status": "running"})

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files for cloud platforms"""
    return send_from_directory('static', filename)

@app.route('/api/text-to-signs', methods=['POST'])
def text_to_signs_api():
    """Convert text to letter sequence"""
    try:
        data = request.json
        text = data.get('text', '')

        if not text:
            return jsonify({"success": False, "error": "No text provided"}), 400

        print(f'📝 Converting text: "{text}"')

        sequence = text_to_letter_sequence(text)

        stats["total_conversions"] += 1
        stats["total_gestures"] += len(sequence)

        return jsonify({
            "success": True,
            "text": text,
            "sequence": sequence,
            "stats": get_stats()
        })

    except Exception as e:
        stats["errors"] += 1
        print(f"❌ Error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/stats')
def get_stats_api():
    """Get system statistics"""
    return jsonify(get_stats())

def get_stats():
    """Calculate current statistics"""
    uptime = (datetime.now() - stats["start_time"]).total_seconds()

    return {
        "total_conversions": stats["total_conversions"],
        "total_gestures": stats["total_gestures"],
        "speech_recognitions": stats["speech_recognitions"],
        "errors": stats["errors"],
        "uptime_seconds": int(uptime),
        "alphabet_letters": len(ISL_LIBRARY)
    }

# ==================== Socket.IO Events ====================
@socketio.on('connect')
def handle_connect():
    """Client connected"""
    print('✅ Client connected')
    emit('status', {'message': 'Connected', 'stats': get_stats()})

@socketio.on('disconnect')
def handle_disconnect():
    """Client disconnected"""
    print('👋 Client disconnected')

@socketio.on('record_speech')
def handle_record_speech():
    """Record speech and convert to fingerspelling"""

    # Disable microphone in cloud environment
    if RUN_MODE == "CLOUD":
        emit("error", {"message": "Microphone recording not supported in cloud deployment"})
        return

    def record_and_convert():
        try:
            print('🎤 Starting speech recognition...')

            with sr.Microphone() as source:
                print('🎤 Listening... Speak now!')
                socketio.emit('recording_status', {'status': 'listening'})

                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)

                print('🔄 Processing speech...')
                socketio.emit('recording_status', {'status': 'processing'})

                try:
                    text = recognizer.recognize_google(audio, language='en-IN')
                    print(f'✅ Recognized: "{text}"')

                    stats["speech_recognitions"] += 1
                    stats["total_conversions"] += 1

                    sequence = text_to_letter_sequence(text)
                    stats["total_gestures"] += len(sequence)

                    socketio.emit('play_signs', {
                        'text': text,
                        'sequence': sequence,
                        'stats': get_stats()
                    })

                except sr.UnknownValueError:
                    print('❌ Could not understand speech')
                    stats["errors"] += 1
                    socketio.emit('error', {
                        'message': 'Could not understand speech. Please speak clearly and try again.'
                    })

                except sr.RequestError as e:
                    print(f'❌ Speech recognition API error: {e}')
                    stats["errors"] += 1
                    socketio.emit('error', {
                        'message': 'Speech recognition service error. Please check your internet connection.'
                    })

        except sr.WaitTimeoutError:
            print('⏱️ No speech detected (timeout)')
            stats["errors"] += 1
            socketio.emit('error', {
                'message': 'No speech detected. Please try again.'
            })

        except Exception as e:
            print(f'❌ Recording error: {e}')
            stats["errors"] += 1
            socketio.emit('error', {
                'message': f'Recording error: {str(e)}'
            })

    threading.Thread(target=record_and_convert, daemon=True).start()

# ==================== Main ====================
if __name__ == '__main__':

    print("\n" + "="*60)
    print("ISL SPEECH TO SIGN AVATAR SYSTEM STARTING")
    print("RUN MODE:", RUN_MODE)
    print("="*60)

    os.makedirs('templates', exist_ok=True)

    PORT = int(os.environ.get("PORT", 5001))

    print("Server starting on port:", PORT)
    print("Health check endpoint: /health")

    socketio.run(
        app,
        host="0.0.0.0",
        port=PORT,
        debug=False,
        use_reloader=False
    )