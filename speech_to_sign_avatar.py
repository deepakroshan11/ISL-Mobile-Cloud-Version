"""
ISL Fingerspelling System - Backend Server
Full A-Z Letter-by-Letter Speech-to-Sign Conversion

Install dependencies:
pip install SpeechRecognition flask flask-socketio pyaudio

Usage:
python speech_to_sign_avatar_fixed.py

Then open: http://localhost:5001
"""

import os
import json
import time
import threading
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_socketio import SocketIO, emit
import speech_recognition as sr

# ==================== ISL ALPHABET LIBRARY (A-Z) ====================
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

# ==================== Flask Setup ====================
app = Flask(__name__)
app.config['SECRET_KEY'] = 'isl-fingerspelling-2024'
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

# ==================== Text to Letter Sequence ====================
def text_to_letter_sequence(text):
    """Convert text to letter-by-letter fingerspelling sequence"""
    # Clean text - remove punctuation, keep only letters and spaces
    clean_text = text.lower().replace(",", "").replace(".", "").replace("!", "").replace("?", "").replace("-", " ")
    chars = list(clean_text)
    
    sequence = []
    
    for i, char in enumerate(chars):
        if char == ' ':
            # Add word space
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
        
        # Check if letter exists in library
        if char in ISL_LIBRARY:
            letter_data = ISL_LIBRARY[char]
            
            # Convert rotation from degrees to radians for frontend
            rot_rad = [r * 3.14159 / 180 for r in letter_data['rot']]
            
            sequence.append({
                'name': letter_data['name'],
                'letter': char,
                'desc': letter_data['desc'],
                'duration': 1.0,
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
                        'time': 1.0,
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
    return send_from_directory('templates', 'avatar_interface.html')

@app.route('/api/text-to-signs', methods=['POST'])
def text_to_signs_api():
    """Convert text to letter sequence"""
    try:
        data = request.json
        text = data.get('text', '')
        
        if not text:
            return jsonify({"success": False, "error": "No text provided"}), 400
        
        print(f'📝 Converting text: "{text}"')
        
        # Convert to letter sequence
        sequence = text_to_letter_sequence(text)
        
        # Update stats
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
    def record_and_convert():
        try:
            print('🎤 Starting speech recognition...')
            
            with sr.Microphone() as source:
                print('🎤 Listening... Speak now!')
                socketio.emit('recording_status', {'status': 'listening'})
                
                # Adjust for ambient noise
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                
                # Listen for speech (5 second timeout, 10 second phrase limit)
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
                
                print('🔄 Processing speech...')
                socketio.emit('recording_status', {'status': 'processing'})
                
                try:
                    # Try English recognition
                    text = recognizer.recognize_google(audio, language='en-IN')
                    print(f'✅ Recognized: "{text}"')
                    
                    # Update stats
                    stats["speech_recognitions"] += 1
                    stats["total_conversions"] += 1
                    
                    # Convert to letter sequence
                    sequence = text_to_letter_sequence(text)
                    stats["total_gestures"] += len(sequence)
                    
                    # Send to frontend
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
    
    # Run in separate thread to avoid blocking
    threading.Thread(target=record_and_convert, daemon=True).start()

# ==================== Main ====================
if __name__ == '__main__':
    print("\n" + "="*70)
    print("🤟 ISL FINGERSPELLING SYSTEM - A-Z LETTER-BY-LETTER")
    print("="*70)
    print()
    print("✨ Features:")
    print(f"   • {len(ISL_LIBRARY)} Letter Signs (A-Z)")
    print("   • Speech-to-Text Recognition")
    print("   • Automatic Fingerspelling")
    print("   • 3D Hand Animation")
    print("   • Real-time Conversion")
    print()
    print("🌐 Interface: http://localhost:5001")
    print("📊 Stats API: http://localhost:5001/api/stats")
    print()
    print("🎤 Usage:")
    print("   1. Type text in the box and click 'Convert to Signs'")
    print("   2. OR click 'Record Speech' and speak")
    print("   3. Watch the 3D hand fingerspell your text!")
    print()
    print("="*70)
    print()
    
    # Create templates folder if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    # Run server
    socketio.run(app, host='0.0.0.0', port=5001, debug=False, allow_unsafe_werkzeug=True)