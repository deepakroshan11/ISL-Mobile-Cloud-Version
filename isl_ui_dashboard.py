# isl_ui_dashboard.py — CLOUD PRODUCTION VERSION
#
# ════════════════════════════════════════════════════════════════════════════
#  CLOUD FIXES APPLIED (4 changes only — everything else identical):
#
#  FIX 1 — Browser camera: HTML now captures + emits browser_frame via socket
#  FIX 2 — Nav links: localhost:5000/5001 → dynamic window.location URLs
#  FIX 3 — _cloud_frame scope: shared_state stored as module-level variable
#  FIX 4 — /avatar route added
#
#  LOCAL CAMERA FIX:
#  FIX 5 — generate_frames() now waits for frame_seq > 0 before streaming.
#           Without this, the MJPEG client receives raw uninitialized
#           SharedMemory bytes (all zeros or garbage) before the detection
#           engine writes the first valid frame — causing static/noise on
#           the built-in laptop webcam display.
#           Also removed unreachable `return` after infinite fallback loop.
#
#  ALL original features preserved:
#  ✅ MJPEG /video_feed from SharedMemory (zero-copy)
#  ✅ Socket.IO state_update events (emotion, word, suggestions, stats)
#  ✅ Speak, Backspace, Reset commands
#  ✅ Emotion bars, Chart.js timeline, word suggestions
#  ✅ async_mode='threading', transports=['polling']
# ════════════════════════════════════════════════════════════════════════════

import time
import threading
import numpy as np
import cv2
import multiprocessing.shared_memory

from queue import Queue, Empty
from flask import Flask, render_template_string, Response, jsonify, request
from flask_socketio import SocketIO

# ── App & SocketIO ────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["SECRET_KEY"] = "isl_secret_2024"

socketio = SocketIO(
    app,
    async_mode="threading",
    cors_allowed_origins="*",
    ping_timeout=60,
    ping_interval=25,
    logger=False,
    engineio_logger=False,
)

# ── Module-level state ────────────────────────────────────────────────────────
_local_state_q: Queue = Queue(100)
FRAME_W, FRAME_H      = 640, 480
_shm_name   = None
_frame_seq  = None

# FIX 3: shared_state stored at module level so on_browser_frame() can
#         set _cloud_frame without being inside dashboard_process() scope.
_shared_state = None

# ── HTML ──────────────────────────────────────────────────────────────────────
HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0, viewport-fit=cover"/>
<title>ISL Gesture Recognition — AI Sign Language</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin/>
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet"/>
<script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.7.2/socket.io.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
:root {
  --bg:    #0d1117;
  --surf:  #131922;
  --card:  #1a2233;
  --bdr:   rgba(212,168,83,.13);
  --bdrm:  rgba(212,168,83,.30);
  --bdrh:  rgba(212,168,83,.58);
  --gold:  #d4a853;
  --goldb: #f0c76a;
  --goldd: #a07c3a;
  --sage:  #7eb89a;
  --saged: #4e7d66;
  --txt:   #f0ede6;
  --txtm:  #a8a096;
  --txtl:  #5a5650;
  --red:   #c0614a;
  --r8:    8px;
  --r14:   14px;
  --fD:    'Playfair Display', Georgia, serif;
  --fB:    'DM Sans', system-ui, sans-serif;
  --fM:    'DM Mono', monospace;
}
*, *::before, *::after { margin:0; padding:0; box-sizing:border-box; }
html, body { height:100%; overflow:hidden; }
body {
  font-family: var(--fB);
  background: var(--bg);
  color: var(--txt);
  -webkit-font-smoothing: antialiased;
  display: flex;
  flex-direction: column;
}

#app { display:flex; flex:1; overflow:hidden; height:100%; }

/* ── Camera pane ── */
#cam-pane {
  flex: 1;
  display: flex;
  flex-direction: column;
  min-width: 0;
  background: var(--bg);
  position: relative;
  overflow: hidden;
}
.topbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 10px 16px 0;
  flex-shrink: 0;
  position: relative;
  z-index: 5;
}
.brand { display:flex; align-items:baseline; gap:8px; }
.brand-t { font-family:var(--fD); font-size:1.1rem; font-weight:700; color:var(--gold); }
.brand-s { font-size:.65rem; color:var(--txtl); text-transform:uppercase; letter-spacing:.07em; }
.live-badge {
  font-size:.65rem; font-family:var(--fM); color:var(--sage);
  background:rgba(126,184,154,.10); border:1px solid rgba(126,184,154,.25);
  padding:3px 9px; border-radius:20px; transition:all .3s;
}
.live-badge.rec { color:var(--gold); background:rgba(212,168,83,.1); border-color:var(--bdrm); animation:pd .9s infinite; }
.live-badge.err { color:var(--red); background:rgba(192,97,74,.1); border-color:rgba(192,97,74,.3); }
@keyframes pd { 0%,100%{opacity:1} 50%{opacity:.55} }

#camera-feed {
  flex: 1; width:100%; display:block; object-fit:cover; background:#000;
}
.cam-overlay {
  position:absolute; bottom:0; left:0; right:0;
  padding:12px 18px 14px;
  background: linear-gradient(to top, rgba(13,17,23,1) 55%, transparent);
  display:flex; align-items:flex-end; justify-content:space-between;
  gap:10px; pointer-events:none; z-index:5;
}
.co-letter {
  font-family:var(--fD); font-size:clamp(2rem,5vw,3.2rem);
  font-weight:700; color:var(--goldb); line-height:1;
}
.co-word {
  font-size:.85rem; font-weight:600; color:var(--gold);
  font-family:var(--fM); letter-spacing:.05em;
}
.co-desc { font-size:.72rem; color:var(--txtm); text-align:right; max-width:200px; line-height:1.45; }
.co-progress { position:absolute; bottom:0; left:0; right:0; height:3px; background:rgba(212,168,83,.08); }
.co-bar {
  height:100%; width:0%;
  background:linear-gradient(90deg, var(--goldd), var(--goldb));
  transition:width .3s ease; position:relative; overflow:hidden;
}
.co-bar::after {
  content:''; position:absolute; inset:0;
  background:linear-gradient(90deg,transparent,rgba(255,255,255,.3),transparent);
  animation:shimmer 1.8s infinite;
}
@keyframes shimmer { from{transform:translateX(-100%)} to{transform:translateX(200%)} }
.cam-placeholder {
  position:absolute; inset:0;
  display:flex; flex-direction:column;
  align-items:center; justify-content:center;
  gap:10px; color:var(--txtl); font-size:.8rem; z-index:2; pointer-events:none;
}
.cam-placeholder .ph-icon { font-size:2.8rem; opacity:.25; }

/* ── Control pane ── */
#ctrl-pane {
  width:340px; flex-shrink:0; display:flex; flex-direction:column;
  background:var(--surf); border-left:1px solid var(--bdr);
  overflow-y:auto; overflow-x:hidden;
}
#ctrl-pane::-webkit-scrollbar { width:4px; }
#ctrl-pane::-webkit-scrollbar-thumb { background:var(--bdrm); border-radius:4px; }

@media (max-width:720px) {
  html,body { overflow:hidden; }
  #app { flex-direction:column; height:100%; }
  #cam-pane { flex:0 0 46vh; min-height:260px; max-height:46vh; width:100%; }
  #ctrl-pane { width:100%; flex:1; border-left:none; border-top:1px solid var(--bdr); overflow-y:auto; }
  .topbar { padding:8px 12px 0; }
  .brand-t { font-size:.95rem; }
}

.cpanel { padding:14px 16px; display:flex; flex-direction:column; gap:12px; }

.sec-label {
  font-size:.65rem; font-weight:600; text-transform:uppercase;
  letter-spacing:.08em; color:var(--txtl);
  display:flex; align-items:center; gap:6px; font-family:var(--fM);
}

/* ── Status pill ── */
.status {
  display:flex; align-items:center; gap:7px; padding:7px 13px;
  border-radius:30px; font-size:.73rem; font-weight:500;
  border:1px solid var(--bdr); background:var(--card); color:var(--txtm);
  transition:all .3s; min-height:34px; font-family:var(--fM);
}
.sdot { width:7px; height:7px; border-radius:50%; background:var(--txtl); flex-shrink:0; transition:all .3s; }
.status.act .sdot { background:var(--sage); box-shadow:0 0 8px var(--sage); animation:pd 1.4s infinite; }
.status.act { border-color:rgba(126,184,154,.3); color:var(--sage); }
.status.err .sdot { background:var(--red); }
.status.err { border-color:rgba(192,97,74,.3); color:var(--red); }

/* ── Text display ── */
.text-display {
  background:var(--card); border:1px solid var(--bdr);
  border-radius:var(--r14); padding:10px 13px; min-height:52px;
}
.cur-word {
  font-size:1.15rem; font-weight:700; color:var(--goldb);
  font-family:var(--fD); line-height:1.2; margin-bottom:4px;
}
.cur-sentence { font-size:.75rem; color:var(--txtm); line-height:1.5; font-family:var(--fM); }

/* ── Buttons ── */
.btn-row { display:grid; grid-template-columns:repeat(3,1fr); gap:7px; }
.btn {
  display:flex; align-items:center; justify-content:center; gap:5px;
  padding:9px 6px; border:1px solid transparent; border-radius:var(--r14);
  font-family:var(--fB); font-size:.76rem; font-weight:600;
  cursor:pointer; transition:all .22s; white-space:nowrap; overflow:hidden;
}
.btn:active { transform:scale(.96); }
.btn:disabled { opacity:.4; cursor:not-allowed; }
.btn-speak {
  background:linear-gradient(135deg,#4e7d66,#7eb89a);
  color:#0d1117; border-color:rgba(126,184,154,.35);
}
.btn-speak:hover { box-shadow:0 4px 16px rgba(126,184,154,.3); }
.btn-back { background:transparent; color:var(--txtm); border-color:var(--bdr); }
.btn-back:hover { border-color:var(--bdrm); color:var(--txt); }
.btn-reset { background:transparent; color:var(--red); border-color:rgba(192,97,74,.3); }
.btn-reset:hover { background:rgba(192,97,74,.08); border-color:rgba(192,97,74,.55); }

.divider { height:1px; background:var(--bdr); margin:0 -16px; }

/* ── Emotion ── */
.emotion-header { display:flex; justify-content:space-between; align-items:center; margin-bottom:8px; }
.emotion-big { font-family:var(--fD); font-size:1.1rem; font-weight:700; color:var(--goldb); }
.emotion-icon { font-size:1.5rem; animation:emojiPop 2.2s ease infinite; }
@keyframes emojiPop { 0%,100%{transform:scale(1)} 50%{transform:scale(1.12)} }

.emo-bars { display:flex; flex-direction:column; gap:5px; }
.emo-bar { display:flex; align-items:center; gap:8px; }
.ebo-label { width:78px; font-size:.72rem; color:var(--txtm); flex-shrink:0; font-family:var(--fB); }
.ebo-track {
  flex:1; height:20px; background:rgba(26,34,51,.8);
  border-radius:10px; overflow:hidden; border:1px solid var(--bdr);
}
.ebo-fill {
  height:100%; transition:width .5s cubic-bezier(.4,0,.2,1);
  display:flex; align-items:center; justify-content:flex-end;
  padding-right:6px; font-size:.62rem; font-weight:700;
  font-family:var(--fM); color:rgba(0,0,0,.75); min-width:28px;
}

/* ── Suggestions ── */
.sug-detected {
  font-size:.65rem; font-family:var(--fM); color:var(--gold);
  background:rgba(212,168,83,.07); border:1px solid var(--bdr);
  border-radius:var(--r8); padding:4px 10px; margin-bottom:5px; letter-spacing:.04em;
}
.sug-list { display:flex; flex-direction:column; gap:4px; }
.sug-item {
  display:flex; align-items:center; gap:9px; padding:7px 11px;
  border-radius:var(--r8); background:var(--card); border:1px solid var(--bdr);
  cursor:pointer; transition:all .2s; font-size:.82rem; color:var(--txt);
}
.sug-item:hover {
  background:rgba(212,168,83,.08); border-color:var(--bdrm);
  transform:translateX(3px); color:var(--goldb);
}
.sug-n {
  font-family:var(--fM); font-size:.65rem; font-weight:700; color:var(--gold);
  background:rgba(212,168,83,.12); border:1px solid rgba(212,168,83,.2);
  width:20px; height:20px; display:flex; align-items:center; justify-content:center;
  border-radius:5px; flex-shrink:0;
}
.sug-empty { text-align:center; padding:16px 0; color:var(--txtl); font-size:.75rem; line-height:1.7; }

/* ── Chart ── */
#emotionChart { height:160px !important; }

/* ── Stats ── */
.stat-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:6px; }
.stat-tile {
  background:var(--card); border:1px solid var(--bdr);
  border-radius:var(--r8); padding:9px 10px; text-align:center; transition:border-color .25s;
}
.stat-tile:hover { border-color:var(--bdrm); }
.sl { font-size:.55rem; text-transform:uppercase; letter-spacing:.07em; color:var(--txtl); font-weight:600; margin-bottom:3px; font-family:var(--fM); }
.sv { font-family:var(--fD); font-size:1.3rem; font-weight:700; color:var(--gold); line-height:1; }

/* ── About ── */
.about-txt { font-size:.75rem; color:var(--txtm); line-height:1.75; }
.about-txt p { margin-bottom:7px; }
.alist { list-style:none; margin-bottom:7px; }
.alist li { padding-left:13px; position:relative; font-size:.73rem; padding-bottom:2px; }
.alist li::before { content:'›'; position:absolute; left:0; color:var(--gold); font-weight:700; }
.acredit { margin-top:10px; padding-top:10px; border-top:1px solid var(--bdr); }
.aname { font-family:var(--fD); color:var(--gold); font-weight:600; font-size:.88rem; }
.arole { font-size:.65rem; color:var(--txtl); margin:2px 0 3px; font-family:var(--fM); }
.aemail { color:var(--sage); font-size:.66px; font-family:var(--fM); }

/* ── Nav ── */
.nav-row { display:flex; gap:7px; padding:0 16px 20px; }
.nav-a {
  flex:1; display:flex; align-items:center; justify-content:center; gap:6px;
  padding:12px 8px; border-radius:var(--r14); border:1px solid var(--bdr);
  background:var(--card); color:var(--txtm); text-decoration:none;
  font-size:.76rem; font-weight:600; transition:all .25s;
}
.nav-a:hover { border-color:var(--bdrm); color:var(--txt); }
.nav-a.on { background:rgba(212,168,83,.10); border-color:var(--bdrh); color:var(--gold); }
</style>
</head>
<body>

<div id="app">

  <!-- ── Camera pane ── -->
  <div id="cam-pane">
    <div class="topbar">
      <div class="brand">
        <span class="brand-t">ISL Gesture Recognition</span>
        <span class="brand-s">Gesture · Emotion · AI</span>
      </div>
      <span class="live-badge" id="live-badge">● Ready</span>
    </div>

    <!-- MJPEG feed — displays processed frame from server SharedMemory -->
    <img id="camera-feed"
         src="/video_feed"
         alt="Camera Feed"
         style="display:none;flex:1;width:100%;object-fit:cover;"
         onload="onVidLoad()"
         onerror="document.getElementById('cam-ph').style.display='flex'"/>

    <div class="cam-placeholder" id="cam-ph">
      <span class="ph-icon">📷</span>
      <span style="font-family:var(--fM);font-size:.75rem">Connecting to camera…</span>
    </div>

    <div class="cam-overlay">
      <div>
        <div style="display:flex;align-items:baseline;gap:8px">
          <span id="co-letter" class="co-letter">🤟</span>
          <span id="co-word"   class="co-word"></span>
        </div>
      </div>
      <div id="co-desc" class="co-desc">Point camera at your hand</div>
      <div class="co-progress"><div class="co-bar" id="co-bar"></div></div>
    </div>
  </div>

  <!-- ── Control pane ── -->
  <div id="ctrl-pane">
    <div class="cpanel">

      <!-- Status -->
      <div class="status" id="status">
        <div class="sdot"></div>
        <span id="stxt">Ready for detection</span>
      </div>

      <!-- Current word -->
      <div class="text-display">
        <div class="cur-word"     id="cur-word">—</div>
        <div class="cur-sentence" id="cur-sentence"></div>
      </div>

      <!-- Buttons -->
      <div class="btn-row">
        <button class="btn btn-speak" onclick="sendCmd('speak')">🔊 Speak</button>
        <button class="btn btn-back"  onclick="sendCmd('backspace')">⌫ Back</button>
        <button class="btn btn-reset" onclick="sendCmd('reset')">🔄 Reset</button>
      </div>

      <div class="divider"></div>

      <!-- Emotion -->
      <div class="sec-label">😊 Emotion Detection</div>
      <div class="emotion-header">
        <span class="emotion-big" id="emo-label">Neutral</span>
        <span class="emotion-icon" id="emo-icon">😐</span>
      </div>
      <div class="emo-bars">
        <div class="emo-bar">
          <div class="ebo-label">😊 Happy</div>
          <div class="ebo-track"><div class="ebo-fill" id="bar-happy" style="width:0%;background:linear-gradient(90deg,#3a8a5c,#7eb89a)">0%</div></div>
        </div>
        <div class="emo-bar">
          <div class="ebo-label">😢 Sad</div>
          <div class="ebo-track"><div class="ebo-fill" id="bar-sad" style="width:0%;background:linear-gradient(90deg,#3a5a8a,#6090c8)">0%</div></div>
        </div>
        <div class="emo-bar">
          <div class="ebo-label">😠 Angry</div>
          <div class="ebo-track"><div class="ebo-fill" id="bar-angry" style="width:0%;background:linear-gradient(90deg,#8a3a3a,#c06060)">0%</div></div>
        </div>
        <div class="emo-bar">
          <div class="ebo-label">😲 Surprise</div>
          <div class="ebo-track"><div class="ebo-fill" id="bar-surprise" style="width:0%;background:linear-gradient(90deg,#8a6a1a,#d4a853)">0%</div></div>
        </div>
        <div class="emo-bar">
          <div class="ebo-label">😐 Neutral</div>
          <div class="ebo-track"><div class="ebo-fill" id="bar-neutral" style="width:100%;background:linear-gradient(90deg,#3a4455,#5a6878)">100%</div></div>
        </div>
      </div>

      <div class="divider"></div>

      <!-- Word Suggestions -->
      <div class="sec-label">💡 Word Suggestions</div>
      <div id="sug-area">
        <div class="sug-empty">Detected letters will appear<br>here with word suggestions</div>
      </div>

      <div class="divider"></div>

      <!-- Emotion Timeline Chart -->
      <div class="sec-label">📈 Emotion Timeline</div>
      <canvas id="emotionChart"></canvas>

      <div class="divider"></div>

      <!-- Statistics -->
      <div class="sec-label">📊 Statistics</div>
      <div class="stat-grid">
        <div class="stat-tile"><div class="sl">FPS</div><div class="sv" id="s-fps">0</div></div>
        <div class="stat-tile"><div class="sl">Letters</div><div class="sv" id="s-letters">0</div></div>
        <div class="stat-tile"><div class="sl">Words</div><div class="sv" id="s-words">0</div></div>
        <div class="stat-tile"><div class="sl">Sentences</div><div class="sv" id="s-sentences">0</div></div>
        <div class="stat-tile"><div class="sl">Emotions</div><div class="sv" id="s-emotions">0</div></div>
        <div class="stat-tile"><div class="sl">Conf.</div><div class="sv" id="s-conf">—</div></div>
      </div>

      <div class="divider"></div>

      <!-- About -->
      <div class="sec-label">ℹ️ About</div>
      <div class="about-txt">
        <p>AI Powered ISL Detection is an assistive AI system designed to improve communication between hearing and deaf individuals.</p>
        <p>This module focuses on real-time gesture and facial emotion recognition using computer vision.</p>
        <p>Useful for:</p>
        <ul class="alist">
          <li>Deaf individuals</li>
          <li>Inclusive classrooms</li>
          <li>Assistive communication systems</li>
          <li>Accessibility and AI research</li>
        </ul>
        <div class="acredit">
          <div class="aname">Deepak Roshan</div>
          <div class="arole">AI Engineer</div>
          <div class="aemail">deepakroshan380@gmail.com</div>
        </div>
      </div>

    </div><!-- cpanel -->

    <!-- FIX 2: Nav links now built dynamically from window.location -->
    <div class="nav-row">
      <a href="/" class="nav-a on" id="nav-gesture">😊 Gesture Recognition</a>
      <a href="#"  class="nav-a"   id="nav-avatar">🤟 Speech-to-Sign</a>
    </div>
  </div><!-- ctrl-pane -->

</div><!-- app -->

<script>
document.addEventListener('DOMContentLoaded', function () {

  // FIX 2: Build nav URLs dynamically so they work on any host/domain
  (function() {
    const proto = location.protocol;
    const host  = location.hostname;
    const port  = location.port;
    if (port && port !== '80' && port !== '443') {
      document.getElementById('nav-gesture').href = proto + '//' + host + ':5000/';
      document.getElementById('nav-avatar').href  = proto + '//' + host + ':5001/';
    } else {
      document.getElementById('nav-gesture').href = proto + '//' + host + '/';
      document.getElementById('nav-avatar').href  = proto + '//' + host + '/avatar/';
    }
  })();

  const EMO_ICONS = { happy:'😊', sad:'😢', angry:'😠', surprise:'😲', neutral:'😐' };
  const EMO_KEYS  = ['happy','sad','angry','surprise','neutral'];

  /* ── Chart.js Emotion Timeline ─────────────────────────── */
  const chartCtx = document.getElementById('emotionChart').getContext('2d');
  const chart = new Chart(chartCtx, {
    type: 'line',
    data: {
      labels: [],
      datasets: [
        { label:'Happy',    data:[], borderColor:'#7eb89a', borderWidth:1.5, tension:.4, fill:false, pointRadius:0 },
        { label:'Sad',      data:[], borderColor:'#6090c8', borderWidth:1.5, tension:.4, fill:false, pointRadius:0 },
        { label:'Angry',    data:[], borderColor:'#c06060', borderWidth:1.5, tension:.4, fill:false, pointRadius:0 },
        { label:'Surprise', data:[], borderColor:'#d4a853', borderWidth:1.5, tension:.4, fill:false, pointRadius:0 },
        { label:'Neutral',  data:[], borderColor:'#5a6878', borderWidth:1.5, tension:.4, fill:false, pointRadius:0 },
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color:'#5a5650', font:{ size:9, family:"'DM Mono'" }, boxWidth:10 } }
      },
      scales: {
        y: {
          beginAtZero:true, max:1,
          ticks: { color:'#5a5650', font:{ size:8 } },
          grid:  { color:'rgba(212,168,83,.05)' }
        },
        x: { ticks: { display:false }, grid: { color:'rgba(212,168,83,.03)' } }
      },
      animation: false
    }
  });

  /* ── Browser Camera (FIX 1) ─────────────────────────────── */
  let _camStream   = null;
  let _camInterval = null;
  const _capCanvas = document.createElement('canvas');
  const _capCtx    = _capCanvas.getContext('2d');

  async function startBrowserCamera() {
    if (_camStream) return;
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      console.warn('Camera API not available — HTTP or unsupported browser');
      return;
    }
    try {
      _camStream = await navigator.mediaDevices.getUserMedia({
        video: { width:{ ideal:640 }, height:{ ideal:480 }, facingMode:'user' },
        audio: false
      });
      const video = document.createElement('video');
      video.srcObject  = _camStream;
      video.autoplay   = true;
      video.playsInline = true;
      video.addEventListener('loadeddata', () => {
        _capCanvas.width  = 320;
        _capCanvas.height = 240;
        _camInterval = setInterval(() => {
          if (document.visibilityState === 'visible' && socket && socket.connected) {
            _capCtx.drawImage(video, 0, 0, 320, 240);
            const frame = _capCanvas.toDataURL('image/jpeg', 0.6);
            socket.emit('browser_frame', { frame: frame.split(',')[1] });
          }
        }, 65); // ~15 fps
      });
    } catch (err) {
      console.error('Camera access error:', err);
    }
  }

  function stopBrowserCamera() {
    if (_camStream) { _camStream.getTracks().forEach(t => t.stop()); _camStream = null; }
    if (_camInterval) { clearInterval(_camInterval); _camInterval = null; }
  }

  /* ── Socket.IO ───────────────────────────────────────────── */
  let socket;
  try {
    socket = io({ transports: ['polling'], reconnectionDelay: 1000 });

    socket.on('connect', () => {
      setStat('act', 'Connected — detecting…');
      document.getElementById('cam-ph').style.display = 'none';
      setBadge('● Live', 'rec');
      startBrowserCamera();
    });

    socket.on('disconnect', () => {
      setStat('err', 'Disconnected');
      document.getElementById('cam-ph').style.display = 'flex';
      setBadge('● Offline', 'err');
      stopBrowserCamera();
    });

    socket.on('state_update', updateUI);

  } catch (e) {
    console.warn('Socket.IO unavailable — demo mode');
  }

  /* ── Video load callback ──────────────────────────────── */
  window.onVidLoad = function () {
    document.getElementById('cam-ph').style.display    = 'none';
    document.getElementById('camera-feed').style.display = 'block';
    setBadge('● Live', 'rec');
  };

  /* ── Main UI update ───────────────────────────────────── */
  function updateUI(d) {
    if (d.overlay_letter) {
      document.getElementById('co-letter').textContent = d.overlay_letter.toUpperCase();
    }
    if (d.current_word !== undefined) {
      const w = d.current_word || '—';
      document.getElementById('co-word').textContent  = w;
      document.getElementById('cur-word').textContent = w;
    }
    if (d.sentence !== undefined) {
      document.getElementById('cur-sentence').textContent = d.sentence || '';
    }
    if (d.overlay_confidence !== undefined) {
      const pct = Math.round(d.overlay_confidence * 100);
      document.getElementById('co-desc').textContent = 'Confidence: ' + pct + '%';
      document.getElementById('s-conf').textContent  = pct + '%';
      document.getElementById('co-bar').style.width  = pct + '%';
    }
    if (d.hand_detected) {
      setStat('act', d.overlay_letter
        ? 'Detecting: ' + d.overlay_letter + '  (' + Math.round((d.overlay_confidence||0)*100) + '%)'
        : 'Hand in frame — signing…');
    } else {
      setStat('', d.current_word ? 'Word so far: ' + d.current_word : 'Ready for detection');
    }
    if (d.fps !== undefined) {
      document.getElementById('s-fps').textContent = Math.round(d.fps);
    }
    if (d.stats) {
      document.getElementById('s-letters').textContent   = d.stats.letters_detected  || 0;
      document.getElementById('s-words').textContent     = d.stats.words_formed       || 0;
      document.getElementById('s-sentences').textContent = d.stats.sentences_formed   || 0;
      document.getElementById('s-emotions').textContent  = d.stats.emotion_changes    || 0;
    }
    if (Array.isArray(d.suggestions)) {
      updateSuggestions(d.suggestions, d.current_word);
    }
    if (d.emotion_scores) {
      updateEmotion(d.emotion || 'neutral', d.emotion_scores);
    }
    if (d.emotion_timeline && d.emotion_timeline.length > 0) {
      updateChart(d.emotion_timeline);
    }
  }

  function updateEmotion(emotion, scores) {
    document.getElementById('emo-label').textContent = emotion.charAt(0).toUpperCase() + emotion.slice(1);
    document.getElementById('emo-icon').textContent  = EMO_ICONS[emotion] || '😐';
    EMO_KEYS.forEach(k => {
      const bar = document.getElementById('bar-' + k);
      if (!bar) return;
      const pct = Math.round(Math.max(0, Math.min(1, scores[k] || 0)) * 100);
      bar.style.width = pct + '%';
      bar.textContent = pct + '%';
    });
  }

  function updateSuggestions(suggestions, currentWord) {
    const area = document.getElementById('sug-area');
    if (!suggestions || !suggestions.length) {
      area.innerHTML = '<div class="sug-empty">Detected letters will appear<br>here with word suggestions</div>';
      return;
    }
    const detectedRow = currentWord
      ? '<div class="sug-detected">Detected: ' + currentWord.toUpperCase() + '</div>' : '';
    area.innerHTML = detectedRow + '<div class="sug-list">' +
      suggestions.map((w, i) =>
        '<div class="sug-item" onclick="window.acceptSug(\'' + w + '\')">' +
        '<span class="sug-n">' + (i+1) + '</span>' + w + '</div>'
      ).join('') + '</div>';
  }

  function updateChart(timeline) {
    if (!timeline || !timeline.length) return;
    const collected = Object.fromEntries(EMO_KEYS.map(k => [k, []]));
    timeline.forEach(tp => {
      const s = tp.scores || {};
      EMO_KEYS.forEach(k => collected[k].push(s[k] || 0));
    });
    chart.data.labels = timeline.map((_, i) => i);
    chart.data.datasets.forEach((ds, i) => ds.data = collected[EMO_KEYS[i]]);
    chart.update('none');
  }

  function setStat(cls, msg) {
    document.getElementById('stxt').textContent = msg;
    document.getElementById('status').className = 'status' + (cls ? ' ' + cls : '');
  }
  function setBadge(msg, cls) {
    const b = document.getElementById('live-badge');
    b.textContent = msg;
    b.className   = 'live-badge' + (cls ? ' ' + cls : '');
  }

  window.sendCmd = function (action) {
    if (action === 'speak') {
      const sentence = document.getElementById('cur-sentence').textContent.trim();
      const word     = document.getElementById('cur-word').textContent.trim();
      const text     = sentence || (word !== '—' ? word : '');
      if (text && socket) socket.emit('command', { action:'speak', text });
    } else {
      if (socket) socket.emit('command', { action });
    }
  };

  window.acceptSug = function (word) {
    if (socket) socket.emit('command', { action:'accept_suggestion', word });
  };

  document.addEventListener('keydown', e => {
    if (e.ctrlKey && e.key === 's')         { e.preventDefault(); window.sendCmd('speak'); }
    if (e.ctrlKey && e.key === 'r')         { e.preventDefault(); window.sendCmd('reset'); }
    if (e.ctrlKey && e.key === 'Backspace') { e.preventDefault(); window.sendCmd('backspace'); }
    if (['1','2','3'].includes(e.key)) {
      const items = document.querySelectorAll('.sug-item');
      const item  = items[parseInt(e.key) - 1];
      if (item) item.click();
    }
  });

});
</script>
</body>
</html>
"""


# ── MJPEG generator reads directly from SharedMemory ──────────────────────────
def generate_frames():
    # FIX 5 (part A): If SharedMemory is not configured, stream a blank frame
    # continuously. Removed unreachable `return` after the infinite loop —
    # the original code had `return` after `while True` which is dead code and
    # caused a StopIteration on some Python/Flask versions.
    if _shm_name is None or _frame_seq is None:
        blank = np.zeros((FRAME_H, FRAME_W, 3), np.uint8)
        _, enc = cv2.imencode(".jpg", blank)
        chunk  = enc.tobytes()
        while True:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + chunk + b"\r\n")
            time.sleep(0.1)

    try:
        shm       = multiprocessing.shared_memory.SharedMemory(name=_shm_name)
        frame_buf = np.ndarray((FRAME_H, FRAME_W, 3), dtype=np.uint8, buffer=shm.buf)
    except Exception as e:
        print(f"⚠️  MJPEG generator: SharedMemory attach failed: {e}")
        return

    # FIX 5 (part B): Wait for the detection engine to write the first valid
    # frame before we start streaming. Without this wait, the MJPEG client
    # receives raw uninitialized SharedMemory bytes immediately, which renders
    # as horizontal static/noise stripes on the browser camera feed.
    # Timeout after 15 seconds (150 × 0.1s) to avoid hanging indefinitely.
    print("⏳ MJPEG: waiting for first valid frame from detection engine…")
    timeout_count = 0
    while _frame_seq.value == 0 and timeout_count < 150:
        time.sleep(0.1)
        timeout_count += 1

    if _frame_seq.value == 0:
        print("⚠️  MJPEG: timed out waiting for first frame — streaming anyway")
    else:
        print(f"✅ MJPEG: first frame ready (seq={_frame_seq.value}) — streaming started")

    last_seq = 0
    try:
        while True:
            if _frame_seq.value != last_seq:
                last_seq = _frame_seq.value
                frame = frame_buf.copy()
                _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                frame_bytes = buffer.tobytes()
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" +
                       frame_bytes + b"\r\n")
            else:
                time.sleep(0.005)
    finally:
        shm.close()


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/cmd", methods=["POST"])
def cmd_route():
    data = request.get_json(force=True, silent=True) or {}
    _local_state_q.put(data)
    return jsonify({"ok": True})


@app.route("/health")
def health():
    return jsonify({"ok": True, "ts": time.time()})


# FIX 4: /avatar route — redirects to the speech-to-sign service
@app.route("/avatar")
@app.route("/avatar/")
def avatar_redirect():
    from flask import redirect
    proto = request.headers.get('X-Forwarded-Proto', 'http')
    host  = request.host.split(':')[0]
    return redirect(proto + '://' + host + '/avatar/', code=302)


# ── Socket.IO events ──────────────────────────────────────────────────────────
@socketio.on("connect")
def on_connect():
    print("🔌 Browser connected (polling)")


@socketio.on("disconnect")
def on_disconnect():
    print("🔌 Browser disconnected")


@socketio.on("command")
def on_cmd(data):
    if data:
        _local_state_q.put(data)


# FIX 1 + FIX 3: browser_frame handler
@socketio.on("browser_frame")
def on_browser_frame(data):
    global _shared_state
    if _shared_state is not None and data and "frame" in data:
        _shared_state._cloud_frame = data["frame"]


# ── Dashboard process entry point ─────────────────────────────────────────────
def dashboard_process(shared_state, shm_name, frame_lock_val, frame_seq):
    global _shm_name, _frame_seq, _shared_state

    _shm_name     = shm_name
    _frame_seq    = frame_seq
    _shared_state = shared_state

    print("=" * 60)
    print("🌐 ISL Web Dashboard Starting...")
    print("=" * 60)
    print("📊 URL          : http://localhost:5000")
    print("📹 Video stream : http://localhost:5000/video_feed")
    print("   (MJPEG from SharedMemory — no WebSocket for video)")
    print("=" * 60)

    ui_queue  = shared_state.ui_queue
    cmd_queue = shared_state.command_queue

    def state_drain():
        print("📥 State drain thread started")
        while True:
            try:
                packet = ui_queue.get(timeout=1.0)
                try:
                    _local_state_q.put_nowait(packet)
                except Exception:
                    pass
            except Exception:
                pass

    threading.Thread(target=state_drain, daemon=True, name="StateDrain").start()

    def state_emit():
        print("📡 State emitter started")
        while True:
            try:
                packet = _local_state_q.get(timeout=1.0)
            except Empty:
                continue
            except Exception:
                continue

            if "action" in packet:
                try:
                    cmd_queue.put_nowait(packet)
                except Exception:
                    pass
                continue

            try:
                socketio.emit("state_update", packet)
            except Exception:
                pass

    threading.Thread(target=state_emit, daemon=True, name="StateEmit").start()

    socketio.run(
        app,
        host="0.0.0.0",
        port=5000,
        debug=False,
        use_reloader=False,
        log_output=False,
    )


# ── Compatibility aliases ─────────────────────────────────────────────────────
def start_ui_server(shared_state_obj, port: int = 5000):
    dashboard_process(shared_state_obj, shm_name=None, frame_lock_val=None, frame_seq=None)


# ── Stand-alone test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000, debug=True, use_reloader=False)