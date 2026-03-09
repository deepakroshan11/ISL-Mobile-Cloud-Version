# unified_launcher.py
# ═══════════════════════════════════════════════════════════════════════════
#  ISL FULL SYSTEM — SINGLE COMMAND LAUNCHER
#  Starts BOTH services from ONE terminal:
#
#    LOCAL  →  python unified_launcher.py
#    CLOUD  →  RUN_MODE=CLOUD python unified_launcher.py
#
#  Port 5000 → ISL Gesture Recognition + Emotion Detection  (launcher.py)
#  Port 5001 → Speech-to-Sign / Fingerspelling Avatar        (speech_to_sign_avatar.py)
#              OR speech_to_sign_avatar_fixed.py  (auto-detected)
#
#  ⚠️  WHY subprocess.Popen (not multiprocessing)?
#      speech_to_sign_avatar*.py calls eventlet.monkey_patch() at import-time,
#      which corrupts the multiprocessing runtime.  Running each script as a
#      fully separate OS process avoids this entirely.
# ═══════════════════════════════════════════════════════════════════════════

import subprocess
import sys
import os
import time
import multiprocessing

# ── Resolve paths ─────────────────────────────────────────────────────────────
PYTHON   = sys.executable          # same venv / interpreter that started us
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RUN_MODE = os.getenv("RUN_MODE", "LOCAL").upper()

# ── Auto-detect which avatar script exists ────────────────────────────────────
# Prefer the fixed/fingerspelling version if present, fall back to original.
AVATAR_CANDIDATES = [
    "speech_to_sign_avatar_fixed.py",
    "speech_to_sign_avatar.py",
]
AVATAR_SCRIPT = None
for _candidate in AVATAR_CANDIDATES:
    _path = os.path.join(BASE_DIR, _candidate)
    if os.path.exists(_path):
        AVATAR_SCRIPT = _path
        break

GESTURE_SCRIPT = os.path.join(BASE_DIR, "launcher.py")

# ── Environment for child processes ──────────────────────────────────────────
CHILD_ENV = {
    **os.environ,
    "RUN_MODE"         : RUN_MODE,
    "PYTHONUNBUFFERED" : "1",        # immediate stdout/stderr in logs
}

# ── Oracle Cloud: open firewall ports (run once manually) ─────────────────────
#   sudo firewall-cmd --permanent --add-port=5000/tcp
#   sudo firewall-cmd --permanent --add-port=5001/tcp
#   sudo firewall-cmd --reload
#   Also open in OCI Console → VCN → Security List → Ingress Rules


# ─────────────────────────────────────────────────────────────────────────────
def start_process(script_path, label):
    """Spawn a child process and return the Popen handle."""
    print(f"  🚀 Starting {label}")
    print(f"     Script : {script_path}")
    proc = subprocess.Popen(
        [PYTHON, script_path],
        env=CHILD_ENV,
        cwd=BASE_DIR,
        stdout=None,   # inherit — all logs appear in the same terminal
        stderr=None,
    )
    print(f"     PID    : {proc.pid}")
    return proc


def stop_all(processes):
    """Gracefully terminate all child processes, then force-kill if needed."""
    print("\n🛑 Shutting down all processes…")
    for label, proc in processes.items():
        if proc and proc.poll() is None:
            print(f"   → Terminating  {label}  (PID {proc.pid})")
            proc.terminate()

    deadline = time.time() + 6
    for label, proc in processes.items():
        if proc:
            remaining = max(0.1, deadline - time.time())
            try:
                proc.wait(timeout=remaining)
                print(f"   ✅ {label} stopped cleanly")
            except subprocess.TimeoutExpired:
                print(f"   ⚡ Force-killing {label}  (PID {proc.pid})")
                proc.kill()

    print("✅ All processes stopped")
    print("=" * 70)


def preflight_check():
    """Verify required scripts exist before starting anything."""
    ok = True

    if not os.path.exists(GESTURE_SCRIPT):
        print(f"❌ MISSING: launcher.py not found at:\n   {GESTURE_SCRIPT}")
        ok = False

    if AVATAR_SCRIPT is None:
        print("❌ MISSING: No avatar script found.")
        print("   Expected one of:", AVATAR_CANDIDATES)
        ok = False

    if not ok:
        print("\n   Make sure unified_launcher.py is in the SAME folder as the other scripts.")
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
def main():
    print()
    print("=" * 70)
    print("  🤟  ISL FULL SYSTEM  —  UNIFIED LAUNCHER")
    print("=" * 70)
    print(f"  RUN_MODE  : {RUN_MODE}")
    print(f"  Python    : {PYTHON}")
    print(f"  Base dir  : {BASE_DIR}")
    print(f"  Gesture   : {os.path.basename(GESTURE_SCRIPT)}  → port 5000")
    if AVATAR_SCRIPT:
        print(f"  Avatar    : {os.path.basename(AVATAR_SCRIPT)}  → port 5001")
    print("=" * 70)
    print()

    preflight_check()

    processes = {
        "ISL Gesture  (5000)" : None,
        "Avatar Server (5001)": None,
    }

    restart_map = {
        "ISL Gesture  (5000)" : GESTURE_SCRIPT,
        "Avatar Server (5001)": AVATAR_SCRIPT,
    }

    # ── Start gesture system first; give camera time to initialise ────────────
    processes["ISL Gesture  (5000)"] = start_process(
        GESTURE_SCRIPT, "ISL Gesture Recognition  (port 5000)")
    print()
    print("   ⏳ Waiting 2 s for camera to initialise…")
    time.sleep(2)
    print()

    # ── Start avatar server ───────────────────────────────────────────────────
    processes["Avatar Server (5001)"] = start_process(
        AVATAR_SCRIPT, "Speech-to-Sign Avatar  (port 5001)")
    time.sleep(1)

    # ── Ready banner ──────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("  ✅  ALL SYSTEMS RUNNING!")
    print("=" * 70)

    if RUN_MODE == "CLOUD":
        public_ip = os.getenv("PUBLIC_IP", "<YOUR_ORACLE_PUBLIC_IP>")
        print(f"  📊 ISL Dashboard   →  http://{public_ip}:5000")
        print(f"  🎭 Avatar System   →  http://{public_ip}:5001")
        print()
        print("  💡 Oracle Cloud checklist:")
        print("     1. Set env var:  export PUBLIC_IP=<your-instance-ip>")
        print("     2. OCI Console → VCN → Security List → add Ingress rules")
        print("        for TCP port 5000 and TCP port 5001  (source 0.0.0.0/0)")
        print("     3. On the VM (run once):")
        print("        sudo firewall-cmd --permanent --add-port=5000/tcp")
        print("        sudo firewall-cmd --permanent --add-port=5001/tcp")
        print("        sudo firewall-cmd --reload")
        print("     4. To auto-start on reboot → see isl.service below")
    else:
        print("  📊 ISL Dashboard   →  http://localhost:5000")
        print("  🎭 Avatar System   →  http://localhost:5001")

    print()
    print("  🛑 Press Ctrl+C to stop everything")
    print("=" * 70)
    print()

    # ── Watch-dog loop — auto-restarts any crashed process ───────────────────
    try:
        while True:
            for label, script in restart_map.items():
                proc = processes[label]
                if proc is not None and proc.poll() is not None:
                    exit_code = proc.returncode
                    print(f"\n⚠️  [{label}] exited with code {exit_code}")
                    print("   Restarting in 3 s…")
                    time.sleep(3)
                    processes[label] = start_process(script, label)
                    print()
            time.sleep(3)

    except KeyboardInterrupt:
        stop_all(processes)
        sys.exit(0)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    multiprocessing.freeze_support()   # no-op on Linux/Mac, required on Win
    main()