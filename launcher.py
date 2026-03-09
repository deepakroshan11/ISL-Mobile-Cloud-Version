# launcher.py — CLOUD PRODUCTION VERSION
#
# ════════════════════════════════════════════════════════════════════════════
#  CLOUD FIX APPLIED (1 change only — everything else identical):
#
#  FIX 1 — SharedMemory cleanup on startup:
#           If the previous run crashed without cleanup, "isl_frame_shm"
#           still exists in /dev/shm on Linux. SharedMemory(create=True)
#           then raises FileExistsError and the entire launcher fails.
#           Fixed by attempting to unlink the old block before creating new.
#
#  ALL original features preserved:
#  ✅ Shared memory block (640×480 BGR, zero-copy frame transfer)
#  ✅ frame_seq counter for dashboard to detect new frames
#  ✅ Auto-restart of crashed child processes (every 2s watch loop)
#  ✅ Clean shutdown with shm.close() + shm.unlink() on Ctrl+C
#  ✅ Core Processor + Web Dashboard spawned as separate processes
# ════════════════════════════════════════════════════════════════════════════

import multiprocessing
import multiprocessing.shared_memory
import ctypes
import time
import sys

from isl_detection    import core_processing_engine, SharedState
from isl_ui_dashboard import dashboard_process

FRAME_W, FRAME_H = 640, 480
FRAME_BYTES      = FRAME_W * FRAME_H * 3   # 921 600 bytes for one BGR frame


def _run_core(shared_state, shm_name, frame_lock_val, frame_seq):
    core_processing_engine(shared_state, shm_name, frame_lock_val, frame_seq)


def _run_dashboard(shared_state, shm_name, frame_lock_val, frame_seq):
    dashboard_process(shared_state, shm_name, frame_lock_val, frame_seq)


if __name__ == "__main__":
    multiprocessing.freeze_support()   # required on Windows

    print("=" * 70)
    print(" 🤟 ISL EMOTION TRANSLATOR - MULTI-PROCESS SYSTEM")
    print("=" * 70)
    print()
    print("🎯 Architecture:")
    print("   📹 Core Processor  → Camera · MediaPipe · TF model")
    print("   🌐 Web Dashboard   → MJPEG video · Socket.IO state")
    print("   🔗 SharedMemory    → Zero-copy frame transfer (no base64 queue)")
    print()
    print("=" * 70)

    # FIX 1: Clean up leftover shared memory from a previous crash.
    # On Linux, named shared memory persists in /dev/shm until explicitly
    # unlinked. If the previous run crashed, creating the same name fails.
    try:
        _old_shm = multiprocessing.shared_memory.SharedMemory(name="isl_frame_shm")
        _old_shm.close()
        _old_shm.unlink()
        print("🧹 Cleaned up leftover shared memory from previous run")
    except FileNotFoundError:
        pass  # normal — no leftover, continue

    # ── Shared memory block ──────────────────────────────────────────────────
    shm        = multiprocessing.shared_memory.SharedMemory(
                     create=True, size=FRAME_BYTES, name="isl_frame_shm")
    frame_seq  = multiprocessing.Value(ctypes.c_uint64, 0)
    frame_lock = multiprocessing.Value(ctypes.c_bool,   False)

    shared_state = SharedState()

    # ── Spawn processes ──────────────────────────────────────────────────────
    core_args = (shared_state, shm.name, frame_lock, frame_seq)
    web_args  = (shared_state, shm.name, frame_lock, frame_seq)

    core_proc = multiprocessing.Process(
        target=_run_core, args=core_args, name="CoreProcessor", daemon=True)
    web_proc  = multiprocessing.Process(
        target=_run_dashboard, args=web_args, name="WebDashboard",  daemon=True)

    print("🚀 Starting Core Processor…")
    core_proc.start()

    # Give core a moment to open the camera before the dashboard opens
    time.sleep(1.5)

    print("🚀 Starting Web Dashboard on port 5000…")
    web_proc.start()

    print()
    print("=" * 70)
    print("✅ SYSTEM READY!")
    print("=" * 70)
    print()
    print("📊 Dashboard    → http://localhost:5000")
    print("📹 Video stream → http://localhost:5000/video_feed")
    print()
    print("⌨️  Keyboard shortcuts (in dashboard):")
    print("   Ctrl+R          → Reset")
    print("   Ctrl+Backspace  → Backspace")
    print("   1 / 2 / 3       → Accept suggestion")
    print()
    print("🛑 Press Ctrl+C to stop all processes")
    print("=" * 70)

    try:
        while True:
            # Auto-restart crashed processes
            if not core_proc.is_alive():
                print("⚠️  Core Processor crashed — restarting…")
                core_proc = multiprocessing.Process(
                    target=_run_core, args=core_args, name="CoreProcessor", daemon=True)
                core_proc.start()

            if not web_proc.is_alive():
                print("⚠️  Web Dashboard crashed — restarting…")
                web_proc = multiprocessing.Process(
                    target=_run_dashboard, args=web_args, name="WebDashboard", daemon=True)
                web_proc.start()

            time.sleep(2)

    except KeyboardInterrupt:
        print("\n🛑 Shutting down…")
        core_proc.terminate()
        web_proc.terminate()
        core_proc.join(timeout=5)
        web_proc.join(timeout=5)
        try:
            shm.close()
            shm.unlink()
        except Exception:
            pass
        print("✅ All processes stopped")
        print("=" * 70)
        sys.exit(0)