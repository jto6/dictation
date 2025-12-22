#!/usr/bin/env python3
"""
Toggle-based Whisper dictation for Claude Code terminal.
Run once to start recording, run again to stop and transcribe.

Usage:
    - Bind to a keyboard shortcut in your DE settings
    - Or use the .desktop launcher
    - Or run directly: ./dictate-toggle.py
"""

import os
import sys
import signal
import subprocess
import tempfile
import time
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import sounddevice as sd
import soundfile as sf
from faster_whisper import WhisperModel

# Configuration
SAMPLE_RATE = 16000
CHANNELS = 1
MODEL_SIZE = "base.en"
DEVICE = "cuda"  # Change to "cpu" if no GPU
COMPUTE_TYPE = "float16"  # Change to "float32" for CPU

# State management
STATE_DIR = Path("/tmp/whisper-dictation")
STATE_FILE = STATE_DIR / "recording_state.json"
AUDIO_FILE = STATE_DIR / "recording.wav"
PID_FILE = STATE_DIR / "recorder.pid"
LOG_FILE = STATE_DIR / "dictation.log"


def log(message: str):
    """Log message with timestamp."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_line = f"[{timestamp}] {message}"
    print(log_line)
    with open(LOG_FILE, "a") as f:
        f.write(log_line + "\n")


def notify(message: str, urgency: str = "normal"):
    """Show desktop notification."""
    try:
        subprocess.run(
            ["notify-send", "-u", urgency, "-t", "2000", "Dictation", message],
            check=False,
            capture_output=True
        )
    except FileNotFoundError:
        pass  # notify-send not available


def is_recording() -> bool:
    """Check if a recording session is active."""
    if not STATE_FILE.exists():
        return False

    try:
        with open(STATE_FILE) as f:
            state = json.load(f)

        # If state file exists, we're recording or starting
        # Even if "starting" flag is set, treat as recording to prevent race
        if state.get("starting"):
            return True

        # Check if the recorder process is still running
        pid = state.get("pid")
        if pid and Path(f"/proc/{pid}").exists():
            return True
        else:
            # Stale state file, clean up
            cleanup_state()
            return False
    except (json.JSONDecodeError, IOError):
        cleanup_state()
        return False


def cleanup_state():
    """Clean up state files."""
    for f in [STATE_FILE, PID_FILE]:
        try:
            f.unlink(missing_ok=True)
        except Exception:
            pass


def start_recording():
    """Start a new recording session."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)

    # Write state BEFORE forking to prevent race condition
    # Use a temporary PID that we'll update after fork
    temp_state = {
        "pid": os.getpid(),  # Temporary, will be updated
        "starting": True,
        "started": datetime.now().isoformat(),
        "audio_file": str(AUDIO_FILE)
    }
    with open(STATE_FILE, "w") as f:
        json.dump(temp_state, f)
        f.flush()
        os.fsync(f.fileno())

    log("Starting recording...")
    notify("🎤 Recording...", "low")

    # Fork a background process to record
    pid = os.fork()

    if pid == 0:
        # Child process - do the recording
        try:
            record_audio()
        except Exception as e:
            log(f"Recording error: {e}")
        sys.exit(0)
    else:
        # Parent process - update state with actual PID
        state = {
            "pid": pid,
            "started": datetime.now().isoformat(),
            "audio_file": str(AUDIO_FILE)
        }
        with open(STATE_FILE, "w") as f:
            json.dump(state, f)
            f.flush()
            os.fsync(f.fileno())

        log(f"Recording started (PID: {pid})")


def record_audio():
    """Record audio until terminated."""
    audio_data = []

    def callback(indata, frames, time_info, status):
        audio_data.append(indata.copy())

    # Set up signal handler to save audio on termination
    def save_and_exit(signum, frame):
        if audio_data:
            audio = np.concatenate(audio_data, axis=0)
            sf.write(str(AUDIO_FILE), audio, SAMPLE_RATE)
            log(f"Audio saved: {len(audio)/SAMPLE_RATE:.1f}s")
        sys.exit(0)

    signal.signal(signal.SIGTERM, save_and_exit)
    signal.signal(signal.SIGINT, save_and_exit)

    # Start recording
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS,
                        dtype=np.float32, callback=callback):
        # Record indefinitely until killed
        while True:
            time.sleep(0.1)


def stop_recording_and_transcribe():
    """Stop recording and transcribe the audio."""
    if not STATE_FILE.exists():
        log("No active recording")
        notify("No active recording", "normal")
        return

    try:
        with open(STATE_FILE) as f:
            state = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        log(f"Error reading state: {e}")
        cleanup_state()
        return

    pid = state.get("pid")
    if not pid:
        log("No recorder PID found")
        cleanup_state()
        return

    # Send SIGTERM to recorder process to trigger audio save
    log(f"Stopping recording (PID: {pid})...")
    notify("⏹️ Processing...", "low")

    try:
        os.kill(pid, signal.SIGTERM)
        # Wait for process to save audio and exit
        for _ in range(50):  # 5 second timeout
            if not Path(f"/proc/{pid}").exists():
                break
            time.sleep(0.1)
    except ProcessLookupError:
        log("Recorder process already exited")

    cleanup_state()

    # Check if audio file was created
    if not AUDIO_FILE.exists():
        log("No audio file created")
        notify("No audio recorded", "normal")
        return

    # Transcribe
    log("Transcribing...")
    text = transcribe_audio(AUDIO_FILE)

    # Clean up audio file
    AUDIO_FILE.unlink(missing_ok=True)

    if text:
        log(f"Result: {text}")
        notify(f"✓ {text[:50]}..." if len(text) > 50 else f"✓ {text}", "low")
        type_text(text)
    else:
        log("No speech detected")
        notify("No speech detected", "normal")


def transcribe_audio(audio_path: Path) -> str:
    """Transcribe audio file using Whisper."""
    log(f"Loading Whisper model '{MODEL_SIZE}'...")
    model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)

    try:
        segments, _ = model.transcribe(
            str(audio_path),
            beam_size=5,
            language="en",
            vad_filter=True,
        )
        return " ".join(seg.text for seg in segments).strip()
    except Exception as e:
        log(f"Transcription error: {e}")
        return ""


def type_text(text: str):
    """Type text using xdotool or wtype."""
    if not text:
        return

    # Small delay to ensure focus is correct
    time.sleep(0.1)

    try:
        # Use xdotool for X11
        subprocess.run(
            ["xdotool", "type", "--clearmodifiers", "--delay", "0", "--", text],
            check=True
        )
        log("Text typed successfully")
    except FileNotFoundError:
        try:
            # Fallback to wtype for Wayland
            subprocess.run(["wtype", text], check=True)
            log("Text typed successfully (wtype)")
        except FileNotFoundError:
            # Last resort - copy to clipboard
            try:
                subprocess.run(
                    ["xclip", "-selection", "clipboard"],
                    input=text.encode(),
                    check=True
                )
                log("Text copied to clipboard (paste with Ctrl+V)")
                notify("Text copied to clipboard", "normal")
            except FileNotFoundError:
                log(f"Could not type text. Output: {text}")
                print(f"\n>>> {text}")


def main():
    """Toggle recording state."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)

    if is_recording():
        stop_recording_and_transcribe()
    else:
        start_recording()


if __name__ == "__main__":
    main()
