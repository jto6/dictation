#!/usr/bin/env python3
"""
Whisper dictation daemon - keeps model loaded for fast transcription.
Listens on a Unix socket for record/stop commands.

Usage:
    1. Start daemon: ./dictate-daemon.py start
    2. Toggle recording: ./dictate-daemon.py toggle
    3. Stop daemon: ./dictate-daemon.py stop
"""

import os
import sys
import socket
import signal
import subprocess
import tempfile
import time
import json
import threading
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
DEVICE = "cuda"  # GPU acceleration
COMPUTE_TYPE = "float16"  # Fast on GPU

# Whisper prompt to bias toward technical/programming vocabulary
INITIAL_PROMPT = """
git commit --message, git push, git pull, git checkout, git branch,
Linux, Ubuntu, Python, JavaScript, TypeScript, Bash, Docker, Kubernetes,
npm install, pip install, sudo apt, Claude Code, API, JSON, YAML, SQL,
pytest, mypy, eslint, webpack, venv, virtualenv, conda,
"""

# Post-processing replacements: {"wrong": "correct"}
# Add your custom replacements here
REPLACEMENTS = {
    "get commit": "git commit",
    "get push": "git push",
    "get pull": "git pull",
    "get checkout": "git checkout",
    "get branch": "git branch",
    "get status": "git status",
    "get diff": "git diff",
    "get log": "git log",
    "get add": "git add",
    "get reset": "git reset",
    "get stash": "git stash",
    "get merge": "git merge",
    "get rebase": "git rebase",
    "get clone": "git clone",
    "get init": "git init",
    "lennox": "Linux",
    "Lennox": "Linux",
    "lennix": "Linux",
    "Lennix": "Linux",
    "pie test": "pytest",
    "pie charm": "PyCharm",
    "jason": "JSON",
    "Jason": "JSON",
    "yam-l": "YAML",
    "yamel": "YAML",
    "my pie": "mypy",
    "my pi": "mypy",
    "E.S. lint": "ESLint",
    "D.O.S.": "DOS",
    "cloudcode": "Claude Code",
    "cloud code": "Claude Code",
    "pseudo": "sudo",
    " - - ": " --",
    "- -message": "--message",
    "- -help": "--help",
    "- -version": "--version",
    "- -verbose": "--verbose",
    "- -force": "--force",
}

# Paths
STATE_DIR = Path("/tmp/whisper-dictation")
SOCKET_PATH = STATE_DIR / "daemon.sock"
PID_FILE = STATE_DIR / "daemon.pid"
LOG_FILE = STATE_DIR / "daemon.log"
AUDIO_FILE = STATE_DIR / "recording.wav"


def apply_replacements(text: str) -> str:
    """Apply post-processing replacements to fix common transcription errors."""
    for wrong, correct in REPLACEMENTS.items():
        text = text.replace(wrong, correct)
    return text


def log(message: str):
    """Log with timestamp."""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    log_line = f"[{timestamp}] {message}"
    print(log_line, flush=True)
    try:
        with open(LOG_FILE, "a") as f:
            f.write(log_line + "\n")
    except Exception:
        pass


def notify(message: str, urgency: str = "normal"):
    """Show desktop notification."""
    try:
        subprocess.Popen(
            ["notify-send", "-u", urgency, "-t", "2000", "Dictation", message],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
    except Exception:
        pass


def type_text(text: str):
    """Type text using xdotool."""
    if not text:
        return

    time.sleep(0.05)  # Brief delay for focus
    try:
        subprocess.run(
            ["xdotool", "type", "--clearmodifiers", "--delay", "0", "--", text],
            check=True,
            capture_output=True
        )
        log(f"Typed: {text[:50]}...")
    except subprocess.CalledProcessError as e:
        log(f"xdotool error: {e}")
    except FileNotFoundError:
        try:
            subprocess.run(["wtype", text], check=True, capture_output=True)
        except Exception:
            log(f"Could not type. Text: {text}")


class DictationDaemon:
    def __init__(self):
        self.model = None
        self.recording = False
        self.audio_data = []
        self.stream = None
        self.running = True
        self.lock = threading.Lock()

    def load_model(self):
        """Load Whisper model."""
        log(f"Loading Whisper model '{MODEL_SIZE}'...")
        start = time.time()
        self.model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
        log(f"Model loaded in {time.time() - start:.2f}s")

    def audio_callback(self, indata, frames, time_info, status):
        """Audio stream callback."""
        if self.recording:
            self.audio_data.append(indata.copy())

    def start_recording(self):
        """Start recording audio."""
        with self.lock:
            if self.recording:
                return "Already recording"

            self.audio_data = []
            self.recording = True

            self.stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype=np.float32,
                callback=self.audio_callback
            )
            self.stream.start()

        log("Recording started")
        notify("🎤 Recording...", "low")
        return "Recording"

    def stop_recording(self) -> str:
        """Stop recording and transcribe."""
        with self.lock:
            if not self.recording:
                return "Not recording"

            self.recording = False

            if self.stream:
                self.stream.stop()
                self.stream.close()
                self.stream = None

            audio_data = self.audio_data
            self.audio_data = []

        if not audio_data:
            log("No audio captured")
            return "No audio"

        audio = np.concatenate(audio_data, axis=0)
        duration = len(audio) / SAMPLE_RATE
        log(f"Captured {duration:.1f}s of audio")

        notify("⏹️ Transcribing...", "low")

        # Save and transcribe
        sf.write(str(AUDIO_FILE), audio, SAMPLE_RATE)

        try:
            start = time.time()
            segments, _ = self.model.transcribe(
                str(AUDIO_FILE),
                beam_size=5,
                language="en",
                vad_filter=True,
                initial_prompt=INITIAL_PROMPT,
            )
            raw_text = " ".join(seg.text for seg in segments).strip()
            text = apply_replacements(raw_text)
            elapsed = time.time() - start
            if text != raw_text:
                log(f"Transcribed in {elapsed:.2f}s: {raw_text} → {text}")
            else:
                log(f"Transcribed in {elapsed:.2f}s: {text}")
        except Exception as e:
            log(f"Transcription error: {e}")
            text = ""
        finally:
            AUDIO_FILE.unlink(missing_ok=True)

        if text:
            notify(f"✓ {text[:40]}..." if len(text) > 40 else f"✓ {text}", "low")
            type_text(text)
            return f"OK: {text}"
        else:
            notify("No speech detected", "normal")
            return "No speech"

    def toggle(self) -> str:
        """Toggle recording state."""
        if self.recording:
            return self.stop_recording()
        else:
            return self.start_recording()

    def handle_client(self, conn):
        """Handle a client connection."""
        try:
            data = conn.recv(1024).decode().strip()
            log(f"Command: {data}")

            if data == "toggle":
                response = self.toggle()
            elif data == "start":
                response = self.start_recording()
            elif data == "stop":
                response = self.stop_recording()
            elif data == "status":
                response = "Recording" if self.recording else "Idle"
            elif data == "quit":
                self.running = False
                response = "Shutting down"
            else:
                response = f"Unknown command: {data}"

            conn.sendall(response.encode())
        except Exception as e:
            log(f"Client error: {e}")
        finally:
            conn.close()

    def run(self):
        """Run the daemon."""
        STATE_DIR.mkdir(parents=True, exist_ok=True)

        # Clean up old socket
        SOCKET_PATH.unlink(missing_ok=True)

        # Create Unix socket
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(str(SOCKET_PATH))
        server.listen(1)
        server.settimeout(1.0)

        # Save PID
        with open(PID_FILE, "w") as f:
            f.write(str(os.getpid()))

        # Load model
        self.load_model()

        log("Daemon ready, listening for commands")
        notify("Dictation daemon started", "low")

        def signal_handler(sig, frame):
            log("Shutdown signal received")
            self.running = False

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        while self.running:
            try:
                conn, _ = server.accept()
                self.handle_client(conn)
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    log(f"Server error: {e}")

        # Cleanup
        server.close()
        SOCKET_PATH.unlink(missing_ok=True)
        PID_FILE.unlink(missing_ok=True)
        log("Daemon stopped")


def send_command(cmd: str) -> str:
    """Send command to daemon."""
    if not SOCKET_PATH.exists():
        return "Daemon not running"

    try:
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.connect(str(SOCKET_PATH))
        client.sendall(cmd.encode())
        response = client.recv(4096).decode()
        client.close()
        return response
    except Exception as e:
        return f"Error: {e}"


def is_daemon_running() -> bool:
    """Check if daemon is running."""
    if not PID_FILE.exists():
        return False
    try:
        pid = int(PID_FILE.read_text().strip())
        return Path(f"/proc/{pid}").exists()
    except Exception:
        return False


def start_daemon():
    """Start the daemon in background."""
    if is_daemon_running():
        print("Daemon already running")
        return

    # Fork to background
    pid = os.fork()
    if pid > 0:
        # Parent - wait briefly then confirm
        time.sleep(0.5)
        if is_daemon_running():
            print(f"Daemon started (PID: {pid})")
        else:
            print("Failed to start daemon")
        return

    # Child - become daemon
    os.setsid()
    pid = os.fork()
    if pid > 0:
        sys.exit(0)

    # Redirect stdio BEFORE creating daemon to avoid duplicate logs
    sys.stdin = open("/dev/null")
    log_handle = open(LOG_FILE, "a", buffering=1)  # Line buffered
    sys.stdout = log_handle
    sys.stderr = log_handle
    os.dup2(log_handle.fileno(), 1)
    os.dup2(log_handle.fileno(), 2)

    daemon = DictationDaemon()
    daemon.run()


def stop_daemon():
    """Stop the daemon."""
    if not is_daemon_running():
        print("Daemon not running")
        return

    response = send_command("quit")
    print(response)


def main():
    STATE_DIR.mkdir(parents=True, exist_ok=True)

    if len(sys.argv) < 2:
        cmd = "toggle"
    else:
        cmd = sys.argv[1]

    if cmd == "start":
        start_daemon()
    elif cmd == "stop":
        stop_daemon()
    elif cmd == "toggle":
        if not is_daemon_running():
            print("Daemon not running. Start with: dictate-daemon.py start")
            sys.exit(1)
        response = send_command("toggle")
        print(response)
    elif cmd == "status":
        if is_daemon_running():
            response = send_command("status")
            print(f"Daemon running, state: {response}")
        else:
            print("Daemon not running")
    else:
        print(f"Usage: {sys.argv[0]} [start|stop|toggle|status]")
        sys.exit(1)


if __name__ == "__main__":
    main()
