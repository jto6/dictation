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
from queue import Queue, Empty
from pathlib import Path
from datetime import datetime

import numpy as np
import sounddevice as sd
import soundfile as sf
from faster_whisper import WhisperModel

# Configuration
SAMPLE_RATE = 16000
CHANNELS = 1

# Model/device/compute settings - will auto-detect best configuration
# Set to specific values to override auto-detection
MODEL_SIZE = "auto"  # "auto", "tiny.en", "base.en", "small.en", "medium.en", "large-v3"
DEVICE = "auto"  # "auto", "cuda", or "cpu"
COMPUTE_TYPE = "auto"  # "auto", "float16", "float32", "int8"

# Audio input device - set to None for system default, or device name/index
# Use `python -c "import sounddevice; print(sounddevice.query_devices())"` to list devices
# Examples: None, 0, "pipewire", "HDA Intel PCH"
AUDIO_DEVICE = None

# Streaming mode settings
SILENCE_THRESHOLD = 0.01  # RMS threshold for silence detection
SILENCE_DURATION = 0.7    # Seconds of silence to trigger phrase transcription
MIN_PHRASE_DURATION = 0.3 # Minimum audio duration to transcribe
PAUSE_PUNCTUATION_THRESHOLD = 1.5  # Silence longer than this = intentional pause, strip punctuation

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
    # Slash commands
    "slash commit": "/commit",
    "slash help": "/help",
    "slash status": "/status",
    "slash ": "/",  # Generic fallback for "slash X" → "/X"
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


def strip_trailing_punctuation(text: str) -> str:
    """Strip trailing sentence-ending punctuation for streaming mode.

    This prevents periods/ellipsis from being inserted when pausing mid-dictation.
    Keeps commas and other mid-sentence punctuation.
    """
    # Strip trailing whitespace first
    text = text.rstrip()
    # Remove sentence-ending punctuation
    while text and text[-1] in '.!?':
        text = text[:-1]
    # Also handle ellipsis that might be separate
    text = text.rstrip()
    if text.endswith('...'):
        text = text[:-3].rstrip()
    elif text.endswith('..'):
        text = text[:-2].rstrip()
    return text


def log(message: str):
    """Log with timestamp."""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    log_line = f"[{timestamp}] {message}"
    # When running as daemon, stdout is redirected to LOG_FILE, so just print.
    # Only write directly to file when stdout is NOT already the log file
    # (i.e., when running in foreground for debugging).
    print(log_line, flush=True)
    # Check if stdout is redirected to the log file (daemon mode)
    try:
        if not hasattr(sys.stdout, 'name') or sys.stdout.name != str(LOG_FILE):
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


def play_sound(sound_name: str):
    """Play a system sound for audio feedback."""
    sound_path = f"/usr/share/sounds/freedesktop/stereo/{sound_name}.oga"
    if not Path(sound_path).exists():
        return

    # Try pw-play (PipeWire), then paplay (PulseAudio), then aplay (ALSA)
    for player in ["pw-play", "paplay", "aplay"]:
        try:
            subprocess.Popen(
                [player, sound_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            return
        except FileNotFoundError:
            continue


# Maximum characters to paste in one chunk to avoid terminal buffer issues
PASTE_CHUNK_SIZE = 300


def reset_modifier_keys():
    """Reset modifier keys to prevent stuck Ctrl/Alt/Shift states."""
    session_type = os.environ.get("XDG_SESSION_TYPE", "").lower()
    try:
        if session_type == "wayland":
            # Release all modifiers via ydotool (29=ctrl, 42=shift, 56=alt)
            subprocess.run(
                ["ydotool", "key", "29:0", "42:0", "56:0"],
                capture_output=True,
                timeout=2
            )
        else:
            # X11: use xdotool keyup for all modifiers
            subprocess.run(
                ["xdotool", "keyup", "ctrl", "shift", "alt", "super"],
                capture_output=True,
                timeout=2
            )
    except Exception:
        pass  # Best effort


def type_text(text: str):
    """Insert text using clipboard paste (more reliable than simulated typing).

    Simulated typing with xdotool/ydotool can drop or reorder characters when
    applications can't keep up with the keystroke rate. Clipboard paste is
    atomic and much more reliable.

    For very long text, paste in chunks to avoid terminal buffer issues.
    """
    if not text:
        return

    time.sleep(0.05)  # Brief delay for focus

    # Detect session type and use appropriate tool
    session_type = os.environ.get("XDG_SESSION_TYPE", "").lower()

    # Split into chunks for long text to avoid overwhelming terminals
    if len(text) > PASTE_CHUNK_SIZE:
        chunks = []
        remaining = text
        while remaining:
            # Try to break at word boundary within chunk size
            if len(remaining) <= PASTE_CHUNK_SIZE:
                chunks.append(remaining)
                break
            # Find last space within chunk size
            chunk = remaining[:PASTE_CHUNK_SIZE]
            last_space = chunk.rfind(' ')
            if last_space > PASTE_CHUNK_SIZE // 2:
                # Break at word boundary
                chunks.append(remaining[:last_space + 1])
                remaining = remaining[last_space + 1:]
            else:
                # No good word boundary, just break at chunk size
                chunks.append(chunk)
                remaining = remaining[PASTE_CHUNK_SIZE:]
        log(f"Splitting {len(text)} chars into {len(chunks)} chunks")
    else:
        chunks = [text]

    for i, chunk in enumerate(chunks):
        if not chunk:
            continue

        try:
            if session_type == "wayland":
                _paste_wayland(chunk)
            else:
                _paste_x11(chunk)

            if len(chunks) > 1 and i < len(chunks) - 1:
                # Delay between chunks to let terminal process
                time.sleep(0.15)
        except Exception as e:
            log(f"Paste error on chunk {i+1}/{len(chunks)}: {e}")
            reset_modifier_keys()
            return

    # Always reset modifier keys after pasting to prevent stuck state
    reset_modifier_keys()

    if len(text) > 50:
        log(f"Pasted: {text[:50]}... ({len(text)} chars)")
    else:
        log(f"Pasted: {text}")


def _paste_wayland(text: str):
    """Type text on Wayland using ydotool simulated keystrokes.

    Direct typing works universally across all applications.
    """
    try:
        subprocess.run(
            ["ydotool", "type", "--key-delay", "10", "--", text],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=60
        )
    except subprocess.TimeoutExpired:
        log(f"ydotool type timed out for {len(text)} chars")
        raise
    except Exception as e:
        log(f"ydotool type error: {e}")
        raise


def _paste_x11(text: str):
    """Type text on X11 using xdotool simulated keystrokes.

    Direct typing works universally across all applications (terminals, Emacs,
    browsers, etc.) unlike clipboard paste which requires Ctrl+V support.
    The delay between keystrokes prevents character dropping in applications
    that can't keep up with rapid input.
    """
    try:
        subprocess.run(
            ["xdotool", "type", "--clearmodifiers", "--delay", "20", "--", text],
            check=True,
            capture_output=True,
            timeout=60
        )
    except subprocess.TimeoutExpired:
        log(f"xdotool type timed out for {len(text)} chars")
        raise
    except Exception as e:
        log(f"xdotool type error: {e}")
        raise


def get_gpu_vram_mb():
    """Get total and available GPU VRAM in MB using nvidia-smi.

    Returns:
        tuple: (total_mb, available_mb) - both 0 if query fails
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            # Get first GPU's VRAM (format: "total, free")
            parts = result.stdout.strip().split('\n')[0].split(',')
            total = int(parts[0].strip())
            free = int(parts[1].strip())
            return total, free
    except Exception:
        pass
    return 0, 0


# Model hierarchy from largest to smallest (for fallback)
MODEL_HIERARCHY = ["large-v3", "medium.en", "small.en", "base.en", "tiny.en"]

# Approximate VRAM requirements in MB (float16)
MODEL_VRAM_REQUIREMENTS = {
    "large-v3": 5000,   # ~5GB
    "medium.en": 2500,  # ~2.5GB
    "small.en": 1500,   # ~1.5GB
    "base.en": 1000,    # ~1GB
    "tiny.en": 500,     # ~0.5GB
}


def select_model_for_vram(vram_mb: int, device: str) -> str:
    """Select the best Whisper model based on available VRAM.

    Prefers English-only (.en) models since they're optimized for English
    and more VRAM-efficient. The multilingual large-v3 is only ~15% better
    than medium.en for English, but uses 2x the VRAM - not worth it unless
    VRAM is truly abundant.

    Approximate VRAM usage (float16):
    - large-v3:  ~5GB  (1550M params, multilingual)
    - medium.en: ~2.5GB (769M params, English-optimized)
    - small.en:  ~1.5GB (244M params, English-optimized)
    - base.en:   ~1GB   (74M params, English-optimized)

    English Word Error Rate (lower is better):
    - large-v3:  ~2.5%
    - medium.en: ~2.9%
    - small.en:  ~3.4%
    - base.en:   ~4.3%
    """
    if device == "cpu":
        # For CPU, use base.en as default (good balance of speed/quality)
        return "base.en"

    # Thresholds based on available VRAM with ~1.5x headroom for inference buffers
    # Model requirements: large-v3 ~5GB, medium.en ~2.5GB, small.en ~1.5GB, base.en ~1GB
    if vram_mb >= 7000:   # 7GB+: large-v3 (5GB + headroom)
        return "large-v3"
    elif vram_mb >= 4000:  # 4GB+: medium.en (2.5GB + headroom)
        return "medium.en"
    elif vram_mb >= 2500:  # 2.5GB+: small.en (1.5GB + headroom)
        return "small.en"
    elif vram_mb >= 1500:  # 1.5GB+: base.en (1GB + headroom)
        return "base.en"
    else:
        return "tiny.en"


def get_model_fallback_chain(starting_model: str) -> list:
    """Get list of models to try, starting from given model down to smallest.

    Returns models from starting_model down through the hierarchy,
    allowing graceful degradation when VRAM is tight.
    """
    try:
        start_idx = MODEL_HIERARCHY.index(starting_model)
        return MODEL_HIERARCHY[start_idx:]
    except ValueError:
        # Unknown model, just return it alone
        return [starting_model]


def detect_best_config():
    """Auto-detect best device, compute type, and model size configuration."""
    import ctranslate2

    device = DEVICE
    compute_type = COMPUTE_TYPE
    model_size = MODEL_SIZE

    # Determine device
    if device == "auto":
        if ctranslate2.get_cuda_device_count() > 0:
            device = "cuda"
            log("CUDA detected")
        else:
            device = "cpu"
            log("No CUDA, using CPU")

    # Determine compute type
    if compute_type == "auto":
        if device == "cpu":
            compute_type = "int8"  # Best for CPU
        else:
            # For CUDA, test which compute types actually work
            # Some GPUs/drivers have issues with float16
            compute_type = "float16"  # Try float16 first

    # Determine model size based on available VRAM (not total)
    if model_size == "auto":
        total_vram, available_vram = get_gpu_vram_mb()
        if total_vram > 0:
            log(f"GPU VRAM: {available_vram} MB available / {total_vram} MB total")
        # Use available VRAM for model selection, not total
        model_size = select_model_for_vram(available_vram, device)
        log(f"Auto-selected model: {model_size}")

    return device, compute_type, model_size


def validate_model(model, device, compute_type):
    """Quick validation that model produces sensible output."""
    if device == "cpu":
        return True  # CPU always works

    # Generate 1 second of silence and transcribe
    # A working model should return empty or near-empty result quickly
    silence = np.zeros(SAMPLE_RATE, dtype=np.float32)
    temp_file = STATE_DIR / "validation.wav"

    try:
        sf.write(str(temp_file), silence, SAMPLE_RATE)

        # Warm-up pass: CUDA/cuDNN compiles optimized kernels on first run,
        # which can take several seconds. Run once untimed to trigger JIT
        # compilation before the actual validation.
        log(f"Warming up {device}/{compute_type}...")
        list(model.transcribe(str(temp_file), language="en", vad_filter=False)[0])

        # Now run the timed validation
        start = time.time()
        segments, _ = model.transcribe(str(temp_file), language="en", vad_filter=False)
        result = " ".join(seg.text for seg in segments).strip()
        elapsed = time.time() - start

        # Validation criteria:
        # - Should complete in reasonable time (< 5s for 1s of audio)
        # - Result should be short (silence shouldn't produce long text)
        if elapsed > 5.0:
            log(f"Validation failed: too slow ({elapsed:.1f}s)")
            return False
        if len(result) > 50:
            log(f"Validation failed: garbage output ({len(result)} chars)")
            return False

        log(f"Validation passed ({elapsed:.2f}s)")
        return True
    except Exception as e:
        log(f"Validation failed: {e}")
        return False
    finally:
        temp_file.unlink(missing_ok=True)


class DictationDaemon:
    def __init__(self):
        self.model = None
        self.recording = False
        self.audio_data = []
        self.stream = None
        self.running = True
        self.lock = threading.Lock()

        # Streaming mode state
        self.streaming_mode = False  # False = batch mode, True = streaming mode
        self.silence_samples = 0     # Count of consecutive silent samples
        self.phrase_audio = []       # Audio buffer for current phrase
        self.transcribe_queue = Queue()  # Queue for phrase transcription
        self.transcribe_thread = None
        self.last_transcribe_time = 0
        self.pending_phrase = None   # (audio_chunks, silence_start_time) when waiting to measure pause
        self.phrase_has_speech = False  # Whether current phrase contains actual speech
        self.previous_text = ""        # Previous transcription for context conditioning

    def load_model(self):
        """Load Whisper model with auto-detection and fallback.

        Fallback strategy:
        1. Start with model selected based on available VRAM
        2. Try float16, then float32 for that model on CUDA
        3. If both fail (likely OOM), try the next smaller model on CUDA
        4. Repeat until base.en is tried
        5. Only fall back to CPU as last resort
        """
        start = time.time()

        device, compute_type, model_size = detect_best_config()
        log(f"Loading Whisper model '{model_size}'...")

        if device == "cuda":
            # Build fallback chain: try progressively smaller models on GPU
            # before falling back to CPU
            model_chain = get_model_fallback_chain(model_size)

            for model_name in model_chain:
                # For each model, try float16 then float32
                for ct in ["float16", "float32"]:
                    try:
                        log(f"Trying {model_name} on cuda/{ct}...")
                        model = WhisperModel(model_name, device="cuda", compute_type=ct)

                        if validate_model(model, "cuda", ct):
                            self.model = model
                            log(f"Model loaded in {time.time() - start:.2f}s (model={model_name}, device=cuda, compute={ct})")
                            return
                        else:
                            log(f"Config cuda/{ct} failed validation, trying next...")
                    except Exception as e:
                        error_msg = str(e).lower()
                        if "out of memory" in error_msg or "oom" in error_msg:
                            log(f"CUDA OOM with {model_name}/{ct}, trying smaller model...")
                            break  # Skip float32, go straight to smaller model
                        log(f"Failed to load {model_name} with cuda/{ct}: {e}")
                        continue

            # All CUDA attempts failed, fall back to CPU
            log("All CUDA configurations failed, falling back to CPU...")
            try:
                # Use base.en for CPU (good speed/quality balance)
                model = WhisperModel("base.en", device="cpu", compute_type="int8")
                if validate_model(model, "cpu", "int8"):
                    self.model = model
                    log(f"Model loaded in {time.time() - start:.2f}s (model=base.en, device=cpu, compute=int8)")
                    return
            except Exception as e:
                log(f"CPU fallback failed: {e}")
        else:
            # CPU mode - just try the requested configuration
            try:
                model = WhisperModel(model_size, device=device, compute_type=compute_type)
                if validate_model(model, device, compute_type):
                    self.model = model
                    log(f"Model loaded in {time.time() - start:.2f}s (model={model_size}, device={device}, compute={compute_type})")
                    return
            except Exception as e:
                log(f"Failed to load model: {e}")

        raise RuntimeError("Could not load model with any configuration")

    def audio_callback(self, indata, frames, time_info, status):
        """Audio stream callback."""
        if not self.recording:
            return

        audio_chunk = indata.copy()

        if not self.streaming_mode:
            # Batch mode: just accumulate audio
            self.audio_data.append(audio_chunk)
        else:
            # Streaming mode: detect phrase boundaries via silence

            # Calculate RMS energy
            rms = np.sqrt(np.mean(audio_chunk**2))
            is_silence = rms < SILENCE_THRESHOLD

            if is_silence:
                self.silence_samples += frames
                # Only accumulate audio if we've had speech (don't buffer pure silence)
                if self.phrase_has_speech:
                    self.phrase_audio.append(audio_chunk)
            else:
                # User is speaking
                self.phrase_audio.append(audio_chunk)
                self.phrase_has_speech = True

                # If we had a pending phrase, queue it now that we know pause duration
                if self.pending_phrase is not None:
                    pending_audio, silence_start = self.pending_phrase
                    actual_pause = time.time() - silence_start
                    self.transcribe_queue.put((pending_audio, actual_pause))
                    self.pending_phrase = None
                    self.last_transcribe_time = time.time()
                self.silence_samples = 0

            # Check if we've hit a phrase boundary (silence duration exceeded)
            silence_duration = self.silence_samples / SAMPLE_RATE
            phrase_duration = sum(len(a) for a in self.phrase_audio) / SAMPLE_RATE

            if (silence_duration >= SILENCE_DURATION and
                phrase_duration >= MIN_PHRASE_DURATION and
                self.phrase_has_speech and
                self.pending_phrase is None and
                time.time() - self.last_transcribe_time > 0.5):
                # Mark phrase as pending - wait to see how long pause actually is
                self.pending_phrase = (self.phrase_audio[:], time.time())
                self.phrase_audio = []
                self.phrase_has_speech = False
                self.silence_samples = 0

            # Also check for max pause timeout (queue even if user hasn't resumed speaking)
            # Use punctuation threshold - no benefit waiting longer since we'd strip punct anyway
            if self.pending_phrase is not None:
                pending_audio, silence_start = self.pending_phrase
                pause_so_far = time.time() - silence_start
                if pause_so_far >= PAUSE_PUNCTUATION_THRESHOLD:
                    self.transcribe_queue.put((pending_audio, pause_so_far))
                    self.pending_phrase = None
                    self.last_transcribe_time = time.time()

    def streaming_transcribe_worker(self):
        """Background worker to transcribe phrases in streaming mode."""
        while self.recording or not self.transcribe_queue.empty():
            try:
                item = self.transcribe_queue.get(timeout=0.1)
            except Empty:
                continue

            # Handle both tuple (audio, silence_duration) and plain list formats
            if isinstance(item, tuple):
                audio_chunks, silence_duration = item
            else:
                audio_chunks = item
                silence_duration = SILENCE_DURATION  # Default, keep punctuation

            if not audio_chunks:
                continue

            audio = np.concatenate(audio_chunks, axis=0)
            duration = len(audio) / SAMPLE_RATE

            if duration < MIN_PHRASE_DURATION:
                continue

            # Save and transcribe
            temp_file = STATE_DIR / f"phrase_{time.time()}.wav"
            try:
                sf.write(str(temp_file), audio, SAMPLE_RATE)
                start = time.time()

                # Use previous transcription as context to help Whisper understand continuity
                # This helps it know "create" might continue with "me a picture"
                context_prompt = INITIAL_PROMPT
                if self.previous_text:
                    # Append recent text (last ~100 chars) to give context
                    recent_context = self.previous_text[-100:].strip()
                    context_prompt = f"{INITIAL_PROMPT}\n{recent_context}"

                segments, _ = self.model.transcribe(
                    str(temp_file),
                    beam_size=5,
                    language="en",
                    vad_filter=True,
                    initial_prompt=context_prompt,
                    condition_on_previous_text=False,  # We provide context via initial_prompt instead
                    repetition_penalty=1.1,  # Penalize repeated tokens
                    no_repeat_ngram_size=3,  # Prevent 3-gram repetitions
                )
                raw_text = " ".join(seg.text for seg in segments).strip()
                text = apply_replacements(raw_text)

                # Strip punctuation for:
                # 1. Intentional pauses (longer silence) - user pausing to paste/think
                # 2. Short phrases (1-2 words) - unlikely to be complete sentences
                word_count = len(text.split())
                should_strip = (silence_duration >= PAUSE_PUNCTUATION_THRESHOLD or
                               word_count <= 2)

                if should_strip:
                    text = strip_trailing_punctuation(text)
                    reason = "short phrase" if word_count <= 2 else "long pause"
                    log(f"Phrase ({duration:.1f}s, pause {silence_duration:.1f}s) transcribed in {time.time() - start:.2f}s: {text} [punct stripped: {reason}]")
                else:
                    log(f"Phrase ({duration:.1f}s, pause {silence_duration:.1f}s) transcribed in {time.time() - start:.2f}s: {text}")

                if text:
                    type_text(text + " ")  # Add space between phrases
                    # Update context for next phrase
                    self.previous_text = (self.previous_text + " " + text).strip()
                    # Keep context from getting too long
                    if len(self.previous_text) > 500:
                        self.previous_text = self.previous_text[-300:]
            except Exception as e:
                log(f"Phrase transcription error: {e}")
            finally:
                temp_file.unlink(missing_ok=True)

    def start_recording(self):
        """Start recording audio."""
        with self.lock:
            if self.recording:
                return "Already recording"

            self.audio_data = []
            self.phrase_audio = []
            self.silence_samples = 0
            self.pending_phrase = None
            self.phrase_has_speech = False
            self.previous_text = ""  # Reset context for new recording session
            self.recording = True

            self.stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype=np.float32,
                callback=self.audio_callback,
                device=AUDIO_DEVICE
            )
            self.stream.start()

            # Start streaming transcription worker if in streaming mode
            if self.streaming_mode:
                self.transcribe_thread = threading.Thread(
                    target=self.streaming_transcribe_worker,
                    daemon=True
                )
                self.transcribe_thread.start()

        mode_str = "streaming" if self.streaming_mode else "batch"
        log(f"Recording started ({mode_str} mode)")
        notify(f"🎤 Recording ({mode_str})...", "low")
        return "Recording"

    def stop_recording(self) -> str:
        """Stop recording and transcribe."""
        with self.lock:
            if not self.recording:
                return "Not recording"

            was_streaming = self.streaming_mode
            self.recording = False

            if self.stream:
                self.stream.stop()
                self.stream.close()
                self.stream = None

            if was_streaming:
                # Streaming mode: queue any remaining audio and wait for worker
                # First, handle any pending phrase
                if self.pending_phrase is not None:
                    pending_audio, silence_start = self.pending_phrase
                    # End of dictation = keep punctuation
                    self.transcribe_queue.put((pending_audio, 0))
                    self.pending_phrase = None
                # Then handle any audio accumulated since the pending phrase
                if self.phrase_audio:
                    # End of dictation = keep punctuation (use low silence duration)
                    self.transcribe_queue.put((self.phrase_audio[:], 0))
                    self.phrase_audio = []

                audio_data = []  # Already transcribed in real-time
            else:
                # Batch mode: get accumulated audio
                audio_data = self.audio_data
                self.audio_data = []

        if was_streaming:
            # Wait for transcription worker to finish
            if self.transcribe_thread and self.transcribe_thread.is_alive():
                self.transcribe_thread.join(timeout=10)
            log("Streaming recording stopped")
            notify("⏹️ Done", "low")
            return "Stopped"

        # Batch mode processing
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
                repetition_penalty=1.1,  # Penalize repeated tokens
                no_repeat_ngram_size=3,  # Prevent 3-gram repetitions
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

    def toggle_mode(self) -> str:
        """Toggle between batch and streaming mode. Can be called mid-recording."""
        with self.lock:
            was_streaming = self.streaming_mode
            self.streaming_mode = not self.streaming_mode
            mode_str = "streaming" if self.streaming_mode else "batch"

            if self.recording:
                if was_streaming and not self.streaming_mode:
                    # Streaming → Batch: flush pending and current phrase, stop worker
                    if self.pending_phrase is not None:
                        pending_audio, _ = self.pending_phrase
                        self.transcribe_queue.put((pending_audio, PAUSE_PUNCTUATION_THRESHOLD + 1))
                        self.pending_phrase = None
                    if self.phrase_audio:
                        # Mode switch = intentional, strip punctuation
                        self.transcribe_queue.put((self.phrase_audio[:], PAUSE_PUNCTUATION_THRESHOLD + 1))
                        self.phrase_audio = []
                    # Signal worker to finish and wait
                    if self.transcribe_thread and self.transcribe_thread.is_alive():
                        # Worker will exit when queue is empty and self.recording check
                        pass  # Let it finish naturally
                    self.audio_data = []  # Start fresh batch accumulation
                elif not was_streaming and self.streaming_mode:
                    # Batch → Streaming: transcribe accumulated audio, start streaming
                    if self.audio_data:
                        # Queue batch audio as one phrase, mode switch = intentional pause
                        self.transcribe_queue.put((self.audio_data[:], PAUSE_PUNCTUATION_THRESHOLD + 1))
                        self.audio_data = []
                    self.phrase_audio = []
                    self.silence_samples = 0
                    # Start streaming worker if not running
                    if not self.transcribe_thread or not self.transcribe_thread.is_alive():
                        self.transcribe_thread = threading.Thread(
                            target=self.streaming_transcribe_worker,
                            daemon=True
                        )
                        self.transcribe_thread.start()

        log(f"Mode changed to: {mode_str}")
        notify(f"Mode: {mode_str}", "normal")
        # Audio feedback: different sounds for each mode
        if self.streaming_mode:
            play_sound("message-new-instant")  # Brighter sound for streaming
        else:
            play_sound("audio-volume-change")  # Subtle click for batch
        return f"Mode: {mode_str}"

    def get_mode(self) -> str:
        """Get current mode."""
        return "streaming" if self.streaming_mode else "batch"

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
                mode = self.get_mode()
                state = "Recording" if self.recording else "Idle"
                response = f"{state} ({mode} mode)"
            elif data == "mode":
                response = self.toggle_mode()
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
    elif cmd == "mode":
        if not is_daemon_running():
            print("Daemon not running. Start with: dictate-daemon.py start")
            sys.exit(1)
        response = send_command("mode")
        print(response)
    elif cmd == "status":
        if is_daemon_running():
            response = send_command("status")
            print(f"Daemon running, state: {response}")
        else:
            print("Daemon not running")
    else:
        print(f"Usage: {sys.argv[0]} [start|stop|toggle|mode|status]")
        sys.exit(1)


if __name__ == "__main__":
    main()
