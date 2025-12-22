#!/usr/bin/env python3
"""
Push-to-talk Whisper dictation using pynput for global hotkeys.
Hold the hotkey to record, release to transcribe and type.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

try:
    from pynput import keyboard
except ImportError:
    print("pynput not installed. Run:")
    print("  source ~/whisper-dictation/venv/bin/activate && pip install pynput")
    sys.exit(1)

from faster_whisper import WhisperModel

# Configuration
SAMPLE_RATE = 16000
CHANNELS = 1
MODEL_SIZE = "base.en"
DEVICE = "cuda"  # Change to "cpu" if no GPU
COMPUTE_TYPE = "float16"  # Change to "float32" for CPU

# Hotkey: Ctrl+Shift+D
HOTKEY = {keyboard.Key.ctrl, keyboard.Key.shift, keyboard.KeyCode.from_char('d')}


class PushToTalkDictation:
    def __init__(self):
        print(f"Loading Whisper model '{MODEL_SIZE}'...")
        self.model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
        print("Model loaded!")

        self.recording = False
        self.audio_data = []
        self.stream = None
        self.current_keys = set()
        self.hotkey_pressed = False

    def audio_callback(self, indata, frames, time_info, status):
        if self.recording:
            self.audio_data.append(indata.copy())

    def start_recording(self):
        if self.recording:
            return

        self.audio_data = []
        self.recording = True
        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=np.float32,
            callback=self.audio_callback
        )
        self.stream.start()
        print("🎤 Recording...")

    def stop_recording(self):
        if not self.recording:
            return None

        self.recording = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

        if not self.audio_data:
            return None

        return np.concatenate(self.audio_data, axis=0)

    def transcribe(self, audio):
        if audio is None or len(audio) == 0:
            return ""

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio, SAMPLE_RATE)
            temp_path = f.name

        try:
            segments, _ = self.model.transcribe(
                temp_path,
                beam_size=5,
                language="en",
                vad_filter=True,
            )
            return " ".join(seg.text for seg in segments).strip()
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def type_text(self, text):
        if not text:
            return
        try:
            subprocess.run(["xdotool", "type", "--clearmodifiers", "--", text], check=True)
        except FileNotFoundError:
            try:
                subprocess.run(["wtype", text], check=True)
            except FileNotFoundError:
                print(f"Output: {text}")

    def on_press(self, key):
        self.current_keys.add(key)

        if HOTKEY.issubset(self.current_keys) and not self.hotkey_pressed:
            self.hotkey_pressed = True
            self.start_recording()

    def on_release(self, key):
        if self.hotkey_pressed and key in HOTKEY:
            self.hotkey_pressed = False
            audio = self.stop_recording()

            if audio is not None and len(audio) > 0:
                print("📝 Transcribing...")
                text = self.transcribe(audio)
                if text:
                    print(f"✅ {text}")
                    self.type_text(text)
                else:
                    print("No speech detected")

        self.current_keys.discard(key)

    def run(self):
        print("\n=== Push-to-Talk Dictation ===")
        print("Hold Ctrl+Shift+D to record")
        print("Release to transcribe and type")
        print("Press Ctrl+C to exit\n")

        with keyboard.Listener(on_press=self.on_press, on_release=self.on_release) as listener:
            try:
                listener.join()
            except KeyboardInterrupt:
                print("\nExiting...")


if __name__ == "__main__":
    ptt = PushToTalkDictation()
    ptt.run()
