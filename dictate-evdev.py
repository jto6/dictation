#!/usr/bin/env python3
"""
Push-to-talk Whisper dictation using evdev for reliable keyboard detection.
Hold the hotkey to record, release to transcribe and type.
"""

import subprocess
import sys
import tempfile
import threading
import signal
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf
import evdev
from evdev import ecodes

from faster_whisper import WhisperModel

# Configuration
SAMPLE_RATE = 16000
CHANNELS = 1
MODEL_SIZE = "base.en"
DEVICE = "cuda"  # Change to "cpu" if no GPU
COMPUTE_TYPE = "float16"  # Change to "float32" for CPU

# Hotkey: Ctrl+Shift+D
HOTKEY_CTRL = {ecodes.KEY_LEFTCTRL, ecodes.KEY_RIGHTCTRL}
HOTKEY_SHIFT = {ecodes.KEY_LEFTSHIFT, ecodes.KEY_RIGHTSHIFT}
HOTKEY_D = ecodes.KEY_D


def find_keyboard():
    """Find the main keyboard device."""
    for path in evdev.list_devices():
        try:
            device = evdev.InputDevice(path)
            caps = device.capabilities()
            if ecodes.EV_KEY in caps:
                keys = caps[ecodes.EV_KEY]
                # Must have alphabet keys and modifiers
                if (ecodes.KEY_A in keys and
                    ecodes.KEY_LEFTCTRL in keys and
                    ecodes.KEY_LEFTSHIFT in keys and
                    'Mouse' not in device.name and
                    'Consumer' not in device.name and
                    'System' not in device.name):
                    return device
        except Exception:
            continue
    return None


class PushToTalkDictation:
    def __init__(self):
        print(f"Loading Whisper model '{MODEL_SIZE}'...")
        self.model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
        print("Model loaded!")

        self.recording = False
        self.audio_data = []
        self.stream = None
        self.pressed_keys = set()
        self.hotkey_active = False
        self.running = True

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
        print("Recording...")

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

    def check_hotkey(self):
        """Check if Ctrl+Shift+D is currently pressed."""
        has_ctrl = bool(self.pressed_keys & HOTKEY_CTRL)
        has_shift = bool(self.pressed_keys & HOTKEY_SHIFT)
        has_d = HOTKEY_D in self.pressed_keys
        return has_ctrl and has_shift and has_d

    def handle_key_event(self, event):
        if event.type != ecodes.EV_KEY:
            return

        key = event.code

        # Track key state
        if event.value == 1:  # Key press
            self.pressed_keys.add(key)
        elif event.value == 0:  # Key release
            self.pressed_keys.discard(key)

        # Check hotkey state
        hotkey_pressed = self.check_hotkey()

        if hotkey_pressed and not self.hotkey_active:
            # Hotkey just activated
            self.hotkey_active = True
            self.start_recording()
        elif not hotkey_pressed and self.hotkey_active:
            # Hotkey just released
            self.hotkey_active = False
            audio = self.stop_recording()

            if audio is not None and len(audio) > 0:
                print("Transcribing...")
                text = self.transcribe(audio)
                if text:
                    print(f"Result: {text}")
                    self.type_text(text)
                else:
                    print("No speech detected")

    def run(self):
        print("\n=== Push-to-Talk Dictation (evdev) ===")
        print("Hold Ctrl+Shift+D to record")
        print("Release to transcribe and type")
        print("Press Ctrl+C to exit\n")

        keyboard = find_keyboard()
        if not keyboard:
            print("ERROR: No keyboard device found!")
            print("Make sure you're in the 'input' group: sudo usermod -aG input $USER")
            sys.exit(1)

        print(f"Using keyboard: {keyboard.name}")
        print(f"Device: {keyboard.path}\n")

        # Handle Ctrl+C gracefully
        def signal_handler(sig, frame):
            print("\nExiting...")
            self.running = False
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        try:
            for event in keyboard.read_loop():
                if not self.running:
                    break
                self.handle_key_event(event)
        except KeyboardInterrupt:
            print("\nExiting...")


if __name__ == "__main__":
    ptt = PushToTalkDictation()
    ptt.run()
