# Whisper Dictation

Voice-to-text dictation for Linux using Whisper with optional GPU acceleration. Press a keyboard shortcut to start recording, speak, press again to transcribe and auto-type the result at your cursor position.

## Features

- **Fast transcription** - ~0.3s for 50 seconds of audio with GPU (RTX 4070), or a few seconds on CPU
- **Toggle-based recording** - Press shortcut to start, press again to stop and transcribe
- **Daemon mode** - Keeps Whisper model in memory for instant response
- **Technical vocabulary** - Optimized for programming terms (git, Linux, Python, etc.)
- **Customizable replacements** - Fix common transcription errors
- **Desktop notifications** - Visual feedback for recording state
- **Auto-types result** - Text appears at cursor position via xdotool

## Requirements

- **Python 3.8+**
- **Linux** (tested on Ubuntu with GNOME)
- **xdotool** (for typing text) - `sudo apt install xdotool`
- **libnotify** (for notifications) - usually pre-installed
- **Audio input device** (microphone)

### For GPU Acceleration (Optional)

- **NVIDIA GPU** with CUDA support
- **NVIDIA driver** (580+ recommended)

## Installation

### Quick Install

```bash
# Clone the repository
git clone https://github.com/yourusername/dictation.git
cd dictation

# Run the installer
./install.sh
```

The install script automatically:
- Creates a Python virtual environment
- Installs dependencies (faster-whisper, sounddevice, etc.)
- Detects NVIDIA GPU and installs cuDNN for acceleration
- Configures GPU or CPU mode appropriately
- Sets up keyboard shortcut (GNOME) - **Super+W** by default
- Configures autostart so the daemon starts on login

### Manual Install

If you prefer manual installation:

```bash
cd dictation

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install faster-whisper sounddevice soundfile numpy

# For GPU support (NVIDIA only - skip for CPU-only)
pip install nvidia-cudnn-cu12

# Make scripts executable
chmod +x *.py *.sh

# Run setup script (GNOME desktop)
./setup-shortcut.sh
```

## Usage

### Basic Usage

1. **Press Super+W** - Recording starts (notification appears)
2. **Speak** your message
3. **Press Super+W** again - Stops, transcribes, and types at cursor

The first toggle after login takes ~2-3 seconds (loads the Whisper model). Subsequent uses are instant.

### Manual Commands

```bash
# Start daemon manually
./start-dictation-daemon.sh start

# Toggle recording
./start-dictation-daemon.sh toggle

# Check status
./start-dictation-daemon.sh status

# Stop daemon
./start-dictation-daemon.sh stop
```

## Customization

### Custom Keyboard Shortcut

```bash
# Use a different shortcut
./setup-shortcut.sh '<Ctrl><Alt>d'    # Ctrl+Alt+D
./setup-shortcut.sh '<Super>F9'       # Super+F9
```

### Fix Transcription Errors

Edit `dictate-daemon.py` and add to the `REPLACEMENTS` dictionary:

```python
REPLACEMENTS = {
    # Existing replacements...
    "get commit": "git commit",
    "lennox": "Linux",

    # Add your own:
    "wrong phrase": "correct phrase",
}
```

After editing, restart the daemon:
```bash
./start-dictation-daemon.sh stop
./start-dictation-daemon.sh start
```

### Add Technical Vocabulary

Edit the `INITIAL_PROMPT` in `dictate-daemon.py` to add terms Whisper should recognize:

```python
INITIAL_PROMPT = """
git commit --message, git push, Linux, Ubuntu, Python,
# Add your terms here:
your_project_name, your_framework, your_terminology,
"""
```

### Switch Between GPU and CPU

Edit `dictate-daemon.py`:

```python
# For GPU (faster, requires nvidia-cudnn-cu12):
DEVICE = "cuda"
COMPUTE_TYPE = "float16"

# For CPU (slower, more compatible):
DEVICE = "cpu"
COMPUTE_TYPE = "int8"
```

### Change Whisper Model Size

Edit `dictate-daemon.py`:

```python
# Options: tiny.en, base.en, small.en, medium.en, large-v3
# Larger = more accurate but slower and more VRAM
MODEL_SIZE = "base.en"  # Good balance of speed/accuracy
```

## How It Works

### Architecture

```
[Keyboard Shortcut] → [start-dictation-daemon.sh toggle]
                              ↓
                    [dictate-daemon.py]
                              ↓
              ┌───────────────┴───────────────┐
              ↓                               ↓
        [Recording]                    [Transcription]
     (sounddevice)                   (faster-whisper)
              ↓                               ↓
        [Audio file]                  [Apply replacements]
                                              ↓
                                      [Type with xdotool]
```

### Daemon Mode

The daemon (`dictate-daemon.py`) runs as a background process:
- Listens on Unix socket `/tmp/whisper-dictation/daemon.sock`
- Keeps Whisper model loaded in memory (~500MB-1GB)
- Responds to toggle/start/stop/status commands

### Autostart

A `.desktop` file is installed to `~/.config/autostart/`:
- Daemon starts automatically on GNOME login
- Model is pre-loaded before you need it
- First dictation is instant (no loading delay)

To disable autostart:
```bash
rm ~/.config/autostart/dictate-daemon-start.desktop
```

## File Structure

```
dictation/
├── install.sh                   # Full auto-install script
├── setup-shortcut.sh            # GNOME keyboard/autostart setup
├── dictate-daemon.py            # Main daemon (keeps model loaded)
├── dictate-toggle.py            # Standalone toggle (slower, loads model each time)
├── dictate-ptt.py               # Push-to-talk mode using pynput
├── dictate-evdev.py             # Push-to-talk mode using evdev
├── start-dictation-daemon.sh    # Wrapper script for daemon mode
├── start-dictation-toggle.sh    # Wrapper for standalone mode
├── start-dictation.sh           # Wrapper for ptt mode
├── start-dictation-evdev.sh     # Wrapper for evdev mode
├── dictate-toggle.desktop       # Desktop launcher for toggle
├── dictate-daemon-start.desktop # Desktop launcher to start daemon
└── venv/                        # Python virtual environment (created by install)
```

## Troubleshooting

### No audio captured
```bash
# Test microphone
arecord -d 3 test.wav && aplay test.wav

# Check audio devices
python3 -c "import sounddevice; print(sounddevice.query_devices())"
```

### Daemon not responding
```bash
# Check logs
tail -f /tmp/whisper-dictation/daemon.log

# Kill and restart
pkill -f dictate-daemon
./start-dictation-daemon.sh start
```

### GPU/CUDA errors
```bash
# Check NVIDIA driver
nvidia-smi

# Verify cuDNN installed
pip show nvidia-cudnn-cu12

# Fall back to CPU (edit dictate-daemon.py)
DEVICE = "cpu"
COMPUTE_TYPE = "int8"
```

### Shortcut not working
- Check Settings → Keyboard → Custom Shortcuts
- Look for "Dictate Toggle"
- Verify no conflicts with existing shortcuts

### Text not typing
```bash
# Test xdotool
xdotool type "hello world"

# For Wayland, you may need wtype instead
sudo apt install wtype
```

## Logs

All logs are written to `/tmp/whisper-dictation/daemon.log`:
```bash
tail -f /tmp/whisper-dictation/daemon.log
```

Example output:
```
[22:55:40.838] Command: toggle
[22:55:40.851] Captured 49.8s of audio
[22:55:41.167] Transcribed in 0.30s: git commit --message "fix bug"
[22:55:41.259] Typed: git commit --message "fix bug"...
```

## License

See [LICENSE](LICENSE) file.
