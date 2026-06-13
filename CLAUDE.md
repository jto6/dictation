# Whisper Dictation - Project Context

## Overview

This is a voice-to-text dictation solution for Linux that uses OpenAI's Whisper model (via faster-whisper) to transcribe speech and automatically type it at the cursor position.

## Architecture

The primary mode is **daemon mode** (`dictate-daemon.py`):

- Runs as a background process
- Keeps the Whisper model loaded in memory for instant response
- Communicates via Unix socket at `/tmp/whisper-dictation/daemon.sock`
- Commands: `start`, `stop`, `toggle`, `status`, `quit`

Alternative modes (less commonly used):

- `dictate-toggle.py` - Standalone toggle mode (loads model each time, slower)
- `dictate-ptt.py` - Push-to-talk using pynput (hold key to record)
- `dictate-evdev.py` - Push-to-talk using evdev (requires input group membership)

## Key Files

| File                        | Purpose                                                                   |
|-----------------------------|---------------------------------------------------------------------------|
| `dictate-daemon.py`         | Main daemon - handles recording, transcription, typing                    |
| `start-dictation-daemon.sh` | Wrapper that activates venv and sets up library paths                     |
| `install.sh`                | Full installation script - creates venv, installs deps, sets up shortcuts |
| `setup-shortcut.sh`         | GNOME-specific keyboard shortcut and autostart setup                      |

## Configuration

All configuration is at the top of `dictate-daemon.py`:

```python
MODEL_SIZE = "base.en"      # Whisper model: tiny.en, base.en, small.en, medium.en, large-v3
DEVICE = "cpu"              # "cpu" or "cuda"
COMPUTE_TYPE = "int8"       # "int8" for CPU, "float16" for GPU
INITIAL_PROMPT = "..."      # Vocabulary hints for Whisper
REPLACEMENTS = {...}        # Post-transcription text replacements
```

## Design Decisions & Watch Items

These are deliberate choices that aren't obvious from the code alone. They travel
with the repo so every machine running this tool shares the same context.

### GPU compute type: `int8_float16` (WATCH)

On CUDA the daemon prefers `int8_float16` over `float16` (see `detect_best_config`
and the CUDA load fallback chain in `dictate-daemon.py`).

- **Why:** with `float16`, `medium.en` silently dropped a spoken phrase
  ("effective at communicating") around a pause. `int8_float16` kept it at
  **identical speed**; `float32` (higher precision) *also* dropped it, so this is
  a borderline beam-search flip from the quantization path, not a precision ladder.
- **Caveat / what to watch:** this was validated on a **single** observed drop.
  It costs nothing in speed, so it's a safe default, but it is *not* proven
  universally better — a different clip could flip the other way. If word drops
  recur, revisit this choice.
- **How to diagnose a suspected drop:** the daemon saves both
  `last-recording-raw.wav` (original) and `last-recording.wav` (compressed) in
  `/tmp/whisper-dictation/`. Transcribe both with the **same** backend and diff;
  if they match, compression is not the cause. Then compare compute types
  (`float16` vs `int8_float16` vs `float32`) on the same audio — backend
  precision, not silence handling, has been the real culprit so far.

### Silence compression: only collapse genuinely long dead air

`compress_silence` leaves normal speech pauses untouched and only collapses
silence runs longer than `LONG_PAUSE_TRIGGER` (8s) down to `LONG_PAUSE_KEEP` (2s),
cutting from the middle so word edges are never clipped.

- **Why:** aggressively compressing every pause re-times natural speech and clips
  the quiet edges of real words, which measurably *degraded* transcription. Raw
  audio transcribed better than over-compressed audio in side-by-side tests.
- **History:** earlier versions added recovery/gap-fill re-transcription passes to
  repair this self-inflicted damage; those caused duplicate/hallucinated text and
  were removed. The single-pass conservative approach is intentional — do not
  reintroduce per-pause compression or secondary re-transcription passes.

## Runtime State

All runtime files are in `/tmp/whisper-dictation/`:

- `daemon.sock` - Unix socket for IPC
- `daemon.pid` - Daemon process ID
- `daemon.log` - Log file
- `recording.wav` - Temporary audio file (deleted after transcription)

## Dependencies

Core Python packages (installed via pip):

- `faster-whisper` - Whisper implementation using CTranslate2
- `sounddevice` - Audio recording
- `soundfile` - WAV file handling
- `numpy` - Audio data processing

GPU support (optional):

- `nvidia-cudnn-cu12` - CUDA Deep Neural Network library

System dependencies:

- `xdotool` - Types text at cursor position (X11)
- `wtype` - Alternative for Wayland
- `notify-send` - Desktop notifications

## Common Tasks

### Adding a new transcription replacement
Edit the `REPLACEMENTS` dict in `dictate-daemon.py` and restart the daemon.

### Changing the keyboard shortcut
Run `./setup-shortcut.sh '<New>shortcut'` or edit GNOME settings manually.

### Debugging transcription issues
Check `/tmp/whisper-dictation/daemon.log` for transcription output and timing.

### Testing without the daemon
Use `dictate-toggle.py` directly - it loads the model each time but is simpler to debug.

## Desktop Integration

The `.desktop` files use `__INSTALL_DIR__` as a placeholder that gets replaced with the actual installation path by `install.sh` or `setup-shortcut.sh` when copying to:

- `~/.config/autostart/` (for daemon autostart)
- `~/.local/share/applications/` (for desktop launchers)
