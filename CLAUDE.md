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

### Mic gain normalization: `normalize_audio` (WATCH)

Before silence detection and Whisper transcription, `normalize_audio` scales the
audio up if the peak amplitude is below `NORMALIZE_THRESHOLD` (0.3). Target peak
after scaling is `NORMALIZE_TARGET` (0.7).

- **Why:** a quiet mic (peak ~0.038, -28 dBFS) has speech RMS well below
  `SILENCE_THRESHOLD` (0.01), so `compress_silence` classifies 99.6% of a real
  recording as dead air, triggers `LONG_PAUSE_TRIGGER`, and collapses the entire
  session to 2 seconds. `normalize_audio` brings the signal to a usable level
  before either check runs.
- **What to watch:** normalization amplifies background noise equally with speech.
  If transcription quality degrades on a noisy-but-quiet mic (hiss becomes
  prominent), lower `NORMALIZE_TARGET` or add a noise gate. On a clean-but-quiet
  mic (laptop/headset at max volume) this is a net win.
- **last-recording-raw.wav** always stores the pre-normalization signal so the
  true mic level is preserved for diagnostics.

### Window-end truncation: batch mode uses `vad_filter=True`

Batch mode transcribes with VAD enabled and deliberately gentle VAD parameters
(`VAD_THRESHOLD` 0.3, `VAD_SPEECH_PAD_MS` 600, `VAD_MIN_SILENCE_MS` 2000 — all
less aggressive than faster-whisper's 0.5 / 400ms defaults).

- **Why:** Whisper decodes in 30-second windows. It can truncate a window early —
  emitting a closing timestamp at the window edge after transcribing only part of
  it. faster-whisper sees `single_timestamp_ending` and does `seek += segment_size`
  (`transcribe.py:1071`), consuming the whole window and *skipping* the
  untranscribed remainder; the word-timestamp seek correction at line 1288 is
  explicitly bypassed in that branch. Several seconds of real speech vanish with
  no error, and `avg_logprob` stays excellent (-0.08 observed), so temperature
  fallback never triggers. Observed twice in one 124s recording: 6s and 13s of
  speech dropped, both exactly at a window boundary.
- **Evidence:** re-transcribing the same audio at 8 different window alignments
  (prepending 0–4.9s of silence), scoring 6 known phrases per run:
  `vad_filter=False` as shipped scored 37/48; `condition_on_previous_text=False`
  38/48; dropping the initial prompt 44/48; `vad_filter=True` **48/48** with zero
  stretched words. The non-VAD variants only ever moved the drop to a different
  spot — they are alignment luck, not fixes.
- **Why this does not contradict "don't discard audio":** the old
  `vad_filter=False` comment assumed VAD would eat real speech. At these settings
  it only removed genuine >1.5s pauses on the test recording and recovered *more*
  speech than the non-VAD path. `normalize_audio` already runs first, so VAD sees
  a healthy signal. If VAD still returns nothing, the daemon logs
  "VAD returned no speech" and retries with `vad_filter=False`.
- **How to spot a recurrence:** a truncation leaves the last decoded word
  stretched across the skipped audio. The daemon logs
  `WARNING: '<word>' spans ...s - speech was likely dropped here` for any word
  lasting `LONG_WORD_WARN` (3s) or more. That warning in `daemon.log` is the only
  trace this failure leaves — a 13.6s "it's" was the fingerprint that found it.

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
