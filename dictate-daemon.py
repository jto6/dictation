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
import re
import shlex
import sys
import socket
import signal
import subprocess
import tempfile
import time
import unicodedata
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
MODEL_SIZE = "medium.en"  # "auto", "tiny.en", "base.en", "small.en", "medium.en", "large-v3"
DEVICE = "auto"  # "auto", "cuda", or "cpu"
COMPUTE_TYPE = "auto"  # "auto", "int8_float16", "float16", "float32", "int8" (auto picks int8_float16 on GPU)

# Audio input device - set to None for system default, or device name/index
# Use `python -c "import sounddevice; print(sounddevice.query_devices())"` to list devices
# Examples: None, 0, "pipewire", "HDA Intel PCH"
AUDIO_DEVICE = None

# Streaming mode settings
SILENCE_THRESHOLD = 0.01  # RMS threshold for silence detection
SILENCE_DURATION = 0.7    # Seconds of silence to trigger phrase transcription
MIN_PHRASE_DURATION = 0.3 # Minimum audio duration to transcribe

# Mic gain normalization. Quiet mics (peak < NORMALIZE_THRESHOLD) have their
# speech RMS fall below SILENCE_THRESHOLD, which makes compress_silence treat
# an entire recording as dead air and collapse it. Normalize before processing
# when the peak is below this fraction of full scale.
NORMALIZE_THRESHOLD = 0.3  # apply gain if peak amplitude is below this
NORMALIZE_TARGET = 0.7     # scale peak up to this value

# Long-pause compression (batch mode). Normal speech pauses (the natural rhythm
# between phrases and sentences) are left COMPLETELY untouched — re-timing them
# degrades transcription, since Whisper relies on natural cadence and a loudness
# threshold clips the quiet edges of real words. Only genuinely long dead-air
# gaps (you stepped away, stopped to read something) are collapsed, because
# Whisper can hallucinate text over long stretches of silence. We cut only from
# the MIDDLE of such a gap, keeping a generous margin of real silence on each
# side, so no word edge is ever touched and no click is introduced.
LONG_PAUSE_TRIGGER = 8.0  # only compress silence runs LONGER than this (seconds)
LONG_PAUSE_KEEP = 2.0     # collapse such runs down to this many seconds

# Adaptive silence cutoff for batch compression. A FIXED RMS cutoff mis-fires on
# quiet mics: even after normalization, the quiet onset/tail of a real word can
# dip below the cutoff, so those windows get absorbed into an adjacent silence
# run. That over-extends the run past LONG_PAUSE_TRIGGER and lets the middle cut
# clip real speech (observed: "Roth conversions. You can determine" → "Rothcon.
# The"). Instead, derive the cutoff from each recording's OWN speech level —
# SILENCE_REL_FRACTION of a high RMS percentile — so it scales with mic gain.
# Clamp it so it never rises above the old fixed value (never MORE aggressive than
# before) and never falls below a small floor (so a near-silent clip, where the
# percentile lands in noise, still has its genuine dead air collapsed).
SILENCE_REL_PERCENTILE = 95     # percentile of window RMS taken as the speech level
SILENCE_REL_FRACTION = 0.08     # silence cutoff = this fraction of that speech level
SILENCE_THRESHOLD_FLOOR = 0.0025  # cutoff is never clamped below this
PAUSE_PUNCTUATION_THRESHOLD = 1.5  # Silence longer than this = intentional pause, strip punctuation

# Whisper prompt to bias toward technical/programming vocabulary
INITIAL_PROMPT = """
git commit --message, git push, git pull, git checkout, git branch,
Linux, Ubuntu, Python, JavaScript, TypeScript, Bash, Docker, Kubernetes,
npm install, pip install, sudo apt, Claude Code, API, JSON, YAML, SQL,
pytest, mypy, eslint, webpack, venv, virtualenv, conda,
"""

# Post-processing replacements: {"wrong": "correct"}
# Add your custom replacements here.
#
# Matching is anchored per-side (see apply_replacements) so a key only replaces a
# whole token, not a fragment inside another word. The anchor chosen for each edge
# depends on that edge's character, because regex \b only marks a word/non-word
# transition and therefore can only hinge on a word character:
#   - word-char edge   (e.g. "jason", "...lint")   -> \b   (blocks "jasonville")
#   - punctuation edge (e.g. "D.O.S.", "- -help")  -> whitespace/string-edge lookaround
#   - whitespace edge  (key's own delimiter, e.g. "slash ") -> no anchor; the literal
#     space already is the boundary and consuming it is intentional.
# So when adding a key, its leading/trailing character determines how it is bounded.
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
    "npack": "NPAC",
    "SDV plus": "SDV+",
    "TI-OVX": "TIOVX",
    "CAS": "kas",
    "Piali": "Pyali",
    "Maren": "Marin",
    "Autosaur": "Autosar",
    "RP message": "RPMsg",
    "Escort": "S-Core"
}

# Post-processing for batch transcription. Empty list = raw (no rewrite).
# The value is an argv list passed to subprocess; the first token must be on
# POST_CMD_SAFELIST to prevent the IPC channel from becoming a generic exec.
DEFAULT_BATCH_POST_CMD: list = []

POST_CMD_SAFELIST = {"text-clean", "bulletize"}

POST_TIMEOUT_SEC = 120  # fall back to raw if wrapper exceeds this


def _read_claude_api_env() -> tuple[dict, str]:
    """Read API env vars and node bin dir from ~/.claude/settings.json.

    Returns (env_dict, node_bin_dir). Both may be empty/empty-string if the
    file is absent or malformed.
    """
    settings_path = Path.home() / ".claude" / "settings.json"
    try:
        with open(settings_path) as f:
            settings = json.load(f)
    except (OSError, ValueError):
        return {}, ""

    env = {k: v for k, v in settings.get("env", {}).items() if isinstance(v, str)}

    node_bin_dir = ""
    helper = settings.get("apiKeyHelper", "")
    if helper:
        try:
            node_exe = shlex.split(helper)[0]
            node_bin_dir = str(Path(node_exe).parent)
        except (ValueError, IndexError):
            pass

    return env, node_bin_dir


# Common Whisper hallucinations (typically appear at end of transcription)
# These are artifacts from YouTube training data
# Common Whisper hallucination patterns (typically at end of transcription).
# These are regex patterns matched case-insensitively against the end of the text.
# Each pattern can optionally match trailing filler like ", and have a great day."
_HALLUCINATION_TAIL = r"(?:[,.!]?\s*and\s+.{0,40})?[.!]?"
HALLUCINATION_PATTERNS = [
    r"thanks?\s+(?:you\s+)?(?:so much\s+)?for\s+watching" + _HALLUCINATION_TAIL,
    r"thanks?\s+(?:you\s+)?(?:so much\s+)?for\s+(?:reading|viewing)" + _HALLUCINATION_TAIL,
    r"thanks?\s+(?:you\s+)?(?:so much\s+)?for\s+listening" + _HALLUCINATION_TAIL,
    r"(?:please\s+)?(?:don'?t\s+forget\s+to\s+)?(?:like\s+and\s+)?subscribe" + _HALLUCINATION_TAIL,
    r"see\s+you(?:\s+\w+){0,2}\s+(?:in\s+the\s+)?next\s+(?:video|time|one)" + _HALLUCINATION_TAIL,
    r"\bbye[\s-]*bye" + _HALLUCINATION_TAIL,
    r"\bbye[.!]?",
    r"thank\s+you\s+for\s+(?:your\s+)?\w+(?:\s+\w+){0,1}" + r"[.!]?",
    r"thank\s+you\s+very\s+much" + _HALLUCINATION_TAIL,
    r"thank\s+you" + _HALLUCINATION_TAIL,
    r"and\s+that'?s\s+it" + _HALLUCINATION_TAIL,
    r"that'?s\s+it" + _HALLUCINATION_TAIL,
    r"that'?s\s+all\s+for\s+(?:today|now)" + _HALLUCINATION_TAIL,
    r"that'?s\s+all" + _HALLUCINATION_TAIL,
    r"i'?ll\s+see\s+you(?:\s+\w+){0,2}\s+(?:in\s+the\s+)?next\s+(?:video|one)" + _HALLUCINATION_TAIL,
    r"have\s+a\s+(?:great|good|nice|wonderful)\s+(?:day|one)" + _HALLUCINATION_TAIL,
    r"(?:in\s+the\s+)?description\s+(?:of\s+this\s+|below\s+)?video" + _HALLUCINATION_TAIL,
    r"(?:check\s+(?:out\s+)?)?(?:the\s+)?links?\s+in\s+the\s+description" + _HALLUCINATION_TAIL,
]
_HALLUCINATION_RE = [re.compile(p + r"[\s.]*$", re.IGNORECASE) for p in HALLUCINATION_PATTERNS]

# Paths
STATE_DIR = Path("/tmp/whisper-dictation")
SOCKET_PATH = STATE_DIR / "daemon.sock"
PID_FILE = STATE_DIR / "daemon.pid"
LOG_FILE = STATE_DIR / "daemon.log"
AUDIO_FILE = STATE_DIR / "recording.wav"
LAST_AUDIO_FILE = STATE_DIR / "last-recording.wav"        # compressed+padded audio sent to Whisper
LAST_RAW_AUDIO_FILE = STATE_DIR / "last-recording-raw.wav"  # original uncompressed audio (for debugging)
POST_CMD_FILE = STATE_DIR / "batch-post-cmd"


def normalize_whitespace(text: str) -> str:
    """Collapse multiple spaces into single spaces."""
    return re.sub(r' {2,}', ' ', text)


# ydotool's `type` looks each character up in a 128-entry keycode table indexed
# by a *signed* char, so any byte >= 0x80 indexes before the array and emits
# arbitrary keystrokes. Fold to ASCII first. This applies only to the ydotool
# path — clipboard and emacsclient inserts handle UTF-8 fine and keep the
# typographic punctuation that text-clean produces.
ASCII_PUNCT_MAP = {
    '‘': "'", '’': "'", '‚': "'", '‛': "'",
    '“': '"', '”': '"', '„': '"', '‟': '"',
    '′': "'", '″': '"',
    '‐': '-', '‑': '-', '‒': '-', '–': '-',
    '—': '-', '―': '-', '−': '-',
    '…': '...', '•': '*', ' ': ' ', '⁄': '/',
}
_ASCII_PUNCT_TABLE = str.maketrans(ASCII_PUNCT_MAP)


def to_ascii(text: str) -> str:
    """Fold text to pure ASCII so ydotool cannot emit out-of-range keycodes."""
    text = text.translate(_ASCII_PUNCT_TABLE)
    if text.isascii():
        return text
    # Strip accents (é -> e), then replace anything still outside ASCII so a
    # dropped character is visible rather than silently corrupting keystrokes.
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = ''.join(c for c in decomposed if not unicodedata.combining(c))
    return ''.join(c if c.isascii() else '?' for c in stripped)


NSP_TRAILING_THRESHOLD = 0.80  # drop trailing segments with no_speech_prob >= this
NSP_MAX_DROP_DURATION = 2.0   # never drop segments longer than this (real speech, not filler)

def normalize_audio(audio: np.ndarray) -> tuple:
    """Apply gain if the recording is too quiet for silence detection to work.

    Returns (normalized_audio, gain_applied). gain_applied is 1.0 when no
    adjustment was made.
    """
    peak = float(np.max(np.abs(audio)))
    if peak < NORMALIZE_THRESHOLD and peak > 0:
        gain = NORMALIZE_TARGET / peak
        return (audio * gain).astype(audio.dtype), gain
    return audio, 1.0


def compress_silence(audio: np.ndarray) -> tuple:
    """Collapse only genuinely long dead-air gaps; leave normal speech untouched.

    A silence run is compressed only if it is longer than LONG_PAUSE_TRIGGER
    (i.e. you stepped away or stopped to read), in which case it is shortened to
    LONG_PAUSE_KEEP seconds.  Every shorter pause — the natural rhythm between
    phrases and sentences — is passed through verbatim, because re-timing normal
    speech degrades transcription (Whisper relies on natural cadence, and the
    RMS threshold clips the quiet edges of real words).

    When a long gap is collapsed we keep half of LONG_PAUSE_KEEP at each end and
    drop only the middle.  The kept margins (~1s each side) far exceed any word's
    quiet onset/offset, so no speech edge is ever touched and the cut sits in
    truly silent audio (no click).

    The silence cutoff is adaptive: derived from this recording's own speech level
    (see SILENCE_REL_* constants) rather than a fixed RMS value, so quiet-mic word
    edges aren't misclassified as silence and absorbed into a run.

    Returns (compressed_audio, seconds_removed, compression_events, cutoff) where
    each event is {raw_start, raw_end, comp_pos, removed} (seconds): raw_start/
    raw_end bracket the gap in the original audio, comp_pos is where the shortened
    gap begins in the compressed audio, and removed is how much silence was
    dropped. cutoff is the adaptive RMS threshold used (for logging/diagnostics).
    """
    window = int(0.02 * SAMPLE_RATE)  # 20ms analysis windows
    trigger = int(LONG_PAUSE_TRIGGER * SAMPLE_RATE)
    keep = int(LONG_PAUSE_KEEP * SAMPLE_RATE)
    n_windows = len(audio) // window
    if n_windows == 0:
        return audio, 0.0, [], SILENCE_THRESHOLD

    rms = np.array([
        np.sqrt(np.mean(audio[i * window:(i + 1) * window] ** 2))
        for i in range(n_windows)
    ])
    # Adaptive cutoff: a small fraction of the recording's speech level, clamped
    # so it is never more aggressive than the old fixed cutoff and never collapses
    # below a floor (which would let mic noise read as speech in a silent clip).
    speech_level = float(np.percentile(rms, SILENCE_REL_PERCENTILE))
    cutoff = float(np.clip(
        speech_level * SILENCE_REL_FRACTION,
        SILENCE_THRESHOLD_FLOOR,
        SILENCE_THRESHOLD,
    ))
    is_silent = rms < cutoff

    kept = []
    compression_events = []
    output_samples = 0
    i = 0
    while i < n_windows:
        if is_silent[i]:
            j = i
            while j < n_windows and is_silent[j]:
                j += 1
            silent_samples = (j - i) * window
            run_start = i * window
            run_end = j * window
            if silent_samples > trigger:
                # Long dead-air gap: keep keep/2 at each end, drop the middle.
                head = keep // 2
                tail = keep - head
                chunk = np.concatenate([
                    audio[run_start: run_start + head],
                    audio[run_end - tail: run_end],
                ])
                compression_events.append({
                    "raw_start": run_start / SAMPLE_RATE,
                    "raw_end": run_end / SAMPLE_RATE,
                    "comp_pos": output_samples / SAMPLE_RATE,
                    "removed": (silent_samples - keep) / SAMPLE_RATE,
                })
            else:
                # Normal pause — pass through untouched.
                chunk = audio[run_start: run_end]
            kept.append(chunk)
            output_samples += len(chunk)
            i = j
        else:
            chunk = audio[i * window: (i + 1) * window]
            kept.append(chunk)
            output_samples += len(chunk)
            i += 1

    remainder = audio[n_windows * window:]
    if len(remainder) > 0:
        kept.append(remainder)

    compressed = np.concatenate(kept) if kept else audio.copy()
    removed = (len(audio) - len(compressed)) / SAMPLE_RATE
    return compressed, removed, compression_events, cutoff


def drop_trailing_high_nsp(segments: list) -> list:
    """Drop trailing segments with high no_speech_prob (Whisper hallucinations).

    When speech ends and there is trailing silence or noise, Whisper often
    emits short filler segments like 'Thanks.' or 'All right.' with elevated
    no_speech_prob values.  Strip them from the tail, but keep at least one
    segment so we never discard a single-segment transcription.

    Only drop segments that are short — long segments are real speech even if
    Whisper assigns a high no_speech_prob.
    """
    cut = len(segments)
    while cut > 1:
        seg = segments[cut - 1]
        nsp = getattr(seg, 'no_speech_prob', 0.0)
        duration = seg.end - seg.start
        if nsp >= NSP_TRAILING_THRESHOLD and duration <= NSP_MAX_DROP_DURATION:
            # If the preceding segment ends mid-sentence, this segment likely
            # completes it — preserve it regardless of nsp.
            prev_text = segments[cut - 2].text.strip()
            if prev_text and prev_text[-1] not in '.!?':
                break
            cut -= 1
        else:
            break
    if cut < len(segments):
        dropped = " ".join(s.text.strip() for s in segments[cut:])
        log(f"Dropped {len(segments) - cut} trailing high-nsp segment(s): '{dropped}'")
        return segments[:cut]
    return segments


def drop_trailing_echo(segments: list) -> list:
    """Drop the final segment if it's a fuzzy echo of the previous segment's tail.

    Whisper sometimes hallucinates a short trailing segment that echoes the
    end of the main transcription with slight word variations (e.g., "4th"
    becomes "3rd").  These are too different for exact overlap removal but
    are clearly not real speech.

    Heuristic: if the last segment is short (< 3s) and > 60% of its unique
    words appear in the last 12 words of the previous segment, drop it.
    """
    if len(segments) < 2:
        return segments
    last = segments[-1]
    prev = segments[-2]
    # Only consider short trailing segments
    if (last.end - last.start) > 3.0:
        return segments
    def norm_words(text):
        return [re.sub(r'[^\w]', '', w.lower()) for w in text.split() if re.sub(r'[^\w]', '', w.lower())]
    last_words = set(norm_words(last.text))
    prev_tail = set(norm_words(prev.text)[-12:])
    if not last_words:
        return segments
    overlap_ratio = len(last_words & prev_tail) / len(last_words)
    if overlap_ratio >= 0.6:
        log(f"Dropped trailing echo segment ({overlap_ratio:.0%} word overlap): '{last.text.strip()}'")
        return segments[:-1]
    return segments


def remove_segment_overlaps(seg_texts: list) -> list:
    """Remove content repeated across adjacent segment boundaries.

    Whisper sometimes ends a segment mid-phrase and repeats the tail words at
    the start of the next segment.  For each consecutive pair we find the
    longest suffix of seg[i] (up to 30 words) that matches the prefix of
    seg[i+1] and strip that prefix.

    Also handles the case where an entire segment is a duplicate of content
    already present in the previous segment (Whisper occasionally re-emits a
    full sentence when the audio has trailing silence).
    """
    def norm_words(text):
        """Return list of (normalized, original_token) for each word."""
        result = []
        for tok in text.split():
            w = re.sub(r'[^\w]', '', tok.lower())
            if w:
                result.append((w, tok))
        return result

    result = list(seg_texts)
    for i in range(len(result) - 1):
        prev = norm_words(result[i])
        nxt  = norm_words(result[i + 1])
        if not prev or not nxt:
            continue

        # Check if nxt is entirely contained at the end of prev (full duplicate).
        nxt_words = [w for w, _ in nxt]
        prev_words = [w for w, _ in prev]
        if nxt_words == prev_words[-len(nxt_words):]:
            log(f"Removed duplicate segment ({len(nxt_words)} words): '{result[i + 1].strip()}'")
            result[i + 1] = ''
            continue

        max_check = min(len(prev), len(nxt), 30)
        overlap = 0
        for k in range(max_check, 2, -1):
            if [w for w, _ in prev[-k:]] == [w for w, _ in nxt[:k]]:
                overlap = k
                break
        if overlap:
            # Skip the first `overlap` word-tokens in result[i+1]
            tokens = result[i + 1].split()
            consumed, skip = 0, 0
            for tok in tokens:
                skip += 1
                if re.sub(r'[^\w]', '', tok.lower()):
                    consumed += 1
                if consumed >= overlap:
                    break
            trimmed = ' '.join(tokens[skip:]).lstrip(' ,.')
            log(f"Removed cross-segment overlap ({overlap} words): dropped '{' '.join(tokens[:skip])}'")
            result[i + 1] = trimmed
    return result


def apply_replacements(text: str) -> str:
    """Apply post-processing replacements to fix common transcription errors."""
    import re

    # Pick the boundary anchor for one edge of a key. See the comment above
    # REPLACEMENTS for why the choice depends on the edge character.
    def anchor(ch, side):
        if ch == '' or ch.isspace():
            return ''
        if ch.isalnum() or ch == '_':
            return r'\b'
        return r'(?<!\S)' if side == 'L' else r'(?!\S)'

    for wrong, correct in REPLACEMENTS.items():
        pattern = anchor(wrong[:1], 'L') + re.escape(wrong) + anchor(wrong[-1:], 'R')
        text = re.sub(pattern, correct, text, flags=re.IGNORECASE)
    return text


def strip_hallucinations(text: str) -> str:
    """Remove common Whisper hallucinations from the end of transcriptions."""
    original = text
    changed = True
    while changed:
        changed = False
        for pattern in _HALLUCINATION_RE:
            m = pattern.search(text)
            if m:
                text = text[:m.start()].rstrip()
                changed = True
    # Strip trailing conjunctions/connectors left dangling after hallucination removal
    text = re.sub(r'\s+(?:and|so|but|or|now|well)\s*$', '', text, flags=re.IGNORECASE)
    if text != original:
        log(f"Stripped hallucination: '{original[len(text):].strip()}'")
    return text


def strip_trailing_ellipsis(text: str) -> str:
    """Strip trailing ellipsis that Whisper adds for incomplete sentences.

    Whisper adds '...' when it detects speech was cut off mid-sentence, or
    hallucinates repeated '...' patterns during trailing silence.
    Always called after transcription regardless of mode.
    """
    import re
    # Strip any number of trailing ellipsis patterns (ASCII or Unicode), space-separated or not
    text = re.sub(r'(\s*(\.\.\.|\.\.|…))+\s*$', '', text)
    return text


def strip_trailing_punctuation(text: str) -> str:
    """Strip trailing sentence-ending punctuation for streaming mode.

    This prevents periods from being inserted when pausing mid-dictation.
    Keeps commas and other mid-sentence punctuation.
    Note: Ellipsis is handled separately by strip_trailing_ellipsis().
    """
    # Strip trailing whitespace first
    text = text.rstrip()
    # Remove sentence-ending punctuation
    while text and text[-1] in '.!?':
        text = text[:-1]
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
            # Release all modifiers via ydotool
            # 29=ctrl, 42=shift, 56=alt, 125=Super_L, 126=Super_R
            subprocess.run(
                ["ydotool", "key", "29:0", "42:0", "56:0", "125:0", "126:0"],
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


def _wm_class_for_window(win_id: str) -> str:
    try:
        out = subprocess.run(
            ["xprop", "-id", win_id, "WM_CLASS"],
            capture_output=True, text=True, timeout=2,
        ).stdout
        return " ".join(m.lower() for m in re.findall(r'"([^"]+)"', out))
    except Exception:
        return ""


def _emacs_insert(text: str) -> bool:
    """Insert text into the buffer displayed in Emacs's most-recently-active
    visible frame, without stealing window focus. Returns True on success.

    A bare `(insert ...)` from emacsclient runs in the server's internal
    " *server*" buffer (invisible), so we explicitly target the visible
    frame's selected window's buffer.
    """
    # Escape for elisp string: backslashes and double quotes only.
    escaped = text.replace("\\", "\\\\").replace('"', '\\"')
    elisp = (
        '(let* ((frame (or (car (visible-frame-list)) (selected-frame)))'
        '       (win (frame-selected-window frame))'
        '       (buf (window-buffer win)))'
        '  (with-current-buffer buf'
        f'    (goto-char (window-point win))'
        f'    (insert "{escaped}")'
        f'    (set-window-point win (point))'
        f'    (buffer-name buf)))'
    )
    try:
        result = subprocess.run(
            ["emacsclient", "-e", elisp],
            check=True, capture_output=True, text=True, timeout=10,
        )
        log(f"emacsclient inserted into buffer {result.stdout.strip()}")
        return True
    except Exception as e:
        log(f"emacsclient insert failed ({e}); falling back to focus+paste")
        return False


def type_text(text: str, target_window_id: str | None = None):
    """Insert text using clipboard paste (more reliable than simulated typing).

    Simulated typing with xdotool/ydotool can drop or reorder characters when
    applications can't keep up with the keystroke rate. Clipboard paste is
    atomic and much more reliable.

    For very long text, paste in chunks to avoid terminal buffer issues.

    If target_window_id is provided (X11 only), routes the paste back to the
    window that was focused when recording started, so the transcription lands
    in the right place even if focus moved during a slow AI rewrite. For Emacs
    targets we use emacsclient (no focus change). For other apps we save the
    currently focused window, activate the target, paste, then restore focus.
    """
    if not text:
        return

    saved_focus_id: str | None = None
    target_is_terminal = False
    if target_window_id:
        target_class = _wm_class_for_window(target_window_id)
        target_is_terminal = any(c in X11_TERMINAL_APP_IDS for c in target_class.split())
        # Emacs: insert via emacsclient — no focus change at all.
        if "emacs" in target_class:
            if _emacs_insert(text):
                log(f"Inserted into Emacs via emacsclient ({len(text)} chars)")
                return
            # else: fall through to focus+paste

        # Other apps: save current focus, activate target, paste, restore.
        try:
            saved_focus_id = subprocess.run(
                ["xdotool", "getactivewindow"],
                capture_output=True, text=True, timeout=2,
            ).stdout.strip() or None
        except Exception:
            saved_focus_id = None
        try:
            subprocess.run(
                ["xdotool", "windowactivate", "--sync", target_window_id],
                check=True, capture_output=True, timeout=2,
            )
            log(f"Refocused target window {target_window_id} for paste "
                f"(will restore focus to {saved_focus_id})")
        except Exception as e:
            log(f"Target window {target_window_id} activate failed ({e}); "
                f"pasting into current focus")
            saved_focus_id = None  # don't restore if we didn't switch

    time.sleep(0.05)  # Brief delay for focus

    # Detect session type and use appropriate tool
    session_type = os.environ.get("XDG_SESSION_TYPE", "").lower()

    # Split into chunks for long text only when using xdotool type (non-terminal, non-emacs).
    # Terminal clipboard paste (xclip + Ctrl+Shift+V) is atomic — chunking causes rapid
    # save/restore cycles that freeze the X11 clipboard and lock up terminal windows.
    if len(text) > PASTE_CHUNK_SIZE and not target_is_terminal:
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

    if saved_focus_id and saved_focus_id != target_window_id:
        try:
            subprocess.run(
                ["xdotool", "windowactivate", "--sync", saved_focus_id],
                check=True, capture_output=True, timeout=2,
            )
            log(f"Restored focus to {saved_focus_id}")
        except Exception as e:
            log(f"Focus restore to {saved_focus_id} failed: {e}")

    log(f"Pasted ({len(text)} chars): {text}")


TERMINAL_APP_IDS = {
    "com.mitchellh.ghostty", "ghostty",
    "gnome-terminal", "gnome-terminal-server",
    "xterm", "urxvt", "alacritty", "kitty", "konsole",
    "terminator", "tilix", "xfce4-terminal", "lxterminal",
    "st", "foot", "wezterm", "rio", "contour", "org.gnome.terminal",
}


def _is_terminal_focused_wayland():
    """Check if the focused Wayland window is a terminal emulator."""
    import re as _re

    # --- Sway ---
    try:
        result = subprocess.run(
            ["swaymsg", "-t", "get_tree"],
            capture_output=True, text=True, timeout=2
        )
        if result.returncode == 0:
            tree = json.loads(result.stdout)
            def find_focused(node):
                if node.get("focused"):
                    return node.get("app_id", "").lower()
                for child in node.get("nodes", []) + node.get("floating_nodes", []):
                    found = find_focused(child)
                    if found is not None:
                        return found
                return None
            app_id = find_focused(tree)
            if app_id:
                return app_id in TERMINAL_APP_IDS
    except Exception:
        pass

    # --- GNOME / generic: xdotool + xprop via XWayland compatibility layer ---
    try:
        win_id = subprocess.run(
            ["xdotool", "getactivewindow"],
            capture_output=True, text=True, timeout=2
        ).stdout.strip()
        if win_id:
            xprop_out = subprocess.run(
                ["xprop", "-id", win_id, "WM_CLASS"],
                capture_output=True, text=True, timeout=2
            ).stdout
            # WM_CLASS(STRING) = "ghostty", "com.mitchellh.ghostty"
            for cls in _re.findall(r'"([^"]+)"', xprop_out):
                if cls.lower() in TERMINAL_APP_IDS:
                    return True
    except Exception:
        pass

    return False


def _run_ydotool(argv: list, timeout: int):
    """Run a ydotool command, raising with its own diagnostics on failure.

    ydotool prints socket errors to stdout rather than stderr (see
    Client/ydotool.c), so both streams have to be captured to learn why a
    call failed — discarding them turns "wrong socket path" into a bare
    "exit status 2".
    """
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        detail = ' '.join((proc.stdout + proc.stderr).split()) or "no output"
        raise RuntimeError(
            f"{' '.join(argv[:2])} failed (exit {proc.returncode}): {detail}"
        )


def _ydotool_type(text: str):
    """Type text with ydotool, replacing \\n with Enter key events.

    ydotool maps \\n (0x0a) to KEY_LINEFEED rather than KEY_ENTER, which
    triggers Ctrl+J in Emacs and other apps. Split on newlines and send
    explicit Enter key (keycode 28) between segments.

    Text is folded to ASCII first; see to_ascii for why.
    """
    ascii_text = to_ascii(text)
    if ascii_text != text:
        folded = ' '.join(sorted({c for c in text if not c.isascii()}))
        log(f"ydotool: folded non-ASCII to ASCII: {folded}")

    parts = ascii_text.split('\n')
    for i, part in enumerate(parts):
        if part:
            _run_ydotool(
                ["ydotool", "type", "--key-delay", "10", "--", part], timeout=60
            )
        if i < len(parts) - 1:
            _run_ydotool(["ydotool", "key", "28:1", "28:0"], timeout=5)


def _paste_wayland(text: str):
    """Paste text on Wayland, using clipboard paste for terminals and ydotool type otherwise.

    Terminals are susceptible to PTY buffer overflow when ydotool type simulates
    keystrokes faster than they can process. Clipboard paste (Ctrl+Shift+V) is
    atomic and avoids this. Other apps (Emacs, browsers, etc.) handle simulated
    keystrokes fine, and some don't support Ctrl+Shift+V.
    """
    if _is_terminal_focused_wayland():
        log("paste: terminal detected, using wl-copy + Ctrl+Shift+V")
        wl_proc = None
        try:
            # Save current clipboard
            saved = subprocess.run(
                ["wl-paste", "--no-newline"],
                capture_output=True, timeout=2
            )
            saved_clip = saved.stdout if saved.returncode == 0 else None

            # wl-copy stays alive as the Wayland clipboard owner and never exits,
            # so use Popen (matching _paste_x11's xclip approach).
            wl_proc = subprocess.Popen(
                ["wl-copy"],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            wl_proc.stdin.write(text.encode("utf-8"))
            wl_proc.stdin.close()

            # Small delay to let wl-copy register as clipboard owner
            time.sleep(0.05)

            # Paste with Ctrl+Shift+V
            subprocess.run(
                ["ydotool", "key", "29:1", "42:1", "47:1", "47:0", "42:0", "29:0"],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5
            )
            log("paste: Ctrl+Shift+V sent")

            # Brief delay to let paste complete before restoring clipboard
            time.sleep(0.1)

            # Kill our wl-copy instance before restoring previous clipboard
            wl_proc.terminate()
            wl_proc.wait(timeout=1)
            wl_proc = None

            # Restore previous clipboard
            if saved_clip is not None:
                restore_proc = subprocess.Popen(
                    ["wl-copy"],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                restore_proc.stdin.write(saved_clip)
                restore_proc.stdin.close()

            return
        except FileNotFoundError:
            log("wl-copy/wl-paste not found, falling back to ydotool type")
        except Exception as e:
            log(f"Clipboard paste failed ({e}), falling back to ydotool type")
        finally:
            if wl_proc is not None:
                wl_proc.terminate()

        # Terminal clipboard paste failed — ydotool type directly
        log("paste: terminal clipboard failed, falling back to _ydotool_type")
        _ydotool_type(text)
        return

    # Non-terminal: try xdotool type first — it handles \t and \n correctly and
    # works for X11/XWayland apps (Emacs, etc.) running under GNOME Wayland.
    try:
        win_id = subprocess.run(
            ["xdotool", "getactivewindow"],
            capture_output=True, text=True, timeout=2
        ).stdout.strip()
        if win_id:
            log(f"paste: non-terminal X11 window {win_id}, using xdotool type")
            subprocess.run(
                ["xdotool", "type", "--clearmodifiers", "--delay", "20", "--", text],
                check=True, capture_output=True, timeout=60
            )
            return
    except Exception:
        pass

    log("paste: native Wayland non-terminal, using _ydotool_type")
    _ydotool_type(text)


X11_TERMINAL_APP_IDS = {
    "com.mitchellh.ghostty", "ghostty", "gnome-terminal", "gnome-terminal-server",
    "xterm", "urxvt", "alacritty", "kitty", "konsole",
    "terminator", "tilix", "xfce4-terminal", "lxterminal",
    "st", "foot", "wezterm", "rio", "contour", "cool-retro-term",
}


def _xdotool_type(text: str):
    """Type text with xdotool, replacing \\n with Return key events.

    xdotool type maps \\n (0x0a) to XK_Linefeed which is C-j in Emacs,
    not the Return key. Split on newlines and send explicit Return between.
    """
    parts = text.split('\n')
    for i, part in enumerate(parts):
        if part:
            subprocess.run(
                ["xdotool", "type", "--clearmodifiers", "--delay", "20", "--", part],
                check=True, capture_output=True, timeout=60,
            )
        if i < len(parts) - 1:
            subprocess.run(
                ["xdotool", "key", "--clearmodifiers", "Return"],
                check=True, capture_output=True, timeout=5,
            )


def _paste_x11(text: str):
    """Paste text on X11 using the best method for the focused app.

    - Terminals: xclip + Ctrl+Shift+V (atomic, preserves all characters)
    - Emacs: xclip + Shift+Insert (= clipboard-yank, inserts text literally
      including \\n and \\t without triggering key bindings)
    - Other: _xdotool_type with \\n split to Return key events
    """
    # Detect focused window using two-step xdotool + xprop (more reliable than
    # the compound "getactivewindow getwindowclassname" form which can silently
    # return empty on some X11 configurations).
    win_class = ""
    try:
        win_id = subprocess.run(
            ["xdotool", "getactivewindow"],
            capture_output=True, text=True, timeout=2
        ).stdout.strip()
        if win_id:
            xprop_out = subprocess.run(
                ["xprop", "-id", win_id, "WM_CLASS"],
                capture_output=True, text=True, timeout=2
            ).stdout
            # WM_CLASS(STRING) = "instance", "Class" — check both components
            classes = [m.lower() for m in re.findall(r'"([^"]+)"', xprop_out)]
            win_class = " ".join(classes)
            log(f"paste: X11 window {win_id} WM_CLASS={win_class!r}")
    except Exception as e:
        log(f"paste: window detection error: {e}")

    is_terminal = any(c in X11_TERMINAL_APP_IDS for c in win_class.split())
    is_emacs = "emacs" in win_class

    if is_terminal or is_emacs:
        label = "terminal" if is_terminal else "emacs"
        log(f"paste: {label} ({win_class}), using xclip clipboard paste")
        xclip_proc = None
        try:
            # Save current clipboard
            saved = subprocess.run(
                ["xclip", "-selection", "clipboard", "-o"],
                capture_output=True, timeout=2
            )
            saved_clip = saved.stdout if saved.returncode == 0 else None

            # xclip stays alive as X11 clipboard owner; use Popen, don't wait.
            xclip_proc = subprocess.Popen(
                ["xclip", "-selection", "clipboard"],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            xclip_proc.stdin.write(text.encode("utf-8"))
            xclip_proc.stdin.close()

            time.sleep(0.05)

            if is_terminal:
                # Terminals: Ctrl+Shift+V (bracketed paste, newlines literal)
                subprocess.run(
                    ["xdotool", "key", "--clearmodifiers", "ctrl+shift+v"],
                    check=True, capture_output=True, timeout=5
                )
            else:
                # Emacs: Shift+Insert = clipboard-yank, inserts text literally
                # including \n and \t without triggering any key bindings.
                subprocess.run(
                    ["xdotool", "key", "--clearmodifiers", "shift+Insert"],
                    check=True, capture_output=True, timeout=5
                )

            time.sleep(0.1)

            xclip_proc.terminate()
            xclip_proc.wait(timeout=1)
            xclip_proc = None

            if saved_clip is not None:
                restore_proc = subprocess.Popen(
                    ["xclip", "-selection", "clipboard"],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                restore_proc.stdin.write(saved_clip)
                restore_proc.stdin.close()

            return
        except FileNotFoundError:
            log("xclip not found, falling back to xdotool type")
        except Exception as e:
            log(f"Clipboard paste failed ({e}), falling back to xdotool type")
        finally:
            if xclip_proc is not None:
                xclip_proc.terminate()

    # Other apps: xdotool type with \n → Return (avoids XK_Linefeed/Ctrl+J)
    log(f"paste: other app ({win_class}), using _xdotool_type")
    _xdotool_type(text)


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
    if vram_mb >= 13000:   # 13GB+: only try if VRAM is excessive! large-v3 (5GB + headroom)
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
            # For CUDA, prefer int8_float16: same speed as float16 but more
            # robust beam-search decisions. Observed float16 (and even float32)
            # silently dropping a phrase around a pause that int8_float16 keeps.
            compute_type = "int8_float16"

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

        # Batch post-processing command — restored from file if present
        self.batch_post_cmd: list = self._load_post_cmd()

        # When set, holds {"post_cmd": list, "streaming_mode": bool} to restore
        # after the next completed transcription (one-shot mode support).
        self.one_shot_restore: dict | None = None

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

        # Window that was focused when batch recording started. Used to refocus
        # before pasting so the user can switch windows during slow AI rewrites.
        self.target_window_id: str | None = None

    def _transcribe_with_cpu_fallback(self, audio_file, **kwargs):
        """Run model.transcribe, retrying on CPU if CUDA runs out of memory."""
        try:
            segments, info = self.model.transcribe(str(audio_file), **kwargs)
            return list(segments), info
        except Exception as e:
            if "out of memory" not in str(e).lower():
                raise
            log(f"CUDA OOM during transcription, retrying on CPU...")
            cpu_model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
            segments, info = cpu_model.transcribe(str(audio_file), **kwargs)
            return list(segments), info

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
                # int8_float16 first (same speed as float16, fewer dropped words),
                # then float16, then float32 as compatibility fallbacks.
                for ct in ["int8_float16", "float16", "float32"]:
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
            if len(self.audio_data) == 0:
                log("First audio chunk received")
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

            audio, _ = normalize_audio(audio)

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

                segments, _ = self._transcribe_with_cpu_fallback(
                    temp_file,
                    beam_size=5,
                    language="en",
                    vad_filter=True,
                    initial_prompt=context_prompt,
                    condition_on_previous_text=False,  # We provide context via initial_prompt instead
                    repetition_penalty=1.1,  # Penalize repeated tokens
                    no_repeat_ngram_size=3,  # Prevent 3-gram repetitions
                )
                raw_text = " ".join(seg.text for seg in segments).strip()
                text = normalize_whitespace(raw_text)
                text = apply_replacements(text)
                text = strip_hallucinations(text)
                text = strip_trailing_ellipsis(text)

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

            # Capture the currently focused window so we can refocus it before
            # pasting. Lets the user start dictation, then switch away and do
            # other work while transcription / AI rewrite runs.
            # Only meaningful in batch mode — streaming pastes immediately.
            self.target_window_id = None
            if not self.streaming_mode:
                try:
                    win_id = subprocess.run(
                        ["xdotool", "getactivewindow"],
                        capture_output=True, text=True, timeout=2,
                    ).stdout.strip()
                    if win_id:
                        self.target_window_id = win_id
                        log(f"Captured target window: {win_id}")
                except Exception as e:
                    log(f"Target window capture failed: {e}")

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
        self.recording_start_time = time.time()
        log(f"Recording started ({mode_str} mode)")
        notify(f"🎤 Recording ({mode_str})...", "low")
        return "Recording"

    def stop_recording(self) -> str:
        """Stop recording and transcribe."""
        log("Stop recording requested")
        with self.lock:
            if not self.recording:
                return "Not recording"

            was_streaming = self.streaming_mode
            self.recording = False
            log("Recording flag set to False")

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
            self._maybe_restore_one_shot()
            return "Stopped"

        # Batch mode processing
        if not audio_data:
            log("No audio captured")
            return "No audio"

        audio = np.concatenate(audio_data, axis=0)
        duration = len(audio) / SAMPLE_RATE
        elapsed = time.time() - self.recording_start_time if hasattr(self, 'recording_start_time') else 0
        log(f"Captured {duration:.1f}s of audio (wall clock: {elapsed:.1f}s, chunks: {len(audio_data)})")

        notify("⏹️ Transcribing...", "low")

        # Save raw audio immediately so it is never lost to a processing error.
        # Also keep a persistent copy of the uncompressed original for debugging
        # (AUDIO_FILE gets overwritten below with the compressed+padded version).
        sf.write(str(AUDIO_FILE), audio, SAMPLE_RATE)
        sf.write(str(LAST_RAW_AUDIO_FILE), audio, SAMPLE_RATE)

        # Normalize quiet mics before silence detection. Without this, a mic
        # whose peak is well below SILENCE_THRESHOLD will have its entire
        # recording classified as dead air and compressed to nothing.
        audio, gain = normalize_audio(audio)
        if gain > 1.0:
            log(f"Applied {gain:.1f}x gain (mic peak was {float(np.max(np.abs(audio/gain))):.4f})")

        # Collapse only genuinely long dead-air gaps (>LONG_PAUSE_TRIGGER) so
        # Whisper doesn't hallucinate over them; normal speech pauses are left
        # untouched (re-timing them degrades transcription).
        audio_tc, silence_removed, compression_events, silence_cutoff = compress_silence(audio)
        if silence_removed > 0.1:
            log(f"Compressed {silence_removed:.1f}s of dead air "
                f"(gaps >{LONG_PAUSE_TRIGGER:.0f}s collapsed to {LONG_PAUSE_KEEP:.0f}s, "
                f"adaptive cutoff {silence_cutoff:.4f})")
            # Map each compressed silence run across both timelines so a "missing
            # speech at Ns" report can be traced in raw vs. compressed audio.
            for ev in compression_events:
                log(
                    f"  silence run: raw {ev['raw_start']:.1f}-{ev['raw_end']:.1f}s "
                    f"({ev['raw_end'] - ev['raw_start']:.1f}s) → compressed at "
                    f"{ev['comp_pos']:.1f}s (-{ev['removed']:.1f}s); "
                    f"raw t>{ev['raw_end']:.1f}s maps to compressed t-{ev['removed']:.1f}s onward"
                )

        # Pad with 0.5s of silence so Whisper finishes the final phrase before the audio ends.
        # Match audio dimensions (mono=1D, stereo=2D).
        silence_shape = (int(0.5 * SAMPLE_RATE),) + audio_tc.shape[1:]
        silence_pad = np.zeros(silence_shape, dtype=audio_tc.dtype)
        audio_padded = np.concatenate([audio_tc, silence_pad])

        # Re-write padded version for transcription
        sf.write(str(AUDIO_FILE), audio_padded, SAMPLE_RATE)

        try:
            start = time.time()
            segments, _ = self._transcribe_with_cpu_fallback(
                AUDIO_FILE,
                beam_size=5,
                language="en",
                vad_filter=False,  # Don't discard any audio - user intentionally started recording
                initial_prompt=INITIAL_PROMPT,
                word_timestamps=True,  # Prevents skipping speech in long segments
                hallucination_silence_threshold=None,  # Disabled: was dropping real speech after pauses
            )
            if segments:
                log(f"Segments: {len(segments)}, last ends at {segments[-1].end:.1f}s (audio: {duration:.1f}s)")
                prev_end = 0.0
                for i, seg in enumerate(segments):
                    gap = seg.start - prev_end
                    gap_flag = f" [GAP {gap:.1f}s]" if gap > 0.5 else ""
                    nsp = getattr(seg, 'no_speech_prob', None)
                    nsp_flag = f" [nsp={nsp:.2f}]" if nsp is not None else ""
                    log(f"  seg {i}: {seg.start:.1f}-{seg.end:.1f}s{gap_flag}{nsp_flag} {seg.text.strip()[:80]}")
                    prev_end = seg.end
            # Drop segments where word rate is physically impossible (Whisper hallucination).
            # Normal max speech is ~5 words/sec; hallucinated filler appears at 8-15 words/sec.
            # Only apply to segments with >=3 words to avoid false positives on short real utterances.
            MAX_WORDS_PER_SEC = 6.0
            MIN_WORDS_FOR_RATE_CHECK = 3
            filtered_segments = []
            for seg in segments:
                duration = seg.end - seg.start
                words = len(seg.text.split())
                if duration > 0 and words >= MIN_WORDS_FOR_RATE_CHECK:
                    rate = words / duration
                    if rate > MAX_WORDS_PER_SEC:
                        log(f"Dropped implausible segment ({rate:.1f} words/sec, {duration:.1f}s): '{seg.text.strip()}'")
                        continue
                filtered_segments.append(seg)
            segments = filtered_segments
            segments = drop_trailing_high_nsp(segments)
            segments = drop_trailing_echo(segments)
            seg_texts = remove_segment_overlaps([seg.text for seg in segments])
            raw_text = " ".join(seg_texts).strip()
            text = normalize_whitespace(raw_text)
            text = apply_replacements(text)
            text = strip_hallucinations(text)
            text = strip_trailing_ellipsis(text)
            elapsed = time.time() - start
            if text != raw_text:
                log(f"Transcribed in {elapsed:.2f}s: {raw_text} → {text}")
            else:
                log(f"Transcribed in {elapsed:.2f}s: {text}")
            log(f"Raw transcript: {text}")
            text = self._run_post_processor(text)
        except Exception as e:
            log(f"Transcription error: {e}")
            text = ""
        finally:
            AUDIO_FILE.rename(LAST_AUDIO_FILE)

        if text:
            notify(f"✓ {text[:40]}..." if len(text) > 40 else f"✓ {text}", "low")
            target = self.target_window_id
            self.target_window_id = None
            type_text(text, target_window_id=target)
            self._maybe_restore_one_shot()
            return f"OK: {text}"
        else:
            self.target_window_id = None
            notify("No speech detected", "normal")
            return "No speech"

    def _load_post_cmd(self) -> list:
        try:
            raw = POST_CMD_FILE.read_text().strip()
            if raw and raw != "raw":
                argv = shlex.split(raw)
                if argv and argv[0] in POST_CMD_SAFELIST:
                    log(f"Restored post-cmd: {shlex.join(argv)}")
                    return argv
        except (OSError, ValueError):
            pass
        return list(DEFAULT_BATCH_POST_CMD)

    def _save_post_cmd(self):
        try:
            cmd = shlex.join(self.batch_post_cmd) if self.batch_post_cmd else "raw"
            POST_CMD_FILE.write_text(cmd)
        except OSError:
            pass

    def set_batch_post_cmd(self, args_str: str) -> str:
        args_str = args_str.strip()
        if args_str in ("", "raw"):
            with self.lock:
                self.batch_post_cmd = []
            self._save_post_cmd()
            return "Batch post-cmd: raw"
        try:
            argv = shlex.split(args_str)
        except ValueError as e:
            return f"Parse error: {e}"
        if not argv or argv[0] not in POST_CMD_SAFELIST:
            return (f"First token must be one of: "
                    f"{', '.join(sorted(POST_CMD_SAFELIST))} (or 'raw')")
        with self.lock:
            self.batch_post_cmd = argv
        self._save_post_cmd()
        if self.streaming_mode:
            log(f"Warning: post-cmd set to '{shlex.join(argv)}' but daemon is in streaming mode; it won't apply until batch mode is active")
        return f"Batch post-cmd: {shlex.join(argv)}"

    def get_batch_post_cmd(self) -> str:
        return shlex.join(self.batch_post_cmd) if self.batch_post_cmd else "raw"

    def set_one_shot(self, rec_mode: str, post_cmd_str: str) -> str:
        """Apply a temporary mode that auto-restores after the next completed transcription."""
        if rec_mode not in ("batch", "streaming"):
            return "one-shot requires rec_mode 'batch' or 'streaming'"
        with self.lock:
            self.one_shot_restore = {
                "post_cmd": list(self.batch_post_cmd),
                "streaming_mode": self.streaming_mode,
            }
        want_streaming = (rec_mode == "streaming")
        if self.streaming_mode != want_streaming:
            self.toggle_mode()
        self.set_batch_post_cmd(post_cmd_str)
        label = post_cmd_str.strip() or "raw"
        log(f"One-shot mode set: {rec_mode}/{label}")
        return f"One-shot: {label}"

    def _maybe_restore_one_shot(self):
        """If a one-shot mode was pending, restore the previous mode now."""
        with self.lock:
            restore = self.one_shot_restore
            if restore is None:
                return
            self.one_shot_restore = None
            self.batch_post_cmd = restore["post_cmd"]
            want_streaming = restore["streaming_mode"]
            self.streaming_mode = want_streaming
        self._save_post_cmd()
        restored_cmd = shlex.join(restore["post_cmd"]) if restore["post_cmd"] else "raw"
        log(f"One-shot restored to: {restored_cmd}")
        notify(f"↩ Restored: {restored_cmd}", "normal")

    def _run_post_processor(self, text: str) -> str:
        cmd = list(self.batch_post_cmd)
        if not cmd:
            return text
        label = shlex.join(cmd)
        # Extend PATH so that `clip`, `text-clean`, and `bulletize` are found.
        # They live in ~/dev/claudecode_workflows/bin/ which may not be on the daemon's PATH.
        extra_bin = str(Path.home() / "dev" / "claudecode_workflows" / "bin")
        env = os.environ.copy()
        env["PATH"] = extra_bin + ":" + env.get("PATH", "")

        # Inject API credentials from ~/.claude/settings.json (absent in daemon env).
        # Also add the node bin dir so llm-rewrite can call the token helper.
        api_env, node_bin_dir = _read_claude_api_env()
        env.update(api_env)
        if node_bin_dir:
            env["PATH"] = node_bin_dir + ":" + env["PATH"]
        try:
            notify(f"✏️ Rewriting ({label})...", "low")
            # Use --stdin to pipe text directly; avoids wl-copy blocking on clipboard write.
            rewritten = subprocess.run(
                [cmd[0], "--stdin"] + cmd[1:],
                input=text, capture_output=True, text=True,
                timeout=POST_TIMEOUT_SEC, check=True, env=env,
            ).stdout
            if not rewritten.strip():
                log(f"Post-processor {label} returned empty; using raw")
                return text
            log(f"Post-processed via {label}: {len(text)} → {len(rewritten)} chars")
            return rewritten
        except subprocess.TimeoutExpired:
            log(f"Post-processor {label} timed out after {POST_TIMEOUT_SEC}s; using raw")
            notify("⚠️ Rewrite timeout — using raw transcript", "normal")
            return text
        except subprocess.CalledProcessError as e:
            log(f"Post-processor {label} failed (exit {e.returncode}): {e.stderr}")
            notify("⚠️ Rewrite failed — using raw transcript", "normal")
            return text
        except FileNotFoundError:
            log(f"Post-processor command not found: {cmd[0]}; using raw")
            notify("⚠️ Rewrite tool missing — using raw transcript", "normal")
            return text

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
                response = (f"{state} ({mode} mode, "
                            f"post={self.get_batch_post_cmd()})")
            elif data == "mode":
                response = self.toggle_mode()
            elif data.startswith("post-cmd "):
                args_str = data[len("post-cmd "):]
                response = self.set_batch_post_cmd(args_str)
            elif data == "post-cmd":
                response = f"Batch post-cmd: {self.get_batch_post_cmd()}"
            elif data.startswith("set-mode "):
                target = data[len("set-mode "):].strip()
                if target not in ("batch", "streaming"):
                    response = "set-mode requires 'batch' or 'streaming'"
                else:
                    want_streaming = (target == "streaming")
                    if self.streaming_mode != want_streaming:
                        response = self.toggle_mode()
                    else:
                        response = f"Mode: {target}"
            elif data.startswith("one-shot "):
                rest = data[len("one-shot "):].strip()
                parts = rest.split(" ", 1)
                if len(parts) == 2:
                    response = self.set_one_shot(parts[0], parts[1])
                elif len(parts) == 1 and parts[0] in ("batch", "streaming"):
                    response = self.set_one_shot(parts[0], "raw")
                else:
                    response = "Usage: one-shot <batch|streaming> <post_cmd>"
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
    elif cmd == "post-cmd":
        if not is_daemon_running():
            print("Daemon not running. Start with: dictate-daemon.py start")
            sys.exit(1)
        if len(sys.argv) >= 3:
            rest = " ".join(sys.argv[2:])
            response = send_command(f"post-cmd {rest}")
        else:
            response = send_command("post-cmd")
        print(response)
    elif cmd == "set-mode":
        if not is_daemon_running():
            print("Daemon not running. Start with: dictate-daemon.py start")
            sys.exit(1)
        if len(sys.argv) < 3:
            print("Usage: dictate-daemon.py set-mode batch|streaming")
            sys.exit(1)
        response = send_command(f"set-mode {sys.argv[2]}")
        print(response)
    elif cmd == "one-shot":
        if not is_daemon_running():
            print("Daemon not running. Start with: dictate-daemon.py start")
            sys.exit(1)
        if len(sys.argv) < 4:
            print("Usage: dictate-daemon.py one-shot <batch|streaming> <post_cmd>")
            sys.exit(1)
        rec_mode = sys.argv[2]
        post_cmd = " ".join(sys.argv[3:])
        response = send_command(f"one-shot {rec_mode} {post_cmd}")
        print(response)
    else:
        print(f"Usage: {sys.argv[0]} "
              f"[start|stop|toggle|mode|"
              f"set-mode <batch|streaming>|"
              f"one-shot <batch|streaming> <post_cmd>|"
              f"post-cmd [raw|<wrapper> [args...]]|status]")
        sys.exit(1)


if __name__ == "__main__":
    main()
