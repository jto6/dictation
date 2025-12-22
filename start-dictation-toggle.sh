#!/bin/bash
# Toggle dictation - run once to start recording, again to stop and transcribe
# For use with Claude Code terminal

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate virtual environment and run toggle script
source venv/bin/activate
exec python3 dictate-toggle.py "$@"
