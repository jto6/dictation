#!/bin/bash
# Wrapper to start/toggle the dictation daemon
# Usage: start-dictation-daemon.sh [start|stop|toggle|mode|status]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

source venv/bin/activate

# Add NVIDIA cuDNN libraries to library path for GPU acceleration
# Dynamically find the Python version in the venv
PYTHON_VERSION=$(python3 -c 'import sys; print(f"python{sys.version_info.major}.{sys.version_info.minor}")')
CUDNN_PATH="$SCRIPT_DIR/venv/lib/$PYTHON_VERSION/site-packages/nvidia/cudnn/lib"
CUBLAS_PATH="$SCRIPT_DIR/venv/lib/$PYTHON_VERSION/site-packages/nvidia/cublas/lib"
if [ -d "$CUDNN_PATH" ]; then
    export LD_LIBRARY_PATH="$CUDNN_PATH:$CUBLAS_PATH:${LD_LIBRARY_PATH:-}"
fi

# Default to toggle if no argument
CMD="${1:-toggle}"

# Auto-start daemon if toggle or mode requested but daemon not running
if [ "$CMD" = "toggle" ] || [ "$CMD" = "mode" ]; then
    if ! python3 dictate-daemon.py status 2>/dev/null | grep -q "running"; then
        echo "Starting daemon..."
        python3 dictate-daemon.py start
        sleep 2  # Wait for model to load
    fi
fi

exec python3 dictate-daemon.py "$CMD"
