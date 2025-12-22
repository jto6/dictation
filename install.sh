#!/bin/bash
# Full installation script for Whisper Dictation
# Run this on a fresh machine to set up everything

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=================================="
echo "Whisper Dictation - Full Install"
echo "=================================="
echo ""

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not installed."
    echo "Install with: sudo apt install python3 python3-venv python3-pip"
    exit 1
fi

# Check for xdotool
if ! command -v xdotool &> /dev/null; then
    echo "Warning: xdotool not found (needed for typing text)"
    echo "Install with: sudo apt install xdotool"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check for NVIDIA GPU
HAS_GPU=false
if command -v nvidia-smi &> /dev/null; then
    if nvidia-smi &> /dev/null; then
        HAS_GPU=true
        echo "✓ NVIDIA GPU detected"
        nvidia-smi --query-gpu=name --format=csv,noheader | head -1
    fi
else
    echo "! No NVIDIA GPU detected - will use CPU mode"
fi
echo ""

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
else
    echo "✓ Virtual environment exists"
fi

# Activate venv
source venv/bin/activate

# Install base dependencies
echo ""
echo "Installing Python dependencies..."
pip install --upgrade pip -q
pip install faster-whisper sounddevice soundfile numpy -q
echo "✓ Base dependencies installed"

# Install GPU support if available
if [ "$HAS_GPU" = true ]; then
    echo ""
    echo "Installing GPU support (nvidia-cudnn-cu12)..."
    pip install nvidia-cudnn-cu12 -q
    echo "✓ GPU support installed"

    # Update daemon to use GPU
    if grep -q 'DEVICE = "cpu"' dictate-daemon.py; then
        sed -i 's/DEVICE = "cpu"/DEVICE = "cuda"/' dictate-daemon.py
        sed -i 's/COMPUTE_TYPE = "int8"/COMPUTE_TYPE = "float16"/' dictate-daemon.py
        echo "✓ Configured for GPU acceleration"
    fi
else
    # Ensure CPU mode
    if grep -q 'DEVICE = "cuda"' dictate-daemon.py; then
        sed -i 's/DEVICE = "cuda"/DEVICE = "cpu"/' dictate-daemon.py
        sed -i 's/COMPUTE_TYPE = "float16"/COMPUTE_TYPE = "int8"/' dictate-daemon.py
        echo "✓ Configured for CPU mode"
    fi
fi

# Make scripts executable
chmod +x dictate-daemon.py dictate-toggle.py
chmod +x start-dictation-daemon.sh start-dictation-toggle.sh
chmod +x setup-shortcut.sh

# Check for GNOME
echo ""
if [[ "$XDG_CURRENT_DESKTOP" == *"GNOME"* ]]; then
    echo "GNOME desktop detected - setting up keyboard shortcut..."
    ./setup-shortcut.sh
else
    echo "Non-GNOME desktop detected: $XDG_CURRENT_DESKTOP"
    echo "Skipping keyboard shortcut setup."
    echo ""
    echo "To set up manually, bind this command to a shortcut:"
    echo "  $SCRIPT_DIR/start-dictation-daemon.sh toggle"
    echo ""
    echo "Desktop launchers installed to ~/.local/share/applications/"
    mkdir -p ~/.local/share/applications
    sed "s|__INSTALL_DIR__|$SCRIPT_DIR|g" dictate-toggle.desktop > ~/.local/share/applications/dictate-toggle.desktop
    sed "s|__INSTALL_DIR__|$SCRIPT_DIR|g" dictate-daemon-start.desktop > ~/.local/share/applications/dictate-daemon-start.desktop
    update-desktop-database ~/.local/share/applications/ 2>/dev/null || true
fi

echo ""
echo "=================================="
echo "Installation complete!"
echo "=================================="
echo ""
echo "Usage:"
echo "  1. Press Super+W to start recording"
echo "  2. Speak your message"
echo "  3. Press Super+W again to stop and transcribe"
echo ""
echo "The daemon will auto-start on login."
echo "For manual control: $SCRIPT_DIR/start-dictation-daemon.sh [start|stop|toggle|status]"
echo ""
echo "See README.md for customization options."
