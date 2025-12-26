#!/bin/bash
# Installation script for Whisper Dictation
# Safe to run multiple times - will update existing installation

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=================================="
echo "Whisper Dictation - Install/Update"
echo "=================================="
echo ""

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not installed."
    echo "Install with: sudo apt install python3 python3-venv python3-pip"
    exit 1
fi

# Check for typing tools (ydotool for Wayland, xdotool for X11)
SESSION_TYPE="${XDG_SESSION_TYPE:-unknown}"
echo "Session type: $SESSION_TYPE"

if [[ "$SESSION_TYPE" == "wayland" ]]; then
    echo ""
    echo "Wayland detected - checking for ydotool..."
    if ! command -v ydotool &> /dev/null; then
        echo "Warning: ydotool not found (needed for typing on Wayland)"
        echo "Install with: sudo apt install ydotool"
        read -p "Install ydotool now? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            sudo apt install -y ydotool
        fi
    fi

    if command -v ydotool &> /dev/null; then
        echo "✓ ydotool found"

        # Check if ydotoold is running
        if ! pgrep -x ydotoold > /dev/null; then
            echo ""
            echo "Setting up ydotool permissions..."

            # Check if user is in input group
            if ! groups | grep -q '\binput\b'; then
                echo "Adding user to 'input' group for uinput access..."
                sudo usermod -aG input "$USER"
                NEED_RELOGIN=true
            fi

            # Create udev rule for uinput
            if [ ! -f /etc/udev/rules.d/99-uinput.rules ]; then
                echo "Creating udev rule for uinput access..."
                echo 'KERNEL=="uinput", GROUP="input", MODE="0660"' | sudo tee /etc/udev/rules.d/99-uinput.rules > /dev/null
                sudo udevadm control --reload-rules
                sudo udevadm trigger
            fi

            # Set up ydotoold autostart
            echo "Setting up ydotoold autostart..."
            mkdir -p ~/.config/autostart
            cp "$SCRIPT_DIR/ydotoold.desktop" ~/.config/autostart/
            echo "✓ ydotoold autostart configured"

            if [ "$NEED_RELOGIN" = true ]; then
                echo ""
                echo "⚠ IMPORTANT: You need to log out and log back in for group membership to take effect!"
                echo "After logging back in, ydotoold will start automatically on login."
                echo "The dictation daemon will also start automatically."
            else
                # Try to start ydotoold now
                echo "Starting ydotoold daemon..."
                ydotoold > /dev/null 2>&1 &
                sleep 1
                if pgrep -x ydotoold > /dev/null; then
                    echo "✓ ydotoold started"
                else
                    echo "⚠ Could not start ydotoold - it will start automatically on next login"
                fi
            fi
        else
            echo "✓ ydotoold is running"
        fi
    fi
else
    # X11 - check for xdotool
    if ! command -v xdotool &> /dev/null; then
        echo "Warning: xdotool not found (needed for typing text)"
        echo "Install with: sudo apt install xdotool"
        read -p "Continue anyway? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    else
        echo "✓ xdotool found"
    fi
fi
echo ""

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

# Create virtual environment (or recreate if incomplete)
if [ ! -f "venv/bin/activate" ]; then
    if [ -d "venv" ]; then
        echo "Removing incomplete virtual environment..."
        rm -rf venv
    fi
    echo "Creating virtual environment..."
    python3 -m venv venv
else
    echo "✓ Virtual environment exists"
fi

# Activate venv
source venv/bin/activate

# Install/update base dependencies
echo ""
echo "Installing/updating Python dependencies..."
pip install --upgrade pip -q
pip install --upgrade faster-whisper sounddevice soundfile numpy -q
echo "✓ Base dependencies installed/updated"

# Install GPU support if available
if [ "$HAS_GPU" = true ]; then
    echo ""
    echo "Installing/updating GPU support (nvidia-cudnn-cu12)..."
    pip install --upgrade nvidia-cudnn-cu12 -q
    echo "✓ GPU support installed"
    echo "  (Device/compute type will be auto-detected at runtime)"
else
    echo "  (Will use CPU mode - auto-detected at runtime)"
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

# Restart daemon if running (to pick up updates)
if [ -f "/tmp/whisper-dictation/daemon.pid" ]; then
    echo ""
    echo "Restarting daemon to apply updates..."
    ./start-dictation-daemon.sh stop 2>/dev/null || true
    sleep 1
    ./start-dictation-daemon.sh start
    echo "✓ Daemon restarted"
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
echo "See CLAUDE.md for configuration options."
