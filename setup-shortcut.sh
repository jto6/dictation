#!/bin/bash
# Setup keyboard shortcuts for dictation on GNOME
# Run this script on any Ubuntu/GNOME machine after installing whisper-dictation

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOGGLE_SCRIPT="$SCRIPT_DIR/start-dictation-daemon.sh"
BINDING_PATH="/org/gnome/settings-daemon/plugins/media-keys/custom-keybindings/dictate-toggle/"
MODE_BINDING_PATH="/org/gnome/settings-daemon/plugins/media-keys/custom-keybindings/dictate-mode/"
SHORTCUT="${1:-<Super>w}"
MODE_SHORTCUT="${2:-<Super><Shift>w}"

# Check if we're on GNOME
if [[ "$XDG_CURRENT_DESKTOP" != *"GNOME"* ]]; then
    echo "Warning: This script is designed for GNOME desktop."
    echo "Detected: $XDG_CURRENT_DESKTOP"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check if toggle script exists
if [[ ! -x "$TOGGLE_SCRIPT" ]]; then
    echo "Error: Toggle script not found or not executable: $TOGGLE_SCRIPT"
    exit 1
fi

echo "Setting up dictation keyboard shortcuts..."
echo "  Script: $TOGGLE_SCRIPT"
echo "  Toggle shortcut: $SHORTCUT"
echo "  Mode shortcut: $MODE_SHORTCUT"
echo ""

# Get current custom keybindings
current=$(gsettings get org.gnome.settings-daemon.plugins.media-keys custom-keybindings)

# Helper function to add a binding if it doesn't exist
add_binding() {
    local path="$1"
    local current="$2"

    if [[ "$current" != *"$path"* ]]; then
        if [[ "$current" == "@as []" ]]; then
            echo "['$path']"
        else
            echo "${current%]*}, '$path']"
        fi
    else
        echo "$current"
    fi
}

# Add both bindings
echo "Setting up keyboard shortcuts..."
current=$(add_binding "$BINDING_PATH" "$current")
current=$(add_binding "$MODE_BINDING_PATH" "$current")
gsettings set org.gnome.settings-daemon.plugins.media-keys custom-keybindings "$current"

# Set the toggle keybinding properties
gsettings set org.gnome.settings-daemon.plugins.media-keys.custom-keybinding:$BINDING_PATH name "Dictate Toggle"
gsettings set org.gnome.settings-daemon.plugins.media-keys.custom-keybinding:$BINDING_PATH command "$TOGGLE_SCRIPT toggle"
gsettings set org.gnome.settings-daemon.plugins.media-keys.custom-keybinding:$BINDING_PATH binding "$SHORTCUT"

# Set the mode keybinding properties
gsettings set org.gnome.settings-daemon.plugins.media-keys.custom-keybinding:$MODE_BINDING_PATH name "Dictate Mode Toggle"
gsettings set org.gnome.settings-daemon.plugins.media-keys.custom-keybinding:$MODE_BINDING_PATH command "$TOGGLE_SCRIPT mode"
gsettings set org.gnome.settings-daemon.plugins.media-keys.custom-keybinding:$MODE_BINDING_PATH binding "$MODE_SHORTCUT"

# Setup autostart
echo "Setting up autostart..."
mkdir -p ~/.config/autostart
sed "s|__INSTALL_DIR__|$SCRIPT_DIR|g" "$SCRIPT_DIR/dictate-daemon-start.desktop" > ~/.config/autostart/dictate-daemon-start.desktop

# Install desktop launchers
echo "Installing desktop launchers..."
mkdir -p ~/.local/share/applications
sed "s|__INSTALL_DIR__|$SCRIPT_DIR|g" "$SCRIPT_DIR/dictate-toggle.desktop" > ~/.local/share/applications/dictate-toggle.desktop
sed "s|__INSTALL_DIR__|$SCRIPT_DIR|g" "$SCRIPT_DIR/dictate-daemon-start.desktop" > ~/.local/share/applications/dictate-daemon-start.desktop
update-desktop-database ~/.local/share/applications/ 2>/dev/null || true

echo ""
echo "✓ Setup complete!"
echo ""
echo "Configured:"
echo "  • Toggle shortcut: $SHORTCUT"
echo "  • Mode shortcut: $MODE_SHORTCUT"
echo "  • Autostart: daemon starts on login"
echo "  • Desktop launchers: installed"
echo ""
echo "Usage:"
echo "  1. Press $SHORTCUT to start recording"
echo "  2. Speak your message"
echo "  3. Press $SHORTCUT again to stop and transcribe"
echo ""
echo "  Press $MODE_SHORTCUT to switch between batch and streaming modes"
echo "    • Batch mode: transcribes all at once when stopped (more accurate)"
echo "    • Streaming mode: transcribes phrases as you speak (more responsive)"
echo ""
echo "To use different shortcuts, run:"
echo "  $0 '<Super>F9' '<Super><Shift>F9'"
echo ""
echo "To verify, check: Settings → Keyboard → Custom Shortcuts"
