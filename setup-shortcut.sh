#!/bin/bash
# Setup keyboard shortcut for dictation toggle on GNOME
# Run this script on any Ubuntu/GNOME machine after installing whisper-dictation

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOGGLE_SCRIPT="$SCRIPT_DIR/start-dictation-daemon.sh"
BINDING_PATH="/org/gnome/settings-daemon/plugins/media-keys/custom-keybindings/dictate-toggle/"
SHORTCUT="${1:-<Super>w}"

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

echo "Setting up dictation keyboard shortcut..."
echo "  Script: $TOGGLE_SCRIPT"
echo "  Shortcut: $SHORTCUT"
echo ""

# Get current custom keybindings
current=$(gsettings get org.gnome.settings-daemon.plugins.media-keys custom-keybindings)

# Check if our binding already exists
if [[ "$current" == *"dictate-toggle"* ]]; then
    echo "Updating existing dictate-toggle shortcut..."
else
    echo "Adding new dictate-toggle shortcut..."
    # Add our binding to the list
    if [[ "$current" == "@as []" ]]; then
        # No existing bindings
        new_bindings="['$BINDING_PATH']"
    else
        # Append to existing bindings
        # Remove trailing ] and add our path
        new_bindings="${current%]*}, '$BINDING_PATH']"
    fi
    gsettings set org.gnome.settings-daemon.plugins.media-keys custom-keybindings "$new_bindings"
fi

# Set the keybinding properties
gsettings set org.gnome.settings-daemon.plugins.media-keys.custom-keybinding:$BINDING_PATH name "Dictate Toggle"
gsettings set org.gnome.settings-daemon.plugins.media-keys.custom-keybinding:$BINDING_PATH command "$TOGGLE_SCRIPT toggle"
gsettings set org.gnome.settings-daemon.plugins.media-keys.custom-keybinding:$BINDING_PATH binding "$SHORTCUT"

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
echo "  • Keyboard shortcut: $SHORTCUT"
echo "  • Autostart: daemon starts on login"
echo "  • Desktop launchers: installed"
echo ""
echo "Usage:"
echo "  1. Press $SHORTCUT to start recording"
echo "  2. Speak your message"
echo "  3. Press $SHORTCUT again to stop, transcribe, and type"
echo ""
echo "To use a different shortcut, run:"
echo "  $0 '<Super>F9'    # Example: Super+F9"
echo "  $0 '<Ctrl><Alt>d' # Example: Ctrl+Alt+D"
echo ""
echo "To verify, check: Settings → Keyboard → Custom Shortcuts"
