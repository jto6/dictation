/* Dictation Focus — minimal focused-window reporter.
 *
 * GNOME Wayland gives normal clients no way to learn which window has focus:
 * XWayland never sets _NET_ACTIVE_WINDOW (so xdotool always fails), and
 * org.gnome.Shell.Introspect.GetWindows / GetRunningApplications / Eval are all
 * access-denied. Extension code runs inside gnome-shell, so it can read
 * global.display.focus_window directly and re-export the one fact the dictation
 * daemon needs: the focused window's WM_CLASS.
 *
 * Deliberately tiny — no signals, no timers, no window enumeration. This runs
 * in the shell process, where a fault takes the whole session down with it.
 */

import Gio from 'gi://Gio';
import {Extension} from 'resource:///org/gnome/shell/extensions/extension.js';

const BUS_NAME = 'org.local.DictationFocus';
const OBJECT_PATH = '/org/local/DictationFocus';

const IFACE = `
<node>
  <interface name="org.local.DictationFocus">
    <method name="GetFocused">
      <arg type="s" direction="out" name="wm_class"/>
    </method>
  </interface>
</node>`;

class FocusService {
    // Returns the focused window's WM_CLASS, or '' when nothing is focused.
    // Mutter sets WM_CLASS from xdg_toplevel.set_app_id for Wayland toplevels,
    // so this is "com.mitchellh.ghostty" for a native Wayland ghostty window
    // and the X11 class for XWayland windows.
    GetFocused() {
        try {
            const win = global.display?.focus_window;
            if (!win)
                return '';
            return win.get_wm_class() || win.get_gtk_application_id() || '';
        } catch (e) {
            return '';
        }
    }
}

export default class DictationFocusExtension extends Extension {
    enable() {
        this._dbus = Gio.DBusExportedObject.wrapJSObject(IFACE, new FocusService());
        this._dbus.export(Gio.DBus.session, OBJECT_PATH);
        this._nameId = Gio.bus_own_name(
            Gio.BusType.SESSION, BUS_NAME, Gio.BusNameOwnerFlags.NONE,
            null, null, null);
    }

    disable() {
        if (this._nameId) {
            Gio.bus_unown_name(this._nameId);
            this._nameId = null;
        }
        if (this._dbus) {
            this._dbus.unexport();
            this._dbus = null;
        }
    }
}
