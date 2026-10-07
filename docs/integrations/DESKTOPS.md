# GNOME and KDE

Both need the AT-SPI accessibility bridge to see which window has focus. `hyprwhspr setup` offers to enable it.

## GNOME/Mutter

GNOME/Mutter lacks layer-shell, so visual feedback uses notifications. Injection depends on the session:

- **Window detection** uses the AT-SPI accessibility bridge — `hyprwhspr setup` offers to enable it (`gsettings set org.gnome.desktop.interface toolkit-accessibility true`). Without it, GNOME can't tell terminals apart and paste falls back to Ctrl+V. An explicit `paste_mode` (with no `applications` rules) skips the probe entirely.
- **GNOME Wayland direct typing:** Mutter blocks `wtype`, so ASCII text is typed directly with `ydotool type` on layouts that keep ASCII where US QWERTY does — checked against your compiled XKB keymap, not the layout name, so Polish, Romanian and the like type directly while German, French and any Dvorak/Colemak variant fall back to clipboard paste. Set `"prefer_clipboard_paste": true` to always use clipboard paste.
- **GNOME X11 clipboard paste:** X11 uses `xclip` and a normal paste chord rather than the Wayland-only direct-typing workaround. GNOME/X11 on Ubuntu 24.04 is the currently validated X11 configuration.
- **Non-Latin layouts** (Thai, Russian, Arabic, …): no physical key produces a `v` keysym, so hyprwhspr briefly switches to a Latin input source for the paste chord and restores your layout after — just keep a Latin source in Settings → Keyboard → Input Sources.

### Waveform overlay

GNOME users who want an animated waveform instead of notifications can install the opt-in GNOME Shell extension in `contrib/gnome-shell-extension/` — it draws inside gnome-shell, always-on-top, without stealing focus:

```bash
cd contrib/gnome-shell-extension
./install.sh
```

Log out and back in if GNOME cannot enable it immediately. The extension only reads hyprwhspr's state and audio-level files — disable it and GNOME falls back to notifications.

## KDE Plasma

Plasma has layer-shell, so the overlay works normally. Only window detection needs setup:

- **Window detection** uses the AT-SPI accessibility bridge — KWin windows are native Wayland, invisible to both compositor IPC and `xdotool`, and Qt apps only register on the bus when the bridge is on. Plasma leaves it off, so `hyprwhspr config focused-window` reports `not detected` and Konsole gets Ctrl+V, which pastes nothing. `hyprwhspr setup` offers to enable it, or: `busctl --user set-property org.a11y.Bus /org/a11y/bus org.a11y.Status IsEnabled b true` (revert with `b false`). Apps started before the change need a restart. An explicit `paste_mode` (with no `applications` rules) skips the probe entirely.
- **Bindings** — needs `at-spi2-core` and `python-gobject` (Arch), or `gir1.2-atspi-2.0` and `python3-gi` (Debian/Ubuntu).
