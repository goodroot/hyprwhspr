# Troubleshooting

Start with `hyprwhspr status --report`. Then read on.

## Reset installation

If you're having persistent issues, completely reset hyprwhspr:

```bash
hyprwhspr uninstall
hyprwhspr setup
```

## CUDA host compiler rejected

When CUDA rejects your system GCC as too new, setup automatically selects the
newest compatible versioned `g++` it can find. If your compiler is installed in
a non-standard location, select it explicitly before running setup:

```bash
export HYPRWHSPR_CUDA_HOST=/path/to/g++
hyprwhspr setup
```

## Common issues

### Something is weird

Restart the service - right click on the waybar icon if you use it, or:

```bash
systemctl --user restart hyprwhspr.service
```

Still weird? Proceed.

### I heard the sound but don't see text

On resume/restart, the microphone often "loses connection" and requires reseating — a Linux quirk not resolvable by hyprwhspr. Reseat your microphone as prompted, and ensure the **right microphone** is set in sound options.

If the default source ends in `.monitor`, select a real input or set
`audio_device_name`. Test it with `hyprwhspr test --live`.

### "Missing Python modules" at startup

An update brought a dependency your environment lacks. The service runs; dictation may not.
`Python dependencies changed since setup` is the gentler cousin. Same fix:

```bash
hyprwhspr setup   # keep the backend, reinstall: yes
```

### Hotkey not working

```bash
# Check service status for hyprwhspr
systemctl --user status hyprwhspr.service

# Check logs
journalctl --user -u hyprwhspr.service -f
```

```bash
# The ydotool paste fallback (GNOME/Mutter) runs as a private child of hyprwhspr.
# After a dictation, confirm the daemon is alive:
pgrep -af 'ydotoold .*hyprwhspr-ydotool.sock'
```

### Why no ydotool service? (private ydotoold)

`wtype` is the primary paste path. On compositors that reject the Wayland
virtual-keyboard protocol (notably GNOME/Mutter), hyprwhspr falls back to its
**own private `ydotoold`** — a child process on a dedicated socket
(`$XDG_RUNTIME_DIR/hyprwhspr-ydotool.sock`), launched lazily and torn down with the
service. It runs rootless via the `uaccess` ACL on `/dev/uinput`; the `input`
group and udev rule from setup are the fallback, and mainly serve the global
hotkey (evdev reads of `/dev/input/event*`).

### Service starts but doesn't work until restarted

`hyprwhspr` must start within an active graphical session.

If the service appears active but hotkeys/transcription don't work until you manually restart, your session environment may not be set up correctly.

Check your session:

```bash
# Verify graphical-session.target is active
systemctl --user is-active graphical-session.target

# Verify the active display environment is available to systemd services
systemctl --user show-environment | grep -E 'WAYLAND_DISPLAY|DISPLAY|XAUTHORITY|NIRI_SOCKET'
```

If `WAYLAND_DISPLAY` is missing, add to `~/.config/hypr/hyprland.conf`:

```bash
# Export session environment to systemd user services
exec-once = dbus-update-activation-environment --systemd WAYLAND_DISPLAY XDG_CURRENT_DESKTOP HYPRLAND_INSTANCE_SIGNATURE
```

hyprwhspr also has a startup fallback: if `WAYLAND_DISPLAY` is unset but a
Wayland socket exists in `XDG_RUNTIME_DIR`, it will use the newest `wayland-*`
socket for its own process and children. The compositor environment export above
is still the recommended fix because it makes the correct display available to
all systemd user services.

For X11, `DISPLAY` must be present and `XAUTHORITY` should be imported when your
session uses it:

```bash
systemctl --user import-environment DISPLAY XAUTHORITY XDG_SESSION_TYPE XDG_CURRENT_DESKTOP
systemctl --user show-environment | grep -E 'DISPLAY|XAUTHORITY|XDG_SESSION_TYPE'
```

Do not set `WAYLAND_DISPLAY` in an explicit `XDG_SESSION_TYPE=x11` session.

**Niri:**

hyprwhspr uses `niri msg --json focused-window` to detect the focused app and choose the correct paste shortcut. That requires `NIRI_SOCKET` to be available in the systemd user environment used by `hyprwhspr.service`.

If `NIRI_SOCKET` is missing, add an environment export to your Niri startup config:

```kdl
spawn-at-startup "dbus-update-activation-environment" "--systemd" "WAYLAND_DISPLAY" "XDG_CURRENT_DESKTOP" "NIRI_SOCKET"
```

**Hyprland:**

If `graphical-session.target` is inactive, you likely need a session manager to activate it.

The recommended approach is to launch Hyprland via [uwsm](https://github.com/Vladimir-csp/uwsm) (it activates `graphical-session.target` and exports the session environment to systemd).

If you *aren't* using a session manager and your system allows it, you can try starting it manually:

```bash
exec-once = systemctl --user start graphical-session.target
```

> **Note:** Some distros set `graphical-session.target` with `RefuseManualStart=yes`, in which case the manual start will fail and you should use a session manager like `uwsm` instead.

Then restart Hyprland or log out and back in.

Run `hyprwhspr validate` to confirm the session is configured correctly.

### Permission denied

```bash
# Fix uinput permissions
hyprwhspr setup

# Log out and back in
```

### No audio input

Is your mic _actually_ available?

```bash
# Check audio devices
pactl list short sources

# Restart PipeWire
systemctl --user restart pipewire
```

### Microphone indicator shows on while idle

If your desktop (GNOME, Ubuntu, etc.) shows the microphone as active whenever the hyprwhspr service is running, you likely have `keepalive_stream` enabled. Disable it:

```jsonc
{
  "keepalive_stream": false
}
```

This is the default. If you previously enabled it to fix `paTimedOut` errors, see [Audio stream keepalive](usage/AUDIO_INPUT.md#audio-stream-keepalive) for the trade-off.

### Audio feedback not working

```bash
# Check if audio feedback is enabled in config
cat ~/.config/hyprwhspr/config.json | grep audio_feedback

# Verify sound files exist (script install uses ~/hyprwhspr/share/assets/)
ls -la /usr/lib/hyprwhspr/share/assets/   # AUR install
ls -la ~/hyprwhspr/share/assets/           # Script install

# Check which audio player is available
which ffplay paplay pw-play aplay
```

**Important:** the default sounds are OGG; `aplay` (ALSA) only supports WAV and will produce white noise on OGG. Install `ffplay` (ffmpeg) or ensure `paplay` (pulseaudio-utils) or `pw-play` (pipewire) is available.

### Model not found

```bash
# Check installed models (routes to active backend)
hyprwhspr model status

# Download a model
hyprwhspr model download base

# Verify model in config
cat ~/.config/hyprwhspr/config.json | grep model
```

### NumPy ABI or dependency verification failure

During setup, hyprwhspr verifies each backend import before downloading a
model. Messages such as `_ARRAY_API not found`, `numpy.dtype size changed`, or
"compiled using NumPy 1.x" mean a compiled package and the NumPy visible in the
managed environment use incompatible binary interfaces. The diagnostic shown
by setup—and retained in `hyprwhspr status`—includes the failing import and the
NumPy versions and paths seen before and after installation.

Re-run `hyprwhspr setup` first. Safe inherited-package conflicts are relocated
automatically, and failed or interrupted Cohere downloads resume from the
Hugging Face cache. If Cohere is already configured, its download can also be
resumed with:

```bash
hyprwhspr model download
```

The package named by the traceback may be the incompatible side rather than
NumPy itself. Cohere setup also verifies and, when safe, relocates the
inherited scientific-package builds it manages.

As a last resort, use the diagnostic paths to identify the conflicting NumPy
and manually install the compatible NumPy version into the managed venv. This
is intentionally not automatic or globally pinned because the correct version
depends on the failing compiled package and platform.

```bash
# Examples only—choose the side indicated by the persisted ABI diagnostic:
~/.local/share/hyprwhspr/venv/bin/python -m pip install --ignore-installed 'numpy<2'
~/.local/share/hyprwhspr/venv/bin/python -m pip install --ignore-installed 'numpy>=2'
```

### Stuck recording state

```bash
# Cancel the stuck recording, audio discarded
hyprwhspr record cancel

# Still stuck: restart
systemctl --user restart hyprwhspr.service

# Check service status
systemctl --user status hyprwhspr.service
```

### This sucks

Doh! We tried.

Wipe the slate clean and remove everything:

```
hyprwhspr uninstall
yay -Rs hyprwhspr
```

Or better yet - create an issue and help us improve.
