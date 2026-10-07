# Hotkeys

Set the shortcut that starts recording:

```jsonc
{
    "primary_shortcut": "CTRL+SHIFT+SPACE"
}
```

## Supported key types

- **Modifiers**: `ctrl`, `alt`, `shift`, `super` (left) or `rctrl`, `ralt`, `rshift`, `rsuper` (right)
- **Function keys**: `f1` through `f24`
- **Letters**: `a` through `z`
- **Numbers**: `1` through `9`, `0`
- **Arrow keys**: `up`, `down`, `left`, `right`
- **Special keys**: `enter`, `space`, `tab`, `esc`, `backspace`, `delete`, `home`, `end`, `pageup`, `pagedown`
- **Lock keys**: `capslock`, `numlock`, `scrolllock`
- **Media keys**: `mute`, `volumeup`, `volumedown`, `play`, `nextsong`, `previoussong`
- **Numpad**: `kp0` through `kp9`, `kpenter`, `kpplus`, `kpminus`

Or use direct evdev key names for any key not in the alias list:

```jsonc
{
    "primary_shortcut": "SUPER+KEY_COMMA"
}
```

Examples:

- `"SUPER+SHIFT+M"` - Super + Shift + M
- `"CTRL+ALT+F1"` - Ctrl + Alt + F1
- `"F12"` - Just F12 (no modifier)
- `"RCTRL+RSHIFT+ENTER"` - Right Ctrl + Right Shift + Enter

## Secondary shortcut with language

Use a different hotkey for a specific language:

```jsonc
{
    "primary_shortcut": "SUPER+ALT+D",    // Uses default language from config
    "secondary_shortcut": "SUPER+ALT+I",  // Optional: second hotkey
    "secondary_language": "it"          // Language for secondary shortcut
}
```

> **Note**: Works with backends that support language parameters:
> - **Local whisper models**: Fully supported (all pywhispercpp models)
> - **Realtime WebSocket**: Fully supported (OpenAI, Google, ElevenLabs)
> - **REST API**: Only if the endpoint accepts a `language` parameter (varies by provider/custom endpoint)

The primary shortcut uses the `language` setting from your config (or auto-detect if `null`); the secondary always uses `secondary_language`.

Configure via CLI:

```bash
hyprwhspr config secondary-shortcut
```

## Cancel shortcut

Bail out of an accidental recording — discards the audio and plays the error sound:

```jsonc
{
    "cancel_shortcut": "SUPER+ESCAPE"  // Any key combo (default: null = disabled)
}
```

Works in all recording modes; **in long-form mode it discards all accumulated segments and resets the session to idle.**

You can also cancel without a dedicated shortcut:

```bash
# Via CLI
hyprwhspr record cancel

# Via FIFO directly (useful for Hyprland binds or sxhkd)
echo cancel > "$XDG_RUNTIME_DIR/hyprwhspr/recording_control"
```

## Hyprland native bindings

Use Hyprland's compositor bindings instead of evdev keyboard grabbing — sometimes better compatibility with keyboard remappers.

Enable in config (`~/.config/hyprwhspr/config.json`):

```jsonc
{
  "use_hypr_bindings": true
}
```

Then add bindings to `~/.config/hypr/hyprland.conf`.

### Toggle mode

Press once to start, press again to stop:

```bash
bindd = SUPER ALT, D, Speech-to-text, exec, /usr/lib/hyprwhspr/config/hyprland/hyprwhspr-tray.sh record
```

### Push-to-talk mode

Hold the key to record, release to stop:

```bash
bind = SUPER ALT, D, exec, echo "start" > "$XDG_RUNTIME_DIR/hyprwhspr/recording_control"
bindr = SUPER ALT, D, exec, echo "release" > "$XDG_RUNTIME_DIR/hyprwhspr/recording_control"
```

`release` stops, or latches a long hold (`push_to_talk_lock_seconds`). A `stop` binding still works; it just never latches.

### Long-form mode

Primary shortcut toggles record/pause/resume; submit shortcut transcribes:

```bash
bindd = SUPER ALT, D, Speech-to-text, exec, /usr/lib/hyprwhspr/config/hyprland/hyprwhspr-tray.sh record
bindd = SUPER ALT, E, Speech-to-text-submit, exec, echo "submit" > "$XDG_RUNTIME_DIR/hyprwhspr/recording_control"
```

### Cancel recording (all modes)

Discard audio without transcribing:

```bash
bind = SUPER, ESCAPE, exec, echo "cancel" > "$XDG_RUNTIME_DIR/hyprwhspr/recording_control"
```

Restart the service to lock in changes:

```bash
systemctl --user restart hyprwhspr
```

## External hotkey systems

Bind [`hyprwhspr record`](CLI.md#recording) in KDE, GNOME, sxhkd, Espanso or anything else. `--lang` overrides the language for that recording — handy for per-language hotkeys:

```bash
# Example: KDE custom shortcuts
# English: hyprwhspr record toggle
# Italian: hyprwhspr record start --lang it
# Cancel:  hyprwhspr record cancel

# Example: Hyprland config
bind = SUPER ALT, D, exec, hyprwhspr record toggle
bind = SUPER ALT, I, exec, hyprwhspr record start --lang it
bind = SUPER, ESCAPE, exec, hyprwhspr record cancel
```

## Running without keyboard access

With `grab_keys: false` (default), hyprwhspr can start even if you are not in the `input` group, but the global shortcut will not work. Control recording via:

- The CLI (`hyprwhspr record toggle`, `hyprwhspr record start`, etc.)
- The `recording_control` FIFO for lowest latency (e.g. bind in Hyprland to `echo start > "$XDG_RUNTIME_DIR/hyprwhspr/recording_control"`)

## Keyboard device selection

If you have multiple input tools (e.g., Espanso, keyd, kmonad), specify which to use:

```jsonc
{
  "selected_device_name": "USB Keyboard"  // Match by device name (recommended)
}
```

Or by device path:

```jsonc
{
  "selected_device_path": "/dev/input/event3"  // Match by exact path
}
```

Device name takes priority if both are set. Use `hyprwhspr keyboard list` to see available devices.

## Keyboard hotplug (docks, Bluetooth)

By default hyprwhspr watches for keyboards plugged in after startup and attaches them automatically:

```jsonc
{
  "keyboard_hotplug": true   // default; set false for startup-only discovery
}
```

**Restricting which keyboards are used (optional):** setting a `keyboard_device_names` allowlist limits attachment — at startup *and* on hot-plug — to just the listed devices. Use it if auto-discovery picks up a device you don't want grabbed, or to pin behavior to a known set of keyboards. Leave it unset to keep the default "attach any keyboard" behavior.

**Interactive configurator (to set the optional allowlist):**

```bash
hyprwhspr keyboard configure
```

This detects your keyboards and pre-selects the real ones (using udev's
keyboard/mouse classification so a fancy mouse isn't picked by mistake). It shows
the recommended set and lets you accept it as-is or adjust the list by number,
then writes `keyboard_device_names` and offers to restart the service so it takes
effect immediately.

If the device names are cryptic and you're not sure which one is your keyboard,
run `hyprwhspr keyboard detect` and press a key — it reports which device the
keypress came from (and its number in `keyboard configure`).

**Manual alternative** — edit `config.json` directly (run `hyprwhspr keyboard list` to find exact device names) and restart the service:

```jsonc
{
  "keyboard_device_names": [
    "AT Translated Set 2 keyboard",
    "SONiX USB Keyboard"
  ]
}
```

`selected_device_name` and `selected_device_path` take priority over this list if set.
