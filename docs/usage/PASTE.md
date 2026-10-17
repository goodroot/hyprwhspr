# Paste and clipboard

hyprwhspr copies dictated text to the clipboard, sends a paste shortcut, then restores your clipboard. Wayland uses `wl-clipboard`; Hyprland prefers its native shortcut dispatcher, while other Wayland sessions prefer `wtype`, with `ydotool key` as a fallback. X11 uses `python-pyperclip` with `xclip`, and `xdotool`/`xprop` for focused-window and terminal detection. `xsel` is also accepted as a clipboard fallback when installed. Most setups need no configuration. GNOME/Mutter and KDE Plasma need the AT-SPI bridge for window detection — see [GNOME and KDE](../integrations/DESKTOPS.md).

## Paste mode

The paste shortcut is auto-detected from the focused window:

- **Terminals** (Ghostty, Kitty, WezTerm, Alacritty, foot, …) → Ctrl+Shift+V
- **Everything else** (editors, browsers, chat apps) → Ctrl+V

Override with `paste_mode` if needed:

```jsonc
{
    "paste_mode": "ctrl_shift"  // "ctrl_shift" (terminal paste) | "ctrl" (GUI paste) | "super" | "alt"
}
```

The default is `null`, which auto-detects the appropriate paste chord from the
focused application. The legacy `shift_paste` setting also accepts `null` for
the same auto-detection behavior.

## App-specific paste keys

Some apps use non-standard paste shortcuts — GUI Emacs, for example, uses Ctrl+Y while Ctrl+V scrolls. Set per-app behavior with `applications`, keyed by window identifier:

```jsonc
{
    "applications": {
        "emacs": { "auto_paste": "ctrl+y" },  // custom paste chord
        "some-app": { "auto_paste": false }    // disable: nothing pasted, clipboard untouched
    }
}
```

Run `hyprwhspr config focused-window` to see the identifiers for the active window. Run straight from a terminal it just reports the terminal, so add a delay and focus the target window before it fires — the output still lands in your terminal:

```bash
sleep 3; hyprwhspr config focused-window   # then click into the app you want
```

Prefer stable app classes over window titles, which change with the open document.

`not detected` means no identifier is available, so `applications` rules can't match — the command says why and what to do about it.

> **Terminal Emacs** (`emacs -nw`, `emacsclient -t`): hyprwhspr sees the terminal, not Emacs, so terminal paste (Ctrl+Shift+V) is normally correct.

## Hyprland NumLock workaround

If pasting makes the focused app treat NumLock as off, try the optional ydotool
backend for [issue #283](https://github.com/goodroot/hyprwhspr/issues/283):

```json
{
    "hyprland_paste_backend": "ydotool"
}
```

This sends paste shortcuts and auto-submit Enter through ydotool exclusively.
The default, `"auto"`, keeps Hyprland's native shortcut dispatcher and its existing
fallbacks. The setting is ignored outside Hyprland. Restart hyprwhspr after changing
configuration; set it back to `"auto"` to revert. Invalid values warn and use `"auto"`.

Both `ydotool` and `ydotoold` must be installed, with access to `/dev/uinput` as
configured by setup. hyprwhspr starts its own private daemon lazily, so the first
paste can take a few extra seconds. See [ydotool troubleshooting](../TROUBLESHOOTING.md).

If ydotool reports a paste failure, hyprwhspr reports failed delivery, leaves the
text on the clipboard, and skips both fallback and auto-submit. In live typing,
the clipboard may contain only the failed chunk; use `hyprwhspr record copy-last`
to recover the retained dictation. A failed auto-submit Enter is logged without
retrying the paste or Enter.

This is a workaround, not a keyboard-state repair. Test NumLock and CapsLock in
your target apps, including the first paste after starting hyprwhspr. ydotool sends
physical keycodes: non-QWERTY layouts and modifier remaps may require the overrides
below. Non-Latin layouts are not automatically switched on Hyprland. A successful
command cannot guarantee that the focused app accepted the shortcut.

## Non-QWERTY layouts

`ydotool` sends physical Linux keycodes, so `Ctrl+KEY_V` may not be `Ctrl+v` on layouts like bepo or dvorak. To fix on Wayland: run `wev`, press the key that types `v`, and copy the printed keycode into `paste_keycode_wev`:

```jsonc
{
    "paste_keycode_wev": 55 // `wev` keycode for the key that types 'v' on your layout
}
```

If you already know the Linux evdev keycode, set `paste_keycode` directly. Non-Latin layouts (Thai, Russian, Arabic, …) are a special case handled on GNOME/Mutter — see [GNOME and KDE](../integrations/DESKTOPS.md#gnomemutter).

## Remapped modifier keys

`ydotool` sends physical keycodes, so desktop-level remaps can change the
generated paste shortcut. For example, with Ctrl and Caps Lock swapped:

```jsonc
{
    "ydotool_modifier_overrides": {
        "ctrl": "capslock"
    }
}
```

Supported modifiers are `ctrl`, `shift`, `alt`, and `super`. Key names are
case-insensitive and may include the `KEY_` prefix; Linux evdev keycodes are
also accepted. The usual `control`, `meta`, `cmd`, and `logo` aliases work here
as they do in paste chords. This setting affects ydotool, whether selected
explicitly or used as a fallback.

## Auto-submit

Automatically press Enter after pasting — aka Dictation YOLO. Handy for chat boxes and search fields; careful elsewhere.

```jsonc
{
    "auto_submit": true   // Send Enter key after paste (default: false)
}
```

## Clipboard behavior

`clipboard_settle_delay` controls the wait between a successful clipboard copy and
the paste shortcut (default `0.15` seconds). It accepts finite nonnegative numbers;
`0` skips the wait. Lower values are an advanced compatibility tradeoff: some
applications may paste stale clipboard contents. Direct typing is unaffected.

hyprwhspr saves your clipboard before injection and restores it afterward — dictated text never permanently overwrites it. To instead clear the clipboard a few seconds after pasting:

```jsonc
{
    "clipboard_behavior": true,         // true = clear clipboard after a delay (default: false = restore previous contents)
    "clipboard_clear_delay": 5.0        // seconds to wait before clearing (only when clipboard_behavior is true)
}
```

## Recover the last dictation

The daemon keeps the last prepared dictation in memory — text only, no audio or
disk history — so you can recover it when a paste does not land.

```bash
hyprwhspr record copy-last   # Copy to clipboard for manual paste
hyprwhspr record paste-last  # Deliver to the currently focused application
hyprwhspr record clear-last  # Forget the retained text
```

Bind these through your compositor so the destination keeps focus — in Hyprland,
`bind = SUPER ALT, V, exec, hyprwhspr record paste-last`. Recovery reuses the
exact prepared text without rerunning hooks, and never sends the auto-submit
Enter. Continuous mode retains the latest segment, not the whole session.
Dictation into an app with injection disabled is never retained.
