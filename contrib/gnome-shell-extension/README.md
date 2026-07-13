# hyprwhspr Waveform OSD (GNOME Shell extension)

A floating, audio-reactive **waveform overlay** for hyprwhspr dictation on
**GNOME/Mutter** — the one compositor where hyprwhspr's built-in `mic-osd`
visualizer can't run (it needs the `gtk4-layer-shell` protocol, which Mutter
does not implement, so hyprwhspr falls back to plain notifications there).

This extension fills that gap by drawing the overlay **inside gnome-shell**
itself, so it is **focus-safe** — it never steals keyboard focus and therefore
never disturbs the window your dictation is being typed into.

![placement](https://github.com/goodroot/hyprwhspr) <!-- bottom-centre pill while recording -->

## How it works

It is a pure status consumer — it reads the files hyprwhspr already writes and
animates accordingly. No audio capture of its own, no extra dependencies:

| File | Meaning |
|------|---------|
| `$XDG_RUNTIME_DIR/hyprwhspr/recording_status` | present ⇒ recording (show the pill) |
| `$XDG_RUNTIME_DIR/hyprwhspr/audio_level` | `0.000`–`1.000`, drives the bar heights |
| `$XDG_RUNTIME_DIR/hyprwhspr/transcript_preview` | optional live transcript line |

Older hyprwhspr versions wrote the first two under `~/.config/hyprwhspr/`;
the extension falls back to those paths automatically.

A pulsing record dot, 27 cyan→violet equalizer bars reacting to the live mic
level, and (when available) the live transcript text underneath.

## Install

```bash
./install.sh
```

Then **log out and back in** if it doesn't appear immediately (on Wayland a new
extension can't be hot-loaded). Start dictation and the pill fades in at the
bottom-centre of the screen.

Manual install (equivalent):

```bash
UUID="hyprwhspr-waveform@ninyawee.github.io"
cp -r "$UUID" ~/.local/share/gnome-shell/extensions/
gnome-extensions enable "$UUID"   # or log out/in, then enable
```

## Tweak the look

- **Colours / size / bar count:** the constants at the top of `extension.js`
  (`N_BARS`, `BAR_MIN`, `BAR_MAX`, `C1`, `C2`).
- **Pill style:** `stylesheet.css`.

After editing, reload with `gnome-extensions disable <uuid> && gnome-extensions enable <uuid>`
(no logout needed once it's been registered once).

## Requirements

- GNOME Shell 45–48 (Wayland or X11).
- hyprwhspr running (any backend) — the extension only visualizes its state.
