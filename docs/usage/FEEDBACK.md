# Feedback

See it listening. Hear it start and stop.

## Themed visualizer

The recording-status indicator — the **mic OSD** — gives visual feedback while recording and auto-matches Noctalia & Omarchy themes.

> Highly recommended!

```jsonc
{
  "mic_osd_enabled": true
}
```

### Display mode depends on your compositor

`mic_osd_enabled` turns the mic OSD on; *how* it's shown is chosen automatically at startup:

- **Overlay mode** — compositors with layer-shell support (Hyprland, Sway, niri, KDE Plasma Wayland) get the animated always-on-top overlay. Requires GTK4, PyCairo, and `gtk4-layer-shell`.
- **Notification mode** — GNOME/Mutter and X11 sessions use desktop notifications (recording / transcribing / inserted), which never steal the focus the paste needs. The layer-shell overlay is Wayland-only. Notifications require `notify-send` (libnotify). "Transcribing…" stays until the text lands; GNOME keeps its own clock.

Set `mic_osd_enabled: false` to turn off both. The service log records which mode was selected:

```bash
journalctl --user -u hyprwhspr.service | grep -E 'Mic-OSD daemon started|status via notifications'
```

### Overlay styles

In overlay mode, `mic_osd_style` picks one of three visualizations (notification mode ignores it):

- `waveform` — the full themed waveform with transcript preview (default)
- `vu_meter` — a VU meter
- `pill` — a compact monochrome status pill: idle dots, live bars while recording, a travelling wave while processing, a pulse on error, and a checkmark on success. Can optionally show an animated live transcript on realtime backends that stream partial results (OpenAI, ElevenLabs).

![Pill OSD states](assets/pill-states.png)

```jsonc
{
  "mic_osd_style": "pill"
}
```

Restart the service after changing the style:

```bash
systemctl --user restart hyprwhspr
```

### Pill live transcript

Shows the last few words while recording. Opt-in; needs `mic_osd_style: pill` plus a `realtime-ws` backend that streams partial transcripts (OpenAI, ElevenLabs, self-hosted — not Gemini yet). Final transcription and paste are unaffected.

```jsonc
{
  "transcription_backend": "realtime-ws",
  "websocket_provider": "elevenlabs",
  "websocket_model": "scribe_v2_realtime",
  "realtime_mode": "transcribe",
  "mic_osd_style": "pill",
  "mic_osd_pill_transcript_enabled": true
}
```

| Setting | Default | Description |
|---|---:|---|
| `mic_osd_pill_transcript_enabled` | `false` | Enable the pill transcript. |
| `mic_osd_pill_transcript_word_limit` | `4` | Recent words shown (1–12). |
| `mic_osd_pill_transcript_idle_timeout_ms` | `1400` | Hide after this much idle time; `0` disables. |

## Audio feedback

Optional sound notifications:

```jsonc
{
    "audio_feedback": true,            // Enable audio feedback (default: true)
    "audio_volume": 0.5,               // General audio volume fallback (0.1 to 1.0, default: 0.5)
    "start_sound_volume": 1.0,         // Start recording sound volume (0.1 to 1.0, default: 1.0)
    "stop_sound_volume": 1.0,          // Stop recording sound volume (0.1 to 1.0, default: 1.0)
    "error_sound_volume": 0.5,         // Error sound volume (0.1 to 1.0, default: 0.5)
    "start_sound_path": "custom-start.ogg",  // Custom start sound (relative to assets)
    "stop_sound_path": "custom-stop.ogg",    // Custom stop sound (relative to assets)
    "error_sound_path": "custom-error.ogg"  // Custom error sound (relative to assets)
}
```

Default sounds included:

- **Start recording**: `ping-up.ogg` (ascending tone)
- **Stop recording**: `ping-down.ogg` (descending tone)
- **Error/blank audio**: `ping-error.ogg` (double-beep)

Custom sounds:

- **Supported formats**: `.ogg`, `.wav`, `.mp3`
- **Fallback**: Uses defaults if custom files don't exist

## Audio ducking

Quiet other audio on record:

```jsonc
{
  "audio_ducking": true,
  "audio_ducking_mode": "duck",
  "audio_ducking_percent": 50
}
```

- `audio_ducking: true` — set true to quiet other audio while recording
- `audio_ducking_mode: "duck"` — `"duck"` lowers volume; `"pause"` pauses media players instead
- `audio_ducking_percent: 50` — how much to reduce volume BY (default 50 = reduce to 50% of original; 70 = reduce to 30%); in pause mode, applies to apps without player controls

Ducking lowers each application stream's volume, not the device master — your speaker setting is untouched and shell volume OSDs don't fire on every recording. Streams that start mid-recording aren't ducked.

Halving a podcast's volume doesn't help; you still miss what was said. Pause mode pauses your players instead and resumes them afterwards, at the same position, with no volume change at all. It uses MPRIS (Firefox, Chromium, Spotify, mpv, VLC, most players); anything without it — game audio, calls, system sounds — is ducked as usual.

Pause mode needs `dbus-python`, an optional dependency installed by `scripts/install-deps.sh`; without it, it falls back to ducking. Only players hyprwhspr paused are resumed, and only if they're still paused when you stop.
