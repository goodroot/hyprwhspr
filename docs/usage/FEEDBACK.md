# Feedback

See it listening. Hear it start and stop.

## Themed visualizer

The recording-status indicator — the **mic OSD** — gives visual feedback while recording and auto-matches Noctalia & Omarchy themes.

```jsonc
{
  "mic_osd_enabled": true
}
```

### Display mode depends on your session

`mic_osd_enabled` turns the mic OSD on; *how* it's shown is chosen automatically at startup:

- **Overlay mode** — the animated always-on-top overlay. On Wayland it needs a layer-shell compositor such as Hyprland, Sway, niri or KDE Plasma, plus GTK4, PyCairo and `gtk4-layer-shell`. On X11 it needs GTK3 and PyCairo. Without a compositing manager, it sits on a black box.
- **Notification mode** — GNOME on Wayland gets desktop notifications instead: recording, transcribing, inserted. They never steal the paste's focus. Requires a notification service and `notify-send` or `gdbus`. The result replaces the processing banner. Timing depends on the desktop.

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

Shows the last few words while recording. Opt-in; needs `mic_osd_style: pill` plus a `realtime-ws` backend that streams partial transcripts (OpenAI, ElevenLabs, Deepgram, self-hosted — not Gemini yet). Final transcription and paste are unaffected.

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

## Bar integrations

Add recording controls and service status to [Waybar](../integrations/WAYBAR.md),
[Noctalia](../integrations/NOCTALIA.md), or a custom
[Quickshell bar](../integrations/QUICKSHELL.md). The Quickshell button also shows
a microphone level meter and transcription spinner, even with `mic_osd_enabled: false`.

Custom bars can read `${XDG_RUNTIME_DIR}/hyprwhspr/visualizer_state` in overlay
and notification modes, and when the OSD is disabled. It reports recording,
paused, processing, success, or error and is removed when recording feedback
ends. The separate `audio_level` file contains the normalized microphone level
during regular dictation.

## Notifications

Choose how much the service tells you. These are the defaults; add only the settings you want to change to [your config](../CONFIGURATION.md), then restart hyprwhspr.

```jsonc
{
  "notification_level": "all",
  "notification_timeout_ms": 5000
}
```

| Level | Shows |
|---|---|
| `all` (default) | Problems, command results, model loading/unloading, and recovery that needs retries. |
| `action_required` | Problems and command results; quiet automatic lifecycle changes. |
| `off` | No service or Waybar click alerts. Logs remain available. |

**Recording status is separate.** `mic_osd_enabled` controls the overlay or recording-status bubble at every level, including `off`. To disable all desktop notifications, set both `notification_level: "off"` and `mic_osd_enabled: false`. Audio feedback has its [own toggle](#audio-feedback).

The first automatic recovery attempt, including after suspend, is silent. If retries are needed, one message explains what is recovering; its outcome replaces it. A blocked recording attempt gets immediate feedback. Microphone and connection recovery keep separate alerts.

| When | What you see |
|---|---|
| The idle timer unloads the model | “Model unloaded after inactivity — reloads on next recording” |
| You record while it loads | “Model loading — keep speaking”, then “Model loading — audio captured” if you stop before it is ready |
| Microphone recovery needs retries | “Recovering microphone…”, then “Microphone recovered” on success |
| Text reaches only the clipboard | “Text copied — paste to insert” |
| No speech is recognized | “No speech recognized — try recording again” |
| A failed model load leaves saved audio | “Transcription failed — audio kept”, with the command to transcribe that file |

A hook that consumes the text shows “Handled by hook”; a capture client shows “✓ Done”. Neither claims the text was inserted. See [memory management](../backends/MEMORY.md#unload-when-idle) for model wake behavior and [long-form recording](RECORDING_MODES.md#long-form-mode) for retrying a saved session.

Temporary alerts use `notification_timeout_ms` (5000 ms by default). The service explicitly closes them after at least four seconds; `0` does not make them permanent. The desktop may use its own banner duration. Recording status has its own lifecycle; clipboard and no-speech results get extra reading time.

Final recovery failures and saved-audio instructions stay until dismissed or resolved. Each saved file keeps its own alert, even after a later successful dictation. Individual microphone-start failures and disconnection notices are temporary, so deliberately muting or unplugging a mic does not leave a permanent warning.

## Tray icon

A system tray icon shows that hyprwhspr is running and what it's doing. It works in any standard tray (StatusNotifierItem): KDE Plasma, XFCE, Waybar's `tray` module, and Quickshell shells such as DankMaterialShell. GNOME needs the AppIndicator extension.

| Icon | State |
|---|---|
| Microphone | Ready to record |
| Record dot | Recording |
| Pause | Long-form session paused |
| Muted microphone | Model loading or unloaded, or transcribing |
| Warning | Microphone, model or long-form submit problem (the tooltip says which) |

Click it for a menu to start or stop recording, unload or load the model, and restart or stop the service. In long-form, it pauses, submits, retries or cancels. Restart and Stop only appear when the service runs as `hyprwhspr.service`.

The icon disappears when the service stops, so no icon means hyprwhspr isn't running. It needs PyGObject, which the visualizer also uses.

It's off by default; setup offers it when you skip the bar integrations. If you use the [Waybar module](../integrations/WAYBAR.md), [Noctalia widget](../integrations/NOCTALIA.md), or [Quickshell button](../integrations/QUICKSHELL.md), you can leave it off. To turn it on:

```jsonc
{
  "tray_icon_enabled": true
}
```

Restart the service after changing it.

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
