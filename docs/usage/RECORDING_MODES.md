# Recording modes

How a press becomes a recording. `recording_mode` picks one.

## Toggle mode

Press to start, press again to stop. The default:

```jsonc
{
    "recording_mode": "toggle"
}
```

## Push-to-talk mode

Hold to record, release to stop:

```jsonc
{
    "recording_mode": "push_to_talk",
    "push_to_talk_lock_seconds": 3.0  // Optional: hold this long to latch hands-free. 0 (default) = off.
}
```

- Hold past `push_to_talk_lock_seconds`, let go: recording **latches on**. Press again to end it.
- Shorter holds stop on release, as always.
- External bindings latch only on key-up `release`, never on `stop` — `stop` always stops.

## Auto mode

Tap or hold; it reads your intent:

```jsonc
{
    "recording_mode": "auto"
}
```

- **Tap** (< 400ms): toggle. Tap again to stop.
- **Hold** (≥ 400ms): push-to-talk. Release to stop.

## Continuous mode

Speak; each pause pastes. Recording rolls on until you press again:

```jsonc
{
    "recording_mode": "continuous",
    "continuous_silence_seconds": 2.0,  // The pause that pastes. Default 2.0.
    "continuous_silence_threshold": 0,  // Silence level. 0 (default) = calibrate each session.
    "silence_timeout": 15               // Optional: 15s of quiet ends the session.
}
```

- The final press pastes what's left.
- Detection off? The log prints the calibrated level; start `continuous_silence_threshold` from there.

## Auto-stop on silence

Go quiet and recording ends on its own — in toggle, auto, or continuous mode:

```jsonc
{
    "silence_timeout": 2.5  // Seconds of quiet. 0 (default) = off.
}
```

- Arms only once you speak; a slow first sentence is safe.
- Hears silence the way continuous mode does, same threshold.
- In continuous mode, choose a longer quiet (say `15`): pauses still paste, the long silence ends it.
- The stop beep marks it. Press to stop sooner.

## Long-form mode

Extended recording with pause/resume support:

```jsonc
{
    "recording_mode": "long_form",
    "long_form_submit_shortcut": "SUPER+ALT+E",  // Required: no default, must be set
    "long_form_temp_limit_mb": 500,              // Optional: max temp storage (default: 500 MB)
    "long_form_auto_save_interval": 300,         // Optional: auto-save interval in seconds (default: 300 = 5 minutes)
    "use_hypr_bindings": false                   // Optional: set true to use Hyprland compositor bindings
}
```

- Primary shortcut toggles recording/pause/resume
- Submit shortcut processes all recorded segments and pastes transcription
- Segments are auto-saved periodically to disk for crash recovery
- Old segments are automatically cleaned up when storage limit is reached
