# Command line

Drive the service from scripts, launchers and [any hotkey system](HOTKEYS.md#external-hotkey-systems).

## Recording

```bash
# Start recording
hyprwhspr record start

# Start recording with specific language
hyprwhspr record start --lang it    # Italian
hyprwhspr record start --lang de    # German
hyprwhspr record start --lang es    # Spanish

# Stop recording (transcribes and pastes)
hyprwhspr record stop

# Push-to-talk key-up: stop, or latch a long hold
hyprwhspr record release

# Cancel recording (discards audio, no transcription)
hyprwhspr record cancel

# Toggle recording on/off
hyprwhspr record toggle
hyprwhspr record toggle --lang it   # Toggle with language override

# Check current status
hyprwhspr record status

# Capture: trigger a recording and stream the transcription to stdout
# Blocks until transcription is complete. Suppresses text injection — use for scripting/piping.
# Self-triggers a recording if none is in progress; attaches to an in-flight recording if one is.
hyprwhspr record capture
hyprwhspr record capture --lang it   # Capture with language override

# Diagnose how the running daemon would process a captured transcript
hyprwhspr record capture --trace-processing
```

## Trace processing

`--trace-processing` prints one UTF-8 JSON document using the running daemon's settings. It reports exact `raw` and
`preprocessed` text, backend/model, recording and silence settings, symbol/hook state, `vad_mode`, and `boundary_mode`.
It does not run hooks, append space, save the transcript, or inject text.

`vad_mode`: `none`; `silero_filter` (pre-inference filtering); `silero_segmented` (ONNX segmentation after the reported
duration gate); `server_vad`; `manual_commit`; or `provider_managed`. `boundary_mode` is `manual_stop`,
`silence_auto_stop`, or `continuous_silence`. Continuous silence flushing and realtime server VAD can turn pauses into
independently punctuated segments. Trace does not change VAD or silence defaults.

## File transcription

Files in. Words out. WAV and MP3; local and REST backends.

```bash
hyprwhspr transcribe recording.mp3
hyprwhspr transcribe recording.wav -o transcript.txt
hyprwhspr transcribe recording.wav --lang fr --clean
```

Stdout by default; `-o` writes UTF-8 text. `--lang` sets the language. `--clean`
applies configured cleanup, without pasting or running hooks.

An idle service lends its loaded model. A busy service says no. No service: the
command loads the backend itself.

File transcription does not work with the realtime WebSocket backend.
