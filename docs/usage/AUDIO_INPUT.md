# Audio input

By default hyprwhspr follows your system's default input source, so changing your microphone in desktop sound settings just works. To pin one instead:

```jsonc
{
  "audio_device_name": "Elgato"
}
```

`audio_device_name` and `audio_device_id` accept the same three forms:

- **A source name** — `alsa_input.usb-Elgato_Systems_Elgato_Wave_XLR_ABC123-00.analog-stereo`. Most reliable; list yours with `pactl list short sources`.
- **A name substring** — `Elgato`. Convenient and stable across reboots. If it matches several sources, your default source wins the tie; otherwise hyprwhspr refuses to guess and falls back.
- **A device index** — `"audio_device_id": 2`. Indices move across reboots, so prefer a name.

On PipeWire/PulseAudio a pinned mic is captured *through* the sound server:

```text
microphone → PipeWire/PulseAudio source → processing (EasyEffects, noise suppression, echo cancellation) → hyprwhspr
```

So anything applied to that source reaches hyprwhspr, and you can pin a virtual source directly:

```jsonc
{
  "audio_device_name": "easyeffects_source"
}
```

If your mic feeds a filter chain, pin the chain's source rather than the hardware. A chain is a separate source, so pinning `Elgato` captures the Elgato source itself — in the graph, but upstream of the processing.

### Capturing raw hardware

An `hw:` (or `plughw:`) prefix opens the ALSA device directly instead:

```jsonc
{
  "audio_device_name": "hw:1,0"
}
```

Raw capture skips **all** sound-server processing and takes the card exclusively, so other apps cannot use the microphone at the same time. When a sound server is running, hyprwhspr warns at startup whenever capture lands on a raw `hw:` device — including when it falls back to one because nothing matched your setting.

## Audio stream keepalive

By default hyprwhspr opens the microphone only while recording. On some hardware (certain USB mics on raw ALSA), the audio device suspends between uses and the first recording after an idle period fails with a `paTimedOut` error. If you see this, enable the keepalive stream:

```jsonc
{
  "keepalive_stream": true
}
```

This holds a silent input stream open in the background so the device stays warm.

**Leave this off unless you need it** — an open input stream triggers the microphone-in-use indicator on most desktops (GNOME, KDE, Ubuntu, etc.), making it appear as though hyprwhspr is always listening. It's not!

## Debug recordings

Hear what the model heard. Keep the last 3 recordings as `.wav` in `$XDG_RUNTIME_DIR/hyprwhspr/recordings` (tmpfs; gone at logout):

```jsonc
{
  "debug_recordings": true
}
```

Audio only, no transcripts. In continuous mode, each pasted chunk counts as one. Long-form keeps its own segments.

## Mute detection

Lets you know when you're disconnected. Mute detection can conflict with Bluetooth microphones — disable it if so:

```jsonc
{
  "mute_detection": false
}
```

Silent recordings are still rejected. Test live input with `hyprwhspr test --live`.
