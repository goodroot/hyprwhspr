<h1 align="center">
    hyprwhspr
</h1>

<p align="center">
    <b>Native speech-to-text for Arch / Omarchy</b> - Fast, accurate and easy system-wide dictation
</p>

<p align="center">
    instant performance | most accurate local models | realtime cloud streaming | themed visuals | supports any cloud provider
</p>

 <p align="center">
    <i>Matches your theme!</i>
 </p>

https://github.com/user-attachments/assets/4c223e85-2916-494f-b7b1-766ce1bdc991

---

- **Optimized for Arch Linux** - Seamless integration with Arch Linux via the AUR
- **Local, very fast defaults** - Instant and accurate speech recognition via in memory [Whisper](https://github.com/goodroot/hyprwhspr?tab=readme-ov-file#whisper-models)
- **Latest local models with GPU support**: Whisper turbo-v3? [Parakeet-v3](https://github.com/goodroot/hyprwhspr?tab=readme-ov-file#parakeet-nvidia)? Use GPU for incredible speed
- **Conversation mode** - Send text to Cloud API and receive LLM response in return
- **Themed visualizer** - Visual feedback when recording, matched to your Omarcy theme
- **Supports any cloud API** -  Use a cloud stt service or any custom localhost model
- **Word overrides and multi-language** - Customize transcriptions, prompt and corrections, in your language
- **Auto-paste anywhere** - Pastes in active buffer without additional keypresses

## Quick start

### Prerequisites

- **[Omarchy](https://omarchy.org/)** or **[Arch Linux](https://archlinux.org/)**
- **NVIDIA GPU** (optional, for CUDA acceleration)
- **AMD GPU** (optional, for ROCm acceleration)

### Quick start

On the AUR:

```bash
# Install for stable
yay -S hyprwhspr

# Or install for bleeding edge
yay -S hyprwhspr-git
```

Then run through the interactive setup:

```bash
# Run interactive setup
hyprwhspr setup
```

**The setup will walk you through the process:**

1. ✅ Configure transcription backend (pywhispercpp, Parakeet-v3, REST API, or Realtime WebSocket)
2. ✅ Download models (if using pywhispercpp backend)
3. ✅ Configure themed visualizer for maximum coolness
4. ✅ Configure Waybar integration (if Waybar is installed)
5. ✅ Set up systemd user services 
6. ✅ Set up permissions
7. ✅ Validate installation

### First use

> Ensure your microphone of choice is available in audio settings!

1. **Log out and back in** (for group permissions)
2. **Press `Super+Alt+D`** to start dictation - _beep!_
3. **Speak naturally**
4. **Press `Super+Alt+D`** again to stop dictation - _boop!_
5. **Bam!** Text appears in active buffer!

Any snags, please [create an issue](https://github.com/goodroot/hyprwhspr/issues/new/choose) or visit [Omarchy Discord](https://discord.com/channels/1390012484194275541/1410373168765468774).

### Updating

```bash
# Update via your AUR helper
yay -Syu hyprwhspr

# If needed, re-run setup (idempotent)
hyprwhspr setup
```

### CLI Commands

After installation, use the `hyprwhspr` CLI to manage your installation:

- `hyprwhspr setup` - Interactive initial setup
- `hyprwhspr config` - Manage configuration (init/show/edit)
- `hyprwhspr waybar` - Manage Waybar integration (install/remove/status)
- `hyprwhspr mic-osd` - Manage microphone visualization overlay (enable/disable/status)
- `hyprwhspr systemd` - Manage systemd services (install/enable/disable/status/restart)
- `hyprwhspr model` - Manage models (download/list/status)
- `hyprwhspr status` - Overall status check
- `hyprwhspr validate` - Validate installation
- `hyprwhspr backend` - Backend management (repair/reset)
- `hyprwhspr state` - State management (show/validate/reset)
- `hyprwhspr uninstall` - Completely remove hyprwhspr and all user data

## Usage

### Global hotkey modes

hyprwhspr supports three configurable interaction modes:

**Toggle mode (default):**

- **`Super+Alt+D`** - Toggle dictation on/off

**Push-to-talk mode:**

- **Hold `Super+Alt+D`** - Start dictation
- **Release `Super+Alt+D`** - Stop dictation

**Auto mode (hybrid tap/hold):**

- **Tap** (< 400ms) - Toggle behavior: tap to start recording, tap again to stop
- **Hold** (>= 400ms) - Push-to-talk behavior: hold to record, release to stop

## Configuration

Edit `~/.config/hyprwhspr/config.json`:

**Minimal config** - only 2 essential options:

```jsonc
{
    "primary_shortcut": "SUPER+ALT+D",
    "model": "base.en"
}
```

**Toggle hotkey mode** (default) - press to start, press again to stop:

```jsonc
{
    "recording_mode": "toggle"
}
```

**Push-to-talk mode** - hold to record, release to stop:

```jsonc
{
    "recording_mode": "push_to_talk"
}
```

**Auto mode (hybrid tap/hold)** - automatically detects your intent:

```jsonc
{
    "recording_mode": "auto"
}
```

- **Tap** (< 400ms) - Toggle behavior: tap to start recording, tap again to stop
- **Hold** (>= 400ms) - Push-to-talk behavior: hold to record, release to stop

**Realtime WebSocket** - low-latency streaming via OpenAI's Realtime API:

Two modes available:

- **transcribe** (default) - Pure speech-to-text, more expensive than HTTP
- **converse** - Voice-to-AI: speak and get AI responses

```jsonc
{
    "transcription_backend": "realtime-ws",
    "websocket_provider": "openai",
    "websocket_model": "gpt-realtime-mini-2025-12-15",
    "realtime_mode": "transcribe",       // "transcribe" or "converse"
    "realtime_timeout": 30,              // Completion timeout (seconds)
    "realtime_buffer_max_seconds": 5     // Max audio buffer before dropping chunks
}
```

**REST API** - use any ASR backend via HTTP API (local or cloud):

**Local Parakeet v3**

Fastest, latest, and apparently the best! GPU accel recommended, not required.

**OpenAI**

Bring an API key from OpenAI, and choose from:

- **GPT-4o Transcribe** - Latest model with best accuracy
- **GPT-4o Mini Transcribe** - Faster, lighter model
- **GPT-4o Mini Transcribe (2025-12-15)** - Updated version of the faster, lighter transcription model
- **GPT Audio Mini (2025-12-15)** - General purpose audio model
- **Whisper 1** - Legacy Whisper model

> For realtime streaming transcription, use the `realtime-ws` backend (see above) with **GPT Realtime Mini (2025-12-15)**.

**Groq**

Bring an API key from Grok, and choose from:

- **Whisper Large V3** - High accuracy processing
- **Whisper Large V3 Turbo** - Fastest transcription speed

**Any arbitrary backend:**

Or connect to any backend, local or cloud, via your own custom backend:

```jsonc
{
    "transcription_backend": "rest-api",
    "rest_endpoint_url": "https://your-server.example.com/transcribe",
    "rest_headers": {                     // optional arbitrary headers
        "authorization": "Bearer your-api-key-here"
    },
    "rest_body": {                        // optional body fields merged with defaults
        "model": "custom-model"
    },
    "rest_api_key": "your-api-key-here",  // equivalent to rest_headers: { authorization: Bearer your-api-key-here }
    "rest_timeout": 30                    // optional, default: 30
}
```

**Custom hotkey** - extensive key support:

```json
{
    "primary_shortcut": "CTRL+SHIFT+SPACE"
}
```

Supported key types:

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

```json
{
    "primary_shortcut": "SUPER+KEY_COMMA"
}
```

Examples:

- `"SUPER+SHIFT+M"` - Super + Shift + M
- `"CTRL+ALT+F1"` - Ctrl + Alt + F1
- `"F12"` - Just F12 (no modifier)
- `"RCTRL+RSHIFT+ENTER"` - Right Ctrl + Right Shift + Enter

**Themed visualizer** - visual feedback, matches your Omarchy theme:

```json
{
  "mic_osd_enabled": true,
}
```

**Word overrides** - customize transcriptions:

```json
{
    "word_overrides": {
        "hyperwhisper": "hyprwhspr",
        "omarchie": "Omarchy"
    }
}
```

**Audio feedback** - optional sound notifications:

```jsonc
{
    "audio_feedback": true,            // Enable audio feedback (default: false)
    "audio_volume": 0.5,               // General audio volume fallback (0.1 to 1.0, default: 0.5)
    "start_sound_volume": 0.5,         // Start recording sound volume (0.1 to 1.0, default: 0.5)
    "stop_sound_volume": 0.5,          // Stop recording sound volume (0.1 to 1.0, default: 0.5)
    "error_sound_volume": 0.5,         // Error sound volume (0.1 to 1.0, default: 0.5)
    "start_sound_path": "custom-start.ogg",  // Custom start sound (relative to assets)
    "stop_sound_path": "custom-stop.ogg",    // Custom stop sound (relative to assets)
    "error_sound_path": "custom-error.ogg"  // Custom error sound (relative to assets)
}
```

**Default sounds included:**

- **Start recording**: `ping-up.ogg` (ascending tone)
- **Stop recording**: `ping-down.ogg` (descending tone)
- **Error/blank audio**: `ping-error.ogg` (double-beep)

**Custom sounds:**

- **Supported formats**: `.ogg`, `.wav`, `.mp3`
- **Fallback**: Uses defaults if custom files don't exist

**Text replacement:** Automatically converts spoken words to symbols / punctuation:

**Punctuation:**

- "period" → "."
- "comma" → ","
- "question mark" → "?"
- "exclamation mark" → "!"
- "colon" → ":"
- "semicolon" → ";"

**Symbols:**

- "at symbol" → "@"
- "hash" → "#"
- "plus" → "+"
- "equals" → "="
- "dash" → "-"
- "underscore" → "_"

**Brackets:**

- "open paren" → "("
- "close paren" → ")"
- "open bracket" → "["
- "close bracket" → "]"
- "open brace" → "{"
- "close brace" → "}"

**Special commands:**

- "new line" → new line
- "tab" → tab character

_Speech-to-text replacement list via [WhisperTux](https://github.com/cjams/whispertux), thanks @cjams!_

**Clipboard behavior** - control what happens to clipboard after text injection:

```jsonc
{
    "clipboard_behavior": false,       // Boolean: true = clear after delay, false = keep (default: false)
    "clipboard_clear_delay": 5.0      // Float: seconds to wait before clearing (default: 5.0, only used if clipboard_behavior is true)
}
```

- **`clipboard_behavior: true`** - Clipboard is automatically cleared after the specified delay
- **`clipboard_clear_delay`** - How long to wait before clearing (only matters when `clipboard_behavior` is `true`)

**Paste behavior** - control how text is pasted into applications:

```jsonc
{
    "paste_mode": "ctrl_shift"   // "super" | "ctrl_shift" | "ctrl"  (default: "ctrl_shift")
}
```

**Paste behavior options:**

- **`"ctrl_shift"`** (default) — Sends Ctrl+Shift+V. Works in most terminals.

- **`"super"`** — Sends Super+V. Omarchy default. Maybe finicky.

- **`"ctrl"`** — Sends Ctrl+V. Standard GUI paste.

**Add dynamic tray icon** to your `~/.config/waybar/config`:

```json
{
    "custom/hyprwhspr": {
        "exec": "/usr/lib/hyprwhspr/config/hyprland/hyprwhspr-tray.sh status",
        "interval": 2,
        "return-type": "json",
        "exec-on-event": true,
        "format": "{}",
        "on-click": "/usr/lib/hyprwhspr/config/hyprland/hyprwhspr-tray.sh toggle",
        "on-click-right": "/usr/lib/hyprwhspr/config/hyprland/hyprwhspr-tray.sh restart",
        "tooltip": true
    }
}
```

**Add CSS styling** to your `~/.config/waybar/style.css`:

```css
@import "/usr/lib/hyprwhspr/config/waybar/hyprwhspr-style.css";
```

**Waybar icon click interactions**:

- **Left-click**: Toggle Hyprwhspr on/off
- **Right-click**: Start Hyprwhspr (if not running)
- **Middle-click**: Restart Hyprwhspr

## Whisper (OpenAI)

**Default model installed:** `ggml-base.en.bin` (~148MB) to `~/.local/share/pywhispercpp/models/`

**GPU Acceleration (NVIDIA & AMD):**

- NVIDIA (CUDA) and AMD (ROCm) are detected automatically; pywhispercpp will use GPU when selected

- **⚠️ AMD ROCm 7.x / HIPBLAS** ROCm 7.0+ introduced breaking changes to hipBLAS datatype signatures. As of now, ggml’s HIP backend is compatible with ROCm 6.x, but ROCm 7.x will fail to build with errors and fallback to CPU.

**CPU performance options** - improve cpu transcription speed:

```jsonc
{
    "threads": 4            // thread count for whisper cpu processing
}
```

**Available models to download:**

- **`tiny`** - Fastest, good for real-time dictation
- **`base`** - Best balance of speed/accuracy (recommended)
- **`small`** - Better accuracy, still fast
- **`medium`** - High accuracy, slower processing
- **`large`** - Best accuracy, **requires GPU acceleration** for reasonable speed
- **`large-v3`** - Latest large model, **requires GPU acceleration** for reasonable speed

**⚠️ GPU required:** Models `large` and `large-v3` require GPU acceleration to perform. 

```bash
cd ~/.local/share/pywhispercpp/models/

# Tiny models (fastest, least accurate)
wget https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin
wget https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin

# Base models (good balance)
wget https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin
wget https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin

# Small models (better accuracy)
wget https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.en.bin
wget https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin

# Medium models (high accuracy)
wget https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.en.bin
wget https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.bin

# Large models (best accuracy, requires GPU)
wget https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large.bin
wget https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3.bin
```

**Update config after downloading:**

```jsonc
{
    "model": "small.en" // Or just small if multi-lingual model. If both available, general model is chosen.
}
```

**Language detection** - control transcription language:

English only speakers use `.en` models which are smaller.

For multi-language detection, ensure you select a model which does not say `.en`:

```jsonc
{
    "language": null // null = auto-detect (default), or specify language code
}
```

Language options:

- **`null`** (default) - Auto-detect language from audio
- **`"en"`** - English transcription
- **`"nl"`** - Dutch transcription  
- **`"fr"`** - French transcription
- **`"de"`** - German transcription
- **`"es"`** - Spanish transcription
- **`etc.`** - Any supported language code

**Whisper prompt** - customize transcription behavior:

```json
{
    "whisper_prompt": "Transcribe with proper capitalization, including sentence beginnings, proper nouns, titles, and standard English capitalization rules."
}
```

The prompt influences how Whisper interprets and transcribes your audio, eg:

- `"Transcribe as technical documentation with proper capitalization, acronyms and technical terminology."`

- `"Transcribe as casual conversation with natural speech patterns."`
  
- `"Transcribe as an ornery pirate on the cusp of scurvy."`

## Parakeet (Nvidia)

Whisper is the default, but any model works via API.

Select Parakeet within `hyprwhspr setup`.

## Troubleshooting

### Reset Installation

If you're having persistent issues, you can completely reset hyprwhspr:

```bash
# Stop services
systemctl --user stop hyprwhspr ydotool

# Remove runtime data
rm -rf ~/.local/share/hyprwhspr/

# Remove user config
rm -rf ~/.config/hyprwhspr/

# Remove system files
sudo rm -rf /usr/lib/hyprwhspr/
```

And then...

```bash
# Then reinstall fresh via AUR
yay -S hyprwhspr
hyprwhspr setup
```

### Common issues

**I heard the sound, but don't see text!** 

It's fairly common in Arch and other distros for the microphone to need to be plugged in and set each time you log in and out of your session, including during a restart. Within sound options, ensure that the microphone is indeed set. The sound utility will show feedback from the microphone if it is.

**Hotkey not working:**

```bash
# Check service status for hyprwhspr
systemctl --user status hyprwhspr.service

# Check logs
journalctl --user -u hyprwhspr.service -f
```

```bash
# Check service statusr for ydotool
systemctl --user status ydotool.service

# Check logs
journalctl --user -u ydotool.service -f
```

**Permission denied:**

```bash
# Fix uinput permissions
hyprwhspr setup

# Log out and back in
```

**No audio input:**

If your mic _actually_ available?

```bash
# Check audio devices
pactl list short sources

# Restart PipeWire
systemctl --user restart pipewire
```

**Audio feedback not working:**

```bash
# Check if audio feedback is enabled in config
cat ~/.config/hyprwhspr/config.json | grep audio_feedback

# Verify sound files exist
ls -la /usr/lib/hyprwhspr/share/assets/

# Check if ffplay/aplay/paplay is available
which ffplay aplay paplay
```

**Model not found:**

```bash
# Check if model exists
ls -la ~/.local/share/pywhispercpp/models/

# Download a different model
cd ~/.local/share/pywhispercpp/models/
wget https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin

# Verify model path in config
cat ~/.config/hyprwhspr/config.json | grep model
```

**Stuck recording state:**

```bash
# Check service health and auto-recover
/usr/lib/hyprwhspr/config/hyprland/hyprwhspr-tray.sh health

# Manual restart if needed
systemctl --user restart hyprwhspr.service

# Check service status
systemctl --user status hyprwhspr.service
```

**Keyboard remappers (keyd / kmonad)**:

If you use a keyboard remapping daemon that grabs evdev devices (e.g. `keyd`, `kmonad`), set:

```json
{
  "grab_keys": false
}
```

This prevents hyprwhspr from taking exclusive control of keyboards and allows it to listen to events normally.

> When grab_keys is disabled, the shortcut is not suppressed and may also trigger other system keybindings.


## Getting help

1. **Check logs**: `journalctl --user -u hyprwhspr.service` `journalctl --user -u ydotool.service`
2. **Verify permissions**: Run the permissions fix script
3. **Test components**: Check ydotool, audio devices, whisper.cpp
4. **Report issues**: [Create an issue](https://github.com/goodroot/hyprwhspr/issues/new/choose) or visit [Omarchy Discord](https://discord.com/channels/1390012484194275541/1410373168765468774) - logging info helpful!

## License

MIT License - see [LICENSE](LICENSE) file.

## Contributing

Create an issue, happy to help!  

For pull requests, also best to start with an issue.

---

**Built with ❤️ in 🇨🇦 for the Omarchy community**

*Integrated and natural speech-to-text.*
