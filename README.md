<h1 align="center">
    hyprwhspr
</h1>

<p align="center">
    <b>Native speech-to-text for Arch / Omarchy</b> - Fast, accurate and easy system-wide dictation
</p>

<p align="center">
    local or cloud | waybar integration | audio and visual feedback | any asr model! | gpu accel
</p>

 <p align="center">
    <i>pssst...un-mute!</i>
 </p>

https://github.com/user-attachments/assets/40cb1837-550c-4e6e-8d61-07ea59898f12

---

- **Optimized for Arch Linux / Omarchy** - Seamless integration with Arch Linux via the AUR
- **Local, very fast defaults** - State-of-the-art, fast speech recognition via in memory [Whisper](https://github.com/goodroot/hyprwhspr?tab=readme-ov-file#whisper-models)
- **Cross-platform GPU support** - Automatic detection and acceleration for NVIDIA (CUDA) / AMD (ROCm) 
- **Supports >any< ASR backend** - [Parakeet-v3](https://github.com/goodroot/hyprwhspr?tab=readme-ov-file#parakeet-nvidia)? Cloud API? New-thing? Use the API and templates
- **Word overrides** - Customize transcriptions, prompt and corrections
- **Multi-lingual** - Use a multi-language model and speak your own language
- **Paste anywhere** - Start record or push to talk, and paste in active buffer without additional keypresses

## Quick start

### Prerequisites

- **[Omarchy](https://omarchy.org/)** or **[Arch Linux](https://archlinux.org/)**
- **NVIDIA GPU** (optional, for CUDA acceleration)
- **AMD GPU** (optional, for ROCm acceleration)

### Quick start

For Arch Linux and Omarchy users, `hyprwhspr` is available via the AUR.

```
# Install hyprwhspr
yay -S hyprwhspr

# Run interactive setup
hyprwhspr setup
```

**The setup will:**

1. ✅ Configure transcription backend (pywhispercpp, Parakeet-v3 or REST API)
2. ✅ Set up systemd user services
3. ✅ Configure Waybar integration (if Waybar is installed)
4. ✅ Download models (if using pywhispercpp backend)
5. ✅ Set up permissions
6. ✅ Validate installation

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
- `hyprwhspr systemd` - Manage systemd services (install/enable/disable/status/restart)
- `hyprwhspr model` - Manage models (download/list/status)
- `hyprwhspr status` - Overall status check
- `hyprwhspr validate` - Validate installation

## Usage

### Global hotkey modes

hyprwhspr supports two configurable interaction modes:

**Toggle mode (default):**

- **`Super+Alt+D`** - Toggle dictation on/off

**Push-to-talk mode:**

- **Hold `Super+Alt+D`** - Start dictation
- **Release `Super+Alt+D`** - Stop dictation

## Configuration

Edit `~/.config/hyprwhspr/config.json`:

**Minimal config** - only 2 essential options:

```jsonc
{
    "primary_shortcut": "SUPER+ALT+D",
    "model": "base.en"
}
```

**Push-to-talk mode** - hold to record, release to stop:

```jsonc
{
    "push_to_talk": true
}
```

- **`push_to_talk: false`** (default) - Toggle mode: press to start, press again to stop
- **`push_to_talk: true`** - Push-to-talk mode: hold to record, release to stop

**REST API backends** - use any ASR backend via HTTP API (can run locally or remotely):

The installer will walk you through remote / Cloud endpoints:

**Parakeet v3**

Fastest, latest, and apparently the best! GPU accel rcommended, not required.

**OpenAI**

- **GPT-4o Transcribe** - Latest model with best accuracy
- **GPT-4o Mini Transcribe** - Faster, lighter model
- **Whisper 1** - Legacy Whisper model

**Groq**

- **Whisper Large V3** - High accuracy processing
- **Whisper Large V3 Turbo** - Fastest transcription speed

**Any arbitrary backend:**

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
    "start_sound_volume": 0.3,        // Start recording sound volume (0.1 to 1.0)
    "stop_sound_volume": 0.3,         // Stop recording sound volume (0.1 to 1.0)
    "start_sound_path": "custom-start.ogg",  // Custom start sound (relative to assets)
    "stop_sound_path": "custom-stop.ogg"     // Custom stop sound (relative to assets)
}
```

**Default sounds included:**

- **Start recording**: `ping-up.ogg` (ascending tone)
- **Stop recording**: `ping-down.ogg` (descending tone)

**Custom sounds:**

- **Supported formats**: `.ogg`, `.wav`, `.mp3`
- **Fallback**: Uses defaults if custom files don't exist

_Thanks for [the sounds](https://github.com/akx/Notifications), @akx!_

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
        "on-click-right": "/usr/lib/hyprwhspr/config/hyprland/hyprwhspr-tray.sh start",
        "on-click-middle": "/usr/lib/hyprwhspr/config/hyprland/hyprwhspr-tray.sh restart",
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

See [hyprwhspr-backends](https://github.com/goodroot/hyprwhspr-backends) for the [Parakeet-tdt-0.6b-v3](https://github.com/goodroot/hyprwhspr-backends/tree/main/backends/parakeet-tdt-0.6b-v3) example.

After that, setup the following to match your backend, and then restart hyprwhspr:

```jsonc
{
    "transcription_backend": "rest-api",
    "rest_endpoint_url": "https://127.0.0.1:8080/transcribe",
    "rest_headers": {
        "authorization": "Bearer your-api-key-here",
        "x-model": "parakeet-tdt-0.6b-v3"
    },
    "rest_body": {
        "temperature": "0.0"
    },
    "rest_api_key": "your-api-key-here",  // optional, equivalent to rest_headers: { authorization: Bearer your-api-key-here }
    "rest_timeout": 60                    // optional, default: 30
}
```

Uses local Python and optionally systemd. Works great with GPU, or set the CPU flag.

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
