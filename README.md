<h1 align="center">
    hyprwhspr
</h1>

<p align="center">
    <b>Native speech-to-text for Arch / Omarchy</b> - Fast, accurate and easy system-wide dictation
</p>

<p align="center">
all local | waybar integration | audio feedback | auto-paste | cpu or gpu | easy setup
</p>

 <p align="center">
    <i>pssst...un-mute!</i>
 </p>

https://github.com/user-attachments/assets/40cb1837-550c-4e6e-8d61-07ea59898f12

---

- **Optimized for Arch Linux / Omarchy** - Seamless integration with [Omarchy](https://omarchy.org/) / [Hyprland](https://github.com/hyprwm/Hyprland) & [Waybar](https://github.com/Alexays/Waybar)
- **Whisper-powered** - State-of-the-art speech recognition via [OpenAI's Whisper](https://github.com/openai/whisper)
- **Cross-platform GPU support** - Automatic detection and acceleration for NVIDIA (CUDA) / AMD (ROCm) 
- **Hot model loading** - pywhispercpp backend keeps models in memory for _fast_ transcription
- **Word overrides** - Customize transcriptions, prompt and corrections
- **Run as user** - Runs in user space, just sudo once for the installer

## Quick start

### Prerequisites

- **[Omarchy](https://omarchy.org/)**
- **NVIDIA GPU** (optional, for CUDA acceleration)
- **AMD GPU** (optional, for ROCm acceleration)

### Installation

"Just works" with Omarchy.

**AUR:**

_New!_

```bash
# Install package
yay -S hyprwhspr

# Setup package
hyprwhspr-setup
```

**Script:**

```bash
# Clone the repository
git clone https://github.com/goodroot/hyprwhspr.git
cd hyprwhspr

# Run the automated installer
./scripts/install-omarchy.sh
```

**The installer will:**

1. ✅ Install system dependencies (ydotool, etc.)
2. ✅ Copy application files to system directory (`/usr/lib/hyprwhspr`)
3. ✅ Set up Python virtual environment in user space (`~/.local/share/hyprwhspr/venv`)
4. ✅ Install pywhispercpp backend
5. ✅ Download base model to user space (`~/.local/share/pywhispercpp/models/ggml-base.en.bin`)
6. ✅ Set up systemd services for hyprwhspr & ydotoolds
7. ✅ Configure Waybar integration
8. ✅ Test everything works

### First use

> Ensure your microphone of choice is available in audio settings!

1. **Log out and back in** (for group permissions)
2. **Press `Super+Alt+D`** to start dictation - _beep!_
3. **Speak naturally**
4. **Press `Super+Alt+D`** again to stop dictation - _boop!_
5. **Bam!** Text appears in active buffer!

## Usage

### Toggle-able global hotkey

- **`Super+Alt+D`** - Toggle dictation on/off

## Configuration

Edit `~/.config/hyprwhspr/config.json`:

**Minimal config** - only 2 essential options:

```jsonc
{
    "primary_shortcut": "SUPER+ALT+D",
    "model": "base.en"
}
```

For choice of model, see [model instructions](#whisper-models).

**Performance options** - improve cpu transcription speed:

```jsonc
{
    "threads": 4            // thread count for whisper cpu processing
}
```

Increase for more CPU parallelism when using CPU; on GPU, modest values are fine.

**Word overrides** - customize transcriptions:

```json
{
    "word_overrides": {
        "hyperwhisper": "hyprwhspr",
        "omarchie": "Omarchy"
    }
}
```

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

### Speech-to-text replacements

Automatically converts spoken words to symbols and punctuation for natural dictation:

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

_Speech-to-text replacement list via [WhisperTux](https://github.com/cjams/whispertux), thanks g@cjams!_

**Clipboard behavior** - control what happens to clipboard after text injection:

```jsonc
{
    "clipboard_behavior": false,       // Boolean: true = clear after delay, false = keep (default: false)
    "clipboard_clear_delay": 5.0      // Float: seconds to wait before clearing (default: 5.0, only used if clipboard_behavior is true)
}
```

**Clipboard behavior options:**
- **`clipboard_behavior: true`** - Clipboard is automatically cleared after the specified delay
- **`clipboard_clear_delay`** - How long to wait before clearing (only matters when `clipboard_behavior` is `true`)

> PRIVACY: hyprwhspr never reads your existing - or any - clipboard / audio content 

**Paste behavior** - control how text is pasted into applications:

```jsonc
{
    "paste_mode": "super"   // "super" | "ctrl_shift" | "ctrl"  (default: "super")
}
```

**Paste behavior options:**

- **`"super"`** (default) — Sends Super+V. Omarchy default.

- **`"ctrl_shift"`** — Sends Ctrl+Shift+V. Works in most terminals.

- **`"ctrl"`** — Sends Ctrl+V. Standard GUI paste.

**Backwards compatibility:**

Older configs using:
```jsonc
{
    "shift_paste": true   // Ctrl+Shift+V
}
```
```jsonc
{
    "shift_paste": false  // Ctrl+V
}
```
still work. If `paste_mode` is present, it takes precedence over `shift_paste`.

**Language detection** - control transcription language:

```jsonc
{
    "language": null    // null = auto-detect (default), or specify language code
}
```

Language options:
- **`null`** (default) - Auto-detect language from audio
- **`"en"`** - Force English transcription
- **`"nl"`** - Force Dutch transcription  
- **`"fr"`** - Force French transcription
- **`"de"`** - Force German transcription
- **`"es"`** - Force Spanish transcription
- **`etc.`** - Any supported language code

> **Note:** Multilingual models (like `base`, `medium`) are preferred over English-only models (like `base.en`) for language auto-detection.

### Waybar integration

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

Add CSS styling to your `~/.config/waybar/style.css`:

```css
@import "/usr/lib/hyprwhspr/config/waybar/hyprwhspr-style.css";
```

Click interactions:

- **Left-click**: Toggle Hyprwhspr on/off
- **Right-click**: Start Hyprwhspr (if not running)
- **Middle-click**: Restart Hyprwhspr

## Advanced Setup

### GPU Acceleration (NVIDIA & AMD)

- NVIDIA (CUDA) and AMD (ROCm) are detected automatically; pywhispercpp will use GPU when available
- No manual build steps required. 
    - If toolchains are present, installer can build pywhispercpp with GPU support; otherwise CPU wheel is used.

### Whisper Models (pywhispercpp)

**Default model installed:** `ggml-base.en.bin` (~148MB) to `~/.local/share/pywhispercpp/models/`

**Available models to download:**

- **`tiny.en`** - Fastest, good for real-time dictation
- **`base.en`** - Best balance of speed/accuracy (recommended)
- **`small.en`** - Better accuracy, still fast
- **`medium.en`** - High accuracy, slower processing
- **`large`** - Best accuracy, **requires GPU acceleration** for reasonable speed
- **`large-v3`** - Latest large model, **requires GPU acceleration** for reasonable speed

**⚠️ GPU Acceleration Required:** Models `large` and `large-v3` require GPU acceleration for reasonable performance. 

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

**Update your config after downloading:**

```json
{
    "model": "small.en" // Or just small if multi-lingual model. If both available, general model is chosen.
}
```

## Architecture

**hyprwhspr is designed as a system package:**

- **`/usr/lib/hyprwhspr/`** - Main installation directory
- **`/usr/lib/hyprwhspr/lib/`** - Python application
- **`~/.local/share/pywhispercpp/models/`** - Whisper models (user space)
- **`~/.config/hyprwhspr/`** - User configuration
- **`~/.config/systemd/user/`** - Systemd service

### Systemd integration

**hyprwhspr uses systemd for reliable service management:**

- **`hyprwhspr.service`** - Main application service with auto-restart
- **`ydotool.service`** - Input injection daemon service
- **Tray integration** - All tray operations use systemd commands
- **Process management** - No manual process killing or starting
- **Service dependencies** - Proper startup/shutdown ordering

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
# Then reinstall fresh
./scripts/install-omarchy.sh
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
/usr/lib/hyprwhspr/scripts/fix-uinput-permissions.sh

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

### Getting help

1. **Check logs**: `journalctl --user -u hyprwhspr.service` `journalctl --user -u ydotool.service`
2. **Verify permissions**: Run the permissions fix script
3. **Test components**: Check ydotool, audio devices, whisper.cpp
4. **Report issues**: Include logs and system information

## License

MIT License - see [LICENSE](LICENSE) file.

## Contributing

Create an issue, happy to help!  

For pull requests, also best to start with an issue.

---

**Built with ❤️ for the Omarchy community**

*Integrated and natural speech-to-text.*
