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
- **NVIDIA GPU support** - Automatic CUDA detection and acceleration
- **Word overrides** - Customize transcriptions, prompt and corrections
- **Run as user** - Runs in user space, just sudo once for the installer

## Quick start

### Prerequisites

- **[Omarchy](https://omarchy.org/)**
- **NVIDIA GPU** (optional, for acceleration)

### Installation

"Just works" with Omarchy.

Any other setups may run into bumps.

If stuck, create an issue or visit the thread in [Omarchy discord](https://discord.com/channels/1390012484194275541/1410373168765468774).

```bash
# Clone the repository
git clone https://github.com/goodroot/hyprwhspr.git
cd hyprwhspr

# Run the automated installer
./scripts/install-omarchy.sh
```

* AUR is wip - please use script!

**The installer will:**

1. ✅ Install system dependencies (ydotool, etc.)
2. ✅ Clone and build whisper.cpp (with CUDA if GPU available)
3. ✅ Download base Whisper models
4. ✅ Set up systemd services for hyprwhspr & ydotoolds
5. ✅ Configure Waybar integration
6. ✅ Test everything works

### First use

> Ensure your microphone of choice is available in audio settings.

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

**Minimal config** - only 3 essential options:

```jsonc
{
    "primary_shortcut": "SUPER+ALT+D",
    "model": "base.en",
    "audio_feedback": true, // Optional
}
```

**Model options**:

- **Default**: `"base.en"` (automatically resolves to `/opt/hyprwhspr/whisper.cpp/models/ggml-base.en.bin`)
- **Tiny (fastest)**: `"tiny.en"`
- **Small (better)**: `"small.en"`
- **Medium (high accuracy)**: `"medium.en"`
- **Large (best accuracy)**: `"large"` ⚠️ **GPU required**
- **Large-v3 (latest)**: `"large-v3"` ⚠️ **GPU required**

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
  
- `"Transcribe as formal business communication with professional language."`

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

_Thanks for the speech-to-text replacement ideas from [WhisperTux](https://github.com/cjams/whispertux), @cjams!_

**Custom sounds:**
- **Supported formats**: `.ogg`, `.wav`, `.mp3`
- **Fallback**: Uses defaults if custom files don't exist

**Clipboard behavior** - control what happens to clipboard after text injection:

```jsonc
{
    "clipboard_behavior": false,       // Boolean: true = clear after delay, false = keep (default: false)
    "clipboard_clear_delay": 5.0      // Float: seconds to wait before clearing (default: 5.0, only used if clipboard_behavior is true)
}
```

**Clipboard behavior options:**

- **`clipboard_behavior: false`** (default) - Transcribed text stays on clipboard indefinitely
- **`clipboard_behavior: true`** - Clipboard is automatically cleared after the specified delay
- **`clipboard_clear_delay`** - How long to wait before clearing (only matters when `clipboard_behavior` is `true`)

**Privacy note:** hyprwhspr never reads your existing clipboard content - it only clears the transcribed text after injection.

**Paste behavior** - control how text is pasted into applications:

```jsonc
{
    "shift_paste": true    // Boolean: true = Ctrl+Shift+V (default), false = Ctrl+V
}
```

**Paste behavior options:**

- **`shift_paste: true`** (default) - Uses Ctrl+Shift+V for pasting
  - ✅ Works in terminals (bash, zsh, fish, etc.)
  - ✅ Probably works everywhere else

- **`shift_paste: false`** - Uses traditional Ctrl+V for pasting
  - ✅ Standard paste behavior
  - ❌ May not work in some terminal applications

**Why Ctrl+Shift+V by default?** Most Linux applications support Ctrl+Shift+V as an alternative paste command, making it work universally across both terminals and GUI applications.

### Waybar integration

Add to your `~/.config/waybar/config`:

```json
{
    "custom/hyprwhspr": {
        "exec": "/opt/hyprwhspr/config/hyprland/hyprwhspr-tray.sh status",
        "interval": 2,
        "return-type": "json",
        "exec-on-event": true,
        "format": "{}",
        "on-click": "/opt/hyprwhspr/config/hyprland/hyprwhspr-tray.sh toggle",
        "on-click-right": "/opt/hyprwhspr/config/hyprland/hyprwhspr-tray.sh start",
        "on-click-middle": "/opt/hyprwhspr/config/hyprland/hyprwhspr-tray.sh restart",
        "tooltip": true
    }
}
```

**Add CSS styling** to your `~/.config/waybar/style.css`:

```css
@import "/opt/hyprwhspr/config/waybar/hyprwhspr-style.css";
```

**Click interactions:**

- **Left-click**: Toggle Hyprwhspr on/off
- **Right-click**: Start Hyprwhspr (if not running)
- **Middle-click**: Restart Hyprwhspr


## Advanced Setup

### NVIDIA GPU Acceleration

Use if adding GPU after GPU-less installation:

```bash
# Build with CUDA support (if NVIDIA detected)
/opt/hyprwhspr/scripts/build-whisper-nvidia.sh

# Test GPU acceleration
/opt/hyprwhspr/scripts/build-whisper-nvidia.sh --test
```

Or re-run install script.

### Whisper Models

**Default model included:** `ggml-base.en.bin` (CPU-optimized, ~1GB)

**Available models to download:**

```bash
cd /opt/hyprwhspr/whisper.cpp

# Tiny models (fastest, least accurate)
sh ./models/download-ggml-model.sh tiny.en      # ~39MB
sh ./models/download-ggml-model.sh tiny         # ~39MB

# Base models (good balance)
sh ./models/download-ggml-model.sh base.en      # ~1GB (default)
sh ./models/download-ggml-model.sh base         # ~1GB

# Small models (better accuracy)
sh ./models/download-ggml-model.sh small.en     # ~244MB
sh ./models/download-ggml-model.sh small        # ~244MB

# Medium models (high accuracy)
sh ./models/download-ggml-model.sh medium.en    # ⚠️ ~769MB
sh ./models/download-ggml-model.sh medium       # ⚠️ ~769MB

# Large models (best accuracy, requires GPU)
sh ./models/download-ggml-model.sh large        # ⚠️ ~1.5GB
sh ./models/download-ggml-model.sh large-v3     # ⚠️ ~1.5GB (latest)
```

**⚠️ GPU Acceleration Required:**

Models `large` and `large-v3` require NVIDIA GPU acceleration for reasonable performance. 

Without a GPU, these models will be extremely slow (10-30 seconds per transcription).

**Model selection guide:**
- **`tiny.en`** - Fastest, good for real-time dictation
- **`base.en`** - Best balance of speed/accuracy (recommended)
- **`small.en`** - Better accuracy, still fast
- **`medium.en`** - High accuracy, slower processing
- **`large`** - Best accuracy, **requires GPU acceleration** for reasonable speed
- **`large-v3`** - Latest large model, **requires GPU acceleration** for reasonable speed

**Update your config after downloading:**

```json
{
    "model": "small.en"
}
```

## Architecture

**hyprwhspr is designed as a system package:**

- **`/opt/hyprwhspr/`** - Main installation directory
- **`/opt/hyprwhspr/lib/`** - Python application
- **`/opt/hyprwhspr/whisper.cpp/`** - Speech recognition engine
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
# Run the reset script (removes everything)
./scripts/reset-hyprwhspr.sh

# Then reinstall fresh
./scripts/install-omarchy.sh
```

The reset script will:

- Stop and disable all services
- Remove installation directory (`/opt/hyprwhspr`)
- Remove user config and data
- Clean up waybar integration
- Remove hyprland scripts
- Clean systemd configuration

### Common issues

**I heard the sound, but don't see text!** 

It's fairly common in Arch and other distros for the microphone to need to be plugged in and set each time you log in and out of your session, including during a restart. Within sound options, ensure that the microphone is indeed set. The sound utility will show feedback from the microphone if it is.

**Hotkey not working:**

```bash
# Check service status
systemctl --user status hyprwhspr.service

# Check logs
journalctl --user -u hyprwhspr.service -f
```

**Permission denied:**

```bash
# Fix uinput permissions
/opt/hyprwhspr/scripts/fix-uinput-permissions.sh

# Log out and back in
```

**No audio input:**

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
ls -la /opt/hyprwhspr/assets/

# Check if ffplay/aplay/paplay is available
which ffplay aplay paplay
```

**Model not found:**

```bash
# Check if model exists
ls -la /opt/hyprwhspr/whisper.cpp/models/

# Download a different model
cd /opt/hyprwhspr/whisper.cpp
sh ./models/download-ggml-model.sh tiny.en

# Verify model path in config
cat ~/.config/hyprwhspr/config.json | grep model
```

**Stuck recording state:**

```bash
# Check service health and auto-recover
/opt/hyprwhspr/config/hyprland/hyprwhspr-tray.sh health

# Manual restart if needed
systemctl --user restart hyprwhspr.service

# Check service status
systemctl --user status hyprwhspr.service
```

### Getting help

1. **Check logs**: `journalctl --user -u hyprwhspr.service`
2. **Verify permissions**: Run the permissions fix script
3. **Test components**: Check ydotool, audio devices, whisper.cpp
4. **Report issues**: Include logs and system information

## Other DEs

Gnome, KDE, etc might work. Untested. Try!

## License

MIT License - see [LICENSE](LICENSE) file.

## Contributing

Create an issue, happy to help!  

For pull requests, also best to start with an issue.

---

**Built with ❤️ for the Omarchy community**

*Integrated and natural speech-to-text.*
