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

> 🔐 **PRIVATE**: hyprwhspr is local and never reads any clipboard / audio content 

## Quick start

### Prerequisites

- **[Omarchy](https://omarchy.org/)** or **[Arch Linux](https://archlinux.org/)**
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

Any snags, please [create an issue](https://github.com/goodroot/hyprwhspr/issues/new/choose) or visit [Omarchy Discord](https://discord.com/channels/1390012484194275541/1410373168765468774).

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

> For choice of model and languages, see [model instructions](https://github.com/goodroot/hyprwhspr/tree/main?tab=readme-ov-file#whisper-models).

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
    "paste_mode": "super"   // "super" | "ctrl_shift" | "ctrl"  (default: "super")
}
```

**Paste behavior options:**

- **`"super"`** (default) — Sends Super+V. Omarchy default.

- **`"ctrl_shift"`** — Sends Ctrl+Shift+V. Works in most terminals.

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

**CPU performance options** - improve cpu transcription speed:

```jsonc
{
    "threads": 4            // thread count for whisper cpu processing
}
```

Increase for more CPU parallelism when using CPU; on GPU, modest values are fine.

## Backend Options

hyprwhspr supports two transcription backends: **local** processing using pywhispercpp, or **remote** processing via speaches.ai or OpenAI API. Choose the option that best fits your needs.

### Local Backend (pywhispercpp) 

**Default model installed:** `ggml-base.en.bin` (~148MB) to `~/.local/share/pywhispercpp/models/`

**GPU Acceleration (NVIDIA & AMD):**

- NVIDIA (CUDA) and AMD (ROCm) are detected automatically; pywhispercpp will use GPU when available
- No manual build steps required. 
    - If toolchains are present, installer can build pywhispercpp with GPU support; otherwise CPU wheel is used.

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

**Update your config after downloading:**

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

### Remote Backend (speaches.ai / OpenAI)

hyprwhspr can use a remote transcription service instead of local processing.

#### Use Cases
- 🚀 Offload processing to a more powerful server
- 🔧 Use custom Whisper models not available locally
- 🌐 Centralized transcription for multiple devices
- ☁️ Use OpenAI's API without local GPU requirements

#### Quick Setup with speaches.ai

1. **Deploy speaches server:**
   ```bash
   # With volume mount to persist models
   docker run -d -p 8000:8000 -v speaches-models:/root/.cache/speaches ghcr.io/speaches-ai/speaches:latest
   ```

   > **Note:** The volume mount `-v speaches-models:/root/.cache/speaches` persists downloaded models across container restarts.

2. **Download a Whisper model:**
   ```bash
   # Download the base model (recommended to start)
   uvx speaches-cli model download Systran/faster-whisper-base

   # Or download other models:
   # uvx speaches-cli model download Systran/faster-distil-whisper-small.en
   # uvx speaches-cli model download Systran/faster-whisper-small
   # uvx speaches-cli model download Systran/faster-whisper-medium
   # uvx speaches-cli model download Systran/faster-whisper-large-v3
   ```

3. **Configure hyprwhspr for remote backend:**

   Edit `~/.config/hyprwhspr/config.json`:
   ```json
   {
     "backend": "remote",
     "remote_backend": {
       "api_url": "http://localhost:8000",
       "model": "Systran/faster-whisper-base"
     }
   }
   ```

4. **Restart hyprwhspr:**
   ```bash
   systemctl --user restart hyprwhspr.service
   ```

#### Configuration Options

**Backend Selection:**
```json
{
  "backend": "local"   // Use local pywhispercpp (default)
}
```
```json
{
  "backend": "remote"  // Use remote API
}
```

**Remote Backend Settings:**

| Setting | Required | Default | Description |
|---------|----------|---------|-------------|
| `api_url` | ✅ Yes | - | Base URL of speaches/OpenAI server |
| `model` | ✅ Yes | - | Model identifier |
| `api_key` | No | `"dummy"` | API key (use "dummy" for speaches) |
| `prompt` | No | `null` | Optional prompt for better accuracy |
| `language` | No | `null` | Language code (e.g., "en", "es") |
| `timeout` | No | `30` | Request timeout in seconds |
| `max_retries` | No | `2` | Automatic retry attempts |

**Available Models:**

*speaches.ai:*
- `Systran/faster-whisper-base` - Good balance (recommended)
- `Systran/faster-whisper-small` - Better accuracy
- `Systran/faster-whisper-medium` - High accuracy
- `Systran/faster-whisper-large-v3` - Best accuracy

*OpenAI:*
- `whisper-1` - Standard Whisper model
- `gpt-4o-transcribe` - High quality
- `gpt-4o-mini-transcribe` - Fast and efficient

#### Example Configurations

**Minimal remote config:**
```json
{
  "backend": "remote",
  "remote_backend": {
    "api_url": "http://192.168.1.100:8000",
    "model": "Systran/faster-whisper-base"
  }
}
```

**Full remote config with all options:**
```json
{
  "backend": "remote",
  "remote_backend": {
    "api_url": "http://your-server:8000",
    "api_key": "dummy",
    "model": "Systran/faster-whisper-small",
    "prompt": "Technical transcription with proper terminology",
    "language": "en",
    "timeout": 30,
    "max_retries": 2
  }
}
```

**Using OpenAI's official API:**
```json
{
  "backend": "remote",
  "remote_backend": {
    "api_url": "https://api.openai.com/v1",
    "api_key": "sk-your-actual-openai-api-key",
    "model": "whisper-1",
    "timeout": 60
  }
}
```

#### Privacy & Security

⚠️ **Important:** When using remote backend, your audio is sent over the network to the remote server.

**Privacy considerations:**
- Ensure you trust the server hosting the API
- Use HTTPS (not HTTP) for sensitive content
- Consider self-hosting speaches for full privacy control
- Review the server's data retention policies

**Self-hosting speaches ensures:**
- ✅ Audio never leaves your infrastructure
- ✅ Full control over data
- ✅ No third-party dependencies

#### Troubleshooting Remote Backend

**Test server connectivity:**
```bash
curl http://your-server:8000/v1/audio/transcriptions \
  -F "file=@test.wav" \
  -F "model=Systran/faster-whisper-base"
```

**Check hyprwhspr logs:**
```bash
journalctl --user -u hyprwhspr.service -f
```

**Common errors:**

*"Connection refused"* - Server not running or wrong URL
```bash
# Verify server is running
docker ps | grep speaches
```

*"remote_backend configuration is required"* - Add config
```json
{
  "backend": "remote",
  "remote_backend": { ... }
}
```

*"openai package not installed"* - Install dependency
```bash
~/.local/share/hyprwhspr/venv/bin/pip install openai
```

**Switch back to local mode:**
```json
{
  "backend": "local"
}
```
Then restart:
```bash
systemctl --user restart hyprwhspr.service
```

#### Advanced: Custom Prompts

Use prompts to improve transcription quality for specific contexts:

**Technical content:**
```json
{
  "remote_backend": {
    "prompt": "Technical documentation with proper capitalization of technology terms like API, HTTP, SQL, Python, Docker."
  }
}
```

**Medical transcription:**
```json
{
  "remote_backend": {
    "prompt": "Medical transcription with proper medical terminology and abbreviations."
  }
}
```

**Casual conversation:**
```json
{
  "remote_backend": {
    "prompt": "Transcribe as natural conversation with informal language."
  }
}
```

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
