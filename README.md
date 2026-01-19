<h1 align="center">
    hyprwhspr
</h1>

<p align="center">
    <b>Native speech-to-text for Linux</b> - Fast, accurate and private system-wide dictation
</p>

<p align="center">
    instant performance | Parakeet / Whisper / REST API | stylish visuals
</p>

 <p align="center">
    <i>Supports Arch, Debian, Ubuntu, Fedora, openSUSE and more</i>
 </p>

https://github.com/user-attachments/assets/4c223e85-2916-494f-b7b1-766ce1bdc991

---

- **Built for Linux** - Native AUR package for Arch, or use Debian/Ubuntu/Fedora/openSUSE
- **Local, very fast defaults** - Instant, private and accurate performance via in-memory models
- **Latest models** - Turbo-v3? Parakeet TDT V3? Latest and greatest
- **onnx-asr for wild CPU speeds** - No GPU? Optimized for great speed on any hardware
- **REST API** - Securely connect to cloud models or your own localhost models
- **Themed visualizer** - Visualizes your voice, will automatch Omarchy theme
- **Word overides and prompts** - Custom hot keys, common words, and more
- **Multi-lingual** - Great performance in many languages
- **Long form mode with saving** - Pause, think, resume, pause: submit... Bam!
- **Auto-paste anywhere** - Instant paste into any active buffer, or even auto enter (optional)
- **Audio ducking 🦆** - Reduces system volume on record (optional)

## Quick start

### Prerequisites

- **Linux** with systemd (Arch, Debian, Ubuntu, Fedora, openSUSE, etc.)
- **Requires a Wayland session** (GNOME, KDE Plasma Wayland, Sway, Hyprland)

- **Waybar** (optional, for status bar)
- **gtk4** (optional, for visualizer)
- **NVIDIA GPU** (optional, for CUDA acceleration)
- **AMD/Intel GPU / APU** (optional, for Vulkan acceleration)

### Quick start (Arch Linux)

On the AUR:

```bash
# Install for stable
yay -S hyprwhspr

# Or install for bleeding edge
yay -S hyprwhspr-git
```

Then run the auto installer, or perform your own:

```bash
# Run interactive setup
hyprwhspr setup
```

**The setup will walk you through the process:**

1. ✅ Configure transcription backend (Parakeet TDT V3, pywhispercpp, REST API, or Realtime WebSocket)
2. ✅ Download models (if using pywhispercpp backend)
3. ✅ Configure themed visualizer for maximum coolness (optional)
4. ✅ Configure Waybar integration (optional)
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

Any snags, please [create an issue](https://github.com/goodroot/hyprwhspr/issues/new/choose).

### Updating

```bash
# Update via your AUR helper
yay -Syu hyprwhspr

# If needed, re-run setup (idempotent)
hyprwhspr setup
```

### Other Linux distros

hyprwhspr can run on any Linux distribution with systemd.

**Quick install (recommended):**

Use the install script to automatically install dependencies for your distro:

```bash
# Download and run the install script
curl -fsSL https://raw.githubusercontent.com/goodroot/hyprwhspr/main/scripts/install-deps.sh | bash

# Clone and run setup
git clone https://github.com/goodroot/hyprwhspr.git ~/hyprwhspr
cd ~/hyprwhspr
./bin/hyprwhspr setup
```

The script supports Ubuntu, Debian, Fedora, and openSUSE.

> Non-Arch distro support is new - please report any snags!

<details>
<summary><b>Manual installation instructions</b></summary>

**Debian / Ubuntu:**

> **Important:** Ubuntu/Debian apt repositories contain an outdated ydotool (0.1.x from 2019) that is **incompatible** with hyprwhspr. With the old version, paste injection outputs garbage instead of text. You must install ydotool 1.0+ manually. The install script above handles this automatically, or see the manual fix below.

```bash
# Install system dependencies (NOTE: do NOT install ydotool from apt)
sudo apt install python3 python3-pip python3-venv git cmake make build-essential \
    python3-dev libportaudio2 python3-numpy python3-scipy python3-evdev \
    python3-requests python3-psutil python3-rich \
    python3-gi gir1.2-gtk-4.0 gir1.2-gtk4layershell-1.0 \
    pipewire pipewire-pulse wl-clipboard wget

# Install ydotool 1.0+ from Debian backports (required!)
wget http://deb.debian.org/debian/pool/main/y/ydotool/ydotool_1.0.4-2~bpo13+1_amd64.deb
sudo dpkg -i ydotool_1.0.4-2~bpo13+1_amd64.deb
sudo apt install -f  # Fix any dependency issues

# Install Python packages not in Debian repos
pip install --user --break-system-packages sounddevice pyperclip

# Clone and run setup
git clone https://github.com/goodroot/hyprwhspr.git ~/hyprwhspr
cd ~/hyprwhspr
./bin/hyprwhspr setup
```

> **Note:** On Ubuntu 22.04 LTS, `gir1.2-gtk4layershell-1.0` may not be available. The mic-osd visualizer will be disabled, but dictation works fine without it.

**Fedora:**

```bash
# Install system dependencies
sudo dnf install python3 python3-pip python3-devel git cmake make gcc-c++ \
    python3-sounddevice python3-numpy python3-scipy python3-evdev \
    python3-pyperclip python3-requests python3-psutil python3-rich \
    python3-gobject gtk4 gtk4-layer-shell \
    pipewire pipewire-pulseaudio ydotool wl-clipboard

# Clone and run setup
git clone https://github.com/goodroot/hyprwhspr.git ~/hyprwhspr
cd ~/hyprwhspr
./bin/hyprwhspr setup
```

**openSUSE:**

```bash
# Install system dependencies
sudo zypper install python3 python3-pip python3-devel git cmake make gcc-c++ \
    python3-sounddevice python3-numpy python3-scipy python3-evdev \
    python3-pyperclip python3-requests python3-psutil python3-rich \
    python3-gobject typelib-1_0-Gtk-4_0 \
    pipewire pipewire-pulseaudio ydotool wl-clipboard

# Optional: For mic-osd visualizer (Tumbleweed only, from community repo)
# sudo zypper addrepo https://download.opensuse.org/repositories/devel:languages:zig/openSUSE_Tumbleweed/devel:languages:zig.repo
# sudo zypper refresh && sudo zypper install gtk4-layer-shell

# Clone and run setup
git clone https://github.com/goodroot/hyprwhspr.git ~/hyprwhspr
cd ~/hyprwhspr
./bin/hyprwhspr setup
```

</details>

**Post-installation (non-Arch distros):**

The setup wizard handles most configuration automatically:

- Creates `~/.local/bin/hyprwhspr` symlink (so the command works from anywhere)
- Configures systemd services
- Sets up permissions (groups, udev rules)

After setup completes:

```bash
# Log out and back in for group permissions to take effect
# Then verify everything is running:
hyprwhspr status
```

> **Note:** On non-Arch systems, the setup will guide you through any missing dependencies. GPU acceleration (CUDA/Vulkan) requires additional packages - the setup will provide instructions.

### CLI Commands

After installation, use the `hyprwhspr` CLI to manage your installation:

- `hyprwhspr setup` - Interactive initial setup
  - `hyprwhspr setup auto` - Automated setup with optional parameters:
    - `--backend {nvidia,vulkan,cpu,onnx-asr}` - Specify backend (default: auto-detect GPU)
    - `--model MODEL` - Model to download (default: base for whisper, auto for onnx-asr)
    - `--no-waybar` - Skip waybar integration
    - `--no-mic-osd` - Disable mic-osd visualization
    - `--no-systemd` - Skip systemd service setup
    - `--hypr-bindings` - Enable Hyprland compositor bindings
- `hyprwhspr config` - Manage configuration (init/show/edit)
- `hyprwhspr waybar` - Manage Waybar integration (install/remove/status)
- `hyprwhspr mic-osd` - Manage microphone visualization overlay (enable/disable/status)
- `hyprwhspr systemd` - Manage systemd services (install/enable/disable/status/restart)
- `hyprwhspr model` - Manage models (download/list/status)
- `hyprwhspr status` - Overall status check
- `hyprwhspr validate` - Validate installation
- `hyprwhspr test` - Test microphone and backend connectivity end-to-end
  - `--live` - Record live audio (3s) instead of using test.wav
  - `--mic-only` - Only test microphone, skip transcription
- `hyprwhspr backend` - Backend management (repair/reset)
- `hyprwhspr state` - State management (show/validate/reset)
- `hyprwhspr uninstall` - Completely remove hyprwhspr and all data

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

**Long-form mode:**

- **`Super+Alt+D`** - Toggle recording/pause (start recording, pause, or resume)
- **`long_form_submit_shortcut`** - Set a key to send, like `SUPER+ALT+E`
- Auto-saves segments periodically (default: every 5 minutes) for crash-safe recording
- Supports pause/resume for extended recording sessions

## Configuration

Edit `~/.config/hyprwhspr/config.json`:

**Minimal config** - only 2 essential options:

```jsonc
{
    "primary_shortcut": "SUPER+ALT+D",
    "model": "base"
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

**Long-form mode** - extended recording with pause/resume support:

```jsonc
{
    "recording_mode": "long_form",
    "long_form_submit_shortcut": "SUPER+ALT+E",  // Required: no default, must be set
    "long_form_temp_limit_mb": 500,              // Optional: max temp storage (default: 500 MB)
    "long_form_auto_save_interval": 300,         // Optional: auto-save interval in seconds (default: 300 = 5 minutes)
    "use_hypr_bindings": false,                   // Optional: set true to use Hyprland compositor bindings
    "grab_keys": false                            // Recommended: false for normal keyboard usage
}
```

- Primary shortcut toggles recording/pause/resume
- Submit shortcut processes all recorded segments and pastes transcription
- Segments are auto-saved periodically to disk for crash recovery
- Old segments are automatically cleaned up when storage limit is reached

**REST API** - use any ASR backend via HTTP API (local or cloud):

**OpenAI**

Bring an API key from OpenAI, and choose from:

- **GPT-4o Transcribe** - Latest model with best accuracy
- **GPT-4o Mini Transcribe** - Faster, lighter model
- **GPT-4o Mini Transcribe (2025-12-15)** - Updated version of the faster, lighter transcription model
- **GPT Audio Mini (2025-12-15)** - General purpose audio model
- **Whisper 1** - Legacy Whisper model

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

**Realtime WebSocket** - low-latency streaming via OpenAI's Realtime API:

> Experimental! 

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

**Secondary shortcut with language** - use a different hotkey for a specific language:

```jsonc
{
    "primary_shortcut": "SUPER+ALT+D",    // Uses default language from config
    "secondary_shortcut": "SUPER+ALT+I",  // Optional: second hotkey
    "secondary_language": "it"          // Language for secondary shortcut
}
```

> **Note**: Works with backends that support language parameters:
> - **REST API**: Works if the endpoint accepts `language` in the request body
> - **Realtime WebSocket**: Fully supported (OpenAI Realtime API)
> - **Local whisper models**: Fully supported (all pywhispercpp models)
> - **Custom REST endpoints**: May not work if the endpoint doesn't accept a language parameter

The primary shortcut continues to use the `language` setting from your config (or auto-detect if set to `null`). The secondary shortcut will always use the configured `secondary_language` when pressed.

Configure via CLI:

```bash
hyprwhspr config secondary-shortcut
```

**Hyprland native input bindings:**

Use Hyprland's compositor bindings instead of evdev keyboard grabbing.

Somtimes better compatibility with keyboard remappers.

Enable in config (`~/.config/hyprwhspr/config.json`):

```json
{
  "use_hypr_bindings": true,
  "grab_keys": false
}
```

Add bindings to `~/.config/hypr/hyprland.conf`:

```bash
# Toggle mode
# Press once to start, press again to stop
bindd = SUPER ALT, D, Speech-to-text, exec, /usr/lib/hyprwhspr/config/hyprland/hyprwhspr-tray.sh record
```

```bash
# Push-to-Talk mode
# Hold key to record, release to stop
bind = SUPER ALT, D, exec, echo "start" > ~/.config/hyprwhspr/recording_control
bindr = SUPER ALT, D, exec, echo "stop" > ~/.config/hyprwhspr/recording_control
```

```bash
# Long-form mode
# Primary shortcut: toggle record/pause/resume
bindd = SUPER ALT, D, Speech-to-text, exec, /usr/lib/hyprwhspr/config/hyprland/hyprwhspr-tray.sh record
# Submit shortcut: submit recording for transcription
bindd = SUPER ALT, E, Speech-to-text-submit, exec, echo "submit" > ~/.config/hyprwhspr/recording_control
```

Restart service to lock in changes:

```bash
systemctl --user restart hyprwhspr
```

**Themed visualizer** - visual feedback, will auto-match Omarchy themes:

> Highly recommended!

```json
{
  "mic_osd_enabled": true,
}
```

**Word overrides** - customize transcriptions:

```json
{
    "word_overrides": {
        "hyper whisper": "hyprwhspr",
    }
}
```

**Audio feedback** - optional sound notifications:

```jsonc
{
    "audio_feedback": true,            // Enable audio feedback (default: false)
    "audio_volume": 0.5,               // General audio volume fallback (0.1 to 1.0, default: 0.5)
    "start_sound_volume": 1.0,         // Start recording sound volume (0.1 to 1.0, default: 1.0)
    "stop_sound_volume": 1.0,          // Stop recording sound volume (0.1 to 1.0, default: 1.0)
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

**Text replacement:** 

Automatically converts spoken words to symbols / punctuation:

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
    "paste_mode": "ctrl_shift"   // "ctrl_shift" | "ctrl" | "super" (default: "ctrl_shift")
}
```

**Paste behavior options:**

- **`"ctrl_shift"`** (default) — Sends Ctrl+Shift+V. Works in most terminals.

- **`"ctrl"`** — Sends Ctrl+V. Standard GUI paste.

- **`"super"`** — Sends Super+V. Maybe finicky.

**Auto-submit** - automatically press Enter after pasting:

> aka Dictation YOLO

```jsonc
{
    "auto_submit": true   // Send Enter key after paste (default: false)
}
```

Useful for chat applications, search boxes, or any input where you want to submit immediately after dictation.

... Be careful!

**Audio ducking** - quiet system volume on record:

```json
{
  "audio_ducking": true,
  "audio_ducking_percent": 70
}
```


- `audio_ducking: true` Set true to enable audio ducking 
- `audio_ducking_percent: 70` -  How much to reduce volume BY (70 = reduces to 30% of original)

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

- **Left-click**: Start/stop recording (auto-starts service if needed)
- **Right-click**: Restart Hyprwhspr service

## Parakeet (Nvidia)

Parakeet V3 via [onnx-asr](https://github.com/istupakov/onnx-asr) is a fantastic project.

It provides very strong accuracy and nigh unbelievable speed on modest CPUs.

Also great for GPUs.

Select Parakeet V3 within `hyprwhspr setup`.

## Whisper (OpenAI)

**Default multi-lingual model installed:** `ggml-base.bin` (~175MB) to `~/.local/share/pywhispercpp/models/`

**GPU Acceleration (NVIDIA & AMD):**

- NVIDIA (CUDA) and AMD/Intel (Vulkan) are detected automatically; pywhispercpp will use GPU when selected

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


## Troubleshooting

### Reset Installation

If you're having persistent issues, completely reset hyprwhspr:

```bash
hyprwhspr uninstall
hyprwhspr setup
```

### Common issues

**Something is weird...**

Right click the Waybar microphone next to the tray to restart the service.

Still weird? Proceed.

**I heard the sound, but don't see text!** 

It's common in Arch and other distros for the microphone to need to be plugged in and set each time you log in and out of your session, including during a restart. Reseat your microphone as prompted if it fails under these conditions. Also, ithin sound options, ensure that the microphone is indeed set. The sound utility will show feedback from the select microphone if it is.

**I updated and something is weird...**

Uninstall everything and setup fresh.

Brute force. And effective.

```bash
hyprwhspr uninstall
hyprwhsp setup
```

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

**Bluetooth mic and flakey recording:**

Mute detection can cause conflicts with Bluetooth microphones. To disable it, add the following to your `~/.config/hyprwhspr/config.json`:

```json
{
  "mute_detection": false
}
```

**This sucks!**

Doh! We tried.

Wipe the slate clean and remove everything:

```
hyprwhspr uninstall
yay -Rs hyprwhspr
```

Or better yet - create an issue and help us improve.


## Getting help

1. **Check logs**: `journalctl --user -u hyprwhspr.service` `journalctl --user -u ydotool.service`
2. **Verify permissions**: Run the permissions fix script
3. **Test components**: Check ydotool, audio devices, whisper.cpp
4. **Report issues**: [Create an issue](https://github.com/goodroot/hyprwhspr/issues/new/choose) - logging info helpful!

## License

MIT License - see [LICENSE](LICENSE) file.

## Contributing

Create an issue, happy to help!  

For pull requests, also best to start with an issue.

---

**Built with ❤️ in 🇨🇦**
