# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**hyprwhspr** is a native speech-to-text dictation system for Arch Linux/Omarchy. It provides fast, accurate, system-wide voice dictation using OpenAI's Whisper (via whisper.cpp) with optional NVIDIA GPU acceleration.

### Key Features
- Toggle-based dictation with global hotkeys (default: Super+Alt+D)
- Runs as user-space systemd services (no root required after setup)
- Auto-paste into active applications via clipboard + hotkey
- Waybar integration for visual status
- Audio feedback (beep on record start/stop)
- Word overrides and custom Whisper prompts

## Installation & Setup

### Installation Commands
```bash
# Install (automated - for Omarchy)
./scripts/install-omarchy.sh

# Reset and reinstall
./scripts/reset-hyprwhspr.sh
./scripts/install-omarchy.sh

# Build Whisper with NVIDIA GPU support
/opt/hyprwhspr/scripts/build-whisper-nvidia.sh

# Test GPU acceleration
/opt/hyprwhspr/scripts/build-whisper-nvidia.sh --test

# Fix permissions issues
/opt/hyprwhspr/scripts/fix-uinput-permissions.sh
```

### Service Management
```bash
# Check service status
systemctl --user status hyprwhspr.service
systemctl --user status ydotool.service

# View logs
journalctl --user -u hyprwhspr.service -f
journalctl --user -u ydotool.service -f

# Restart services
systemctl --user restart hyprwhspr.service
systemctl --user restart ydotool.service
```

### Download Whisper Models
```bash
cd /opt/hyprwhspr/whisper.cpp
sh ./models/download-ggml-model.sh tiny.en      # Fastest
sh ./models/download-ggml-model.sh base.en      # Default, balanced
sh ./models/download-ggml-model.sh small.en     # Better accuracy
sh ./models/download-ggml-model.sh medium.en    # High accuracy
sh ./models/download-ggml-model.sh large-v3     # Best (requires GPU)
```

## Architecture

### Installation Layout
hyprwhspr is designed as a **system package installed to /opt**:

- `/opt/hyprwhspr/` - Main installation directory
- `/opt/hyprwhspr/lib/` - Python application code
- `/opt/hyprwhspr/whisper.cpp/` - Whisper.cpp clone and models
- `/opt/hyprwhspr/config/` - Systemd, Waybar, and Hyprland integration files
- `/opt/hyprwhspr/scripts/` - Installation and utility scripts
- `/opt/hyprwhspr/share/assets/` - Audio feedback files (ping-up.ogg, ping-down.ogg)
- `~/.config/hyprwhspr/` - User configuration directory
- `~/.config/systemd/user/` - User systemd services

### Core Components

#### 1. Main Application (`lib/main.py`)
- Entry point: `hyprwhsprApp` class
- Orchestrates all components
- Runs in headless mode (no GUI)
- Maintains recording state via file: `~/.config/hyprwhspr/recording_status`

#### 2. Audio Capture (`lib/src/audio_capture.py`)
- Uses sounddevice library for audio input
- Captures at 16kHz mono (optimized for Whisper)
- Supports device selection and auto-discovery
- Provides real-time audio level monitoring
- Threading-based recording with proper cleanup

#### 3. Whisper Manager (`lib/src/whisper_manager.py`)
- Interfaces with whisper.cpp binary
- Manages model loading and transcription
- Supports configurable Whisper prompts
- Handles temporary WAV file creation/cleanup
- Model path: `/opt/hyprwhspr/whisper.cpp/models/ggml-{model}.bin`
- Binary path: `/opt/hyprwhspr/whisper.cpp/build/bin/whisper-cli`

#### 4. Text Injection (`lib/src/text_injector.py`)
- **Strategy**: Clipboard + paste hotkey (Ctrl+Shift+V or Ctrl+V)
- Uses ydotool for keystroke injection
- Preprocesses text with speech-to-text replacements:
  - "period" → "."
  - "comma" → ","
  - "new line" → newline character
  - See `_preprocess_text()` for full list
- Applies user-defined word overrides
- Optional clipboard clearing after delay

#### 5. Global Shortcuts (`lib/src/global_shortcuts.py`)
- Uses evdev for hardware-level keyboard capture
- Parses key combinations (e.g., "SUPER+ALT+D")
- Requires access to /dev/input devices (uinput group membership)
- Supports debouncing to prevent double-triggers
- Thread-based event loop with select()

#### 6. Configuration Manager (`lib/src/config_manager.py`)
- Loads/saves `~/.config/hyprwhspr/config.json`
- Default settings:
  - `primary_shortcut`: "SUPER+ALT+D"
  - `model`: "base"
  - `audio_feedback`: true/false
  - `shift_paste`: true (Ctrl+Shift+V) / false (Ctrl+V)
  - `clipboard_behavior`: false (keep) / true (clear after delay)
  - `word_overrides`: {} (custom replacements)
  - `whisper_prompt`: (transcription guidance)

#### 7. Audio Manager (`lib/src/audio_manager.py`)
- Plays start/stop audio feedback
- Uses ffplay, aplay, or paplay
- Default sounds: ping-up.ogg, ping-down.ogg

### Systemd Integration

**Services:**
- `hyprwhspr.service` - Main Python application
- `ydotool.service` - Input injection daemon

**Key design:**
- All operations use systemd commands (start/stop/restart)
- Auto-restart on failure
- Waits for Hyprland Wayland session
- No manual process management

### Waybar Integration

**Tray Script:** `/opt/hyprwhspr/config/hyprland/hyprwhspr-tray.sh`

Commands:
- `status` - Returns JSON for Waybar display
- `toggle` - Toggle hyprwhspr on/off
- `start` - Start service
- `restart` - Restart service
- `health` - Check and auto-recover stuck states

**Configuration:**
- Waybar config: `~/.config/waybar/config` (custom/hyprwhspr section)
- Waybar styles: `/opt/hyprwhspr/config/waybar/hyprwhspr-style.css`

## Development Notes

### Python Dependencies
From `requirements.txt`:
- sounddevice (audio capture)
- numpy, scipy (audio processing)
- evdev (keyboard shortcuts)
- pyperclip (clipboard operations)
- psutil (process management)
- json5 (config parsing)
- rich (CLI formatting)

### Key Design Decisions

1. **Installation to /opt**: Allows clean separation between system package and user config
2. **Systemd-first**: All lifecycle management through systemd for reliability
3. **Clipboard paste strategy**: More reliable than direct typing, works in terminals
4. **Evdev for shortcuts**: Hardware-level capture works system-wide
5. **16kHz mono audio**: Optimized for Whisper.cpp performance
6. **Recording status file**: Simple IPC between Python app and tray script

### File Paths to Remember
- Config: `~/.config/hyprwhspr/config.json`
- Recording status: `~/.config/hyprwhspr/recording_status` (created during recording)
- Temp audio: `lib/temp/` (created automatically)
- Whisper binary: `/opt/hyprwhspr/whisper.cpp/build/bin/whisper-cli`
- Models: `/opt/hyprwhspr/whisper.cpp/models/ggml-*.bin`

### Troubleshooting Commands

```bash
# Check if ydotool socket exists
ls -la $XDG_RUNTIME_DIR/.ydotool_socket

# Test audio devices
pactl list short sources

# Check uinput permissions
ls -la /dev/uinput
groups | grep uinput

# View real-time logs
journalctl --user -u hyprwhspr.service -f

# Health check and auto-recover
/opt/hyprwhspr/config/hyprland/hyprwhspr-tray.sh health
```

## Common Patterns

### Adding a New Speech-to-Text Replacement
Edit `lib/src/text_injector.py`, method `_preprocess_text()`:
```python
replacements = {
    r'\bnew_phrase\b': 'replacement',
    # Add more...
}
```

### Changing Default Configuration
Edit `lib/src/config_manager.py`, `default_config` dictionary.

### Adding a New Audio Feedback Sound
1. Place .ogg/.wav/.mp3 file in `/opt/hyprwhspr/share/assets/`
2. Reference in config: `"start_sound_path": "custom.ogg"`

### Model Selection in Config
```json
{
  "model": "small.en"  // Will resolve to ggml-small.en.bin
}
```

## Testing

The installer includes comprehensive testing:
- `./scripts/test-services.sh` - Test systemd services
- Tray health checks - Auto-detect stuck recording states
- GPU test: `/opt/hyprwhspr/scripts/build-whisper-nvidia.sh --test`

## Important Constraints

- **Requires uinput group membership** - User must be in `uinput` group and relogin
- **Wayland/Hyprland specific** - Designed for Hyprland compositor
- **ydotool dependency** - Required for paste injection
- **GPU for large models** - Large/large-v3 models need NVIDIA GPU for speed
- **Omarchy-optimized** - Tested primarily on Omarchy/Arch Linux
