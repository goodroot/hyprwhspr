# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**hyprwhspr** is a native speech-to-text application for Arch Linux/Omarchy with Hyprland desktop environment. It provides fast, accurate, and easy system-wide dictation using OpenAI's Whisper, with automatic GPU acceleration support (NVIDIA CUDA and AMD ROCm).

**Key Features:**
- Global hotkey dictation (Super+Alt+D by default)
- Hot model loading via pywhispercpp backend
- GPU acceleration support
- Waybar integration
- Audio feedback
- Text injection into any application
- Word overrides and custom prompts
- Automatic punctuation/symbol conversion

## Common Development Commands

### Installation & Setup
```bash
# Full installation for Omarchy/Arch
./scripts/install-omarchy.sh

# Only install/update systemd services
./scripts/install-services.sh

# Test service configuration
./scripts/test-services.sh

# Fix uinput permissions
/usr/lib/hyprwhspr/scripts/fix-uinput-permissions.sh
```

### Running the Application
```bash
# Via systemd service (recommended)
systemctl --user start hyprwhspr.service
systemctl --user stop hyprwhspr.service
systemctl --user restart hyprwhspr.service
systemctl --user status hyprwhspr.service

# View logs
journalctl --user -u hyprwhspr.service -f
journalctl --user -u ydotool.service -f

# Direct execution (development)
./bin/hyprwhspr

# Or directly with Python
source ~/.local/share/hyprwhspr/venv/bin/activate
python3 lib/main.py
```

### Dependencies
Python dependencies are managed via `requirements.txt`:
- **Audio processing:** sounddevice, numpy, scipy
- **Global shortcuts:** evdev, pyperclip
- **Whisper integration:** pywhispercpp (v1.3.3)
- **System integration:** psutil, rich

Dependencies are installed into a user-space virtual environment at `~/.local/share/hyprwhspr/venv/`.

### Service Management
The application uses two systemd services:
- **hyprwhspr.service** - Main application with auto-restart
- **ydotool.service** - Input injection daemon

Both are user-level services (no root required).

## Architecture

### Directory Structure
```
/home/will/Applications/hyprwhspr
├── bin/hyprwhspr                    # Main launcher script
├── lib/
│   ├── main.py                      # Entry point, application orchestrator
│   └── src/                         # Core modules
│       ├── config_manager.py        # Configuration management (JSON-based)
│       ├── whisper_manager.py       # Speech-to-text via pywhispercpp
│       ├── audio_capture.py         # Audio recording from microphone
│       ├── audio_manager.py         # Audio feedback (start/stop sounds)
│       ├── text_injector.py         # Text injection into applications
│       ├── global_shortcuts.py      # Global keyboard shortcuts (evdev)
│       └── logger.py                # Logging utilities
├── scripts/
│   ├── install-omarchy.sh           # Main installation script
│   ├── install-services.sh          # Systemd service setup
│   ├── test-services.sh             # Service testing
│   └── fix-uinput-permissions.sh    # Permission fix
├── config/                          # Configuration templates
│   ├── hyprland/                    # Hyprland/Waybar integration
│   ├── systemd/                     # Systemd service templates
│   └── waybar/                      # Waybar module files
├── share/assets/                    # Sound files and resources
└── requirements.txt                 # Python dependencies
```

### Core Application Flow

**Entry Point:** `lib/main.py` - `hyprwhsprApp` class

1. **Initialization:**
   - `ConfigManager` - Loads config from `~/.config/hyprwhspr/config.json`
   - `AudioCapture` - Sets up audio device for recording
   - `AudioManager` - Configures audio feedback sounds
   - `WhisperManager` - Loads Whisper model into memory
   - `TextInjector` - Sets up text injection mechanism
   - `GlobalShortcuts` - Registers global hotkey (default: Super+Alt+D)

2. **Runtime Flow:**
   - User presses hotkey → `_on_shortcut_triggered()`
   - If not recording → `_start_recording()` → audio feedback → start capture
   - If recording → `_stop_recording()` → audio feedback → process audio
   - `_process_audio()` → Whisper transcription → `_inject_text()` → paste

3. **System Integration:**
   - **Audio:** Uses `sounddevice` for capture, 16kHz sample rate
   - **Shortcuts:** Uses `evdev` for global keyboard monitoring
   - **Text Injection:** Clipboard-based with configurable paste method
   - **Model Loading:** pywhispercpp keeps model hot in memory
   - **GPU Acceleration:** Automatic detection (CUDA/ROCm/Vulkan/CPU)

### Configuration System

**Location:** `~/.config/hyprwhspr/config.json`

**Key Settings:**
- `primary_shortcut` - Global hotkey (format: "SUPER+ALT+D")
- `model` - Whisper model name ("base", "small", "medium", "large", etc.)
- `threads` - CPU thread count for processing
- `language` - Language code (null for auto-detect)
- `word_overrides` - Dictionary of word replacements
- `whisper_prompt` - Transcription prompt/guidance
- `paste_mode` - "super" | "ctrl_shift" | "ctrl"
- `clipboard_behavior` - Auto-clear clipboard after injection
- `audio_feedback` - Enable/disable sound notifications

**Defaults** are defined in `config_manager.py:16-30`.

### Service Architecture

**Systemd Services** are defined in `config/systemd/`:

**hyprwhspr.service:**
- Runs the main Python application
- User-level service
- Restarts on failure
- Depends on ydotool.service

**ydotool.service:**
- Runs ydotool daemon for input injection
- User-level service
- Required for text injection functionality

Both services are started automatically on login if enabled.

### Waybar Integration

**Files:** `config/waybar/hyprwhspr-style.css`, `config/hyprland/hyprwhspr-tray.sh`

The tray script `hyprwhspr-tray.sh` provides:
- Status monitoring (reads from `~/.config/hyprwhspr/recording_status`)
- Start/stop/toggle operations via systemd
- Waybar JSON output for dynamic icon display

Click interactions:
- Left-click: Toggle dictation
- Right-click: Start (if not running)
- Middle-click: Restart service

## Installation Details

**Installation Directory:** `/usr/lib/hyprwhspr/` (read-only system files)
**User Data:** `~/.local/share/hyprwhspr/` (Python venv, runtime data)
**Config:** `~/.config/hyprwhspr/` (user configuration)
**Models:** `~/.local/share/pywhispercpp/models/` (Whisper model files)

**Installation Process** (`scripts/install-omarchy.sh:1-200`):
1. Detects actual user (supports sudo usage)
2. Creates directory structure
3. Copies system files to `/usr/lib/hyprwhspr/`
4. Sets up Python venv in user space
5. Installs pywhispercpp backend
6. Downloads base Whisper model
7. Configures systemd services
8. Sets up Waybar integration
9. Runs health checks

**Models:** Default model is `ggml-base.en.bin` (~148MB). GPU models (large, large-v3) require NVIDIA CUDA or AMD ROCm.

## Text Injection System

**Mechanism:** Clipboard-based with configurable paste methods:

1. **Super Mode (default):** Copies text to clipboard, sends Super+V
2. **Ctrl+Shift Mode:** Copies to clipboard, sends Ctrl+Shift+V
3. **Ctrl Mode:** Copies to clipboard, sends Ctrl+V

**Punctuation Replacement:** Automatic conversion of spoken punctuation to symbols (e.g., "period" → ".", "comma" → ",") - defined in `text_injector.py`.

**Clipboard Management:** Configurable auto-clear after delay via `clipboard_behavior` setting.

## Development Notes

- **No test suite:** The project doesn't include automated tests
- **No build system:** Uses simple Python script execution, no setup.py or pyproject.toml
- **Single entry point:** All functionality flows through `lib/main.py`
- **Hot model loading:** pywhispercpp keeps Whisper model in memory for fast transcription
- **Event-driven:** Based on global shortcut events, not a GUI loop
- **State tracking:** Simple booleans for recording/processing state
- **No logging file:** Uses console output, logs via `logger.py` module
- **Configuration-driven:** Behavior primarily controlled via config.json

## Troubleshooting Resources

- Service status: `systemctl --user status hyprwhspr.service ydotool.service`
- Service logs: `journalctl --user -u hyprwhspr.service -f`
- Audio devices: `pactl list short sources`
- Model files: `ls -la ~/.local/share/pywhispercpp/models/`
- Config file: `cat ~/.config/hyprwhspr/config.json`
- Health check: `/usr/lib/hyprwhspr/config/hyprland/hyprwhspr-tray.sh health`

## External Dependencies

**System-level (installed by installer):**
- ydotool - Input injection daemon
- PipeWire/PulseAudio - Audio system
- systemd - Service management

**Python (in requirements.txt):**
- pywhispercpp (1.3.3) - Whisper backend with hot model loading
- sounddevice - Audio capture
- evdev - Global keyboard shortcuts
- pyperclip - Clipboard access
- numpy, scipy - Audio processing
- psutil - System utilities
- rich - Terminal formatting
