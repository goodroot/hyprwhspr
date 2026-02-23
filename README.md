<h1 align="center">
    hyprwhspr
</h1>

<p align="center">
    <b>Native speech-to-text for Linux</b> - Fast, accurate and private system-wide dictation
</p>

<p align="center">
    instant performance | Parakeet / Whisper / ElevenLabs / REST API | stylish visuals
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
- **REST API or websockets** - Secure, fast wires to top clouds like ElevenLabs
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

1. ✅ Configure transcription backend (Parakeet TDT V3, Whisper, REST API, or Realtime WebSocket)
2. ✅ Download models
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

### CLI commands

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
  - `hyprwhspr config show --all` - Show all settings including defaults
- `hyprwhspr waybar` - Manage Waybar integration (install/remove/status)
- `hyprwhspr mic-osd` - Manage microphone visualization overlay (enable/disable/status)
- `hyprwhspr systemd` - Manage systemd services (install/enable/disable/status/restart)
- `hyprwhspr model` - Manage models (download/list/status)
- `hyprwhspr status` - Overall status check
- `hyprwhspr validate` - Validate installation
- `hyprwhspr test` - Test microphone and backend connectivity end-to-end
  - `--live` - Record live audio (3s) instead of using test.wav
  - `--mic-only` - Only test microphone, skip transcription
- `hyprwhspr keyboard` - Keyboard device management (list/test)
- `hyprwhspr backend` - Backend management (repair/reset)
- `hyprwhspr state` - State management (show/validate/reset)
- `hyprwhspr uninstall` - Completely remove hyprwhspr and all data

## Documentation

For full configuration and customization, see the **[Configuration guide](docs/CONFIGURATION.md)**.

- [Minimal configuration](docs/CONFIGURATION.md#minimal-configuration)
- [Recording modes](docs/CONFIGURATION.md#recording-modes) -- toggle, push-to-talk, auto, long-form
- [Custom hotkeys](docs/CONFIGURATION.md#custom-hotkeys) -- key support, secondary shortcuts, Hyprland bindings
- [Transcription backends](docs/CONFIGURATION.md#transcription-backends) -- REST API, Realtime WebSocket
- [Models](docs/CONFIGURATION.md#models) -- Parakeet, Whisper
- [Audio and visual feedback](docs/CONFIGURATION.md#audio-and-visual-feedback) -- visualizer, audio feedback, ducking
- [Text processing](docs/CONFIGURATION.md#text-processing) -- word overrides, filler words, symbol replacements
- [Paste and clipboard behavior](docs/CONFIGURATION.md#paste-and-clipboard-behavior) -- paste mode, non-QWERTY, auto-submit
- [Integrations](docs/CONFIGURATION.md#integrations) -- Waybar, Hyprland bindings, external hotkey systems
- [Troubleshooting](docs/CONFIGURATION.md#troubleshooting)

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
