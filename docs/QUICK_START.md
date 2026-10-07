# Quick start

## Prerequisites

- **Linux** with systemd (Arch, Debian, Ubuntu, Fedora, openSUSE, etc.)
- **Python 3.11-3.14**
- **Wayland or X11 session** (GNOME, KDE Plasma, Sway, Hyprland, Niri, etc.).
- **Clipboard/window tools:**
    - `wl-clipboard` and `wtype` on Wayland
    - `python-pyperclip`, `xclip`, `xdotool`, and `xprop` on X11 (installed by the dependency script)
- **Waybar or Noctalia** (optional, for status bar)
- **gtk4 + PyCairo** (optional, for visualizer)
- **NVIDIA GPU** (optional, for CUDA acceleration)
- **AMD/Intel GPU / APU** (optional, for Vulkan acceleration)

## Install

### Arch Linux

On the AUR:

```bash
# Install for stable
yay -S hyprwhspr

# Or install for bleeding edge
yay -S hyprwhspr-git
```

Then run the interactive setup:

```bash
hyprwhspr setup
```

### Ubuntu, Debian, Fedora, openSUSE

```bash
curl -fsSL https://hyprwhspr.com/install.sh | bash
```

Installs dependencies, clones to `~/.local/share/hyprwhspr/src`, and walks you through setup.

<details>
<summary>Manual install</summary>

```bash
# Clone the repo
git clone https://github.com/goodroot/hyprwhspr.git
cd hyprwhspr

# Install dependencies for your distro
./scripts/install-deps.sh

# Run interactive setup
./bin/hyprwhspr setup
```

</details>

**Setup then walks you through:**

1. ✅ Configure transcription backend (Cohere Transcribe, Parakeet TDT V3, Whisper, Qwen3-ASR, REST API, or Realtime WebSocket)
2. ✅ Download models
3. ✅ Configure themed visualizer for maximum coolness (optional)
4. ✅ Configure bar integration for your shell -- Waybar or Noctalia (optional)
5. ✅ Set up systemd user services 
6. ✅ Set up permissions
7. ✅ Validate installation

## First use

> Ensure your microphone of choice is available in audio settings!

1. **Log out and back in** (for group permissions)
2. **Press `Super+Alt+D`** to start dictation - _beep!_
3. **Speak naturally**
4. **Press `Super+Alt+D`** again to stop dictation - _boop!_
5. **Bam!** Text appears in active buffer!

> **What you'll see while recording:** on layer-shell compositors (Hyprland, Sway, niri, KDE) the animated mic OSD overlay -- on Noctalia / Omarchy it auto-matches your live shell theme; on GNOME/Mutter you may need to make additional changes. See [Themed visualizer](usage/FEEDBACK.md#themed-visualizer) for details.

Any snags, please [create an issue](https://github.com/goodroot/hyprwhspr/issues/new/choose).

## Updating

```bash
# Arch: update via your AUR helper
yay -Syu hyprwhspr

# Managed release installs
hyprwhspr update

# Legacy bootstrap installs: re-run the installer
curl -fsSL https://hyprwhspr.com/install.sh | bash

# Either way, setup is idempotent if you need to re-run it
hyprwhspr setup
```

Managed release rollout and recovery: [installation guide](MANAGED_INSTALLATION.md).
