# Quick start

## Prerequisites

- **Linux** with systemd (Arch, Debian, Ubuntu, Fedora, openSUSE, etc.)
- **Python 3.11-3.15**
- **Wayland or X11 session** (GNOME, KDE Plasma, Sway, Hyprland, Niri, etc.).
- **Clipboard/window tools:**
    - `wl-clipboard` and `wtype` on Wayland
    - `python-pyperclip`, `xclip`, `xdotool`, and `xprop` on X11 (installed by the dependency script)
- **Waybar or Noctalia** (optional, for status bar)
- **gtk4 + PyCairo** (optional, for visualizer)
- **NVIDIA GPU** (optional, for CUDA acceleration)
- **AMD/Intel GPU / APU** (optional, for Vulkan acceleration)

## Install

### Nix and NixOS (beta)

Use the [Nix flake and NixOS module](integrations/NIX.md) for a packaged
Whisper.cpp CPU runtime. Nix installations use configuration and model commands
instead of the interactive installer. We'd appreciate
[issue reports](https://github.com/goodroot/hyprwhspr/issues).

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

Installs dependencies and the latest release, then walks you through setup.

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

**Setup:** pick a backend, review your preferences and choose desktop integrations. Setup asks before changing your bar or desktop shortcuts. Backend and model downloads can take several minutes. Follow the final checks and any remaining steps before trying dictation.

Want to choose everything? Answer no to recommended settings.

For automatic backend selection and default settings:

```bash
hyprwhspr setup auto
```

Authentication may still ask for your password.

## First use

> Ensure your microphone of choice is available in audio settings!

1. **Log out and back in** if setup asked you to
2. **Press `Super+Alt+D`** to start dictation - _beep!_
3. **Speak naturally**
4. **Press `Super+Alt+D`** again to stop dictation - _boop!_
5. **Bam!** Text appears in active buffer!

> **Recording indicator:** when enabled and its dependencies are available, supported desktops show an overlay or a status notification. Noctalia / Omarchy can match the overlay to your shell theme. See [Themed visualizer](usage/FEEDBACK.md#themed-visualizer) for details.

Any snags, please [create an issue](https://github.com/goodroot/hyprwhspr/issues/new/choose).

## Updating

```bash
# Arch: update via your AUR helper
yay -Syu hyprwhspr

# Installed with install.sh
hyprwhspr update

# Old git-clone installs: re-run the installer once to move to releases
curl -fsSL https://hyprwhspr.com/install.sh | bash

# Either way, setup is idempotent if you need to re-run it
hyprwhspr setup
```

Managed release rollout and recovery: [installation guide](MANAGED_INSTALLATION.md).
