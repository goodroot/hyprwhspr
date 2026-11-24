<h1 align="center">
    hyprwhspr
</h1>

<p align="center">
    <b>Native speech-to-text for Linux</b> - Fast, accurate, private and hackable system-wide dictation
</p>

<p align="center">
    instant performance | Cohere / Parakeet / Whisper / Qwen3-ASR / ElevenLabs / REST API | stylish visuals
</p>

 <p align="center">
    <i>Supports NixOS, Arch, Debian, Ubuntu, Fedora, openSUSE and more</i>
 </p>

 <p align="center">
    <i> <a href="https://hyprwhspr.com">hyprwhspr.com</a></i>
 </p>

https://github.com/user-attachments/assets/4c223e85-2916-494f-b7b1-766ce1bdc991

---

- **Built for Linux** - Native AUR package for Arch, or use Debian/Ubuntu/Fedora/openSUSE/NixOS
- **Very fast defaults** - Nearly instant and accurate performance via in-memory models
- **Offline and private** - Transcription never leaves your machine. Cloud APIs are fully optional. 
- **Latest models** - Cohere Transcribe? Orukeet? Parakeet TDT V3? Qwen3-ASR? Latest and greatest
- **GPU memory efficient** - Limit or zero memory usage easily, more for other local models
- **onnx-asr for wild CPU speeds** - No GPU? Optimized for great speed on any hardware
- **Strong CJK** - Qwen3-ASR for Chinese, Japanese and Korean
- **Translation** - Translate non-English to English with a single config
- **REST API or websockets** - Secure, fast wires to top clouds like Gemini, ElevenLabs, Deepgram
- **Live streaming** - Self-host NeMo or Phonon: watch words form, or type as you speak (experimental)
- **Themed visualizer** - Visualizes your voice, will automatch Omarchy and Noctalia themes
- **Word overrides and prompts** - Custom hot keys, common words, and more
- **Multi-lingual** - Great performance in many languages
- **Long form mode with saving** - Pause, think, resume, pause: submit... Bam!
- **Auto-paste anywhere** - Instant paste into any active buffer, or even auto enter (optional)
- **Audio ducking 🦆** - Reduces or pauses other audio on record (optional)

---

**Why hyprwhspr?** There are a lotta dictation apps. This one is built ground-up for performance and stability: your hardware will auto-optimize to the **best possible accuracy and speed**. Actively maintained, at the leading edge, fully featured, strictly offline unless you use an API, and works on both Wayland and X11.

---

## Quick start

### Prerequisites

- **Linux** with systemd (Arch, Debian, Ubuntu, Fedora, openSUSE, NixOS etc.)
- **Python 3.11-3.15**
- **Wayland or X11 session** (GNOME, KDE Plasma, Sway, Hyprland, Niri, etc.).
- **Clipboard/window tools:**
    - `wl-clipboard` and `wtype` on Wayland
    - `xdotool` as a fallback for Sway XWayland apps like Steam and Wine
    - `python-pyperclip`, `xclip`, `xdotool`, and `xprop` on X11 (installed by the dependency script)
- **[Waybar](docs/integrations/WAYBAR.md), [Noctalia](docs/integrations/NOCTALIA.md), or [Quickshell](docs/integrations/QUICKSHELL.md)** (optional, for status bar)
- **gtk4 + PyCairo** (optional, for visualizer)
- **NVIDIA GPU** (optional, for CUDA acceleration)
- **AMD/Intel GPU / APU** (optional, for Vulkan acceleration)

### Install

#### Arch Linux

Install from the AUR:

```bash
# Stable
yay -S hyprwhspr

# Or bleeding edge
yay -S hyprwhspr-git
```

#### Ubuntu, Debian, Fedora, openSUSE

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

#### Nix and NixOS (beta)

Use the [Nix flake and NixOS module](docs/integrations/NIX.md) for x86_64 Linux
with Whisper.cpp CPU transcription. 

### Setup

Choose your backend and desktop integrations. If the installer hasn't already
walked you through setup, run:

```bash
hyprwhspr setup
```

### First use

> Ensure your microphone of choice is available in audio settings!

1. **Log out and back in** if setup asked you to
2. **Press `Super+Alt+D`** to start dictation - _beep!_
3. **Speak naturally**
4. **Press `Super+Alt+D`** again to stop dictation - _boop!_
5. **Bam!** Text appears in active buffer!

> **Recording indicator:** when enabled and its dependencies are available, supported desktops show an overlay or a status notification. Noctalia / Omarchy can match the overlay to your shell theme. See [Themed visualizer](docs/usage/FEEDBACK.md#themed-visualizer) for details.

Any snags, please [create an issue](https://github.com/goodroot/hyprwhspr/issues/new/choose).

### Updating

```bash
# Arch: update via your AUR helper
yay -Syu hyprwhspr

# Installed with install.sh
hyprwhspr update
```

For an old git-clone install, re-run the installer once to move to managed releases:

```bash
curl -fsSL https://hyprwhspr.com/install.sh | bash
```

See the [installation guide](docs/MANAGED_INSTALLATION.md) for managed release
updates and recovery, or the [Nix guide](docs/integrations/NIX.md) for Nix updates.

### CLI commands

- `hyprwhspr setup` - Interactive initial setup
- `hyprwhspr update [--version VERSION]` - Update a managed release
- `hyprwhspr install status` - Inspect managed installation state
- `hyprwhspr install repair [--python PATH]` - Repair a managed installation
- `hyprwhspr config` - Manage configuration (`show` / `show --all` / `edit` / `secondary-shortcut`)
- `hyprwhspr model` - Manage models (`download` / `list` / `status` / `unload` / `reload`)
- `hyprwhspr record` - External hotkey control (`start` / `stop` / `toggle` / `cancel` / `capture` / `status` / `copy-last` / `paste-last` / `clear-last`)
- `hyprwhspr transcribe INPUT` - Transcribe a WAV or MP3 file to stdout (use `-o PATH` to write a file)
- `hyprwhspr status` - Overall status check
- `hyprwhspr validate` - Validate installation
- `hyprwhspr test` - Test microphone and transcription end-to-end
- `hyprwhspr keyboard` - List/test keyboard devices (`list` / `test`)
- `hyprwhspr waybar` / `hyprwhspr noctalia` - Manage bar integration
- `hyprwhspr mic-osd` - Enable/disable the mic OSD (`enable` / `disable` / `status`)
- `hyprwhspr systemd` - Manage systemd services
- `hyprwhspr uninstall` - Remove hyprwhspr; preserve settings, credentials and models
- `hyprwhspr uninstall --purge` - Also remove recorded personal files (`--keep-models` preserves models)

## Documentation

All of it at **[hyprwhspr.com/docs](https://hyprwhspr.com/docs/)**, or here:

- [Configuration](docs/CONFIGURATION.md) -- the config file, validation, minimal setup
- [Recording modes](docs/usage/RECORDING_MODES.md) · [Hotkeys](docs/usage/HOTKEYS.md) · [Paste](docs/usage/PASTE.md) · [Text processing](docs/usage/TEXT_PROCESSING.md) · [Feedback](docs/usage/FEEDBACK.md)
- [Backends](docs/backends/OVERVIEW.md) · [Memory](docs/backends/MEMORY.md) -- which model, and freeing it
- [Troubleshooting](docs/TROUBLESHOOTING.md)

## Getting help

- **Check logs**: `journalctl --user -u hyprwhspr.service`
- **[Troubleshooting](docs/TROUBLESHOOTING.md)** covers common issues
- **Still stuck?** [Create an issue](https://github.com/goodroot/hyprwhspr/issues/new/choose) - logs helpful!

## License

MIT License - see [LICENSE](LICENSE) file.

## Contributing

Start with an issue - happy to help! Pull requests best begin there too.

To run your fork in an activated development environment:

```bash
git clone <your-fork>
cd hyprwhspr
python -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -r requirements-onnx-asr.txt
./bin/hyprwhspr
```

An activated venv inside the checkout runs the app and CLI. You still need system
dependencies (see the manual install above) and [a configured backend and models](docs/backends/PARAKEET.md).
Stop the service first: `systemctl --user stop hyprwhspr.service`.

Upvotes on the AUR and/or GitHub stars are also a nice way to show appreciation.

---

**Built with ❤️ in 🇨🇦**
