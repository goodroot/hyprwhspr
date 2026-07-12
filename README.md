<h1 align="center">
    hyprwhspr
</h1>

<p align="center">
    <b>Native speech-to-text for Linux</b> - Fast, accurate and private system-wide dictation
</p>

<p align="center">
    instant performance | Cohere / Parakeet / Whisper / Gemini / ElevenLabs / REST API | stylish visuals
</p>

 <p align="center">
    <i>Supports Arch, Debian, Ubuntu, Fedora, openSUSE and more</i>
 </p>

 <p align="center">
    <i> <a href="https://hyprwhspr.com">hyprwhspr.com</a></i>
 </p>

https://github.com/user-attachments/assets/4c223e85-2916-494f-b7b1-766ce1bdc991

---

- **Built for Linux** - Native AUR package for Arch, or use Debian/Ubuntu/Fedora/openSUSE
- **Local, very fast defaults** - Instant, private and accurate performance via in-memory models
- **Latest models** - Cohere Transcribe? Turbo-v3? Parakeet TDT V3? Latest and greatest
- **GPU memory efficient** - Limit or zero memory usage easily, more for other local models
- **onnx-asr for wild CPU speeds** - No GPU? Optimized for great speed on any hardware
- **Translation** - Translate non-English to English with a single config
- **REST API or websockets** - Secure, fast wires to top clouds like Gemini, ElevenLabs
- **Themed visualizer** - Visualizes your voice, will automatch Omarchy theme
- **Word overides and prompts** - Custom hot keys, common words, and more
- **Multi-lingual** - Great performance in many languages
- **Long form mode with saving** - Pause, think, resume, pause: submit... Bam!
- **Auto-paste anywhere** - Instant paste into any active buffer, or even auto enter (optional)
- **Audio ducking 🦆** - Reduces system volume on record (optional)

---

**Why hyprwhspr?** There are a lotta dictation apps. This one is designed to work great for most people, and in particular those with highest end machines. If you've a recent Nvidia card, hyprwhspr is designed from the ground up for the **best possible accuracy and speed**. Other setups will run as well as they can on the hardware you bring. It's also actively maintained, and fully featured. Works on  anything with Wayland.

---

## Quick start

### Prerequisites

- **Linux** with systemd (Arch, Debian, Ubuntu, Fedora, openSUSE, etc.)
- **Requires a Wayland session** (GNOME, KDE Plasma Wayland, Sway, Hyprland, Niri)

- **Waybar or Noctalia** (optional, for status bar)
- **gtk4 + PyCairo** (optional, for visualizer)
- **NVIDIA GPU** (optional, for CUDA acceleration)
- **AMD/Intel GPU / APU** (optional, for Vulkan acceleration)

### Install

#### Arch Linux

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

#### Ubuntu, Debian, Fedora, openSUSE

```bash
curl -fsSL https://hyprwhspr.com/install.sh | bash
```

Installs dependencies, clones to `~/.local/share/hyprwhspr/src`, and walks you through setup. 


Re-run it any time to update.

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

1. ✅ Configure transcription backend (Cohere Transcribe, Parakeet TDT V3, Whisper, REST API, or Realtime WebSocket)
2. ✅ Download models
3. ✅ Configure themed visualizer for maximum coolness (optional)
4. ✅ Configure bar integration for your shell -- Waybar or Noctalia (optional)
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

> **What you'll see while recording:** on layer-shell compositors (Hyprland, Sway, niri, KDE) the animated mic OSD overlay -- on Noctalia / Omarchy it auto-matches your live shell theme; on GNOME/Mutter you may need to make additional changes. See [Themed visualizer](docs/CONFIGURATION.md#themed-visualizer) for details.

Any snags, please [create an issue](https://github.com/goodroot/hyprwhspr/issues/new/choose).

### Updating

```bash
# Arch: update via your AUR helper
yay -Syu hyprwhspr

# Other distros: re-run the installer
curl -fsSL https://hyprwhspr.com/install.sh | bash

# Either way, setup is idempotent if you need to re-run it
hyprwhspr setup
```

### CLI commands

After installation, use the `hyprwhspr` CLI to manage your installation:

- `hyprwhspr setup` - Interactive initial setup
- `hyprwhspr config` - Manage configuration (`show` / `show --all` / `edit` / `secondary-shortcut`)
- `hyprwhspr model` - Manage models (`download` / `list` / `status` / `unload` / `reload`)
- `hyprwhspr record` - External hotkey control (`start` / `stop` / `toggle` / `cancel` / `capture` / `status`)
- `hyprwhspr status` - Overall status check
- `hyprwhspr validate` - Validate installation
- `hyprwhspr test` - Test microphone and transcription end-to-end
- `hyprwhspr keyboard` - List/test keyboard devices (`list` / `test`)
- `hyprwhspr waybar` / `hyprwhspr noctalia` - Manage bar integration
- `hyprwhspr mic-osd` - Enable/disable the mic OSD (`enable` / `disable` / `status`)
- `hyprwhspr systemd` - Manage systemd services
- `hyprwhspr uninstall` - Remove hyprwhspr and user data

For the full command reference, see the **[Configuration guide](docs/CONFIGURATION.md)**.

## Documentation

For full configuration and customization, see the **[Configuration guide](docs/CONFIGURATION.md)**.

- [Minimal configuration](docs/CONFIGURATION.md#minimal-configuration)
- [Recording modes](docs/CONFIGURATION.md#recording-modes) -- toggle, push-to-talk, auto, long-form
- [Custom hotkeys](docs/CONFIGURATION.md#custom-hotkeys) -- key support, secondary shortcuts, Hyprland bindings
- [Backends](docs/CONFIGURATION.md#backends) -- Cohere Transcribe, Parakeet, Whisper, REST API, Realtime WebSocket
- [GPU resource management](docs/CONFIGURATION.md#gpu-resource-management) -- unload/reload model to free VRAM
- [Audio and visual feedback](docs/CONFIGURATION.md#audio-and-visual-feedback) -- visualizer, audio feedback, ducking
- [Text processing](docs/CONFIGURATION.md#text-processing) -- word overrides, filler words, symbol replacements
- [Paste and clipboard behavior](docs/CONFIGURATION.md#paste-and-clipboard-behavior) -- paste mode, per-app paste keys, non-QWERTY, auto-submit
- [Integrations](docs/CONFIGURATION.md#integrations) -- Waybar, Noctalia, Hyprland bindings, external hotkey systems
- [Troubleshooting](docs/CONFIGURATION.md#troubleshooting)

## Getting help

1. **Check logs**: `journalctl --user -u hyprwhspr.service`
2. **Verify permissions**: Run the permissions fix script
3. **Test components**: Check ydotool, audio devices, whisper.cpp
4. **Report issues**: [Create an issue](https://github.com/goodroot/hyprwhspr/issues/new/choose) - logging info helpful!

## License

MIT License - see [LICENSE](LICENSE) file.

## Contributing

Create an issue, happy to help!  

For pull requests, also best to start with an issue.

If you want, compute credits from [opub.dev](https://opub.dev/github/goodroot/hyprwhspr) are always welcome!

---

**Built with ❤️ in 🇨🇦**
