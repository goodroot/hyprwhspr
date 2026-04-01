<h1 align="center">
    hyprwhspr
</h1>

<p align="center">
    <b>Native speech-to-text for Linux</b> - Fast, accurate and private system-wide dictation
</p>

<p align="center">
    instant performance | Cohere / Parakeet / Whisper / ElevenLabs / REST API | stylish visuals
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

1. ✅ Configure transcription backend (Cohere Transcribe, Parakeet TDT V3, Whisper, REST API, or Realtime WebSocket)
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

```bash
# Clone the repo
git clone https://github.com/goodroot/hyprwhspr.git
cd hyprwhspr

# Install dependencies for your distro (Ubuntu, Debian, Fedora, openSUSE)
./scripts/install-deps.sh

# Run interactive setup
./bin/hyprwhspr setup
```

After setup, log out and back in for group permissions, then:

```bash
hyprwhspr status
```

> Non-Arch distro support is new - please report any snags!

### CLI commands

After installation, use the `hyprwhspr` CLI to manage your installation:

- `hyprwhspr setup` - Interactive initial setup
- `hyprwhspr config` - Manage configuration (`show` / `show --all` / `edit`)
- `hyprwhspr model` - Manage models (`download` / `list` / `unload` / `reload`)
- `hyprwhspr status` - Overall status check
- `hyprwhspr validate` - Validate installation
- `hyprwhspr test` - Test microphone and transcription end-to-end
- `hyprwhspr waybar` - Manage Waybar integration
- `hyprwhspr systemd` - Manage systemd services
- `hyprwhspr record` - External hotkey control (`start` / `stop` / `toggle`)

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
