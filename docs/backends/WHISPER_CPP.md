# whisper.cpp

Local Whisper via [pywhispercpp](https://github.com/abdeladim-s/pywhispercpp).

Run `hyprwhspr setup` and select **Whisper**. On AMD/Intel, setup uses whisper.cpp with Vulkan.

For whisper.cpp on CPU or NVIDIA: `hyprwhspr setup auto --backend cpu` or `--backend nvidia`.

On x86-64, setup fetches a pre-built wheel: CUDA 12, or Vulkan on glibc 2.41+ (Arch, Fedora 42+, Debian 13). Otherwise it builds, and a Vulkan build needs:

- Arch: setup installs it
- Debian/Ubuntu: `libvulkan-dev glslc spirv-headers`
- Fedora: `vulkan-headers vulkan-loader-devel glslc spirv-headers-devel`
- openSUSE: `vulkan-devel shaderc spirv-headers`

If the GPU build fails, you get CPU, and the service says so at every start. Install what's missing, re-run setup, reinstall the backend.

**Best for:** modern NVIDIA cards or discrete AMD/Intel (via Vulkan) — extremely fast on GPU with `large-v3` or `large-v3-turbo`.

## Available models

Models stored in: `~/.local/share/pywhispercpp/models/`

| Model | Size | Notes |
|-------|------|-------|
| `tiny` / `tiny.en` | ~75 MB | Fastest |
| `base` / `base.en` | ~148 MB | Recommended (default) |
| `small` / `small.en` | ~488 MB | Better accuracy |
| `medium` / `medium.en` | ~1.5 GB | High accuracy |
| `large-v3` | ~2.9 GB | Best accuracy, **requires GPU** |
| `large-v3-turbo` | ~1.6 GB | Fast + accurate, **requires GPU** |

> **GPU required:** `large-v3` and `large-v3-turbo` require GPU acceleration for reasonable speed.

Download a specific model by name:

```bash
hyprwhspr model download base
hyprwhspr model download small.en
```

Set model in config (pywhispercpp only — faster-whisper uses `faster_whisper_model`):

```jsonc
{
    "model": "small.en",  // .en = English-only; omit suffix for multilingual
    // "threads": 6       // optional; omit for auto = min(8, CPU count)
}
```

## Voice activity detection

Optional native Silero VAD strips silence before inference — the same hallucination mitigation faster-whisper ships, off by default here because it needs an extra ~1 MB model (`ggml-silero-v5.1.2.bin`, auto-downloaded to the models directory on first use):

```jsonc
{
    "pywhispercpp_use_vad": true   // default: false
}
```

If the download fails (e.g. offline), the service logs a warning and continues without VAD.

Language, prompts, translation and decoding: [Language and prompts](LANGUAGE.md).
