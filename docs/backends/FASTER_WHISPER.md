# faster-whisper

Local Whisper via [faster-whisper](https://github.com/SYSTRAN/faster-whisper).

Run `hyprwhspr setup` and select **Whisper**. On NVIDIA or CPU, setup uses faster-whisper.

**Best for:** CPU users wanting faster inference than whisper.cpp, or NVIDIA GPU users where VRAM is constrained — INT8 quantization runs `large-v3-turbo` in ~3.1 GB vs ~6 GB for float16. AMD/Intel GPU users should use Parakeet or whisper.cpp instead (CTranslate2 does not support Vulkan or ROCm).

Built-in Silero VAD strips silence before inference — the most effective mitigation for Whisper's hallucination loops on longer recordings.

```jsonc
{
    "transcription_backend": "faster-whisper",
    "faster_whisper_model": "large-v3-turbo",   // CUDA; use "base" or "small" for CPU
    "faster_whisper_device": "auto",             // auto | cuda | cpu
    "faster_whisper_compute_type": "auto",       // auto → int8; float16/float32 trade memory for precision
    "faster_whisper_vad_filter": true            // Silero VAD (default: true)
}
```

## Available models

| Model | Size (INT8) | Notes |
|-------|-------------|-------|
| `tiny` | ~75 MB | Fastest |
| `base` | ~145 MB | Recommended for CPU |
| `small` | ~484 MB | Better accuracy |
| `medium` | ~1.5 GB | High accuracy |
| `large-v3` | ~3.1 GB | Best accuracy (needs GPU) |
| `large-v3-turbo` | ~1.6 GB | **Recommended for CUDA** |
| `distil-large-v3` | ~1.5 GB | Distilled, CPU/GPU balance |

Models stored in: `~/.cache/huggingface/hub/`

Language, prompts, translation and decoding: [Language and prompts](LANGUAGE.md).
