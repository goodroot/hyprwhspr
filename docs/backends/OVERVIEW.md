# Choosing a backend

**Quick pick by hardware:**

- **NVIDIA GPU** → [Cohere Transcribe](COHERE_TRANSCRIBE.md)
- **AMD / Intel GPU** → [Parakeet.cpp](PARAKEET_CPP.md) · [whisper.cpp](WHISPER_CPP.md) (Vulkan) for 99 languages
- **CPU only** → [Parakeet](PARAKEET.md) or [faster-whisper](FASTER_WHISPER.md)
- **ARM64** → [Parakeet](PARAKEET.md)
- **Chinese, Japanese or Korean** → [Qwen3-ASR](QWEN3_ASR.md)
- **No local setup** → [REST API](REST_API.md)

| Model | Engine | Runs on | Arch | Speed | Accuracy | Memory | Languages |
|-------|--------|---------|------|:-----:|:--------:|-------:|-----------|
| Parakeet v3 | ONNX | CPU · NVIDIA | x64 · ARM64 | ●●● | ●●○ | 1 GB | 25 European |
| Orukeet † | ONNX | CPU · NVIDIA | x64 · ARM64 | ●●● | ●●○ | 1 GB | 25 European |
| Parakeet v3 † | Parakeet.cpp | CPU · Vulkan | x64 · ARM64 | ●●● | ●●○ | 0.9 GB | 25 European |
| Whisper turbo | whisper.cpp · faster-whisper | CPU · NVIDIA · Vulkan | x64 · ARM64 | ●●○ | ●●○ | 1.6 GB | 99 |
| Cohere Transcribe | PyTorch | CPU · NVIDIA | x64 · ARM64 | ●●● | ●●● | 4 GB | 14 |
| Qwen3-ASR 1.7B † | llama.cpp | CPU · Vulkan | x64 | ●●○ | ●●○ · ●●● CJK | 2.4 GB | 30 |

Speed on each model's best hardware. Accuracy from the [Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard). Memory is RAM or VRAM, roughly. ARM64 runs on CPU. Whisper ships smaller models: [faster-whisper](FASTER_WHISPER.md#available-models), [whisper.cpp](WHISPER_CPP.md#available-models). † Experimental.

Cloud or your own server: [REST API](REST_API.md) and [Realtime WebSocket](REALTIME_WEBSOCKET.md). Speed and accuracy follow the provider.

`hyprwhspr setup auto` picks Whisper for your hardware: faster-whisper on NVIDIA or CPU, whisper.cpp on AMD/Intel.

## Model commands

`hyprwhspr model` commands route automatically to the configured local backend.
For cloud backends (`rest-api` and `realtime-ws`), model operations are not
applicable and exit nonzero.

```bash
hyprwhspr model status            # Check if model is downloaded/cached
hyprwhspr model list              # Show model info for active backend
hyprwhspr model download [model]  # Download or re-download model
hyprwhspr model unload            # Free its memory; the service keeps running
hyprwhspr model reload            # Reload the model after an unload
```

Models are downloaded automatically during `hyprwhspr setup`; use `model download` to re-download if needed. To free memory, see [Memory management](MEMORY.md).
