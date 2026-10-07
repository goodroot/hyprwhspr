# Memory management

A local model lives in memory while the service runs: VRAM on a GPU, RAM on CPU. Cloud backends hold none.

## What each model needs

Roughly, at default settings:

| Model | Backend | Memory |
|-------|---------|-------:|
| Parakeet v3 · Orukeet | [Parakeet](PARAKEET.md) | 1 GB |
| Parakeet v3 | [Parakeet.cpp](PARAKEET_CPP.md) | 0.9 GB |
| Whisper `base` | [whisper.cpp](WHISPER_CPP.md) · [faster-whisper](FASTER_WHISPER.md) | ~150 MB |
| Whisper `small` | whisper.cpp · faster-whisper | ~490 MB |
| Whisper `large-v3-turbo` | whisper.cpp · faster-whisper | 1.6 GB |
| Whisper `large-v3` | whisper.cpp · faster-whisper | ~3 GB |
| Qwen3-ASR 0.6B | [Qwen3-ASR](QWEN3_ASR.md) | ~1.0 GB + context |
| Qwen3-ASR 1.7B | Qwen3-ASR | ~2.4 GB + context |
| Cohere Transcribe | [Cohere Transcribe](COHERE_TRANSCRIBE.md) | 4 GB VRAM · 8 GB RAM on CPU |
| — | [REST API](REST_API.md) · [Realtime WebSocket](REALTIME_WEBSOCKET.md) | none |

Whisper figures are model sizes; expect some overhead on top. Qwen3-ASR's context adds ~0.9 GB at the default 8192 tokens.

## Spending less

- **Smaller model.** Whisper `base` or `small`, Qwen3-ASR `0.6b-q8_0`.
- **Quantize.** Parakeet runs `int8` by default (`onnx_asr_quantization`); keep it. faster-whisper's `auto` compute type is `int8`; `float16` and `float32` cost more.
- **Keep Cohere on bfloat16.** Its GPU default. `float32` doubles the footprint.
- **Trim Qwen3-ASR's context.** `qwen3_asr_ctx_size` sizes the KV cache, ~112 KiB per token on the 1.7B. `null` means llama.cpp's 32000, ~3.5 GB.
- **Move to CPU.** Frees VRAM, spends RAM.
- **Go remote.** A cloud or self-hosted backend keeps memory off this machine.
- **Share the model.** `hyprwhspr transcribe` borrows an idle service's model rather than loading a second.

## Unload and reload

Need the GPU back for a game or a local LLM? Unload the model. The service stays up, shortcuts stay bound:

```bash
hyprwhspr model unload   # Free the model's VRAM or RAM
hyprwhspr model reload   # Load it again
```

While unloaded, recording is refused with a notification, and the Waybar tray shows a `󰒲` sleep icon. Works on every local backend; Qwen3-ASR stops its sidecar. Cloud backends have nothing to unload.

Bind both in `~/.config/hypr/hyprland.conf`:

```bash
# Free the GPU before starting a local LLM
bindd = SUPER ALT, U, Unload speech model, exec, hyprwhspr model unload

# Reclaim dictation when done
bindd = SUPER ALT, L, Reload speech model, exec, hyprwhspr model reload
```
