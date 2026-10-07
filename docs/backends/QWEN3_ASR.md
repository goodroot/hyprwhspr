# Qwen3-ASR

[Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR), local through a pinned llama.cpp sidecar. Best for Chinese, Japanese and Korean. Thirty languages; 22 Chinese dialects.

Run `hyprwhspr setup`; choose **Qwen3-ASR**. It adds no Python packages—only the runtime (17–34 MB) and model pair.

```jsonc
{
    "transcription_backend": "qwen3-asr",
    "qwen3_asr_model": "1.7b-q8_0",   // 1.7b-q8_0 (quality) or 0.6b-q8_0 (smaller)
    "qwen3_asr_device": "auto",       // auto | cpu | vulkan
    "qwen3_asr_timeout": 180,         // sidecar request timeout, seconds (1-600)
    "qwen3_asr_ctx_size": 8192        // llama-server context, tokens (512-65536); null = llama.cpp default
}
```

`auto` keeps an installed runtime. Otherwise: Vulkan when found, CPU when not. Vulkan covers NVIDIA, AMD and Intel; there is no Linux CUDA build.

`qwen3_asr_ctx_size` sizes the KV cache, ~112 KiB per token on the 1.7B. 8192 is ~0.9 GiB and covers ~5 minutes of audio; requests carry at most 2. `null` restores llama.cpp's 32000, ~3.5 GiB.

Long audio splits at pauses, then joins again. Set `language` when you can; without it, the first useful segment guides the rest.

## Available models

| Model | Size | Notes |
|-------|------|-------|
| `1.7b-q8_0` | ~2.4 GB | **Recommended** · best quality |
| `0.6b-q8_0` | ~1.0 GB | Smaller · faster · less accurate |

Models stored in: `~/.local/share/hyprwhspr/qwen3-asr/models/`

## Languages

Chinese, Cantonese, English, Japanese, Korean, Arabic, German, French, Spanish, Portuguese, Italian, Russian, Dutch, Polish, Turkish, Thai, Vietnamese, Hindi, Indonesian, Malay and more.

Leave `language` unset to detect it. Set an ISO code to hold it steady.

> **Note:** No streaming, timestamps, alignment or prompts. `whisper_prompt_*` does not apply.
