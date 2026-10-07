# Cohere Transcribe

**#1 on the [Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard)** — 5.42 average WER across 9 benchmarks vs. Whisper large-v3's 7.44, at ~3× the throughput. [Benchmark details](https://huggingface.co/blog/CohereLabs/cohere-transcribe-03-2026-release).

**Supported languages:** English, German, French, Italian, Spanish, Portuguese, Greek, Dutch, Polish, Arabic, Vietnamese, Chinese, Japanese, Korean

This model has no language detection, so `language` must be set: `null` transcribes as English, and
any code outside the list above is refused rather than transcribed. `hyprwhspr status` shows which
language is in effect.

**Memory:** 4 GB VRAM (bfloat16), or 8 GB RAM on CPU (float32).

## Setup

Cohere Transcribe is a **gated model** on HuggingFace — you must accept the license before downloading.

1. Accept the license agreement at: [huggingface.co/CohereLabs/cohere-transcribe-03-2026](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026)
2. Generate a read token at: [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
3. Run `hyprwhspr setup` and select **Cohere** — you will be prompted for your token

The model (~4 GB) is downloaded during setup. Your token is securely stored locally in `~/.config/hyprwhspr/credentials.json` and never shared.

## Configuration

```jsonc
{
    "transcription_backend": "cohere-transcribe",
    "cohere_transcribe_device": "auto",      // auto | cuda | cpu
    "cohere_transcribe_dtype": "bfloat16",   // bfloat16 (GPU default) | float32 (CPU)
    "cohere_transcribe_compile": false       // torch.compile for faster throughput (adds warmup on first call)
}
```

On GPU the model always runs in **bfloat16** (its native precision). A `float16` setting is accepted but coerced to bfloat16 — true float16 (IEEE half) overflows the model's attention mask, so it is not usable here. Use `float32` (default on CPU) to force full precision.

Model stored in: `~/.cache/huggingface/hub/models--CohereLabs--cohere-transcribe-03-2026/`
