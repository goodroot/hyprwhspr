# Language and prompts

Which language to listen for, and how to steer the words. Mostly Whisper; notes say where else each applies.

## Language detection

English-only speakers can use the smaller `.en` models; for multi-language detection, pick a model without the `.en` suffix:

```jsonc
{
    "language": null // null = auto-detect (default), or specify language code
}
```

Auto-detect (`null`, the default) runs an extra detection pass per utterance. Setting `language` explicitly skips it and lowers latency.

Whisper accepts any of its 99 supported language codes — the core set:

| Code | Language | Code | Language |
|------|----------|------|----------|
| `en` | English | `pt` | Portuguese |
| `de` | German | `nl` | Dutch |
| `fr` | French | `pl` | Polish |
| `es` | Spanish | `ru` | Russian |
| `it` | Italian | `zh` | Chinese |
| `ja` | Japanese | `ko` | Korean |

For the full list, see the [Whisper language codes](https://github.com/openai/whisper/blob/main/whisper/tokenizer.py).

## Whisper prompt

Customize transcription behavior:

```jsonc
{
    "whisper_prompt": "Transcribe as technical documentation, keeping acronyms uppercase."
}
```

The prompt influences how Whisper interprets and transcribes your audio, eg:

- `"Transcribe as technical documentation with proper capitalization, acronyms and technical terminology."`
- `"Transcribe as casual conversation with natural speech patterns."`
- `"Transcribe as an ornery pirate on the cusp of scurvy."`

A `whisper_prompt` applies to every language, and a prompt written in one language pulls
transcription toward that language. That is why the shipped capitalization default lives in
`whisper_prompt_en` instead — see below.

## Translation

Translate non-English speech into English:

```jsonc
{
    "task": "translate",
    "language": "it"  // optional: set source language, or null to auto-detect
}
```

- **`"transcribe"`** (default) - Output in the source language
- **`"translate"`** - Translate speech into English

> **Note**: Supported by `faster-whisper` and `pywhispercpp` backends. `qwen3-asr` honours `language` but has no
> translation task. `language` and `task` are independent — setting a non-English language does not imply translation.

## Language-specific prompts

Set a per-language prompt using `whisper_prompt_{lang}`:

```jsonc
{
    "whisper_prompt_de": "Transkribiere auf Deutsch. Verwende Schweizer Rechtschreibung: kein ß, immer ss."
}
```

- The language comes from `language`, `secondary_language` or `--lang`; `pywhispercpp`
  and `faster-whisper` auto-detect it when unset
- Prompts do not apply to `qwen3-asr`, which sends no prompt field (it uses `language`
  for the hint and auto-detects when unset)
- Falls back to `whisper_prompt` if no language-specific prompt is configured
- With neither set, English audio gets the shipped `whisper_prompt_en` capitalization prompt and
  other languages get no prompt

## Decoding strategy

Controls how Whisper searches for the best transcription. Applies to `pywhispercpp` and `faster-whisper` backends.

```jsonc
{
    "sampling_strategy": "beam_search",  // "beam_search" (default) or "greedy"
    "beam_size": 5                       // number of candidates to track (beam_search only)
}
```

- **`"beam_search"`** (default) — keeps the top N candidate sequences in parallel and picks the best overall result. Matches `whisper-cli` defaults. Better accuracy, especially for non-English audio and noisy input.
- **`"greedy"`** — picks the single highest-probability word at each step. Faster, lower quality.
- **`beam_size`** — higher values (e.g. `8`–`10`) can improve accuracy at the cost of speed. Default `5` is a good balance for real-time dictation.

> **Note**: `sampling_strategy` is locked in at model load time for `pywhispercpp`. Changing it requires a service restart.
