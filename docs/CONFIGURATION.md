# Configuration guide

Perform basic setup and configuration via `hyprwhspr setup`.

Or use the CLI, or edit `~/.config/hyprwhspr/config.json` directly.

The config file uses **sparse storage** and will only contain values you've explicitly changed from the defaults. 

This keeps your config clean and means upstream default changes apply automatically on update.

There is also a `$schema` reference for IDE autocompletion and validation:

```jsonc
{
    "$schema": "https://raw.githubusercontent.com/goodroot/hyprwhspr/main/share/config.schema.json",
}
```

To view your overrides or the full resolved config:

```bash
hyprwhspr config show        # Show your overrides only
hyprwhspr config show --all  # Show all settings including defaults
```

## Minimal configuration

Only 2 essential options:

```jsonc
{
    "primary_shortcut": "SUPER+ALT+D",
    "model": "base"
}
```

## Recording modes

### Toggle mode

Toggle hotkey mode (default) - press to start, press again to stop:

```jsonc
{
    "recording_mode": "toggle"
}
```

### Push-to-talk mode

Hold to record, release to stop:

```jsonc
{
    "recording_mode": "push_to_talk"
}
```

### Auto mode

Hybrid tap/hold - automatically detects your intent:

```jsonc
{
    "recording_mode": "auto"
}
```

- **Tap** (< 400ms) - Toggle behavior: tap to start recording, tap again to stop
- **Hold** (>= 400ms) - Push-to-talk behavior: hold to record, release to stop

### Continuous mode (Experimental)

Press to start, speak naturally, and when you pause for a couple seconds the text is automatically transcribed and pasted. 

Press again to stop:

```jsonc
{
    "recording_mode": "continuous",
    "continuous_silence_seconds": 2.0,  // Optional: seconds of silence before auto-paste (default: 2.0)
    "continuous_silence_threshold": 0    // Optional: 0 = auto-calibrate from noise floor (default). Set manually if needed.
}
```

- Recording continues after each auto-paste, so you can keep dictating
- To make it "faster", lower continous silence threshold
- The final press stops recording and pastes any remaining audio
- The silence threshold is auto-calibrated from your mic's noise floor at the start of each session. If detection feels off, set `continuous_silence_threshold` manually (check logs for the auto-calibrated value as a starting point)

### Long-form mode

Extended recording with pause/resume support:

```jsonc
{
    "recording_mode": "long_form",
    "long_form_submit_shortcut": "SUPER+ALT+E",  // Required: no default, must be set
    "long_form_temp_limit_mb": 500,              // Optional: max temp storage (default: 500 MB)
    "long_form_auto_save_interval": 300,         // Optional: auto-save interval in seconds (default: 300 = 5 minutes)
    "use_hypr_bindings": false,                   // Optional: set true to use Hyprland compositor bindings
}
```

- Primary shortcut toggles recording/pause/resume
- Submit shortcut processes all recorded segments and pastes transcription
- Segments are auto-saved periodically to disk for crash recovery
- Old segments are automatically cleaned up when storage limit is reached

## Custom hotkeys

Extensive key support:

```json
{
    "primary_shortcut": "CTRL+SHIFT+SPACE"
}
```

### Supported key types

- **Modifiers**: `ctrl`, `alt`, `shift`, `super` (left) or `rctrl`, `ralt`, `rshift`, `rsuper` (right)
- **Function keys**: `f1` through `f24`
- **Letters**: `a` through `z`
- **Numbers**: `1` through `9`, `0`
- **Arrow keys**: `up`, `down`, `left`, `right`
- **Special keys**: `enter`, `space`, `tab`, `esc`, `backspace`, `delete`, `home`, `end`, `pageup`, `pagedown`
- **Lock keys**: `capslock`, `numlock`, `scrolllock`
- **Media keys**: `mute`, `volumeup`, `volumedown`, `play`, `nextsong`, `previoussong`
- **Numpad**: `kp0` through `kp9`, `kpenter`, `kpplus`, `kpminus`

Or use direct evdev key names for any key not in the alias list:

```json
{
    "primary_shortcut": "SUPER+KEY_COMMA"
}
```

Examples:

- `"SUPER+SHIFT+M"` - Super + Shift + M
- `"CTRL+ALT+F1"` - Ctrl + Alt + F1
- `"F12"` - Just F12 (no modifier)
- `"RCTRL+RSHIFT+ENTER"` - Right Ctrl + Right Shift + Enter

### Secondary shortcut with language

Use a different hotkey for a specific language:

```jsonc
{
    "primary_shortcut": "SUPER+ALT+D",    // Uses default language from config
    "secondary_shortcut": "SUPER+ALT+I",  // Optional: second hotkey
    "secondary_language": "it"          // Language for secondary shortcut
}
```

> **Note**: Works with backends that support language parameters:
> - **REST API**: Works if the endpoint accepts `language` in the request body
> - **Realtime WebSocket**: Fully supported (OpenAI Realtime API)
> - **Local whisper models**: Fully supported (all pywhispercpp models)
> - **Custom REST endpoints**: May not work if the endpoint doesn't accept a language parameter

The primary shortcut continues to use the `language` setting from your config (or auto-detect if set to `null`). 

The secondary shortcut will always use the configured `secondary_language` when pressed.

Configure via CLI:

```bash
hyprwhspr config secondary-shortcut
```

### Cancel shortcut

Useful when you trigger the shortcut by accident or start speaking and want to bail out.

Plays the error sound on cancel so you have clear audio feedback.

```jsonc
{
    "cancel_shortcut": "SUPER+ESCAPE"  // Any key combo (default: null = disabled)
}
```

The cancel shortcut works in all recording modes. 

**In long-form mode it discards all accumulated segments and resets the session to idle.**

You can also cancel without a dedicated shortcut:

```bash
# Via CLI
hyprwhspr record cancel

# Via FIFO directly (useful for Hyprland binds or sxhkd)
echo cancel > ~/.config/hyprwhspr/recording_control
```

### Hyprland native bindings

Use Hyprland's compositor bindings instead of evdev keyboard grabbing.

Sometimes better compatibility with keyboard remappers.

Enable in config (`~/.config/hyprwhspr/config.json`):

```json
{
  "use_hypr_bindings": true,
}
```

Add bindings to `~/.config/hypr/hyprland.conf`:

```bash
# Toggle mode
# Press once to start, press again to stop
bindd = SUPER ALT, D, Speech-to-text, exec, /usr/lib/hyprwhspr/config/hyprland/hyprwhspr-tray.sh record
```

```bash
# Push-to-Talk mode
# Hold key to record, release to stop
bind = SUPER ALT, D, exec, echo "start" > ~/.config/hyprwhspr/recording_control
bindr = SUPER ALT, D, exec, echo "stop" > ~/.config/hyprwhspr/recording_control
```

```bash
# Long-form mode
# Primary shortcut: toggle record/pause/resume
bindd = SUPER ALT, D, Speech-to-text, exec, /usr/lib/hyprwhspr/config/hyprland/hyprwhspr-tray.sh record
# Submit shortcut: submit recording for transcription
bindd = SUPER ALT, E, Speech-to-text-submit, exec, echo "submit" > ~/.config/hyprwhspr/recording_control
```

```bash
# Cancel recording (all modes) - discard audio without transcribing
bind = SUPER, ESCAPE, exec, echo "cancel" > ~/.config/hyprwhspr/recording_control
```

Restart service to lock in changes:

```bash
systemctl --user restart hyprwhspr
```

### Running without keyboard access

With `grab_keys: false` (default), hyprwhspr can start even if you are not in the `input` group. 

The global shortcut will not work.

Control recording via:

* CLI (`hyprwhspr record toggle`, `hyprwhspr record start`, etc.) 
& the `recording_control` file (e.g. bind in Hyprland to `echo start > ~/.config/hyprwhspr/recording_control`)

## Backends

**Quick pick by hardware:**

- **NVIDIA GPU** → Cohere Transcribe is the leading edge · whisper.cpp (`large-v3-turbo`) for speed
- **AMD / Intel GPU** → whisper.cpp (Vulkan)
- **CPU only** → Parakeet or faster-whisper
- **No local setup** → REST API

For up-to-date accuracy rankings across open-source models, see the [Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard).

| Backend | Privacy | GPU | Speed | Languages | Accuracy | Notes |
|---------|---------|-----|-------|-----------|----------|-------|
| Cohere Transcribe | Local | NVIDIA or CPU | Fast | 14 | Best | Gated model, HF token required |
| Parakeet | Local | NVIDIA or CPU | Fast | Multi | Excellent | — |
| faster-whisper | Local | NVIDIA or CPU | Fast | 99 | Very good | — |
| whisper.cpp | Local | NVIDIA, AMD/Intel, CPU | Very fast | 99 | Very good | — |
| REST API | Cloud | — | Varies | Varies | Varies | Cohere, OpenAI, Groq, Regolo |
| Realtime WebSocket | Cloud | — | Real-time | Varies | Varies | OpenAI, ElevenLabs |

---

### Model commands

`hyprwhspr model` commands route automatically to the configured backend:

```bash
hyprwhspr model status            # Check if model is downloaded/cached
hyprwhspr model list              # Show model info for active backend
hyprwhspr model download [model]  # Download or re-download model
```

Models are downloaded automatically during `hyprwhspr setup`. 

Use `model download` to re-download if needed.

---

### Cohere Transcribe

**#1 on the [Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard)**

_5.42 average WER across 9 benchmarks, outperforms Whisper large-v3 (7.44 WER) at 3× the throughput._

[![Cohere Transcribe benchmark results](https://cdn-uploads.huggingface.co/production/uploads/6867ac274d8d690302fd0378/VtvUqMr9ibvv47Wj3gvI3.png)](https://huggingface.co/blog/CohereLabs/cohere-transcribe-03-2026-release)

| Benchmark | Cohere Transcribe | Whisper large-v3 |
|-----------|:-----------------:|:----------------:|
| LibriSpeech clean | **1.25** | 2.7 |
| LibriSpeech other | **2.37** | 5.2 |
| TedLium | **2.49** | 4.2 |
| SPGISpeech | **3.08** | 4.7 |
| VoxPopuli | **5.87** | 9.1 |
| Gigaspeech | **9.33** | 10.3 |
| Earnings22 | **10.84** | 12.7 |
| **Average WER** | **5.42** | 7.44 |

Lower is better. 

Full benchmark details: [Cohere Transcribe release blog](https://huggingface.co/blog/CohereLabs/cohere-transcribe-03-2026-release).

**Supported languages:** English, German, French, Italian, Spanish, Portuguese, Greek, Dutch, Polish, Arabic, Vietnamese, Chinese, Japanese, Korean

**Requirements:** ~4 GB VRAM (bfloat16), or CPU with ~8 GB RAM (float32) — slower on CPU

#### Setup

Cohere Transcribe is a **gated model** on HuggingFace — you must accept the license before downloading.

1. Accept the license agreement at: [huggingface.co/CohereLabs/cohere-transcribe-03-2026](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026)
2. Generate a read token at: [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
3. Run `hyprwhspr setup` and select **[8] Cohere Transcribe** — you will be prompted for your token

The model (~4 GB) is downloaded during setup. Your token is securely stored locally in `~/.config/hyprwhspr/credentials.json` and never shared.

#### Configuration

```jsonc
{
    "transcription_backend": "cohere-transcribe",
    "cohere_transcribe_device": "auto",      // auto | cuda | cpu
    "cohere_transcribe_dtype": "bfloat16",   // bfloat16 (GPU default) | float32 (CPU)
    "cohere_transcribe_compile": false       // torch.compile for faster throughput (adds warmup on first call)
}
```

Model stored in: `~/.cache/huggingface/hub/models--CohereLabs--cohere-transcribe-03-2026/`

---

### Parakeet

Parakeet TDT V3 via [onnx-asr](https://github.com/istupakov/onnx-asr). 

Typically this model requires a large GPU — 

onnx-asr makes it run well on CPU with a very small accuracy trade-off.

**Requirements:** ~1 GB RAM (CPU) or VRAM (GPU)

#### Setup

Run `hyprwhspr setup` and select **[1] Parakeet**. The model (~1 GB) is downloaded during setup.

#### Configuration

```jsonc
{
    "transcription_backend": "onnx-asr",
    "onnx_asr_model": "nemo-parakeet-tdt-0.6b-v3",  // default
    "onnx_asr_quantization": "int8",                  // int8 (default) | fp32
    "onnx_asr_use_vad": true                          // Silero VAD (default: true)
}
```

Model stored in: `~/.cache/huggingface/hub/`

---

### faster-whisper

Local Whisper via [faster-whisper](https://github.com/SYSTRAN/faster-whisper). 

Run `hyprwhspr setup` and select **[4] faster-whisper** to install.

**Best for:**

CPU users wanting faster inference than whisper.cpp, or NVIDIA GPU users where VRAM is constrained. 

INT8 quantization runs `large-v3-turbo` in ~3.1 GB vs ~6 GB for float16.

AMD/Intel GPU users should use Parakeet or whisper.cpp instead — CTranslate2 does not support Vulkan or ROCm.

Built-in Silero VAD strips silence before inference — the most effective mitigation for Whisper's hallucination loops on longer recordings.

```jsonc
{
    "transcription_backend": "faster-whisper",
    "faster_whisper_model": "large-v3-turbo",   // CUDA; use "base" or "small" for CPU
    "faster_whisper_device": "auto",             // auto | cuda | cpu
    "faster_whisper_compute_type": "auto",       // auto → int8 on cuda, float32 on cpu; set "int8" on cpu for speed
    "faster_whisper_vad_filter": true            // Silero VAD (default: true)
}
```

#### Available models

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

---

### whisper.cpp

Local Whisper via [pywhispercpp](https://github.com/abdeladim-s/pywhispercpp). 

Run `hyprwhspr setup` and select **[2] Whisper CPU**, **[3] Whisper NVIDIA**, or **[5] Whisper AMD/Intel (Vulkan)**.

**Best for:**

Modern NVIDIA cards or discrete AMD/Intel use (via Vulkan). 

Extremely fast on GPU with `large-v3` or `large-v3-turbo`.

#### Available models

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
    "model": "small.en"  // .en = English-only; omit suffix for multilingual
}
```

#### Language detection

English only speakers use `.en` models which are smaller.

For multi-language detection, ensure you select a model which does not say `.en`:

```jsonc
{
    "language": null // null = auto-detect (default), or specify language code
}
```

Language options:

- **`null`** (default) - Auto-detect language from audio
- **`"en"`** - English transcription
- **`"nl"`** - Dutch transcription
- **`"fr"`** - French transcription
- **`"de"`** - German transcription
- **`"es"`** - Spanish transcription
- **`etc.`** - Any supported language code

#### Whisper prompt

Customize transcription behavior:

```jsonc
{
    "whisper_prompt": "Transcribe with proper capitalization, including sentence beginnings, proper nouns, titles, and standard English capitalization rules."
}
```

The prompt influences how Whisper interprets and transcribes your audio, eg:

- `"Transcribe as technical documentation with proper capitalization, acronyms and technical terminology."`

- `"Transcribe as casual conversation with natural speech patterns."`

- `"Transcribe as an ornery pirate on the cusp of scurvy."`

#### Translation

Translate non-English speech into English:

```jsonc
{
    "task": "translate",
    "language": "it"  // optional: set source language, or null to auto-detect
}
```

- **`"transcribe"`** (default) - Output in the source language
- **`"translate"`** - Translate speech into English

> **Note**: Supported by `faster-whisper` and `pywhispercpp` backends. `language` and `task` are independent — setting a non-English language does not imply translation.

#### Language-specific prompts

Set a per-language prompt using `whisper_prompt_{lang}`:

```jsonc
{
    "whisper_prompt": "Transcribe with proper capitalization.",
    "whisper_prompt_de": "Transkribiere auf Deutsch. Verwende Schweizer Rechtschreibung: kein ß, immer ss."
}
```

- Falls back to `whisper_prompt` if no language-specific prompt is configured
- Only applies when a language is active (via `language`, `secondary_language`, or `--lang`)

#### Decoding strategy

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

---

### REST API

Use any ASR backend via HTTP API (local or cloud).

#### Cohere 🇨🇦

[Sign up at dashboard.cohere.com](https://dashboard.cohere.com/welcome/register) — Canadian-hosted, same as local Apache 2.0 model.

- **Cohere Transcribe** — #1 Open ASR Leaderboard, 5.42 avg WER, 14 languages

> **Note:** Cohere's API requires a `language` parameter. Set `"language": "en"` (or your language code) in your config alongside the backend selection.

#### OpenAI

Bring an API key from OpenAI, and choose from:

- **GPT-4o Transcribe** - Latest model with best accuracy
- **GPT-4o Mini Transcribe** - Faster, lighter model
- **GPT-4o Mini Transcribe (2025-12-15)** - Updated version of the faster, lighter transcription model
- **GPT Audio Mini (2025-12-15)** - General purpose audio model
- **Whisper 1** - Legacy Whisper model

#### Groq

Bring an API key from Groq, and choose from:

- **Whisper Large V3** - High accuracy processing
- **Whisper Large V3 Turbo** - Fastest transcription speed

#### Regolo

Bring an API key from [Regolo](https://regolo.ai/), European-hosted with zero data retention (GDPR):

- **Faster Whisper Large V3** - High accuracy, zero data retention (GDPR)

#### Custom backend

Connect to any backend, local or cloud, via your own custom configuration:

```jsonc
{
    "transcription_backend": "rest-api",
    "rest_endpoint_url": "https://your-server.example.com/transcribe",
    "rest_headers": {                     // optional arbitrary headers
        "authorization": "Bearer your-api-key-here"
    },
    "rest_body": {                        // optional body fields merged with defaults
        "model": "custom-model"
    },
    "rest_api_key": "your-api-key-here",  // equivalent to rest_headers: { authorization: Bearer your-api-key-here }
    "rest_timeout": 30                    // optional, default: 30
}
```

---

### Realtime WebSocket

Low-latency streaming transcription.

> Experimental!

#### OpenAI Realtime

Two modes available:

- **transcribe** (default) - Pure speech-to-text, more expensive than HTTP
- **converse** - Voice-to-AI: speak and get AI responses

```jsonc
{
    "transcription_backend": "realtime-ws",
    "websocket_provider": "openai",
    "websocket_model": "gpt-realtime-mini-2025-12-15",
    "realtime_mode": "transcribe",       // "transcribe" or "converse"
    "realtime_timeout": 30,              // Advanced: seconds to wait after stop for final transcript
    "realtime_buffer_max_seconds": 5     // Advanced: max unsent audio backlog (seconds) before dropping old chunks
}
```

#### ElevenLabs Scribe v2

Ultra-low latency (~150ms) streaming transcription. 

Bring an API key from [ElevenLabs](https://elevenlabs.io/) with speech-to-text capabilities enabled.

Uses native 16kHz audio (no resampling) and auto-reconnects on connection drops.

- **transcribe** (default) - speech-to-text

```jsonc
{
    "transcription_backend": "realtime-ws",
    "websocket_provider": "elevenlabs",
    "websocket_model": "scribe_v2_realtime",
    "realtime_timeout": 30,              // Advanced: seconds to wait after stop for final transcript
    "realtime_buffer_max_seconds": 5     // Advanced: max unsent audio backlog (seconds) before dropping old chunks
}
```

## Audio and visual feedback

### Themed visualizer

Visual feedback that will auto-match Omarchy themes.

> Highly recommended!

```json
{
  "mic_osd_enabled": true,
}
```

### Audio feedback

Optional sound notifications:

```jsonc
{
    "audio_feedback": true,            // Enable audio feedback (default: false)
    "audio_volume": 0.5,               // General audio volume fallback (0.1 to 1.0, default: 0.5)
    "start_sound_volume": 1.0,         // Start recording sound volume (0.1 to 1.0, default: 1.0)
    "stop_sound_volume": 1.0,          // Stop recording sound volume (0.1 to 1.0, default: 1.0)
    "error_sound_volume": 0.5,         // Error sound volume (0.1 to 1.0, default: 0.5)
    "start_sound_path": "custom-start.ogg",  // Custom start sound (relative to assets)
    "stop_sound_path": "custom-stop.ogg",    // Custom stop sound (relative to assets)
    "error_sound_path": "custom-error.ogg"  // Custom error sound (relative to assets)
}
```

Default sounds included:

- **Start recording**: `ping-up.ogg` (ascending tone)
- **Stop recording**: `ping-down.ogg` (descending tone)
- **Error/blank audio**: `ping-error.ogg` (double-beep)

Custom sounds:

- **Supported formats**: `.ogg`, `.wav`, `.mp3`
- **Fallback**: Uses defaults if custom files don't exist

### Audio stream keepalive

By default hyprwhspr opens the microphone only while recording. 

On some hardware (certain USB mics on raw ALSA), the first recording after an idle period fails with a `paTimedOut` error because the audio device suspends between uses.

If you see this, enable the keepalive stream:

```json
{
  "keepalive_stream": true
}
```

This holds a silent input stream open in the background so the device stays warm. 

**Leave this off unless you need it** — an open input stream triggers the microphone-in-use indicator on most desktops (GNOME, KDE, Ubuntu, etc.), making it appear as though hyprwhspr is always listening. It's not!

### Audio ducking

Quiet system volume on record:

```jsonc
{
  "audio_ducking": true,
  "audio_ducking_percent": 70
}
```

- `audio_ducking: true` Set true to enable audio ducking 
- `audio_ducking_percent: 70` -  How much to reduce volume BY (70 = reduces to 30% of original)

## Text processing

### Word overrides

Customize transcriptions:

```json
{
    "word_overrides": {
        "hyper whisper": "hyprwhspr",
        "um": ""
    }
}
```

Use empty string `""` to delete words entirely.

Single-character overrides match anywhere in a word (not just at word boundaries):

```json
{
    "word_overrides": {
        "ß": "ss"
    }
}
```

- `"Straße"` → `"Strasse"`, `"Fuß"` → `"Fuss"`, etc.
- Multi-character overrides use whole-word matching only

### Filler word filtering

Remove common filler words automatically:

```jsonc
{
    "filter_filler_words": true,  // Enable automatic filler word removal (default: false)
    "filler_words": ["uh", "um", "er", "ah", "eh", "hmm", "hm", "mm", "mhm"]  // Customize list
}
```

When enabled, filler words are removed before text injection. Customize the list to match your speech patterns.

### Symbol replacements

Automatically converts spoken words to symbols / punctuation.

Toggle this behavior in `~/.config/hyprwhspr/config.json`:

```jsonc
{
    "symbol_replacements": true  // default: true (set false to disable speech-to-symbol replacements)
}
```

**Punctuation:**

- "period" → "."
- "comma" → ","
- "question mark" → "?"
- "exclamation mark" → "!"
- "colon" → ":"
- "semicolon" → ";"

**Symbols:**

- "at symbol" → "@"
- "hash" → "#"
- "plus" → "+"
- "equals" → "="
- "dash" → "-"
- "underscore" → "_"

**Brackets:**

- "open paren" → "("
- "close paren" → ")"
- "open bracket" → "["
- "close bracket" → "]"
- "open brace" → "{"
- "close brace" → "}"

**Special commands:**

- "new line" → new line
- "tab" → tab character

## Paste and clipboard behavior

### Paste mode

hyprwhspr auto-detects the correct paste shortcut based on the focused window:

- **Terminals** (Ghostty, Kitty, WezTerm, Alacritty, foot, etc.) → Ctrl+Shift+V
- **Everything else** (editors, browsers, chat apps) → Ctrl+V

No configuration needed for most setups. Override with `paste_mode` if your app needs something different:

```jsonc
{
    "paste_mode": "ctrl_shift",  // "ctrl_shift" | "ctrl" | "super" | "alt"
}
```

Options:

- **`"ctrl_shift"`** — Sends Ctrl+Shift+V. Standard terminal paste.
- **`"ctrl"`** — Sends Ctrl+V. Standard GUI paste.
- **`"super"`** — Sends Super+V.
- **`"alt"`** — Sends Alt+V.

### Non-QWERTY layouts

`ydotool` sends physical Linux keycodes, so `Ctrl+KEY_V` might not be `Ctrl+v` on your layout (bepo, dvorak, etc.).

Quick way to fix it on Wayland (no arithmetic):

- Run `wev` in terminal
- Press the key that types `v` on your layout
- Copy the printed `keycode` into `paste_keycode_wev`

```jsonc
{
    "paste_keycode_wev": 55 // `wev` keycode for the key that types 'v' on your layout
}
```

Advanced (if you already know the Linux evdev keycode): set `paste_keycode` directly.

### Auto-submit

Automatically press Enter after pasting.

> aka Dictation YOLO

```jsonc
{
    "auto_submit": true   // Send Enter key after paste (default: false)
}
```

Useful for chat applications, search boxes, or any input where you want to submit immediately after dictation.

... Be careful!

### Clipboard behavior

hyprwhspr saves your clipboard before injection and restores it automatically afterward — your clipboard contents are never permanently overwritten by dictated text.

The paste hotkey is sent via `wtype` (Wayland virtual-keyboard protocol), which works correctly in all applications including Kitty-protocol terminals (Ghostty, Kitty, WezTerm). `ydotool` is used as a fallback when `wtype` is not installed.

## Integrations

### Waybar

Add dynamic tray icon to your `~/.config/waybar/config`:

```jsonc
{
    "custom/hyprwhspr": {
        "exec": "/usr/lib/hyprwhspr/config/hyprland/hyprwhspr-tray.sh status",
        "interval": 2,
        "return-type": "json",
        "exec-on-event": true,
        "format": "{}",
        "on-click": "/usr/lib/hyprwhspr/config/hyprland/hyprwhspr-tray.sh toggle",
        "on-click-right": "/usr/lib/hyprwhspr/config/hyprland/hyprwhspr-tray.sh restart",
        "tooltip": true
    }
}
```

Add CSS styling to your `~/.config/waybar/style.css`:

```css
@import "/usr/lib/hyprwhspr/config/waybar/hyprwhspr-style.css";
```

Waybar icon click interactions:

- **Left-click**: Start/stop recording (auto-starts service if needed)
- **Right-click**: Restart Hyprwhspr service

### Keyboard device selection

If you have multiple input tools (e.g., Espanso, keyd, kmonad), specify which to use:

```json
{
  "selected_device_name": "USB Keyboard"  // Match by device name (recommended)
}
```

Or by device path:

```jsonc
{
  "selected_device_path": "/dev/input/event3"  // Match by exact path
}
```

Device name takes priority if both are set. Use `hyprwhspr keyboard list` to see available devices.

### External hotkey systems

Control recording via CLI (Espanso, KDE, GNOME, etc.) - set these terminal commands however is appropriate:

```bash
# Start recording
hyprwhspr record start

# Start recording with specific language
hyprwhspr record start --lang it    # Italian
hyprwhspr record start --lang de    # German
hyprwhspr record start --lang es    # Spanish

# Stop recording (transcribes and pastes)
hyprwhspr record stop

# Cancel recording (discards audio, no transcription)
hyprwhspr record cancel

# Toggle recording on/off
hyprwhspr record toggle
hyprwhspr record toggle --lang it   # Toggle with language override

# Check current status
hyprwhspr record status
```

The `--lang` parameter overrides the default language for that recording session. 

This is useful for multilingual users who want different hotkeys for different languages.

Then bind these commands to your preferred hotkeys in KDE, GNOME, sxhkd, or any other hotkey system:

```bash
# Example: KDE custom shortcuts
# English: hyprwhspr record toggle
# Italian: hyprwhspr record start --lang it
# Cancel:  hyprwhspr record cancel

# Example: Hyprland config
bind = SUPER ALT, D, exec, hyprwhspr record toggle
bind = SUPER ALT, I, exec, hyprwhspr record start --lang it
bind = SUPER, ESCAPE, exec, hyprwhspr record cancel
```

### Mute detection

Lets you know when you're disconnected.

Note, mute detection can cause conflicts with Bluetooth microphones. 

To disable it, add the following to your `~/.config/hyprwhspr/config.json`:

```jsonc
{
  "mute_detection": false
}
```

## GPU resource management

Free GPU VRAM without stopping the service - useful before running a game, or other GPU-intensive workload.

The service keeps running with all keyboard shortcuts active. 

Recording is blocked while the model is unloaded, with a desktop notification on attempt.

```bash
# Unload model from GPU memory (service stays alive, shortcuts still active)
hyprwhspr model unload

# Reload model back into memory when ready to dictate again
hyprwhspr model reload
```

Only applies to local-model backends (Cohere Transcribe, `pywhispercpp`, `faster-whisper`, `onnx-asr`). 

No-op for `rest-api` and `realtime-ws` (those hold no local GPU memory).

The Waybar tray shows a `󰒲` sleep icon while the model is unloaded.

### Hyprland keybindings

For quick access, bind unload and reload to keys in `~/.config/hypr/hyprland.conf`:

```bash
# Free GPU before starting a local LLM
bindd = SUPER ALT, U, Unload Whisper model, exec, hyprwhspr model unload

# Reclaim dictation when done
bindd = SUPER ALT, L, Reload Whisper model, exec, hyprwhspr model reload
```

## Troubleshooting

### Reset installation

If you're having persistent issues, completely reset hyprwhspr:

```bash
hyprwhspr uninstall
hyprwhspr setup
```

### Common issues

#### Something is weird

Restart the service - right click on the waybar icon if you use it, or:

```bash
systemctl --user restart hyprwhspr.service
```

Still weird? Proceed.

#### I heard the sound but don't see text

On resume/restart, often the microphone "loses connection" and requires reseating.

This is a Linux quirk and not resolvable by hyprwhspr.

Reseat your microphone as prompted if it fails under these conditions. 

Also, within sound options, ensure that the **right microphone** is indeed set. 

#### Hotkey not working

```bash
# Check service status for hyprwhspr
systemctl --user status hyprwhspr.service

# Check logs
journalctl --user -u hyprwhspr.service -f
```

```bash
# Check service status for ydotool
systemctl --user status ydotool.service

# Check logs
journalctl --user -u ydotool.service -f
```

If you are using the hyprland input method, do you have shortcuts?

#### Service starts but doesn't work until restarted

`hyprwhspr` must start within an active graphical session.

If the service appears active but hotkeys/transcription don't work until you manually restart, your session environment may not be set up correctly.

Check your session:

```bash
# Verify graphical-session.target is active
systemctl --user is-active graphical-session.target

# Verify Wayland env is available to systemd services
systemctl --user show-environment | grep WAYLAND_DISPLAY
```

If `WAYLAND_DISPLAY` is missing, add to `~/.config/hypr/hyprland.conf`:

```bash
# Export session environment to systemd user services
exec-once = dbus-update-activation-environment --systemd WAYLAND_DISPLAY XDG_CURRENT_DESKTOP HYPRLAND_INSTANCE_SIGNATURE
```

**Hyprland:**

If `graphical-session.target` is inactive, you likely need a session manager to activate it. 

The recommended approach is to launch Hyprland via [uwsm](https://github.com/Vladimir-csp/uwsm) (it activates `graphical-session.target` and exports the session environment to systemd).

If you *aren't* using a session manager and your system allows it, you can try starting it manually:

```bash
exec-once = systemctl --user start graphical-session.target
```

> **Note:** Some distros set `graphical-session.target` with `RefuseManualStart=yes`, in which case the manual start will fail and you should use a session manager like `uwsm` instead.

Then restart Hyprland or log out and back in.

Run `hyprwhspr validate` to confirm the session is configured correctly.

#### Permission denied

```bash
# Fix uinput permissions
hyprwhspr setup

# Log out and back in
```

#### No audio input

Is your mic _actually_ available?

```bash
# Check audio devices
pactl list short sources

# Restart PipeWire
systemctl --user restart pipewire
```

#### Microphone indicator shows on while idle

If your desktop (GNOME, Ubuntu, etc.) shows the microphone as active whenever the hyprwhspr service is running, you likely have `keepalive_stream` enabled. Disable it:

```json
{
  "keepalive_stream": false
}
```

This is the default. If you previously enabled it to fix `paTimedOut` errors, see [Audio stream keepalive](#audio-stream-keepalive) for the trade-off.

#### Audio feedback not working

```bash
# Check if audio feedback is enabled in config
cat ~/.config/hyprwhspr/config.json | grep audio_feedback

# Verify sound files exist (script install uses ~/hyprwhspr/share/assets/)
ls -la /usr/lib/hyprwhspr/share/assets/   # AUR install
ls -la ~/hyprwhspr/share/assets/           # Script install

# Check which audio player is available
which ffplay paplay pw-play aplay
```

**Important:** 

The default sounds are OGG format. `aplay` (ALSA) only supports WAV files and will produce white noise if used with OGG. 

Install `ffplay` (ffmpeg) or ensure `paplay` (pulseaudio-utils) or `pw-play` (pipewire) is available for OGG playback.

#### Model not found

```bash
# Check installed models (routes to active backend)
hyprwhspr model status

# Download a model
hyprwhspr model download base

# Verify model in config
cat ~/.config/hyprwhspr/config.json | grep model
```

#### Stuck recording state

```bash
# Check service health and auto-recover
/usr/lib/hyprwhspr/config/hyprland/hyprwhspr-tray.sh health

# Manual restart if needed
systemctl --user restart hyprwhspr.service

# Check service status
systemctl --user status hyprwhspr.service
```

#### This sucks

Doh! We tried.

Wipe the slate clean and remove everything:

```
hyprwhspr uninstall
yay -Rs hyprwhspr
```

Or better yet - create an issue and help us improve.
