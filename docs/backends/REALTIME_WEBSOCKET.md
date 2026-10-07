# Realtime WebSocket

Persistent WebSocket streaming. `realtime_mode` selects `transcribe` (speech-to-text, the default) or `converse`
(voice-to-AI), but converse support varies by provider:

| Provider | `transcribe` | `converse` |
| --- | --- | --- |
| OpenAI | Yes | Yes, on a `gpt-realtime-*` model |
| Google Gemini | Yes | Yes |
| ElevenLabs | Yes | No — hyprwhspr ignores `realtime_mode` |
| [Self-hosted](#self-hosted-streaming) | Yes | OpenAI-compatible servers only |

## OpenAI Realtime

Dedicated transcription models, all requiring `realtime_mode: "transcribe"`:

| Model | Transcript arrives | Notes |
| --- | --- | --- |
| `gpt-transcribe` | After you stop recording | Recommended: accurate, fast, inexpensive |
| `gpt-live-transcribe` | Live, while you speak | Best OSD previews; higher cost |
| `gpt-realtime-whisper` | Live, while you speak | Legacy |

None of the three support `converse` — that needs a `gpt-realtime-*` model, which `hyprwhspr setup` also offers.
All three disable server-side VAD and commit the turn when recording stops. `realtime_transcription_delay`
(partial-result latency vs. accuracy) and the continuous waveform OSD preview apply only to the two live models.

```jsonc
{
    "transcription_backend": "realtime-ws",
    "websocket_provider": "openai",
    "websocket_model": "gpt-transcribe",
    "realtime_mode": "transcribe",       // "transcribe" or "converse"
    "realtime_timeout": 30,              // Advanced: seconds to wait after stop for final transcript
    "realtime_buffer_max_seconds": 5     // Advanced: max unsent audio backlog (seconds) before dropping old chunks
}
```

Proxies that reject transcription-only sessions, such as CLIProxyAPI with ChatGPT/Codex OAuth, need a full
Realtime session. The URL picks the session model; `websocket_model` picks the transcriber:

```jsonc
{
    "transcription_backend": "realtime-ws",
    "websocket_provider": "openai",
    "websocket_url": "wss://your-proxy.example.com/v1/realtime?model=gpt-realtime",
    "websocket_model": "gpt-live-transcribe",
    "realtime_mode": "transcribe",
    "realtime_transcription_session_type": "realtime"   // default: "transcription"
}
```

Dictation is unchanged; no assistant reply is requested.

In `converse` mode, `realtime_conversation_history` controls what the provider retains between turns:

- `"turn"` (default) - deletes each completed turn while reusing the WebSocket, so every turn starts with empty
  context.
- `"session"` - keeps prior turns for conversational context. The provider re-sends and bills that audio every
  turn, so cost climbs as the session grows. Choose this only if you want a multi-turn assistant.

For a stateless voice-to-AI workflow:

```jsonc
{
    "transcription_backend": "realtime-ws",
    "websocket_provider": "openai",
    "websocket_model": "gpt-realtime-2.1",
    "realtime_mode": "converse",
    "realtime_conversation_history": "turn"
}
```

All OpenAI Realtime models stream 24 kHz PCM audio. For GPT Transcribe and GPT Live Transcribe, hyprwhspr sends the
scalar `language` setting as OpenAI's single-entry `languages` hint and resolves the prompt through
`whisper_prompt_<language>` → `whisper_prompt` → `whisper_prompt_en` when the language is English or
unset → omitted. For GPT Realtime Whisper it sends a plain scalar
`language` and no prompt.

## Google Gemini

Realtime streaming transcription via Google's Gemini Live API.

Bring an API key from [Google AI Studio](https://aistudio.google.com/).

Uses native 16kHz audio (no resampling) and server-side VAD.

- **transcribe** (default) - speech-to-text via gemini's live transcript events
- **converse** - voice-to-AI: speak and get AI responses

```jsonc
{
    "transcription_backend": "realtime-ws",
    "websocket_provider": "google",
    "websocket_model": "gemini-3.1-flash-live-preview",  // or gemini-2.5-flash-native-audio-preview-12-2025
    "realtime_mode": "transcribe",           // "transcribe" or "converse"
    "realtime_timeout": 30,                  // Advanced: seconds to wait after stop for final transcript
    "realtime_buffer_max_seconds": 5         // Advanced: max unsent audio backlog (seconds) before dropping old chunks
}
```

## ElevenLabs Scribe v2

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

## Self-hosted streaming

Run your own server; no API key needed. `hyprwhspr setup` → **Realtime WS** lists NeMo-Speech.cpp, Phonon, and any
OpenAI Realtime-compatible server.

**NeMo-Speech.cpp** — English, [live typing](#live-typing). Get the
[server](https://github.com/NVIDIA/NeMo-Speech.cpp/releases) and
[`nemotron-speech-streaming-en-0.6b.q8_0.gguf`](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b/resolve/main/nemotron-speech-streaming-en-0.6b.q8_0.gguf)
(700MB, ~2.7GB VRAM). Serve 1.12s chunks; the 160ms default is noticeably less accurate:

```bash
nemo-speech serve --asr-model nemotron-speech-streaming-en-0.6b.q8_0.gguf \
    --asr.streaming.rnnt_right_context 13
```

```jsonc
{
    "transcription_backend": "realtime-ws",
    "websocket_provider": "custom",
    "websocket_model": "nemotron-speech-streaming-en-0.6b",
    "websocket_url": "ws://127.0.0.1:8080/v1/realtime",
    "websocket_sample_rate": 16000,
    "websocket_session_format": "flat",
    "websocket_live_text": "append_only"
}
```

**[Phonon](https://github.com/fermionresearch/phonon)** — English, live OSD preview. One stream per recording:
toggle, push-to-talk or auto.

```jsonc
{
    "transcription_backend": "realtime-ws",
    "websocket_provider": "custom",
    "websocket_protocol": "phonon",
    "websocket_model": "phonon-2",
    "websocket_url": "wss://asr.example.com/v1/audio/stream"
}
```

**Anything else** — describe how it speaks:

| Setting | Values | Default |
| --- | --- | --- |
| `websocket_protocol` | `openai-realtime`, `phonon` | `openai-realtime` |
| `websocket_sample_rate` | 8000–96000 | protocol's own |
| `websocket_session_format` | `nested`, `flat` (older servers) | `nested` |
| `websocket_live_text` | `none`, `revisable`, `append_only` | `none` |

`revisable` shows live text in the OSD. `append_only` means sent words never change, which allows live typing.

## Live typing

> Experimental.

Words land as you speak. Needs a model whose words never change once sent (`append_only`): today, NeMo.

```jsonc
{
    "realtime_live_typing": true
}
```

- Typed words stay; at stop, the rest arrives with its punctuation.
- Enter, the trailing space and the clipboard restore wait for stop.
- Cancel keeps what's typed.
- Off with a `post_transcription_hook`, `record capture`, continuous or long-form. `hyprwhspr config validate` says why.
