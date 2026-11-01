# REST API

Use any ASR backend via HTTP API (local or cloud).

## Cohere 🇨🇦

[Sign up at dashboard.cohere.com](https://dashboard.cohere.com/welcome/register) — Canadian-hosted, same as local Apache 2.0 model.

- **Cohere Transcribe** — #1 Open ASR Leaderboard, 5.42 avg WER, 14 languages

> **Note:** Cohere's API requires a `language` parameter. Set `"language": "en"` (or your language code) in your config alongside the backend selection.

## OpenAI

Bring an API key from OpenAI, and choose from:

- **GPT-4o Transcribe** - Latest model with best accuracy
- **GPT-4o Mini Transcribe** - Faster, lighter model
- **GPT-4o Mini Transcribe (2025-12-15)** - Updated version of the faster, lighter transcription model
- **GPT Audio Mini (2025-12-15)** - General purpose audio model
- **Whisper 1** - Legacy Whisper model

## ElevenLabs

Bring an API key from [ElevenLabs](https://elevenlabs.io/app/settings/api-keys) for **Scribe v2** batch transcription (90+ languages).
`rest_api_provider: "elevenlabs"` sends the stored key in the `xi-api-key` header ElevenLabs expects. Requests carrying that header don't follow redirects, so the key never reaches another host.

```jsonc
{
    "transcription_backend": "rest-api",
    "rest_endpoint_url": "https://api.elevenlabs.io/v1/speech-to-text",
    "rest_api_provider": "elevenlabs",
    "rest_body": {
        "model_id": "scribe_v2",
        "language_code": "en",                          // optional; omit to auto-detect
        "keyterms": ["Claude Code", "Hyprland", "useEffect"]  // optional vocabulary biasing
    }
}
```

It reuses the key `hyprwhspr setup` stores when you pick ElevenLabs (for example for Scribe v2 Realtime).
ElevenLabs reads `language_code`, not `language`, so set the language in `rest_body`.
Keyterms add a 20% surcharge, and more than 100 of them bill every request as at least 20 seconds; see [ElevenLabs' limits](https://elevenlabs.io/docs/api-reference/speech-to-text/convert) (under 50 characters and at most 5 words per keyterm).

## Groq

Bring an API key from Groq, and choose from:

- **Whisper Large V3** - High accuracy processing
- **Whisper Large V3 Turbo** - Fastest transcription speed

## Regolo

Bring an API key from [Regolo](https://regolo.ai/), European-hosted with zero data retention (GDPR):

- **Faster Whisper Large V3** - High accuracy, zero data retention (GDPR)

## Custom backend

Connect to any backend, local or cloud, via your own custom configuration:

```jsonc
{
    "transcription_backend": "rest-api",
    "rest_endpoint_url": "https://your-server.example.com/transcribe",
    "rest_fallback_endpoint_urls": [      // optional ordered fallback endpoints
        "https://fallback.example.com/v1/audio/transcriptions"
    ],
    "rest_headers": {                     // optional arbitrary headers
        "authorization": "Bearer your-api-key-here"
    },
    "rest_body": {                        // optional body fields merged with defaults
        "model": "custom-model",
        "keywords": ["one", "two"]        // arrays are sent as repeated form fields
    },
    "rest_api_key": "your-api-key-here",  // equivalent to rest_headers: { authorization: Bearer your-api-key-here }
    "rest_timeout": 30,                   // optional, default: 30; per endpoint
    "rest_audio_format": "wav"            // optional audio format sent to the endpoint: "wav" (default) | "mp3"
}
```

`rest_fallback_endpoint_urls` are mirrors of the primary, tried in order. They get the same key, headers and body.
Failover happens only when an endpoint can't be reached or answers 429 or 5xx.
Behind a keyed `https://` primary, `http://` fallbacks are skipped.
