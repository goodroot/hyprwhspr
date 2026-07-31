"""Capabilities for OpenAI transcription models used over Realtime WebSocket."""


OPENAI_TRANSCRIPTION_ONLY_MODELS = frozenset({
    'gpt-transcribe',
    'gpt-live-transcribe',
    'gpt-realtime-whisper',
})

OPENAI_MANUAL_COMMIT_MODELS = frozenset({
    'gpt-transcribe',
    'gpt-live-transcribe',
    'gpt-realtime-whisper',
})

OPENAI_LANGUAGE_CONTEXT_MODELS = frozenset({
    'gpt-transcribe',
    'gpt-live-transcribe',
})

# These models can emit transcript text while audio is still arriving. Keep
# delay configuration and continuous waveform previews limited to this group.
OPENAI_CONTINUOUS_TRANSCRIPTION_MODELS = frozenset({
    'gpt-live-transcribe',
    'gpt-realtime-whisper',
})
