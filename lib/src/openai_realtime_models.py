"""Capabilities for OpenAI transcription models used over Realtime WebSocket.

Capabilities live on the model entries in provider_registry so a model is
described in one place. Unknown models (custom endpoints) report no
capabilities, which keeps them on server VAD and the singular `language` field.
"""

try:
    from .provider_registry import get_realtime_capabilities
except ImportError:
    from provider_registry import get_realtime_capabilities


def _capability(model_id, name):
    if not model_id:
        return False
    return bool(get_realtime_capabilities('openai', model_id).get(name))


def is_transcription_only(model_id) -> bool:
    """The model rejects realtime_mode="converse"."""
    return _capability(model_id, 'transcription_only')


def uses_manual_commit(model_id) -> bool:
    """The session disables server VAD and commits the turn on stop."""
    return _capability(model_id, 'manual_commit')


def uses_language_context(model_id) -> bool:
    """The model takes a `languages` array and an optional `prompt`."""
    return _capability(model_id, 'language_context')


def is_continuous(model_id) -> bool:
    """The model emits transcript deltas while audio is still arriving."""
    return _capability(model_id, 'continuous')
