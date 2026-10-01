"""
Wire protocols the realtime-ws backend can speak.

A protocol is how bytes and events travel, not who hosts the server. Built-in
cloud providers name theirs in provider_registry ('websocket_protocol');
everything else, self-hosted servers included, is `custom` plus an explicit
`websocket_protocol` setting, defaulting to OpenAI Realtime.
"""

import importlib
from dataclasses import dataclass
from typing import Optional, Tuple

try:
    from .provider_registry import get_provider, get_realtime_capabilities
except ImportError:
    from provider_registry import get_provider, get_realtime_capabilities


DEFAULT_PROTOCOL = 'openai-realtime'

# What a model's live text promises while audio is still arriving:
#   none         nothing worth showing before the turn ends
#   revisable    streamed text the model may still rewrite (OSD preview only)
#   append_only  emitted words never change (safe to type while speaking)
LIVE_TEXT_MODES = ('none', 'revisable', 'append_only')


@dataclass(frozen=True)
class RealtimeProtocol:
    id: str
    client: str                    # 'module:Class', imported on first use
    deps: str                      # PLAN_SPECS key
    modes: Tuple[str, ...] = ('transcribe',)
    vad: Optional[str] = None      # trace label; None: derived from mode and model
    uses_instructions: bool = False
    derives_url: bool = False      # build the URL from the provider endpoint plus intent/model query


PROTOCOLS = {
    'openai-realtime': RealtimeProtocol(
        id='openai-realtime',
        client='realtime_client:RealtimeClient',
        deps='realtime',
        modes=('transcribe', 'converse'),
        uses_instructions=True,
        derives_url=True,
    ),
    'gemini-live': RealtimeProtocol(
        id='gemini-live',
        client='gemini_realtime_client:GeminiRealtimeClient',
        deps='realtime',
        modes=('transcribe', 'converse'),
        vad='server_vad',
        uses_instructions=True,
    ),
    'elevenlabs': RealtimeProtocol(
        id='elevenlabs',
        client='elevenlabs_realtime_client:ElevenLabsRealtimeClient',
        deps='elevenlabs',
        vad='provider_managed',
    ),
}


def protocol_for_provider(provider_id: Optional[str]) -> RealtimeProtocol:
    """Protocol a built-in provider speaks; custom and unknown get the default."""
    provider = get_provider(provider_id) if provider_id else None
    return PROTOCOLS[(provider or {}).get('websocket_protocol', DEFAULT_PROTOCOL)]


def resolve_protocol(provider_id: Optional[str], configured: Optional[str] = None) -> RealtimeProtocol:
    """The configured protocol for custom endpoints, else the provider's own.

    Raises ValueError for an unknown protocol name.
    """
    if provider_id == 'custom' and configured:
        protocol = PROTOCOLS.get(str(configured).strip().lower())
        if protocol is None:
            raise ValueError(f'Unknown websocket_protocol: {configured!r}')
        return protocol
    return protocol_for_provider(provider_id)


def live_text_mode(provider_id: Optional[str], model_id: Optional[str],
                   configured: Optional[str] = None) -> str:
    """Live-text promise: configured for custom endpoints, else from model caps."""
    if provider_id == 'custom':
        mode = str(configured or 'none').strip().lower()
        return mode if mode in LIVE_TEXT_MODES else 'none'
    caps = get_realtime_capabilities(provider_id, model_id)
    if caps.get('live_text') in LIVE_TEXT_MODES:
        return caps['live_text']
    return 'revisable' if caps.get('continuous') else 'none'


def load_client_class(protocol: RealtimeProtocol):
    """Import the protocol's client class, package-relative when possible."""
    module_name, class_name = protocol.client.split(':')
    if __package__:
        module = importlib.import_module(f'.{module_name}', __package__)
    else:
        module = importlib.import_module(module_name)
    return getattr(module, class_name)
