"""
Provider registry for remote transcription backends
Defines known providers, their models, and configuration templates
"""

from typing import Dict, List, Optional, Tuple


DEFAULT_MODEL_BACKENDS: Tuple[str, ...] = ('rest-api',)


# Provider registry with known cloud transcription providers.
#
# 'backends' lists which setup pickers offer a model; it defaults to
# DEFAULT_MODEL_BACKENDS when omitted. 'realtime' carries the model's Realtime
# WebSocket capabilities, which drive session payloads and mode guards.
PROVIDERS: Dict[str, Dict] = {
    'openai': {
        'name': 'OpenAI',
        'endpoint': 'https://api.openai.com/v1/audio/transcriptions',
        'websocket_endpoint': 'wss://api.openai.com/v1/realtime',
        'api_key_prefix': 'sk-',
        'api_key_description': 'OpenAI API key (starts with sk-)',
        'models': {
            'gpt-transcribe': {
                'name': 'GPT Transcribe',
                'description': 'Recommended: accurate, fast, and inexpensive',
                'backends': ('realtime-ws',),
                'realtime': {
                    'default_mode': 'transcribe',
                    'transcription_only': True,
                    'manual_commit': True,
                    'language_context': True,
                    'continuous': False,
                },
            },
            'gpt-live-transcribe': {
                'name': 'GPT Live Transcribe',
                'description': 'Live streaming with the best OSD previews; higher cost',
                'backends': ('realtime-ws',),
                'realtime': {
                    'default_mode': 'transcribe',
                    'transcription_only': True,
                    'manual_commit': True,
                    'language_context': True,
                    'continuous': True,
                },
            },
            'gpt-realtime-whisper': {
                'name': 'GPT Realtime Whisper',
                'description': 'Legacy realtime streaming transcription model',
                'backends': ('realtime-ws',),
                'realtime': {
                    'default_mode': 'transcribe',
                    'transcription_only': True,
                    'manual_commit': True,
                    'language_context': False,
                    'continuous': True,
                },
            },
            'gpt-realtime-2.1': {
                'name': 'GPT-Realtime-2.1',
                'description': 'Voice-to-AI: speak and get spoken replies',
                'backends': ('realtime-ws',),
                'realtime': {'default_mode': 'converse'},
            },
            'gpt-realtime-2.1-mini': {
                'name': 'GPT-Realtime-2.1 mini',
                'description': 'Voice-to-AI, cost-efficient',
                'backends': ('realtime-ws',),
                'realtime': {'default_mode': 'converse'},
            },
            'gpt-4o-transcribe': {
                'name': 'GPT-4o Transcribe',
                'description': 'Latest model with best accuracy',
                'body': {'model': 'gpt-4o-transcribe'}
            },
            'gpt-4o-mini-transcribe': {
                'name': 'GPT-4o Mini Transcribe',
                'description': 'Faster, lighter model',
                'body': {'model': 'gpt-4o-mini-transcribe'}
            },
            'gpt-4o-mini-transcribe-2025-12-15': {
                'name': 'GPT-4o Mini Transcribe (2025-12-15)',
                'description': 'Updated version of the faster, lighter transcription model',
                'body': {'model': 'gpt-4o-mini-transcribe-2025-12-15'}
            },
            'gpt-audio-mini-2025-12-15': {
                'name': 'GPT Audio Mini (2025-12-15)',
                'description': 'General purpose audio model',
                'body': {'model': 'gpt-audio-mini-2025-12-15'}
            },
            'whisper-1': {
                'name': 'Whisper 1',
                'description': 'Legacy Whisper model',
                'body': {'model': 'whisper-1'}
            }
        }
    },
    'groq': {
        'name': 'Groq',
        'endpoint': 'https://api.groq.com/openai/v1/audio/transcriptions',
        'api_key_prefix': 'gsk_',
        'api_key_description': 'Groq API key (starts with gsk_)',
        'models': {
            'whisper-large-v3': {
                'name': 'Whisper Large V3',
                'description': 'High accuracy processing',
                'body': {'model': 'whisper-large-v3'}
            },
            'groq-whisper-large-v3-turbo': {
                'name': 'Whisper Large V3 Turbo',
                'description': 'Fastest transcription speed',
                'body': {'model': 'whisper-large-v3-turbo'}
            },
        },
    },
    'regolo': {
        'name': 'Regolo',
        'endpoint': 'https://api.regolo.ai/v1/audio/transcriptions',
        'api_key_prefix': None,
        'api_key_description': 'Regolo API key',
        'models': {
            'faster-whisper-large-v3': {
                'name': 'Faster Whisper Large V3',
                'description': 'High accuracy, zero data retention (GDPR)',
                'body': {'model': 'faster-whisper-large-v3'}
            },
        },
    },
    'cohere': {
        'name': 'Cohere (🇨🇦)',
        'endpoint': 'https://api.cohere.com/v2/audio/transcriptions',
        'api_key_prefix': None,
        'api_key_description': 'Cohere API key (from dashboard.cohere.com)',
        'models': {
            'cohere-transcribe-03-2026': {
                'name': 'Cohere Transcribe',
                'description': '#1 Open ASR Leaderboard — 5.42 avg WER, 14 languages',
                'body': {'model': 'cohere-transcribe-03-2026'},
            },
        },
    },
    'google': {
        'name': 'Google (Gemini)',
        'endpoint': 'https://generativelanguage.googleapis.com/v1beta/audio/transcriptions',
        'websocket_endpoint': 'wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent',
        'api_key_prefix': None,
        'api_key_description': 'Google AI API key (from aistudio.google.com)',
        'models': {
            'gemini-3.1-flash-live-preview': {
                'name': 'Gemini 3.1 Flash Live (Preview)',
                'description': 'Fast, low-latency realtime streaming',
                'backends': ('realtime-ws',),
                'realtime': {'default_mode': 'transcribe'},
            },
            'gemini-2.5-flash-native-audio-preview-12-2025': {
                'name': 'Gemini 2.5 Flash Native Audio (Preview)',
                'description': 'Native audio, sub-second conversational streaming',
                'backends': ('realtime-ws',),
                'realtime': {'default_mode': 'transcribe'},
            },
        }
    },
    'elevenlabs': {
        'name': 'ElevenLabs',
        'endpoint': 'https://api.elevenlabs.io/v1/speech-to-text',
        'websocket_endpoint': 'wss://api.elevenlabs.io/v1/speech-to-text/realtime',
        'api_key_header': 'xi-api-key',
        'api_key_prefix': None,
        'api_key_description': 'ElevenLabs API key',
        'models': {
            'scribe_v2': {
                'name': 'Scribe v2',
                'description': 'Batch transcription, 90+ languages',
                'body': {'model_id': 'scribe_v2'},
                'backends': (),  # reference entry; no setup picker offers it
            },
            'scribe_v2_realtime': {
                'name': 'Scribe v2 Realtime',
                'description': 'Ultra-low latency (~150ms), 90+ languages',
                'backends': ('realtime-ws',),
                'realtime': {'default_mode': 'transcribe'},
            }
        }
    }
}


def get_provider(provider_id: str) -> Optional[Dict]:
    """Get provider configuration by ID"""
    return PROVIDERS.get(provider_id)


def list_providers() -> List[Tuple[str, str, List[str]]]:
    """
    List all available providers with their models.
    
    Returns:
        List of tuples: (provider_id, provider_name, [model_ids])
    """
    result = []
    for provider_id, provider_data in PROVIDERS.items():
        model_ids = list(provider_data['models'].keys())
        result.append((provider_id, provider_data['name'], model_ids))
    return result


def get_provider_models(provider_id: str) -> Optional[Dict[str, Dict]]:
    """Get all models for a provider"""
    provider = get_provider(provider_id)
    if provider:
        return provider.get('models')
    return None


def model_backends(model_data: Dict) -> Tuple[str, ...]:
    """Backends whose setup picker offers this model."""
    return tuple(model_data.get('backends', DEFAULT_MODEL_BACKENDS))


def get_models_for_backend(provider_id: str, backend: str) -> Dict[str, Dict]:
    """Models a provider offers for the given transcription backend."""
    models = get_provider_models(provider_id) or {}
    return {
        model_id: model_data
        for model_id, model_data in models.items()
        if backend in model_backends(model_data)
    }


def get_realtime_capabilities(provider_id: str, model_id: str) -> Dict:
    """Realtime WebSocket capabilities for a model, empty when unknown."""
    models = get_provider_models(provider_id) or {}
    return (models.get(model_id) or {}).get('realtime') or {}


def get_realtime_mode(provider_id: str, model_id: str) -> str:
    """Mode a model expects: 'transcribe' or 'converse'."""
    return get_realtime_capabilities(provider_id, model_id).get(
        'default_mode', 'transcribe'
    )


def get_model_config(provider_id: str, model_id: str) -> Optional[Dict]:
    """Get configuration for a specific provider/model combination"""
    provider = get_provider(provider_id)
    if not provider:
        return None
    
    models = provider.get('models', {})
    model_config = models.get(model_id)
    if not model_config:
        return None
    
    return {
        'endpoint': provider['endpoint'],
        'body': model_config.get('body', {}).copy(),
        'model_name': model_config.get('name', model_id),
        'model_description': model_config.get('description', '')
    }


def validate_api_key(provider_id: str, api_key: str) -> Tuple[bool, Optional[str]]:
    """
    Validate API key format for a provider.
    
    Returns:
        (is_valid, error_message)
    """
    provider = get_provider(provider_id)
    if not provider:
        return False, f"Unknown provider: {provider_id}"
    
    if not api_key:
        return False, "API key is required"

    prefix = provider.get('api_key_prefix')
    if prefix and not api_key.startswith(prefix):
        return False, f"API key should start with '{prefix}'"
    
    if len(api_key) < 10:  # Basic length check
        return False, "API key appears too short"
    
    return True, None
