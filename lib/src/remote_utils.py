"""
Utilities for remote transcription backend
"""

import numpy as np
from io import BytesIO
import wave


def numpy_to_wav_bytes(audio_data: np.ndarray, sample_rate: int) -> BytesIO:
    """Convert numpy array to WAV file bytes"""
    # Convert float32 to int16 for WAV format
    audio_int16 = (audio_data * 32767).astype(np.int16)

    # Create WAV file in memory
    wav_buffer = BytesIO()
    with wave.open(wav_buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_int16.tobytes())

    wav_buffer.seek(0)
    return wav_buffer


def build_transcribe_params(
    remote_config: dict,
    audio_data: np.ndarray,
    sample_rate: int,
    global_language: str = None,
    global_prompt: str = None
) -> dict:
    """Build transcription parameters for OpenAI-compatible API

    Args:
        remote_config: Remote backend configuration
        audio_data: Audio as numpy array
        sample_rate: Sample rate of audio
        global_language: Global language setting
        global_prompt: Global whisper prompt setting

    Returns:
        Dict of parameters for OpenAI audio.transcriptions.create()
    """
    # Convert audio to WAV bytes
    wav_bytes = numpy_to_wav_bytes(audio_data, sample_rate)

    # Base parameters
    params = {
        'model': remote_config['model'],
        'file': ('audio.wav', wav_bytes, 'audio/wav'),
        'response_format': 'text'
    }

    # Optional prompt from global config
    if global_prompt:
        params['prompt'] = global_prompt

    # Optional language from global config
    if global_language:
        params['language'] = global_language

    return params


def validate_remote_config(config_manager) -> bool:
    """Validate remote backend configuration"""
    remote_config = config_manager.get_remote_config()

    if not remote_config:
        print("ERROR: 'remote_backend' configuration is required when backend='remote'")
        print("\nAdd to ~/.config/hyprwhspr/config.json:")
        print('  "remote_backend": {')
        print('    "api_url": "http://localhost:8000",')
        print('    "model": "Systran/faster-whisper-base"')
        print('  }')
        return False

    if not remote_config.get('api_url'):
        print("ERROR: remote_backend.api_url is required")
        return False

    if not remote_config.get('model'):
        print("ERROR: remote_backend.model is required")
        return False

    return True
