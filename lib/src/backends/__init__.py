"""
Transcription backend classes.

BACKENDS maps canonical backend names to their classes. Names missing here
(including the pywhispercpp hardware variants 'cpu'/'nvidia'/'vulkan') fall
back to the default pywhispercpp backend in WhisperManager. Backends are
registered as they are extracted from whisper_manager.py.
"""

from .base import TranscriptionBackend
from .rest_api_backend import RestApiBackend

BACKENDS = {
    RestApiBackend.name: RestApiBackend,
}

__all__ = ['BACKENDS', 'TranscriptionBackend', 'RestApiBackend']
