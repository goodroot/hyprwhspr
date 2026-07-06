"""
Transcription backend classes.

BACKENDS maps canonical backend names to their classes. Names missing here
(including the pywhispercpp hardware variants 'cpu'/'nvidia'/'vulkan') fall
back to the default pywhispercpp backend in WhisperManager. Backends are
registered as they are extracted from whisper_manager.py.
"""

from .base import TranscriptionBackend

BACKENDS = {}

__all__ = ['BACKENDS', 'TranscriptionBackend']
