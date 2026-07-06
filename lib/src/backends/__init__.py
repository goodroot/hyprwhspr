"""
Transcription backend classes.

BACKENDS maps canonical backend names to their classes. Names missing here
(including the pywhispercpp hardware variants 'cpu'/'nvidia'/'vulkan') fall
back to the default pywhispercpp backend in WhisperManager. Backends are
registered as they are extracted from whisper_manager.py.
"""

from .base import TranscriptionBackend
from .onnx_asr_backend import OnnxAsrBackend
from .rest_api_backend import RestApiBackend

BACKENDS = {
    OnnxAsrBackend.name: OnnxAsrBackend,
    RestApiBackend.name: RestApiBackend,
}

__all__ = ['BACKENDS', 'TranscriptionBackend', 'OnnxAsrBackend', 'RestApiBackend']
