"""
Transcription backend classes.

BACKENDS maps canonical backend names to their classes. Names missing here
(including the pywhispercpp hardware variants 'cpu'/'nvidia'/'vulkan') fall
back to the default pywhispercpp backend in WhisperManager. Backends are
registered as they are extracted from whisper_manager.py.
"""

from .base import TranscriptionBackend
from .cohere_backend import CohereBackend
from .faster_whisper_backend import FasterWhisperBackend
from .onnx_asr_backend import OnnxAsrBackend
from .pywhispercpp_backend import PywhispercppBackend
from .realtime_ws_backend import RealtimeWsBackend
from .rest_api_backend import RestApiBackend

BACKENDS = {
    CohereBackend.name: CohereBackend,
    FasterWhisperBackend.name: FasterWhisperBackend,
    OnnxAsrBackend.name: OnnxAsrBackend,
    PywhispercppBackend.name: PywhispercppBackend,
    RealtimeWsBackend.name: RealtimeWsBackend,
    RestApiBackend.name: RestApiBackend,
}

__all__ = [
    'BACKENDS',
    'TranscriptionBackend',
    'CohereBackend',
    'FasterWhisperBackend',
    'OnnxAsrBackend',
    'PywhispercppBackend',
    'RealtimeWsBackend',
    'RestApiBackend',
]
