"""
Transcription backend classes.

BACKENDS maps canonical backend names to their classes. Names missing here
(including the pywhispercpp hardware variants 'cpu'/'nvidia'/'vulkan') fall
back to the default pywhispercpp backend in WhisperManager. Adding a new
backend means adding a module here and registering its class in BACKENDS.
"""

from .base import TranscriptionBackend
from .cohere_backend import CohereBackend
from .faster_whisper_backend import FasterWhisperBackend
from .onnx_asr_backend import OnnxAsrBackend
from .pywhispercpp_backend import PywhispercppBackend
from .qwen3_asr_backend import Qwen3AsrBackend
from .realtime_ws_backend import RealtimeWsBackend
from .rest_api_backend import RestApiBackend

BACKENDS = {
    CohereBackend.name: CohereBackend,
    FasterWhisperBackend.name: FasterWhisperBackend,
    OnnxAsrBackend.name: OnnxAsrBackend,
    PywhispercppBackend.name: PywhispercppBackend,
    Qwen3AsrBackend.name: Qwen3AsrBackend,
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
    'Qwen3AsrBackend',
    'RealtimeWsBackend',
    'RestApiBackend',
]
