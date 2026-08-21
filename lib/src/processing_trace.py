"""Diagnostic description of daemon-side transcript processing."""

import math

try:
    from .backend_utils import normalize_backend
    from .openai_realtime_models import uses_manual_commit
    from .text_injector import preprocess_text
except ImportError:
    from backend_utils import normalize_backend
    from openai_realtime_models import uses_manual_commit
    from text_injector import preprocess_text


def _setting(config, name, default=None):
    return config.get_setting(name, default)


def _finite_float_setting(config, name, default):
    try:
        value = float(_setting(config, name, default))
    except (TypeError, ValueError, OverflowError):
        return float(default)
    return value if math.isfinite(value) else float(default)


def _onnx_duration_gate(config):
    try:
        value = float(_setting(config, 'onnx_asr_vad_min_duration', 30))
    except (TypeError, ValueError, OverflowError):
        value = 30.0
    return max(0.0, value)


def _backend_model(config, backend):
    if backend == 'onnx-asr':
        return _setting(config, 'onnx_asr_model', None)
    if backend == 'faster-whisper':
        return _setting(config, 'faster_whisper_model', None)
    if backend == 'realtime-ws':
        return _setting(config, 'websocket_model', None)
    if backend == 'cohere-transcribe':
        return 'CohereLabs/cohere-transcribe-03-2026'
    return _setting(config, 'model', None)


def classify_vad_mode(config):
    """Classify how speech filtering or provider turn boundaries are made."""
    backend = normalize_backend(_setting(config, 'transcription_backend', 'pywhispercpp'))
    if backend in ('pywhispercpp', 'cpu', 'nvidia', 'vulkan'):
        return 'silero_filter' if _setting(config, 'pywhispercpp_use_vad', False) else 'none'
    if backend == 'faster-whisper':
        return 'silero_filter' if _setting(config, 'faster_whisper_vad_filter', True) else 'none'
    if backend == 'onnx-asr':
        return 'silero_segmented' if _setting(config, 'onnx_asr_use_vad', True) else 'none'
    if backend == 'rest-api':
        return 'provider_managed'
    if backend != 'realtime-ws':
        return 'none'

    provider = str(_setting(config, 'websocket_provider', '') or '').lower()
    model = _setting(config, 'websocket_model', None)
    realtime_mode = _setting(config, 'realtime_mode', 'transcribe')
    if provider == 'elevenlabs':
        return 'provider_managed'
    if provider == 'google':
        return 'server_vad'
    if realtime_mode == 'converse' or uses_manual_commit(model):
        return 'manual_commit'
    return 'server_vad'


def classify_boundary_mode(config):
    """Describe client-side recording boundaries separately from backend VAD."""
    recording_mode = _setting(config, 'recording_mode', 'toggle')
    if recording_mode == 'continuous':
        return 'continuous_silence'
    silence_timeout = _finite_float_setting(config, 'silence_timeout', 0)
    if recording_mode in ('toggle', 'auto') and silence_timeout > 0:
        return 'silence_auto_stop'
    return 'manual_stop'


def build_processing_trace(raw, config):
    """Return JSON-serializable processing output and active daemon settings."""
    backend = normalize_backend(_setting(config, 'transcription_backend', 'pywhispercpp'))
    trace = {
        'raw': raw,
        'preprocessed': preprocess_text(raw, config),
        'backend': backend,
        'model': _backend_model(config, backend),
        'recording_mode': _setting(config, 'recording_mode', 'toggle'),
        'symbol_replacements': bool(_setting(config, 'symbol_replacements', True)),
        'hook_present': bool(str(_setting(config, 'post_transcription_hook', '') or '').strip()),
        'silence_timeout': _finite_float_setting(config, 'silence_timeout', 0),
        'continuous_silence_seconds': _finite_float_setting(config, 'continuous_silence_seconds', 2.0),
        'continuous_silence_threshold': _finite_float_setting(config, 'continuous_silence_threshold', 0),
        'vad_mode': classify_vad_mode(config),
        'boundary_mode': classify_boundary_mode(config),
    }
    if backend == 'onnx-asr' and _setting(config, 'onnx_asr_use_vad', True):
        trace['onnx_vad_min_duration'] = _onnx_duration_gate(config)
    return trace
