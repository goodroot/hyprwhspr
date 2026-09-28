"""Shared ONNX selection and production loader; optional imports stay lazy."""

DEFAULT_MODEL = 'nemo-parakeet-tdt-0.6b-v3'
# prepare() exit code for a cache miss or corrupt file that a download can fix.
NEEDS_DOWNLOAD = 3


def resolve_model(settings, explicit=None):
    # Orukeet ships only int8 files and ignores quantization, so the user's
    # Parakeet setting is carried through untouched.
    model = explicit or settings.get('onnx_asr_model', DEFAULT_MODEL)
    return model, settings.get('onnx_asr_quantization', 'int8'), settings.get('onnx_asr_use_vad', True)


def load_model(model, quantization='int8', use_vad=True, *, offline=False, repair=False):
    import onnx_asr
    if model == 'orukeet':
        try:
            from .orukeet import download_model
        except ImportError:
            from orukeet import download_model
        directory = download_model(offline=offline, repair=repair)
        loaded = onnx_asr.load_model('nemo-conformer-tdt', path=directory, quantization='int8')
    elif quantization:
        loaded = onnx_asr.load_model(model, quantization=quantization)
    else:
        loaded = onnx_asr.load_model(model)
    vad_model = loaded.with_vad(onnx_asr.load_vad('silero')) if use_vad else None
    return loaded, vad_model


def prepare(selection, offline):
    """Setup's readiness check, run in the backend venv; returns an exit code."""
    try:
        from .orukeet import ChecksumError
    except ImportError:
        from orukeet import ChecksumError
    try:
        load_model(*selection, offline=offline, repair=not offline)
    except (FileNotFoundError, ChecksumError):
        # Offline misses (Hub and Orukeet alike) and corrupt files are fixable by
        # downloading; anything else is a real error and propagates.
        if offline:
            return NEEDS_DOWNLOAD
        raise
    return 0
