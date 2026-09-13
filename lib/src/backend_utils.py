"""Backend utilities and constants for hyprwhspr"""

import re


_HARDWARE_DEVICE_TYPES = ('INTEGRATED_GPU', 'DISCRETE_GPU', 'VIRTUAL_GPU')
_DEVICE_TYPE_LINE_RE = re.compile(r'deviceType\s*=\s*PHYSICAL_DEVICE_TYPE_(\w+)')


def vulkaninfo_has_hardware_gpu(summary: str) -> bool:
    """Return True if vulkaninfo --summary lists at least one non-software device.

    vulkaninfo --summary prints one block per Vulkan device, each with a
    `deviceType = PHYSICAL_DEVICE_TYPE_...` line. A hardware GPU has deviceType
    INTEGRATED_GPU, DISCRETE_GPU, or VIRTUAL_GPU; llvmpipe and other software
    renderers report PHYSICAL_DEVICE_TYPE_CPU (or _OTHER). On Mesa the llvmpipe
    fallback ICD is always present alongside the real driver, so the previous
    substring check (`'llvmpipe' in output`) was a false negative on every Mesa
    system with a real GPU.
    """
    for match in _DEVICE_TYPE_LINE_RE.finditer(summary):
        if match.group(1) in _HARDWARE_DEVICE_TYPES:
            return True
    return False


def normalize_backend(backend: str) -> str:
    """Normalize backend name for backward compatibility.

    Maps old backend names to new names:
    - 'local' -> 'pywhispercpp'
    - 'remote' -> 'rest-api'
    - 'amd' -> 'vulkan' (AMD/Intel now uses Vulkan instead of ROCm)

    Args:
        backend: Backend name (may use old naming)

    Returns:
        Normalized backend name
    """
    if backend == 'local':
        return 'pywhispercpp'
    elif backend == 'remote':
        return 'rest-api'
    elif backend == 'amd':
        return 'vulkan'
    return backend


# Backends that install packages into the local venv (vs. remote API backends).
# Single source of truth — used by setup, install validation, and repair.
# 'amd' is accepted pre-normalization (normalize_backend maps it to 'vulkan').
LOCAL_INSTALL_BACKENDS = ('cpu', 'nvidia', 'amd', 'vulkan', 'onnx-asr', 'faster-whisper', 'cohere-transcribe', 'qwen3-asr')

# Python module each local backend needs importable from the venv.
# Single source of truth — used to verify installs and detect missing backends.
BACKEND_IMPORT_MODULES = {
    'pywhispercpp': 'pywhispercpp',
    'cpu': 'pywhispercpp',
    'nvidia': 'pywhispercpp',
    'amd': 'pywhispercpp',
    'vulkan': 'pywhispercpp',
    'onnx-asr': 'onnx_asr',
    'faster-whisper': 'faster_whisper',
    'cohere-transcribe': 'transformers',
    # A bundled executable, not an importable Python package; callers must
    # validate its runtime manifest rather than attempt an import.
    'qwen3-asr': None,
}

# Languages CohereLabs/cohere-transcribe accepts. The model has no language
# detection, so an unset or unsupported language is a user-visible problem;
# kept here so the CLI can name them without loading the model.
COHERE_LANGUAGES = ('ar', 'de', 'el', 'en', 'es', 'fr', 'it', 'ja', 'ko', 'nl', 'pl', 'pt', 'vi', 'zh')

# English names for the languages Qwen3-ASR recognises, keyed by the ISO codes
# used everywhere else in hyprwhspr. llama.cpp forwards a transcription's
# `language` field by appending it to a natural-language prompt
# ("Transcribe audio to text (language: %s)"), and Qwen's own vocabulary is
# language *names* — it emits "language Japanese<asr_text>…" — so an ISO code
# is a poor hint. Codes absent here fall through unmapped rather than guessing.
LANGUAGE_NAMES = {
    'ar': 'Arabic', 'cs': 'Czech', 'da': 'Danish', 'de': 'German', 'el': 'Greek',
    'en': 'English', 'es': 'Spanish', 'fa': 'Persian', 'fi': 'Finnish',
    'fil': 'Filipino', 'fr': 'French', 'hi': 'Hindi', 'hu': 'Hungarian',
    'id': 'Indonesian', 'it': 'Italian', 'ja': 'Japanese', 'ko': 'Korean',
    'mk': 'Macedonian', 'ms': 'Malay', 'nl': 'Dutch', 'pl': 'Polish',
    'pt': 'Portuguese', 'ro': 'Romanian', 'ru': 'Russian', 'sv': 'Swedish',
    'th': 'Thai', 'tr': 'Turkish', 'vi': 'Vietnamese', 'yue': 'Cantonese',
    'zh': 'Chinese',
}


def language_name(language):
    """Map an ISO code to the English language name, or pass it through."""
    if not language:
        return language
    return LANGUAGE_NAMES.get(language.strip().lower().replace('_', '-'), language)

# Backend display names for CLI output
# Single source of truth for user-facing backend names
BACKEND_DISPLAY_NAMES = {
    'pywhispercpp': 'Local (pywhispercpp)',
    'onnx-asr': 'Parakeet TDT V3 (onnx-asr, CPU/GPU)',
    'cohere-transcribe': 'Cohere Transcribe 2B (transformers, CPU/GPU)',
    'rest-api': 'REST API',
    'realtime-ws': 'Realtime WebSocket',
    'cpu': 'Whisper CPU (pywhispercpp)',
    'nvidia': 'Whisper NVIDIA (CUDA)',
    'amd': 'Whisper AMD/Intel (Vulkan)',
    'vulkan': 'Whisper AMD/Intel (Vulkan)',
    'faster-whisper': 'faster-whisper (CTranslate2, CPU/CUDA)',
    'qwen3-asr': 'Qwen3-ASR (llama.cpp, experimental)',
}
