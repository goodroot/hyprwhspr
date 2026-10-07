"""Backend utilities and constants for hyprwhspr"""

import gc
import re
import sys
from urllib.parse import urlsplit, urlunsplit


def _is_valid_endpoint_url(url, schemes):
    """Validate a supported scheme and authority without contacting the host."""
    if (not isinstance(url, str) or not url.startswith(schemes)
            or any(char.isspace() or ord(char) < 32 or ord(char) == 127 for char in url)):
        return False
    try:
        parsed = urlsplit(url)
        return bool(parsed.hostname) and parsed.port != 0
    except ValueError:
        return False


def is_valid_websocket_url(url):
    """A ws:// or wss:// URL with a host, checked without contacting it."""
    return _is_valid_endpoint_url(url, ('ws://', 'wss://'))


def is_valid_http_url(url):
    """An http:// or https:// URL with a host, checked without contacting it."""
    return _is_valid_endpoint_url(url, ('http://', 'https://'))


def endpoint_key(url):
    """Identity of a valid HTTP(S) URL for de-duplication.

    Scheme and host compare case-insensitively and a trailing slash is
    ignored, so "https://Host/x" and "https://host/x/" are one endpoint.
    """
    parsed = urlsplit(url)
    return (parsed.scheme.lower(), (parsed.hostname or '').lower(), parsed.port,
            parsed.path.rstrip('/'), parsed.query)


def redact_url(url):
    """Return a log-safe endpoint URL.

    Keeps the scheme, host, port, and path so the target stays identifiable
    while dropping embedded userinfo (which may carry credentials) and the
    query/fragment (which may carry tokens).
    """
    if not isinstance(url, str) or not url:
        return url
    try:
        parts = urlsplit(url)
        hostname = parts.hostname
    except ValueError:
        # Unparseable (e.g. a malformed IPv6 literal): keep nothing that
        # could carry a secret.
        return '<redacted-endpoint>'
    # A log-safe endpoint needs a real, non-empty authority. Without one the
    # "path" may actually be userinfo/query data (e.g. "https:/user:pw@host/x"
    # or a scheme-less "user:pw@host/x"), so echo nothing at all.
    if not parts.netloc or not hostname:
        return '<redacted-endpoint>'
    netloc = parts.netloc
    if '@' in netloc:
        # rsplit keeps the real authority even if userinfo contains '@'.
        netloc = netloc.rsplit('@', 1)[1]
    return urlunsplit((parts.scheme, netloc, parts.path, '', ''))


def release_memory() -> bool:
    """Free a just-dropped model now, before anything new is allocated.

    Collects cycles so native destructors run, then returns torch's cached
    CUDA blocks to the driver. Never imports torch: that alone costs ~0.5 GB.
    Returns True if the CUDA cache was cleared.
    """
    gc.collect()
    torch = sys.modules.get('torch')
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
        return True
    return False


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
LOCAL_INSTALL_BACKENDS = ('cpu', 'nvidia', 'amd', 'vulkan', 'parakeet-cpp', 'onnx-asr', 'faster-whisper', 'cohere-transcribe', 'qwen3-asr')

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
    'parakeet-cpp': None,
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

# One line per local backend, shown before setup installs it. Keep the facts
# in step with the model table in docs/backends/OVERVIEW.md.
BACKEND_SUMMARIES = {
    'onnx-asr': 'Parakeet via ONNX · CPU or NVIDIA · ~1 GB',
    'parakeet-cpp': 'Parakeet.cpp · CPU or Vulkan · ~0.9 GB · experimental',
    'faster-whisper': 'Whisper via faster-whisper · CPU or NVIDIA',
    'cpu': 'Whisper via whisper.cpp · CPU',
    'nvidia': 'Whisper via whisper.cpp · NVIDIA (CUDA) · may compile from source',
    'vulkan': 'Whisper via whisper.cpp · Vulkan (AMD/Intel) · may compile from source',
    'cohere-transcribe': 'Cohere Transcribe · 4 GB VRAM, or 8 GB RAM on CPU · Hugging Face token required',
    'qwen3-asr': 'Qwen3-ASR via llama.cpp · CPU or Vulkan · ~2.4 GB · experimental',
}

# Backend display names for CLI output
# Single source of truth for user-facing backend names
BACKEND_DISPLAY_NAMES = {
    'parakeet-cpp': 'Parakeet.cpp (experimental, CPU/Vulkan)',
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
