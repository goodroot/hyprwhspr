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
LOCAL_INSTALL_BACKENDS = ('cpu', 'nvidia', 'amd', 'vulkan', 'onnx-asr', 'faster-whisper', 'cohere-transcribe')

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
}

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
}
