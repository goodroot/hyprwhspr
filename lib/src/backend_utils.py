"""Backend utilities and constants for hyprwhspr"""

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


# Backend display names for CLI output
# Single source of truth for user-facing backend names
BACKEND_DISPLAY_NAMES = {
    'pywhispercpp': 'Local (pywhispercpp)',
    'onnx-asr': 'Parakeet TDT V3 (onnx-asr, CPU/GPU)',
    'rest-api': 'REST API',
    'realtime-ws': 'Realtime WebSocket (experimental)',
    'cpu': 'Whisper CPU (pywhispercpp)',
    'nvidia': 'Whisper NVIDIA (CUDA)',
    'amd': 'Whisper AMD/Intel (Vulkan)',
    'vulkan': 'Whisper AMD/Intel (Vulkan)',
    'faster-whisper': 'faster-whisper (CTranslate2, CPU/CUDA)',
}
