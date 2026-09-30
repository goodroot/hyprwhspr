"""Check an installed pywhispercpp wheel: Vulkan backend registered, sample transcribed.

Usage: python verify_vulkan_wheel.py <audio.wav> <expected phrase>

Loads the wheel's libggml, lists the compiled-in ggml backends, then
transcribes the sample with the tiny.en model and looks for the phrase.
Set GGML_VK_VISIBLE_DEVICES to include a software device such as lavapipe.
"""

import ctypes
import sys
import sysconfig
from pathlib import Path


def registered_backends():
    import _pywhispercpp  # noqa: F401  (loads the wheel's ggml libraries)

    platlib = Path(sysconfig.get_paths()['platlib'])
    candidates = sorted(platlib.glob('libggml.so*')) + sorted(platlib.glob('*/libggml.so*'))
    if not candidates:
        sys.exit(f'no libggml.so under {platlib}')
    lib = ctypes.CDLL(str(candidates[0]))
    lib.ggml_backend_reg_count.restype = ctypes.c_size_t
    lib.ggml_backend_reg_get.restype = ctypes.c_void_p
    lib.ggml_backend_reg_get.argtypes = [ctypes.c_size_t]
    lib.ggml_backend_reg_name.restype = ctypes.c_char_p
    lib.ggml_backend_reg_name.argtypes = [ctypes.c_void_p]
    return [lib.ggml_backend_reg_name(lib.ggml_backend_reg_get(i)).decode()
            for i in range(lib.ggml_backend_reg_count())]


def main():
    audio, expected = sys.argv[1], sys.argv[2].lower()

    backends = registered_backends()
    print(f'ggml backends: {backends}', flush=True)
    if 'Vulkan' not in backends:
        sys.exit('Vulkan backend is not compiled into this wheel')

    from pywhispercpp.model import Model

    model = Model('tiny.en')
    text = ' '.join(segment.text for segment in model.transcribe(audio)).strip()
    print(f'transcript: {text}', flush=True)
    if expected not in text.lower():
        sys.exit(f'expected {expected!r} in the transcript')


if __name__ == '__main__':
    main()
