"""Pinned Parakeet.cpp v0.5.0 runtime/model contract (no native loading at import)."""

import ctypes
import platform
import subprocess
import sys

try:
    from .paths import DATA_DIR
    from .qwen3_asr_runtime import detect_device
except ImportError:
    from paths import DATA_DIR
    from qwen3_asr_runtime import detect_device

RELEASE = 'v0.5.0'
ABI_VERSION = 6
BASE_URL = 'https://github.com/mudler/parakeet.cpp/releases/download/' + RELEASE
RUNTIMES_DIR = DATA_DIR / 'runtime' / 'parakeet-cpp'
MODELS_DIR = DATA_DIR / 'parakeet-cpp' / 'models'
MODEL_ID = 'tdt-0.6b-v3-q8_0'
MODEL = {
    'repo': 'mudler/parakeet-cpp-gguf',
    'revision': '9e421aa66101085dd3dabe9a0c621568ca7c5a3f',
    'filename': 'tdt-0.6b-v3-q8_0.gguf',
    'size': 940663680,
    'sha256': '4d69a4a6683f4f2d952bad794c1357ca6eb628027695b4699c5a9ad4cd07d757',
}
# Source: GitHub release API asset digests; HF pinned LFS metadata.
ASSETS = {
    ('arm64', 'cpu'): ('parakeet-v0.5.0-lib-linux-cpu-arm64.tar.gz', 902807, 'c932e2bc46d763d51e9cb3955fd6e7b0d60291d07bbd1054bcca09e6e3cce8da'),
    ('x64', 'cpu'): ('parakeet-v0.5.0-lib-linux-cpu-x64.tar.gz', 990547, '07cc1095924a134fe5dc7e6e44247fea4eada70ba56a663f73281b96da0624f2'),
    ('arm64', 'vulkan'): ('parakeet-v0.5.0-lib-linux-vulkan-arm64.tar.gz', 14629431, '2a6159b1e6df8748a428862f2bba6b22e1348d37bffdd0ff2e1fa5717271975c'),
    ('x64', 'vulkan'): ('parakeet-v0.5.0-lib-linux-vulkan-x64.tar.gz', 18482077, 'a8e7a5af6ef82088a71b0e34708c31bf6ab9f69624bcd2cff2ad7e8c11e7e494'),
}
ARCHIVE_FILES = ('libparakeet.so', 'parakeet_capi.h', 'LICENSE', 'README.md')


def architecture():
    arch = {'x86_64': 'x64', 'amd64': 'x64', 'aarch64': 'arm64',
            'arm64': 'arm64'}.get(platform.machine().lower())
    if platform.system() != 'Linux' or arch is None:
        raise RuntimeError('Parakeet.cpp requires Linux x64 or ARM64')
    return arch


def library_path(device):
    if device not in ('cpu', 'vulkan'):
        raise ValueError(f'Unsupported Parakeet.cpp device: {device}')
    return RUNTIMES_DIR / RELEASE / architecture() / device / 'libparakeet.so'


def selection_path():
    return RUNTIMES_DIR / RELEASE / architecture() / 'selected-device'


def model_path():
    return MODELS_DIR / MODEL['filename']


def model_installed():
    path = model_path()
    return path.is_file() and path.stat().st_size == MODEL['size']


def bind_library(path):
    """Declare only the ABI-6 surface used here. Owned strings stay pointers."""
    library = ctypes.CDLL(str(path))
    signatures = {
        'abi_version': ([], ctypes.c_int),
        'load': ([ctypes.c_char_p], ctypes.c_void_p),
        'free': ([ctypes.c_void_p], None),
        'transcribe_pcm': ([ctypes.c_void_p, ctypes.POINTER(ctypes.c_float),
                            ctypes.c_int, ctypes.c_int, ctypes.c_int], ctypes.c_void_p),
        'free_string': ([ctypes.c_void_p], None),
        'last_error': ([ctypes.c_void_p], ctypes.c_char_p),
    }
    version = library.parakeet_capi_abi_version
    version.argtypes = []
    version.restype = ctypes.c_int
    if version() != ABI_VERSION:
        raise RuntimeError(f'Parakeet.cpp ABI mismatch (requires {ABI_VERSION})')
    for name, (args, result) in signatures.items():
        function = getattr(library, 'parakeet_capi_' + name)
        function.argtypes = args
        function.restype = result
    return library


def probe_library(path):
    """A broken loader/ABI cannot crash or indefinitely block the installer."""
    code = ("import sys; sys.path.insert(0, sys.argv[1]); "
            "from parakeet_cpp_runtime import bind_library; bind_library(sys.argv[2])")
    from pathlib import Path
    try:
        result = subprocess.run([sys.executable, '-c', code,
                                 str(Path(__file__).resolve().parent), str(path)],
                                capture_output=True, text=True, timeout=20, check=False)
        return result.returncode == 0, (result.stderr or result.stdout).strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)


def resolve_device(config, force_probe=False, probe=True):
    """Pick the runtime device. probe=False trusts the installer's selection:
    it was probed before being written, and the service must not spawn probes."""
    requested = config.get_setting('parakeet_cpp_device', 'auto')
    if requested not in ('auto', 'cpu', 'vulkan'):
        raise ValueError('parakeet_cpp_device must be auto, cpu, or vulkan')
    if requested != 'auto':
        return requested
    if not force_probe:
        try:
            selected = selection_path().read_text().strip()
        except OSError:
            selected = None
        order = [selected] if selected in ('cpu', 'vulkan') else []
        order += [d for d in ('vulkan', 'cpu') if d not in order]
        for device in order:
            path = library_path(device)
            if path.is_file() and (not probe or probe_library(path)[0]):
                return device
    return detect_device()


def is_installed(config):
    try:
        path = library_path(resolve_device(config, probe=False))
        return model_installed() and path.is_file() and probe_library(path)[0]
    except (OSError, ValueError, RuntimeError):
        return False
