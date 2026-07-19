"""Authoritative dependency plan specifications and resolution."""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

try:
    from .dependency_manifest import fingerprint, parse_graph
except ImportError:
    from dependency_manifest import fingerprint, parse_graph


PLAN_SPECS = {
    'pywhispercpp': ('requirements-pywhispercpp.txt', ('sounddevice', 'numpy', 'soxr', 'pywhispercpp'), 'pywhispercpp'),
    'rest': ('requirements-rest.txt', ('sounddevice', 'numpy', 'soxr', 'requests'), 'rest'),
    'realtime': ('requirements-realtime.txt', ('sounddevice', 'numpy', 'soxr', 'websocket'), 'realtime'),
    'elevenlabs': ('requirements-realtime-elevenlabs.txt', ('sounddevice', 'numpy', 'soxr', 'elevenlabs'), 'elevenlabs'),
    'cohere': ('requirements-cohere-transcribe.txt', ('sounddevice', 'numpy', 'soxr', 'transformers', 'torch'), 'cohere'),
    'onnx-cpu': ('requirements-onnx-asr.txt', ('sounddevice', 'numpy', 'soxr', 'onnx_asr'), 'onnx'),
    'onnx-gpu': ('requirements-onnx-asr-gpu.txt', ('sounddevice', 'numpy', 'soxr', 'onnx_asr'), 'onnx'),
    'faster-cpu': ('requirements-faster-whisper.txt', ('sounddevice', 'numpy', 'soxr', 'faster_whisper'), 'faster-whisper'),
    'faster-cuda': ('requirements-faster-whisper-cuda.txt', ('sounddevice', 'numpy', 'soxr', 'faster_whisper'), 'faster-whisper'),
}


@dataclass(frozen=True)
class DependencyPlan:
    manifest: Path
    manifests: tuple[Path, ...]
    fingerprint: str
    required_imports: tuple[str, ...]
    family: str
    accelerated_variant: Optional[str] = None


def plan_key(backend: str, provider: Optional[str], variant: Optional[str], error: Callable):
    if backend in ('cpu', 'nvidia', 'amd', 'vulkan', 'pywhispercpp'):
        return 'pywhispercpp'
    if backend in ('rest-api', 'remote'):
        return 'rest'
    if backend == 'realtime-ws':
        return 'elevenlabs' if provider == 'elevenlabs' else 'realtime'
    if backend == 'cohere-transcribe':
        return 'cohere'
    if backend == 'onnx-asr':
        return 'onnx-gpu' if variant in ('gpu', 'cuda') else 'onnx-cpu'
    if backend == 'faster-whisper':
        return 'faster-cuda' if variant in ('gpu', 'cuda') else 'faster-cpu'
    raise error(f"No dependency manifest is defined for backend {backend!r}")


def resolve(root: Path, backend: str, provider: Optional[str], variant: Optional[str], error: Callable):
    filename, imports, family = PLAN_SPECS[plan_key(backend, provider, variant, error)]
    selected = Path(root) / filename
    manifests = parse_graph(selected, error).manifests
    return DependencyPlan(selected.resolve(), manifests, fingerprint(manifests), imports, family, variant)
