"""Backends expose one capability contract; callers never branch on backend names."""
import ast
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib" / "src"))

from backends import BACKENDS, TranscriptionBackend  # noqa: E402
from backends.pywhispercpp_backend import PywhispercppBackend  # noqa: E402
from whisper_manager import WhisperManager  # noqa: E402

SURFACE = ('get_streaming_callback', 'apply_partial_callback', 'last_connect_failure',
           'discard_audio', 'update_language', 'close')


class FakeConfig:
    def __init__(self, backend):
        self.backend = backend

    def get_setting(self, key, default=None):
        return self.backend if key == 'transcription_backend' else default


class BackendContractTests(unittest.TestCase):
    def test_every_backend_implements_the_contract(self):
        for name, cls in BACKENDS.items():
            self.assertTrue(issubclass(cls, TranscriptionBackend), name)
            self.assertEqual(cls.name, name)
            for attr in SURFACE:
                self.assertTrue(hasattr(cls, attr), f'{name}.{attr}')

    def test_capability_flags(self):
        streaming = {name for name, cls in BACKENDS.items() if cls.streams_audio}
        background = {name for name, cls in BACKENDS.items() if cls.loads_in_background}
        self.assertEqual(streaming, {'realtime-ws'})
        # faster-whisper stays synchronous: its init mutates process env and dlopens CUDA globally.
        self.assertEqual(background, {'cohere-transcribe', 'qwen3-asr', 'onnx-asr'})

    def test_base_streaming_surface_is_harmless(self):
        backend = TranscriptionBackend(WhisperManager(config_manager=FakeConfig('rest-api')))
        self.assertIsNone(backend.get_streaming_callback())
        self.assertIsNone(backend.last_connect_failure)
        for call in (lambda: backend.apply_partial_callback(print), backend.discard_audio,
                     lambda: backend.update_language('en'), backend.close):
            self.assertIsNone(call())

    def test_manager_reads_background_loading_from_the_configured_class(self):
        for configured, expected in (('onnx-asr', True), ('faster-whisper', False),
                                     ('realtime-ws', False), ('cpu', False), ('unknown', False)):
            manager = WhisperManager(config_manager=FakeConfig(configured))
            self.assertEqual(manager.configured_backend_loads_in_background(), expected, configured)
        self.assertFalse(PywhispercppBackend.loads_in_background)

    def test_background_loaders_never_write_the_environment_in_initialize(self):
        # initialize() runs on a worker thread for these; os.environ writes race other threads.
        import inspect
        import textwrap
        for name, cls in BACKENDS.items():
            if not cls.loads_in_background:
                continue
            tree = ast.parse(textwrap.dedent(inspect.getsource(cls.initialize)))
            for node in ast.walk(tree):
                target = node.targets[0] if isinstance(node, ast.Assign) else None
                call = node.func if isinstance(node, ast.Call) else None
                writes = (isinstance(target, ast.Subscript) and ast.unparse(target.value) == 'os.environ') or \
                         (isinstance(call, ast.Attribute) and ast.unparse(call) in
                          ('os.environ.setdefault', 'os.environ.update', 'os.putenv'))
                self.assertFalse(writes, f'{name}.initialize writes os.environ: {ast.unparse(node)}')

    def test_manager_has_no_backend_name_checks_on_objects(self):
        # Capabilities replace `backend.name == ...` branches on live backend objects.
        tree = ast.parse((ROOT / 'lib' / 'src' / 'whisper_manager.py').read_text(encoding='utf-8'))
        offenders = [node.lineno for node in ast.walk(tree)
                     if isinstance(node, ast.Compare) and isinstance(node.left, ast.Attribute)
                     and node.left.attr == 'name']
        self.assertEqual(offenders, [])


if __name__ == '__main__':
    unittest.main()
