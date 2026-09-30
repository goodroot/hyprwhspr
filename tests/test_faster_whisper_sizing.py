"""faster-whisper sizes CPU loads for dictation, not the CTranslate2 defaults."""
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib" / "src"))

from backends import faster_whisper_backend  # noqa: E402
from backends.faster_whisper_backend import FasterWhisperBackend  # noqa: E402
from tests.text_injector_helpers import ConfigStub  # noqa: E402


def _manager(settings):
    return types.SimpleNamespace(config=ConfigStub(settings), ready=False, current_model=None)


class FasterWhisperSizingTests(unittest.TestCase):
    def _load(self, settings, force_cpu=None):
        whisper_model = mock.Mock()
        fake = types.SimpleNamespace(WhisperModel=whisper_model)
        backend = FasterWhisperBackend(_manager(settings))
        with mock.patch.dict(sys.modules, {'faster_whisper': fake}), \
                mock.patch.object(faster_whisper_backend, 'log'):
            if force_cpu is None:
                self.assertTrue(backend.initialize())
            else:
                self.assertTrue(backend.reinitialize(force_cpu=force_cpu))
        return whisper_model.call_args.kwargs

    def test_auto_is_int8_on_cpu_and_threads_are_honored(self):
        # float32 held ~3-4x the weights in RAM; CTranslate2 ignored `threads`.
        kwargs = self._load({'faster_whisper_device': 'cpu', 'threads': 6})
        self.assertEqual((kwargs['compute_type'], kwargs['cpu_threads']), ('int8', 6))

    def test_explicit_compute_type_is_respected(self):
        kwargs = self._load({'faster_whisper_device': 'cpu', 'faster_whisper_compute_type': 'float32'})
        self.assertEqual(kwargs['compute_type'], 'float32')

    def test_cpu_fallback_reinit_is_int8(self):
        kwargs = self._load({'threads': 3}, force_cpu=True)
        self.assertEqual((kwargs['device'], kwargs['compute_type'], kwargs['cpu_threads']), ('cpu', 'int8', 3))


if __name__ == '__main__':
    unittest.main()
