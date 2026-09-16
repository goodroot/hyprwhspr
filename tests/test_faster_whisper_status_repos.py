import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'lib'))
sys.path.insert(0, str(ROOT / 'lib' / 'src'))

from cli import models


class FasterWhisperModelStatusTests(unittest.TestCase):
    def _cache_dir(self, *repos):
        """Build a fake ~/.cache/huggingface/hub with the given model repos."""
        cell = {}

        class _FakeHome(type(Path())):
            @staticmethod
            def home():
                return cell['home']

        with tempfile.TemporaryDirectory() as tmp:
            cell['home'] = Path(tmp)
            hub = cell['home'] / '.cache' / 'huggingface' / 'hub'
            hub.mkdir(parents=True)
            for repo, marker in repos:
                model_dir = hub / f'models--{repo}'
                model_dir.mkdir(parents=True)
                (model_dir / marker).write_bytes(b'x' * 1024)
            with mock.patch.object(models, 'Path', _FakeHome):
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    models.faster_whisper_model_status()
            return output.getvalue()

    def test_lists_legacy_systran_models(self):
        rendered = self._cache_dir(
            ('Systran--faster-whisper-large-v3', 'model.bin'))
        self.assertIn('Installed faster-whisper models:', rendered)
        self.assertIn('- large-v3', rendered)

    def test_lists_mobiuslabsgmbh_turbo_models(self):
        rendered = self._cache_dir(
            ('mobiuslabsgmbh--faster-whisper-large-v3-turbo', 'model.bin'))
        self.assertIn('- large-v3-turbo', rendered)

    def test_lists_models_from_both_namespaces(self):
        rendered = self._cache_dir(
            ('Systran--faster-whisper-large-v3', 'model.bin'),
            ('mobiuslabsgmbh--faster-whisper-large-v3-turbo', 'model.bin'))
        self.assertIn('- large-v3', rendered)
        self.assertIn('- large-v3-turbo', rendered)

    def test_empty_cache_warns(self):
        rendered = self._cache_dir()
        self.assertIn('No faster-whisper models found', rendered)


if __name__ == '__main__':
    unittest.main()