"""pywhispercpp loads the installed model file, never a name it would download."""
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib" / "src"))

from backends import pywhispercpp_backend  # noqa: E402
from backends.pywhispercpp_backend import PywhispercppBackend  # noqa: E402
from tests.text_injector_helpers import ConfigStub  # noqa: E402


def _manager(settings):
    return types.SimpleNamespace(config=ConfigStub(settings), ready=False, current_model=None)


class PywhispercppModelFileTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.models = Path(tmp.name)
        patcher = mock.patch.object(pywhispercpp_backend, 'PYWHISPERCPP_MODELS_DIR', self.models)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.settings = {'language': 'en'}
        self.backend = PywhispercppBackend(_manager(self.settings))

    def _model_arg(self, model_name):
        model = mock.Mock()
        fake = types.ModuleType('pywhispercpp.model')
        fake.Model = model
        with mock.patch.dict(sys.modules, {'pywhispercpp': types.ModuleType('pywhispercpp'),
                                           'pywhispercpp.model': fake}):
            with mock.patch.object(pywhispercpp_backend, 'log'):
                self.backend._create_pywhisper_model(model_name, 4)
        return model.call_args.kwargs['model']

    def test_english_only_install_loads_that_file_for_english(self):
        # A bare name makes pywhispercpp download the multilingual model it can't find.
        english = self.models / 'ggml-base.en.bin'
        english.write_bytes(b'x')
        self.assertEqual(self._model_arg('base'), str(english))

    def test_other_languages_never_get_the_english_only_file(self):
        # .en models force English: German or auto-detect still needs the download.
        (self.models / 'ggml-base.en.bin').write_bytes(b'x')
        for language in ('de', None):
            with self.subTest(language=language):
                self.settings['language'] = language
                self.assertEqual(self._model_arg('base'), 'base')

    def test_truncated_multilingual_file_does_not_hide_a_valid_english_one(self):
        (self.models / 'ggml-base.bin').write_bytes(b'x')
        with open(self.models / 'ggml-base.en.bin', 'wb') as f:
            f.truncate(10000001)
        self.assertTrue(self.backend._validate_model_file('base'))

    def test_multilingual_file_wins_when_both_exist(self):
        (self.models / 'ggml-base.en.bin').write_bytes(b'x')
        multilingual = self.models / 'ggml-base.bin'
        multilingual.write_bytes(b'x')
        self.assertEqual(self._model_arg('base'), str(multilingual))

    def test_missing_model_is_refused_before_loading(self):
        with mock.patch.object(pywhispercpp_backend, 'log'), \
                mock.patch.object(self.backend, '_create_pywhisper_model') as create:
            self.assertFalse(self.backend.initialize())
        create.assert_not_called()


if __name__ == '__main__':
    unittest.main()
