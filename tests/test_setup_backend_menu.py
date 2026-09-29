"""Setup backend menu: one entry per model family, one confirmation per step."""
import contextlib
import io
import types
import unittest
from unittest.mock import patch

from tests.test_setup_command_scope import setup
import backend_installer


class WhisperEngineTests(unittest.TestCase):
    def test_installed_whisper_engine_is_kept(self):
        for current, expected in (('cpu', 'cpu'), ('nvidia', 'nvidia'), ('vulkan', 'vulkan'),
                                  ('pywhispercpp', 'cpu'), ('faster-whisper', 'faster-whisper')):
            with patch.object(backend_installer, 'detect_gpu_type') as detect:
                self.assertEqual(setup._resolve_whisper_engine(current), expected)
            detect.assert_not_called()

    def test_fresh_install_follows_the_gpu(self):
        for gpu, expected in (('nvidia', 'faster-whisper'), ('cpu', 'faster-whisper'),
                              ('vulkan', 'vulkan')):
            for current in (None, 'onnx-asr', 'rest-api'):
                with patch.object(backend_installer, 'detect_gpu_type', return_value=gpu):
                    self.assertEqual(setup._resolve_whisper_engine(current), expected)


class BackendMenuTests(unittest.TestCase):
    def _select(self, current, answers, settings=None, confirm=True):
        """Drive _prompt_backend_selection; returns (result, prompts asked)."""
        answers = iter(answers)
        asked = []

        def ask(prompt, choices=None, default=None, **_):
            asked.append((prompt, default))
            answer = next(answers)
            return default if answer is None else answer

        def confirm_ask(prompt, **_):
            asked.append((prompt, None))
            return confirm
        settings = dict(settings or {})
        if current:
            settings.setdefault('transcription_backend', current)
        with patch.object(setup, '_detect_current_backend', return_value=current), \
             patch.object(setup, 'Prompt', types.SimpleNamespace(ask=ask)), \
             patch.object(setup, 'Confirm', types.SimpleNamespace(ask=confirm_ask)), \
             patch.object(backend_installer, 'detect_gpu_type', return_value='nvidia'), \
             contextlib.redirect_stdout(io.StringIO()):
            return setup._prompt_backend_selection(settings), asked

    def test_every_backend_defaults_to_its_menu_entry(self):
        for backend, entry in (('onnx-asr', '1'), ('parakeet-cpp', '1'), ('faster-whisper', '2'),
                               ('cpu', '2'), ('nvidia', '2'), ('vulkan', '2'), ('amd', '2'),
                               ('pywhispercpp', '2'), ('cohere-transcribe', '3'),
                               ('qwen3-asr', '4'), ('rest-api', '5'), ('realtime-ws', '6')):
            with self.subTest(backend=backend):
                _, asked = self._select(backend, [None, None], confirm=False)
                self.assertEqual(asked[0], ('Select backend', entry))

    def test_whisper_entry_installs_faster_whisper_on_nvidia(self):
        result, asked = self._select(None, ['2'])
        self.assertEqual(result, ('faster-whisper', False, False, None))
        self.assertEqual(len(asked), 1)

    def test_switching_local_backends_asks_nothing_extra(self):
        result, asked = self._select('faster-whisper', ['1', '3'])
        self.assertEqual(result, ('parakeet-cpp', False, True, None))
        self.assertEqual([prompt for prompt, _ in asked], ['Select backend', 'Select Parakeet'])

    def test_changing_onnx_model_is_not_a_reinstall(self):
        result, asked = self._select('onnx-asr', ['1', '2'])
        self.assertEqual(result, ('onnx-asr', False, False, 'orukeet'))
        self.assertNotIn('Reinstall backend?', [prompt for prompt, _ in asked])

    def test_same_choice_offers_reinstall(self):
        for confirm, reinstall in ((True, True), (False, False)):
            result, asked = self._select('cohere-transcribe', ['3'], confirm=confirm)
            self.assertEqual(result, ('cohere-transcribe', False, reinstall, None))
            self.assertIn('Reinstall backend?', [prompt for prompt, _ in asked])


if __name__ == '__main__':
    unittest.main()
