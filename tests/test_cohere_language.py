import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib"))
sys.path.insert(0, str(ROOT / "lib" / "src"))

from backend_utils import COHERE_LANGUAGES
from backends.cohere_backend import CohereBackend


class FakeConfig:
    def __init__(self, values=None):
        self.values = values or {}

    def get_setting(self, key, default=None):
        return self.values.get(key, default)


class FakeManager:
    def __init__(self, config):
        self.config = config
        self.current_model = None
        self.ready = False
        self._last_use_time = 0.0


class FakeCohereModel:
    """Mimics the transcribe() contract of CohereLabs/cohere-transcribe."""

    def __init__(self):
        self.config = type("CohereConfig", (), {
            "supported_languages": ["ar", "de", "el", "en", "es", "fr", "it",
                                    "ja", "ko", "nl", "pl", "pt", "vi", "zh"],
        })()
        self.calls = []

    def transcribe(self, processor, audio_arrays, sample_rates, language, compile):
        self.calls.append(language)
        return ["hello"]


def fake_torch_and_transformers(model):
    """Stand in for the torch/transformers imports inside initialize()."""
    torch = mock.MagicMock()
    torch.cuda.is_available.return_value = False

    loaded = mock.MagicMock()
    loaded.to.return_value = model
    model.eval = mock.MagicMock()

    transformers = mock.MagicMock()
    transformers.AutoModelForSpeechSeq2Seq.from_pretrained.return_value = loaded

    return mock.patch.dict(sys.modules, {"torch": torch, "transformers": transformers})


class CohereLanguageTests(unittest.TestCase):
    """The model cannot detect a language, so both gaps must be visible (#233 follow-up)."""

    def _backend(self, values=None):
        backend = CohereBackend(FakeManager(FakeConfig(values)))
        backend._cohere_model = FakeCohereModel()
        backend._cohere_processor = object()
        return backend

    def _transcribe(self, backend, language_override=None):
        with mock.patch("desktop_notify.notify") as notify:
            text = backend.transcribe(
                np.zeros(16000, dtype=np.float32), language_override=language_override
            )
        return text, notify

    def test_unset_language_uses_english_and_says_so_once(self):
        backend = self._backend()
        with mock.patch("builtins.print") as printed:
            self._transcribe(backend)
            self._transcribe(backend)

        self.assertEqual(backend._cohere_model.calls, ["en", "en"])
        notices = [c for c in printed.call_args_list if "cannot auto-detect" in str(c)]
        self.assertEqual(len(notices), 1)

    def test_supported_override_passes_through(self):
        backend = self._backend({"language": "en"})
        text, _ = self._transcribe(backend, language_override="pt")

        self.assertEqual(backend._cohere_model.calls, ["pt"])
        self.assertEqual(text, "hello")

    def test_unsupported_language_refuses_instead_of_failing_silently(self):
        backend = self._backend({"language": "sv"})
        text, notify = self._transcribe(backend)

        self.assertEqual(text, "")
        self.assertEqual(backend._cohere_model.calls, [])
        self.assertEqual(notify.call_count, 1)
        self.assertIn("sv", notify.call_args.args[1])

    def test_unsupported_language_notifies_once_not_per_recording(self):
        backend = self._backend({"language": "sv"})
        self._transcribe(backend)
        _, notify = self._transcribe(backend)

        self.assertEqual(notify.call_count, 0)

    def test_supported_languages_falls_back_when_model_has_no_list(self):
        backend = self._backend()
        backend._cohere_model.config = object()

        self.assertEqual(backend._supported_languages(), COHERE_LANGUAGES)

    def test_initialize_ignores_secondary_language_without_a_shortcut(self):
        backend = self._backend({"secondary_language": "no"})
        reported = []
        backend._reject_language = lambda language, setting: reported.append((language, setting))

        with fake_torch_and_transformers(backend._cohere_model), \
                mock.patch("backends.cohere_backend.get_credential", return_value=None):
            self.assertTrue(backend.initialize())

        self.assertEqual(reported, [])

    def test_initialize_reports_bad_languages_without_failing(self):
        backend = self._backend({
            "language": "sv",
            "secondary_language": "no",
            "secondary_shortcut": "SUPER+ALT+I",
        })
        reported = []
        backend._reject_language = lambda language, setting: reported.append((language, setting))

        with fake_torch_and_transformers(backend._cohere_model), \
                mock.patch("backends.cohere_backend.get_credential", return_value=None):
            self.assertTrue(backend.initialize())

        self.assertEqual(reported, [("sv", "language"), ("no", "secondary_language")])
        self.assertTrue(backend.ready)


if __name__ == "__main__":
    unittest.main()
