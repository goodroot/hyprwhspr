import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib"))
sys.path.insert(0, str(ROOT / "lib" / "src"))

import numpy as np

from backends.base import TranscriptionBackend

ENGLISH_DEFAULT = (
    "Transcribe with proper capitalization, including sentence beginnings, "
    "proper nouns, titles, and standard English capitalization rules."
)


class FakeConfig:
    """Stands in for ConfigManager: defaults merged in, defaults still visible."""

    def __init__(self, overrides=None):
        self.default_config = {"whisper_prompt": "", "whisper_prompt_en": ENGLISH_DEFAULT}
        self.config = dict(self.default_config)
        self.config.update(overrides or {})

    def get_setting(self, key, default=None):
        return self.config.get(key, default)


class FakeManager:
    def __init__(self, config):
        self.config = config


def resolve(language, overrides=None):
    return TranscriptionBackend(FakeManager(FakeConfig(overrides))).resolve_whisper_prompt(language)


class WhisperPromptResolutionTests(unittest.TestCase):
    """The shipped English prompt must never reach non-English audio (#233)."""

    def test_non_english_gets_no_prompt_by_default(self):
        self.assertEqual(resolve("pt"), (None, "none"))

    def test_non_english_uses_its_own_prompt(self):
        self.assertEqual(
            resolve("pt", {"whisper_prompt_pt": "Transcreva em português."}),
            ("Transcreva em português.", "whisper_prompt_pt"),
        )

    def test_english_keeps_the_shipped_default(self):
        self.assertEqual(resolve("en"), (ENGLISH_DEFAULT, "whisper_prompt_en"))

    def test_unknown_language_falls_back_to_english(self):
        self.assertEqual(resolve(None), (ENGLISH_DEFAULT, "whisper_prompt_en"))

    def test_configured_global_prompt_beats_the_shipped_default(self):
        self.assertEqual(
            resolve("en", {"whisper_prompt": "Talk like a pirate."}),
            ("Talk like a pirate.", "whisper_prompt"),
        )

    def test_language_prompt_beats_configured_global_prompt(self):
        self.assertEqual(
            resolve("de", {"whisper_prompt": "Global.", "whisper_prompt_de": "Deutsch."}),
            ("Deutsch.", "whisper_prompt_de"),
        )

    def test_global_prompt_applies_to_every_language(self):
        self.assertEqual(
            resolve("pt", {"whisper_prompt": "Global."}),
            ("Global.", "whisper_prompt"),
        )


class FakeFasterWhisperModel:
    """Mimics WhisperModel's detect_language/transcribe contract (faster-whisper 1.2.1)."""

    def __init__(self, detected=("pt", 0.99), is_multilingual=True, detect_error=None):
        self.detected = detected
        self.detect_error = detect_error
        self.model = type("Ct2Model", (), {"is_multilingual": is_multilingual})()
        self.transcribe_kwargs = None

    def detect_language(self, audio):
        if self.detect_error:
            raise self.detect_error
        return (*self.detected, [])

    def transcribe(self, audio, **kwargs):
        self.transcribe_kwargs = kwargs
        return [], None


class FasterWhisperPromptWiringTests(unittest.TestCase):
    """The detected language must reach both the prompt and the decoder."""

    def _transcribe(self, model, overrides=None):
        from backends.faster_whisper_backend import FasterWhisperBackend

        manager = FakeManager(FakeConfig(overrides))
        manager.current_model = "large-v3-turbo"
        manager._last_use_time = 0.0
        backend = FasterWhisperBackend(manager)
        backend._faster_whisper_model = model
        backend.transcribe(np.zeros(16000, dtype=np.float32))
        return model.transcribe_kwargs

    def test_detected_language_drives_the_prompt(self):
        kwargs = self._transcribe(FakeFasterWhisperModel(detected=("pt", 0.99)))
        self.assertEqual(kwargs["language"], "pt")
        self.assertNotIn("initial_prompt", kwargs)

    def test_detected_english_keeps_the_shipped_prompt(self):
        kwargs = self._transcribe(FakeFasterWhisperModel(detected=("en", 0.98)))
        self.assertEqual(kwargs["initial_prompt"], ENGLISH_DEFAULT)

    def test_english_only_model_skips_detection(self):
        model = FakeFasterWhisperModel(is_multilingual=False, detect_error=RuntimeError("boom"))
        kwargs = self._transcribe(model)
        self.assertEqual(kwargs["language"], "en")

    def test_detection_failure_falls_back_to_internal_detection(self):
        model = FakeFasterWhisperModel(detect_error=RuntimeError("boom"))
        kwargs = self._transcribe(model)
        self.assertNotIn("language", kwargs)
        self.assertEqual(kwargs["initial_prompt"], ENGLISH_DEFAULT)


class FakePywhispercppModel:
    def __init__(self, detected=("pt", 1.0)):
        self.detected = detected
        self.transcribe_kwargs = None

    def auto_detect_language(self, audio):
        return self.detected, {}

    def transcribe(self, audio, **kwargs):
        self.transcribe_kwargs = kwargs
        return []


class PywhispercppPromptWiringTests(unittest.TestCase):
    def _transcribe(self, model, overrides=None):
        from backends.pywhispercpp_backend import PywhispercppBackend

        manager = FakeManager(FakeConfig(overrides))
        manager.current_model = "large-v3-turbo"
        manager._last_use_time = 0.0
        backend = PywhispercppBackend(manager)
        backend._pywhisper_model = model
        backend.transcribe(np.zeros(16000, dtype=np.float32))
        return model.transcribe_kwargs

    def test_auto_detected_portuguese_gets_no_english_prompt(self):
        kwargs = self._transcribe(FakePywhispercppModel(detected=("pt", 1.0)))
        self.assertEqual(kwargs["language"], "pt")
        self.assertNotIn("initial_prompt", kwargs)

    def test_auto_detected_english_keeps_the_shipped_prompt(self):
        kwargs = self._transcribe(FakePywhispercppModel(detected=("en", 0.97)))
        self.assertEqual(kwargs["initial_prompt"], ENGLISH_DEFAULT)


if __name__ == "__main__":
    unittest.main()
