import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib" / "src"))

import openai_realtime_models as caps  # noqa: E402
from provider_registry import (  # noqa: E402
    PROVIDERS,
    get_models_for_backend,
    get_realtime_mode,
    model_backends,
)


class ModelBackendRoutingTests(unittest.TestCase):
    """Every model declares which setup picker offers it, in one place."""

    def test_models_default_to_the_rest_picker(self):
        self.assertEqual(model_backends({}), ("rest-api",))

    def test_realtime_models_stay_out_of_the_rest_picker(self):
        rest_models = get_models_for_backend("openai", "rest-api")
        for model_id in ("gpt-transcribe", "gpt-live-transcribe", "gpt-realtime-2.1"):
            self.assertNotIn(model_id, rest_models)

    def test_rest_models_stay_out_of_the_realtime_picker(self):
        realtime_models = get_models_for_backend("openai", "realtime-ws")
        for model_id in ("whisper-1", "gpt-4o-transcribe"):
            self.assertNotIn(model_id, realtime_models)

    def test_elevenlabs_offers_only_its_realtime_model(self):
        self.assertEqual(list(get_models_for_backend("elevenlabs", "rest-api")), [])
        self.assertEqual(
            list(get_models_for_backend("elevenlabs", "realtime-ws")),
            ["scribe_v2_realtime"],
        )

    def test_every_realtime_model_declares_capabilities(self):
        # A realtime model without a 'realtime' block would silently fall back to
        # server VAD and the singular language field.
        for provider_id in PROVIDERS:
            for model_id, model_data in get_models_for_backend(
                provider_id, "realtime-ws"
            ).items():
                self.assertIn("realtime", model_data, f"{provider_id}/{model_id}")


class RealtimeModeTests(unittest.TestCase):
    def test_transcription_models_default_to_transcribe(self):
        for model_id in (
            "gpt-transcribe",
            "gpt-live-transcribe",
            "gpt-realtime-whisper",
        ):
            self.assertEqual(get_realtime_mode("openai", model_id), "transcribe")

    def test_conversational_models_default_to_converse(self):
        for model_id in ("gpt-realtime-2.1", "gpt-realtime-2.1-mini"):
            self.assertEqual(get_realtime_mode("openai", model_id), "converse")

    def test_unknown_and_custom_models_default_to_transcribe(self):
        self.assertEqual(get_realtime_mode("custom", "qwen-audio-realtime"), "transcribe")
        self.assertEqual(get_realtime_mode("openai", "not-a-model"), "transcribe")


class TranscriptionCapabilityTests(unittest.TestCase):
    def test_gpt_transcribe_commits_manually_without_live_deltas(self):
        self.assertTrue(caps.is_transcription_only("gpt-transcribe"))
        self.assertTrue(caps.uses_manual_commit("gpt-transcribe"))
        self.assertTrue(caps.uses_language_context("gpt-transcribe"))
        self.assertFalse(caps.is_continuous("gpt-transcribe"))

    def test_live_transcribe_streams_deltas_continuously(self):
        self.assertTrue(caps.uses_language_context("gpt-live-transcribe"))
        self.assertTrue(caps.is_continuous("gpt-live-transcribe"))

    def test_realtime_whisper_streams_without_language_context(self):
        self.assertTrue(caps.is_continuous("gpt-realtime-whisper"))
        self.assertFalse(caps.uses_language_context("gpt-realtime-whisper"))

    def test_conversational_models_are_not_transcription_only(self):
        self.assertFalse(caps.is_transcription_only("gpt-realtime-2.1"))
        self.assertFalse(caps.uses_manual_commit("gpt-realtime-2.1"))

    def test_unknown_models_report_no_capabilities(self):
        # Custom endpoints keep server VAD and the singular language field.
        for model_id in ("qwen-audio-3.0-realtime-plus", "", None):
            self.assertFalse(caps.is_transcription_only(model_id))
            self.assertFalse(caps.uses_manual_commit(model_id))
            self.assertFalse(caps.uses_language_context(model_id))
            self.assertFalse(caps.is_continuous(model_id))


if __name__ == "__main__":
    unittest.main()
