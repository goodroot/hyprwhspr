import json
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib" / "src"))

from processing_trace import build_processing_trace, classify_boundary_mode, classify_vad_mode


class ConfigStub:
    def __init__(self, settings=None):
        self.settings = settings or {}

    def get_setting(self, name, default=None):
        return self.settings.get(name, default)

    def get_word_overrides(self):
        return self.settings.get("word_overrides", {})

    def get_filter_filler_words(self):
        return self.settings.get("filter_filler_words", False)

    def get_filler_words(self):
        return self.settings.get("filler_words", [])


class ProcessingTraceTests(unittest.TestCase):
    def test_trace_is_unicode_safe_and_uses_preprocessing_only(self):
        config = ConfigStub({
            "transcription_backend": "onnx-asr",
            "onnx_asr_model": "parakeet-日本語",
            "onnx_asr_use_vad": True,
            "onnx_asr_vad_min_duration": 12.5,
            "symbol_replacements": True,
            "post_transcription_hook": "dangerous-hook",
            "append_trailing_space": True,
        })
        with mock.patch.object(subprocess, "run") as run:
            trace = build_processing_trace("Café new line. 東京", config)
        run.assert_not_called()
        self.assertEqual(trace["raw"], "Café new line. 東京")
        self.assertEqual(trace["preprocessed"], "Café\n東京")
        self.assertEqual(trace["model"], "parakeet-日本語")
        self.assertTrue(trace["hook_present"])
        self.assertEqual(trace["onnx_vad_min_duration"], 12.5)
        encoded = json.dumps(trace, ensure_ascii=False)
        self.assertEqual(json.loads(encoded), trace)
        self.assertIn("東京", encoded)
        self.assertIn(r"\n", encoded)
        self.assertFalse(trace["preprocessed"].endswith(" "))

    def test_vad_classification_matrix(self):
        cases = [
            ({"transcription_backend": "pywhispercpp"}, "none"),
            ({"transcription_backend": "cpu", "pywhispercpp_use_vad": True}, "silero_filter"),
            ({"transcription_backend": "faster-whisper", "faster_whisper_vad_filter": True}, "silero_filter"),
            ({"transcription_backend": "onnx-asr", "onnx_asr_use_vad": True}, "silero_segmented"),
            ({"transcription_backend": "onnx-asr", "onnx_asr_use_vad": False}, "none"),
            ({"transcription_backend": "realtime-ws", "websocket_provider": "openai", "websocket_model": "gpt-transcribe"}, "manual_commit"),
            ({"transcription_backend": "realtime-ws", "websocket_provider": "openai", "websocket_model": "gpt-4o-mini-transcribe"}, "server_vad"),
            ({"transcription_backend": "realtime-ws", "websocket_provider": "google"}, "server_vad"),
            ({"transcription_backend": "realtime-ws", "websocket_provider": "elevenlabs"}, "provider_managed"),
            ({"transcription_backend": "rest-api"}, "provider_managed"),
        ]
        for settings, expected in cases:
            with self.subTest(settings=settings):
                self.assertEqual(classify_vad_mode(ConfigStub(settings)), expected)

    def test_client_boundary_classification(self):
        self.assertEqual(
            classify_boundary_mode(ConfigStub({"recording_mode": "continuous"})),
            "continuous_silence",
        )
        self.assertEqual(
            classify_boundary_mode(ConfigStub({"recording_mode": "toggle", "silence_timeout": 2})),
            "silence_auto_stop",
        )
        self.assertEqual(
            classify_boundary_mode(ConfigStub({"recording_mode": "auto", "silence_timeout": "3"})),
            "silence_auto_stop",
        )
        self.assertEqual(classify_boundary_mode(ConfigStub()), "manual_stop")

    def test_trace_reports_effective_numeric_settings(self):
        trace = build_processing_trace("hello", ConfigStub({
            "transcription_backend": "onnx-asr",
            "onnx_asr_use_vad": True,
            "onnx_asr_vad_min_duration": "invalid",
            "silence_timeout": "3",
            "continuous_silence_seconds": "1.5",
            "continuous_silence_threshold": "nan",
        }))
        self.assertEqual(trace["silence_timeout"], 3.0)
        self.assertEqual(trace["continuous_silence_seconds"], 1.5)
        self.assertEqual(trace["continuous_silence_threshold"], 0.0)
        self.assertEqual(trace["onnx_vad_min_duration"], 30.0)

    def test_empty_trace_is_still_one_json_document(self):
        trace = build_processing_trace("", ConfigStub())
        encoded = json.dumps(trace, ensure_ascii=False) + "\n"
        self.assertEqual(json.loads(encoded), trace)
        self.assertEqual(trace["raw"], "")
        self.assertEqual(trace["preprocessed"], "")


if __name__ == "__main__":
    unittest.main()
