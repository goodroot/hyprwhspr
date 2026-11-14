import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class OnnxAsrVadRoutingTests(unittest.TestCase):
    def _parse_onnx_backend(self):
        return ast.parse(
            (ROOT / "lib" / "src" / "backends" / "onnx_asr_backend.py").read_text(encoding="utf-8")
        )

    def _find_function(self, tree, name):
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == name:
                return node
        return None

    def test_onnx_vad_model_is_kept_separate_from_direct_model(self):
        import sys
        from unittest.mock import Mock, patch
        if str(ROOT / 'lib' / 'src') not in sys.path:
            sys.path.insert(0, str(ROOT / 'lib' / 'src'))
        from onnx_model import load_model
        runtime = Mock()
        with patch.dict(sys.modules, {'onnx_asr': runtime}):
            direct, vad = load_model('custom', use_vad=True)
        self.assertIs(direct, runtime.load_model.return_value)
        self.assertIs(vad, direct.with_vad.return_value)
        direct.with_vad.assert_called_once_with(runtime.load_vad.return_value, batch_size=1)

    def test_vad_batch_size_is_configurable_and_clamped(self):
        import sys
        from unittest.mock import Mock, patch
        if str(ROOT / 'lib' / 'src') not in sys.path:
            sys.path.insert(0, str(ROOT / 'lib' / 'src'))
        from onnx_model import load_model, vad_batch_size
        runtime = Mock()
        with patch.dict(sys.modules, {'onnx_asr': runtime}):
            direct, _ = load_model('custom', use_vad=True, vad_batch_size=4)
        direct.with_vad.assert_called_once_with(runtime.load_vad.return_value, batch_size=4)
        for raw, expected in (({}, 1), (4, 4), (2.9, 2), (0, 1), (-3, 1), ('8', 1),
                              (None, 1), (True, 1), (float('nan'), 1), (10**9, 64)):
            settings = raw if isinstance(raw, dict) else {'onnx_asr_vad_batch_size': raw}
            self.assertEqual(vad_batch_size(settings), expected, raw)

    def test_active_provider_reads_the_session_and_never_raises(self):
        import sys
        import types
        if str(ROOT / 'lib' / 'src') not in sys.path:
            sys.path.insert(0, str(ROOT / 'lib' / 'src'))
        from onnx_model import active_provider

        def model(**attrs):
            return types.SimpleNamespace(asr=types.SimpleNamespace(**attrs))

        def session(*providers):
            return types.SimpleNamespace(get_providers=lambda: list(providers))

        fallback = session('CPUExecutionProvider')
        self.assertEqual(active_provider(model(_encoder=fallback)), 'CPUExecutionProvider')
        self.assertEqual(
            active_provider(model(_model=session('CUDAExecutionProvider', 'CPUExecutionProvider'))),
            'CUDAExecutionProvider')
        for unreadable in (object(), model(), model(_encoder=session()), model(_encoder=object())):
            self.assertIsNone(active_provider(unreadable))

    def test_vad_timing_covers_the_lazy_segment_generator(self):
        import sys
        import types
        from unittest.mock import patch
        if str(ROOT / 'lib' / 'src') not in sys.path:
            sys.path.insert(0, str(ROOT / 'lib' / 'src'))
        from backends import onnx_asr_backend
        import numpy as np

        clock = [100.0]

        def segments():
            clock[0] += 36.0  # the real work happens while segments are consumed
            yield types.SimpleNamespace(text='hello')

        settings = {'onnx_asr_vad_min_duration': 0}
        manager = types.SimpleNamespace(
            config=types.SimpleNamespace(get_setting=lambda key, default=None: settings.get(key, default)))
        backend = onnx_asr_backend.OnnxAsrBackend(manager)
        backend._onnx_asr_model = object()
        backend._onnx_asr_vad_model = types.SimpleNamespace(recognize=lambda audio, sample_rate: segments())
        logged = []
        with patch.object(onnx_asr_backend, 'log', logged.append), \
             patch.object(onnx_asr_backend.time, 'time', lambda: clock[0]):
            text = backend.transcribe(np.zeros(16000, dtype=np.float32))
        self.assertEqual(text, 'hello')
        self.assertIn('[ONNX-ASR] Transcription completed in 36.00s', logged)

    def test_onnx_vad_is_duration_gated_at_transcription_time(self):
        tree = self._parse_onnx_backend()
        transcribe_onnx = self._find_function(tree, "transcribe")
        self.assertIsNotNone(transcribe_onnx)

        reads_threshold = any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "_get_onnx_asr_vad_min_duration"
            for node in ast.walk(transcribe_onnx)
        )
        compares_duration_to_threshold = any(
            isinstance(node, ast.Compare)
            and isinstance(node.left, ast.Name)
            and node.left.id == "audio_duration"
            and any(
                isinstance(comparator, ast.Name) and comparator.id == "vad_min_duration"
                for comparator in node.comparators
            )
            for node in ast.walk(transcribe_onnx)
        )

        self.assertTrue(reads_threshold)
        self.assertTrue(compares_duration_to_threshold)

    def test_onnx_vad_threshold_comes_from_config(self):
        tree = self._parse_onnx_backend()
        threshold_func = self._find_function(tree, "_get_onnx_asr_vad_min_duration")
        self.assertIsNotNone(threshold_func)

        reads_config_key = any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get_setting"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "onnx_asr_vad_min_duration"
            for node in ast.walk(threshold_func)
        )
        self.assertTrue(reads_config_key)

    def test_onnx_recognize_receives_capture_sample_rate(self):
        tree = self._parse_onnx_backend()
        transcribe_onnx = self._find_function(tree, "transcribe")
        self.assertIsNotNone(transcribe_onnx)

        recognize_calls = [
            node for node in ast.walk(transcribe_onnx)
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "recognize"
            )
        ]

        self.assertTrue(
            any(
                any(
                    keyword.arg == "sample_rate"
                    and isinstance(keyword.value, ast.Name)
                    and keyword.value.id == "sample_rate"
                    for keyword in call.keywords
                )
                for call in recognize_calls
            )
        )


if __name__ == "__main__":
    unittest.main()
