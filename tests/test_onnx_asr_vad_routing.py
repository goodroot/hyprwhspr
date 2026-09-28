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
        direct.with_vad.assert_called_once_with(runtime.load_vad.return_value)

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
