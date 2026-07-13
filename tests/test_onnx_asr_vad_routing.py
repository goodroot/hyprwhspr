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
        tree = self._parse_onnx_backend()
        init_onnx = self._find_function(tree, "initialize")
        self.assertIsNotNone(init_onnx)

        assigns_vad_model = False
        overwrites_direct_model_with_vad = False
        for node in ast.walk(init_onnx):
            if not isinstance(node, ast.Assign):
                continue
            target_names = [
                target.attr
                for target in node.targets
                if isinstance(target, ast.Attribute)
            ]
            calls_with_vad = (
                isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute)
                and node.value.func.attr == "with_vad"
            )
            if "_onnx_asr_vad_model" in target_names and calls_with_vad:
                assigns_vad_model = True
            if "_onnx_asr_model" in target_names and calls_with_vad:
                overwrites_direct_model_with_vad = True

        self.assertTrue(assigns_vad_model)
        self.assertFalse(overwrites_direct_model_with_vad)

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
