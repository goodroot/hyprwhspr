import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class MicOSDMainFallbackTests(unittest.TestCase):
    def test_transcript_preview_fallback_uses_runtime_dir(self):
        tree = ast.parse((ROOT / "lib" / "mic_osd" / "main.py").read_text(encoding="utf-8"))

        uses_runtime = False
        assigns_preview_from_runtime = False
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == "runtime_dir" for target in node.targets)
            ):
                uses_runtime = True
            if (
                isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == "TRANSCRIPT_PREVIEW_FILE" for target in node.targets)
                and isinstance(node.value, ast.BinOp)
                and isinstance(node.value.left, ast.Name)
                and node.value.left.id == "runtime_dir"
            ):
                assigns_preview_from_runtime = True

        self.assertTrue(uses_runtime)
        self.assertTrue(assigns_preview_from_runtime)


if __name__ == "__main__":
    unittest.main()
