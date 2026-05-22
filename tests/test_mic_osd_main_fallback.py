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

    def test_hide_clears_preview_file_before_visibility_return(self):
        tree = ast.parse((ROOT / "lib" / "mic_osd" / "main.py").read_text(encoding="utf-8"))

        hide_func = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "_hide":
                hide_func = node
                break

        self.assertIsNotNone(hide_func)

        clear_line = None
        visible_return_line = None
        for node in ast.walk(hide_func):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "_clear_preview_file"
            ):
                clear_line = node.lineno
            if (
                isinstance(node, ast.If)
                and isinstance(node.test, ast.UnaryOp)
                and isinstance(node.test.op, ast.Not)
                and isinstance(node.test.operand, ast.Attribute)
                and node.test.operand.attr == "visible"
            ):
                visible_return_line = node.lineno

        self.assertIsNotNone(clear_line)
        self.assertIsNotNone(visible_return_line)
        self.assertLess(clear_line, visible_return_line)

    def test_cleanup_uses_preview_file_helper(self):
        tree = ast.parse((ROOT / "lib" / "mic_osd" / "main.py").read_text(encoding="utf-8"))

        cleanup_func = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "_cleanup":
                cleanup_func = node
                break

        self.assertIsNotNone(cleanup_func)

        calls_helper = any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "_clear_preview_file"
            for node in ast.walk(cleanup_func)
        )
        self.assertTrue(calls_helper)

    def test_state_poll_updates_window_visualizer_state(self):
        tree = ast.parse((ROOT / "lib" / "mic_osd" / "main.py").read_text(encoding="utf-8"))

        poll_func = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "_poll_state_file":
                poll_func = node
                break

        self.assertIsNotNone(poll_func)

        calls_window_state = any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "set_visualizer_state"
            for node in ast.walk(poll_func)
        )
        self.assertTrue(calls_window_state)


if __name__ == "__main__":
    unittest.main()
