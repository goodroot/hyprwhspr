import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class MainStartupSafetyTests(unittest.TestCase):
    def test_realtime_partial_callback_registration_is_guarded(self):
        tree = ast.parse((ROOT / "lib" / "main.py").read_text(encoding="utf-8"))

        guarded = False
        for node in ast.walk(tree):
            if not isinstance(node, ast.If):
                continue
            test = node.test
            if (
                isinstance(test, ast.Call)
                and isinstance(test.func, ast.Name)
                and test.func.id == "hasattr"
                and len(test.args) == 2
                and isinstance(test.args[1], ast.Constant)
                and test.args[1].value == "set_realtime_partial_callback"
            ):
                for child in ast.walk(node):
                    if (
                        isinstance(child, ast.Call)
                        and isinstance(child.func, ast.Attribute)
                        and child.func.attr == "set_realtime_partial_callback"
                    ):
                        guarded = True

        self.assertTrue(guarded)

    def test_show_mic_osd_clears_preview_before_showing(self):
        tree = ast.parse((ROOT / "lib" / "main.py").read_text(encoding="utf-8"))

        show_func = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "_show_mic_osd":
                show_func = node
                break

        self.assertIsNotNone(show_func)

        clear_line = None
        show_line = None
        for node in ast.walk(show_func):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr == "clear_preview_text":
                clear_line = node.lineno
            elif node.func.attr == "show":
                show_line = node.lineno

        self.assertIsNotNone(clear_line)
        self.assertIsNotNone(show_line)
        self.assertLess(clear_line, show_line)


if __name__ == "__main__":
    unittest.main()
