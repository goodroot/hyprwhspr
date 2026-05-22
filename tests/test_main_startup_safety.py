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


if __name__ == "__main__":
    unittest.main()
