import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class MainStartupSafetyTests(unittest.TestCase):
    def _find_function(self, tree, name):
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == name:
                return node
        return None

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

        show_func = self._find_function(tree, "_show_mic_osd")
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

    def test_stop_recording_clears_preview_before_processing_state(self):
        tree = ast.parse((ROOT / "lib" / "main.py").read_text(encoding="utf-8"))

        stop_func = self._find_function(tree, "_stop_recording")
        self.assertIsNotNone(stop_func)

        clear_line = None
        processing_line = None
        for node in ast.walk(stop_func):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Attribute) and node.func.attr == "_clear_mic_osd_preview_text":
                clear_line = node.lineno
            elif (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "_set_visualizer_state"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == "processing"
            ):
                processing_line = node.lineno

        self.assertIsNotNone(clear_line)
        self.assertIsNotNone(processing_line)
        self.assertLess(clear_line, processing_line)

    def test_reset_stale_state_scrubs_transcript_preview(self):
        tree = ast.parse((ROOT / "lib" / "main.py").read_text(encoding="utf-8"))

        reset_func = self._find_function(tree, "_reset_stale_state")
        self.assertIsNotNone(reset_func)

        references_preview_file = any(
            isinstance(node, ast.Name) and node.id == "TRANSCRIPT_PREVIEW_FILE"
            for node in ast.walk(reset_func)
        )
        self.assertTrue(references_preview_file)

    def test_migrate_legacy_state_files_creates_compat_symlinks(self):
        tree = ast.parse((ROOT / "lib" / "main.py").read_text(encoding="utf-8"))

        migrate_func = self._find_function(tree, "_migrate_legacy_state_files")
        self.assertIsNotNone(migrate_func)

        calls_symlink = any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "symlink_to"
            for node in ast.walk(migrate_func)
        )
        self.assertTrue(calls_symlink)

        constants = {
            node.value
            for node in ast.walk(migrate_func)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        for name in ("recording_control", "recording_status", "audio_level"):
            self.assertIn(name, constants)

        app_class = next(
            node for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef) and node.name == "hyprwhsprApp"
        )
        init_func = self._find_function(app_class, "__init__")
        self.assertIsNotNone(init_func)
        called_from_init = any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "_migrate_legacy_state_files"
            for node in ast.walk(init_func)
        )
        self.assertTrue(called_from_init)

    def test_cancel_cleanup_clears_transcript_preview(self):
        tree = ast.parse((ROOT / "lib" / "main.py").read_text(encoding="utf-8"))

        cleanup_func = self._find_function(tree, "_cleanup_recording_state")
        self.assertIsNotNone(cleanup_func)

        clears_preview = any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "_clear_mic_osd_preview_text"
            for node in ast.walk(cleanup_func)
        )
        self.assertTrue(clears_preview)

    def test_process_audio_finally_clears_transcript_preview(self):
        tree = ast.parse((ROOT / "lib" / "main.py").read_text(encoding="utf-8"))

        process_func = self._find_function(tree, "_process_audio")
        self.assertIsNotNone(process_func)

        clears_in_finally = False
        for node in ast.walk(process_func):
            if not isinstance(node, ast.Try):
                continue
            for finalizer_node in node.finalbody:
                for child in ast.walk(finalizer_node):
                    if (
                        isinstance(child, ast.Call)
                        and isinstance(child.func, ast.Attribute)
                        and child.func.attr == "_clear_mic_osd_preview_text"
                    ):
                        clears_in_finally = True

        self.assertTrue(clears_in_finally)

    def test_inject_text_checks_injector_result_before_success_log(self):
        tree = ast.parse((ROOT / "lib" / "main.py").read_text(encoding="utf-8"))

        inject_func = self._find_function(tree, "_inject_text")
        self.assertIsNotNone(inject_func)

        result_checked_line = None
        success_log_line = None
        for node in ast.walk(inject_func):
            if (
                isinstance(node, ast.UnaryOp)
                and isinstance(node.op, ast.Not)
                and isinstance(node.operand, ast.Call)
                and isinstance(node.operand.func, ast.Attribute)
                and node.operand.func.attr == "inject_text"
            ):
                result_checked_line = node.lineno
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "print"
                and node.args
                and isinstance(node.args[0], ast.JoinedStr)
                and any(
                    isinstance(part, ast.Constant) and "[INJECT] Text injected" in str(part.value)
                    for part in node.args[0].values
                )
            ):
                success_log_line = node.lineno

        self.assertIsNotNone(result_checked_line)
        self.assertIsNotNone(success_log_line)
        self.assertLess(result_checked_line, success_log_line)


if __name__ == "__main__":
    unittest.main()
