import ast
import builtins
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib"))


class MicOSDMainFallbackTests(unittest.TestCase):
    def _stub_gtk_modules(self):
        gtk_module = types.SimpleNamespace(Application=object)
        glib_module = types.SimpleNamespace()
        gi_module = types.SimpleNamespace(require_version=lambda *args: None)
        gi_repository = types.SimpleNamespace(Gtk=gtk_module, GLib=glib_module)
        return {
            "gi": gi_module,
            "gi.repository": gi_repository,
        }

    def test_main_import_degrades_cleanly_when_cairo_missing(self):
        for module_name in (
            "mic_osd.main",
            "mic_osd.window",
            "mic_osd.visualizations",
            "mic_osd.visualizations.base",
            "mic_osd.visualizations.waveform",
            "mic_osd.visualizations.vu_meter",
        ):
            sys.modules.pop(module_name, None)

        original_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "cairo":
                raise ImportError("no cairo")
            return original_import(name, *args, **kwargs)

        with mock.patch.dict(sys.modules, self._stub_gtk_modules()), \
                mock.patch("builtins.__import__", side_effect=fake_import):
            import mic_osd.main as main_module

        self.assertIsNotNone(main_module._MIC_OSD_IMPORT_ERROR)
        with mock.patch.object(sys, "argv", ["mic-osd"]):
            self.assertEqual(main_module.main(), 1)

    def test_fallback_paths_use_runtime_dir(self):
        tree = ast.parse((ROOT / "lib" / "mic_osd" / "main.py").read_text(encoding="utf-8"))

        expected = {
            "RECORDING_STATUS_FILE",
            "VISUALIZER_STATE_FILE",
            "TRANSCRIPT_PREVIEW_FILE",
            "MIC_OSD_LEVEL_FEED_FILE",
        }
        uses_runtime = False
        assigned_from_runtime = set()
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == "runtime_dir" for target in node.targets)
            ):
                uses_runtime = True
            if (
                isinstance(node, ast.Assign)
                and isinstance(node.value, ast.BinOp)
                and isinstance(node.value.left, ast.Name)
                and node.value.left.id == "runtime_dir"
            ):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in expected:
                        assigned_from_runtime.add(target.id)

        self.assertTrue(uses_runtime)
        self.assertEqual(assigned_from_runtime, expected)

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
