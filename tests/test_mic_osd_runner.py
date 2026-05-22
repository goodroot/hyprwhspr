import sys
import tempfile
import types
import unittest
import builtins
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib"))

from mic_osd.runner import MicOSDRunner
import mic_osd.runner as runner_module


class MicOSDRunnerTests(unittest.TestCase):
    def _import_window_with_stubs(self):
        for module_name in ("mic_osd.window",):
            sys.modules.pop(module_name, None)

        cairo_module = types.SimpleNamespace(
            FONT_SLANT_NORMAL=0,
            FONT_WEIGHT_NORMAL=0,
            Context=object,
        )
        gtk_module = types.SimpleNamespace(
            Window=object,
            DrawingArea=type(
                "DrawingArea",
                (),
                {
                    "set_content_width": lambda self, value: None,
                    "set_content_height": lambda self, value: None,
                    "set_draw_func": lambda self, value: None,
                },
            ),
            CssProvider=object,
            StyleContext=types.SimpleNamespace(add_provider_for_display=lambda *args: None),
            STYLE_PROVIDER_PRIORITY_APPLICATION=0,
        )
        gdk_module = types.SimpleNamespace(Display=types.SimpleNamespace(get_default=lambda: None))
        glib_module = types.SimpleNamespace(Error=Exception)
        layer_shell_module = types.SimpleNamespace(
            init_for_window=lambda *args: None,
            set_namespace=lambda *args: None,
            set_layer=lambda *args: None,
            set_anchor=lambda *args: None,
            set_margin=lambda *args: None,
            set_exclusive_zone=lambda *args: None,
            set_keyboard_mode=lambda *args: None,
            Layer=types.SimpleNamespace(OVERLAY=0),
            Edge=types.SimpleNamespace(BOTTOM=0, LEFT=1, RIGHT=2, TOP=3),
            KeyboardMode=types.SimpleNamespace(NONE=0),
        )
        gi_module = types.SimpleNamespace(require_version=lambda *args: None)
        gi_repository = types.SimpleNamespace(
            Gtk=gtk_module,
            Gdk=gdk_module,
            GLib=glib_module,
            Gtk4LayerShell=layer_shell_module,
        )

        patcher = mock.patch.dict(
            sys.modules,
            {
                "cairo": cairo_module,
                "gi": gi_module,
                "gi.repository": gi_repository,
            },
        )
        with patcher:
            import mic_osd.window as window_module
        return window_module, cairo_module

    def test_preview_text_is_written_as_utf8_with_restrictive_permissions(self):
        with tempfile.TemporaryDirectory() as tmp:
            preview_file = Path(tmp) / "hyprwhspr" / "transcript_preview"
            original = runner_module.TRANSCRIPT_PREVIEW_FILE
            runner_module.TRANSCRIPT_PREVIEW_FILE = preview_file
            try:
                text = "cafe 東京"
                MicOSDRunner().set_preview_text(text)

                self.assertEqual(preview_file.read_bytes(), text.encode("utf-8"))
                self.assertEqual(preview_file.read_text(encoding="utf-8"), text)
                self.assertEqual(preview_file.parent.stat().st_mode & 0o777, 0o700)
                self.assertEqual(preview_file.stat().st_mode & 0o777, 0o600)
            finally:
                runner_module.TRANSCRIPT_PREVIEW_FILE = original

    def test_window_module_imports_with_cairo_available(self):
        window_module, cairo_module = self._import_window_with_stubs()

        self.assertIs(window_module.cairo, cairo_module)

    def test_is_available_returns_false_when_cairo_missing(self):
        original_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "cairo":
                raise ImportError("no cairo")
            return original_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=fake_import):
            self.assertFalse(MicOSDRunner.is_available())

    def test_text_extents_support_tuple_and_attribute_shapes(self):
        window_module, _ = self._import_window_with_stubs()
        window = object.__new__(window_module.OSDWindow)

        class TupleContext:
            def text_extents(self, text):
                return (0, 0, len(text) * 5, 10, 0, 0)

        class ObjectContext:
            def text_extents(self, text):
                return types.SimpleNamespace(width=len(text) * 5, height=10)

        self.assertEqual(window._text_width(TupleContext(), "abcd"), 20)
        self.assertEqual(window._text_height(TupleContext(), "abcd"), 10)
        self.assertEqual(window._text_width(ObjectContext(), "abcd"), 20)
        self.assertEqual(window._text_height(ObjectContext(), "abcd"), 10)

    def test_preview_text_preserves_spaces_but_trims_newlines(self):
        with tempfile.TemporaryDirectory() as tmp:
            preview_file = Path(tmp) / "hyprwhspr" / "transcript_preview"
            original = runner_module.TRANSCRIPT_PREVIEW_FILE
            runner_module.TRANSCRIPT_PREVIEW_FILE = preview_file
            try:
                MicOSDRunner().set_preview_text(" cafe 東京 \n")

                self.assertEqual(preview_file.read_text(encoding="utf-8"), " cafe 東京 ")
            finally:
                runner_module.TRANSCRIPT_PREVIEW_FILE = original

    def test_clear_preview_text_removes_stale_runtime_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            preview_file = Path(tmp) / "hyprwhspr" / "transcript_preview"
            original = runner_module.TRANSCRIPT_PREVIEW_FILE
            runner_module.TRANSCRIPT_PREVIEW_FILE = preview_file
            try:
                MicOSDRunner().set_preview_text("stale preview")
                self.assertTrue(preview_file.exists())

                MicOSDRunner().clear_preview_text()

                self.assertFalse(preview_file.exists())
            finally:
                runner_module.TRANSCRIPT_PREVIEW_FILE = original

    def test_hide_clears_preview_without_delayed_rewrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            preview_file = Path(tmp) / "hyprwhspr" / "transcript_preview"
            original_file = runner_module.TRANSCRIPT_PREVIEW_FILE
            original_interval = MicOSDRunner.PREVIEW_WRITE_INTERVAL_SECONDS
            runner_module.TRANSCRIPT_PREVIEW_FILE = preview_file
            MicOSDRunner.PREVIEW_WRITE_INTERVAL_SECONDS = 60.0
            runner = MicOSDRunner()
            try:
                runner.set_preview_text("first")
                runner.set_preview_text("ignored throttled update")

                runner.hide()

                self.assertFalse(preview_file.exists())
            finally:
                runner_module.TRANSCRIPT_PREVIEW_FILE = original_file
                MicOSDRunner.PREVIEW_WRITE_INTERVAL_SECONDS = original_interval

    def test_high_frequency_preview_updates_are_throttled_without_timer(self):
        with tempfile.TemporaryDirectory() as tmp:
            preview_file = Path(tmp) / "hyprwhspr" / "transcript_preview"
            original_file = runner_module.TRANSCRIPT_PREVIEW_FILE
            original_interval = MicOSDRunner.PREVIEW_WRITE_INTERVAL_SECONDS
            runner_module.TRANSCRIPT_PREVIEW_FILE = preview_file
            MicOSDRunner.PREVIEW_WRITE_INTERVAL_SECONDS = 60.0
            runner = MicOSDRunner()
            try:
                runner.set_preview_text("first")
                runner.set_preview_text("second")
                runner.set_preview_text("third")

                self.assertEqual(preview_file.read_text(encoding="utf-8"), "first")
                self.assertFalse(hasattr(runner, "_preview_flush_timer"))
            finally:
                runner_module.TRANSCRIPT_PREVIEW_FILE = original_file
                MicOSDRunner.PREVIEW_WRITE_INTERVAL_SECONDS = original_interval

    def test_requirements_include_pycairo_for_service_environment(self):
        requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")

        self.assertIn("pycairo", requirements.lower())


if __name__ == "__main__":
    unittest.main()
