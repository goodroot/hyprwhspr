import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib"))

from mic_osd.runner import MicOSDRunner
import mic_osd.runner as runner_module


class MicOSDRunnerTests(unittest.TestCase):
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

        with mock.patch.dict(
            sys.modules,
            {
                "cairo": cairo_module,
                "gi": gi_module,
                "gi.repository": gi_repository,
            },
        ):
            import mic_osd.window as window_module

        self.assertIs(window_module.cairo, cairo_module)


if __name__ == "__main__":
    unittest.main()
