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


class FakeTimer:
    def __init__(self, delay, callback, args=()):
        self.delay = delay
        self.callback = callback
        self.args = args
        self.daemon = False
        self.started = False
        self.cancelled = False

    def start(self):
        self.started = True

    def cancel(self):
        self.cancelled = True

    def fire(self):
        self.callback(*self.args)


class FakeCairoContext:
    def __init__(self):
        self.shown_text = []

    def select_font_face(self, *args):
        pass

    def set_font_size(self, *args):
        pass

    def text_extents(self, text):
        return (0, 0, len(text) * 5, 10, 0, 0)

    def set_source_rgba(self, *args):
        pass

    def rectangle(self, *args):
        pass

    def fill(self):
        pass

    def move_to(self, *args):
        pass

    def show_text(self, text):
        self.shown_text.append(text)


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

    def test_preview_text_write_uses_atomic_replace(self):
        with tempfile.TemporaryDirectory() as tmp:
            preview_file = Path(tmp) / "hyprwhspr" / "transcript_preview"
            original_file = runner_module.TRANSCRIPT_PREVIEW_FILE
            original_replace = runner_module.os.replace
            replace_calls = []
            runner_module.TRANSCRIPT_PREVIEW_FILE = preview_file
            try:
                def replace_spy(src, dst):
                    replace_calls.append((Path(src), Path(dst)))
                    original_replace(src, dst)

                with mock.patch.object(runner_module.os, "replace", side_effect=replace_spy):
                    MicOSDRunner().set_preview_text("atomic preview")

                self.assertEqual(preview_file.read_text(encoding="utf-8"), "atomic preview")
                self.assertEqual(len(replace_calls), 1)
                temp_path, final_path = replace_calls[0]
                self.assertNotEqual(temp_path, final_path)
                self.assertEqual(final_path, preview_file)
                self.assertTrue(temp_path.name.startswith(".transcript_preview."))
                self.assertFalse(temp_path.exists())
            finally:
                runner_module.TRANSCRIPT_PREVIEW_FILE = original_file

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

    def test_pid_validation_rejects_unrelated_process_cmdline(self):
        with mock.patch.object(runner_module.os, "kill", return_value=None), \
                mock.patch.object(runner_module.Path, "read_bytes", return_value=b"python\0not-osd\0"):
            self.assertFalse(MicOSDRunner._is_mic_osd_daemon_pid(12345))

    def test_pid_validation_accepts_mic_osd_daemon_cmdline(self):
        cmdline = b"python3\0-c\0from mic_osd.main import main\nsys.argv = ['mic-osd', '--daemon']\0"
        with mock.patch.object(runner_module.os, "kill", return_value=None), \
                mock.patch.object(runner_module.Path, "read_bytes", return_value=cmdline):
            self.assertTrue(MicOSDRunner._is_mic_osd_daemon_pid(12345))

    def test_pid_validation_accepts_daemon_environment_marker(self):
        with mock.patch.object(runner_module.os, "kill", return_value=None), \
                mock.patch.object(runner_module.Path, "read_bytes", return_value=b"HYPRWHSPR_MIC_OSD_DAEMON=1\0"):
            self.assertTrue(MicOSDRunner._is_mic_osd_daemon_pid(12345))

    def test_ensure_daemon_reuses_python_c_daemon_after_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            pid_file = Path(tmp) / "mic_osd.pid"
            pid_file.write_text("12345", encoding="utf-8")
            original_pid_file = runner_module.MIC_OSD_PID_FILE
            runner_module.MIC_OSD_PID_FILE = pid_file

            cmdline = b"python3\0-c\0from mic_osd.main import main\nsys.argv = ['mic-osd', '--daemon']\0"
            try:
                with mock.patch.object(runner_module.os, "kill", return_value=None), \
                        mock.patch.object(runner_module.Path, "read_bytes", side_effect=[b"", cmdline]), \
                        mock.patch.object(runner_module.subprocess, "Popen", return_value=types.SimpleNamespace()) as popen:
                    runner = MicOSDRunner()

                    self.assertTrue(runner._ensure_daemon())

                popen.assert_called_once()
                self.assertEqual(popen.call_args.args[0], ['true'])
                self.assertEqual(runner._orphaned_daemon_pid, 12345)
                self.assertTrue(pid_file.exists())
            finally:
                runner_module.MIC_OSD_PID_FILE = original_pid_file

    def test_orphaned_pid_signal_revalidates_before_sending(self):
        runner = MicOSDRunner()
        runner._process = types.SimpleNamespace(pid=999, poll=lambda: None)
        runner._orphaned_daemon_pid = 12345

        with mock.patch.object(MicOSDRunner, "is_available", return_value=True), \
                mock.patch.object(runner, "_is_mic_osd_daemon_pid", return_value=False), \
                mock.patch.object(runner_module.os, "kill") as kill:
            self.assertFalse(runner.show())

        kill.assert_not_called()
        self.assertIsNone(runner._process)
        self.assertIsNone(runner._orphaned_daemon_pid)

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

    def test_preview_text_draws_only_while_recording(self):
        window_module, _ = self._import_window_with_stubs()
        window = object.__new__(window_module.OSDWindow)
        window._preview_text = "live partial"
        window._visualizer_state = "processing"

        processing_cr = FakeCairoContext()
        window._draw_preview_text(processing_cr, 400, 68)

        window._visualizer_state = "recording"
        recording_cr = FakeCairoContext()
        window._draw_preview_text(recording_cr, 400, 68)

        self.assertEqual(processing_cr.shown_text, [])
        self.assertEqual(recording_cr.shown_text, ["live partial"])

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

    def test_hide_cancels_pending_preview_flush(self):
        with tempfile.TemporaryDirectory() as tmp:
            preview_file = Path(tmp) / "hyprwhspr" / "transcript_preview"
            original_file = runner_module.TRANSCRIPT_PREVIEW_FILE
            original_interval = MicOSDRunner.PREVIEW_WRITE_INTERVAL_SECONDS
            timers = []
            runner_module.TRANSCRIPT_PREVIEW_FILE = preview_file
            MicOSDRunner.PREVIEW_WRITE_INTERVAL_SECONDS = 60.0
            runner = MicOSDRunner()
            try:
                def make_timer(*args, **kwargs):
                    timer = FakeTimer(*args, **kwargs)
                    timers.append(timer)
                    return timer

                with mock.patch.object(runner_module.threading, "Timer", side_effect=make_timer):
                    runner.set_preview_text("first")
                    runner.set_preview_text("stale pending")

                runner.hide()
                timers[0].fire()

                self.assertTrue(timers[0].cancelled)
                self.assertFalse(preview_file.exists())
            finally:
                runner_module.TRANSCRIPT_PREVIEW_FILE = original_file
                MicOSDRunner.PREVIEW_WRITE_INTERVAL_SECONDS = original_interval

    def test_high_frequency_preview_updates_are_coalesced(self):
        with tempfile.TemporaryDirectory() as tmp:
            preview_file = Path(tmp) / "hyprwhspr" / "transcript_preview"
            original_file = runner_module.TRANSCRIPT_PREVIEW_FILE
            original_interval = MicOSDRunner.PREVIEW_WRITE_INTERVAL_SECONDS
            timers = []
            runner_module.TRANSCRIPT_PREVIEW_FILE = preview_file
            MicOSDRunner.PREVIEW_WRITE_INTERVAL_SECONDS = 60.0
            runner = MicOSDRunner()
            try:
                def make_timer(*args, **kwargs):
                    timer = FakeTimer(*args, **kwargs)
                    timers.append(timer)
                    return timer

                with mock.patch.object(runner_module.threading, "Timer", side_effect=make_timer):
                    runner.set_preview_text("first")
                    runner.set_preview_text("second")
                    runner.set_preview_text("third")

                self.assertEqual(preview_file.read_text(encoding="utf-8"), "first")
                self.assertEqual(len(timers), 1)
                self.assertTrue(timers[0].started)

                timers[0].fire()

                self.assertEqual(preview_file.read_text(encoding="utf-8"), "third")
            finally:
                runner_module.TRANSCRIPT_PREVIEW_FILE = original_file
                MicOSDRunner.PREVIEW_WRITE_INTERVAL_SECONDS = original_interval

    def test_requirements_include_pycairo_for_service_environment(self):
        requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")

        self.assertIn("pycairo", requirements.lower())


if __name__ == "__main__":
    unittest.main()
