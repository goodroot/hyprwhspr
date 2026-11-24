import sys
import tempfile
import threading
import types
import unittest
import builtins
import signal
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
            STYLE_PROVIDER_PRIORITY_APPLICATION=600,
            STYLE_PROVIDER_PRIORITY_USER=800,
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

    def test_custom_css_cannot_override_transparent_outer_surface(self):
        module, _ = self._import_window_with_stubs()
        transparency, custom = mock.Mock(), mock.Mock()
        display = object()
        with mock.patch.object(module, '_transparency_provider', None), \
                mock.patch.object(module.Gtk, 'CssProvider', side_effect=[transparency, custom]), \
                mock.patch.object(module.Gdk.Display, 'get_default', return_value=display), \
                mock.patch.object(module.Gtk.StyleContext, 'add_provider_for_display') as add:
            module.load_css('/optional/cosmetics.css')
        custom.load_from_path.assert_called_once_with('/optional/cosmetics.css')
        self.assertEqual(add.call_args_list, [
            mock.call(display, transparency, module.TRANSPARENCY_PRIORITY),
            mock.call(display, custom, module.Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION),
        ])
        self.assertGreater(module.TRANSPARENCY_PRIORITY, module.Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION)

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

    def test_pill_word_layout_uses_space_advance_instead_of_zero_ink_width(self):
        window_module, _ = self._import_window_with_stubs()
        window = object.__new__(window_module.OSDWindow)

        class SpaceHasNoInkContext:
            def text_extents(self, text):
                if text == " ":
                    return (0, 0, 0, 0, 6, 0)
                return (0, 0, len(text) * 5, 10, len(text) * 5, 0)

        positions, total = window._word_layout(
            SpaceHasNoInkContext(),
            ("one", "two"),
            100,
        )

        self.assertEqual(total, 36)
        self.assertEqual(positions, (32, 53))

    def test_pill_long_token_is_ellipsized_to_available_width(self):
        window_module, _ = self._import_window_with_stubs()
        window = object.__new__(window_module.OSDWindow)

        class AdvanceContext:
            def text_extents(self, text):
                advance = len(text) * 5
                return (0, 0, advance, 10, advance, 0)

        text = window._ellipsize_pill_token(
            AdvanceContext(),
            "averylongunbrokentoken",
            40,
        )

        self.assertEqual(text, "averylo…")
        self.assertLessEqual(window._text_advance(AdvanceContext(), text), 40)

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

    def test_hide_cancels_show_blocked_on_first_feed_frame(self):
        source_entered = threading.Event()
        source_release = threading.Event()
        hidden = threading.Event()
        signals = []
        show_results = []

        def blocked_source():
            source_entered.set()
            if not source_release.wait(5):
                raise TimeoutError("test did not release level source")
            return 0.0, []

        def record_signal(pid, sig):
            if sig in (signal.SIGUSR1, signal.SIGUSR2):
                signals.append(sig)
            if sig == signal.SIGUSR2:
                hidden.set()

        runner = MicOSDRunner(level_source=blocked_source)
        runner._process = types.SimpleNamespace(pid=4242, poll=lambda: None)
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(runner_module, "MIC_OSD_LEVEL_FEED_FILE", Path(tmp) / "feed"), \
                mock.patch.object(runner_module, "TRANSCRIPT_PREVIEW_FILE", Path(tmp) / "preview"), \
                mock.patch.object(runner, "is_available", return_value=True), \
                mock.patch.object(runner, "_ensure_daemon", return_value=True), \
                mock.patch.object(runner_module.os, "kill", side_effect=record_signal):
            showing = threading.Thread(target=lambda: show_results.append(runner.show()))
            hiding = threading.Thread(target=runner.hide)
            showing.start()
            try:
                self.assertTrue(source_entered.wait(2), "show did not reach level source")
                hiding.start()
                self.assertTrue(hidden.wait(2), "hide waited for the blocked level source")
                self.assertTrue(hiding.is_alive(), "feed teardown should still be blocked")
            finally:
                source_release.set()
                showing.join(5)
                if hiding.ident is not None:
                    hiding.join(5)
                runner._stop_level_feed()
            self.assertFalse(showing.is_alive())
            self.assertFalse(hiding.is_alive())
            self.assertEqual(signals, [signal.SIGUSR2])
            self.assertEqual(show_results, [False])

            # Cancellation must not prevent the next recording from showing.
            self.assertTrue(runner.show())
            runner.hide()
            self.assertEqual(signals[-2:], [signal.SIGUSR1, signal.SIGUSR2])

    def test_hide_during_daemon_startup_does_not_start_a_feed(self):
        runner = MicOSDRunner(level_source=lambda: (0.0, []))
        startup_entered = threading.Event()
        startup_release = threading.Event()
        results = []
        signals = []

        def start_daemon():
            startup_entered.set()
            startup_release.wait(5)
            runner._process = types.SimpleNamespace(pid=4242, poll=lambda: None)
            return True

        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(runner_module, "MIC_OSD_LEVEL_FEED_FILE", Path(tmp) / "feed"), \
                mock.patch.object(runner_module, "TRANSCRIPT_PREVIEW_FILE", Path(tmp) / "preview"), \
                mock.patch.object(runner, "is_available", return_value=True), \
                mock.patch.object(runner, "_ensure_daemon", side_effect=start_daemon), \
                mock.patch.object(runner_module.os, "kill", side_effect=lambda pid, sig: signals.append(sig)):
            worker = threading.Thread(target=lambda: results.append(runner.show()))
            worker.start()
            try:
                self.assertTrue(startup_entered.wait(2))
                runner.hide()
                startup_release.set()
                worker.join(5)
                self.assertFalse(worker.is_alive())
                self.assertEqual(results, [False])
                self.assertNotIn(signal.SIGUSR1, signals)
                self.assertFalse((Path(tmp) / "feed").exists())
            finally:
                startup_release.set()
                worker.join(5)
                runner._stop_level_feed()

    def test_new_show_waits_for_previous_hide_teardown(self):
        runner = MicOSDRunner(level_source=lambda: (0.0, []))
        runner._process = types.SimpleNamespace(pid=4242, poll=lambda: None)
        teardown_entered = threading.Event()
        teardown_release = threading.Event()
        show_finished = threading.Event()
        signals = []

        def blocked_teardown():
            teardown_entered.set()
            teardown_release.wait(5)

        def show_again():
            runner.show()
            show_finished.set()

        def record_signal(pid, sig):
            if sig in (signal.SIGUSR1, signal.SIGUSR2):
                signals.append(sig)

        with mock.patch.object(runner, "_stop_level_feed", side_effect=blocked_teardown), \
                mock.patch.object(runner, "clear_preview_text"), \
                mock.patch.object(runner, "is_available", return_value=True), \
                mock.patch.object(runner, "_ensure_daemon", return_value=True), \
                mock.patch.object(runner_module.os, "kill", side_effect=record_signal):
            hiding = threading.Thread(target=runner.hide)
            showing = threading.Thread(target=show_again)
            hiding.start()
            try:
                self.assertTrue(teardown_entered.wait(2))
                showing.start()
                self.assertFalse(
                    show_finished.wait(0.1),
                    "new show overtook the previous hide teardown",
                )
                self.assertNotIn(signal.SIGUSR1, signals)
            finally:
                teardown_release.set()
                hiding.join(5)
                if showing.ident is not None:
                    showing.join(5)

            self.assertFalse(hiding.is_alive())
            self.assertFalse(showing.is_alive())
            self.assertEqual(signals, [signal.SIGUSR2, signal.SIGUSR1])

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
