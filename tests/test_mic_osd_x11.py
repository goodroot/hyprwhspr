"""X11 mic-OSD overlay."""

import importlib
from contextlib import contextmanager
import io
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib"))

from mic_osd.backend import GTK3, GTK4, bottom_center, select_backend
from mic_osd.runner import MicOSDRunner
import mic_osd.runner as runner_module
from src.cli import mic_osd as mic_osd_cli


class BackendSelectionTests(unittest.TestCase):
    def test_x11_display_selects_gtk3(self):
        self.assertEqual(select_backend({'DISPLAY': ':0'}), GTK3)

    def test_wayland_selects_gtk4_even_with_xwayland(self):
        self.assertEqual(
            select_backend({'DISPLAY': ':0', 'WAYLAND_DISPLAY': 'wayland-1'}),
            GTK4,
        )

    def test_forced_backend_wins(self):
        self.assertEqual(
            select_backend({'DISPLAY': ':0', 'HYPRWHSPR_MIC_OSD_BACKEND': 'gtk4'}),
            GTK4,
        )
        self.assertEqual(
            select_backend({'WAYLAND_DISPLAY': 'wayland-1',
                            'HYPRWHSPR_MIC_OSD_BACKEND': 'gtk3'}),
            GTK3,
        )

    def test_no_display_keeps_the_wayland_path(self):
        self.assertEqual(select_backend({}), GTK4)


class OverlayPlacementTests(unittest.TestCase):
    def test_overlay_is_centered_above_the_bottom_edge(self):
        geometry = types.SimpleNamespace(x=0, y=0, width=1920, height=1080)
        self.assertEqual(bottom_center(geometry, 400, 68, 130), (760, 882))

    def test_secondary_monitor_offsets_the_overlay(self):
        geometry = types.SimpleNamespace(x=1920, y=0, width=1280, height=720)
        self.assertEqual(bottom_center(geometry, 400, 68, 130), (2360, 522))

    def test_overlay_never_leaves_a_small_screen(self):
        geometry = types.SimpleNamespace(x=0, y=0, width=200, height=100)
        self.assertEqual(bottom_center(geometry, 400, 68, 130), (0, 0))


class X11AvailabilityTests(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.object(MicOSDRunner, "_x11_availability", None)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_session_needs_a_display_and_no_wayland(self):
        with mock.patch.dict(os.environ, {'DISPLAY': ':0'}, clear=True):
            self.assertTrue(MicOSDRunner.x11_session())
        with mock.patch.dict(
                os.environ, {'DISPLAY': ':0', 'WAYLAND_DISPLAY': 'wayland-1'}, clear=True):
            self.assertFalse(MicOSDRunner.x11_session())
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(MicOSDRunner.x11_session())

    def test_no_probe_without_an_x11_session(self):
        with mock.patch.object(MicOSDRunner, "x11_session", return_value=False), \
                mock.patch("mic_osd.runner.subprocess.run") as run:
            self.assertFalse(MicOSDRunner.x11_available())
        run.assert_not_called()

    def test_probe_accepts_a_gtk3_environment(self):
        result = types.SimpleNamespace(stdout=b"1\n", returncode=0)
        with mock.patch.object(MicOSDRunner, "x11_session", return_value=True), \
                mock.patch("mic_osd.runner.subprocess.run", return_value=result):
            self.assertTrue(MicOSDRunner.x11_available())

    def test_probe_rejects_a_missing_gtk3(self):
        result = types.SimpleNamespace(stdout=b"", returncode=1)
        with mock.patch.object(MicOSDRunner, "x11_session", return_value=True), \
                mock.patch("mic_osd.runner.subprocess.run", return_value=result):
            self.assertFalse(MicOSDRunner.x11_available())


class AvailabilityTests(unittest.TestCase):
    def test_x11_alone_makes_the_overlay_available(self):
        with mock.patch.object(MicOSDRunner, "_system_dependencies_available",
                               return_value=False), \
                mock.patch.object(MicOSDRunner, "_bundled_dependencies_available",
                                  return_value=False), \
                mock.patch.object(MicOSDRunner, "x11_available", return_value=True):
            self.assertTrue(MicOSDRunner.is_available())

    def test_unavailable_without_layer_shell_and_x11(self):
        with mock.patch.object(MicOSDRunner, "_system_dependencies_available",
                               return_value=False), \
                mock.patch.object(MicOSDRunner, "_bundled_dependencies_available",
                                  return_value=False), \
                mock.patch.object(MicOSDRunner, "x11_available", return_value=False):
            self.assertFalse(MicOSDRunner.is_available())


    def test_x11_reports_its_runtime(self):
        with mock.patch.object(MicOSDRunner, "x11_available", return_value=True), \
                mock.patch.object(MicOSDRunner, "_system_dependencies_available",
                                  return_value=False), \
                mock.patch.object(MicOSDRunner, "_bundled_dependencies_available",
                                  return_value=False):
            self.assertEqual(MicOSDRunner.runtime_source(), "X11 GTK3")

    def test_cli_accepts_a_gtk3_only_setup(self):
        with tempfile.TemporaryDirectory() as tmp:
            python = Path(tmp) / "bin" / "python"
            python.parent.mkdir()
            python.touch()
            run = mock.Mock(return_value=types.SimpleNamespace(returncode=0, stdout="AVAILABLE: X11 GTK3"))
            with mock.patch.object(mic_osd_cli, "VENV_DIR", Path(tmp)), \
                    mock.patch.object(mic_osd_cli.subprocess, "run", run):
                mic_osd_cli._query_mic_osd_availability()
        code = run.call_args.args[0][2]
        output = io.StringIO()
        with mock.patch.object(MicOSDRunner, "x11_available", return_value=True), \
                mock.patch("sys.stdout", output):
            exec(code, {})
        self.assertEqual(output.getvalue().strip(), "AVAILABLE: X11 GTK3")

    def test_missing_x11_dependencies_name_gtk3_without_layer_shell_advice(self):
        with mock.patch.object(MicOSDRunner, "x11_session", return_value=True), \
                mock.patch.object(MicOSDRunner, "x11_available", return_value=False):
            reason = MicOSDRunner.get_unavailable_reason()
        self.assertIn("GTK3", reason)
        self.assertIn("PyCairo", reason)
        self.assertNotIn("GTK4", reason)
        self.assertNotIn("layer-shell", reason)

    def test_available_x11_has_no_dependency_error(self):
        with mock.patch.object(MicOSDRunner, "x11_session", return_value=True), \
                mock.patch.object(MicOSDRunner, "x11_available", return_value=True):
            self.assertEqual(MicOSDRunner.get_unavailable_reason(), "")

    def test_wayland_keeps_layer_shell_dependency_advice(self):
        with mock.patch.object(MicOSDRunner, '_system_status', return_value=('layer-shell', '')), \
                mock.patch.object(MicOSDRunner, 'x11_session', return_value=False), \
                mock.patch.object(MicOSDRunner, 'x11_available', return_value=False), \
                mock.patch.object(MicOSDRunner, '_bundled_dependencies_available', return_value=False), \
                mock.patch.object(MicOSDRunner, '_get_distro_packages', return_value=('gtk-package', 'layer-package')):
            self.assertEqual(MicOSDRunner.get_unavailable_reason(),
                             'gtk4-layer-shell not installed. Install: layer-package')


class ClickThroughTests(unittest.TestCase):
    @contextmanager
    def _window_module(self):
        stubs = {
            "gi": types.SimpleNamespace(require_version=lambda *args: None),
            "gi.repository": types.SimpleNamespace(
                Gtk=types.SimpleNamespace(Window=object),
                Gdk=types.SimpleNamespace(WindowTypeHint=types.SimpleNamespace(NOTIFICATION=1)),
                GLib=types.SimpleNamespace()),
            "cairo": types.SimpleNamespace(Region=lambda: self.region),
        }
        self.region = object()
        with mock.patch.dict(sys.modules, stubs):
            sys.modules.pop("mic_osd.window_x11", None)
            sys.modules.pop("mic_osd.render", None)
            yield importlib.import_module("mic_osd.window_x11")

    def test_window_gets_an_empty_x_input_shape(self):
        with self._window_module() as window_x11:
            gdk_window = mock.Mock()
            fake = types.SimpleNamespace(get_window=lambda: gdk_window)
            window_x11.OSDWindow._make_click_through(fake)
        gdk_window.input_shape_combine_region.assert_called_once_with(self.region, 0, 0)

    def test_window_never_requests_focus_or_taskbar_entries(self):
        with self._window_module() as window_x11:
            window = mock.Mock(_width=400, _height=68)
            window_x11.OSDWindow._setup_window(window)
        window.set_accept_focus.assert_called_once_with(False)
        window.set_focus_on_map.assert_called_once_with(False)
        window.set_skip_taskbar_hint.assert_called_once_with(True)
        window.set_skip_pager_hint.assert_called_once_with(True)
        window.set_keep_above.assert_called_once_with(True)
        window.connect.assert_any_call('realize', window._on_realize)
        window.connect.assert_any_call('map-event', window._on_map_event)

    def test_realize_and_mapping_apply_empty_input_shape(self):
        with self._window_module() as window_x11:
            window = mock.Mock()
            window._make_click_through.side_effect = lambda: window_x11.OSDWindow._make_click_through(window)
            window_x11.OSDWindow._on_realize(window, window)
            self.assertFalse(window_x11.OSDWindow._on_map_event(window, window, object()))
        self.assertEqual(window.get_window.return_value.input_shape_combine_region.call_args_list,
                         [mock.call(self.region, 0, 0), mock.call(self.region, 0, 0)])
        self.assertEqual(window._reposition.call_count, 2)


class DaemonBackendTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.pid_file = Path(temp.name) / "mic_osd.pid"
        patcher = mock.patch.object(runner_module, "MIC_OSD_PID_FILE", self.pid_file)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _start(self, x11_available=False, layer_shell_error=None):
        runner = MicOSDRunner()
        if layer_shell_error is None:
            layer_env = mock.Mock(return_value={"KEEP": "1"})
        else:
            layer_env = mock.Mock(side_effect=layer_shell_error)
        with mock.patch.object(runner, "_layer_shell_environment", layer_env), \
                mock.patch.object(MicOSDRunner, "x11_available",
                                  return_value=x11_available), \
                mock.patch.object(MicOSDRunner, "layer_shell_active",
                                  side_effect=AssertionError("probed layer-shell")), \
                mock.patch("mic_osd.runner.time.sleep"), \
                mock.patch("mic_osd.runner.subprocess.Popen") as popen:
            popen.return_value.poll.return_value = None
            started = runner._ensure_daemon()
        return started, popen

    def test_layer_shell_daemon_keeps_the_gtk4_backend(self):
        started, popen = self._start()
        self.assertTrue(started)
        env = popen.call_args.kwargs["env"]
        self.assertEqual(env["HYPRWHSPR_MIC_OSD_BACKEND"], "gtk4")
        self.assertEqual(env["HYPRWHSPR_MIC_OSD_DAEMON"], "1")

    def test_x11_without_layer_shell_uses_the_gtk3_backend(self):
        started, popen = self._start(x11_available=True)
        self.assertTrue(started)
        env = popen.call_args.kwargs["env"]
        self.assertEqual(env["HYPRWHSPR_MIC_OSD_BACKEND"], "gtk3")
        self.assertEqual(env["HYPRWHSPR_MIC_OSD_DAEMON"], "1")

    def test_orphan_is_reused_without_a_probe(self):
        self.pid_file.write_text("4244")
        with mock.patch.object(MicOSDRunner, "_is_mic_osd_daemon_pid", return_value=True):
            started, popen = self._start()
        self.assertTrue(started)
        self.assertEqual(popen.call_args.args[0], ['true'])

    def test_without_layer_shell_or_x11_no_daemon_starts(self):
        started, popen = self._start(
            layer_shell_error=runner_module.LayerShellResolutionError("no layer shell"),
            x11_available=False,
        )
        self.assertFalse(started)
        popen.assert_not_called()
