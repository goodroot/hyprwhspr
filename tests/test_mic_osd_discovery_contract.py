"""Mic-OSD discovery contract: dependencies, session support, backend choice.

Real service, runner and CLI code runs against a fake host. GTK probes are
answered by what they ask for, so these tests outlive refactors of where the
probes live.
"""

import os
import sys
import tempfile
import types
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib"))

from tests.test_suspend_resume_recovery import _import_main_isolated, patch_app_global
import mic_osd.runner as runner_module
from mic_osd.notification_presenter import NotificationPresenter
from mic_osd.runner import MicOSDRunner
from src import visualizer_runtime
from src.cli import mic_osd as mic_osd_cli

SYSTEM_LIBRARY = "/usr/lib/libgtk4-layer-shell.so.1.3.0"
WAYLAND = {"WAYLAND_DISPLAY": "wayland-1"}
XWAYLAND = {"WAYLAND_DISPLAY": "wayland-1", "DISPLAY": ":0"}
X11 = {"DISPLAY": ":0"}


class FakeHost:
    """Answers GTK probes from declared facts and records which ran."""

    def __init__(self, env, *, cairo=True, gtk4=True, layer_shell=True, gtk3=False,
                 bundled=False, compositor=False, libraries=(SYSTEM_LIBRARY,),
                 daemon_survives=True):
        self.env = env
        self.cairo = cairo
        self.gtk4 = gtk4
        self.layer_shell = layer_shell
        self.gtk3 = gtk3
        self.bundled = bundled
        self.compositor = compositor
        self.libraries = libraries
        self.daemon_survives = daemon_survives
        self.probes = []
        self.popen = mock.Mock(side_effect=self._popen)

    def _gtk4(self, env):
        bundled = (env or {}).get("FAKE_BUNDLED") == "1" and self.bundled
        return self.cairo and self.gtk4 and (self.layer_shell or bundled)

    def run(self, argv, **kwargs):
        code = argv[2]
        env = kwargs.get("env")
        if "'Gtk', '3.0'" in code:
            kind, ok = "gtk3", self.cairo and self.gtk3
            out = "1" if ok else ""
        elif "is_supported" in code:
            kind, ok = "support", self._gtk4(env)
            out = ("1" if self.compositor else "0") if ok else ""
        elif "/proc/self/maps" in code:
            kind, ok = "library", self.gtk4 and self.layer_shell
            out = "\n".join(self.libraries) if ok else ""
        elif "'layer-shell'" in code:
            kind, ok = "system", True
            out = ("cairo" if not self.cairo else "gtk4" if not self.gtk4
                   else "layer-shell" if not self.layer_shell else "ok")
        elif "Gtk4LayerShell" in code:
            kind = "bundled" if (env or {}).get("FAKE_BUNDLED") == "1" else "system"
            ok = self._gtk4(env)
            out = ""
        else:
            raise AssertionError(f"unexpected subprocess: {argv!r}")
        self.probes.append(kind)
        returncode = 0 if ok else 1
        if not kwargs.get("text"):
            return types.SimpleNamespace(returncode=returncode, stdout=out.encode(), stderr=b"")
        return types.SimpleNamespace(returncode=returncode, stdout=out, stderr="")

    def _popen(self, argv, **kwargs):
        process = mock.Mock(pid=4242, returncode=1)
        process.poll.return_value = None if self.daemon_survives else 1
        process.stderr.read.return_value = b"boom"
        return process

    def _gi(self):
        def require_version(namespace, version):
            missing = {"Gtk": not self.gtk4, "Gtk4LayerShell": not self.layer_shell}
            if missing.get(namespace) or version == "3.0":
                raise ValueError(f"Namespace {namespace} not available")
        return types.SimpleNamespace(require_version=require_version)

    def install(self, stack):
        """Patch the process so discovery sees this host."""
        tmp = stack.enter_context(tempfile.TemporaryDirectory())
        base = {"PATH": os.environ.get("PATH", ""), "HOME": tmp}
        stack.enter_context(mock.patch.dict(os.environ, {**base, **self.env}, clear=True))
        stack.enter_context(mock.patch.dict(sys.modules, {
            "gi": self._gi(), "cairo": types.SimpleNamespace() if self.cairo else None}))
        stack.enter_context(mock.patch("subprocess.run", side_effect=self.run))
        stack.enter_context(mock.patch("subprocess.Popen", self.popen))
        stack.enter_context(mock.patch("time.sleep"))
        stack.enter_context(mock.patch("builtins.print"))
        stack.enter_context(mock.patch.object(
            runner_module, "MIC_OSD_PID_FILE", Path(tmp) / "mic_osd.pid"))
        stack.enter_context(mock.patch.object(
            visualizer_runtime, "is_complete", return_value=self.bundled))
        stack.enter_context(mock.patch.object(
            visualizer_runtime, "bundled_environment",
            side_effect=lambda env=None: {**(os.environ if env is None else env),
                                          "FAKE_BUNDLED": "1"}))
        stack.enter_context(mock.patch.object(
            mic_osd_cli, "VENV_DIR", Path(tmp) / "no-venv"))
        for cache in ("_bundled_availability", "_x11_availability", "_layer_shell_preload",
                      "_system_availability"):
            if hasattr(MicOSDRunner, cache):
                stack.enter_context(mock.patch.object(MicOSDRunner, cache, None))
        return self

    @property
    def daemon_env(self):
        daemons = [c for c in self.popen.call_args_list if c.args[0] != ["true"]]
        if not daemons:
            return None
        assert len(daemons) == 1, daemons
        return daemons[0].kwargs["env"]


class ServiceSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app_class = _import_main_isolated().hyprwhsprApp

    def _start(self, host, *, enabled=True):
        app = object.__new__(self.app_class)
        settings = {"mic_osd_enabled": enabled, "mic_osd_style": "pill"}
        app.config = mock.Mock()
        app.config.get_setting.side_effect = lambda key, default=None: settings.get(key, default)
        app.audio_capture = types.SimpleNamespace(get_viz_frame=mock.Mock())
        with ExitStack() as stack:
            host.install(stack)
            stack.enter_context(mock.patch.object(
                NotificationPresenter, "is_available", return_value=True))
            stack.enter_context(patch_app_global(self.app_class, "log", mock.Mock()))
            stack.enter_context(patch_app_global(
                self.app_class, "_looks_like_wlroots_session", lambda: False))
            app._init_mic_osd()
        return app._mic_osd_runner

    def assertOverlay(self, selected, host, backend):
        self.assertIsInstance(selected, MicOSDRunner)
        self.assertEqual(host.daemon_env["HYPRWHSPR_MIC_OSD_BACKEND"], backend)

    def assertNotifications(self, selected, host):
        self.assertIsInstance(selected, NotificationPresenter)
        self.assertIsNone(host.daemon_env)

    def test_supported_wayland_starts_gtk4_with_the_typelib_library(self):
        host = FakeHost(WAYLAND, compositor=True)
        self.assertOverlay(self._start(host), host, "gtk4")
        self.assertEqual(host.daemon_env["LD_PRELOAD"], SYSTEM_LIBRARY)
        self.assertNotIn("gtk3", host.probes)

    def test_installed_dependencies_do_not_imply_compositor_support(self):
        host = FakeHost(WAYLAND, compositor=False)
        self.assertNotifications(self._start(host), host)
        self.assertIn("support", host.probes)

    def test_xwayland_alone_never_selects_gtk3(self):
        host = FakeHost(XWAYLAND, gtk3=True, compositor=False)
        self.assertNotifications(self._start(host), host)
        self.assertNotIn("gtk3", host.probes)

    def test_x11_selects_gtk3_without_layer_shell_preload(self):
        host = FakeHost(X11, gtk3=True, gtk4=False, layer_shell=False)
        self.assertOverlay(self._start(host), host, "gtk3")
        self.assertNotIn("LD_PRELOAD", host.daemon_env)

    def test_x11_without_gtk3_uses_notifications(self):
        host = FakeHost(X11, gtk3=False)
        self.assertNotifications(self._start(host), host)

    def test_system_runtime_precedes_the_bundle(self):
        host = FakeHost(WAYLAND, compositor=True, bundled=True)
        self.assertOverlay(self._start(host), host, "gtk4")
        self.assertNotIn("FAKE_BUNDLED", host.daemon_env)
        self.assertNotIn("bundled", host.probes)

    def test_bundle_serves_hosts_without_system_layer_shell(self):
        host = FakeHost(WAYLAND, layer_shell=False, bundled=True, compositor=True)
        self.assertOverlay(self._start(host), host, "gtk4")
        self.assertEqual(host.daemon_env["FAKE_BUNDLED"], "1")
        self.assertNotIn("library", host.probes)

    def test_ambiguous_layer_shell_library_falls_back_to_notifications(self):
        host = FakeHost(WAYLAND, compositor=True,
                        libraries=("/a/libgtk4-layer-shell.so", "/b/libgtk4-layer-shell.so.0"))
        self.assertNotifications(self._start(host), host)

    def test_failed_daemon_start_falls_back_to_notifications(self):
        host = FakeHost(WAYLAND, compositor=True, daemon_survives=False)
        self.assertIsInstance(self._start(host), NotificationPresenter)
        self.assertEqual(host.daemon_env["HYPRWHSPR_MIC_OSD_BACKEND"], "gtk4")

    def test_missing_dependencies_use_notifications(self):
        host = FakeHost(WAYLAND, layer_shell=False, compositor=True)
        self.assertNotifications(self._start(host), host)

    def test_disabled_osd_performs_no_discovery(self):
        host = FakeHost(WAYLAND, compositor=True)
        self.assertIsNone(self._start(host, enabled=False))
        self.assertEqual(host.probes, [])
        host.popen.assert_not_called()


class CliDependencyTests(unittest.TestCase):
    def _query(self, host):
        with ExitStack() as stack:
            host.install(stack)
            return mic_osd_cli._query_mic_osd_availability()

    def test_cli_checks_dependencies_not_the_compositor(self):
        host = FakeHost(WAYLAND, compositor=False)
        self.assertEqual(self._query(host), (True, "", "system"))
        self.assertNotIn("support", host.probes)

    def test_enable_succeeds_where_the_compositor_lacks_layer_shell(self):
        host = FakeHost(WAYLAND, compositor=False)
        config = mock.Mock()
        with ExitStack() as stack:
            host.install(stack)
            stack.enter_context(mock.patch.object(mic_osd_cli, "ConfigManager", return_value=config))
            stack.enter_context(mock.patch.object(mic_osd_cli, "log_success"))
            stack.enter_context(mock.patch.object(mic_osd_cli, "log_info"))
            self.assertTrue(mic_osd_cli.mic_osd_enable())
        config.set_setting.assert_called_once_with("mic_osd_enabled", True)

    def test_enable_refuses_missing_dependencies(self):
        host = FakeHost(WAYLAND, layer_shell=False)
        config = mock.Mock()
        with ExitStack() as stack:
            host.install(stack)
            stack.enter_context(mock.patch.object(mic_osd_cli, "ConfigManager", return_value=config))
            error = stack.enter_context(mock.patch.object(mic_osd_cli, "log_error"))
            self.assertFalse(mic_osd_cli.mic_osd_enable())
        config.set_setting.assert_not_called()
        self.assertIn("gtk4-layer-shell", error.call_args.args[0])

    def test_system_runtime_reported_ahead_of_the_bundle(self):
        host = FakeHost(WAYLAND, bundled=True)
        self.assertEqual(self._query(host), (True, "", "system"))
        self.assertNotIn("bundled", host.probes)

    def test_bundled_runtime_reported_without_system_layer_shell(self):
        host = FakeHost(WAYLAND, layer_shell=False, bundled=True)
        source = f"bundled {visualizer_runtime.GTK4_LAYER_SHELL_VERSION}"
        self.assertEqual(self._query(host), (True, "", source))

    def test_x11_reports_gtk3(self):
        host = FakeHost(X11, gtk3=True, gtk4=False, layer_shell=False)
        self.assertEqual(self._query(host), (True, "", "X11 GTK3"))

    def test_x11_without_gtk3_names_gtk3(self):
        host = FakeHost(X11, gtk4=False, layer_shell=False)
        available, reason, source = self._query(host)
        self.assertFalse(available)
        self.assertIn("GTK3", reason)
        self.assertEqual(source, "unavailable")

    def test_xwayland_reports_layer_shell_dependencies(self):
        host = FakeHost(XWAYLAND, gtk3=True, layer_shell=False)
        available, reason, _source = self._query(host)
        self.assertFalse(available)
        self.assertIn("gtk4-layer-shell", reason)
        self.assertNotIn("gtk3", host.probes)

    def test_missing_cairo_is_named(self):
        host = FakeHost(WAYLAND, cairo=False)
        available, reason, _source = self._query(host)
        self.assertFalse(available)
        self.assertIn("PyCairo", reason)


if __name__ == "__main__":
    unittest.main()
