"""mic_osd.runtime: probes, library resolution, child environments, diagnostics."""

import os
import subprocess
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib"))

import mic_osd.runtime as runtime


def completed(stdout="", stderr="", returncode=0):
    return types.SimpleNamespace(stdout=stdout, stderr=stderr, returncode=returncode)


class SystemProbeTests(unittest.TestCase):
    def test_reports_the_first_missing_piece(self):
        for status in ("ok", "cairo", "gtk4", "layer-shell"):
            with self.subTest(status), \
                    mock.patch.object(runtime.subprocess, "run",
                                      return_value=completed(f"{status}\n")) as run:
                self.assertEqual(runtime.system_status(), (status, ""))
            self.assertEqual(run.call_args.kwargs["timeout"], runtime.PROBE_TIMEOUT)

    def test_crashed_probe_keeps_its_last_error_line(self):
        crash = completed(stderr="Traceback\nSegmentation fault\n", returncode=-11)
        with mock.patch.object(runtime.subprocess, "run", return_value=crash):
            self.assertEqual(runtime.system_status(), (None, "Segmentation fault"))

    def test_timed_out_probe_is_explained(self):
        timeout = subprocess.TimeoutExpired("python3", runtime.PROBE_TIMEOUT)
        with mock.patch.object(runtime.subprocess, "run", side_effect=timeout):
            status, detail = runtime.system_status()
        self.assertIsNone(status)
        self.assertIn("timed out", detail)

    def test_probe_failure_reason_carries_the_detail(self):
        reason = runtime.gtk4_reason(None, "Segmentation fault", ("g", "l"))
        self.assertEqual(reason, "GTK4 dependency check failed: Segmentation fault")

    def test_reasons_name_packages(self):
        packages = ("gtk-pkg", "layer-pkg")
        self.assertEqual(runtime.gtk4_reason("cairo", packages=packages),
                         "PyCairo bindings not installed. Install: gtk-pkg")
        self.assertEqual(runtime.gtk4_reason("gtk4", packages=packages),
                         "GTK4 bindings not installed. Install: gtk-pkg")
        self.assertEqual(runtime.gtk4_reason("layer-shell", packages=packages),
                         "gtk4-layer-shell not installed. Install: layer-pkg")


class EnvironmentTests(unittest.TestCase):
    def test_system_environment_is_fresh_and_preloads_first(self):
        base = {"LD_PRELOAD": "/usr/lib/libgtk4-layer-shell.so:/old.so"}
        env = runtime.gtk4_environment(runtime.RUNTIME_SYSTEM, "/lib/libgtk4-layer-shell.so.1", base)
        self.assertEqual(env["LD_PRELOAD"], "/lib/libgtk4-layer-shell.so.1 /old.so")
        self.assertEqual(base["LD_PRELOAD"], "/usr/lib/libgtk4-layer-shell.so:/old.so")
        self.assertIsNot(env, base)

    def test_default_base_copies_os_environ(self):
        with mock.patch.dict(os.environ, {"KEEP": "1"}):
            env = runtime.gtk4_environment(runtime.RUNTIME_SYSTEM, "/lib/libgtk4-layer-shell.so.1")
            self.assertNotIn("libgtk4-layer-shell", os.environ.get("LD_PRELOAD", ""))
        self.assertEqual(env["KEEP"], "1")

    def test_library_probe_strips_inherited_layer_shell_preloads(self):
        with mock.patch.object(runtime.subprocess, "run",
                               return_value=completed("/lib/libgtk4-layer-shell.so.1\n")) as run:
            library = runtime.mapped_layer_shell_library(
                {"LD_PRELOAD": "/x/libgtk4-layer-shell.so /old.so"})
        self.assertEqual(library, "/lib/libgtk4-layer-shell.so.1")
        self.assertEqual(run.call_args.kwargs["env"]["LD_PRELOAD"], "/old.so")
        self.assertEqual(run.call_args.kwargs["timeout"], runtime.PROBE_TIMEOUT)


class ProcessIsolationTests(unittest.TestCase):
    def test_discovery_modules_import_without_gtk(self):
        code = (
            "import sys; sys.path.insert(0, sys.argv[1]);"
            "import mic_osd, mic_osd.runtime, mic_osd.runner;"
            "bound = [m for m in ('gi', 'cairo', 'gi.repository.Gtk') if m in sys.modules];"
            "print(','.join(bound))"
        )
        result = subprocess.run([sys.executable, "-c", code, str(ROOT / "lib")],
                                capture_output=True, text=True, timeout=30,
                                env={"PATH": os.environ.get("PATH", "")})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "")


if __name__ == "__main__":
    unittest.main()
