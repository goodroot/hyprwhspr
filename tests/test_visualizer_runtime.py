import hashlib
import io
import json
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib"))
sys.path.insert(0, str(ROOT / "lib" / "src"))

import backend_installer
from mic_osd.runner import MicOSDRunner
from src import visualizer_runtime
from src.cli import mic_osd as mic_osd_cli
import mic_osd.runner as runner_module


class VisualizerRuntimeTests(unittest.TestCase):
    def _make_bundle(self, path, *, unsafe=False):
        manifest = json.dumps({
            "version": visualizer_runtime.GTK4_LAYER_SHELL_VERSION,
            "commit": visualizer_runtime.GTK4_LAYER_SHELL_COMMIT,
        }).encode()
        files = {
            "lib/libgtk4-layer-shell.so.0": b"library",
            "lib/girepository-1.0/Gtk4LayerShell-1.0.typelib": b"typelib",
            "LICENSE": b"MIT",
            "manifest.json": manifest,
        }
        if unsafe:
            files["../escape"] = b"bad"
        with tarfile.open(path, "w:gz") as bundle:
            for name, data in files.items():
                member = tarfile.TarInfo(name)
                member.size = len(data)
                bundle.addfile(member, io.BytesIO(data))

    def test_noble_derivative_x86_64_is_supported(self):
        with tempfile.TemporaryDirectory() as tmp:
            release = Path(tmp) / "os-release"
            release.write_text(
                'ID=tuxedo\nID_LIKE="ubuntu debian"\nUBUNTU_CODENAME=noble\n',
                encoding="utf-8",
            )
            with mock.patch.object(visualizer_runtime.platform, "machine", return_value="x86_64"):
                self.assertTrue(visualizer_runtime.is_noble_x86_64(release))
            with mock.patch.object(visualizer_runtime.platform, "machine", return_value="aarch64"):
                self.assertFalse(visualizer_runtime.is_noble_x86_64(release))

    def test_bundle_environment_is_child_only_and_preserves_existing_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "runtime"
            original = {"GI_TYPELIB_PATH": "/system/typelibs", "LD_PRELOAD": "/old.so"}
            env = visualizer_runtime.environment_for(root, original)
            self.assertEqual(original["GI_TYPELIB_PATH"], "/system/typelibs")
            self.assertTrue(env["GI_TYPELIB_PATH"].startswith(str(root)))
            self.assertIn("/system/typelibs", env["GI_TYPELIB_PATH"])
            self.assertIn(str(root / "lib" / "libgtk4-layer-shell.so.0"), env["LD_PRELOAD"])
            self.assertIn("/old.so", env["LD_PRELOAD"])

    def test_system_runtime_takes_precedence_over_bundle(self):
        with (
            mock.patch.object(MicOSDRunner, "_system_dependencies_available", return_value=True),
            mock.patch.object(MicOSDRunner, "_bundled_dependencies_available") as bundled,
        ):
            self.assertTrue(MicOSDRunner.is_available())
            self.assertEqual(MicOSDRunner.runtime_source(), "system")
        bundled.assert_not_called()

    def test_private_runtime_environment_is_used_for_layer_shell_child(self):
        with (
            mock.patch.object(MicOSDRunner, "_system_dependencies_available", return_value=False),
            mock.patch.object(MicOSDRunner, "_layer_shell_ld_preload", return_value=""),
            mock.patch.object(runner_module.visualizer_runtime, "is_complete", return_value=True),
            mock.patch.object(
                runner_module.visualizer_runtime,
                "bundled_environment",
                return_value={"PRIVATE_RUNTIME": "1"},
            ),
        ):
            self.assertEqual(MicOSDRunner._layer_shell_environment(), {"PRIVATE_RUNTIME": "1"})

    def test_bundled_availability_probe_is_memoized(self):
        completed = mock.Mock(returncode=0)
        with (
            mock.patch.object(MicOSDRunner, "_bundled_availability", None),
            mock.patch.object(runner_module.visualizer_runtime, "is_complete", return_value=True),
            mock.patch.object(runner_module.subprocess, "run", return_value=completed) as run,
        ):
            self.assertTrue(MicOSDRunner._bundled_dependencies_available())
            self.assertTrue(MicOSDRunner._bundled_dependencies_available())
        run.assert_called_once()

    def test_system_runtime_without_glob_does_not_mix_in_bundle(self):
        with (
            mock.patch.object(MicOSDRunner, "_system_dependencies_available", return_value=True),
            mock.patch.object(MicOSDRunner, "_layer_shell_ld_preload", return_value=""),
            mock.patch.object(runner_module.visualizer_runtime, "is_complete", return_value=True),
            mock.patch.object(runner_module.visualizer_runtime, "bundled_environment") as bundled,
            mock.patch.object(runner_module.os, "environ", {"SYSTEM": "1"}),
        ):
            self.assertEqual(MicOSDRunner._layer_shell_environment(), {"SYSTEM": "1"})
        bundled.assert_not_called()

    def test_download_requires_matching_pinned_checksum(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            payload = b"runtime"
            digest = hashlib.sha256(payload).hexdigest()

            class Response(io.BytesIO):
                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    self.close()

            def urlopen(url, timeout):
                self.assertEqual(timeout, 60)
                return Response(payload)

            with (
                mock.patch.object(backend_installer.urllib.request, "urlopen", side_effect=urlopen),
                mock.patch.object(
                    backend_installer.visualizer_runtime,
                    "GTK4_LAYER_SHELL_SHA256",
                    digest,
                ),
            ):
                asset, digest = backend_installer._download_visualizer_runtime(directory)
            self.assertEqual(asset.read_bytes(), payload)
            self.assertEqual(digest, hashlib.sha256(payload).hexdigest())

    def test_download_rejects_checksum_mismatch(self):
        class Response(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *args):
                self.close()

        def urlopen(url, timeout):
            return Response(b"tampered")

        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(backend_installer.urllib.request, "urlopen", side_effect=urlopen), \
                mock.patch.object(
                    backend_installer.visualizer_runtime,
                    "GTK4_LAYER_SHELL_SHA256",
                    "0" * 64,
                ):
            with self.assertRaisesRegex(RuntimeError, "pinned checksum"):
                backend_installer._download_visualizer_runtime(Path(tmp))

    def test_unsafe_archive_is_rejected_without_writing_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "bundle.tar.gz"
            destination = root / "destination"
            self._make_bundle(archive, unsafe=True)
            with self.assertRaisesRegex(RuntimeError, "unsafe layout"):
                backend_installer._extract_visualizer_runtime(archive, destination, "0" * 64)
            self.assertFalse((root / "escape").exists())

    def test_supported_host_installs_bundle_atomically(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            versions = root / "runtime" / "gtk4-layer-shell"
            target = versions / visualizer_runtime.GTK4_LAYER_SHELL_VERSION / "x86_64"
            old_version = versions / "1.2.0" / "x86_64"
            old_version.mkdir(parents=True)
            archive = root / "bundle.tar.gz"
            self._make_bundle(archive)
            with (
                mock.patch.object(backend_installer.visualizer_runtime, "is_noble_x86_64", return_value=True),
                mock.patch.object(backend_installer.visualizer_runtime, "runtime_dir", return_value=target),
                mock.patch.object(backend_installer.visualizer_runtime, "versions_dir", return_value=versions),
                mock.patch.object(backend_installer.visualizer_runtime, "is_complete", return_value=False),
                mock.patch.object(backend_installer, "_visualizer_runtime_imports", side_effect=[False, True]),
                mock.patch.object(
                    backend_installer,
                    "_download_visualizer_runtime",
                    return_value=(archive, "0" * 64),
                ),
            ):
                self.assertTrue(backend_installer.install_gtk4_layer_shell_runtime(Path("python")))
            self.assertTrue((target / "lib" / "libgtk4-layer-shell.so.0").is_file())
            self.assertFalse(any(target.parent.glob(".gtk4-layer-shell-*")))
            self.assertFalse(old_version.exists())

    def test_valid_cached_bundle_is_reused_without_download(self):
        with (
            mock.patch.object(backend_installer, "_visualizer_runtime_imports", side_effect=[False, True]),
            mock.patch.object(backend_installer.visualizer_runtime, "is_noble_x86_64", return_value=True),
            mock.patch.object(backend_installer.visualizer_runtime, "is_complete", return_value=True),
            mock.patch.object(backend_installer, "_download_visualizer_runtime") as download,
        ):
            self.assertTrue(backend_installer.install_gtk4_layer_shell_runtime(Path("python")))
        download.assert_not_called()

    def test_unsupported_host_never_downloads_bundle(self):
        with (
            mock.patch.object(backend_installer, "_visualizer_runtime_imports", return_value=False),
            mock.patch.object(backend_installer.visualizer_runtime, "is_noble_x86_64", return_value=False),
            mock.patch.object(backend_installer, "_download_visualizer_runtime") as download,
        ):
            self.assertFalse(backend_installer.install_gtk4_layer_shell_runtime(Path("python")))
        download.assert_not_called()

    def test_failed_bundle_validation_preserves_existing_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "runtime"
            target.mkdir()
            (target / "old").write_text("keep", encoding="utf-8")
            archive = root / "bundle.tar.gz"
            self._make_bundle(archive)
            with (
                mock.patch.object(backend_installer.visualizer_runtime, "is_noble_x86_64", return_value=True),
                mock.patch.object(backend_installer.visualizer_runtime, "runtime_dir", return_value=target),
                mock.patch.object(backend_installer.visualizer_runtime, "is_complete", return_value=False),
                mock.patch.object(backend_installer, "_visualizer_runtime_imports", side_effect=[False, False]),
                mock.patch.object(
                    backend_installer,
                    "_download_visualizer_runtime",
                    return_value=(archive, "0" * 64),
                ),
            ):
                self.assertFalse(backend_installer.install_gtk4_layer_shell_runtime(Path("python")))
            self.assertEqual((target / "old").read_text(encoding="utf-8"), "keep")

    def test_status_reports_bundled_runtime_source(self):
        config = mock.Mock()
        config.get_setting.return_value = True
        output = io.StringIO()
        with (
            mock.patch.object(mic_osd_cli, "ConfigManager", return_value=config),
            mock.patch.object(
                mic_osd_cli,
                "_query_mic_osd_availability",
                return_value=(True, "", "bundled 1.3.0"),
            ),
            mock.patch("sys.stdout", output),
        ):
            self.assertTrue(mic_osd_cli.mic_osd_status())
        self.assertIn("Layer-shell runtime: bundled 1.3.0", output.getvalue())

    def test_cli_child_computes_runtime_source_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            venv = Path(tmp) / "venv"
            python = venv / "bin" / "python"
            python.parent.mkdir(parents=True)
            python.touch()
            completed = mock.Mock(returncode=0, stdout="AVAILABLE: bundled 1.3.0\n")
            with (
                mock.patch.object(mic_osd_cli, "VENV_DIR", venv),
                mock.patch.object(mic_osd_cli.subprocess, "run", return_value=completed) as run,
            ):
                self.assertEqual(
                    mic_osd_cli._query_mic_osd_availability(),
                    (True, "", "bundled 1.3.0"),
                )
            code = run.call_args.args[0][2]
            self.assertEqual(code.count("runtime_source()"), 1)
            self.assertNotIn("is_available()", code)
            self.assertEqual(run.call_args.kwargs["timeout"], 10)

    def test_missing_visualizer_requirements_keeps_nothing_to_do_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            with (
                mock.patch.object(backend_installer, "HYPRWHSPR_ROOT", tmp),
                mock.patch.object(backend_installer, "_should_skip_pygobject", return_value=False),
                mock.patch.object(
                    backend_installer, "install_gtk4_layer_shell_runtime"
                ) as install_runtime,
            ):
                self.assertTrue(
                    backend_installer.install_visualizer_deps(Path(tmp) / "bin" / "pip")
                )
        install_runtime.assert_called_once()

    def test_post_install_gc_failure_does_not_flip_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            versions = root / "runtime" / "gtk4-layer-shell"
            target = versions / visualizer_runtime.GTK4_LAYER_SHELL_VERSION / "x86_64"
            archive = root / "bundle.tar.gz"
            self._make_bundle(archive)
            with (
                mock.patch.object(backend_installer.visualizer_runtime, "is_noble_x86_64", return_value=True),
                mock.patch.object(backend_installer.visualizer_runtime, "runtime_dir", return_value=target),
                mock.patch.object(backend_installer.visualizer_runtime, "versions_dir", return_value=versions),
                mock.patch.object(backend_installer.visualizer_runtime, "is_complete", return_value=False),
                mock.patch.object(backend_installer, "_visualizer_runtime_imports", side_effect=[False, True]),
                mock.patch.object(
                    backend_installer,
                    "_download_visualizer_runtime",
                    return_value=(archive, "0" * 64),
                ),
                mock.patch.object(backend_installer.Path, "iterdir", side_effect=OSError("gc failed")),
            ):
                self.assertTrue(backend_installer.install_gtk4_layer_shell_runtime(Path("python")))
            self.assertTrue((target / "lib" / "libgtk4-layer-shell.so.0").exists())


if __name__ == "__main__":
    unittest.main()
