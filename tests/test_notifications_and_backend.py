import sys
import tempfile
import types
import unittest
import importlib
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib"))
sys.path.insert(0, str(ROOT / "lib" / "src"))

import desktop_notify  # noqa: E402
from mic_osd.notification_presenter import NotificationPresenter  # noqa: E402
import backend_installer  # noqa: E402


class NotificationCompatibilityTests(unittest.TestCase):
    def test_notification_presenter_imports_from_lib_only_path(self):
        module_names = ["mic_osd.notification_presenter", "desktop_notify"]
        saved_modules = {
            name: sys.modules.pop(name)
            for name in module_names
            if name in sys.modules
        }
        lib_path = str(ROOT / "lib")
        stripped_path = [
            path for path in sys.path
            if path != str(ROOT / "lib" / "src") and path != ""
        ]
        try:
            with mock.patch.object(sys, "path", [lib_path] + stripped_path):
                module = importlib.import_module("mic_osd.notification_presenter")
            self.assertTrue(hasattr(module, "NotificationPresenter"))
        finally:
            for name in module_names:
                sys.modules.pop(name, None)
            sys.modules.update(saved_modules)

    def test_presenter_preserves_zero_active_timeout(self):
        presenter = NotificationPresenter(active_timeout_ms=0)
        self.assertEqual(presenter._active_timeout_ms, 0)

    def test_send_notification_falls_back_to_gdbus_when_notify_send_has_no_id(self):
        def which(cmd):
            return f"/usr/bin/{cmd}" if cmd in {"notify-send", "gdbus"} else None

        results = [
            types.SimpleNamespace(returncode=0, stdout=b"", stderr=b""),
            types.SimpleNamespace(returncode=0, stdout=b"(uint32 42,)", stderr=b""),
        ]

        with (
            mock.patch("desktop_notify.shutil.which", side_effect=which),
            mock.patch("desktop_notify.subprocess.run", side_effect=results) as run,
        ):
            nid = desktop_notify.send_notification_with_id(
                "hyprwhspr", "Recording", timeout_ms=5000)

        self.assertEqual(nid, 42)
        self.assertEqual(run.call_count, 2)
        self.assertEqual(run.call_args_list[0].args[0][0], "notify-send")
        self.assertEqual(run.call_args_list[1].args[0][0], "gdbus")


class BackendInstallerStateTests(unittest.TestCase):
    def test_dependency_manifests_are_backend_and_provider_specific(self):
        with mock.patch.object(backend_installer, "HYPRWHSPR_ROOT", str(ROOT)):
            cpu = backend_installer.dependency_manifests("cpu")
            rest = backend_installer.dependency_manifests("rest-api", "openai")
            realtime = backend_installer.dependency_manifests("realtime-ws", "openai")
            eleven = backend_installer.dependency_manifests("realtime-ws", "elevenlabs")

        self.assertEqual(cpu[-1].name, "requirements-pywhispercpp.txt")
        self.assertEqual(rest[-1].name, "requirements-rest.txt")
        self.assertEqual(realtime[-1].name, "requirements-realtime.txt")
        self.assertEqual(eleven[-1].name, "requirements-realtime-elevenlabs.txt")

    def test_missing_installed_backend_with_matching_hash_does_not_reinstall(self):
        state = {"dependency_manifest_hash": "same-hash"}

        def get_state(key):
            return state.get(key, "")

        def set_state(key, value):
            state[key] = value

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            venv_dir = tmp_path / "venv"
            (venv_dir / "bin").mkdir(parents=True)
            pip_bin = venv_dir / "bin" / "pip"
            pip_bin.touch()

            completed = types.SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

            with (
                mock.patch.object(backend_installer, "install_system_dependencies") as install_system,
                mock.patch.object(backend_installer, "_check_mise_active", return_value=False),
                mock.patch.object(backend_installer, "setup_python_venv", return_value=pip_bin),
                mock.patch.object(backend_installer, "VENV_DIR", venv_dir),
                mock.patch.object(backend_installer, "HYPRWHSPR_ROOT", str(ROOT)),
                mock.patch.object(backend_installer, "dependency_manifest_hash", return_value="same-hash"),
                mock.patch.object(backend_installer, "get_state", side_effect=get_state),
                mock.patch.object(backend_installer, "set_state", side_effect=set_state),
                mock.patch.object(backend_installer, "commit_dependency_state") as commit_state,
                mock.patch.object(backend_installer, "set_install_state"),
                mock.patch.object(backend_installer, "run_command", return_value=completed),
                mock.patch.object(backend_installer, "download_pywhispercpp_model", return_value=True),
                mock.patch.object(backend_installer, "install_pywhispercpp_cpu", return_value=False) as install_cpu,
            ):
                self.assertTrue(backend_installer.install_backend("cpu"))

        install_cpu.assert_not_called()
        install_system.assert_not_called()
        committed_plan = commit_state.call_args.args[0]
        self.assertEqual(committed_plan.accelerated_variant, "cpu")


if __name__ == "__main__":
    unittest.main()
