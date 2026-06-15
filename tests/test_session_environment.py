import os
import stat as stat_module
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib" / "src"))

from session_environment import ensure_wayland_display


class SessionEnvironmentTests(unittest.TestCase):
    def _socket_file_at(self, path, mtime=100):
        path.write_text("", encoding="utf-8")
        path.chmod(0o600)
        os.utime(path, (mtime, mtime))
        return path

    def _non_socket_file_at(self, path, mtime=100):
        path.write_text("", encoding="utf-8")
        path.chmod(0o644)
        os.utime(path, (mtime, mtime))
        return path

    def _socket_mode_patch(self):
        # The sandbox can deny AF_UNIX bind() in temp dirs, so tests mark fake
        # sockets with a distinct mode while production uses the real stat check.
        return mock.patch(
            "session_environment.stat.S_ISSOCK",
            side_effect=lambda mode: stat_module.S_IMODE(mode) == 0o600,
        )

    def test_does_nothing_when_wayland_display_already_set(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._socket_file_at(Path(tmpdir) / "wayland-1")
            with (
                mock.patch.dict(
                    os.environ,
                    {"WAYLAND_DISPLAY": "wayland-existing", "XDG_RUNTIME_DIR": tmpdir},
                    clear=True,
                ),
                mock.patch("builtins.print") as print_mock,
                self._socket_mode_patch(),
            ):
                ensure_wayland_display()
                self.assertEqual(os.environ.get("WAYLAND_DISPLAY"), "wayland-existing")

        print_mock.assert_not_called()

    def test_sets_wayland_display_when_one_socket_exists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._socket_file_at(Path(tmpdir) / "wayland-1")
            with (
                mock.patch.dict(os.environ, {"XDG_RUNTIME_DIR": tmpdir}, clear=True),
                self._socket_mode_patch(),
            ):
                ensure_wayland_display()
                self.assertEqual(os.environ.get("WAYLAND_DISPLAY"), "wayland-1")

    def test_chooses_newest_socket_when_multiple_exist(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime_dir = Path(tmpdir)
            self._socket_file_at(runtime_dir / "wayland-0", mtime=100)
            self._socket_file_at(runtime_dir / "wayland-1", mtime=200)

            with (
                mock.patch.dict(os.environ, {"XDG_RUNTIME_DIR": tmpdir}, clear=True),
                self._socket_mode_patch(),
            ):
                ensure_wayland_display()
                self.assertEqual(os.environ.get("WAYLAND_DISPLAY"), "wayland-1")

    def test_ignores_non_socket_files_matching_wayland_pattern(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime_dir = Path(tmpdir)
            self._non_socket_file_at(runtime_dir / "wayland-9", mtime=300)
            self._socket_file_at(runtime_dir / "wayland-1", mtime=100)

            with (
                mock.patch.dict(os.environ, {"XDG_RUNTIME_DIR": tmpdir}, clear=True),
                self._socket_mode_patch(),
            ):
                ensure_wayland_display()
                self.assertEqual(os.environ.get("WAYLAND_DISPLAY"), "wayland-1")

    def test_does_nothing_when_runtime_dir_has_no_sockets(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime_dir = Path(tmpdir)
            self._non_socket_file_at(runtime_dir / "wayland-0")

            with (
                mock.patch.dict(os.environ, {"XDG_RUNTIME_DIR": tmpdir}, clear=True),
                self._socket_mode_patch(),
            ):
                ensure_wayland_display()
                self.assertIsNone(os.environ.get("WAYLAND_DISPLAY"))

    def test_does_not_raise_when_xdg_runtime_dir_unset_or_missing(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            ensure_wayland_display()
            self.assertIsNone(os.environ.get("WAYLAND_DISPLAY"))

        with mock.patch.dict(os.environ, {"XDG_RUNTIME_DIR": "/path/that/does/not/exist"}, clear=True):
            ensure_wayland_display()
            self.assertIsNone(os.environ.get("WAYLAND_DISPLAY"))


if __name__ == "__main__":
    unittest.main()
