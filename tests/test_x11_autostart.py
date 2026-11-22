"""X11 desktop sessions never activate graphical-session.target.

The systemd user unit is therefore also started from an autostart entry that
runs inside the session, where DISPLAY and XAUTHORITY exist.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'lib' / 'src'))

from cli import systemd  # noqa: E402


class X11AutostartTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.home = Path(temp.name)
        env = {'XDG_CONFIG_HOME': str(self.home / 'config'),
               'XDG_SESSION_TYPE': 'x11'}
        patch = mock.patch.dict(os.environ, env)
        patch.start()
        self.addCleanup(patch.stop)
        self.session = mock.patch.object(systemd, '_graphical_session_active',
                                         return_value=False)
        self.session.start()
        self.addCleanup(self.session.stop)
        self.path = systemd.x11_autostart_path()

    def test_x11_writes_an_entry_that_starts_the_service(self):
        self.assertTrue(systemd.install_x11_autostart())
        text = self.path.read_text(encoding='utf-8')
        self.assertIn('[Desktop Entry]', text)
        self.assertIn('systemctl --user start hyprwhspr.service', text)
        self.assertIn('import-environment DISPLAY XAUTHORITY', text)
        self.assertIn(systemd.AUTOSTART_MARKER, text)

    def test_rerun_is_idempotent(self):
        self.assertTrue(systemd.install_x11_autostart())
        once = self.path.read_text(encoding='utf-8')
        self.assertTrue(systemd.install_x11_autostart())
        self.assertEqual(self.path.read_text(encoding='utf-8'), once)

    def test_wayland_sessions_use_graphical_session_target(self):
        with mock.patch.dict(os.environ, {'XDG_SESSION_TYPE': 'wayland'}):
            self.assertFalse(systemd.install_x11_autostart())
        self.assertFalse(self.path.exists())

    def test_active_graphical_session_target_needs_no_entry(self):
        with mock.patch.object(systemd, '_graphical_session_active', return_value=True):
            self.assertFalse(systemd.install_x11_autostart())
        self.assertFalse(self.path.exists())

    def test_disable_removes_only_the_managed_entry(self):
        self.assertTrue(systemd.install_x11_autostart())
        self.assertTrue(systemd.remove_x11_autostart())
        self.assertFalse(self.path.exists())

    def test_remove_leaves_a_foreign_file_alone(self):
        self.path.parent.mkdir(parents=True)
        self.path.write_text('[Desktop Entry]\nName=mine\n', encoding='utf-8')
        self.assertFalse(systemd.remove_x11_autostart())
        self.assertTrue(self.path.exists())

    def test_a_foreign_autostart_file_is_left_untouched(self):
        self.path.parent.mkdir(parents=True)
        self.path.write_text('[Desktop Entry]\nName=mine\n', encoding='utf-8')
        self.assertFalse(systemd.install_x11_autostart())
        self.assertEqual(self.path.read_text(encoding='utf-8'),
                         '[Desktop Entry]\nName=mine\n')


if __name__ == '__main__':
    unittest.main()
