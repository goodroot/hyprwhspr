"""The OSD theme poll runs only while visible and catches up on show."""
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib"))

from mic_osd import theme  # noqa: E402


class ThemeWatcherPauseTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.css = Path(tmp.name) / "mic-osd.css"
        self.css.write_text("a{}")
        os.utime(self.css, (1, 1))
        self.glib = types.SimpleNamespace(timeout_add=mock.Mock(side_effect=[11, 12, 13]),
                                          source_remove=mock.Mock())
        stubs = {"gi": types.SimpleNamespace(),
                 "gi.repository": types.SimpleNamespace(GLib=self.glib)}
        for patcher in (mock.patch.dict(sys.modules, stubs),
                        mock.patch.object(theme, "SHELL_THEME_CSS", self.css)):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.watcher = theme.ThemeWatcher()
        self.watcher._theme_link = Path(tmp.name) / "no-omarchy"
        self.watcher._reload_theme = mock.Mock()

    def test_change_while_paused_is_applied_on_resume(self):
        self.watcher.start()
        self.watcher.stop()
        self.glib.source_remove.assert_called_once_with(11)
        os.utime(self.css, (2, 2))

        self.watcher.resume()

        self.watcher._reload_theme.assert_called_once()
        self.assertEqual(self.watcher._timer_id, 12)

    def test_resume_while_polling_adds_no_second_timer(self):
        self.watcher.start()
        self.watcher.resume()
        self.glib.timeout_add.assert_called_once()
        self.watcher._reload_theme.assert_not_called()


if __name__ == "__main__":
    unittest.main()
