"""Recording status, not visualization feed freshness, governs auto-hide.

Feed publishing can fail while capture continues. A lost or stale waveform
must not hide an active recording; clearing recording status must hide it
on the next auto-hide callback even when the feed remains fresh.
"""

import importlib.util
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib"))


class MicOSDStuckOverlayTests(unittest.TestCase):
    def setUp(self):
        runtime = tempfile.TemporaryDirectory()
        self.addCleanup(runtime.cleanup)
        runtime_dir = Path(runtime.name)
        self.status = runtime_dir / "recording_status"
        self.feed = runtime_dir / "mic_osd_level_feed"
        self.preview = runtime_dir / "transcript_preview"

        # Load isolated modules with only hardware/GUI dependencies stubbed.
        # Exercise the real feed reader and OSD lifecycle without GTK or audio.
        audio_spec = importlib.util.spec_from_file_location(
            "_stuck_overlay_audio", ROOT / "lib" / "mic_osd" / "audio.py"
        )
        audio_module = importlib.util.module_from_spec(audio_spec)
        with mock.patch.dict(sys.modules, {"sounddevice": mock.Mock()}):
            audio_spec.loader.exec_module(audio_module)
        audio_module.AudioMonitor = mock.Mock(
            side_effect=AssertionError("A fresh controller feed must not open a microphone")
        )
        glib = types.SimpleNamespace(
            timeout_add=mock.Mock(return_value=1),
            timeout_add_seconds=mock.Mock(return_value=7),
            source_remove=mock.Mock(),
        )
        stubs = {
            "gi": types.SimpleNamespace(require_version=lambda *args: None),
            "gi.repository": types.SimpleNamespace(Gtk=types.SimpleNamespace(), GLib=glib),
            "mic_osd.window": types.SimpleNamespace(OSDWindow=mock.Mock(), load_css=mock.Mock()),
            "mic_osd.audio": audio_module,
            "mic_osd.visualizations": types.SimpleNamespace(VISUALIZATIONS={"waveform": object}),
            "mic_osd.theme": types.SimpleNamespace(ThemeWatcher=mock.Mock()),
        }
        main_spec = importlib.util.spec_from_file_location(
            "mic_osd._stuck_overlay_main", ROOT / "lib" / "mic_osd" / "main.py"
        )
        self.main = importlib.util.module_from_spec(main_spec)
        with mock.patch.dict(sys.modules, stubs):
            main_spec.loader.exec_module(self.main)
        self.main.RECORDING_STATUS_FILE = self.status
        self.main.MIC_OSD_LEVEL_FEED_FILE = self.feed
        self.main.TRANSCRIPT_PREVIEW_FILE = self.preview
        self.rearm = glib.timeout_add_seconds

        # Fix the reader's clock so feed freshness never depends on test speed.
        clock = mock.patch.object(audio_module.time, "time", return_value=1000.0)
        clock.start()
        self.addCleanup(clock.stop)

    def _show_recording(self):
        self.status.write_text("true")
        self.feed.write_text("0.5 0.25 0.5")
        os.utime(self.feed, (1000.0, 1000.0))
        app = self.main.MicOSD(daemon=True)
        app.window = mock.Mock()
        app._show()
        app.window.reset_mock()
        self.rearm.reset_mock()
        return app

    def _assert_recording_stays_visible(self, app):
        self.assertFalse(app._auto_hide_callback())
        self.assertTrue(app.visible)
        app.window.set_visible.assert_not_called()
        self.rearm.assert_called_once()

    def test_feed_stalling_during_recording_does_not_hide_overlay(self):
        app = self._show_recording()
        os.utime(self.feed, (0.0, 0.0))

        self._assert_recording_stays_visible(app)

    def test_feed_disappearing_during_recording_does_not_hide_overlay(self):
        app = self._show_recording()
        self.feed.unlink()

        self._assert_recording_stays_visible(app)

    def test_inactive_recording_hides_overlay_even_with_fresh_feed(self):
        for status in ("false", None):
            with self.subTest(status=status):
                app = self._show_recording()
                if status is None:
                    self.status.unlink()
                else:
                    self.status.write_text(status)

                self.assertFalse(app._auto_hide_callback())

                self.assertFalse(app.visible)
                app.window.set_visible.assert_called_once_with(False)
                self.rearm.assert_not_called()


if __name__ == "__main__":
    unittest.main()
