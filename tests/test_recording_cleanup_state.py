"""Regression tests for end-of-recording state surviving a stalled teardown (#249).

Cleanup used to write recording_status last, after the steps that can block for
minutes when the capture stream stops responding. Consumers that read the file
(overlay auto-hide, `record toggle`) then saw a recording that had already been
abandoned - the visualizer stayed on screen and the next toggle sent 'stop'.
The end state has to be published before any step that can block.
"""

import sys
import tempfile
import threading
import types
import unittest
from pathlib import Path
from unittest import mock

from tests.test_suspend_resume_recovery import _import_main_isolated


class RecordingCleanupStateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.main = _import_main_isolated()

    def _app(self, status_path):
        app = self.main.hyprwhsprApp.__new__(self.main.hyprwhsprApp)
        app.playback_suppressor = types.SimpleNamespace(is_active=False, restore=mock.Mock())
        app._recording_lock = threading.Lock()
        app._recording_finalizing = threading.Event()
        patcher = mock.patch.object(self.main, 'RECORDING_STATUS_FILE', status_path)
        patcher.start()
        self.addCleanup(patcher.stop)
        return app

    def test_end_of_recording_is_published_before_blocking_teardown(self):
        with tempfile.TemporaryDirectory() as tmp:
            status = Path(tmp) / 'recording_status'
            status.write_text('true')
            app = self._app(status)

            hidden = []
            observed = {}
            app._hide_mic_osd = mock.Mock(side_effect=lambda: hidden.append(True))
            app._notify_capture = mock.Mock(side_effect=lambda *args, **kwargs: observed.update(
                status_still_set=status.exists(),
                overlay_already_hidden=bool(hidden),
            ))
            app._autostop_stop_silence_monitor = mock.Mock()
            app._clear_mic_osd_preview_text = mock.Mock()
            app._stop_audio_level_monitoring = mock.Mock()

            app._cleanup_recording_state()

            # First potentially blocking step already sees the recording ended.
            self.assertFalse(observed['status_still_set'])
            self.assertTrue(observed['overlay_already_hidden'])
            self.assertFalse(status.exists())

    def test_cleanup_still_restores_playback_and_stops_monitoring(self):
        with tempfile.TemporaryDirectory() as tmp:
            status = Path(tmp) / 'recording_status'
            status.write_text('true')
            app = self._app(status)
            app.playback_suppressor = types.SimpleNamespace(is_active=True, restore=mock.Mock())
            app._hide_mic_osd = mock.Mock()
            app._notify_capture = mock.Mock()
            app._autostop_stop_silence_monitor = mock.Mock()
            app._clear_mic_osd_preview_text = mock.Mock()
            app._stop_audio_level_monitoring = mock.Mock()

            app._cleanup_recording_state()

            app._stop_audio_level_monitoring.assert_called_once()
            app.playback_suppressor.restore.assert_called_once()
