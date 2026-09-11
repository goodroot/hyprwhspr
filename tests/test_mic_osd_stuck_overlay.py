"""Regression tests for the overlay outliving the controller's capture (#249).

recording_status is only written when a recording starts or ends, while the
level feed is rewritten every tick. A controller that stalls in teardown (a
capture stream that stopped responding is the usual cause) therefore leaves
recording_status at 'true' indefinitely, and the auto-hide - which only ever
trusted that file - kept re-arming until the overlay was pinned on screen for
good. A feed that stopped advancing proves the capture is gone.
"""

import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib"))

try:
    from mic_osd import main as mic_osd_main
except ImportError as exc:  # GTK/layer-shell are optional; CI has neither
    mic_osd_main = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = mic_osd_main._MIC_OSD_IMPORT_ERROR


@unittest.skipIf(mic_osd_main is None or IMPORT_ERROR is not None,
                 "mic-osd GUI stack unavailable")
class MicOSDStuckOverlayTests(unittest.TestCase):
    def setUp(self):
        runtime = tempfile.TemporaryDirectory()
        self.addCleanup(runtime.cleanup)
        runtime_dir = Path(runtime.name)
        self.status = runtime_dir / 'recording_status'
        self.feed = runtime_dir / 'mic_osd_level_feed'
        self.preview = runtime_dir / 'transcript_preview'
        for name, path in (
            ('RECORDING_STATUS_FILE', self.status),
            ('MIC_OSD_LEVEL_FEED_FILE', self.feed),
            ('TRANSCRIPT_PREVIEW_FILE', self.preview),
        ):
            patcher = mock.patch.object(mic_osd_main, name, path)
            patcher.start()
            self.addCleanup(patcher.stop)

        # Timer arming is observed rather than scheduled: no main loop runs here.
        timer = mock.patch.object(mic_osd_main.GLib, 'timeout_add_seconds', return_value=7)
        self.rearm = timer.start()
        self.addCleanup(timer.stop)

    def _visible_app(self, feed_liveness=True):
        app = mic_osd_main.MicOSD.__new__(mic_osd_main.MicOSD)
        app.visible = True
        app.window = None  # _hide() returns before it touches GTK
        app.audio_monitor = None
        app._auto_hide_timeout_id = None
        app._last_preview_text = None
        app._feed_liveness = feed_liveness
        return app

    def _write_feed(self, age_seconds=0.0):
        self.feed.write_text('0.0 0.0 0.0')
        stamp = time.time() - age_seconds
        os.utime(self.feed, (stamp, stamp))

    def test_dead_feed_hides_overlay_despite_stale_status(self):
        self.status.write_text('true')
        self._write_feed(age_seconds=10.0)
        app = self._visible_app()

        self.assertFalse(app._auto_hide_callback())

        self.assertFalse(app.visible)
        self.rearm.assert_not_called()

    def test_live_feed_keeps_overlay_while_recording(self):
        self.status.write_text('true')
        self._write_feed()
        app = self._visible_app()

        self.assertFalse(app._auto_hide_callback())

        self.assertTrue(app.visible)
        self.rearm.assert_called_once()

    def test_missing_status_hides_overlay(self):
        app = self._visible_app()

        self.assertFalse(app._auto_hide_callback())

        self.assertFalse(app.visible)

    def test_stale_status_is_still_believed_without_a_feed(self):
        # Overlay shown without a feed (controller publishes no frames, so the
        # daemon opened its own stream): there is no liveness signal to check.
        self.status.write_text('true')
        app = self._visible_app(feed_liveness=False)

        self.assertFalse(app._auto_hide_callback())

        self.assertTrue(app.visible)
        self.rearm.assert_called_once()
