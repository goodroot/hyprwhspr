"""Tests for the opt-in debug_recordings setting (#264)."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from tests.test_suspend_resume_recovery import FakeConfig, _import_main_isolated


class DebugRecordingsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.main = _import_main_isolated()

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name) / 'recordings'
        patcher = mock.patch.object(self.main, 'DEBUG_RECORDINGS_DIR', self.dir)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _app(self, values):
        app = self.main.hyprwhsprApp.__new__(self.main.hyprwhsprApp)
        app.config = FakeConfig(values)
        app.audio_capture = mock.Mock()
        app.audio_capture.save_audio_to_wav.side_effect = (
            lambda audio_data, filename: Path(filename).write_bytes(b'RIFF')
        )
        return app

    def test_off_by_default(self):
        app = self._app({})
        app._save_debug_recording(np.zeros(10, dtype=np.float32))
        app.audio_capture.save_audio_to_wav.assert_not_called()
        self.assertFalse(self.dir.exists())

    def test_keeps_only_the_newest(self):
        app = self._app({'debug_recordings': 2})
        stamps = ['20260101-000000', '20260101-000001', '20260101-000002']
        with mock.patch.object(self.main.time, 'strftime', side_effect=stamps):
            for _ in stamps:
                app._save_debug_recording(np.zeros(10, dtype=np.float32))

        remaining = sorted(p.name for p in self.dir.glob('*.wav'))
        self.assertEqual(remaining, ['20260101-000001.wav', '20260101-000002.wav'])


if __name__ == '__main__':
    unittest.main()
