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

    def _record(self, app, count):
        stamps = [f'20260101-00000{i}' for i in range(count)]
        with mock.patch.object(self.main.time, 'strftime', side_effect=stamps):
            for _ in stamps:
                app._save_debug_recording(np.zeros(10, dtype=np.float32))
        return stamps

    def test_off_by_default(self):
        app = self._app({})
        app._save_debug_recording(np.zeros(10, dtype=np.float32))
        app.audio_capture.save_audio_to_wav.assert_not_called()
        self.assertFalse(self.dir.exists())

    def test_keeps_only_the_newest_three(self):
        stamps = self._record(self._app({'debug_recordings': True}), 4)
        remaining = sorted(p.stem for p in self.dir.glob('*.wav'))
        self.assertEqual(remaining, stamps[1:])

    def test_non_bool_value_still_caps(self):
        # A hand-edited number or string used to break pruning and pile up WAVs
        self._record(self._app({'debug_recordings': '20'}), 5)
        self.assertEqual(len(list(self.dir.glob('*.wav'))), 3)

    def test_private_directory(self):
        self._record(self._app({'debug_recordings': True}), 1)
        self.assertEqual(self.dir.stat().st_mode & 0o777, 0o700)


if __name__ == '__main__':
    unittest.main()
