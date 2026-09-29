"""Tests for the opt-in debug_recordings setting (#264)."""

import sys
import tempfile
import threading
import unittest
import wave
from pathlib import Path
from unittest import mock

import numpy as np

from tests.test_suspend_resume_recovery import FakeConfig, _import_main_isolated, patch_app_global

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'lib' / 'src'))


class DebugRecordingsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.main = _import_main_isolated()

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name) / 'recordings'
        patcher = patch_app_global(
            self.main.hyprwhsprApp._save_debug_recording, 'DEBUG_RECORDINGS_DIR', self.dir)
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
        remaining = sorted(p.stem[:len(stamps[0])] for p in self.dir.glob('*.wav'))
        self.assertEqual(remaining, stamps[1:])

    def test_same_second_saves_do_not_overwrite(self):
        app = self._app({'debug_recordings': True})
        with mock.patch.object(self.main.time, 'strftime', return_value='20260101-000000'), \
                mock.patch.object(self.main.time, 'time_ns', side_effect=[1_000_000, 2_000_000]):
            app._save_debug_recording(np.zeros(10, dtype=np.float32))
            app._save_debug_recording(np.zeros(10, dtype=np.float32))
        self.assertEqual(len(list(self.dir.glob('*.wav'))), 2)

    def test_continuous_flush_saves_the_flushed_audio(self):
        # Continuous mode transcribes at each pause; the stop path only sees the tail
        app = self._app({'debug_recordings': True})
        audio = np.full(16000, 0.1, dtype=np.float32)
        app.audio_capture.flush_buffer.return_value = audio
        app.audio_capture.sample_rate = 16000
        app._continuous_flush_lock = threading.Lock()
        app._continuous_transcription_done = threading.Event()
        app._is_zero_volume = mock.Mock(return_value=False)

        with mock.patch.object(self.main.threading, 'Thread') as thread:
            app._continuous_flush_audio()

        thread.return_value.start.assert_called_once()
        app.audio_capture.save_audio_to_wav.assert_called_once()
        self.assertIs(app.audio_capture.save_audio_to_wav.call_args.args[0], audio)

    def test_non_bool_value_still_caps(self):
        # A hand-edited number or string used to break pruning and pile up WAVs
        self._record(self._app({'debug_recordings': '20'}), 5)
        self.assertEqual(len(list(self.dir.glob('*.wav'))), 3)

    def test_private_directory(self):
        self._record(self._app({'debug_recordings': True}), 1)
        self.assertEqual(self.dir.stat().st_mode & 0o777, 0o700)


class SaveAudioToWavTests(unittest.TestCase):
    def test_out_of_range_samples_clip_instead_of_wrapping(self):
        with mock.patch.dict(sys.modules, {'sounddevice': mock.Mock()}):
            import audio_capture
            capture = audio_capture.AudioCapture.__new__(audio_capture.AudioCapture)
        capture.channels = 1
        capture.sample_rate = 16000

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'loud.wav'
            capture.save_audio_to_wav(np.array([1.5, -1.5, 0.5], dtype=np.float32), str(path))
            with wave.open(str(path), 'rb') as wav_file:
                samples = np.frombuffer(wav_file.readframes(3), dtype=np.int16)

        self.assertEqual(samples.tolist(), [32767, -32767, 16383])


if __name__ == '__main__':
    unittest.main()
