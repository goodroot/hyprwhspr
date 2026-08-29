import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'lib' / 'src'))

import audio_file  # noqa: E402


class AudioFileTests(unittest.TestCase):
    def _decode(self, suffix, samples, rate=44100):
        with tempfile.TemporaryDirectory() as tempdir:
            path = Path(tempdir) / f'audio{suffix}'
            path.touch()
            fake_soundfile = mock.Mock()
            fake_soundfile.read.return_value = (samples, rate)

            with mock.patch.dict(sys.modules, {'soundfile': fake_soundfile}):
                result = audio_file.decode_audio_file(path)
            fake_soundfile.read.assert_called_once_with(
                str(path), dtype='float32', always_2d=True
            )
            return result

    def test_decodes_wav_mono_as_contiguous_float32(self):
        samples, rate = self._decode(
            '.wav', np.array([[0.1], [0.2]], dtype=np.float32), 22050
        )
        self.assertEqual(rate, 22050)
        self.assertEqual(samples.dtype, np.float32)
        self.assertTrue(samples.flags['C_CONTIGUOUS'])
        np.testing.assert_allclose(samples, [0.1, 0.2])

    def test_decodes_mp3_and_downmixes_channels(self):
        samples, rate = self._decode(
            '.MP3', np.array([[1.0, -1.0], [0.5, 0.25]], dtype=np.float32)
        )
        self.assertEqual(rate, 44100)
        np.testing.assert_allclose(samples, [0.0, 0.375])

    def test_rejects_missing_unsupported_empty_and_decode_failure(self):
        with self.assertRaisesRegex(audio_file.AudioFileError, 'does not exist'):
            audio_file.decode_audio_file('/missing/input.wav')
        with tempfile.TemporaryDirectory() as tempdir:
            unsupported = Path(tempdir) / 'audio.flac'
            unsupported.touch()
            with self.assertRaisesRegex(audio_file.AudioFileError, 'Unsupported'):
                audio_file.decode_audio_file(unsupported)

            wav = Path(tempdir) / 'audio.wav'
            wav.touch()
            fake_soundfile = mock.Mock()
            fake_soundfile.read.return_value = (
                np.empty((0, 1), dtype=np.float32), 16000
            )
            with mock.patch.dict(sys.modules, {'soundfile': fake_soundfile}):
                with self.assertRaisesRegex(audio_file.AudioFileError, 'no samples'):
                    audio_file.decode_audio_file(wav)

            fake_soundfile.read.side_effect = RuntimeError('bad stream')
            with mock.patch.dict(sys.modules, {'soundfile': fake_soundfile}):
                with self.assertRaisesRegex(audio_file.AudioFileError, 'bad stream'):
                    audio_file.decode_audio_file(wav)

    def test_missing_soundfile_is_an_audio_error_not_system_exit(self):
        with tempfile.TemporaryDirectory() as tempdir:
            wav = Path(tempdir) / 'audio.wav'
            wav.touch()
            with mock.patch.dict(sys.modules, {'soundfile': None}):
                with self.assertRaisesRegex(
                        audio_file.AudioFileError, 'hyprwhspr setup'):
                    audio_file.decode_audio_file(wav)


if __name__ == '__main__':
    unittest.main()
