import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib" / "src"))

from audio_resampler import ResamplingError, resample_audio


def _fake_resample(samples, source_rate, target_rate, quality="HQ"):
    count = round(len(samples) * target_rate / source_rate)
    old_x = np.linspace(0.0, 1.0, len(samples), endpoint=False)
    new_x = np.linspace(0.0, 1.0, count, endpoint=False)
    return np.interp(new_x, old_x, samples)


class AudioResamplerTests(unittest.TestCase):
    def setUp(self):
        self.soxr = mock.patch.dict(
            sys.modules, {"soxr": types.SimpleNamespace(resample=_fake_resample)}
        )
        self.soxr.start()

    def tearDown(self):
        self.soxr.stop()

    def test_common_rate_conversions_have_exact_float32_mono_output(self):
        for source_rate, target_rate in ((48000, 16000), (44100, 16000), (48000, 24000)):
            with self.subTest(source_rate=source_rate, target_rate=target_rate):
                audio = np.sin(2 * np.pi * 440 * np.arange(source_rate) / source_rate)
                result = resample_audio(audio, source_rate, target_rate)
                self.assertEqual(result.shape, (target_rate,))
                self.assertEqual(result.dtype, np.float32)
                self.assertGreater(float(np.max(np.abs(result))), 0.9)

    def test_unchanged_rate_normalizes_dtype_and_shape(self):
        result = resample_audio(np.array([0, 1, -1], dtype=np.int16), 16000, 16000)
        self.assertEqual(result.dtype, np.float32)
        self.assertEqual(result.shape, (3,))

    def test_rejects_empty_multichannel_and_invalid_rates(self):
        for audio, source, target in (
            (np.array([]), 48000, 16000),
            (np.zeros((4, 2)), 48000, 16000),
            (np.zeros(4), 0, 16000),
            (np.array([np.nan]), 16000, 16000),
        ):
            with self.subTest(source=source, target=target):
                with self.assertRaises(ValueError):
                    resample_audio(audio, source, target)

    def test_backend_failure_is_explicit(self):
        failing = types.SimpleNamespace(resample=mock.Mock(side_effect=RuntimeError("boom")))
        with mock.patch.dict(sys.modules, {"soxr": failing}):
            with self.assertRaisesRegex(ResamplingError, "48000Hz -> 16000Hz"):
                resample_audio(np.zeros(48, dtype=np.float32), 48000, 16000)


if __name__ == "__main__":
    unittest.main()
