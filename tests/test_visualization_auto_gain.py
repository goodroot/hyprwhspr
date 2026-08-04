import importlib
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib"))


class AutoGainTests(unittest.TestCase):
    def setUp(self):
        self._clear_visualization_modules()
        cairo = types.SimpleNamespace(
            Context=object,
            LinearGradient=object,
            FONT_SLANT_NORMAL=0,
            FONT_WEIGHT_NORMAL=0,
        )
        self.cairo_patch = mock.patch.dict(sys.modules, {"cairo": cairo})
        self.cairo_patch.start()
        base = importlib.import_module("mic_osd.visualizations.base")
        self.AutoGain = base.AutoGain
        self.WaveformVisualization = importlib.import_module(
            "mic_osd.visualizations.waveform"
        ).WaveformVisualization

    def tearDown(self):
        self.cairo_patch.stop()

    @staticmethod
    def _clear_visualization_modules():
        for name in tuple(sys.modules):
            if name == "mic_osd.visualizations" or name.startswith(
                "mic_osd.visualizations."
            ):
                sys.modules.pop(name, None)

    def test_hot_mic_keeps_previous_fixed_gain(self):
        gain = self.AutoGain(min_gain=4.0, noise_floor=0.002)
        self.assertEqual(gain.update(0.3), 4.0)

    def test_quiet_mic_is_boosted(self):
        """Real Focusrite Scarlett capture peaked at 0.031 per bucket."""
        gain = self.AutoGain(min_gain=4.0, noise_floor=0.002)
        self.assertGreater(gain.update(0.031) * 0.031, 0.5)

    def test_room_tone_is_not_boosted(self):
        gain = self.AutoGain(min_gain=4.0, noise_floor=0.002)
        for _ in range(60):
            result = gain.update(0.0004)
        self.assertEqual(result, 4.0)

    def test_gain_is_capped(self):
        gain = self.AutoGain(min_gain=4.0, noise_floor=0.0)
        self.assertLessEqual(gain.update(1e-9), self.AutoGain.MAX_GAIN)

    def test_envelope_releases_after_loud_passage(self):
        gain = self.AutoGain(min_gain=4.0, noise_floor=0.002)
        gain.update(0.5)
        for _ in range(200):
            gain.update(0.02)
        self.assertLess(gain.envelope, 0.5)

    def test_waveform_shows_quiet_input(self):
        """Regression: a distant XLR mic rendered as a flat line under the
        old fixed amplification of 4.0."""
        waveform = self.WaveformVisualization()
        for _ in range(10):
            waveform.update(0.05, np.full(32, 0.0108))
        self.assertGreater(float(waveform.bar_heights.max()), 0.2)

    def test_reset_clears_a_loud_recordings_envelope(self):
        """The meter is frozen while the OSD is hidden, so without a reset a
        loud recording would hold the gain down through the next quiet one."""
        gain = self.AutoGain(min_gain=4.0, noise_floor=0.002)
        gain.update(0.8)
        self.assertEqual(gain.update(0.01), 4.0)  # suppressed by stale envelope
        gain.reset()
        self.assertGreater(gain.update(0.01) * 0.01, 0.5)

    def test_waveform_silence_stays_flat(self):
        waveform = self.WaveformVisualization()
        for _ in range(30):
            waveform.update(0.0, np.full(32, 0.0004))
        self.assertLess(float(waveform.bar_heights.max()), 0.05)


if __name__ == "__main__":
    unittest.main()
