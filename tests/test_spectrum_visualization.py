import importlib
import sys
import time
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib"))


class SpectrumVisualizationTests(unittest.TestCase):
    def setUp(self):
        for name in tuple(sys.modules):
            if name == "mic_osd.visualizations" or name.startswith(
                "mic_osd.visualizations."
            ):
                sys.modules.pop(name, None)
        cairo = types.SimpleNamespace(Context=object)
        self.cairo_patch = mock.patch.dict(sys.modules, {"cairo": cairo})
        self.cairo_patch.start()
        module = importlib.import_module("mic_osd.visualizations.spectrum")
        self.SpectrumVisualization = module.SpectrumVisualization
        self.VisualizerState = module.VisualizerState

    def tearDown(self):
        self.cairo_patch.stop()

    def _advance(self, visualization, seconds=0.05):
        visualization._last_update = time.monotonic() - seconds

    def test_style_declares_large_cavasik_like_surface(self):
        visualization = self.SpectrumVisualization()

        self.assertEqual(visualization.preferred_width, 560)
        self.assertEqual(visualization.preferred_height, 180)
        self.assertTrue(visualization.show_preview)

    def test_silence_decays_to_flat_center_line(self):
        visualization = self.SpectrumVisualization()
        self._advance(visualization)

        visualization.update(0.0, np.zeros(96))

        np.testing.assert_allclose(visualization.bar_heights, 0.0)

    def test_frequency_energy_lifts_matching_bars(self):
        visualization = self.SpectrumVisualization()
        self._advance(visualization)
        spectrum = np.full(96, 1e-7)
        spectrum[12:18] = 0.08

        visualization.update(0.2, spectrum)

        self.assertGreater(float(np.max(visualization.bar_heights[12:18])), 0.5)
        self.assertLess(float(np.max(visualization.bar_heights[50:])), 0.05)

    def test_processing_state_has_travelling_wave(self):
        visualization = self.SpectrumVisualization()
        visualization.set_state("processing")
        self._advance(visualization)

        visualization.update(0.0, np.zeros(96))

        self.assertEqual(
            visualization.state_manager.current_state,
            self.VisualizerState.PROCESSING,
        )
        self.assertGreater(float(np.ptp(visualization.bar_heights)), 0.1)

    def test_registry_exposes_spectrum_style(self):
        module = importlib.import_module("mic_osd.visualizations")
        self.assertIs(
            module.VISUALIZATIONS["spectrum"],
            self.SpectrumVisualization,
        )


if __name__ == "__main__":
    unittest.main()
