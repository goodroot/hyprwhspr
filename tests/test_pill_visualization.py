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


class PillVisualizationTests(unittest.TestCase):
    def setUp(self):
        self._clear_visualization_modules()

        cairo = types.SimpleNamespace(Context=object)
        self.cairo_patch = mock.patch.dict(sys.modules, {"cairo": cairo})
        self.cairo_patch.start()
        module = importlib.import_module("mic_osd.visualizations.pill")
        self.PillVisualization = module.PillVisualization
        self.VisualizerState = module.VisualizerState

    def tearDown(self):
        self.cairo_patch.stop()

    @staticmethod
    def _clear_visualization_modules():
        for name in tuple(sys.modules):
            if name == "mic_osd.visualizations" or name.startswith(
                "mic_osd.visualizations."
            ):
                sys.modules.pop(name, None)

    def _advance(self, visualization, seconds=0.05):
        visualization._last_update = time.monotonic() - seconds

    def test_silence_stays_as_idle_dots(self):
        visualization = self.PillVisualization()
        self._advance(visualization)
        visualization.update(0.0, np.zeros(13))
        np.testing.assert_allclose(visualization.bar_heights, 0.0)

    def test_normal_speech_lifts_bars(self):
        visualization = self.PillVisualization()
        self._advance(visualization)
        visualization.update(0.05, np.full(13, 0.05))
        self.assertGreater(float(np.max(visualization.bar_heights)), 0.2)

    def test_below_noise_gate_remains_idle(self):
        visualization = self.PillVisualization()
        self._advance(visualization)
        visualization.update(0.004, np.full(13, 0.004))
        np.testing.assert_allclose(visualization.bar_heights, 0.0)

    def test_processing_state_generates_animated_wave(self):
        visualization = self.PillVisualization()
        visualization.set_state("processing")
        self._advance(visualization)
        visualization.update(0.0, np.zeros(13))
        self.assertEqual(
            visualization.state_manager.current_state,
            self.VisualizerState.PROCESSING,
        )
        self.assertGreater(float(np.ptp(visualization.bar_heights)), 0.01)

    def test_compact_style_exposes_pill_transcript_preview(self):
        visualization = self.PillVisualization()
        self.assertTrue(visualization.show_preview)
        self.assertEqual(visualization.preview_mode, "pill")

    def test_tall_surface_keeps_pill_at_bottom(self):
        visualization = self.PillVisualization()
        _, y, _, pill_height = visualization._pill_geometry(400, 84)
        self.assertEqual(y, 84 - pill_height - 4)

    def test_registry_exposes_the_pill_style(self):
        module = importlib.import_module("mic_osd.visualizations")
        self.assertIs(module.VISUALIZATIONS["pill"], self.PillVisualization)


if __name__ == "__main__":
    unittest.main()
