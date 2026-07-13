import importlib
import json
import os
import sys
import tempfile
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

    def test_compact_style_hides_transcript_preview(self):
        visualization = self.PillVisualization()
        self.assertFalse(visualization.show_preview)

    def test_daemon_registry_applies_the_configured_pill(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir) / "hyprwhspr"
            config_dir.mkdir()
            (config_dir / "config.json").write_text(
                json.dumps({"mic_osd_style": "pill"}),
                encoding="utf-8",
            )

            self._clear_visualization_modules()
            with mock.patch.dict(
                os.environ,
                {
                    "HYPRWHSPR_MIC_OSD_DAEMON": "1",
                    "XDG_CONFIG_HOME": temp_dir,
                },
                clear=True,
            ):
                module = importlib.import_module("mic_osd.visualizations")

            self.assertIs(
                module.VISUALIZATIONS["waveform"],
                module.PillVisualization,
            )


if __name__ == "__main__":
    unittest.main()
