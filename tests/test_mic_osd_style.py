import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib"))

from mic_osd.runner import MicOSDRunner
from mic_osd.osd_config import MicOSDConfig


class MicOSDStyleTests(unittest.TestCase):
    def test_default_style_is_waveform(self):
        self.assertEqual(MicOSDRunner()._style, "waveform")

    def test_configured_style_is_used(self):
        for style in ("pill", "spectrum"):
            with self.subTest(style=style):
                self.assertEqual(MicOSDRunner(style=style)._style, style)

    def test_unknown_style_falls_back_to_waveform(self):
        for style in ("unknown", "", None, ["pill"], True):
            self.assertEqual(MicOSDRunner(style=style)._style, "waveform")

    def test_osd_config_clamps_layout_and_keeps_valid_db_range(self):
        values = {
            "mic_osd_spectrum_width": 20,
            "mic_osd_spectrum_meter_height": 9999,
            "mic_osd_spectrum_background_opacity": 4,
            "mic_osd_spectrum_noise_floor_db": -30,
            "mic_osd_spectrum_peak_db": -60,
            "mic_osd_transcript_font_size": 100,
            "mic_osd_transcript_max_lines": 0,
        }
        config = MicOSDConfig.from_getter(values.get)

        self.assertEqual(config.spectrum_width, 280)
        self.assertEqual(config.spectrum_meter_height, 360)
        self.assertEqual(config.spectrum_background_opacity, 1.0)
        self.assertEqual(config.spectrum_peak_db, -24.0)
        self.assertEqual(config.transcript_font_size, 72.0)
        self.assertEqual(config.transcript_max_lines, 1)


if __name__ == "__main__":
    unittest.main()
