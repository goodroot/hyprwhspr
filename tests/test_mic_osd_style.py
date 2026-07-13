import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib"))

from mic_osd.runner import MicOSDRunner


class MicOSDStyleTests(unittest.TestCase):
    def test_default_style_is_waveform(self):
        self.assertEqual(MicOSDRunner()._style, "waveform")

    def test_configured_style_is_used(self):
        self.assertEqual(MicOSDRunner(style="pill")._style, "pill")

    def test_unknown_style_falls_back_to_waveform(self):
        for style in ("unknown", "", None, ["pill"], True):
            self.assertEqual(MicOSDRunner(style=style)._style, "waveform")


if __name__ == "__main__":
    unittest.main()
