import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib" / "src"))

from segment_manager import SegmentManager


class SegmentManagerTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.manager = SegmentManager()
        self.manager.segments_dir = Path(self.temp_dir.name)
        self.manager.start_session()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_failed_wav_write_leaves_no_partial_final_or_temp(self):
        audio = np.array([0.1, -0.1], dtype=np.float32)
        with mock.patch("segment_manager.wave.open", side_effect=OSError("write failed")):
            self.assertIsNone(self.manager.save_segment(audio))

        self.assertEqual(list(self.manager.segments_dir.iterdir()), [])
        self.assertEqual(self.manager.segments, [])

    def test_concatenation_fails_closed_for_missing_segment(self):
        first = self.manager.save_segment(np.array([0.1, 0.2], dtype=np.float32))
        self.assertIsNotNone(first)
        self.manager.segments.append(self.manager.segments_dir / "missing.wav")

        self.assertIsNone(self.manager.concatenate_all())

    def test_concatenate_readable_includes_unsaved_audio(self):
        self.manager.save_segment(np.array([0.25, 0.5], dtype=np.float32))
        self.manager.segments.append(self.manager.segments_dir / "missing.wav")
        unsaved = np.array([0.75], dtype=np.float32)

        combined = self.manager.concatenate_readable(unsaved)

        self.assertEqual(len(combined), 3)
        self.assertAlmostEqual(float(combined[-1]), 0.75)


if __name__ == "__main__":
    unittest.main()
