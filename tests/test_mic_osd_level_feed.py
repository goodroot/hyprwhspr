"""Tests for the mic-OSD level feed (issue #205).

The OSD meter is fed from the main process's capture stream via a runtime
file instead of opening its own audio stream, so it always reflects the
device actually being recorded and never contends for exclusive ALSA devices.
"""

import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib"))

from mic_osd.audio import FeedLevelSource
from mic_osd.runner import MicOSDRunner
import mic_osd.runner as runner_module


class FeedLevelSourceTests(unittest.TestCase):
    def _write_feed(self, path, level, buckets):
        path.write_text(" ".join(f"{v:.6f}" for v in [level, *buckets]))

    def test_reads_level_and_samples_from_fresh_feed(self):
        with tempfile.TemporaryDirectory() as tmp:
            feed = Path(tmp) / "mic_osd_level_feed"
            self._write_feed(feed, 0.25, [0.1, 0.2, 0.3])

            source = FeedLevelSource(feed)
            self.assertAlmostEqual(source.get_level(), 0.25)
            np.testing.assert_allclose(source.get_samples(), [0.1, 0.2, 0.3])

    def test_available_only_when_file_is_fresh(self):
        with tempfile.TemporaryDirectory() as tmp:
            feed = Path(tmp) / "mic_osd_level_feed"
            self.assertFalse(FeedLevelSource.available(feed))

            self._write_feed(feed, 0.5, [0.5])
            self.assertTrue(FeedLevelSource.available(feed))
            self.assertFalse(FeedLevelSource.available(feed, max_age=0.0))

    def test_stale_feed_decays_to_silence(self):
        with tempfile.TemporaryDirectory() as tmp:
            feed = Path(tmp) / "mic_osd_level_feed"
            self._write_feed(feed, 0.9, [0.9])
            source = FeedLevelSource(feed)
            self.assertAlmostEqual(source.get_level(), 0.9)

            source.STALE_AFTER_SECONDS = 0.0
            self.assertEqual(source.get_level(), 0.0)
            self.assertEqual(len(source.get_samples()), 0)

    def test_missing_feed_reads_as_silence(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = FeedLevelSource(Path(tmp) / "never_written")
            self.assertEqual(source.get_level(), 0.0)
            self.assertEqual(len(source.get_samples()), 0)

    def test_garbage_content_keeps_last_good_frame(self):
        with tempfile.TemporaryDirectory() as tmp:
            feed = Path(tmp) / "mic_osd_level_feed"
            self._write_feed(feed, 0.4, [0.4, 0.4])
            source = FeedLevelSource(feed)
            self.assertAlmostEqual(source.get_level(), 0.4)

            feed.write_text("not floats at all")
            self.assertAlmostEqual(source.get_level(), 0.4)


class MicOSDRunnerLevelFeedTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.feed_file = Path(self._tmp.name) / "hyprwhspr" / "mic_osd_level_feed"
        self._original_feed_file = runner_module.MIC_OSD_LEVEL_FEED_FILE
        runner_module.MIC_OSD_LEVEL_FEED_FILE = self.feed_file

    def tearDown(self):
        runner_module.MIC_OSD_LEVEL_FEED_FILE = self._original_feed_file
        self._tmp.cleanup()

    def test_writes_frames_readable_by_feed_source(self):
        runner = MicOSDRunner(level_source=lambda: (0.125, [0.1, 0.2]))
        runner._write_level_feed_frame()

        self.assertTrue(FeedLevelSource.available(self.feed_file))
        source = FeedLevelSource(self.feed_file)
        self.assertAlmostEqual(source.get_level(), 0.125)
        np.testing.assert_allclose(source.get_samples(), [0.1, 0.2])
        self.assertEqual(self.feed_file.stat().st_mode & 0o777, 0o600)

    def test_none_frame_writes_zero_level(self):
        runner = MicOSDRunner(level_source=lambda: None)
        runner._write_level_feed_frame()

        source = FeedLevelSource(self.feed_file)
        self.assertEqual(source.get_level(), 0.0)
        self.assertEqual(len(source.get_samples()), 0)

    def test_start_writes_first_frame_synchronously_and_stop_removes_file(self):
        runner = MicOSDRunner(level_source=lambda: (0.5, [0.5]))
        runner._start_level_feed()
        try:
            # First frame must exist before any thread scheduling happens
            self.assertTrue(self.feed_file.exists())
            self.assertIsNotNone(runner._level_feed_thread)
        finally:
            runner._stop_level_feed()

        self.assertFalse(self.feed_file.exists())
        self.assertIsNone(runner._level_feed_thread)

    def test_feed_thread_keeps_writing(self):
        wrote_twice = threading.Event()
        calls = []

        def source():
            calls.append(time.monotonic())
            if len(calls) >= 2:
                wrote_twice.set()
            return (0.3, [0.3])

        runner = MicOSDRunner(level_source=source)
        runner._start_level_feed()
        try:
            self.assertTrue(wrote_twice.wait(timeout=2.0))
        finally:
            runner._stop_level_feed()

    def test_no_level_source_means_no_feed(self):
        runner = MicOSDRunner()
        runner._start_level_feed()
        self.assertIsNone(runner._level_feed_thread)
        self.assertFalse(self.feed_file.exists())

    def test_hide_stops_feed(self):
        runner = MicOSDRunner(level_source=lambda: (0.5, [0.5]))
        runner._start_level_feed()
        runner.hide()
        self.assertIsNone(runner._level_feed_thread)
        self.assertFalse(self.feed_file.exists())


class GetVizFrameTests(unittest.TestCase):
    """Exercise AudioCapture.get_viz_frame's chunk reduction in isolation."""

    def _make_capture(self):
        sys.path.insert(0, str(ROOT / "lib" / "src"))
        import audio_capture
        capture = object.__new__(audio_capture.AudioCapture)
        capture.lock = threading.Lock()
        capture._viz_chunk = None
        capture._viz_chunk_time = 0.0
        capture.current_level = 0.0
        return capture

    def test_returns_none_without_chunk(self):
        capture = self._make_capture()
        self.assertIsNone(capture.get_viz_frame())

    def test_returns_none_when_chunk_is_stale(self):
        capture = self._make_capture()
        capture._viz_chunk = np.ones(1024, dtype=np.float32)
        capture._viz_chunk_time = time.monotonic() - 5.0
        capture.current_level = 0.7
        self.assertIsNone(capture.get_viz_frame())

    def test_reduces_chunk_to_bucket_rms(self):
        capture = self._make_capture()
        chunk = np.zeros(1024, dtype=np.float32)
        chunk[512:] = 0.5  # Second half loud, first half silent
        capture._viz_chunk = chunk
        capture._viz_chunk_time = time.monotonic()
        capture.current_level = 0.05

        frame = capture.get_viz_frame(num_buckets=32)
        self.assertIsNotNone(frame)
        level, buckets = frame
        # Level is display-scaled (raw * 10, clamped) to match get_audio_level()
        self.assertAlmostEqual(level, 0.5)
        self.assertEqual(len(buckets), 32)
        self.assertTrue(all(b == 0.0 for b in buckets[:16]))
        self.assertTrue(all(abs(b - 0.5) < 1e-6 for b in buckets[16:]))

    def test_display_level_is_clamped_to_one(self):
        capture = self._make_capture()
        capture._viz_chunk = np.ones(1024, dtype=np.float32) * 0.5
        capture._viz_chunk_time = time.monotonic()
        capture.current_level = 0.4  # * 10 = 4.0, must clamp to 1.0

        level, _ = capture.get_viz_frame(num_buckets=32)
        self.assertEqual(level, 1.0)

    def test_short_chunk_falls_back_to_abs_samples(self):
        capture = self._make_capture()
        capture._viz_chunk = np.array([-0.2, 0.1], dtype=np.float32)
        capture._viz_chunk_time = time.monotonic()
        capture.current_level = 0.15

        frame = capture.get_viz_frame(num_buckets=32)
        self.assertIsNotNone(frame)
        _, buckets = frame
        self.assertEqual(len(buckets), 2)
        self.assertAlmostEqual(buckets[0], 0.2, places=5)


if __name__ == "__main__":
    unittest.main()
