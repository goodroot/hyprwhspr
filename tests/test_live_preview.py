"""Live preview decodes closed segments only, and never the pasted text."""

import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'lib' / 'src'))

import live_preview  # noqa: E402
from live_preview import PREVIEW_RATE, LivePreview  # noqa: E402

SECOND = PREVIEW_RATE


def silence(seconds):
    return np.zeros(int(seconds * PREVIEW_RATE), dtype=np.float32)


class LivePreviewTests(unittest.TestCase):
    def make(self, segment, decode, on_text=None):
        # A long interval keeps the worker idle; the tests pump by hand.
        preview = LivePreview(segment=segment, decode=decode, on_text=on_text, interval=60)
        self.addCleanup(preview.stop)
        return preview

    def test_a_closed_segment_is_decoded_once(self):
        decoded = []

        def segment(audio):
            # One segment, closed 0.25 s before the end of the buffer.
            end = audio.size - SECOND // 4
            return [(0, end)] if end > 2 * SECOND else []

        def decode(audio):
            decoded.append(audio.size)
            return 'hello'

        texts = []
        preview = self.make(segment, decode, texts.append)
        preview.feed(silence(3))
        preview.pump()
        self.assertEqual(decoded, [int(3 * SECOND - SECOND / 4)])
        self.assertEqual(texts, ['hello'])
        preview.feed(silence(1))
        preview.pump()
        self.assertEqual(len(decoded), 1, 'a decoded segment must not be decoded again')

    def test_a_segment_that_may_still_grow_is_deferred(self):
        decoded = []
        preview = self.make(lambda audio: [(0, audio.size)],
                            lambda audio: decoded.append(1) or 'x')
        preview.feed(silence(1))
        preview.pump()
        self.assertEqual(decoded, [])

    def test_new_segments_are_appended_in_order(self):
        def segment(audio):
            # 1 s blocks, each closed by the next one's silence.
            blocks = [(i * SECOND, (i + 1) * SECOND) for i in range(audio.size // SECOND - 1)]
            return blocks

        texts = []
        counter = iter(range(100))
        with mock.patch.object(live_preview, '_MIN_PUBLISH_INTERVAL_S', 0):
            preview = self.make(segment, lambda audio: f'w{next(counter)}', texts.append)
            preview.feed(silence(4))
            preview.pump()
            preview.feed(silence(1))
            preview.pump()
        self.assertTrue(texts)
        self.assertEqual(texts[-1], 'w0 w1 w2')

    def test_decode_failure_disables_preview_without_raising(self):
        def decode(audio):
            raise RuntimeError('model died')

        preview = self.make(lambda audio: [(0, SECOND)], decode)
        preview.feed(silence(2))
        preview.pump()  # must not raise
        self.assertTrue(preview.disabled)
        preview.feed(silence(1))  # and capture keeps working
        preview.pump()

    def test_segmentation_failure_disables_preview_without_raising(self):
        def segment(audio):
            raise RuntimeError('vad died')

        preview = self.make(segment, lambda audio: 'x')
        preview.feed(silence(2))
        preview.pump()
        self.assertTrue(preview.disabled)

    def test_text_is_truncated_to_the_newest_characters(self):
        long_text = 'x' * 400
        texts = []
        with mock.patch.object(live_preview, '_MIN_PUBLISH_INTERVAL_S', 0):
            preview = self.make(lambda audio: [(0, SECOND)], lambda audio: long_text,
                                texts.append)
            preview.feed(silence(2))
            preview.pump()
        self.assertEqual(len(texts[-1]), live_preview._MAX_PREVIEW_CHARS)
        self.assertTrue(texts[-1].endswith('x'))

    def test_capture_audio_is_converted_to_16k(self):
        seen = []
        preview = self.make(lambda audio: seen.append(audio.size) or [],
                            lambda audio: '')
        convert = lambda audio, source, target: np.zeros(
            int(len(audio) * target / source), dtype=np.float32)
        with mock.patch.object(live_preview, 'resample_audio', side_effect=convert):
            preview.feed(np.zeros(48000, dtype=np.float32), sample_rate=48000)
        preview.pump()
        self.assertEqual(seen[0], SECOND)

    def test_a_failed_conversion_disables_preview_without_raising(self):
        preview = self.make(lambda audio: [(0, SECOND)], lambda audio: 'x')
        with mock.patch.object(live_preview, 'resample_audio',
                               side_effect=RuntimeError('no resampler')):
            preview.feed(np.zeros(48000, dtype=np.float32), sample_rate=48000)
        self.assertTrue(preview.disabled)

    def test_stop_is_idempotent_and_ignores_later_feeds(self):
        decoded = []
        preview = self.make(lambda audio: [(0, SECOND)],
                            lambda audio: decoded.append(1) or 'x')
        preview.stop()
        preview.stop()
        preview.feed(silence(2))
        preview.pump()
        self.assertEqual(decoded, [])

    def test_no_callback_still_decodes_and_keeps_text(self):
        preview = self.make(lambda audio: [(0, SECOND)], lambda audio: 'kept')
        preview.feed(silence(2))
        preview.pump()
        self.assertEqual(preview.text, 'kept')


if __name__ == '__main__':
    unittest.main()
