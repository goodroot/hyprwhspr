"""The ONNX backend previews with segments but pastes the batch result."""

import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'lib' / 'src'))

from backends.onnx_asr_backend import OnnxAsrBackend  # noqa: E402

SECOND = 16000


class FakeModel:
    """Recognize returns a segment marker for short audio, batch text for long."""

    def __init__(self, text='batch words'):
        self.calls = []
        self.text = text

    def recognize(self, audio, sample_rate=16000):
        self.calls.append(int(len(audio)))
        if len(audio) <= 2 * SECOND:
            return types.SimpleNamespace(text='preview')
        return types.SimpleNamespace(text=self.text)


class FakeVadAdapter:
    """One closed segment, everything but the last second."""

    class Vad:
        @staticmethod
        def segment_batch(waveforms, lengths, sample_rate):
            total = int(lengths[0])
            end = total - SECOND
            yield iter([(0, end)] if end > SECOND else [])

    def __init__(self):
        self.vad = self.Vad()


def make_backend(model=None, vad=None):
    backend = OnnxAsrBackend(mock.Mock())
    backend._onnx_asr_model = model
    backend._onnx_asr_vad_model = vad
    return backend


class OnnxLivePreviewTests(unittest.TestCase):
    def test_preview_callback_is_offered_when_model_and_vad_are_loaded(self):
        backend = make_backend(FakeModel(), FakeVadAdapter())
        self.addCleanup(backend.finish_preview)
        self.assertTrue(callable(backend.preview_audio_callback(lambda text: None)))

    def test_preview_is_unavailable_without_a_model(self):
        backend = make_backend(None, FakeVadAdapter())
        self.assertIsNone(backend.preview_audio_callback(lambda text: None))

    def test_preview_is_unavailable_without_vad(self):
        backend = make_backend(FakeModel(), None)
        self.assertIsNone(backend.preview_audio_callback(lambda text: None))

    def test_preview_text_reaches_the_callback(self):
        backend = make_backend(FakeModel(), FakeVadAdapter())
        self.addCleanup(backend.finish_preview)
        texts = []
        feed = backend.preview_audio_callback(texts.append)
        feed(np.zeros(3 * SECOND, dtype=np.float32), 16000)
        backend._preview.pump()
        self.assertEqual(texts, ['preview'])

    def test_paste_still_comes_from_the_batch_decode(self):
        model = FakeModel('authoritative text')
        backend = make_backend(model, FakeVadAdapter())
        self.addCleanup(backend.finish_preview)
        feed = backend.preview_audio_callback(lambda text: None)
        feed(np.zeros(3 * SECOND, dtype=np.float32), 16000)
        backend._preview.pump()
        self.assertEqual(backend.transcribe(np.zeros(4 * SECOND, dtype=np.float32)), 'authoritative text')
        self.assertEqual(model.calls[-1], 4 * SECOND)

    def test_finish_preview_stops_and_forgets_the_worker(self):
        backend = make_backend(FakeModel(), FakeVadAdapter())
        backend.preview_audio_callback(lambda text: None)
        preview = backend._preview
        backend.finish_preview()
        self.assertIsNone(backend._preview)
        self.assertTrue(preview.stopped)

    def test_unload_finishes_the_preview(self):
        backend = make_backend(FakeModel(), FakeVadAdapter())
        backend.preview_audio_callback(lambda text: None)
        backend.unload()
        self.assertIsNone(backend._preview)
        self.assertIsNone(backend._onnx_asr_model)


if __name__ == '__main__':
    unittest.main()
