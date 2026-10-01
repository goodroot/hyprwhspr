import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib"))
sys.path.insert(0, str(ROOT / "lib" / "src"))

from backends.cohere_backend import CohereBackend


class FakeConfig:
    def __init__(self, values=None):
        self.values = values or {}

    def get_setting(self, key, default=None):
        return self.values.get(key, default)


class FakeManager:
    def __init__(self, config):
        self.config = config
        self.current_model = "old-model"
        self.ready = True
        self._last_use_time = 0.0


class CohereReinitializeTests(unittest.TestCase):
    def _modules(self, cuda_available, events, load_model=None,
                 before_processor=None, before_model=None):
        torch = mock.MagicMock()
        torch.cuda.is_available.return_value = cuda_available
        torch.bfloat16 = "bfloat16"
        torch.float32 = "float32"
        torch.cuda.empty_cache.side_effect = lambda: events.append("empty_cache")

        processor = object()
        model = mock.MagicMock()
        model.to.side_effect = lambda device: events.append("move_model") or model
        transformers = mock.MagicMock()
        transformers.AutoProcessor.from_pretrained.side_effect = (
            lambda *args, **kwargs: (before_processor() if before_processor else None)
            or events.append("load_processor") or processor
        )
        if load_model is None:
            transformers.AutoModelForSpeechSeq2Seq.from_pretrained.side_effect = (
                lambda *args, **kwargs: (before_model() if before_model else None)
                or events.append("load_model") or model
            )
        else:
            transformers.AutoModelForSpeechSeq2Seq.from_pretrained.side_effect = load_model
        return torch, transformers, processor, model

    def _backend(self, device="auto"):
        backend = CohereBackend(FakeManager(FakeConfig({
            "cohere_transcribe_device": device,
        })))
        backend._cohere_model = object()
        backend._cohere_processor = object()
        backend._cohere_compile_done = True
        return backend

    def test_cuda_releases_old_model_and_clears_cache_before_loading(self):
        backend = self._backend()
        events = []
        original_unload = backend.unload

        def unload():
            original_unload()
            events.append("unload")

        backend.unload = unload
        def assert_unloaded():
            self.assertIsNone(backend._cohere_model)
            self.assertIsNone(backend._cohere_processor)

        def assert_model_unloaded():
            self.assertIsNone(backend._cohere_model)

        torch, transformers, processor, model = self._modules(
            True, events, before_processor=assert_unloaded,
            before_model=assert_model_unloaded,
        )
        with mock.patch.dict(sys.modules, {"torch": torch, "transformers": transformers}), \
                mock.patch("backends.cohere_backend.get_credential", return_value=None), \
                mock.patch("gc.collect", side_effect=lambda: events.append("collect")):
            self.assertTrue(backend.reinitialize())

        self.assertEqual(events, ["unload", "collect", "empty_cache", "load_processor", "load_model", "move_model"])
        self.assertIs(backend._cohere_processor, processor)
        self.assertIs(backend._cohere_model, model)
        self.assertFalse(backend._cohere_compile_done)

    def test_cpu_reinitialize_collects_without_cuda_cache_clear(self):
        backend = self._backend(device="cpu")
        events = []
        torch, transformers, _, _ = self._modules(False, events)
        with mock.patch.dict(sys.modules, {"torch": torch, "transformers": transformers}), \
                mock.patch("backends.cohere_backend.get_credential", return_value=None), \
                mock.patch("gc.collect", side_effect=lambda: events.append("collect")):
            self.assertTrue(backend.reinitialize())

        self.assertEqual(events, ["collect", "load_processor", "load_model", "move_model"])
        torch.cuda.empty_cache.assert_not_called()

    def test_failed_reload_leaves_old_model_released_and_returns_false(self):
        backend = self._backend()
        events = []

        def fail_load(*args, **kwargs):
            events.append("load_model")
            raise RuntimeError("load failed")

        torch, transformers, _, _ = self._modules(True, events, load_model=fail_load)
        with mock.patch.dict(sys.modules, {"torch": torch, "transformers": transformers}), \
                mock.patch("backends.cohere_backend.get_credential", return_value=None), \
                mock.patch("gc.collect", side_effect=lambda: events.append("collect")):
            self.assertFalse(backend.reinitialize())

        self.assertEqual(events, ["collect", "empty_cache", "load_processor", "load_model"])
        self.assertIsNone(backend._cohere_model)
        self.assertIsNone(backend._cohere_processor)
        self.assertFalse(backend.is_loaded)
        self.assertFalse(backend._cohere_compile_done)


if __name__ == "__main__":
    unittest.main()
