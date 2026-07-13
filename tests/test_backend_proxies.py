import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib" / "src"))

from backends import BACKENDS, TranscriptionBackend  # noqa: E402
from whisper_manager import WhisperManager  # noqa: E402


class FakeConfig:
    def get_setting(self, key, default=None):
        return default

    def get_temp_directory(self):
        return "/tmp"


class BackendProxyTests(unittest.TestCase):
    def test_shared_state_writes_through_to_manager(self):
        manager = WhisperManager(config_manager=FakeConfig())
        backend = TranscriptionBackend(manager)

        backend.ready = True
        self.assertTrue(manager.ready)

        backend.current_model = "base"
        self.assertEqual(manager.current_model, "base")

        backend._last_use_time = 123.0
        self.assertEqual(manager._last_use_time, 123.0)

        self.assertIs(backend.config, manager.config)
        self.assertEqual(backend.temp_dir, manager.temp_dir)

    def test_all_backends_registered(self):
        self.assertEqual(
            set(BACKENDS),
            {
                "pywhispercpp",
                "onnx-asr",
                "faster-whisper",
                "cohere-transcribe",
                "realtime-ws",
                "rest-api",
            },
        )
        for name, cls in BACKENDS.items():
            self.assertEqual(cls.name, name)
            self.assertTrue(issubclass(cls, TranscriptionBackend))


if __name__ == "__main__":
    unittest.main()
