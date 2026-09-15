import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

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
                "qwen3-asr",
            },
        )
        for name, cls in BACKENDS.items():
            self.assertEqual(cls.name, name)
            self.assertTrue(issubclass(cls, TranscriptionBackend))

    def test_backend_registry_import_does_not_require_requests(self):
        script = f'''\
import importlib.abc
import sys

class BlockRequests(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "requests" or fullname.startswith("requests."):
            raise ModuleNotFoundError("No module named 'requests'")
        return None

sys.meta_path.insert(0, BlockRequests())
sys.path.insert(0, {str(ROOT / "lib" / "src")!r})
from whisper_manager import WhisperManager
from backends import BACKENDS
assert "qwen3-asr" in BACKENDS
assert WhisperManager is not None
'''
        result = subprocess.run(
            [sys.executable, '-I', '-c', script],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_rest_backend_loads_requests_when_initialized(self):
        config = mock.Mock()
        config.get_setting.side_effect = lambda key, default=None: {
            'rest_endpoint_url': 'https://example.invalid/transcribe',
        }.get(key, default)
        manager = SimpleNamespace(
            config=config,
            current_model='old',
            ready=False,
            temp_dir='/tmp',
            _last_use_time=0.0,
        )
        backend = BACKENDS['rest-api'](manager)
        client = object()

        with mock.patch(
                'backends.rest_api_backend.require_package',
                return_value=client,
        ) as require:
            self.assertTrue(backend.initialize())

        require.assert_called_once_with('requests')
        self.assertIs(backend._requests, client)


if __name__ == "__main__":
    unittest.main()
