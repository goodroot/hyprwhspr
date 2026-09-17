"""Orukeet remains optional and fails closed on invalid downloaded files."""

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib"))
from src.backends.onnx_asr_backend import OnnxAsrBackend
from src.backends import orukeet


class OrukeetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        entries = {}
        for name in orukeet.FILES:
            data = name.encode()
            (self.path / name).write_bytes(data)
            entries[name] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        manifest = self.path / "manifest.json"
        manifest.write_text(json.dumps({"files": entries}))
        self.manifest_hash = orukeet.sha256(manifest)
        self.calls = []
        self.hub = types.ModuleType("huggingface_hub")
        self.hub.hf_hub_download = self.fetch
        self.errors = types.ModuleType("huggingface_hub.errors")
        self.errors.LocalEntryNotFoundError = FileNotFoundError

    def fetch(self, **kwargs):
        self.calls.append(kwargs)
        self.assertEqual(kwargs["repo_id"], orukeet.REPO_ID)
        self.assertEqual(kwargs["revision"], orukeet.REVISION)
        return str(self.path / Path(kwargs["filename"]).name)

    def download(self, offline=True):
        with patch.dict(sys.modules, {"huggingface_hub": self.hub, "huggingface_hub.errors": self.errors}), \
             patch.object(orukeet, "MANIFEST_SHA256", self.manifest_hash):
            return orukeet.download_model(offline=offline)

    def test_cached_files_are_verified_without_network(self):
        self.assertEqual(self.download(), self.path)
        self.assertEqual(len(self.calls), len(orukeet.FILES) + 1)
        self.assertTrue(all(call["local_files_only"] for call in self.calls))

    def test_corrupt_manifest_or_weights_are_rejected(self):
        for name in ["manifest.json", "encoder-model.int8.onnx", "config.json"]:
            with self.subTest(name=name):
                original = (self.path / name).read_bytes()
                (self.path / name).write_bytes(b"corrupt")
                with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                    self.download()
                (self.path / name).write_bytes(original)

    def test_missing_offline_file_never_downloads(self):
        def missing(**kwargs):
            self.calls.append(kwargs)
            raise FileNotFoundError("missing")
        self.hub.hf_hub_download = missing
        with self.assertRaises(FileNotFoundError):
            self.download()
        self.assertEqual(len(self.calls), 1)
        self.assertTrue(self.calls[0]["local_files_only"])

    def backend(self, model, quantization='int8'):
        settings = {'onnx_asr_model': model, 'onnx_asr_quantization': quantization, 'onnx_asr_use_vad': False}
        manager = types.SimpleNamespace(config=types.SimpleNamespace(get_setting=lambda key, default=None: settings.get(key, default)), ready=False, current_model=None)
        return OnnxAsrBackend(manager)

    def test_orukeet_uses_existing_tdt_loader(self):
        runtime = types.ModuleType('onnx_asr')
        from unittest.mock import Mock
        runtime.load_model = Mock(return_value=object())
        with patch.dict(sys.modules, {'onnx_asr': runtime}), patch.object(orukeet, 'download_model', return_value=self.path):
            backend = self.backend('orukeet')
            self.assertTrue(backend.initialize())
            runtime.load_model.assert_called_once_with('nemo-conformer-tdt', path=self.path, quantization='int8')
            backend.unload()
            self.assertFalse(backend.is_loaded)

    def test_default_keeps_existing_loader(self):
        runtime = types.ModuleType('onnx_asr')
        from unittest.mock import Mock
        runtime.load_model = Mock(return_value=object())
        with patch.dict(sys.modules, {'onnx_asr': runtime}), patch.object(orukeet, 'download_model') as download:
            self.assertTrue(self.backend('nemo-parakeet-tdt-0.6b-v3').initialize())
            download.assert_not_called()
            runtime.load_model.assert_called_once_with('nemo-parakeet-tdt-0.6b-v3', quantization='int8')

    def test_unsupported_quantization_fails_before_download(self):
        runtime = types.ModuleType('onnx_asr')
        with patch.dict(sys.modules, {'onnx_asr': runtime}), patch.object(orukeet, 'download_model') as download:
            self.assertFalse(self.backend('orukeet', 'fp32').initialize())
            download.assert_not_called()


if __name__ == '__main__':
    unittest.main()
