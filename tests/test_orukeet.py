"""Orukeet remains optional and fails closed on invalid downloaded files."""

import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib" / "src"))
from backends.onnx_asr_backend import OnnxAsrBackend
import orukeet


class CacheMiss(OSError):
    """Stand-in for the Hub's LocalEntryNotFoundError, not a disk I/O error."""


class OrukeetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        # The backend checks cache_state(); keep it off the user's real HF cache.
        cache = patch.object(orukeet, 'hub_cache_dir', return_value=Path(self.temp.name))
        cache.start()
        self.addCleanup(cache.stop)
        self.path = Path(self.temp.name) / 'models--oruk--orukeet' / 'snapshots' / orukeet.REVISION / orukeet.SUBFOLDER
        self.path.mkdir(parents=True)
        entries = {}
        for name in orukeet.FILES:
            data = name.encode()
            (self.path / name).write_bytes(data)
            entries[name] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        manifest = self.path / "manifest.json"
        manifest.write_text(json.dumps({"files": entries}))
        self.manifest_hash = orukeet.sha256(manifest)
        self.remote_files = {p.name: p.read_bytes() for p in self.path.iterdir()}
        self.calls = []
        self.hub = types.ModuleType("huggingface_hub")
        self.hub.hf_hub_download = self.fetch
        self.errors = types.ModuleType("huggingface_hub.errors")
        self.errors.LocalEntryNotFoundError = CacheMiss

    def fetch(self, **kwargs):
        self.calls.append(kwargs)
        self.assertEqual(kwargs["repo_id"], orukeet.REPO_ID)
        self.assertEqual(kwargs["revision"], orukeet.REVISION)
        self.assertEqual(Path(kwargs["filename"]).parent.as_posix(), orukeet.SUBFOLDER)
        path = self.path / Path(kwargs["filename"]).name
        if kwargs.get("local_files_only"):
            if not path.exists():
                raise CacheMiss(path.name)
        else:
            path.write_bytes(self.remote_files[path.name])
        return str(path)

    def download(self, offline=True, repair=False):
        with patch.dict(sys.modules, {"huggingface_hub": self.hub, "huggingface_hub.errors": self.errors}), \
             patch.object(orukeet, "MANIFEST_SHA256", self.manifest_hash):
            return orukeet.download_model(self.temp.name, offline=offline, repair=repair)

    def test_cached_files_are_verified_without_network(self):
        self.assertEqual(self.download(), self.path)
        self.assertEqual(len(self.calls), 0)
        self.assertTrue(all(call["local_files_only"] for call in self.calls))

    def test_empty_cache_downloads_pinned_files_then_reuses_them_offline(self):
        for path in self.path.iterdir():
            path.unlink()
        self.assertEqual(self.download(offline=False), self.path)
        names = ["manifest.json", *orukeet.FILES]
        self.assertEqual(len(self.calls), 2 * len(names))
        for index, name in enumerate(names):
            cached, online = self.calls[index * 2:index * 2 + 2]
            self.assertTrue(cached["local_files_only"])
            self.assertNotIn("local_files_only", online)
            self.assertEqual(cached["filename"], f"{orukeet.SUBFOLDER}/{name}")
            self.assertEqual(online["filename"], cached["filename"])
        self.calls.clear()
        self.assertEqual(self.download(), self.path)
        self.assertTrue(all(call["local_files_only"] for call in self.calls))

    def test_partial_cache_downloads_only_missing_files(self):
        missing = "decoder_joint-model.int8.onnx"
        (self.path / missing).unlink()
        self.assertEqual(self.download(offline=False), self.path)
        online = [call for call in self.calls if not call.get("local_files_only")]
        self.assertEqual([call["filename"] for call in online],
                         [f"{orukeet.SUBFOLDER}/{missing}"])

    def test_network_failure_preserves_cache_and_allows_retry(self):
        missing = "decoder_joint-model.int8.onnx"
        (self.path / missing).unlink()

        def unavailable(**kwargs):
            if not kwargs.get("local_files_only"):
                raise ConnectionError("network unavailable")
            return self.fetch(**kwargs)

        self.hub.hf_hub_download = unavailable
        with self.assertRaisesRegex(ConnectionError, "network unavailable"):
            self.download(offline=False)
        self.assertFalse((self.path / missing).exists())
        for name, data in self.remote_files.items():
            if name != missing:
                self.assertEqual((self.path / name).read_bytes(), data)
        self.hub.hf_hub_download = self.fetch
        self.assertEqual(self.download(offline=False), self.path)

    def test_cache_permission_error_does_not_attempt_network(self):
        self.hub.hf_hub_download = Mock(side_effect=PermissionError("cache unreadable"))
        with self.assertRaises(PermissionError):
            self.download(offline=False)
        self.hub.hf_hub_download.assert_called_once()
        self.assertTrue(self.hub.hf_hub_download.call_args.kwargs["local_files_only"])

    def test_corrupt_manifest_or_weights_are_rejected(self):
        for name in ["manifest.json", "encoder-model.int8.onnx", "config.json"]:
            with self.subTest(name=name):
                original = (self.path / name).read_bytes()
                (self.path / name).write_bytes(b"corrupt")
                with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                    self.download()
                (self.path / name).write_bytes(original)

    def test_same_size_corruption_is_rejected_without_redownloading(self):
        for name in ["manifest.json", "encoder-model.int8.onnx", "config.json"]:
            with self.subTest(name=name):
                original = (self.path / name).read_bytes()
                (self.path / name).write_bytes(bytes([original[0] ^ 1]) + original[1:])
                self.calls.clear()
                with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                    self.download(offline=False)
                self.assertTrue(all(call["local_files_only"] for call in self.calls))
                if name == "manifest.json":
                    self.assertEqual(len(self.calls), 1)
                (self.path / name).write_bytes(original)

    def test_verified_cache_skips_rehashing_until_a_file_changes(self):
        self.assertEqual(self.download(), self.path)
        with patch.object(orukeet, 'sha256', wraps=orukeet.sha256) as digest:
            self.assertEqual(self.download(), self.path)
        # Only the small release manifest is hashed on a warm start.
        self.assertEqual(digest.call_count, 1)
        weight = self.path / 'encoder-model.int8.onnx'
        original = weight.read_bytes()
        weight.write_bytes(bytes([original[0] ^ 1]) + original[1:])
        stat = weight.stat()
        os.utime(weight, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1))
        with self.assertRaisesRegex(ValueError, 'checksum mismatch: encoder'):
            self.download()

    def test_unwritable_stamp_still_returns_verified_model(self):
        with patch.object(Path, 'write_text', side_effect=PermissionError('read-only cache')):
            self.assertEqual(self.download(), self.path)

    def test_missing_offline_file_never_downloads(self):
        (self.path / 'manifest.json').unlink()
        with self.assertRaises(FileNotFoundError):
            self.download()
        self.assertEqual(self.calls, [])

    def test_missing_offline_weight_never_downloads(self):
        (self.path / 'encoder-model.int8.onnx').unlink()
        with self.assertRaises(FileNotFoundError):
            self.download()
        self.assertEqual(self.calls, [])

    def backend(self, model, quantization='int8'):
        settings = {'onnx_asr_model': model, 'onnx_asr_quantization': quantization, 'onnx_asr_use_vad': False}
        manager = types.SimpleNamespace(config=types.SimpleNamespace(get_setting=lambda key, default=None: settings.get(key, default)), ready=False, current_model=None)
        return OnnxAsrBackend(manager)

    def test_orukeet_uses_existing_tdt_loader(self):
        runtime = types.ModuleType('onnx_asr')
        runtime.load_model = Mock(return_value=object())
        with patch.dict(sys.modules, {'onnx_asr': runtime}), patch.object(orukeet, 'download_model', return_value=self.path):
            backend = self.backend('orukeet')
            self.assertTrue(backend.initialize())
            runtime.load_model.assert_called_once_with('nemo-conformer-tdt', path=self.path, quantization='int8')
            backend.unload()
            self.assertFalse(backend.is_loaded)

    def test_default_keeps_existing_loader(self):
        runtime = types.ModuleType('onnx_asr')
        runtime.load_model = Mock(return_value=object())
        with patch.dict(sys.modules, {'onnx_asr': runtime}), patch.object(orukeet, 'download_model') as download:
            self.assertTrue(self.backend('nemo-parakeet-tdt-0.6b-v3').initialize())
            download.assert_not_called()
            runtime.load_model.assert_called_once_with('nemo-parakeet-tdt-0.6b-v3', quantization='int8')

    def test_cache_state_is_read_only_and_tracks_verification(self):
        self.assertEqual(orukeet.cache_state(self.temp.name), 'unverified')
        with patch.object(orukeet, 'sha256') as digest:
            orukeet.cache_state(self.temp.name)
        digest.assert_not_called()
        self.assertFalse((self.path / orukeet.STAMP).exists())
        self.download()
        self.assertEqual(orukeet.cache_state(self.temp.name), 'verified')
        weight = self.path / 'vocab.txt'
        stat = weight.stat()
        os.utime(weight, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1))
        self.assertEqual(orukeet.cache_state(self.temp.name), 'unverified')
        weight.unlink()
        self.assertEqual(orukeet.cache_state(self.temp.name), 'missing')

    def test_repair_refetches_only_the_corrupt_file(self):
        name = 'encoder-model.int8.onnx'
        original = (self.path / name).read_bytes()
        (self.path / name).write_bytes(bytes([original[0] ^ 1]) + original[1:])
        self.assertEqual(self.download(offline=False, repair=True), self.path)
        self.assertEqual((self.path / name).read_bytes(), original)
        forced = [call['filename'] for call in self.calls if call.get('force_download')]
        self.assertEqual(forced, [f'{orukeet.SUBFOLDER}/{name}'])

    def test_offline_repair_still_fails_closed(self):
        (self.path / 'config.json').write_bytes(b'corrupt')
        with self.assertRaises(orukeet.ChecksumError):
            self.download(offline=True, repair=True)
        self.assertEqual(self.calls, [])

    def test_failed_verification_does_not_load_model_or_mark_ready(self):
        runtime = types.ModuleType('onnx_asr')
        runtime.load_model = Mock()
        with patch.dict(sys.modules, {'onnx_asr': runtime}), \
             patch.object(orukeet, 'download_model', side_effect=ValueError('checksum mismatch')):
            backend = self.backend('orukeet')
            self.assertFalse(backend.initialize())
            self.assertFalse(backend.is_loaded)
            self.assertFalse(backend.ready)
            runtime.load_model.assert_not_called()

    def test_repair_hint_only_for_checksum_errors(self):
        runtime = types.ModuleType('onnx_asr')
        for error, hinted in ((orukeet.ChecksumError('Orukeet checksum mismatch: vocab.txt'), True),
                              (ValueError('checksum mismatch in some library'), False)):
            with patch.dict(sys.modules, {'onnx_asr': runtime}), \
                 patch.object(orukeet, 'download_model', side_effect=error), \
                 contextlib.redirect_stdout(io.StringIO()) as out, \
                 contextlib.redirect_stderr(io.StringIO()):
                self.assertFalse(self.backend('orukeet').initialize())
            self.assertEqual("hyprwhspr model download" in out.getvalue(), hinted)

    def test_first_start_hint_only_while_cache_is_unverified(self):
        runtime = types.ModuleType('onnx_asr')
        runtime.load_model = Mock(return_value=object())
        for state, hinted in (('unverified', True), ('missing', True), ('verified', False)):
            with patch.dict(sys.modules, {'onnx_asr': runtime}), \
                 patch.object(orukeet, 'download_model', return_value=self.path), \
                 patch.object(orukeet, 'cache_state', return_value=state), \
                 contextlib.redirect_stdout(io.StringIO()) as out:
                self.assertTrue(self.backend('orukeet').initialize())
            self.assertEqual('first start' in out.getvalue(), hinted, state)

    def test_orukeet_ignores_the_quantization_setting(self):
        runtime = types.ModuleType('onnx_asr')
        runtime.load_model = Mock(return_value=object())
        with patch.dict(sys.modules, {'onnx_asr': runtime}), patch.object(orukeet, 'download_model', return_value=self.path):
            self.assertTrue(self.backend('orukeet', None).initialize())
        runtime.load_model.assert_called_once_with('nemo-conformer-tdt', path=self.path, quantization='int8')

if __name__ == '__main__':
    unittest.main()
