"""Model publication and legacy corruption regressions; no network or native code."""
import hashlib
import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'lib' / 'src'))
import backend_installer as installer
import whisper_models


class ModelDownloadSafetyTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.payload = b'verified model'
        self.metadata = (len(self.payload), hashlib.sha256(self.payload).hexdigest())
        for name in ('PYWHISPERCPP_MODELS_DIR', 'QWEN3_ASR_MODELS_DIR'):
            self.stack.enter_context(mock.patch.object(installer, name, self.root))
        self.stack.enter_context(mock.patch.dict(whisper_models.WHISPER_MODELS,
                                                {'base': self.metadata, 'large': self.metadata}))
        self.stack.enter_context(mock.patch.object(installer, 'VAD_MODEL_METADATA', self.metadata))
        self.stack.enter_context(mock.patch.object(installer, 'get_state', return_value=None))
        self.state = self.stack.enter_context(mock.patch.object(installer, 'set_state'))
        self.stack.enter_context(mock.patch('managed_install.record_ownership', return_value=True))
        self.retrieve = self.stack.enter_context(mock.patch.object(installer.urllib.request, 'urlretrieve'))
        self.stack.enter_context(mock.patch.object(installer, '_verified_models', set()))

    def test_old_partial_above_previous_size_floor_is_invalid(self):
        path = self.root / 'ggml-small.en.bin'
        with path.open('wb') as f:
            f.truncate(30_000_000)
        self.assertFalse(installer.check_model_validity(path))

    def test_official_hash_overrides_a_saved_hash_of_corrupt_bytes(self):
        path = self.root / 'ggml-base.bin'
        bad = b'x' * len(self.payload)
        path.write_bytes(bad)
        with mock.patch.object(installer, 'get_state', return_value=hashlib.sha256(bad).hexdigest()):
            self.assertFalse(installer.check_model_validity(path))

    def test_repeated_loads_hash_a_pinned_model_once(self):
        path = self.root / 'ggml-base.bin'
        path.write_bytes(self.payload)
        with mock.patch.object(installer, 'compute_file_hash', wraps=installer.compute_file_hash) as digest:
            for _ in range(3):
                self.assertTrue(installer.check_model_validity(path))
        self.assertEqual(digest.call_count, 1)

    def test_recorded_digest_skips_hashing(self):
        path = self.root / 'ggml-base.bin'
        path.write_bytes(self.payload)
        with mock.patch.object(installer, 'get_state', return_value=self.metadata[1]), \
             mock.patch.object(installer, 'compute_file_hash') as digest:
            self.assertTrue(installer.check_model_validity(path))
        digest.assert_not_called()

    def test_replaced_file_is_hashed_again(self):
        path = self.root / 'ggml-base.bin'
        path.write_bytes(self.payload)
        self.assertTrue(installer.check_model_validity(path))
        path.unlink()
        path.write_bytes(b'x' * len(self.payload))
        self.assertFalse(installer.check_model_validity(path))

    def test_custom_hash_mismatch_does_not_fall_back_to_size(self):
        path = self.root / 'ggml-custom.bin'
        with path.open('wb') as f:
            f.truncate(30_000_000)
        with mock.patch.object(installer, 'get_state', return_value='wrong'):
            self.assertFalse(installer.check_model_validity(path))

    def test_invalid_download_preserves_existing_file_and_state(self):
        path = self.root / 'ggml-base.bin'
        path.write_bytes(b'old')
        self.retrieve.side_effect = lambda url, dest, **kw: dest.write_bytes(b'x' * len(self.payload))
        self.assertFalse(installer.download_pywhispercpp_model('base'))
        self.assertEqual(path.read_bytes(), b'old')
        self.assertEqual(list(self.root.iterdir()), [path])
        self.state.assert_not_called()

    def test_large_alias_downloads_pinned_large_v3(self):
        self.retrieve.side_effect = lambda url, dest, **kw: dest.write_bytes(self.payload)
        self.assertTrue(installer.download_pywhispercpp_model('large'))
        url = self.retrieve.call_args.args[0]
        self.assertIn(whisper_models.WHISPER_MODEL_REVISION, url)
        self.assertTrue(url.endswith('/ggml-large-v3.bin'))
        self.assertEqual((self.root / 'ggml-large.bin').read_bytes(), self.payload)

    def test_unsupported_name_does_not_write_or_download(self):
        self.assertFalse(installer.download_pywhispercpp_model('../bad'))
        self.retrieve.assert_not_called()
        self.assertEqual(list(self.root.iterdir()), [])

    def test_vad_overlapping_downloads_publish_only_complete_files(self):
        final = self.root / installer.VAD_MODEL_FILENAME
        paths = []
        def download(url, dest, **kw):
            paths.append(dest)
            dest.write_bytes(b'partial')
            if len(paths) == 1:
                self.assertFalse(final.exists())
                self.assertTrue(installer.download_vad_model())
            dest.write_bytes(self.payload)
        self.retrieve.side_effect = download
        self.assertTrue(installer.download_vad_model())
        self.assertNotEqual(*paths)
        self.assertEqual(final.read_bytes(), self.payload)
        self.assertEqual(list(self.root.iterdir()), [final])

    def test_vad_failure_and_cancellation_clean_up(self):
        for error in (OSError('network'), KeyboardInterrupt()):
            with self.subTest(error=type(error)):
                def download(url, dest, **kw):
                    dest.write_bytes(b'partial')
                    raise error
                self.retrieve.side_effect = download
                if isinstance(error, KeyboardInterrupt):
                    with self.assertRaises(KeyboardInterrupt):
                        installer.download_vad_model()
                else:
                    self.assertFalse(installer.download_vad_model())
                self.assertEqual(list(self.root.iterdir()), [])

    def test_vad_legacy_corruption_is_replaced(self):
        final = self.root / installer.VAD_MODEL_FILENAME
        final.write_bytes(b'x' * 600_000)
        self.retrieve.side_effect = lambda url, dest, **kw: dest.write_bytes(self.payload)
        self.assertTrue(installer.download_vad_model())
        self.assertEqual(final.read_bytes(), self.payload)

    def _qwen(self):
        metadata = {'repo': 'test/models', 'revision': 'pinned',
                    'decoder': ('decoder.gguf', *self.metadata),
                    'projector': ('projector.gguf', *self.metadata)}
        self.stack.enter_context(mock.patch.dict(installer.QWEN3_ASR_MODELS, {'test': metadata}))

    def test_qwen_overlapping_downloads_keep_their_own_staging_files(self):
        self._qwen()
        paths = []
        def download(url, dest, **kw):
            paths.append(dest)
            dest.write_bytes(b'partial')
            if len(paths) == 1:
                self.assertTrue(installer.download_qwen3_asr_model('test'))
                self.assertEqual(dest.read_bytes(), b'partial')
            dest.write_bytes(self.payload)
        self.retrieve.side_effect = download
        self.assertTrue(installer.download_qwen3_asr_model('test'))
        self.assertEqual(len(set(paths)), len(paths))
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), ['decoder.gguf', 'projector.gguf'])

    def test_qwen_cancellation_cleans_own_partial(self):
        self._qwen()
        other = self.root / '.other.partial'
        other.write_bytes(b'other download')
        def download(url, dest, **kw):
            dest.write_bytes(b'partial')
            raise KeyboardInterrupt
        self.retrieve.side_effect = download
        with self.assertRaises(KeyboardInterrupt):
            installer.download_qwen3_asr_model('test')
        self.assertEqual(list(self.root.iterdir()), [other])
