"""Native backend/installer tests: no model, native code, network or host writes."""
import ctypes
import hashlib
import io
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import threading
import types
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'lib' / 'src'))
import parakeet_cpp_runtime as runtime
import parakeet_cpp_installer as installer
from backends.parakeet_cpp_backend import ParakeetCppBackend
from whisper_manager import WhisperManager
from tests.text_injector_helpers import ConfigStub


class NativeTests(unittest.TestCase):
    def setUp(self):
        self.manager = types.SimpleNamespace(config=ConfigStub(), ready=False, current_model=None)
        self.backend = ParakeetCppBackend(self.manager)
        self.library = mock.Mock()
        self.library.parakeet_capi_load.return_value = 123
        self.backend._library = self.library
        self.backend._context = 123

    def test_exact_signatures_and_abi(self):
        self.library.parakeet_capi_abi_version.return_value = 6
        with mock.patch.object(runtime.ctypes, 'CDLL', return_value=self.library):
            self.assertIs(runtime.bind_library('/mock/library'), self.library)
        self.assertEqual(self.library.parakeet_capi_load.argtypes, [ctypes.c_char_p])
        self.assertIs(self.library.parakeet_capi_load.restype, ctypes.c_void_p)
        self.assertEqual(self.library.parakeet_capi_transcribe_pcm.argtypes,
                         [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_int])
        self.assertIs(self.library.parakeet_capi_transcribe_pcm.restype, ctypes.c_void_p)
        self.assertIsNone(self.library.parakeet_capi_free_string.restype)
        self.assertIsNone(self.library.parakeet_capi_free.restype)
        self.assertIs(self.library.parakeet_capi_last_error.restype, ctypes.c_char_p)
        self.library.parakeet_capi_abi_version.return_value = 5
        with mock.patch.object(runtime.ctypes, 'CDLL', return_value=self.library):
            with self.assertRaisesRegex(RuntimeError, 'ABI'):
                runtime.bind_library('/mock/library')

    def test_contiguous_float_mono_unicode_and_owned_output(self):
        result = ctypes.create_string_buffer('  こんにちは café  '.encode())
        pointer = ctypes.addressof(result)
        def transcribe(ctx, audio, count, rate, decoder):
            self.assertEqual((ctx, count, rate, decoder), (123, 5, 16000, 0))
            np.testing.assert_array_equal(np.ctypeslib.as_array(audio, shape=(count,)), np.arange(10)[::2])
            return pointer
        self.library.parakeet_capi_transcribe_pcm.side_effect = transcribe
        self.assertEqual(self.backend.transcribe(np.arange(10, dtype=np.float64)[::2]), 'こんにちは café')
        self.library.parakeet_capi_free_string.assert_called_once_with(pointer)

    def test_frees_output_even_when_decoding_fails(self):
        result = ctypes.create_string_buffer(b'\xff')
        self.library.parakeet_capi_transcribe_pcm.return_value = ctypes.addressof(result)
        self.assertEqual(self.backend.transcribe(np.ones(8)), '')
        self.library.parakeet_capi_free_string.assert_called_once_with(ctypes.addressof(result))

    def test_native_error_is_borrowed_and_resampling_is_shared(self):
        self.library.parakeet_capi_transcribe_pcm.return_value = None
        self.library.parakeet_capi_last_error.return_value = b'inference failed'
        with mock.patch.object(self.backend, '_resample_audio', return_value=np.ones(4)) as resample:
            self.assertEqual(self.backend.transcribe(np.ones(8), 32000), '')
            resample.assert_called_once()
            self.assertEqual(resample.call_args.args[1:], (32000, 16000))
        self.library.parakeet_capi_free_string.assert_not_called()
        self.library.parakeet_capi_last_error.assert_called_once_with(123)

    def test_empty_invalid_audio_and_unload(self):
        for audio in (np.array([]), np.ones((2, 3)), np.array([float('nan')])):
            self.assertEqual(self.backend.transcribe(audio), '')
        self.library.parakeet_capi_transcribe_pcm.assert_not_called()
        self.backend.cleanup()
        self.backend.unload()
        self.library.parakeet_capi_free.assert_called_once_with(123)
        self.assertFalse(self.backend.is_loaded)
        self.assertFalse(self.manager.ready)

    def test_initialize_failure_and_reinitialization(self):
        self.backend._context = None
        with (mock.patch.object(runtime, 'resolve_device', return_value='cpu'),
              mock.patch.object(runtime, 'library_path', return_value=mock.Mock(is_file=lambda: True)),
              mock.patch.object(runtime, 'model_installed', return_value=True),
              mock.patch.object(runtime, 'bind_library', return_value=self.library)):
            self.library.parakeet_capi_load.return_value = None
            self.assertFalse(self.backend.initialize())
            self.assertFalse(self.manager.ready)
            self.library.parakeet_capi_load.return_value = 456
            self.assertTrue(self.backend.initialize())
            self.assertTrue(self.backend.initialize())
            self.assertEqual(self.library.parakeet_capi_load.call_count, 2)
            self.assertTrue(self.backend.reinitialize())
            self.library.parakeet_capi_free.assert_called_once_with(456)

    def test_cleanup_waits_for_manager_model_lock(self):
        manager = WhisperManager(config_manager=ConfigStub({'transcription_backend': 'parakeet-cpp'}))
        backend = ParakeetCppBackend(manager)
        backend._context, backend._library = 123, self.library
        manager._backend = backend
        entered = threading.Event()
        def cleanup():
            entered.set()
            manager.cleanup()
        with manager._model_lock:
            worker = threading.Thread(target=cleanup)
            worker.start()
            self.assertTrue(entered.wait(1))
            self.library.parakeet_capi_free.assert_not_called()
        worker.join(2)
        self.assertFalse(worker.is_alive())
        self.library.parakeet_capi_free.assert_called_once_with(123)

    def test_resume_reinitialize_waits_for_inference(self):
        manager = WhisperManager(config_manager=ConfigStub({'transcription_backend': 'parakeet-cpp'}))
        backend = ParakeetCppBackend(manager)
        backend._context, backend._library = 123, self.library
        manager._backend = backend
        with mock.patch.object(backend, 'initialize', return_value=True):
            with manager._model_lock:
                worker = threading.Thread(target=manager.reinitialize_after_resume)
                worker.start()
                worker.join(0.2)
                self.assertTrue(worker.is_alive())
                self.library.parakeet_capi_free.assert_not_called()
            worker.join(2)
        self.assertFalse(worker.is_alive())
        self.library.parakeet_capi_free.assert_called_once_with(123)


class RuntimeTests(unittest.TestCase):
    def test_architecture_matrix_and_rejection(self):
        for host, expected in [('x86_64', 'x64'), ('amd64', 'x64'), ('aarch64', 'arm64'), ('arm64', 'arm64')]:
            with mock.patch.object(runtime.platform, 'system', return_value='Linux'), mock.patch.object(runtime.platform, 'machine', return_value=host):
                self.assertEqual(runtime.architecture(), expected)
                for device in ('cpu', 'vulkan'):
                    self.assertIn((expected, device), runtime.ASSETS)
        for system, host in [('Darwin', 'arm64'), ('Linux', 'i686')]:
            with mock.patch.object(runtime.platform, 'system', return_value=system), mock.patch.object(runtime.platform, 'machine', return_value=host):
                with self.assertRaises(RuntimeError):
                    runtime.architecture()

    def test_auto_preserves_selection_reinstall_probes_and_explicit_wins(self):
        with tempfile.TemporaryDirectory() as temp, mock.patch.object(runtime, 'RUNTIMES_DIR', Path(temp)), mock.patch.object(runtime, 'architecture', return_value='x64'), mock.patch.object(runtime, 'probe_library', return_value=(True, '')), mock.patch.object(runtime, 'detect_device', return_value='vulkan') as detect:
            for device in ('cpu', 'vulkan'):
                path = runtime.library_path(device)
                path.parent.mkdir(parents=True)
                path.touch()
            runtime.selection_path().write_text('cpu')
            self.assertEqual(runtime.resolve_device(ConfigStub()), 'cpu')
            detect.assert_not_called()
            self.assertEqual(runtime.resolve_device(ConfigStub(), force_probe=True), 'vulkan')
            self.assertEqual(runtime.resolve_device(ConfigStub({'parakeet_cpp_device': 'cpu'}), True), 'cpu')

    def test_service_resolution_never_probes_and_skips_missing_selection(self):
        with tempfile.TemporaryDirectory() as temp, mock.patch.object(runtime, 'RUNTIMES_DIR', Path(temp)), mock.patch.object(runtime, 'architecture', return_value='x64'), mock.patch.object(runtime, 'probe_library') as probe, mock.patch.object(runtime, 'detect_device', return_value='vulkan'):
            path = runtime.library_path('cpu')
            path.parent.mkdir(parents=True)
            path.touch()
            # Stale selection: its library is gone, so the installed one wins.
            runtime.selection_path().write_text('vulkan')
            self.assertEqual(runtime.resolve_device(ConfigStub(), probe=False), 'cpu')
            path.unlink()
            self.assertEqual(runtime.resolve_device(ConfigStub(), probe=False), 'vulkan')
            probe.assert_not_called()

    def test_probe_is_bounded_and_loader_failure_is_reported(self):
        with mock.patch.object(runtime.subprocess, 'run', side_effect=subprocess.TimeoutExpired('probe', 20)) as run:
            self.assertFalse(runtime.probe_library('/mock/library')[0])
            self.assertEqual(run.call_args.kwargs['timeout'], 20)
        with mock.patch.object(runtime.subprocess, 'run', return_value=types.SimpleNamespace(returncode=1, stdout='', stderr='missing GLIBC')):
            self.assertEqual(runtime.probe_library('/mock/library'), (False, 'missing GLIBC'))


class InstallerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name, value in [('RUNTIMES_DIR', self.root / 'runtime'), ('MODELS_DIR', self.root / 'models')]:
            patcher = mock.patch.object(runtime, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = mock.patch.object(installer, '_record')
        self.record = patcher.start()
        self.addCleanup(patcher.stop)

    def archive(self, name=None, kind=tarfile.REGTYPE):
        path = self.root / 'test.tar.gz'
        with tarfile.open(path, 'w:gz') as bundle:
            for filename in runtime.ARCHIVE_FILES:
                info = tarfile.TarInfo('package/' + filename)
                info.size = 3
                bundle.addfile(info, io.BytesIO(b'abc'))
            if name:
                info = tarfile.TarInfo(name)
                info.type = kind
                info.linkname = '/tmp/escape'
                bundle.addfile(info, io.BytesIO(b''))
        return path

    def test_safe_archive_and_rejects_unexpected_members(self):
        destination = self.root / 'output'
        destination.mkdir()
        installer.extract_runtime(self.archive(), destination, 'package')
        self.assertEqual(set(p.name for p in destination.iterdir()), set(runtime.ARCHIVE_FILES))
        for index, (name, kind) in enumerate([('../escape', tarfile.REGTYPE), ('/tmp/escape', tarfile.REGTYPE), ('package/extra.so', tarfile.REGTYPE), ('package/libparakeet.so', tarfile.SYMTYPE), ('package/../escape', tarfile.REGTYPE)]):
            with self.subTest(name=name):
                out = self.root / str(index)
                out.mkdir()
                with self.assertRaises(RuntimeError):
                    installer.extract_runtime(self.archive(name, kind), out, 'package')

    def test_checksum_failure_and_interruption_preserve_installed_runtime(self):
        with mock.patch.object(runtime, 'architecture', return_value='x64'):
            target = runtime.library_path('cpu')
            target.parent.mkdir(parents=True)
            target.write_bytes(b'old')
            for failure in (False, KeyboardInterrupt()):
                download = mock.Mock(side_effect=failure if isinstance(failure, BaseException) else None)
                with mock.patch.object(installer, '_primitives', return_value=(download, lambda *args: False)):
                    with self.assertRaises(BaseException):
                        installer.install_runtime('cpu', force=True)
                self.assertEqual(target.read_bytes(), b'old')
                self.assertFalse(list(target.parent.parent.glob('.parakeet-*')))

    def test_automatic_compatibility_fallback_but_explicit_vulkan_fails(self):
        with (mock.patch.object(runtime, 'resolve_device', return_value='vulkan'),
              mock.patch.object(installer, 'install_runtime', side_effect=[RuntimeError('GLIBC'), None]) as install,
              mock.patch.object(installer, 'download_model', return_value=True),
              mock.patch.object(runtime, 'selection_path', return_value=self.root / 'selected')):
            self.assertTrue(installer.install_payload(ConfigStub()))
            self.assertEqual([c.args[0] for c in install.call_args_list], ['vulkan', 'cpu'])
            self.assertEqual((self.root / 'selected').read_text().strip(), 'cpu')
        with mock.patch.object(runtime, 'resolve_device', return_value='vulkan'), mock.patch.object(installer, 'install_runtime', side_effect=RuntimeError('GLIBC')) as install:
            self.assertFalse(installer.install_payload(ConfigStub({'parakeet_cpp_device': 'vulkan'})))
            self.assertEqual(install.call_count, 1)

    def test_model_atomic_verification_and_ownership(self):
        metadata = dict(runtime.MODEL, size=3, sha256=hashlib.sha256(b'new').hexdigest())
        def matches(path, size, digest):
            return path.is_file() and path.stat().st_size == size and hashlib.sha256(path.read_bytes()).hexdigest() == digest
        target = runtime.model_path()
        target.parent.mkdir(parents=True)
        target.write_bytes(b'old')
        with mock.patch.object(runtime, 'MODEL', metadata):
            for content, success in [(b'bad', False), (b'new', True)]:
                with mock.patch.object(installer, '_primitives', return_value=(lambda url, path, size: path.write_bytes(content), matches)):
                    self.assertEqual(installer.download_model(), success)
                self.assertEqual(target.read_bytes(), b'new' if success else b'old')
            self.record.assert_called_once_with(target, 'model')
            self.assertFalse(list(target.parent.glob('.parakeet-*')))
