"""#303: a legacy GPU install must migrate to its GPU plan, not silently to CPU."""
import io
import json
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib" / "src"))

import backend_installer  # noqa: E402
import dependency_plan  # noqa: E402
import managed_install as managed  # noqa: E402

UPSTREAM = f'https://github.com/{managed.REPOSITORY}.git'


def fingerprint(backend, variant):
    return dependency_plan.resolve(ROOT, backend, None, variant, ValueError).fingerprint


class LegacyGpuMigrationTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.install = managed.Installation(self.root / 'data', self.root / 'state')
        (self.install.data / 'src/.git').mkdir(parents=True)
        self.install.state.mkdir(parents=True, exist_ok=True)
        config = mock.patch.object(self.install, 'config_home', return_value=self.root / 'config')
        config.start()
        self.addCleanup(config.stop)

    def selection(self, state, root=ROOT):
        (self.install.state / 'install-state.json').write_text(json.dumps(state))
        with mock.patch.object(managed, 'run', return_value=mock.Mock(stdout=UPSTREAM + '\n')), \
                redirect_stdout(io.StringIO()):
            return self.install.legacy_selection(root)

    def state(self, backend, variant):
        # The four keys a legacy install really wrote, plus the configured backend.
        value = fingerprint(backend, variant)
        return {'install_state': 'completed', 'backend': backend, 'dependency_plan_fingerprint': value,
                'dependency_manifest_hash': value, 'dependency_family': 'x'}

    def gpu_package(self, name):
        site = self.install.data / 'venv/lib/python3.14/site-packages' / name
        site.mkdir(parents=True)

    def configure(self, backend, **settings):
        config = self.root / 'config/hyprwhspr/config.json'
        config.parent.mkdir(parents=True, exist_ok=True)
        config.write_text(json.dumps(dict(settings, transcription_backend=backend)))

    def test_configured_family_overrides_stale_whisper_state(self):
        self.configure('onnx-asr')
        state = dict(self.state('onnx-asr', 'gpu'), installed_backend='nvidia', accelerated_variant='nvidia')
        self.assertEqual(self.selection(state), {'backend': 'onnx-asr', 'variant': 'gpu'})

    def test_realtime_migration_preserves_provider_dependency_plan(self):
        self.configure('realtime-ws', websocket_provider='elevenlabs')
        selection = self.selection({'installed_backend': 'cpu', 'accelerated_variant': 'gpu'})
        self.assertEqual(selection, {'backend': 'realtime-ws', 'provider': 'elevenlabs', 'variant': None})
        plan = dependency_plan.resolve(ROOT, selection['backend'], selection['provider'], selection['variant'], ValueError)
        self.assertIn('elevenlabs', plan.required_imports)

    def test_legacy_rocm_variant_maps_to_managed_amd(self):
        self.configure('pywhispercpp')
        for state in ({'installed_backend': 'rocm'},
                      {'installed_backend': 'rocm', 'accelerated_variant': 'rocm'}):
            self.assertEqual(self.selection(state), {'backend': 'amd', 'variant': 'amd'})

    def test_recorded_whisper_cpu_fallback_is_preserved(self):
        for backend in ('pywhispercpp', 'nvidia', 'amd', 'vulkan'):
            with self.subTest(backend=backend):
                self.configure(backend)
                self.assertEqual(self.selection({'installed_backend': 'cpu'}), {'backend': 'cpu', 'variant': 'cpu'})

    def test_recorded_gpu_fingerprint_selects_the_gpu_plan(self):
        for backend, variant, key in (('onnx-asr', 'gpu', 'onnx-gpu'), ('faster-whisper', 'cuda', 'faster-cuda')):
            selection = self.selection(self.state(backend, variant))
            self.assertEqual(selection, {'backend': backend, 'variant': variant})
            self.assertEqual(dependency_plan.plan_key(backend, None, selection['variant'], ValueError), key)

    def test_config_backend_and_legacy_hash_without_explicit_release_root(self):
        config = self.root / 'config/hyprwhspr/config.json'
        config.parent.mkdir(parents=True)
        config.write_text(json.dumps({'transcription_backend': 'onnx-asr'}))
        original_path = sys.path[:]
        selection = self.selection({'dependency_manifest_hash': fingerprint('onnx-asr', 'gpu')}, root=None)
        self.assertEqual(selection, {'backend': 'onnx-asr', 'variant': 'gpu'})
        self.assertEqual(sys.path, original_path)

    def test_recorded_cpu_fingerprint_stays_cpu_even_with_a_gpu_package_left_behind(self):
        self.gpu_package('onnxruntime_gpu-1.32.0.dist-info')
        self.assertEqual(self.selection(self.state('onnx-asr', None)), {'backend': 'onnx-asr', 'variant': None})

    def test_gpu_fingerprint_matches_against_the_old_checkout_when_manifests_changed(self):
        state = self.state('onnx-asr', 'gpu')
        for name in ('requirements-onnx-asr-gpu.txt', 'requirements-onnx-asr.txt', 'requirements.txt'):
            shutil.copy(ROOT / name, self.install.data / 'src' / name)
        payload = self.root / 'payload'
        shutil.copytree(ROOT / 'lib/src', payload / 'lib/src', ignore=shutil.ignore_patterns('__pycache__'))
        for name in ('requirements-onnx-asr-gpu.txt', 'requirements-onnx-asr.txt', 'requirements.txt'):
            (payload / name).write_text((ROOT / name).read_text() + '# newer release\n')
        self.assertEqual(self.selection(state, payload)['variant'], 'gpu')

    def test_installed_gpu_package_decides_when_the_fingerprint_matches_nothing(self):
        stale = {'install_state': 'completed', 'backend': 'onnx-asr', 'dependency_plan_fingerprint': 'f' * 64}
        self.assertIsNone(self.selection(stale)['variant'])
        self.gpu_package('onnxruntime_gpu-1.32.0.dist-info')
        self.assertEqual(self.selection(stale)['variant'], 'gpu')
        self.assertEqual(self.selection({'backend': 'onnx-asr'}, root=None)['variant'], 'gpu')

    def test_faster_whisper_gpu_package_is_its_own_evidence(self):
        self.gpu_package('onnxruntime_gpu-1.32.0.dist-info')
        self.assertIsNone(self.selection({'backend': 'faster-whisper'})['variant'])
        self.gpu_package('nvidia_cudnn_cu12-9.1.0.dist-info')
        self.assertEqual(self.selection({'backend': 'faster-whisper'})['variant'], 'cuda')

    def test_explicit_and_whisper_cpp_variants_are_unchanged(self):
        self.assertEqual(self.selection({'backend': 'onnx-asr', 'accelerated_variant': 'gpu'})['variant'], 'gpu')
        self.assertEqual(self.selection({'installed_backend': 'nvidia'}), {'backend': 'nvidia', 'variant': 'nvidia'})
        self.assertEqual(self.selection({'backend': 'cohere-transcribe'})['variant'], None)

    def test_legacy_installer_now_records_the_variant_and_clears_it(self):
        with mock.patch.object(backend_installer, 'STATE_DIR', self.root / 'legacy-state'), \
                mock.patch.object(backend_installer, 'STATE_FILE', self.root / 'legacy-state/install-state.json'):
            backend_installer.commit_dependency_state(dependency_plan.resolve(ROOT, 'nvidia', None, 'nvidia', ValueError))
            for variant in ('gpu', None):
                backend_installer.commit_dependency_state(dependency_plan.resolve(ROOT, 'onnx-asr', None, variant, ValueError))
                state = json.loads((self.root / 'legacy-state/install-state.json').read_text())
                self.assertEqual(state.get('accelerated_variant'), variant)
                self.assertNotIn('installed_backend', state)


if __name__ == '__main__':
    unittest.main()
