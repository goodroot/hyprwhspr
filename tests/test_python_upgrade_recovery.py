"""Python replacement and bootstrap recovery, with all host operations mocked."""
from contextlib import ExitStack
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import urllib.request
from unittest import mock

from tests.test_managed_install import ManagedFixture
import managed_install as managed

ROOT = Path(__file__).resolve().parents[1]


class PythonUpgradeRecoveryTests(ManagedFixture):
    def test_bootstrap_repair_pins_release_unless_explicitly_overridden(self):
        current = self.root / 'xdg-data/hyprwhspr/current.json'
        managed.atomic_json(current, {'version': 'v1.0.0'})
        source = (ROOT / 'scripts/install.sh').read_text().split("<<'PY'\n", 1)[1].split('\nPY\n', 1)[0]
        helper = b'# fixture helper; never executed\n'
        checksum = hashlib.sha256(helper).hexdigest()
        for explicit in ('', 'v2.0.0'):
            version = explicit or 'v1.0.0'
            base = f'https://example.test/{version}'
            release = {'draft': False, 'prerelease': False, 'tag_name': version,
                       'assets': [{'name': name, 'browser_download_url': f'{base}/{name}'}
                                  for name in ('managed_install.py', 'SHA256SUMS')]}
            responses = [io.BytesIO(json.dumps(release).encode()),
                         io.BytesIO(f'{checksum}  managed_install.py\n'.encode()),
                         io.BytesIO(helper)]
            def invoke(command):
                self.assertEqual(Path(command[3]).read_bytes(), helper)
                self.assertEqual(command[:3], ['/fixture/python3.15', '-I', '-B'])
                self.assertEqual(command[4:], ['bootstrap', '--version', version,
                                             '--python', '/fixture/python3.15', '--repair'])
                return 0
            with self.subTest(explicit=explicit), \
                    mock.patch.object(sys, 'argv', ['bootstrap', explicit, '/fixture/python3.15', '1', '/fixture/python3.15']), \
                    mock.patch.object(urllib.request, 'install_opener'), \
                    mock.patch.object(urllib.request, 'urlopen', side_effect=responses) as fetch, \
                    mock.patch.object(subprocess, 'call', side_effect=invoke) as call:
                with self.assertRaises(SystemExit) as result:
                    exec(compile(source, 'bootstrap', 'exec'), {})
                self.assertEqual(result.exception.code, 0)
                call.assert_called_once()
                self.assertEqual([c.args[0].full_url for c in fetch.call_args_list],
                                 [f'https://api.github.com/repos/goodroot/hyprwhspr/releases/tags/{version}',
                                  f'{base}/SHA256SUMS', f'{base}/managed_install.py'])
                self.assertEqual(managed.read_json(current), {'version': 'v1.0.0'})

    def prepare_environments(self, stack):
        payload = self.install.data / 'releases/fixture'
        payload.mkdir(parents=True)
        for name in ('requirements-cli.txt', 'requirements.txt'):
            (payload / name).write_text('rich\n')
        (payload / 'requirements-rest.txt').write_text('-r requirements.txt\n')
        identity = {'path': '/fixture/python3', 'version': [3, 14, 0], 'identity': 'Python 3.14'}
        selection = {'backend': 'rest-api'}
        tx = {'work': str(self.root / 'work'), 'created': []}
        def run(command, **kwargs):
            # Stand in for venv creation without executing any interpreter.
            self.assertIn('venv', command)
            Path(command[-1]).mkdir(parents=True)
            return mock.Mock(returncode=0)
        runner = stack.enter_context(mock.patch.object(managed, 'run', side_effect=run))
        pip = stack.enter_context(mock.patch.object(managed.subprocess, 'run'))
        stack.enter_context(mock.patch.object(self.install, 'healthy', return_value=True))
        old = {'format': 1, 'version': 'v1.0.0', 'root': str(payload),
               'python': identity, 'selection': selection}
        for kind in ('cli', 'backend'):
            old[kind] = self.install.build(payload, identity, kind, selection, None, tx, False)
        self.install.journal.unlink()
        managed.atomic_json(self.install.current, old)
        runner.reset_mock()
        pip.reset_mock()
        return old, runner, pip

    def test_same_path_new_identity_rebuilds_both_environments(self):
        with ExitStack() as stack:
            old, runner, pip = self.prepare_environments(stack)
            identity = dict(old['python'], version=[3, 15, 0], identity='Python 3.15')
            tx = {'work': str(self.root / 'work'), 'created': []}
            for kind in ('cli', 'backend'):
                # Healthy, unchanged environments still reuse the previous key.
                reused = self.install.build(Path(old['root']), old['python'], kind,
                                            old['selection'], old, tx, False)
                self.assertEqual(reused, old[kind])
            runner.assert_not_called()
            for kind in ('cli', 'backend'):
                rebuilt = self.install.build(Path(old['root']), identity, kind,
                                             old['selection'], old, tx, False)
                self.assertNotEqual(rebuilt['path'], old[kind]['path'])
                self.assertEqual(rebuilt['key']['python'], identity)
            self.assertEqual(runner.call_count, 2)
            self.assertEqual(pip.call_count, 2)

    def test_failed_replacement_build_keeps_old_generation(self):
        with ExitStack() as stack:
            old, runner, pip = self.prepare_environments(stack)
            identity = dict(old['python'], version=[3, 15, 0], identity='Python 3.15')
            stack.enter_context(mock.patch.object(managed, 'interpreter', return_value=identity))
            stack.enter_context(mock.patch.object(managed, 'verify_payload', return_value={'version': 'v1.0.0'}))
            service = stack.enter_context(mock.patch.object(self.install, 'service'))
            activate = stack.enter_context(mock.patch.object(self.install, 'install_launcher'))
            pip.side_effect = [mock.Mock(returncode=0), subprocess.CalledProcessError(1, ['pip'])]
            with self.assertRaises(subprocess.CalledProcessError):
                self.install.update(local_payload=True)
            self.assertEqual(runner.call_count, 2)
            self.assertEqual(managed.read_json(self.install.current), old)
            for kind in ('cli', 'backend'):
                self.assertTrue(Path(old[kind]['path']).is_dir())
            self.assertEqual(set((self.install.data / 'environments').iterdir()),
                             {Path(old[kind]['path']) for kind in ('cli', 'backend')})
            service.assert_not_called()
            activate.assert_not_called()
