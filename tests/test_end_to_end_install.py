"""A real install/update/uninstall against a real payload on a real filesystem.

Every other test in this suite mocks Installation.update, so the pipeline the
install reviews keep probing - extract, verify, journal, activate, record, remove -
was never actually executed end to end. This builds a genuine release payload with
scripts/build-release.py (stdlib only, no network), runs the real lifecycle against
a scratch HOME, and drives the external commands through PATH shims rather than
mock.patch so the subprocess plumbing is exercised too.

These seams are stubbed, and no others:
  - download(), which is fed the locally built archive instead of GitHub;
  - Installation.build, which would pip-install into a fresh venv;
  - package_installed(), which reads the developer's real /usr/lib/hyprwhspr;
  - stable(), which is a fifteen-second wall-clock settle, not logic;
  - scripts/install-deps.sh, which installs distro packages with sudo and is
    documented in update() as being outside application rollback.

The systemctl shim reports an empty FragmentPath, so check_integrations' unit
rewrite branch is not reached here; it is covered by the unit tests instead.
"""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import uuid
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'lib/src'))
import managed_install as managed

SHIMS = ('systemctl', 'hyprctl', 'udevadm', 'gpasswd', 'sudo', 'git')


def build_payload(output):
    """Build a real release payload; returns the metadata dict."""
    subprocess.run([sys.executable, str(ROOT / 'scripts/build-release.py'), 'v1.0.0', str(output)],
                   check=True, capture_output=True, text=True, cwd=ROOT)
    return json.loads((output / 'hyprwhspr-release.json').read_text(encoding='utf-8'))


class EndToEndInstallTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = tempfile.TemporaryDirectory(prefix='hyprwhspr-payload-')
        cls.addClassCleanup(cls.payload.cleanup)
        cls.metadata = build_payload(Path(cls.payload.name))

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='hyprwhspr-e2e-')
        self.addCleanup(self.temp.cleanup)
        self.home = Path(self.temp.name)
        self.calls = self.home / 'shim-calls.log'
        bindir = self.home / 'bin'
        bindir.mkdir()
        for name in SHIMS:
            shim = bindir / name
            # Every shim logs and exits 0. Nothing re-executes: a test must never
            # be able to run a real privileged command on the developer's machine.
            # systemctl additionally answers `show`, so the installer sees a running
            # service and the stop -> activate -> start path is exercised.
            if name == 'systemctl':
                body = ('#!/bin/sh\n'
                        'printf "systemctl %s\\n" "$*" >> "$CALLS"\n'
                        'case "$*" in\n'
                        '  *show*) [ -f "$STATE_ACTIVE" ] && printf '
                        '"ActiveState=active\\nSubState=running\\nNRestarts=0\\nMainPID=1\\nFragmentPath=\\n" ;;\n'
                        '  *stop*) rm -f "$STATE_ACTIVE" ;;\n'
                        '  *start*|*enable*) : > "$STATE_ACTIVE" ;;\n'
                        'esac\n'
                        'exit 0\n')
            else:
                body = '#!/bin/sh\nprintf "%s %s\\n" "' + name + '" "$*" >> "$CALLS"\nexit 0\n'
            shim.write_text(body, encoding='utf-8')
            shim.chmod(0o755)

        self.env = mock.patch.dict(os.environ, {
            'HOME': str(self.home),
            'XDG_CONFIG_HOME': str(self.home / '.config'),
            'XDG_DATA_HOME': str(self.home / '.local/share'),
            'XDG_STATE_HOME': str(self.home / '.local/state'),
            'XDG_RUNTIME_DIR': str(self.home / 'run'),
            'CALLS': str(self.calls),
            'STATE_ACTIVE': str(self.home / 'service-active'),
            'PATH': f'{bindir}{os.pathsep}{os.environ["PATH"]}',
        })
        self.env.start()
        self.addCleanup(self.env.stop)
        (self.home / 'run').mkdir()
        self.install = managed.Installation()

    def mark_service_active(self):
        """Make the systemctl shim report a running service, as on a real update."""
        Path(os.environ['STATE_ACTIVE']).write_text('', encoding='utf-8')

    def shim_calls(self):
        return self.calls.read_text(encoding='utf-8').splitlines() if self.calls.exists() else []

    def fake_download(self, url, destination, maximum=None):
        """Serve the locally built release instead of GitHub."""
        source = getattr(self, 'payload_override', None) or Path(self.payload.name)
        if 'releases' in url and url.endswith(('per_page=100', '/tags/v1.0.0')):
            body = [{'draft': False, 'prerelease': False, 'tag_name': 'v1.0.0', 'assets': [
                {'name': 'hyprwhspr-release.json', 'browser_download_url': 'https://x/meta'},
                {'name': self.metadata['archive'], 'browser_download_url': 'https://x/archive'}]}]
            Path(destination).write_text(json.dumps(body if url.endswith('100') else body[0]),
                                         encoding='utf-8')
        elif url.endswith('/meta'):
            shutil.copy2(source / 'hyprwhspr-release.json', destination)
        else:
            shutil.copy2(source / self.metadata['archive'], destination)

    def fake_build(self, root, identity, kind, selection, old, tx, force):
        """Create an environment without pip or the network.

        A forced rebuild produces a genuinely new environment, as the real build
        does, so callers can drive a real re-activation rather than the
        "already installed and healthy" short-circuit.
        """
        suffix = f'{kind}-forced-{uuid.uuid4().hex[:8]}' if force else f'{kind}-test'
        target = self.install.data / 'environments' / suffix
        (target / 'bin').mkdir(parents=True, exist_ok=True)
        (target / 'bin/python').write_text('#!/bin/sh\nexit 0\n', encoding='utf-8')
        (target / 'bin/python').chmod(0o755)
        tx.setdefault('created', []).append(str(target))
        return {'path': str(target), 'key': {'kind': kind, 'build': suffix}}

    def install_once(self, **kwargs):
        identity = {'path': sys.executable, 'version': list(sys.version_info[:3]),
                    'identity': sys.version}
        deps = ROOT / 'scripts/install-deps.sh'

        def no_host_changes(command, **kwargs):
            if len(command) > 1 and str(command[1]).endswith('install-deps.sh'):
                with self.calls.open('a') as log:
                    log.write('install-deps.sh (skipped)\n')
                return subprocess.CompletedProcess(command, 0, '', '')
            return real_run(command, **kwargs)

        real_run = managed.run
        with mock.patch.object(managed, 'run', no_host_changes), \
             mock.patch.object(managed, 'download', self.fake_download), \
             mock.patch.object(managed.Installation, 'build', self.fake_build), \
             mock.patch.object(managed.Installation, 'build_backend',
                               lambda s, root, i, sel, old, tx, force: (
                                   self.fake_build(root, i, 'backend', sel, old, tx, force), sel)), \
             mock.patch.object(managed, 'interpreter', return_value=identity), \
             mock.patch.object(managed.Installation, 'stable', lambda s: None), \
             mock.patch.object(managed.Installation, 'package_installed', lambda s: False):
            return self.install.update(**kwargs)

    @unittest.skipUnless(importlib.util.find_spec('jsonschema') is not None,
                         'the config validator subprocess needs jsonschema')
    def test_repair_validates_destination_config_before_activation(self):
        first = self.install_once()
        destination = self.home / 'new config'
        config = destination / 'hyprwhspr/config.json'
        config.parent.mkdir(parents=True)
        config.write_text('{malformed', encoding='utf-8')
        real_run = managed.run

        def validate_with_test_python(command, **kwargs):
            # The fixture's fake venv Python always exits zero. Execute the real
            # validator using the test interpreter, keeping host commands mocked.
            if len(command) > 2 and command[1:3] == ['-s', '-c']:
                command = [sys.executable, *command[1:]]
            return real_run(command, **kwargs)

        with mock.patch.dict(os.environ, {'XDG_CONFIG_HOME': str(destination)}), \
             mock.patch.object(managed, 'run', side_effect=validate_with_test_python):
            with self.assertRaisesRegex(RuntimeError, 'Configuration validation failed') as failure:
                self.install_once(repair=True)
        self.assertIn(str(config), str(failure.exception))
        self.assertIn('JSONDecodeError', str(failure.exception))
        self.assertEqual(managed.read_json(self.install.current), first)
        self.assertEqual(config.read_text(encoding='utf-8'), '{malformed')

    def test_repair_repins_the_config_root_to_the_current_session(self):
        first = self.install_once()
        destination = self.home / 'new config'
        (destination / 'hyprwhspr').mkdir(parents=True)
        self.assertEqual(first['xdg']['config'], str(self.home / '.config'))
        with mock.patch.dict(os.environ, {'XDG_CONFIG_HOME': str(destination)}):
            repaired = self.install_once(repair=True)
        self.assertEqual(repaired['xdg']['config'], str(destination),
                         'install repair did not re-pin the config root')
        # And an ordinary update from elsewhere must not move it back.
        with mock.patch.dict(os.environ, {'XDG_CONFIG_HOME': str(self.home / 'somewhere-else')}):
            carried = self.install_once(force_backend=True)
        self.assertEqual(carried['xdg']['config'], str(destination))

    def test_real_install_then_uninstall_leaves_no_managed_state(self):
        self.mark_service_active()
        generation = self.install_once()

        # The payload really was extracted and verified.
        root = Path(generation['root'])
        self.assertTrue((root / 'lib/src/managed_install.py').is_file())
        self.assertEqual(managed.verify_payload(root)['version'], 'v1.0.0')
        self.assertEqual(generation['version'], 'v1.0.0')

        # Entry points exist and the pointer is live.
        self.assertTrue((self.install.data / 'launcher').is_file())
        self.assertTrue((self.install.data / 'interpreter').is_file())
        self.assertEqual(managed.read_json(self.install.current)['id'], generation['id'])

        # XDG roots were pinned from this scratch HOME, not the developer's.
        self.assertEqual(generation['xdg']['config'], str(self.home / '.config'))

        # Ownership was recorded for what was created.
        receipt = self.install.read_receipt()
        self.assertIn(str(self.install.data / 'launcher'), receipt['files'])

        # Real subprocesses ran, through PATH, not through mock.patch. A fresh
        # install deploys no unit (that is `systemd install`), so the probe is
        # what proves the plumbing rather than a daemon-reload.
        self.assertTrue(any('systemctl --user show' in call for call in self.shim_calls()),
                        self.shim_calls())
        # With the service reported active, activation must stop and restart it.
        # Match whole arguments: the `show` probe's NRestarts property contains
        # the substring 'start', which made a looser assertion vacuously true.
        issued = [call.split() for call in self.shim_calls()]
        self.assertTrue(any(args[:3] == ['systemctl', '--user', 'stop'] for args in issued), issued)
        self.assertTrue(any(args[:3] == ['systemctl', '--user', 'start'] for args in issued), issued)

        managed.uninstall(['--yes', '--purge'])

        self.assertFalse(self.install.current.exists())
        self.assertFalse((self.install.data / 'launcher').exists())
        leftovers = [p for p in (self.install.data / 'releases').glob('*')] if (
            self.install.data / 'releases').exists() else []
        self.assertEqual(leftovers, [], f'release payloads survived uninstall: {leftovers}')

    def test_repeating_an_update_is_idempotent_and_accumulates_nothing(self):
        first = self.install_once()
        second = self.install_once()
        # Same release and same environments: the second run must recognise the
        # generation as current rather than staging a duplicate.
        self.assertEqual(first['id'], second['id'])
        self.assertEqual(managed.read_json(self.install.current)['id'], first['id'])
        roots = list((self.install.data / 'releases').glob('*'))
        self.assertEqual(len(roots), 1, f'payloads accumulated across updates: {roots}')
        staged = list((self.install.data / 'transactions').glob('*')) if (
            self.install.data / 'transactions').exists() else []
        self.assertEqual(staged, [], f'transaction work directories leaked: {staged}')
        self.assertFalse(self.install.journal.exists())

    def test_a_tampered_payload_is_refused_and_nothing_is_activated(self):
        """The archive checksum must actually gate activation, not just be computed."""
        tampered = Path(self.temp.name) / 'tampered'
        tampered.mkdir()
        for name in (self.metadata['archive'], 'hyprwhspr-release.json'):
            shutil.copy2(Path(self.payload.name) / name, tampered / name)
        blob = bytearray((tampered / self.metadata['archive']).read_bytes())
        blob[-64:] = bytes(64)  # corrupt the tail, leave the recorded digest intact
        (tampered / self.metadata['archive']).write_bytes(bytes(blob))

        source = Path(self.payload.name)
        try:
            self.payload_override = tampered
            with self.assertRaises((ValueError, OSError)):
                self.install_once()
        finally:
            self.payload_override = None
        self.assertFalse(self.install.current.exists(),
                         'a tampered payload was activated')
        self.assertEqual(list((self.install.data / 'releases').glob('*'))
                         if (self.install.data / 'releases').exists() else [], [])

    def test_a_corrupt_generation_pointer_does_not_block_the_lifecycle(self):
        self.install_once()
        self.install.current.write_text('{truncated', encoding='utf-8')
        # Every command used to abort here with a bare JSONDecodeError.
        self.assertEqual(self.install.generation_state(), {})
        self.assertEqual(str(self.install.config_home()), os.environ['XDG_CONFIG_HOME'])
        self.assertTrue(list(self.install.data.glob('current.json.unreadable-*')))
        self.assertIsNotNone(self.install_once(), 'a fresh install could not recover')

    def test_install_pins_xdg_against_a_later_hostile_environment(self):
        first = self.install_once()
        pinned = first['xdg']['config']
        self.assertEqual(pinned, str(self.home / '.config'))
        # Run a real second lifecycle from a session with a different config root.
        with mock.patch.dict(os.environ, {'XDG_CONFIG_HOME': str(self.home / 'tty-session')}):
            # force_backend makes this a real rebuild rather than the
            # "already installed and healthy" short-circuit, so the new
            # generation genuinely re-evaluates the pinned roots.
            carried = self.install_once(force_backend=True)
            self.assertNotEqual(carried['id'], first['id'],
                                'expected a real rebuild, not the healthy short-circuit')
            self.assertEqual(carried['xdg']['config'], pinned,
                             'the pinned config root did not survive an update')
            self.assertEqual(str(managed.Installation().config_home()), pinned)
            unit = managed.service_content(Path(carried['root']), self.install.data / 'launcher')
        self.assertIn(f'Environment="XDG_CONFIG_HOME={pinned}"', unit)
        self.assertNotIn('tty-session', unit)


if __name__ == '__main__':
    unittest.main()
