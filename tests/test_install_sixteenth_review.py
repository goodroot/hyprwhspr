"""Journal inspection, replay preconditions and lifecycle edge cases."""
from contextlib import ExitStack
import io
import json
import os
from pathlib import Path
import sys
from unittest import mock

from tests import test_managed_install as fixtures
from tests import test_install_fifteenth_review as fifteenth
from tests.test_install_second_review import noctalia, ROOT
import managed_install as managed
import managed_integrations as integrations


class SixteenthReviewTests(fixtures.ManagedFixture):
    def test_invalid_journal_status_is_readonly_and_recovery_preserves_evidence(self):
        root = self.install.data / 'releases/preserved'
        root.mkdir(parents=True)
        current = {'root': str(root)}
        managed.atomic_json(self.install.current, current)
        for value in ('{broken', '[]', '{}', '{"phase":"activated","created":[]}'):
            with self.subTest(value=value):
                self.install.state.mkdir(parents=True, exist_ok=True)
                self.install.journal.write_text(value)
                status = self.install.status()
                self.assertIn(str(self.install.journal), status['transaction']['error'])
                self.assertEqual(self.install.journal.read_text(), value)
                self.install.recover()
                self.assertFalse(self.install.journal.exists())
                self.assertTrue(any(p.read_text() == value for p in self.install.state.glob('transaction-unreadable-*.json')))
                self.assertEqual(managed.read_json(self.install.current), current)
                self.install.remove([str(root)])
                self.assertTrue(root.exists())

    def test_stale_rollback_cannot_delete_recreated_or_modified_config(self):
        for snapshot in (None, {'content': 'original', 'mode': 0o644}):
            with self.subTest(snapshot=snapshot):
                path = self.root / 'bindings.conf'
                path.write_text('installer output')
                queue = self.install.state / 'deferred-restoration.json'
                managed.atomic_json(queue, {'files': {str(path): {'snapshot': snapshot,
                    'expected': {'sha256': managed.digest(path)}}}, 'protected': []})
                real_unlink = Path.unlink
                def unlink(candidate, *args, **kwargs):
                    if candidate == queue:
                        raise PermissionError('cannot retire queue')
                    return real_unlink(candidate, *args, **kwargs)
                with mock.patch.object(Path, 'unlink', unlink):
                    self.install.restore_integrations()
                self.assertTrue(queue.exists())
                path.write_text('new user configuration')
                for _ in range(2):
                    self.install.recover()
                    self.assertEqual(path.read_text(), 'new user configuration')
                self.assertTrue(queue.exists())

    def test_old_snapshot_without_precondition_preserves_file(self):
        path = self.root / 'config'
        path.write_text('user configuration')
        managed.atomic_json(self.install.state / 'deferred-restoration.json',
            {'files': {str(path): {'snapshot': None}}, 'protected': []})
        self.install.restore_integrations()
        self.assertEqual(path.read_text(), 'user configuration')

    def test_noctalia_shared_theme_drift_does_not_fail_owned_removal(self):
        plugin = self.root / 'plugin'
        plugin.mkdir()
        widget = plugin / 'widget.luau'
        widget.write_text('owned widget')
        settings, output = self.root / 'settings.toml', self.root / 'mic-osd.css'
        settings.write_text('Noctalia updated settings')
        output.write_text('Noctalia regenerated theme')
        managed.atomic_json(self.install.receipt, {'files': {
            str(widget): {'sha256': managed.digest(widget)},
            **{str(path): {'sha256': 'old', 'original': 'before', 'restore_sha256': 'old'} for path in (settings, output)}}})
        dst = {'plugin_dir': plugin, 'settings': settings, 'template_input': self.root / 'input', 'template_output': output}
        with mock.patch.dict(os.environ, {'HYPRWHSPR_GENERATION': '{}'}), \
                mock.patch.object(noctalia, '_noctalia_paths', return_value=dst), \
                mock.patch.object(noctalia, 'HYPRWHSPR_ROOT', str(ROOT)), \
                mock.patch.object(noctalia.shutil, 'which', return_value=None), \
                mock.patch.object(managed, 'Installation', return_value=self.install), \
                mock.patch.object(integrations, 'Installation', return_value=self.install):
            self.assertTrue(noctalia.setup_noctalia('remove'))
        self.assertFalse(widget.exists())
        self.assertEqual(settings.read_text(), 'Noctalia updated settings')
        self.assertEqual(output.read_text(), 'Noctalia regenerated theme')

    def test_uninstall_aborts_before_removing_entrypoints_if_daemon_survives_stop(self):
        launcher = self.install.data / 'launcher'
        launcher.parent.mkdir(parents=True)
        launcher.write_text('launcher')
        generation = {'root': str(self.install.data / 'releases/active')}
        managed.atomic_json(self.install.current, generation)
        managed.atomic_json(self.install.receipt, {'files': {str(launcher): managed.file_receipt(launcher, 'integration')}})
        with mock.patch.object(managed, 'Installation', return_value=self.install), \
                mock.patch.object(self.install, 'check_integrations'), \
                mock.patch.object(self.install, 'service_state', return_value={'ActiveState': 'activating'}), \
                mock.patch.object(self.install, 'service') as service, \
                mock.patch.object(self.install, 'daemon_running', return_value=True):
            with self.assertRaisesRegex(RuntimeError, 'Daemon did not stop gracefully'):
                managed.uninstall(['--yes'])
        service.assert_called_once_with('stop')
        self.assertTrue(launcher.exists())
        self.assertEqual(managed.read_json(self.install.current), generation)
        self.assertIn(str(launcher), managed.read_json(self.install.receipt)['files'])

    def test_bootstrap_missing_checksum_assets_and_entries_have_named_errors(self):
        import urllib.request
        source = (ROOT / 'scripts/install.sh').read_text().split("<<'PY'\n", 1)[1].split('\nPY\n', 1)[0]
        for with_checksums in (False, True):
            assets = [{'name': 'managed_install.py', 'browser_download_url': 'https://example/helper'}]
            if with_checksums:
                assets.append({'name': 'SHA256SUMS', 'browser_download_url': 'https://example/checks'})
            release = {'draft': False, 'prerelease': False, 'tag_name': 'v1.0.0', 'assets': assets}
            responses = [io.BytesIO(json.dumps([release]).encode()), io.BytesIO(b'\nchecksum another-file\n')]
            with mock.patch.object(sys, 'argv', ['bootstrap', '', '/python', '0', '']), \
                    mock.patch.object(urllib.request, 'urlopen', side_effect=responses):
                with self.assertRaises(SystemExit) as result:
                    exec(compile(source, 'install.sh bootstrap', 'exec'), {})
            self.assertIn('checksum' if with_checksums else 'No compatible', str(result.exception))

    def test_group_helper_import_works_through_package_path(self):
        import builtins
        import importlib
        sys.path.insert(0, str(ROOT / 'lib'))
        self.addCleanup(sys.path.remove, str(ROOT / 'lib'))
        with mock.patch.dict(sys.modules, {'src.managed_install': managed}):
            package_uninstall = importlib.import_module('src.cli.uninstall')
            with mock.patch.object(fifteenth, 'uninstall', package_uninstall):
                call = fifteenth.FifteenthReviewTests.legacy_uninstall(self)
                managed.atomic_json(self.install.receipt, {'permissions': [
                    {'kind': 'group', 'group': 'input', 'user': 'test-user', 'added': True}]})
                real_import = builtins.__import__
                def guarded(name, *args, **kwargs):
                    level = kwargs.get('level', args[3] if len(args) > 3 else 0)
                    if name == 'managed_install' and level == 0:
                        raise ImportError('flat import forbidden')
                    return real_import(name, *args, **kwargs)
                with mock.patch.object(builtins, '__import__', guarded), mock.patch.object(managed, 'remove_added_group') as remove:
                    call()
                remove.assert_called_once()
