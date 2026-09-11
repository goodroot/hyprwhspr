"""Rollback isolation, atomic units and truthful desktop removal."""
from contextlib import ExitStack
import io
import os
from pathlib import Path
from unittest import mock

from tests import test_managed_install as fixtures
from tests import test_install_sixth_review as sixth
from tests.test_install_second_review import noctalia, ROOT
import managed_install as managed
import managed_integrations as integrations


class FourteenthReviewTests(fixtures.ManagedFixture):
    def test_noctalia_preserved_plugin_reports_incomplete(self):
        plugin = self.root / 'plugin'
        plugin.mkdir()
        widget = plugin / 'widget.luau'
        widget.write_text('modified plugin')
        dst = {'plugin_dir': plugin, 'settings': self.root / 'settings',
               'template_input': self.root / 'input', 'template_output': self.root / 'output'}
        with mock.patch.dict(os.environ, {'HYPRWHSPR_GENERATION': '{}'}), \
                mock.patch.object(noctalia, '_noctalia_paths', return_value=dst), \
                mock.patch.object(noctalia, 'HYPRWHSPR_ROOT', str(ROOT)), \
                mock.patch.object(noctalia.shutil, 'which', return_value=None), \
                mock.patch.object(managed, 'Installation', return_value=self.install), \
                mock.patch.object(integrations, 'Installation', return_value=self.install):
            self.assertFalse(noctalia.setup_noctalia('remove'))
            with self.assertRaisesRegex(RuntimeError, 'Noctalia removal incomplete'):
                noctalia.noctalia_command('remove')
        self.assertTrue(widget.exists())

    def test_atomic_owned_write_preserves_previous_unit_and_receipt_on_failure(self):
        unit = self.root / 'hyprwhspr.service'
        unit.write_text('old unit')
        unit.chmod(0o640)
        managed.atomic_json(self.install.receipt, {'files': {str(unit): managed.file_receipt(unit, 'integration')}})
        before = self.install.receipt.read_bytes()
        with mock.patch.object(managed, 'Installation', return_value=self.install):
            with mock.patch.object(managed.os, 'replace', side_effect=OSError('disk full')):
                with self.assertRaises(OSError):
                    managed.write_owned(unit, 'new unit')
            self.assertEqual(unit.read_text(), 'old unit')
            self.assertEqual(self.install.receipt.read_bytes(), before)
            managed.write_owned(unit, 'new unit')
        self.assertEqual(unit.read_text(), 'new unit')
        self.assertEqual(unit.stat().st_mode & 0o777, 0o640)

    def test_each_failed_restore_isolated_pointer_and_other_files_recover(self):
        for operation in ('unlink', 'mkdir', 'symlink', 'replace', 'text', 'receipt'):
            with self.subTest(operation=operation):
                bad = self.root / f'bad-{operation}'
                good = self.root / f'good-{operation}'
                bad.write_text('replacement')
                good.write_text('replacement')
                staged = self.install.data / 'environments' / operation
                staged.mkdir(parents=True)
                old = {'root': 'old'}
                snapshot = None if operation == 'unlink' else (
                    {'target': 'original-target'} if operation in ('mkdir', 'symlink', 'replace') else
                    {'content': 'original', 'mode': 0o640})
                managed.atomic_json(self.install.current, {'root': str(staged)})
                managed.atomic_json(self.install.journal, {'phase': 'activated', 'old': old,
                    'created': [str(staged)], 'running': True,
                    'integrations': {str(bad): snapshot, str(good): {'content': 'original', 'mode': 0o644}},
                    'receipt': {'files': {}}})
                with ExitStack() as stack:
                    stack.enter_context(mock.patch.object(self.install, 'service_state', return_value={}))
                    service = stack.enter_context(mock.patch.object(self.install, 'service'))
                    stack.enter_context(mock.patch.object(self.install, 'daemon_running', return_value=False))
                    if operation in ('unlink', 'mkdir', 'symlink'):
                        method = {'unlink': 'unlink', 'mkdir': 'mkdir', 'symlink': 'symlink_to'}[operation]
                        actual = getattr(Path, method)
                        def fail(path, *args, **kwargs):
                            if path == bad or (operation == 'mkdir' and path == bad.parent) or (operation == 'symlink' and path.name.startswith('.' + bad.name)):
                                raise PermissionError(operation)
                            return actual(path, *args, **kwargs)
                        # For mkdir, fail once so independent files/queue writes can proceed.
                        if operation == 'mkdir':
                            calls = [False]
                            def fail(path, *args, **kwargs):
                                if path == bad.parent and not calls[0]:
                                    calls[0] = True
                                    raise PermissionError('mkdir')
                                return actual(path, *args, **kwargs)
                        stack.enter_context(mock.patch.object(Path, method, fail))
                    elif operation == 'replace':
                        actual = managed.os.replace
                        def fail(source, dest):
                            if Path(dest) == bad:
                                raise PermissionError('replace')
                            return actual(source, dest)
                        stack.enter_context(mock.patch.object(managed.os, 'replace', fail))
                    elif operation == 'text':
                        actual = managed.atomic_text
                        def fail(path, *args):
                            if path == bad:
                                raise PermissionError('text')
                            return actual(path, *args)
                        stack.enter_context(mock.patch.object(managed, 'atomic_text', fail))
                    else:
                        actual = managed.atomic_json
                        calls = [False]
                        def fail(path, data):
                            if path == self.install.receipt and not calls[0]:
                                calls[0] = True
                                raise PermissionError('receipt')
                            return actual(path, data)
                        stack.enter_context(mock.patch.object(managed, 'atomic_json', fail))
                    self.install.recover()
                    self.assertEqual(managed.read_json(self.install.current), old)
                    self.assertEqual(good.read_text(), 'original')
                    self.assertFalse(self.install.journal.exists())
                    self.assertTrue(staged.exists())
                    self.assertIn(str(bad), self.install.status()['deferred_restoration']['files'])
                    service.assert_called_with('start')
                # Once access is restored, the next lifecycle recovery converges.
                with mock.patch.object(self.install, 'service'), mock.patch.object(self.install, 'service_state', return_value={}):
                    self.install.recover()
                self.assertFalse((self.install.state / 'deferred-restoration.json').exists())
                self.assertFalse(staged.exists())
                if snapshot is None:
                    self.assertFalse(bad.exists())
                elif 'target' in snapshot:
                    self.assertEqual(os.readlink(bad), 'original-target')
                else:
                    self.assertEqual(bad.read_text(), 'original')

    def test_unreadable_bar_siblings_preserve_without_aborting_migration(self):
        legacy, _, style, _ = sixth.SixthReviewTests.seed_bars(self)
        shim = self.install.data / 'integrations/legacy-tray.sh'
        target = style.parent / 'hyprwhspr-style.css'
        source = legacy / 'config/waybar/hyprwhspr-style.css'
        for path in (shim, target, source):
            with self.subTest(path=path):
                actual = Path.read_text
                def read(candidate, *args, **kwargs):
                    if candidate == path:
                        raise PermissionError(str(path))
                    return actual(candidate, *args, **kwargs)
                # Existing targets exercise their guarded reads.
                if path != source:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text('existing')
                with mock.patch.object(Path, 'read_text', read), mock.patch.object(self.install, 'bar_processes', return_value={}), mock.patch('sys.stdout', new_callable=io.StringIO) as output:
                    changes = self.install.bar_migrations()
                self.assertIn(str(path), output.getvalue())
                self.assertNotIn(style, changes)
                if path != source:
                    path.unlink()

    def test_adopted_unit_replace_failure_leaves_complete_unit_and_rolls_back(self):
        unit = self.root / 'hyprwhspr.service'
        unit.write_text('complete old unit')
        root = self.install.data / 'releases/old'
        root.mkdir(parents=True)
        env = self.install.data / 'environments/cli'
        env.mkdir(parents=True)
        old = {'version': 'v1.0.0', 'root': str(root), 'python': {'path': '/python'},
               'cli': {'path': str(env)}, 'backend': None, 'selection': None}
        managed.atomic_json(self.install.current, old)
        actual = managed.os.replace
        failed = []
        def replace(source, dest):
            if Path(dest) == unit and not failed:
                self.assertEqual(unit.read_text(), 'complete old unit')
                failed.append(True)
                raise OSError('unit disk full')
            return actual(source, dest)
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(managed.os, 'replace', replace))
            stack.enter_context(mock.patch.object(managed, 'interpreter', return_value=old['python']))
            stack.enter_context(mock.patch.object(managed, 'verify_payload', return_value={'version': old['version']}))
            stack.enter_context(mock.patch.object(managed, 'service_content', return_value='complete new unit'))
            stack.enter_context(mock.patch.object(managed, 'run'))
            for name, result in [('build', {**old['cli'], 'key': 'changed'}), ('build_backend', (None, None)),
                    ('validate_config', None), ('check_integrations', str(unit)), ('service_state', {}),
                    ('daemon_running', False), ('binding_migrations', {}), ('bar_migrations', {}),
                    ('legacy_selection', None), ('install_launcher', None)]:
                stack.enter_context(mock.patch.object(self.install, name, return_value=result))
            with self.assertRaisesRegex(OSError, 'unit disk full'):
                self.install.update(local_payload=True)
        self.assertTrue(failed)
        self.assertEqual(unit.read_text(), 'complete old unit')
        self.assertEqual(managed.read_json(self.install.current), old)
        self.assertFalse(self.install.journal.exists())
